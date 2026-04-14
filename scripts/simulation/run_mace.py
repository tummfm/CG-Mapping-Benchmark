import argparse
import contextlib
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument(
    "--model", type=str, help="Model path", required=True
)
parser.add_argument("--mol", type=str, help="Molecule to simulate", required=True)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output", default=True
)
parser.add_argument(
    "--xla_mem_fraction",
    type=float,
    default=None,
    help="Override XLA_PYTHON_CLIENT_MEM_FRACTION (e.g. 0.6 on shared GPUs).",
)
parser.add_argument(
    "--equilibrate",
    type=float,
    default=0.0,
    metavar="TIME_IN_PS",
    help="Run an equilibration stage at dt=0.5 fs for TIME_IN_PS before production.",
)
parser.add_argument(
    "--dt",
    type=float,
    default=None,
    metavar="DT_IN_FS",
    help="Override simulation timestep from config (in fs).",
)
parser.add_argument("--gamma", type=float, default=None, help="Override DEFAULT_SIM_CONFIG.gamma")
parser.add_argument(
    "--dt-values-fs",
    type=float,
    nargs="+",
    default=None,
    metavar="DT_IN_FS",
    help="Override DEFAULT_SIM_CONFIG.dt_values_fs with one or more fs values.",
)
parser.add_argument("--print-every", type=float, default=None, help="Override DEFAULT_SIM_CONFIG.print_every")
parser.add_argument("--sim-mode", type=str, default=None, help="Override DEFAULT_SIM_CONFIG.sim_mode")
parser.add_argument("--ensemble", type=str, default=None, help="Override DEFAULT_SIM_CONFIG.ensemble")
parser.add_argument("--t-total", type=float, default=None, help="Override DEFAULT_SIM_CONFIG.t_total (ps)")
parser.add_argument("--n-chains", type=int, default=None, help="Override DEFAULT_SIM_CONFIG.n_chains")
parser.add_argument("--kT", type=float, default=None, help="Override DEFAULT_SIM_CONFIG.kT")
parser.add_argument("--T", type=float, default=None, help="Override DEFAULT_SIM_CONFIG.T (K)")
parser.add_argument(
    "--prngkey-seed",
    type=int,
    default=None,
    help="Override DEFAULT_SIM_CONFIG.PRNGKey_seed",
)
args = parser.parse_args()

# Set device
if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

if args.xla_mem_fraction is not None:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)
else:
    # Respect externally provided value and use a safer default on shared GPUs.
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")

import json
import pickle
import numpy as onp
from jax import numpy as jnp, tree_util
import jax
import time

from chemtrain.data import preprocessing
from chemtrain import quantity, util
from chemtrain.compose import (
    mace_jax as mace_jax_compose,
    mace_jax_cg as mace_jax_compose_cg,
    mace_jax_bond as mace_jax_compose_bond,
)
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from jax import random
from chemtrain.ensemble import sampling
from jax_md import partition, space, simulate, energy
from jax_md_mod import custom_quantity
from cgbench.core import dataset
from cgbench.core.config import (
    DEFAULT_SIM_CONFIG as SIM_CONFIG,
    DEFAULT_MACE_CONFIG,
)

# -------------------------
# Configuration handling
# -------------------------
model_path = args.model
base_dir = os.path.dirname(model_path)

# Load MACE config
mace_config_path = os.path.join(base_dir, "config.json")
if os.path.exists(mace_config_path):
    with open(mace_config_path, "r") as f:
        # load
        MACE_CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"Config file {mace_config_path} not found.")

# Load training config
train_config_path = os.path.join(base_dir, "train_config.json")
if os.path.exists(train_config_path):
    with open(train_config_path, "r") as f:
        TRAIN_CONFIG = json.load(f)
else:
    raise FileNotFoundError(f"Train config file {train_config_path} not found.")

config = SIM_CONFIG.copy()
config["sim_mol"] = args.mol
config["type"] = MACE_CONFIG["type"]
config["cg_map"] = MACE_CONFIG.get("CG_map", None)
config["t_eq"] = float(args.equilibrate)

if args.gamma is not None:
    config["gamma"] = float(args.gamma)
if args.dt_values_fs is not None:
    config["dt_values_fs"] = [float(v) for v in args.dt_values_fs]
if args.print_every is not None:
    config["print_every"] = float(args.print_every)
if args.sim_mode is not None:
    config["sim_mode"] = str(args.sim_mode)
if args.ensemble is not None:
    config["ensemble"] = str(args.ensemble)
if args.t_total is not None:
    config["t_total"] = float(args.t_total)
if args.n_chains is not None:
    config["n_chains"] = int(args.n_chains)
if args.kT is not None:
    config["kT"] = float(args.kT)
if args.T is not None:
    config["T"] = float(args.T)
if args.prngkey_seed is not None:
    config["PRNGKey_seed"] = int(args.prngkey_seed)

if args.dt is not None:
    if args.dt <= 0:
        raise ValueError(f"--dt must be positive (fs). Got {args.dt}.")
    config["dt_values_fs"] = [float(args.dt)]

if args.verbose:
    print("-" * 50)
    for key, value in MACE_CONFIG.items():
        print(f"Found MACE config: {key}: {value}")
    print("-" * 50)
    for key, value in config.items():
        print(f"Using Sim config: {key}: {value}")
    print("-" * 50)

# -------------------------
# Load dataset
# -------------------------
def _load_simulation_dataset(mol, train_ratio, val_ratio, cg_map, model_type):
    def _infer_nmol_from_map(data_obj, default: int):
        map_obj = getattr(data_obj, "map_obj", None)
        if map_obj is not None:
            if hasattr(map_obj, "n_mols"):
                return int(getattr(map_obj, "n_mols"))
            if hasattr(map_obj, "n_replicas"):
                return int(getattr(map_obj, "n_replicas"))
        return int(default)

    basic_loaders = {
        "capped_ala": (dataset.Capped_Ala_Dataset, 1),
        "capped_ala2": (dataset.Capped_Ala2_Dataset, 1),
        "hexane": (dataset.Hexane_Dataset, 100),
        "capped_ala15": (dataset.Capped_Ala15_Dataset, 1),
        "capped_pro": (dataset.Capped_Pro_Dataset, 1),
        "capped_thr": (dataset.Capped_Thr_Dataset, 1),
        "capped_gly": (dataset.Capped_Gly_Dataset, 1),
    }

    if mol in basic_loaders:
        cls, nmol = basic_loaders[mol]
        data_obj = cls(train_ratio=train_ratio, val_ratio=val_ratio)
        return mol, data_obj, nmol, False

    if mol == "benzene_crystal":
        data_obj = dataset.BenzeneCrystal_Dataset(train_ratio=train_ratio, val_ratio=val_ratio)
        nmol = _infer_nmol_from_map(data_obj, default=128)
        return mol, data_obj, nmol, False

    if mol == "tip3p" or mol == "tip3p-water":
        data_obj = dataset.TIP3P_water_Dataset(train_ratio=train_ratio, val_ratio=val_ratio)
        nmol = _infer_nmol_from_map(data_obj, default=901)
        return mol, data_obj, nmol, False

    if mol == "benzene_crystal_288":
        data_obj = dataset.BenzeneCrystal288_Dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            cg_map=cg_map,
            prefer_cg_cache=(model_type == "CG"),
        )
        nmol = int(getattr(data_obj, "nmol", data_obj.map_obj.n_replicas))
        return mol, data_obj, nmol, False

    if mol in ("CATH", "cath", "cath_full", "cath_quarter", "cath_test"):
        dataset_key = "cath_full" if mol in ("CATH", "cath") else mol
        map_name = cg_map or "coreBetaMap2"

        cache_candidates = []
        if dataset_key == "cath_full":
            cache_candidates.extend(
                [
                    f"/ds/project/franz/Datasets/CATH/CATH_full_{map_name}.npz",
                    f"/ds/project/franz/Datasets/CATH_full_{map_name}.npz",
                ]
            )
        elif dataset_key == "cath_quarter":
            cache_candidates.append(f"/ds/project/franz/Datasets/CATH_quarter_{map_name}.npz")
        elif dataset_key == "cath_test":
            cache_candidates.append(f"/ds/project/franz/Datasets/CATH_test_{map_name}.npz")

        cached_path = next((p for p in cache_candidates if os.path.exists(p)), None)
        if args.verbose and cached_path is not None:
            print(f"Using CATH cached dataset: {cached_path}")

        data_obj = dataset.CATH_Dataset(
            dataset_key=dataset_key,
            cg_strategy=map_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            cached_dataset_path=cached_path,
        )
        return mol, data_obj, 1, False

    uncapped = {
        "1UBQ": dataset.UBQ1_Dataset,
        "1IFC": dataset.IFC1_Dataset,
        "1MJC": dataset.MJC1_Dataset,
        "1QX5": dataset.QX5_1_Dataset,
        "6LYT": dataset.LYT6_Dataset,
    }
    if mol in uncapped:
        data_obj = uncapped[mol](
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=False,
        )
        return mol, data_obj, 1, True

    raise ValueError(
        "Invalid molecule. Use 'capped_ala', 'capped_ala2', 'capped_ala15', "
        "'hexane', 'benzene_crystal', 'tip3p', 'tip3p-water', 'capped_pro', 'capped_thr', 'capped_gly', "
        "'benzene_crystal_288', '1UBQ', '1IFC', "
        "'1MJC', '1QX5', '6LYT', 'CATH', 'cath_full', 'cath_quarter', or 'cath_test'."
    )


(
    config["sim_mol"],
    data,
    config["nmol"],
    _force_single_chain,
) = _load_simulation_dataset(
    mol=config["sim_mol"],
    train_ratio=MACE_CONFIG["train_ratio"],
    val_ratio=MACE_CONFIG["val_ratio"],
    cg_map=MACE_CONFIG.get("CG_map"),
    model_type=MACE_CONFIG.get("type"),
)

if _force_single_chain:
    # Uncapped-protein workflows are single-chain only.
    config["n_chains"] = 1

# AT
if MACE_CONFIG["type"] == "AT":
    if hasattr(data, "load_traj"):
        data.load_traj()
    dataset = data.dataset_U
    species = data.species
    masses = data.masses
    n_species = data.n_species
# CG
elif MACE_CONFIG["type"] == "CG":
    data.coarse_grain(map=MACE_CONFIG["CG_map"])
    dataset = data.cg_dataset_U
    species = data.cg_species
    masses = data.cg_masses
    n_species = data.n_cg_species
else:
    raise ValueError("Invalid simulation type. Use 'AT' or 'CG'.")

_is_cath_like_model = (
    "cath" in model_path.lower() or "spice" in model_path.lower() or "fm" in model_path.lower()
)
_cg_map = MACE_CONFIG.get("CG_map")

# LEGACY SUPPORT
# Historical species remapping for CATH-like checkpoints is only valid for
# coreBetaMap2. For other CG maps (e.g. martini3), keep dataset-derived species.
if _is_cath_like_model and _cg_map == "coreBetaMap2" and config["sim_mol"] == "capped_ala":
    species = jnp.array([1, 24, 22, 9, 23, 2])  # C-ACE, N-ALA, CA, CB, C-ALA, N-NME

if _is_cath_like_model and _cg_map == "coreBetaMap2" and config["sim_mol"] == "capped_thr":
    species = jnp.array([1, 24, 22, 4, 23, 2])  # C-ACE, N-ALA, CA, CB, C-ALA, N-NME

if _is_cath_like_model and _cg_map == "coreBetaMap2" and config["sim_mol"] == "capped_pro":
    species = jnp.array([1, 24, 14, 22, 23, 2])  # C-ACE, N-PRO, CB, CA, C-PRO, N-NME

if _is_cath_like_model and _cg_map == "coreBetaMap2" and config["sim_mol"] == "capped_ala15":
    species = jnp.array([1] + [24, 22, 9, 23] * 15 + [2])  # C-ACE, N-PRO, CB, CA, C-PRO, N-NME

print(f"[SPECIES]: {species}")

# -------------------------
# Neighbor list setup
# -------------------------
r_cutoff = MACE_CONFIG["r_cutoff"]
box = data.box

if box is not None:
    displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)
else:
    displacement_fn, _ = space.free()

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
    preprocessing.allocate_neighborlist(
        dataset["training"],
        displacement_fn,
        box,
        r_cutoff=MACE_CONFIG["r_cutoff"],
        mask_key="mask",
        box_key="box" if box is not None else None,
        format=partition.Sparse,
        batch_size=10,
        capacity_multiplier=1.2,
    )
)

# -------------------------
# Model initialization
# -------------------------
# Setup MACE config (in mace-jax format)
mace_cfg = {
    "r_cutoff": MACE_CONFIG["r_cutoff"],
    "hidden_irreps": MACE_CONFIG.get(
        "hidden_irreps", DEFAULT_MACE_CONFIG["hidden_irreps"]
    ),
    "MLP_irreps": MACE_CONFIG.get("readout_mlp_irreps", "16x0e"),
    "num_interactions": MACE_CONFIG.get(
        "num_interactions", DEFAULT_MACE_CONFIG["num_interactions"]
    ),
    "max_ell": MACE_CONFIG.get("max_ell", DEFAULT_MACE_CONFIG["max_ell"]),
    "correlation": MACE_CONFIG.get("correlation", DEFAULT_MACE_CONFIG["correlation"]),
    "n_radial_basis": MACE_CONFIG.get(
        "n_radial_basis", DEFAULT_MACE_CONFIG["n_radial_basis"]
    ),
    "output_irreps": MACE_CONFIG.get("output_irreps", "1x0e"),
    "use_so3": MACE_CONFIG.get("use_so3", False),
}

cueq_config = CuEquivarianceConfig(
    enabled=True,
    layout=("mul_ir"),
    group=("O3"),
    optimize_all=True,
    conv_fusion=True,
)
if MACE_CONFIG.get("use_so3", False):
    print("[NOTE] Using SO(3) equivariance (no CuEquivariance support)")
    cueq_config = None

template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
    displacement=displacement_fn,
    r_cutoff=MACE_CONFIG["r_cutoff"],
    n_species=100,
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors,
    mode="energy",
    use_custom_batch_fn=False,  # Not needed for simulation
    mace_config=mace_cfg,
    cueq_config=cueq_config,
)

variables = template_vars


species_init = jnp.asarray(species)


def energy_fn_template(params):
    vars = {**variables}
    vars["params"] = params

    def energy_fn(pos, neighbor, species=species_init, **dynamic_kwargs):
        dynamic_kwargs.setdefault("box", box)
        pots = gnn_energy_fn(vars, pos, neighbor, species=species, **dynamic_kwargs)

        # Subtract the provided atomic energies
        atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
        mapped_species = jnp.argmax(
            species[:, None] == atomic_numbers[None, :], axis=-1
        )
        pots -= jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)[
            mapped_species
        ] * dynamic_kwargs.get("mask", 1.0)

        return jnp.sum(pots)

    return energy_fn


if args.verbose:
    print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

# -------------------------
# Load model parameters
# -------------------------
model_path = args.model
base_dir = os.path.dirname(model_path)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")
energy_params = onp.load(model_path, allow_pickle=True)
energy_params = tree_util.tree_map(jnp.asarray, energy_params)
energy_fn = energy_fn_template(energy_params)

# Setup base dir
outdir = f"{base_dir}/simulation_{config['ensemble']}_T={config['T']}K/"
os.makedirs(outdir, exist_ok=True)


# -------------------------
# Simulator initialization
# -------------------------
def init_simulator(
    dataset,
    energy_fn,
    masses,
    nbrs_init,
    kT,
    dt,
    n_chains,
    gamma,
    t_eq,
    t_total,
    r_init_override=None,
):
    key = random.PRNGKey(config["PRNGKey_seed"])
    _prosol_uncapped = {"1UBQ", "1IFC", "1MJC", "1QX5", "6LYT"}
    if config["sim_mol"] in _prosol_uncapped and n_chains != 1:
        raise ValueError(f"{config['sim_mol']} simulation only supports n_chains=1.")

    if "box" in dataset["validation"]:
        _, shift_fn = space.periodic_general(
            dataset["validation"]["box"][0], fractional_coordinates=True
        )
    else:
        _, shift_fn = space.free()

    if r_init_override is not None:
        r_init = jnp.asarray(r_init_override)
        if r_init.ndim != 3:
            raise ValueError(
                "r_init_override must have shape (n_chains, n_atoms, 3). "
                f"Got shape {r_init.shape}."
            )
        if r_init.shape[0] != n_chains:
            raise ValueError(
                "r_init_override first dimension must match n_chains. "
                f"Got {r_init.shape[0]} vs n_chains={n_chains}."
            )
    elif config["sim_mol"] in _prosol_uncapped:
        # Start strictly from the first reference frame.
        r_init = dataset["training"]["R"][:1]

    elif (
        config["sim_mode"] == "stability"
    ):  # take the first structure and repeat for n_chains
        r_init = dataset["validation"]["R"][0]
        # repeat for n_chains
        r_init = jnp.tile(r_init, (n_chains, 1, 1))

    elif config["sim_mode"] == "helix":  # use predefined helix indices
        indices = onp.load(
            "/home/franz/Ala15_100_min_helix_indices.npy", allow_pickle=True
        )
        combined_dataset = onp.concatenate(
            [dataset["training"]["R"], dataset["validation"]["R"]], axis=0
        )

        # assert that the indices are within the range of the combined dataset
        assert (
            jnp.max(indices) < combined_dataset.shape[0]
        ), "Indices exceed dataset size."

        r_init = combined_dataset[indices]

    elif config["sim_mode"] == "speed":  # random selection, single chain
        key, split = random.split(key)
        selection = random.choice(
            split,
            jnp.arange(dataset["validation"]["R"].shape[0]),
            shape=(1,),
            replace=False,
        )
        r_init = dataset["validation"]["R"][selection]

    else:  # sampling mode: random selection, n_chains
        key, split = random.split(key)
        selection = random.choice(
            split,
            jnp.arange(dataset["validation"]["R"].shape[0]),
            shape=(n_chains,),
            replace=False,
        )
        r_init = dataset["validation"]["R"][selection]

    if config["ensemble"] == "NVT":
        init_simulator_fn = simulate.nvt_langevin
        sim_kwargs = {"kT": kT, "gamma": gamma, "dt": dt}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init}

    elif config["ensemble"] == "NVE":
        init_simulator_fn = simulate.nve
        sim_kwargs = {"dt": dt, "kT": kT}
        init_sim_kwargs = {"mass": masses, "neighbor": nbrs_init, "kT": kT}

    else:
        raise ValueError(f"Unknown ensemble: {config['ensemble']}. Use NVT or NVE.")

    init_ref_state, sim_template = sampling.initialize_simulator_template(
        init_simulator_fn,
        shift_fn=shift_fn,
        nbrs=init_sim_kwargs["neighbor"],
        init_with_PRNGKey=True,
        extra_simulator_kwargs=sim_kwargs,
    )

    # Init reference state
    key, split = random.split(key)
    reference_state = init_ref_state(
        split, r_init, energy_or_force_fn=energy_fn, init_sim_kwargs=init_sim_kwargs
    )

    # Setup evaluation timings
    eval_timings = sampling.process_printouts(
        time_step=dt,
        total_time=t_total,
        t_equilib=t_eq,
        print_every=config["print_every"],
    )

    # Setup quantities to record
    quantities = {
        "kT": custom_quantity.temperature,
        "epot": custom_quantity.energy_wrapper(lambda _: energy_fn),
        # "force": custom_quantity.force_wrapper(lambda _: energy_fn),
        # "etot": custom_quantity.total_energy_wrapper(lambda _: energy_fn),
    }

    # Initialize trajectory generator
    traj_gen = sampling.trajectory_generator_init(
        sim_template,
        lambda _: energy_fn,
        eval_timings,
        quantities=quantities,
        vmap_sim_batch=config["n_chains"],
        vmap_batch=config["n_chains"],
    )

    return reference_state, jax.jit(traj_gen)


def visualise(traj_path, dataset):
    from cgbench.plotting import molecules as visualise_traj

    vis_fn_map = {
        "capped_ala": visualise_traj.vis_capped_ala,
        "capped_ala2": visualise_traj.vis_capped_ala,
        "tip3p": visualise_traj.vis_tip3p_water,
        "tip3p-water": visualise_traj.vis_tip3p_water,
        "hexane": visualise_traj.vis_hexane,
        "benzene_crystal": visualise_traj.vis_benzene_crystal,
        "benzene_crystal_288": visualise_traj.vis_benzene_crystal,
        "capped_ala15": visualise_traj.vis_capped_ala15,
        "capped_pro": visualise_traj.vis_capped_pro,
        "capped_thr": visualise_traj.vis_capped_thr,
        "capped_gly": visualise_traj.vis_capped_gly,
        "1UBQ": visualise_traj.vis_staticframe_protein,
        "1IFC": visualise_traj.vis_staticframe_protein,
        "1MJC": visualise_traj.vis_staticframe_protein,
        "1QX5": visualise_traj.vis_staticframe_protein,
        "6LYT": visualise_traj.vis_staticframe_protein,
    }

    if config["sim_mol"] in vis_fn_map:
        vis_fn = vis_fn_map[config["sim_mol"]]
        vis_fn(
            traj_path,
            config,
            type=MACE_CONFIG["type"],
            dataset=dataset,
            cg_map=MACE_CONFIG["CG_map"],
        )
    elif str(config["sim_mol"]).lower() in (
        "cath",
        "cath_full",
        "cath_quarter",
        "cath_test",
    ):
        print(f"No dedicated visualizer for {config['sim_mol']}; skipping plotting.")
    else:
        raise ValueError(
            "Invalid molecule. Use 'capped_ala', 'capped_ala2', 'hexane', 'benzene_crystal', 'tip3p-water', "
            "'capped_ala15', 'capped_pro', 'capped_thr', 'capped_gly', "
            "'1UBQ', '1IFC', '1MJC', '1QX5', or '6LYT'."
        )


def _extract_last_positions(traj_pos, n_chains):
    """Return last-frame positions with shape (n_chains, n_atoms, 3)."""
    arr = jnp.asarray(traj_pos)
    if arr.ndim == 4:
        # (n_chains, n_frames, n_atoms, 3)
        return arr[:, -1, :, :]
    if arr.ndim == 3:
        # (n_frames, n_atoms, 3) -> only valid for single-chain runs
        if n_chains != 1:
            raise ValueError(
                "Received single-chain trajectory positions but n_chains != 1."
            )
        return arr[-1:, :, :]
    raise ValueError(f"Unexpected trajectory position shape: {arr.shape}")


def _save_outputs_and_plot(save_dir, traj_state, dt_ps, t_total_ps, box, dataset_obj):
    """Persist trajectory outputs and run plotting for a simulation stage."""
    with open(os.path.join(save_dir, "trajectory.pkl"), "wb") as f:
        pickle.dump(traj_state.trajectory.position, f)

    with open(os.path.join(save_dir, "traj_state_aux.pkl"), "wb") as f:
        pickle.dump(traj_state.aux, f)

    config_ = config.copy()
    config_["dt"] = dt_ps
    config_["t_total"] = t_total_ps
    config_.pop("dt_values_fs", None)
    config_["box"] = float(box[0][0]) if box is not None else None  # cubic box side length in nm

    with open(os.path.join(save_dir, "traj_config.json"), "w") as cf:
        json.dump(config_, cf, indent=4)

    try:
        visualise(save_dir, dataset_obj)
    except Exception as e:
        print(f"Error during visualisation: {e}")


class _TeeStream:
    """Mirror stream writes to terminal and a file."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _tee_prints_to_file(log_path):
    """Redirect stdout/stderr to both terminal and a log file."""
    with open(log_path, "a", buffering=1) as log_file:
        tee_stdout = _TeeStream(sys.__stdout__, log_file)
        tee_stderr = _TeeStream(sys.__stderr__, log_file)
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(
            tee_stderr
        ):
            yield


dt_values_fs = config["dt_values_fs"]
dt_values_ps = [dt_fs * 0.001 for dt_fs in dt_values_fs]  # convert to ps


def _format_dt_for_path(dt_fs):
    """Format dt (fs) for stable folder names (e.g., 6.0 -> 6)."""
    dt_val = float(dt_fs)
    if dt_val.is_integer():
        return str(int(dt_val))
    return format(dt_val, "g")

for dt_fs, dt_ps in zip(dt_values_fs, dt_values_ps):
    # Update config for this dt
    config["dt"] = dt_ps

    print(f"\nStarting simulation for dt = {dt_fs} fs ({dt_ps} ps)...")
    dt_fs_label = _format_dt_for_path(dt_fs)
    folder_name = f"traj_mol={config['sim_mol']}_dt={dt_fs_label}_teq={config['t_eq']}_t={config['t_total']}_nmol={config['nmol']}_nchains={config['n_chains']}_mode={config['sim_mode']}_seed={config['PRNGKey_seed']}_gamma={config['gamma']}/"
    save_dir = os.path.join(outdir, folder_name)
    log_path = os.path.join(save_dir, "simulation.log")

    os.makedirs(save_dir, exist_ok=True)

    with _tee_prints_to_file(log_path):
        print(f"Logging simulation output to {log_path}")

        # Skip simulation if folder already exists and already has trajectory output.
        if os.path.exists(os.path.join(save_dir, "trajectory.pkl")):
            print(
                f"Directory {save_dir} already has trajectory output. Skipping simulation for dt = {dt_fs} fs."
            )
            visualise(save_dir, data)
            continue

        r_init_override = None
        equil_time_ps = float(args.equilibrate)
        if equil_time_ps > 0.0:
            equil_dt_ps = 0.002
            equil_dir = os.path.join(save_dir, f"equilibration_t={equil_time_ps}ps")
            os.makedirs(equil_dir, exist_ok=True)

            print(
                f"Starting equilibration for {equil_time_ps} ps at dt=0.5 fs in {equil_dir}"
            )
            eq_state, eq_generator = init_simulator(
                dataset,
                energy_fn,
                masses,
                nbrs_init,
                kT=config["kT"],
                dt=equil_dt_ps,
                n_chains=config["n_chains"],
                gamma=config["gamma"],
                t_eq=0.0,
                t_total=equil_time_ps,
            )
            eq_start = time.time()
            eq_traj_state = eq_generator(None, eq_state)
            eq_elapsed = time.time() - eq_start
            eq_elapsed_norm = eq_elapsed / max(1, config["n_chains"])
            print(
                f"Equilibration completed in {eq_elapsed_norm:.2f} seconds "
                f"(normalized by n_chains={config['n_chains']})."
            )

            _save_outputs_and_plot(
                save_dir=equil_dir,
                traj_state=eq_traj_state,
                dt_ps=equil_dt_ps,
                t_total_ps=equil_time_ps,
                box=box,
                dataset_obj=data,
            )
            r_init_override = _extract_last_positions(
                eq_traj_state.trajectory.position, config["n_chains"]
            )

        reference_state, traj_generator = init_simulator(
            dataset,
            energy_fn,
            masses,
            nbrs_init,
            kT=config["kT"],
            dt=dt_ps,
            n_chains=config["n_chains"],
            gamma=config["gamma"],
            t_eq=config["t_eq"],
            t_total=config["t_total"],
            r_init_override=r_init_override,
        )

        # time the trajectory generation
        start_time = time.time()
        traj_state = traj_generator(None, reference_state)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time_norm = elapsed_time / max(1, config["n_chains"])
        print(
            f"Simulation completed in {elapsed_time_norm:.2f} seconds "
            f"(normalized by n_chains={config['n_chains']})."
        )

        # calculate ns/day
        total_sim_time_ps = config["t_total"] * config["n_chains"]
        ns_per_day = (total_sim_time_ps / elapsed_time) * (86400 / 1000)
        print(f"Performance: {ns_per_day:.2f} ns/day")

        _save_outputs_and_plot(
            save_dir=save_dir,
            traj_state=traj_state,
            dt_ps=dt_ps,
            t_total_ps=config["t_total"],
            box=box,
            dataset_obj=data,
        )

        print(f"Finished dt = {dt_fs} fs. Results saved to {save_dir}.")
