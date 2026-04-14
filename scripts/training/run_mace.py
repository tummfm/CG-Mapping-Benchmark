"""
MACE-JAX Training Script for CG and AT Molecular Simulations.

This script trains a MACE model using the MACE-JAX implementation with the
chemtrain framework. Supports various molecules and coarse-graining mappings.
"""

import argparse
import os
import sys


parser = argparse.ArgumentParser(description="Train MACE model using MACE-JAX")
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, help="CG mapping to use", required=True)
parser.add_argument("--mol", type=str, help="Molecule to use", required=True)
parser.add_argument("--stride", type=int, help="Subsample dataset", default=None)
parser.add_argument(
    "--rcut", type=float, help="Cutoff radius for neighbor list", default=0.5
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output", default=False
)
parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging")
parser.add_argument(
    "--swa-start",
    type=int,
    default=None,
    help="Epoch index to start SWA snapshots (default: last 25 percent of epochs)",
)
parser.add_argument(
    "--swa-every",
    type=int,
    default=1,
    help="Collect SWA snapshot every N epochs after start",
)
parser.add_argument(
    "--swa-min-snapshots",
    type=int,
    default=2,
    help="Minimum SWA snapshots required before using SWA params",
)
parser.add_argument(
    "--no-swa-prefer",
    action="store_true",
    help="Keep raw best params for eval/save even when SWA params are available",
)
parser.add_argument(
    "--use-so3",
    action="store_true",
    help="Use SO(3) equivariance in MACE instead of O(3) (no cueq support yet)",
)
parser.add_argument("--batch-size", type=int, default=None, help="Override DEFAULT_TRAIN_CONFIG.batch_size")
parser.add_argument("--init-lr", type=float, default=None, help="Override DEFAULT_TRAIN_CONFIG.init_lr")
parser.add_argument("--num-epochs", type=int, default=None, help="Override DEFAULT_TRAIN_CONFIG.num_epochs")
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="Backward-compatible alias for --num-epochs",
)
parser.add_argument("--decay-rate", type=float, default=None, help="Override DEFAULT_TRAIN_CONFIG.decay_rate")
parser.add_argument("--optimizer", type=str, default=None, help="Override DEFAULT_TRAIN_CONFIG.optimizer")
parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for dataset loading")
args = parser.parse_args()

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.97"

import json
import copy
import cloudpickle as pickle
import numpy as onp
import optax
from jax import numpy as jnp, random, tree_util
from jax_md import partition, space, energy
from collections import OrderedDict, defaultdict
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

from chemtrain import trainers
from chemtrain.data import preprocessing
from chemtrain.compose import mace_jax as mace_jax_compose

from cgbench.core import dataset
from cgbench.core.config import (
    DEFAULT_MACE_CONFIG,
    DEFAULT_TRAIN_CONFIG,
)

MACE_CONFIG = copy.deepcopy(DEFAULT_MACE_CONFIG)
TRAIN_CONFIG = copy.deepcopy(DEFAULT_TRAIN_CONFIG)

if args.batch_size is not None:
    TRAIN_CONFIG["batch_size"] = int(args.batch_size)
if args.init_lr is not None:
    TRAIN_CONFIG["init_lr"] = float(args.init_lr)
if args.num_epochs is not None:
    TRAIN_CONFIG["num_epochs"] = int(args.num_epochs)
if args.epochs is not None:
    TRAIN_CONFIG["num_epochs"] = int(args.epochs)
if args.decay_rate is not None:
    TRAIN_CONFIG["decay_rate"] = float(args.decay_rate)
if args.optimizer is not None:
    TRAIN_CONFIG["optimizer"] = str(args.optimizer)

if TRAIN_CONFIG.get("optimizer", "adam+decay") != "adam+decay":
    raise ValueError(
        "This script currently supports only optimizer='adam+decay'. "
        f"Got optimizer='{TRAIN_CONFIG['optimizer']}'."
    )

# Update MACE config with command line arguments
MACE_CONFIG["r_cutoff"] = args.rcut
MACE_CONFIG["mol"] = args.mol
MACE_CONFIG["CG_map"] = args.cgmap
MACE_CONFIG["type"] = "CG" if MACE_CONFIG["CG_map"] != "AT" else "AT"
MACE_CONFIG["use_so3"] = True if args.use_so3 else False

used_prestrided_cache = False



def _compute_swa_start_epoch(total_epochs, explicit_start):
    if explicit_start is not None:
        return max(0, min(explicit_start, max(0, total_epochs - 1)))
    # Default policy: start in the last quarter of training.
    late_phase_epochs = max(1, total_epochs // 4)
    return max(0, total_epochs - late_phase_epochs)


def _save_params(path, params):
    cpu_params = tree_util.tree_map(onp.asarray, params)
    with open(path, "wb") as f:
        pickle.dump(cpu_params, f)


class SWAManager:
    """Collects epoch snapshots and maintains a running SWA average."""

    def __init__(self, enabled, start_epoch, every, min_snapshots, prefer):
        self.enabled = enabled
        self.start_epoch = int(start_epoch)
        self.every = max(1, int(every))
        self.min_snapshots = max(1, int(min_snapshots))
        self.prefer = prefer
        self.snapshot_count = 0
        self.swa_params = None
        self.active = False

    def _should_collect(self, epoch):
        if not self.enabled:
            return False
        if epoch < self.start_epoch:
            return False
        return ((epoch - self.start_epoch) % self.every) == 0

    def _average(self, params):
        if self.swa_params is None:
            self.swa_params = tree_util.tree_map(lambda x: x, params)
            self.snapshot_count = 1
            return

        self.snapshot_count += 1
        count = float(self.snapshot_count)
        self.swa_params = tree_util.tree_map(
            lambda avg, new: avg + (new - avg) / count,
            self.swa_params,
            params,
        )

    def post_epoch(self, trainer, *args, **kwargs):
        del args, kwargs
        epoch = int(trainer._epoch)
        if not self._should_collect(epoch):
            return
        self._average(trainer.params)
        print(
            f"[SWA] Collected snapshot {self.snapshot_count} at epoch {epoch} "
            f"(start={self.start_epoch}, every={self.every})"
        )

    def post_training(self, trainer, *args, **kwargs):
        del trainer, args, kwargs
        self.active = self.enabled and (self.snapshot_count >= self.min_snapshots)
        if self.enabled:
            print(
                f"[SWA] Training finished with {self.snapshot_count} snapshots. "
                f"Active={self.active} (min_snapshots={self.min_snapshots}, prefer={self.prefer})"
            )


# -------------------------
# Helper Functions
# -------------------------

def get_train_config():
    """Create training config compatible with the optimizer setup."""
    return OrderedDict(
        optimizer=OrderedDict(
            init_lr=TRAIN_CONFIG["init_lr"],
            lr_decay=TRAIN_CONFIG["decay_rate"],
            epochs=TRAIN_CONFIG["num_epochs"],
            batch=TRAIN_CONFIG["batch_size"],
            cache=100,
            power="exponential",
            weight_decay=1e-3,
            type="ADAM",
            optimizer_kwargs=OrderedDict(
                b1=0.9,
                b2=0.995,
                eps=1e-8,
            ),
        ),
        gammas=OrderedDict(
            U=1e-3,
            F=1e-2,
        ),
        swa=OrderedDict(
            enabled=bool(args.swa),
            start_epoch=args.swa_start,
            every=max(1, int(args.swa_every)),
            min_snapshots=max(1, int(args.swa_min_snapshots)),
            prefer=not bool(args.no_swa_prefer),
        ),
    )


def init_optimizer(config, dataset_dict, key="optimizer"):
    """Initialize the optimizer with learning rate schedule."""
    num_samples = 1
    if "U" in dataset_dict["training"]:
        num_samples = dataset_dict["training"]["U"].shape[0]
    elif "F" in dataset_dict["training"]:
        num_samples = dataset_dict["training"]["F"].shape[0]
    else:
        print("No energy or force data available")
        exit()

    transition_steps = int(config[key]["epochs"] * num_samples) // config[key]["batch"]

    if config[key].get("power") == "exponential":
        lr_schedule_fm = optax.exponential_decay(
            config[key]["init_lr"],
            transition_steps,
            config[key]["lr_decay"],
        )
    else:
        lr_schedule_fm = optax.polynomial_schedule(
            config[key]["init_lr"],
            config[key]["lr_decay"] * config[key]["init_lr"],
            config[key].get("power", 2.0),
            transition_steps,
        )

    if args.verbose:
        print(f"Decay LR with power {config[key].get('power', 2.0)}")

    transforms = []

    if config[key].get("normalize"):
        transforms.append(optax.scale_by_param_block_norm())

    if config[key]["type"] == "ADAM":
        transforms.append(
            optax.scale_by_adam(
                b1=config[key]["optimizer_kwargs"]["b1"],
                b2=config[key]["optimizer_kwargs"]["b2"],
                eps=config[key]["optimizer_kwargs"]["eps"],
                eps_root=config[key]["optimizer_kwargs"]["eps"] ** 0.5,
                nesterov=True,
            )
        )
    else:
        raise NotImplementedError(f"Optimizer {config[key]['type']} not implemented.")

    weight_decay = config[key].get("weight_decay")
    if weight_decay is not None:
        transforms.append(optax.transforms.add_decayed_weights(weight_decay))

    optimizer_fm = optax.chain(
        *transforms,
        optax.scale_by_learning_rate(lr_schedule_fm, flip_sign=True),
    )

    return optimizer_fm


# -------------------------
# Load dataset
# -------------------------
def _load_training_dataset(mol, train_ratio, val_ratio, cg_map, stride):
    mol_alias = {
        "BenzeneCrystal": "benzene_crystal",
        "CATH": "cath_full",
        "cath": "cath_full",
    }
    mol_normalized = mol_alias.get(mol, mol)

    basic_loaders = {
        "capped_ala": dataset.Capped_Ala_Dataset,
        "capped_ala2": dataset.Capped_Ala2_Dataset,
        "capped_ala15": dataset.Capped_Ala15_Dataset,
        "hexane": dataset.Hexane_Dataset,
        "benzene_crystal": dataset.BenzeneCrystal_Dataset,
        "capped_pro": dataset.Capped_Pro_Dataset,
        "capped_thr": dataset.Capped_Thr_Dataset,
        "capped_gly": dataset.Capped_Gly_Dataset,
        "spice_dipeptides": dataset.SPICE_Dipeptides,
        "tip3p": dataset.TIP3P_water_Dataset
    }

    if mol_normalized in basic_loaders:
        data_obj = basic_loaders[mol_normalized](
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return mol_normalized, data_obj, False

    if mol_normalized in ("cath_full", "cath_quarter", "cath_test"):
        mol_size = mol_normalized.split("_", 1)[1]
        cached_path = f"/ds/project/franz/Datasets/CATH_{mol_size}_{cg_map}.npz"
        used_stride_cache = False
        if stride == 10:
            stride_cached_path = f"/ds/project/franz/Datasets/CATH_{mol_size}_{cg_map}_stride=10.npz"
            if os.path.exists(stride_cached_path):
                cached_path = stride_cached_path
                used_stride_cache = True
                print(f"Using pre-strided cache: {cached_path}")
        data_obj = dataset.CATH_Dataset(
            dataset_key=mol_normalized,
            cg_strategy=cg_map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            cached_dataset_path=cached_path,
        )
        return mol_normalized, data_obj, used_stride_cache
    raise ValueError(
        "Invalid molecule. Use 'capped_ala', 'capped_ala2', 'capped_ala15', "
        "'hexane', 'benzene_crystal', 'benzene_crystal_288', "
        "'capped_pro', 'capped_thr', 'capped_gly', 'spice_dipeptides', "
        "'cath_full', 'cath_quarter', 'cath_test', or 'CATH'"
    )


MACE_CONFIG["mol"], data, used_prestrided_cache = _load_training_dataset(
    mol=MACE_CONFIG["mol"],
    train_ratio=MACE_CONFIG["train_ratio"],
    val_ratio=MACE_CONFIG["val_ratio"],
    cg_map=MACE_CONFIG["CG_map"],
    stride=args.stride,
)

# AT
if MACE_CONFIG["type"] == "AT":
    if hasattr(data, "load_traj"):
        data.load_traj(workers=args.workers) if isinstance(data, dataset.MixedDataset) else data.load_traj()
    dataset_raw = data.dataset_U
    displacement_fn = data.displacement_fn_U
    species = data.species
    n_species = data.n_species

# CG
elif MACE_CONFIG["type"] == "CG":
    cg_cache_path = None
    _cg_kwargs = {"workers": args.workers} if isinstance(data, dataset.MixedDataset) else {}
    if MACE_CONFIG["mol"] in ["cath_full", "cath_quarter", "cath_test"]:
        _mol_size = MACE_CONFIG["mol"].split("_", 1)[1]  # "full", "quarter", or "test"
        cached_path = getattr(
            data,
            "cached_dataset_path",
            f"/ds/project/franz/Datasets/CATH_{_mol_size}_{MACE_CONFIG['CG_map']}.npz",
        )
        cg_cache_path = cached_path
        if os.path.exists(cached_path):
            data.coarse_grain(map=MACE_CONFIG["CG_map"], cached_dataset_path=cached_path, **_cg_kwargs)
        else:
            data.coarse_grain(map=MACE_CONFIG["CG_map"], **_cg_kwargs)
    else:
        data.coarse_grain(map=MACE_CONFIG["CG_map"], **_cg_kwargs)

    # Persist newly mapped CATH datasets so future runs can load them directly.
    if cg_cache_path and (not os.path.exists(cg_cache_path)):
        combined_cg = getattr(data, "cg_dataset", None)
        if isinstance(combined_cg, dict):
            os.makedirs(os.path.dirname(cg_cache_path), exist_ok=True)
            onp.savez(cg_cache_path, **combined_cg)
            print(f"Saved coarse-grained cache to {cg_cache_path}")

    if "spice" in MACE_CONFIG["mol"]:
        dataset_raw = data.cg_dataset_X
        displacement_fn = data.displacement_fn_X
    elif "cath" in MACE_CONFIG["mol"]:
        displacement_fn, _ = space.periodic_general(box=data.box, fractional_coordinates=True)
        dataset_raw = data.cg_dataset_U
    else:
        dataset_raw = data.cg_dataset_U
        displacement_fn = data.displacement_fn_U

    species = data.cg_species
    n_species = data.n_cg_species
else:
    raise ValueError("Invalid simulation type. Use 'AT' or 'CG'.")

# Dont train on energies
if 'U' in dataset_raw['training']:
    print("[CG Train] Dropping energies from dataset")
    del dataset_raw['training']['U']
    del dataset_raw['validation']['U']
    if 'testing' in dataset_raw:
        del dataset_raw['testing']['U']

if args.stride is not None:
    if used_prestrided_cache and args.stride == 10:
        print("Stride=10 pre-strided cache loaded; skipping in-memory subsampling.")
    else:
        print(f"Subsampling dataset with stride {args.stride}")
        print(f"Shape of train R before subsampling: {dataset_raw['training']['R'].shape}")
        for split in dataset_raw.keys():
            for key in dataset_raw[split].keys():
                dataset_raw[split][key] = dataset_raw[split][key][::args.stride]
        print(f"Shape of train R after subsampling: {dataset_raw['training']['R'].shape}")

# dataset_U / cg_dataset_U already contains fractional coordinates
# (converted in BaseDataset.__init__ via io.scale_dataset)
dataset_frac = dataset_raw

if args.verbose:
    print(f"Training set size: {dataset_frac['training']['R'].shape[0]}")
    print(f"Validation set size: {dataset_frac['validation']['R'].shape[0]}")

# -------------------------
# Setup output directory
# -------------------------
use_so3_ = "SO3" if args.use_so3 else "O3"
output_dir = f"outputs/MLP_train/{MACE_CONFIG['mol'].capitalize()}_map={MACE_CONFIG['CG_map']}_tr={MACE_CONFIG['train_ratio']}_rcut={MACE_CONFIG['r_cutoff']}_epochs={TRAIN_CONFIG['num_epochs']}_int={MACE_CONFIG['num_interactions']}_corr={MACE_CONFIG['correlation']}_seed={MACE_CONFIG['PRNGKey_seed']}_swa={'True' if args.swa else 'False'}_stride={args.stride}_maxL={MACE_CONFIG['max_ell']}_eq={use_so3_}"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Setup neighbor list and MACE model
# -------------------------
box = data.box

if box is None:
    print("No box provided, using non-periodic neighbor list space.free().")

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
    preprocessing.allocate_neighborlist(
        dataset_frac["training"],
        displacement_fn,
        box,
        r_cutoff=MACE_CONFIG["r_cutoff"],
        mask_key="mask",
        box_key="box" if box is not None else None,
        format=partition.Dense,
        batch_size=100,
    )
)

if args.verbose:
    print(
        f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, Avg neighbors: {avg_num_neighbors}"
    )

# Setup MACE config (in mace-jax format)
mace_cfg = {
    "r_cutoff": MACE_CONFIG["r_cutoff"], 
    "hidden_irreps": MACE_CONFIG["hidden_irreps"],
    "MLP_irreps": MACE_CONFIG.get("readout_mlp_irreps", "16x0e"),
    "num_interactions": MACE_CONFIG["num_interactions"],
    "max_ell": MACE_CONFIG["max_ell"],
    "correlation": MACE_CONFIG["correlation"],
    "n_radial_basis": MACE_CONFIG["n_radial_basis"],
    "output_irreps": MACE_CONFIG.get("output_irreps", "1x0e"),
    "use_so3": args.use_so3 if hasattr(args, "use_so3") else False,
}

cueq_config = CuEquivarianceConfig(
    enabled=True,
    layout=("mul_ir"),
    group=("O3"),
    optimize_all=True,
    conv_fusion=True,
)
if args.use_so3:
    print("[NOTE] Using SO(3) equivariance (no CuEquivariance support)")
    cueq_config = None


# Initialize MACE-JAX model (using mace_jax_compose.mace_jax_neighborlist)
template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
    displacement=displacement_fn,
    r_cutoff=MACE_CONFIG["r_cutoff"],
    n_species=100,
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors,
    mode="energy",
    use_custom_batch_fn=True,  # Required for batched training
    mace_config=mace_cfg,
    cueq_config=cueq_config,
)

init_params = template_vars["params"]
variables = template_vars

species_init = jnp.asarray(dataset_frac["training"]["species"][0])


def energy_fn_template(params):
    vars = {**variables}
    vars["params"] = params

    def energy_fn(position, neighbor, **kwargs):
        species = kwargs.pop("species", species_init)
        mask    = kwargs.pop("mask", jnp.ones(position.shape[0], dtype=jnp.bool_))

        pots = gnn_energy_fn(vars, position, neighbor, species=species, **kwargs)
        if pots.ndim == 2 and pots.shape[-1] == 1:
            pots = pots.squeeze(-1)

        atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
        atomic_energies = jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)
        mapped_species = jnp.argmax(species[:, None] == atomic_numbers[None, :], axis=-1)

        pots = (pots - atomic_energies[mapped_species]) * mask
        return jnp.sum(pots)

    return energy_fn


# Update neighborlist with initial positions
key = random.PRNGKey(MACE_CONFIG["PRNGKey_seed"])
r_init = jnp.asarray(dataset_frac["training"]["R"][0])
mask_init = jnp.asarray(dataset_frac["training"]["mask"][0])
nbrs_init = nbrs_init.update(r_init, mask=mask_init)

# -------------------------
# Setup optimizer
# -------------------------
train_config = get_train_config()
swa_start_epoch = _compute_swa_start_epoch(
    train_config["optimizer"]["epochs"],
    train_config["swa"]["start_epoch"],
)
train_config["swa"]["start_epoch"] = swa_start_epoch

if train_config["swa"]["enabled"]:
    print(
        "[SWA] Enabled with "
        f"start_epoch={train_config['swa']['start_epoch']}, "
        f"every={train_config['swa']['every']}, "
        f"min_snapshots={train_config['swa']['min_snapshots']}, "
        f"prefer={train_config['swa']['prefer']}"
    )
else:
    print("[SWA] Disabled")

optimizer_fm = init_optimizer(train_config, dataset_frac)

if args.verbose:
    num_samples = dataset_frac["training"]["R"].shape[0]
    batch_size = train_config["optimizer"]["batch"]
    epochs = train_config["optimizer"]["epochs"]
    total_steps = (epochs * num_samples) // batch_size
    print(f"Total steps: {total_steps}")
    print(f"Training on {num_samples} samples.")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")

# -------------------------
# Setup trainer
# -------------------------
trainer_fm = trainers.ForceMatching(
    init_params,
    optimizer_fm,
    energy_fn_template,
    nbrs_init,
    log_file=f"{output_dir}/force_matching.log",
    batch_per_device=train_config["optimizer"]["batch"],
    batch_cache=train_config["optimizer"]["cache"],
    gammas=train_config["gammas"],
)

swa_manager = SWAManager(
    enabled=train_config["swa"]["enabled"],
    start_epoch=train_config["swa"]["start_epoch"],
    every=train_config["swa"]["every"],
    min_snapshots=train_config["swa"]["min_snapshots"],
    prefer=train_config["swa"]["prefer"],
)
# Trainer task hooks pass `trainer` only to plain functions, not bound methods.
# Wrap SWA callbacks so they receive the trainer instance explicitly.
trainer_fm.add_task(
    "post_epoch",
    lambda trainer, *args, **kwargs: swa_manager.post_epoch(trainer, *args, **kwargs),
)
trainer_fm.add_task(
    "post_training",
    lambda trainer, *args, **kwargs: swa_manager.post_training(trainer, *args, **kwargs),
)

trainer_fm.set_dataset(dataset_frac["training"], stage="training")
trainer_fm.set_dataset(dataset_frac["validation"], stage="validation", include_all=True)
if "testing" in dataset_frac:
    trainer_fm.set_dataset(dataset_frac["testing"], stage="testing", include_all=True)

# -------------------------
# Run training and save results
# -------------------------
epochs = train_config["optimizer"]["epochs"]
trainer_fm.train(epochs)

raw_final_params = tree_util.tree_map(lambda x: x, trainer_fm.params)
raw_best_params = tree_util.tree_map(lambda x: x, trainer_fm.best_params)

swa_params_available = swa_manager.active and (swa_manager.swa_params is not None)
selected_eval_params = raw_best_params
if swa_params_available and swa_manager.prefer:
    selected_eval_params = swa_manager.swa_params
    print("[SWA] Using SWA params for eval/save path.")
elif swa_params_available:
    print("[SWA] SWA params available, but keeping raw params for eval/save path.")

trainer_fm.save_trainer(f"{output_dir}/trainer.pkl", format=".pkl")
_save_params(f"{output_dir}/best_params.pkl", selected_eval_params)
_save_params(f"{output_dir}/final_params.pkl", raw_final_params)
if swa_params_available:
    _save_params(f"{output_dir}/swa_params.pkl", swa_manager.swa_params)

# Save configs as json
with open(f"{output_dir}/config.json", "w") as f:
    json.dump(MACE_CONFIG, f, indent=4)
# Save training config as json (convert OrderedDict to regular dict for JSON serialization)
with open(f"{output_dir}/train_config.json", "w") as f:
    json.dump(
        dict(train_config),
        f,
        indent=4,
        default=lambda o: dict(o) if isinstance(o, OrderedDict) else o,
    )

# -------------------------
# Plot and save results
# -------------------------
from cgbench.plotting.training import plot_predictions, plot_convergence

# Plot training convergence
plot_convergence(trainer_fm, output_dir)

batch_size = train_config["optimizer"]["batch"]

predictions_val = trainer_fm.predict(
    dataset_frac["validation"],
    selected_eval_params,
    batch_size=batch_size,
)
predictions_val = tree_util.tree_map(onp.asarray, predictions_val)
plot_predictions(
    predictions_val, dataset_frac["validation"], output_dir, name="preds_validation"
)

if "testing" in dataset_frac:
    predictions_test = trainer_fm.predict(
        dataset_frac["testing"],
        selected_eval_params,
        batch_size=batch_size,
    )
    predictions_test = tree_util.tree_map(onp.asarray, predictions_test)
    onp.savez(f"{output_dir}/predictions_test.npz", **predictions_test)
    plot_predictions(
        predictions_test, dataset_frac["testing"], output_dir, name="preds_testing"
    )
