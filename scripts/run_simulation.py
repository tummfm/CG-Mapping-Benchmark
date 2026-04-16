"""Unified simulation script (backend loaded from config.json)."""

import argparse
import contextlib
import json
import os
import pickle
import sys
import time

from utils import (
	allocate_neighborlist,
	build_mace_config,
	configure_runtime_environment,
	init_mace_model_and_template,
	init_nequip_model,
	load_model_artifacts,
	load_simulation_dataset,
	prepare_dataset_bundle,
)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--model", type=str, help="Path to best_params.pkl", required=True)
parser.add_argument("--mol", type=str, help="Molecule to simulate", required=True)
parser.add_argument("--verbose", action="store_true", default=True)
parser.add_argument(
	"--xla_mem_fraction",
	type=float,
	default=None,
	help="Override XLA_PYTHON_CLIENT_MEM_FRACTION.",
)
parser.add_argument(
	"--equilibrate",
	type=float,
	default=0.0,
	metavar="TIME_IN_PS",
	help="Run an equilibration stage before production.",
)
parser.add_argument("--dt", type=float, default=None, metavar="DT_IN_FS")
parser.add_argument("--gamma", type=float, default=None)
parser.add_argument(
	"--dt-values-fs",
	type=float,
	nargs="+",
	default=None,
	metavar="DT_IN_FS",
)
parser.add_argument("--print-every", type=float, default=None)
parser.add_argument("--sim-mode", type=str, default=None)
parser.add_argument("--ensemble", type=str, default=None)
parser.add_argument("--t-total", type=float, default=None)
parser.add_argument("--n-chains", type=int, default=None)
parser.add_argument("--T", type=float, default=None)
parser.add_argument("--prngkey-seed", type=int, default=None)
args = parser.parse_args()

configure_runtime_environment(
	device=args.device,
	default_mem_fraction="0.99",
	xla_mem_fraction=args.xla_mem_fraction,
	respect_existing=True,
)

import cloudpickle as _cpickle
import numpy as onp
import jax
from jax import numpy as jnp, random, tree_util

from chemtrain.ensemble import sampling
from chemtrain import quantity
from jax_md import simulate, space
from jax_md_mod import custom_quantity
from cgbench.core.config import DEFAULT_SIM_CONFIG as SIM_CONFIG


def _apply_sim_overrides(config):
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
	if args.T is not None:
		config["T"] = float(args.T)
	if args.prngkey_seed is not None:
		config["PRNGKey_seed"] = int(args.prngkey_seed)
	if args.dt is not None:
		if args.dt <= 0:
			raise ValueError(f"--dt must be positive (fs). Got {args.dt}.")
		config["dt_values_fs"] = [float(args.dt)]


model_path = args.model
base_dir, MODEL_CONFIG, TRAIN_CONFIG = load_model_artifacts(model_path)

if "model" not in MODEL_CONFIG:
	raise ValueError(
		"config.json is missing required field 'model'. "
		"Please retrain with run_fm_training.py --model mace|nequip so this metadata is saved."
	)

backend = str(MODEL_CONFIG["model"]).strip().lower()
if backend not in {"mace", "nequip"}:
	raise ValueError(
		f"Invalid config.json model='{MODEL_CONFIG['model']}'. Expected 'mace' or 'nequip'."
	)

print(f"[Model] Detected backend from config.json: {backend}")

config = SIM_CONFIG.copy()
config["sim_mol"] = args.mol
config["type"] = MODEL_CONFIG["type"]
config["cg_map"] = MODEL_CONFIG.get("CG_map", None)
config["t_eq"] = float(args.equilibrate)
_apply_sim_overrides(config)
config["kT"] = config["T"] * quantity.kb

if args.verbose:
	print("-" * 50)
	for key, value in MODEL_CONFIG.items():
		print(f"Found model config: {key}: {value}")
	print("-" * 50)
	for key, value in config.items():
		print(f"Using Sim config: {key}: {value}")
	print("-" * 50)

config["sim_mol"], data, config["nmol"] = load_simulation_dataset(
	mol=config["sim_mol"],
	train_ratio=MODEL_CONFIG["train_ratio"],
	val_ratio=MODEL_CONFIG["val_ratio"],
	cg_map=MODEL_CONFIG.get("CG_map"),
	verbose=args.verbose,
)

bundle = prepare_dataset_bundle(
	data,
	MODEL_CONFIG["type"],
	MODEL_CONFIG.get("CG_map"),
	use_position_dataset=False,
)
dataset = bundle["dataset"]
species = bundle["species"]
masses = bundle["masses"]
n_species = bundle["n_species"]
box = bundle["box"]
displacement_fn = bundle["displacement_fn"]

print(f"[SPECIES]: {species}")

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = allocate_neighborlist(
	dataset["training"],
	displacement_fn,
	box,
	MODEL_CONFIG["r_cutoff"],
	format_name="Sparse",
	batch_size=10,
	capacity_multiplier=1.2,
)

species_init = jnp.asarray(species)

if backend == "mace":
	mace_cfg = build_mace_config(
		MODEL_CONFIG,
		use_so3=MODEL_CONFIG.get("use_so3", False),
	)
	_init_params, _gnn_energy_fn_template, _model_config = init_mace_model_and_template(
		displacement_fn,
		MODEL_CONFIG["r_cutoff"],
		avg_num_neighbors,
		species_default=species_init,
		mace_cfg=mace_cfg,
		n_species=100,
		per_particle=False,
		use_custom_batch_fn=False,
		use_so3=MODEL_CONFIG.get("use_so3", False),
		enable_cueq=not MODEL_CONFIG.get("use_so3", False),
	)
else:
	_init_fn, _nequip_energy_fn = init_nequip_model(
		displacement_fn,
		MODEL_CONFIG["r_cutoff"],
		n_species,
		max_edges,
		avg_num_neighbors,
	)

	def _gnn_energy_fn_template(params):
		def energy_fn(pos, neighbor, **dynamic_kwargs):
			dynamic_kwargs.setdefault("species", species_init)
			dynamic_kwargs.setdefault(
				"mask",
				jnp.ones(pos.shape[0], dtype=jnp.bool_),
			)
			return _nequip_energy_fn(params, pos, neighbor, **dynamic_kwargs)

		return energy_fn


def _wrap_with_prior_if_needed(base_energy_fn_template):
	prior_type = MODEL_CONFIG.get("prior")
	if prior_type is None:
		return base_energy_fn_template

	if prior_type == "bi":
		prior_pkl = os.path.join(base_dir, "priors_bi.pkl")
		if not os.path.exists(prior_pkl):
			raise FileNotFoundError(
				f"config.json specifies prior='bi' but priors_bi.pkl not found in {base_dir}."
			)
		from cgbench.core.prior import get_prior_energy_fn_template as build_prior
	elif prior_type == "1dfunc":
		prior_pkl = os.path.join(base_dir, "priors_1dfunc.pkl")
		if not os.path.exists(prior_pkl):
			raise FileNotFoundError(
				f"config.json specifies prior='1dfunc' but priors_1dfunc.pkl not found in {base_dir}."
			)
		from cgbench.core.prior import (
			get_1dfunc_prior_energy_fn_template as build_prior,
		)
	else:
		raise ValueError(f"Unknown prior type in config.json: '{prior_type}'")

	with open(prior_pkl, "rb") as f:
		prior_data = _cpickle.load(f)

	prior_template = build_prior(prior_data, displacement_fn)
	print(
		f"[Prior] Loaded '{prior_type}' priors: "
		f"{len(prior_data.get('bonds', {}))} bonds, "
		f"{len(prior_data.get('angles', {}))} angles, "
		f"{len(prior_data.get('dihedrals', {}))} dihedrals."
	)

	def combined_template(params):
		gnn_fn = base_energy_fn_template(params)
		prior_fn = prior_template(params)

		def energy_fn(pos, neighbor, **kwargs):
			return gnn_fn(pos, neighbor, **kwargs) + prior_fn(pos, neighbor, **kwargs)

		return energy_fn

	print(f"[Prior] energy_fn_template = GNN + {prior_type} prior.")
	return combined_template


energy_fn_template = _wrap_with_prior_if_needed(_gnn_energy_fn_template)

if args.verbose:
	print(f"Max neighbors: {max_neighbors}, max edges: {max_edges}")

if not os.path.exists(model_path):
	raise FileNotFoundError(f"Model file {model_path} not found.")
energy_params = onp.load(model_path, allow_pickle=True)
energy_params = tree_util.tree_map(jnp.asarray, energy_params)
energy_fn = energy_fn_template(energy_params)

outdir = f"{base_dir}/simulation_{config['ensemble']}_T={config['T']}K/"
os.makedirs(outdir, exist_ok=True)


def init_simulator(
	dataset_dict,
	energy_fn_impl,
	masses_arr,
	nbrs,
	kT,
	dt,
	n_chains,
	gamma,
	t_eq,
	t_total,
	r_init_override=None,
):
	key = random.PRNGKey(config["PRNGKey_seed"])
	prosol_uncapped = {"1UBQ", "1IFC", "1MJC", "1QX5", "6LYT"}
	if config["sim_mol"] in prosol_uncapped and n_chains != 1:
		raise ValueError(f"{config['sim_mol']} simulation only supports n_chains=1.")

	if "box" in dataset_dict["validation"]:
		_, shift_fn = space.periodic_general(
			dataset_dict["validation"]["box"][0],
			fractional_coordinates=True,
		)
	else:
		_, shift_fn = space.free()

	if r_init_override is not None:
		r_init = jnp.asarray(r_init_override)
	elif config["sim_mol"] in prosol_uncapped:
		r_init = dataset_dict["training"]["R"][:1]
	elif config["sim_mode"] == "stability":
		r_init = dataset_dict["validation"]["R"][0]
		r_init = jnp.tile(r_init, (n_chains, 1, 1))
	elif config["sim_mode"] == "helix":
		indices = onp.load("/home/franz/Ala15_100_min_helix_indices.npy", allow_pickle=True)
		combined_dataset = onp.concatenate(
			[dataset_dict["training"]["R"], dataset_dict["validation"]["R"]], axis=0
		)
		if int(jnp.max(indices)) >= combined_dataset.shape[0]:
			raise ValueError("Helix indices exceed dataset size.")
		r_init = combined_dataset[indices]
	elif config["sim_mode"] == "speed":
		key, split = random.split(key)
		selection = random.choice(
			split,
			jnp.arange(dataset_dict["validation"]["R"].shape[0]),
			shape=(1,),
			replace=False,
		)
		r_init = dataset_dict["validation"]["R"][selection]
	else:
		key, split = random.split(key)
		selection = random.choice(
			split,
			jnp.arange(dataset_dict["validation"]["R"].shape[0]),
			shape=(n_chains,),
			replace=False,
		)
		r_init = dataset_dict["validation"]["R"][selection]

	if config["ensemble"] == "NVT":
		init_simulator_fn = simulate.nvt_langevin
		sim_kwargs = {"kT": kT, "gamma": gamma, "dt": dt}
		init_sim_kwargs = {"mass": masses_arr, "neighbor": nbrs}
	elif config["ensemble"] == "NVE":
		init_simulator_fn = simulate.nve
		sim_kwargs = {"dt": dt, "kT": kT}
		init_sim_kwargs = {"mass": masses_arr, "neighbor": nbrs, "kT": kT}
	else:
		raise ValueError(f"Unknown ensemble: {config['ensemble']}. Use NVT or NVE.")

	init_ref_state, sim_template = sampling.initialize_simulator_template(
		init_simulator_fn,
		shift_fn=shift_fn,
		nbrs=init_sim_kwargs["neighbor"],
		init_with_PRNGKey=True,
		extra_simulator_kwargs=sim_kwargs,
	)

	key, split = random.split(key)
	reference_state = init_ref_state(
		split,
		r_init,
		energy_or_force_fn=energy_fn_impl,
		init_sim_kwargs=init_sim_kwargs,
	)

	eval_timings = sampling.process_printouts(
		time_step=dt,
		total_time=t_total,
		t_equilib=t_eq,
		print_every=config["print_every"],
	)

	quantities = {
		"kT": custom_quantity.temperature,
		"epot": custom_quantity.energy_wrapper(lambda _: energy_fn_impl),
	}

	traj_gen = sampling.trajectory_generator_init(
		sim_template,
		lambda _: energy_fn_impl,
		eval_timings,
		quantities=quantities,
		vmap_sim_batch=config["n_chains"],
		vmap_batch=config["n_chains"],
	)
	return reference_state, jax.jit(traj_gen)


def visualise(traj_path, dataset_obj):
	from cgbench.plotting import molecules as visualise_traj

	vis_fn_map = {
		"capped_ala": visualise_traj.vis_capped_ala,
		"capped_ala2": visualise_traj.vis_capped_ala,
		"tip3p": visualise_traj.vis_tip3p_water,
		"tip3p-water": visualise_traj.vis_tip3p_water,
		"hexane": visualise_traj.vis_hexane,
		"benzene_crystal": visualise_traj.vis_benzene_crystal,
		"capped_ala15": visualise_traj.vis_capped_ala15,
		"capped_pro": visualise_traj.vis_capped_pro,
		"capped_thr": visualise_traj.vis_capped_thr,
		"capped_gly": visualise_traj.vis_capped_gly,
		"3bpa": visualise_traj.vis_3bpa,
		"3bpa_biased": visualise_traj.vis_3bpa,
		"ThreeBPA_biased": visualise_traj.vis_3bpa,
		"azobenzene_biased": visualise_traj.vis_azobenzene,
		"Azobenzene_biased": visualise_traj.vis_azobenzene,
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
			type=MODEL_CONFIG["type"],
			dataset=dataset_obj,
			cg_map=MODEL_CONFIG["CG_map"],
		)
	elif str(config["sim_mol"]).lower() in ("cath", "cath_full", "cath_quarter", "cath_test"):
		print(f"No dedicated visualizer for {config['sim_mol']}; skipping plotting.")
	else:
		raise ValueError(f"No visualizer registered for molecule: {config['sim_mol']}")


def _extract_last_positions(traj_pos, n_chains):
	arr = jnp.asarray(traj_pos)
	if arr.ndim == 4:
		return arr[:, -1, :, :]
	if arr.ndim == 3:
		if n_chains != 1:
			raise ValueError("Single-chain trajectory received but n_chains != 1.")
		return arr[-1:, :, :]
	raise ValueError(f"Unexpected trajectory position shape: {arr.shape}")


def _save_outputs_and_plot(save_dir, traj_state, dt_ps, t_total_ps, box_obj, dataset_obj):
	with open(os.path.join(save_dir, "trajectory.pkl"), "wb") as f:
		pickle.dump(traj_state.trajectory.position, f)

	with open(os.path.join(save_dir, "traj_state_aux.pkl"), "wb") as f:
		pickle.dump(traj_state.aux, f)

	config_ = config.copy()
	config_["dt"] = dt_ps
	config_["t_total"] = t_total_ps
	config_.pop("dt_values_fs", None)
	config_["box"] = float(box_obj[0][0]) if box_obj is not None else None

	with open(os.path.join(save_dir, "traj_config.json"), "w") as cf:
		json.dump(config_, cf, indent=4)

	try:
		visualise(save_dir, dataset_obj)
	except Exception as e:
		print(f"Error during visualisation: {e}")


class _TeeStream:
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
	with open(log_path, "a", buffering=1) as log_file:
		tee_stdout = _TeeStream(sys.__stdout__, log_file)
		tee_stderr = _TeeStream(sys.__stderr__, log_file)
		with (
			contextlib.redirect_stdout(tee_stdout),
			contextlib.redirect_stderr(tee_stderr),
		):
			yield


def _format_dt_for_path(dt_fs):
	dt_val = float(dt_fs)
	if dt_val.is_integer():
		return str(int(dt_val))
	return format(dt_val, "g")


dt_values_fs = config["dt_values_fs"]
dt_values_ps = [dt_fs * 0.001 for dt_fs in dt_values_fs]

for dt_fs, dt_ps in zip(dt_values_fs, dt_values_ps):
	config["dt"] = dt_ps
	dt_fs_label = _format_dt_for_path(dt_fs)
	folder_name = (
		f"traj_mol={config['sim_mol']}_"
		f"dt={dt_fs_label}_"
		f"teq={config['t_eq']}_"
		f"t={config['t_total']}_"
		f"nmol={config['nmol']}_"
		f"nchains={config['n_chains']}_"
		f"mode={config['sim_mode']}_"
		f"seed={config['PRNGKey_seed']}_"
		f"gamma={config['gamma']}/"
	)
	save_dir = os.path.join(outdir, folder_name)
	log_path = os.path.join(save_dir, "simulation.log")
	os.makedirs(save_dir, exist_ok=True)

	with _tee_prints_to_file(log_path):
		print(f"Starting simulation for dt = {dt_fs} fs ({dt_ps} ps)")
		print(f"Logging simulation output to {log_path}")

		if os.path.exists(os.path.join(save_dir, "trajectory.pkl")):
			print(f"Found existing trajectory in {save_dir}; skipping run.")
			visualise(save_dir, data)
			continue

		r_init_override = None
		equil_time_ps = float(args.equilibrate)
		if equil_time_ps > 0.0:
			equil_dt_ps = 0.002
			equil_dir = os.path.join(save_dir, f"equilibration_t={equil_time_ps}ps")
			os.makedirs(equil_dir, exist_ok=True)

			print(f"Starting equilibration for {equil_time_ps} ps at dt=0.5 fs")
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
			print(
				f"Equilibration completed in {eq_elapsed / max(1, config['n_chains']):.2f} seconds "
				f"(normalized by n_chains={config['n_chains']})."
			)

			_save_outputs_and_plot(
				save_dir=equil_dir,
				traj_state=eq_traj_state,
				dt_ps=equil_dt_ps,
				t_total_ps=equil_time_ps,
				box_obj=box,
				dataset_obj=data,
			)
			r_init_override = _extract_last_positions(
				eq_traj_state.trajectory.position,
				config["n_chains"],
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

		start_time = time.time()
		traj_state = traj_generator(None, reference_state)
		elapsed_time = time.time() - start_time

		print(
			f"Simulation completed in {elapsed_time / max(1, config['n_chains']):.2f} seconds "
			f"(normalized by n_chains={config['n_chains']})."
		)

		total_sim_time_ps = config["t_total"] * config["n_chains"]
		ns_per_day = (total_sim_time_ps / elapsed_time) * (86400 / 1000)
		print(f"Performance: {ns_per_day:.2f} ns/day")

		_save_outputs_and_plot(
			save_dir=save_dir,
			traj_state=traj_state,
			dt_ps=dt_ps,
			t_total_ps=config["t_total"],
			box_obj=box,
			dataset_obj=data,
		)

		print(f"Finished dt = {dt_fs} fs. Results saved to {save_dir}.")
