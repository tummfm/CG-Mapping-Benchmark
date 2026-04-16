"""Unified force-matching training script (MACE default)."""

import argparse
import copy
import json
import os
from collections import OrderedDict

from utils import (
	apply_stride,
	allocate_neighborlist,
	build_mace_config,
	configure_runtime_environment,
	drop_energy_targets,
	init_mace_model_and_template,
	init_nequip_model,
	load_training_dataset,
	prepare_dataset_bundle,
)


parser = argparse.ArgumentParser(description="Train force-matching model (MACE or NequIP)")
parser.add_argument(
	"--model",
	type=str,
	required=True,
	choices=["mace", "nequip"],
	help="Model backend to train.",
)
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, help="CG mapping to use", required=True)
parser.add_argument("--mol", type=str, help="Molecule to use", required=True)
parser.add_argument("--stride", type=int, help="Subsample dataset", default=None)
parser.add_argument("--rcut", type=float, help="Cutoff radius", default=0.5)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--swa", action="store_true", help="Enable SWA")
parser.add_argument("--swa-start", type=int, default=None)
parser.add_argument("--swa-every", type=int, default=1)
parser.add_argument("--swa-min-snapshots", type=int, default=2)
parser.add_argument("--no-swa-prefer", action="store_true")
parser.add_argument("--use-so3", action="store_true")
parser.add_argument("--batch-size", type=int, default=None)
parser.add_argument("--init-lr", type=float, default=None)
parser.add_argument("--num-epochs", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--decay-rate", type=float, default=None)
parser.add_argument("--optimizer", type=str, default=None)
parser.add_argument(
	"--prior",
	type=str,
	default=None,
	choices=["bi", "1dfunc"],
	help=(
		"Add a fixed bonded prior to the energy function (CG only). "
		"'bi' = tabulated Boltzmann-inversion PMF; "
		"'1dfunc' = fitted harmonic/Fourier prior."
	),
)
args = parser.parse_args()

configure_runtime_environment(
	device=args.device,
	default_mem_fraction="0.97",
	respect_existing=False,
)

import cloudpickle as pickle
import numpy as onp
import optax
from jax import numpy as jnp, random, tree_util

from chemtrain import trainers
from cgbench.core.config import (
	DEFAULT_MACE_CONFIG,
	DEFAULT_NEQUIP_CONFIG,
	DEFAULT_TRAIN_CONFIG,
)


def _compute_swa_start_epoch(total_epochs, explicit_start):
	if explicit_start is not None:
		return max(0, min(explicit_start, max(0, total_epochs - 1)))
	return max(0, total_epochs - max(1, total_epochs // 4))


def _save_params(path, params):
	cpu_params = tree_util.tree_map(onp.asarray, params)
	with open(path, "wb") as f:
		pickle.dump(cpu_params, f)


class SWAManager:
	"""Collect epoch snapshots and maintain a running SWA average."""

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

	def post_epoch(self, trainer, *_, **__):
		epoch = int(trainer._epoch)
		if not self._should_collect(epoch):
			return
		self._average(trainer.params)
		print(
			f"[SWA] Collected snapshot {self.snapshot_count} at epoch {epoch} "
			f"(start={self.start_epoch}, every={self.every})"
		)

	def post_training(self, *_args, **_kwargs):
		self.active = self.enabled and (self.snapshot_count >= self.min_snapshots)
		if self.enabled:
			print(
				f"[SWA] Training finished with {self.snapshot_count} snapshots. "
				f"Active={self.active} (min_snapshots={self.min_snapshots}, prefer={self.prefer})"
			)


def get_train_config(train_defaults):
	return OrderedDict(
		optimizer=OrderedDict(
			init_lr=train_defaults["init_lr"],
			lr_decay=train_defaults["decay_rate"],
			epochs=train_defaults["num_epochs"],
			batch=train_defaults["batch_size"],
			cache=100,
			power="exponential",
			weight_decay=1e-3,
			type="ADAM",
			optimizer_kwargs=OrderedDict(b1=0.9, b2=0.995, eps=1e-8),
		),
		gammas=OrderedDict(U=1e-3, F=1e-2),
		swa=OrderedDict(
			enabled=bool(args.swa),
			start_epoch=args.swa_start,
			every=max(1, int(args.swa_every)),
			min_snapshots=max(1, int(args.swa_min_snapshots)),
			prefer=not bool(args.no_swa_prefer),
		),
	)


def init_optimizer(config, dataset_dict):
	num_samples = 1
	if "U" in dataset_dict["training"]:
		num_samples = dataset_dict["training"]["U"].shape[0]
	elif "F" in dataset_dict["training"]:
		num_samples = dataset_dict["training"]["F"].shape[0]
	else:
		raise ValueError("No energy or force data available.")

	transition_steps = int(config["optimizer"]["epochs"] * num_samples) // config[
		"optimizer"
	]["batch"]

	lr_schedule = optax.exponential_decay(
		config["optimizer"]["init_lr"],
		transition_steps,
		config["optimizer"]["lr_decay"],
	)

	transforms = [
		optax.scale_by_adam(
			b1=config["optimizer"]["optimizer_kwargs"]["b1"],
			b2=config["optimizer"]["optimizer_kwargs"]["b2"],
			eps=config["optimizer"]["optimizer_kwargs"]["eps"],
			eps_root=config["optimizer"]["optimizer_kwargs"]["eps"] ** 0.5,
			nesterov=True,
		),
		optax.transforms.add_decayed_weights(config["optimizer"].get("weight_decay", 0.0)),
	]

	return optax.chain(
		*transforms,
		optax.scale_by_learning_rate(lr_schedule, flip_sign=True),
	)


MODEL_CONFIG = copy.deepcopy(
	DEFAULT_MACE_CONFIG if args.model == "mace" else DEFAULT_NEQUIP_CONFIG
)
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

MODEL_CONFIG["model"] = args.model
MODEL_CONFIG["r_cutoff"] = args.rcut
MODEL_CONFIG["mol"] = args.mol
MODEL_CONFIG["CG_map"] = args.cgmap
MODEL_CONFIG["type"] = "CG" if MODEL_CONFIG["CG_map"] != "AT" else "AT"
if args.model == "mace":
	MODEL_CONFIG["use_so3"] = bool(args.use_so3)

MODEL_CONFIG["mol"], data, used_prestrided_cache = load_training_dataset(
	mol=MODEL_CONFIG["mol"],
	train_ratio=MODEL_CONFIG["train_ratio"],
	val_ratio=MODEL_CONFIG["val_ratio"],
	cg_map=MODEL_CONFIG["CG_map"],
	stride=args.stride,
	verbose=args.verbose,
)

bundle = prepare_dataset_bundle(
	data,
	MODEL_CONFIG["type"],
	MODEL_CONFIG["CG_map"],
	use_position_dataset=True,
)
dataset_raw = bundle["dataset"]
displacement_fn = bundle["displacement_fn"]
box = bundle["box"]
n_species = bundle["n_species"]

if "U" in dataset_raw["training"]:
	print("[CG Train] Dropping energies from dataset")
drop_energy_targets(dataset_raw)

if args.stride is not None:
	if used_prestrided_cache and args.stride == 10:
		print("Stride=10 pre-strided cache loaded; skipping in-memory subsampling.")
	else:
		print(f"Subsampling dataset with stride {args.stride}")
		apply_stride(dataset_raw, args.stride)

if args.verbose:
	print(f"Training set size: {dataset_raw['training']['R'].shape[0]}")
	print(f"Validation set size: {dataset_raw['validation']['R'].shape[0]}")

use_so3_tag = "SO3" if args.use_so3 else "O3"
model_tag = str(MODEL_CONFIG["model"]).lower()
arch_tag = ""
if model_tag == "mace":
	arch_tag = (
		f"_int={MODEL_CONFIG.get('num_interactions')}_"
		f"corr={MODEL_CONFIG.get('correlation')}_"
		f"maxL={MODEL_CONFIG.get('max_ell')}_"
		f"eq={use_so3_tag}"
	)
output_dir = (
	f"outputs/MLP_train_{model_tag}/"
	f"{MODEL_CONFIG['mol'].capitalize()}_"
	f"map={MODEL_CONFIG['CG_map']}_"
	f"tr={MODEL_CONFIG['train_ratio']}_"
	f"rcut={MODEL_CONFIG['r_cutoff']}_"
	f"epochs={TRAIN_CONFIG['num_epochs']}_"
	f"seed={MODEL_CONFIG['PRNGKey_seed']}_"
	f"prior={args.prior}_"
	f"stride={args.stride}"
	f"{arch_tag}"
)
os.makedirs(output_dir, exist_ok=True)

prior_energy_fn_template = None
if args.prior is not None:
	if MODEL_CONFIG["type"] != "CG":
		raise ValueError("--prior is only supported for CG models.")

	MODEL_CONFIG["prior"] = args.prior

	import cloudpickle as _cpickle
	from cgbench.core.prior import BoltzmannPrior, get_prior_energy_fn_template
	from cgbench.plotting.priors import plot_bonded_priors

	print("[Prior] Computing Boltzmann-inversion priors on training split ...")
	_bi = BoltzmannPrior(data, T=300.0)
	all_priors = _bi.compute_all_priors(split="training", cg=True)

	_prior_pkl = f"{output_dir}/priors_bi.pkl"
	with open(_prior_pkl, "wb") as _f:
		_cpickle.dump(all_priors, _f)
	print(f"[Prior] Saved BI priors to {_prior_pkl}")

	_bi_plot = plot_bonded_priors(all_priors, output_dir)
	if _bi_plot:
		print(f"[Prior] Saved BI prior plot to {_bi_plot}")

	if args.prior == "bi":
		prior_energy_fn_template = get_prior_energy_fn_template(
			all_priors,
			displacement_fn,
		)
	elif args.prior == "1dfunc":
		from cgbench.core.prior import (
			fit_1dfunc_priors,
			get_1dfunc_prior_energy_fn_template,
		)
		from cgbench.plotting.priors import plot_1dfunc_priors

		fitted_priors = fit_1dfunc_priors(all_priors)
		_fitted_pkl = f"{output_dir}/priors_1dfunc.pkl"
		with open(_fitted_pkl, "wb") as _f:
			_cpickle.dump(fitted_priors, _f)
		print(f"[Prior] Saved 1dfunc priors to {_fitted_pkl}")

		_fit_plot = plot_1dfunc_priors(all_priors, fitted_priors, output_dir)
		if _fit_plot:
			print(f"[Prior] Saved 1dfunc prior plot to {_fit_plot}")

		prior_energy_fn_template = get_1dfunc_prior_energy_fn_template(
			fitted_priors,
			displacement_fn,
		)

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = allocate_neighborlist(
	dataset_raw["training"],
	displacement_fn,
	box,
	MODEL_CONFIG["r_cutoff"],
	format_name="Dense" if args.model == "mace" else "Sparse",
	batch_size=100,
)

if args.verbose:
	print(
		f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, Avg neighbors: {avg_num_neighbors}"
	)

species_init = jnp.asarray(dataset_raw["training"]["species"][0])
r_init = jnp.asarray(dataset_raw["training"]["R"][0])
mask_init = jnp.asarray(dataset_raw["training"]["mask"][0])

if args.model == "mace":
	mace_cfg = build_mace_config(MODEL_CONFIG, use_so3=args.use_so3)
	init_params, gnn_energy_fn_template, _model_config = init_mace_model_and_template(
		displacement_fn,
		MODEL_CONFIG["r_cutoff"],
		avg_num_neighbors,
		species_default=species_init,
		mace_cfg=mace_cfg,
		n_species=100,
		per_particle=False,
		use_custom_batch_fn=False,
		use_so3=args.use_so3,
		enable_cueq=not args.use_so3,
	)
else:
	init_fn, gnn_energy_fn = init_nequip_model(
		displacement_fn,
		MODEL_CONFIG["r_cutoff"],
		n_species,
		max_edges,
		avg_num_neighbors,
	)

	def gnn_energy_fn_template(params):
		def energy_fn(position, neighbor, **kwargs):
			species = kwargs.pop("species", species_init)
			mask = kwargs.pop("mask", jnp.ones(position.shape[0], dtype=jnp.bool_))
			return gnn_energy_fn(params, position, neighbor, species=species, mask=mask, **kwargs)

		return energy_fn

	key = random.PRNGKey(MODEL_CONFIG["PRNGKey_seed"])
	init_params = init_fn(key, r_init, nbrs_init, species=species_init, mask=mask_init)

if prior_energy_fn_template is not None:
	_prior_eft = prior_energy_fn_template

	def energy_fn_template(params):
		gnn_fn = gnn_energy_fn_template(params)
		prior_fn = _prior_eft(params)

		def energy_fn(position, neighbor, **kwargs):
			return gnn_fn(position, neighbor, **kwargs) + prior_fn(
				position,
				neighbor,
				**kwargs,
			)

		return energy_fn

	print(f"[Prior] energy_fn_template = GNN + {args.prior} prior.")
else:
	energy_fn_template = gnn_energy_fn_template

nbrs_init = nbrs_init.update(r_init, mask=mask_init)

train_config = get_train_config(TRAIN_CONFIG)
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

optimizer_fm = init_optimizer(train_config, dataset_raw)

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
trainer_fm.add_task(
	"post_epoch",
	lambda trainer, *a, **k: swa_manager.post_epoch(trainer, *a, **k),
)
trainer_fm.add_task(
	"post_training",
	lambda trainer, *a, **k: swa_manager.post_training(trainer, *a, **k),
)

trainer_fm.set_dataset(dataset_raw["training"], stage="training")
trainer_fm.set_dataset(dataset_raw["validation"], stage="validation", include_all=True)
if "testing" in dataset_raw:
	trainer_fm.set_dataset(dataset_raw["testing"], stage="testing", include_all=True)

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

with open(f"{output_dir}/config.json", "w") as f:
	json.dump(MODEL_CONFIG, f, indent=4)
with open(f"{output_dir}/train_config.json", "w") as f:
	json.dump(
		dict(train_config),
		f,
		indent=4,
		default=lambda o: dict(o) if isinstance(o, OrderedDict) else o,
	)

from cgbench.plotting.training import plot_convergence, plot_predictions

plot_convergence(trainer_fm, output_dir)

batch_size = train_config["optimizer"]["batch"]
predictions_val = trainer_fm.predict(
	dataset_raw["validation"],
	selected_eval_params,
	batch_size=batch_size,
)
predictions_val = tree_util.tree_map(onp.asarray, predictions_val)
plot_predictions(
	predictions_val,
	dataset_raw["validation"],
	output_dir,
	name="preds_validation",
)

if "testing" in dataset_raw:
	predictions_test = trainer_fm.predict(
		dataset_raw["testing"],
		selected_eval_params,
		batch_size=batch_size,
	)
	predictions_test = tree_util.tree_map(onp.asarray, predictions_test)
	onp.savez(f"{output_dir}/predictions_test.npz", **predictions_test)
	plot_predictions(
		predictions_test,
		dataset_raw["testing"],
		output_dir,
		name="preds_testing",
	)
