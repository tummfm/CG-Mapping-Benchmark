"""One-shot linear force matching for CG spline potentials.

Mirrors run_fm_training.py for the --model spline case, but replaces the
iterative Adam-based training with the normal-equations approach from OpenMSCG:
one pass through the data to build X^T X and X^T y, then a single lstsq solve.
No epochs, no optimizer, no learning-rate schedule.
"""

import argparse
import json
import os
import time

from utils import (
    apply_stride,
    configure_runtime_environment,
    drop_energy_targets,
    load_training_dataset,
)

parser = argparse.ArgumentParser(
    description="CG spline force matching via one-shot normal equations"
)
parser.add_argument("--device", type=str, default=None, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, required=True, help="CG mapping name")
parser.add_argument("--mol", type=str, required=True, help="Molecule name")
parser.add_argument("--stride", type=int, default=None, help="Subsample dataset")
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--n-knots-nb", type=int, default=20,
                    help="Knots for non-bonded splines")
parser.add_argument("--n-knots-bond", type=int, default=20,
                    help="Knots for bond splines")
parser.add_argument("--n-knots-angle", type=int, default=20,
                    help="Knots for angle splines")
parser.add_argument("--n-knots-dihedral", type=int, default=20,
                    help="Knots for dihedral splines")
parser.add_argument("--ridge-alpha", type=float, default=0.0,
                    help="L2 regularisation on spline coefficients (0 = none)")
parser.add_argument("--percentile-lo", type=float, default=0.0,
                    help="Lower percentile for bonded knot range estimation (default 0.0)")
parser.add_argument("--percentile-hi", type=float, default=100.0,
                    help="Upper percentile for bonded knot range estimation (default 100.0)")
parser.add_argument("--train-ratio", type=float, default=0.9)
parser.add_argument("--val-ratio", type=float, default=0.1)
parser.add_argument("--nb-range-frames", type=int, default=500,
                    help="Frames used to estimate non-bonded knot range; "
                         "bonded range uses 4x this value (default 500)")
parser.add_argument("--batch", type=int, default=16,
                    help="Batch size for normal-equation accumulation (default: 16)")
parser.add_argument(
    "--spline-type", type=str, default="b-spline",
    choices=["b-spline", "cubic-spline"],
    help="Spline basis: 'b-spline' (default) or 'cubic-spline'",
)


def _parse_interaction_types(s: str) -> str:
    valid = set("band")
    bad = set(s) - valid
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown interaction character(s): {''.join(sorted(bad))}. "
            "Use b=bond, a=angle, d=dihedral, n=nonbonded."
        )
    return s


parser.add_argument(
    "--type", type=_parse_interaction_types, default="band",
    metavar="TYPES",
    help="Interactions to fit: b=bond a=angle d=dihedral n=nonbonded "
         "(e.g. --type ba fits only bonds and angles; default: band)",
)

args = parser.parse_args()

configure_runtime_environment(device=args.device, xla_mem_fraction="0.97")

import cloudpickle as pickle
import numpy as onp
import jax
import jax.numpy as jnp
from jax_md import partition
from chemtrain.data import preprocessing

from cgbench.core.config import DEFAULT_SPLINE_CONFIG
from cgbench.core.prior import SplineModel, BoltzmannPrior
from cgbench.core.spline import SplineForceMatcher

# ---- model config -----------------------------------------------------------

enabled_types: set[str] = set(args.type)

MODEL_CONFIG = {
    **DEFAULT_SPLINE_CONFIG,
    "model": "spline_lsq",
    "mol": args.mol,
    "CG_map": args.cgmap,
    "type": "CG",
    "interaction_types": args.type,
    "n_knots_nb": args.n_knots_nb,
    "n_knots_bond": args.n_knots_bond,
    "n_knots_angle": args.n_knots_angle,
    "n_knots_dihedral": args.n_knots_dihedral,
    "ridge_alpha": args.ridge_alpha,
    "percentile_lo": args.percentile_lo,
    "percentile_hi": args.percentile_hi,
    "nb_range_frames": args.nb_range_frames,
    "train_ratio": args.train_ratio,
    "val_ratio": args.val_ratio,
    "spline_type": args.spline_type,
}

# ---- dataset ----------------------------------------------------------------

MODEL_CONFIG["mol"], data, _ = load_training_dataset(
    mol=MODEL_CONFIG["mol"],
    train_ratio=MODEL_CONFIG["train_ratio"],
    val_ratio=MODEL_CONFIG["val_ratio"],
    cg_map=MODEL_CONFIG["CG_map"],
    stride=args.stride,
    verbose=args.verbose,
)

data.coarse_grain(MODEL_CONFIG["CG_map"])
dataset_dict = data.cg_dataset_X
species = data.cg_species
displacement_fn = data.displacement_fn_X
box = data.box

# rcut = half the shortest box side (maximum unambiguous cutoff under PBC)
_box_mat = onp.asarray(box, dtype=onp.float64)
rcut = float(0.5 * onp.linalg.norm(_box_mat, axis=1).min())
MODEL_CONFIG["r_cutoff"] = rcut
if args.verbose:
    print(f"[rcut] Derived rcut = {rcut:.4f} nm (half shortest box side)")

drop_energy_targets(dataset_dict)

if args.stride is not None:
    print(f"Subsampling dataset with stride {args.stride}")
    apply_stride(dataset_dict, args.stride)

n_train = dataset_dict["training"]["R"].shape[0]
n_val = dataset_dict["validation"]["R"].shape[0]
if args.verbose:
    print(f"Training frames:   {n_train}")
    print(f"Validation frames: {n_val}")

# ---- output directory -------------------------------------------------------

_nb_tag   = f"_nb={args.n_knots_nb}"       if "n" in enabled_types else ""
_bond_tag = f"_bond={args.n_knots_bond}"   if "b" in enabled_types else ""
_ang_tag  = f"_ang={args.n_knots_angle}"   if "a" in enabled_types else ""
_dih_tag  = f"_dih={args.n_knots_dihedral}" if "d" in enabled_types else ""
arch_tag = (
    f"_type={args.type}"
    f"{_nb_tag}{_bond_tag}{_ang_tag}{_dih_tag}"
    f"_ridge={args.ridge_alpha}"
    f"_stype={args.spline_type}"
)
output_dir = (
    f"outputs/Model=spline_lsq/"
    f"{MODEL_CONFIG['mol'].capitalize()}_"
    f"map={MODEL_CONFIG['CG_map']}_"
    f"tr={MODEL_CONFIG['train_ratio']}_"
    f"stride={args.stride}"
    f"{arch_tag}"
)
os.makedirs(output_dir, exist_ok=True)

# ---- spline model -----------------------------------------------------------

spline_model = SplineModel(
    dataset=data,
    rcut=MODEL_CONFIG["r_cutoff"],
    n_knots_nb=args.n_knots_nb,
    n_knots_bond=args.n_knots_bond,
    n_knots_angle=args.n_knots_angle,
    n_knots_dihedral=args.n_knots_dihedral,
    percentile_lo=args.percentile_lo,
    percentile_hi=args.percentile_hi,
    max_frames_nb=args.nb_range_frames,
    spline_type=args.spline_type,
)

_spline_pkl = f"{output_dir}/spline_model.pkl"

# ---- disable interaction types not in --type --------------------------------

if "b" not in enabled_types:
    spline_model._bond_terms = []
    spline_model._bond_x_grids = onp.empty((0, spline_model.n_knots_bond))
    spline_model._n_bond_types = 0
    print("[SplineModel] Bonds disabled via --type")
if "a" not in enabled_types:
    spline_model._angle_terms = []
    spline_model._angle_x_grids = onp.empty((0, spline_model.n_knots_angle))
    spline_model._n_angle_types = 0
    print("[SplineModel] Angles disabled via --type")
if "d" not in enabled_types:
    spline_model._dihedral_terms = []
    spline_model._dihedral_x_grids = onp.empty((0, spline_model.n_knots_dihedral))
    spline_model._n_dihedral_types = 0
    print("[SplineModel] Dihedrals disabled via --type")
if "n" not in enabled_types:
    spline_model._nb_species_pairs = []
    spline_model._nb_x_grids = onp.empty((0, spline_model.n_knots_nb))
    spline_model._n_nb_types = 0
    print("[SplineModel] Non-bonded disabled via --type")

spline_model.save_data(_spline_pkl)
print(f"[Spline] Saved model topology/grids to {_spline_pkl}")

# ---- print parametrised interactions ----------------------------------------

_sp = onp.asarray(spline_model._cg_species)

print("[SplineModel] Parametrised interactions:")

_rows = []  # (label, grid, unit)

_bond_sp = {}
for i, j, tid in spline_model._bond_terms:
    _bond_sp.setdefault(tid, (int(_sp[i]), int(_sp[j])))
for tid in range(spline_model._n_bond_types):
    si, sj = _bond_sp[tid]
    _rows.append((f"Bond ({si},{sj})", spline_model._bond_x_grids[tid], "nm"))

_angle_sp = {}
for i, j, k, tid in spline_model._angle_terms:
    _angle_sp.setdefault(tid, (int(_sp[i]), int(_sp[j]), int(_sp[k])))
for tid in range(spline_model._n_angle_types):
    si, sj, sk = _angle_sp[tid]
    _rows.append((f"Angle ({si},{sj},{sk})", spline_model._angle_x_grids[tid], "rad"))

_dih_sp = {}
for i, j, k, l, tid in spline_model._dihedral_terms:
    _dih_sp.setdefault(tid, (int(_sp[i]), int(_sp[j]), int(_sp[k]), int(_sp[l])))
for tid in range(spline_model._n_dihedral_types):
    si, sj, sk, sl = _dih_sp[tid]
    _rows.append((f"Dihedral ({si},{sj},{sk},{sl})", spline_model._dihedral_x_grids[tid], "rad"))

for tid, (si, sj) in enumerate(spline_model._nb_species_pairs):
    _rows.append((f"NonBond ({si},{sj})", spline_model._nb_x_grids[tid], "nm"))

_w = max((len(label) for label, _, _ in _rows), default=0)
for label, g, unit in _rows:
    res = (g[-1] - g[0]) / (len(g) - 1)
    print(f"  {label:<{_w}}  xmin={g[0]:8.4f}  xmax={g[-1]:8.4f}  res={res:.6f} {unit}  [{len(g)} knots]")

# ---- neighbor list ----------------------------------------------------------

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
    preprocessing.allocate_neighborlist(
        dataset_dict["training"],
        displacement_fn,
        box,
        r_cutoff=MODEL_CONFIG["r_cutoff"],
        mask_key="mask",
        box_key="box" if box is not None else None,
        format=partition.Sparse,
        batch_size=100,
        capacity_multiplier=2.0,
    )
)
if args.verbose:
    print(f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, "
          f"Avg neighbors: {avg_num_neighbors:.1f}")

# ---- force matcher ----------------------------------------------------------

matcher = SplineForceMatcher(
    spline_model=spline_model,
    displacement_fn=displacement_fn,
    ridge_alpha=args.ridge_alpha,
)
print(f"[SplineLSQ] Total parameters: {matcher.n_params}")

# ---- training loop (accumulate normal equations) ----------------------------

R_train = dataset_dict["training"]["R"]
F_train = dataset_dict["training"]["F"]
masks_train = dataset_dict["training"].get("mask")
boxes_train = dataset_dict["training"].get("box")  # per-frame box or None
species_jax = jnp.asarray(species)

def _get_box(i):
    if boxes_train is not None:
        return jnp.asarray(boxes_train[i])
    return box

BATCH_SIZE = args.batch

def _make_nbrs_batch(batch_start, R_batch, mask_batch):
    B = R_batch.shape[0]
    nbrs_list = []
    for j in range(B):
        box_j = _get_box(batch_start + j)
        nbrs_list.append(nbrs_init.update(
            R_batch[j],
            mask=mask_batch[j] if mask_batch is not None else None,
            box=box_j,
        ))
    return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *nbrs_list)

print(f"[SplineLSQ] Compiling loading function on first batch (B={BATCH_SIZE}) …")
t0 = time.time()
_R0 = jnp.asarray(R_train[:BATCH_SIZE])
_mask0 = jnp.asarray(masks_train[:BATCH_SIZE]) if masks_train is not None else None
_nbrs0 = _make_nbrs_batch(0, _R0, _mask0)
matcher.accumulate_batch(_R0, _nbrs0, F_train[:BATCH_SIZE], _mask0, species_jax)
print(f"[SplineLSQ] Compilation done in {time.time() - t0:.1f}s")

t0 = time.time()
last_report = 0
for batch_start in range(BATCH_SIZE, n_train, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_train)
    R_batch = jnp.asarray(R_train[batch_start:batch_end])
    mask_batch = jnp.asarray(masks_train[batch_start:batch_end]) if masks_train is not None else None
    nbrs_batch = _make_nbrs_batch(batch_start, R_batch, mask_batch)
    matcher.accumulate_batch(R_batch, nbrs_batch, F_train[batch_start:batch_end],
                              mask_batch, species_jax)
    n_done = matcher.n_frames
    if n_done - last_report >= 50_000 or batch_end == n_train:
        elapsed = time.time() - t0
        fps = (n_done - BATCH_SIZE) / elapsed
        print(f"[SplineLSQ] Frame {n_done:>{len(str(n_train))}}/{n_train}"
              f"  {elapsed:.0f}s  ({fps:.1f} frames/s)")
        last_report = n_done

total_acc = time.time() - t0
print(f"[SplineLSQ] Accumulated {matcher.n_frames} frames in {total_acc:.1f}s")

# ---- solve ------------------------------------------------------------------

print("[SplineLSQ] Solving normal equations …")
t_solve = time.time()
params, chi2, rank = matcher.solve()
print(f"[SplineLSQ] Solved in {time.time() - t_solve:.3f}s"
      f"  |  chi2 = {chi2:.6f}  |  rank = {rank}/{matcher.n_params}")

# ---- save params ------------------------------------------------------------

params_path = f"{output_dir}/params.pkl"
cpu_params = {k: onp.asarray(v) for k, v in params.items()}
with open(params_path, "wb") as f:
    pickle.dump(cpu_params, f)
print(f"[SplineLSQ] Saved params to {params_path}")

# ---- validation MSE ---------------------------------------------------------

print("[SplineLSQ] Evaluating on validation set …")
energy_fn = spline_model.get_energy_fn_template(displacement_fn)(params)

@jax.jit
def _predict_forces(R, nbrs, species, mask):
    return -jax.grad(lambda R: energy_fn(R, nbrs, species=species, mask=mask))(R)

R_val = dataset_dict["validation"]["R"]
F_val = dataset_dict["validation"]["F"]
masks_val = dataset_dict["validation"].get("mask")
boxes_val = dataset_dict["validation"].get("box")

F_pred_all = onp.zeros_like(F_val)
sq_sum = 0.0
n_valid_components = 0

for i in range(n_val):
    R_i = jnp.asarray(R_val[i])
    mask_i = jnp.asarray(masks_val[i]) if masks_val is not None else None
    box_i = jnp.asarray(boxes_val[i]) if boxes_val is not None else box
    nbrs_i = nbrs_init.update(R_i, mask=mask_i, box=box_i)
    F_pred_i = onp.asarray(_predict_forces(R_i, nbrs_i, species_jax, mask_i))
    F_pred_all[i] = F_pred_i
    if mask_i is not None:
        m = onp.asarray(mask_i, dtype=bool)
        sq_sum += float(onp.sum((F_pred_i[m] - F_val[i][m]) ** 2))
        n_valid_components += int(m.sum()) * 3
    else:
        sq_sum += float(onp.sum((F_pred_i - F_val[i]) ** 2))
        n_valid_components += F_pred_i.size

val_mse = sq_sum / max(1, n_valid_components)
val_rmse = onp.sqrt(val_mse)
print(f"[SplineLSQ] Validation RMSE: {val_rmse:.4f} kJ/(mol·nm)")

# ---- save config ------------------------------------------------------------

with open(f"{output_dir}/config.json", "w") as f:
    json.dump(MODEL_CONFIG, f, indent=4)

# ---- force prediction plot --------------------------------------------------

from cgbench.plotting.training import plot_predictions

plot_predictions(
    predictions={"F": F_pred_all},
    reference_data={"F": F_val},
    out_dir=output_dir,
    name="preds_validation",
)
print(f"[SplineLSQ] Saved force prediction plot to {output_dir}/preds_validation.png")

# ---- spline shape plot ------------------------------------------------------

from cgbench.plotting.priors import plot_splines

_bi_ref = BoltzmannPrior(data, T=300.0)
_bi_priors = _bi_ref.compute_all_priors(split="training", cg=True,
                                        nb_max_frames=args.nb_range_frames)
_spline_plot = plot_splines(spline_model, params, output_dir, bi_priors=_bi_priors)
if _spline_plot:
    print(f"[SplineLSQ] Saved spline shape plot to {_spline_plot}")
