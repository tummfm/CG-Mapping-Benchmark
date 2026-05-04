"""Test: load the capped_ala test dataset using the production BaseDataset workflow.

Files under tests/capped_ala/:
  md.tpr       – GROMACS topology (used for topology metadata)
  md.gro       – single-frame structure (config fallback)
  first200.trr – empty-frame TRR (no positions/forces stored)
  topol.top    – GROMACS topology text

Because the TRR has no stored coordinates or forces, the test synthesises a
small NPZ (20 frames, 22 atoms) and places it at a temp path so that the
production load_traj() code-path (npz → train/val split) is exercised.
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cgbench.core.config import MD_DATASET_PATHS
from cgbench.core.dataset import Capped_Ala_Dataset

# ---------------------------------------------------------------------------
# Test data location
# ---------------------------------------------------------------------------
TEST_DIR = os.path.join(os.path.dirname(__file__), "capped_ala")
N_ATOMS = 22   # ACE(6) + ALA(10) + NME(6) — must match md.tpr
N_FRAMES = 20  # small synthetic trajectory
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Build a synthetic NPZ that matches the topology atom count
# ---------------------------------------------------------------------------
_tmpdir = tempfile.mkdtemp(prefix="cgbench_test_")
_NPZ_PATH = os.path.join(_tmpdir, "capped_ala_test.npz")

# Positions in nm — keep them inside a small box so PBC makes sense
R_synth = RNG.uniform(0.0, 1.0, (N_FRAMES, N_ATOMS, 3)).astype(np.float32)
F_synth = RNG.normal(0.0, 100.0, (N_FRAMES, N_ATOMS, 3)).astype(np.float32)

np.savez_compressed(_NPZ_PATH, R=R_synth, F=F_synth)
print(f"Synthetic NPZ written: {_NPZ_PATH}  (shape R={R_synth.shape})")

# ---------------------------------------------------------------------------
# Redirect dataset config to local test files + synthetic NPZ
# ---------------------------------------------------------------------------
MD_DATASET_PATHS["capped_ala"].update(
    {
        "path": _NPZ_PATH,
        "config": os.path.join(TEST_DIR, "md.gro"),
        "topology": os.path.join(TEST_DIR, "md.tpr"),
        "selection": "not resname SOL WAT HOH",
    }
)
# Remove any traj/traj_forces keys so load_traj() hits the npz branch.
MD_DATASET_PATHS["capped_ala"].pop("traj", None)
MD_DATASET_PATHS["capped_ala"].pop("traj_forces", None)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def check(condition, msg):
    if not condition:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  OK  {msg}")


# ---------------------------------------------------------------------------
# 1. Instantiate dataset (loads topology, builds map object)
# ---------------------------------------------------------------------------
print("\n=== 1. Instantiating Capped_Ala_Dataset ===")
ds = Capped_Ala_Dataset(train_ratio=0.7, val_ratio=0.1, shuffle=False, cache_cg=False)

# --- species ---
species = np.asarray(ds.species)
print(f"  n_atoms  : {len(species)}")
print(f"  n_species: {ds.n_species}")
check(len(species) == N_ATOMS, f"n_atoms == {N_ATOMS}")
check(ds.n_species > 0, "n_species > 0")

# --- box ---
box = np.asarray(ds.box)
print(f"  box shape: {box.shape}")
print(f"  box:\n{box}")
check(box.shape == (3, 3), f"box shape is (3,3), got {box.shape}")
check(np.all(np.diag(box) > 0), "box diagonal entries are positive")

# --- bonds / angles / dihedrals ---
bonds = np.asarray(ds.bonds) if ds.bonds is not None and len(ds.bonds) > 0 else None
angles = np.asarray(ds.angles) if ds.angles is not None and len(ds.angles) > 0 else None
dihedrals = ds.dihedrals

print(f"  bonds     : {bonds.shape if bonds is not None else None}")
print(f"  angles    : {angles.shape if angles is not None else None}")
print(f"  dihedrals : {np.asarray(dihedrals).shape if dihedrals is not None and len(dihedrals) > 0 else None}")
check(bonds is not None and len(bonds) > 0, "bonds are loaded and non-empty")
check(bonds.ndim == 2 and bonds.shape[1] == 2, "bonds array has shape (N, 2)")
check(angles is not None and len(angles) > 0, "angles are loaded and non-empty")
check(angles.ndim == 2 and angles.shape[1] == 3, "angles array has shape (N, 3)")
check(
    dihedrals is not None and len(dihedrals) > 0,
    "dihedrals are loaded and non-empty",
)

# Bond indices must reference valid atom indices
check(np.all(bonds >= 0) and np.all(bonds < N_ATOMS), "bond indices in [0, n_atoms)")

# --- residue / atom names ---
res_names = ds.topology_residue_names
atom_names = ds.topology_atom_names
res_ids = ds.topology_residue_ids
print(f"  residue_names (per atom): {res_names}")
print(f"  atom_names              : {atom_names}")
check(res_names is not None, "topology_residue_names is set")
check(atom_names is not None, "topology_atom_names is set")
check(len(res_names) == N_ATOMS, "residue_names has one entry per atom")
check(len(atom_names) == N_ATOMS, "atom_names has one entry per atom")

for expected_res in ("ACE", "ALA", "NME"):
    check(
        expected_res in res_names,
        f"residue '{expected_res}' is present in topology_residue_names",
    )

# Unique residue sequence should be ACE → ALA → NME
unique_res = list(dict.fromkeys(res_names))  # order-preserving dedup
check(unique_res == ["ACE", "ALA", "NME"], f"residue order is ACE/ALA/NME, got {unique_res}")

# --- masses ---
masses = np.asarray(ds.masses)
print(f"  masses shape: {masses.shape}  (first 5: {masses[:5]})")
check(len(masses) == N_ATOMS, "masses length matches n_atoms")
check(np.all(masses > 0), "all masses are positive")

# --- available CG maps ---
available_maps = ds.map_obj.get_available_maps()
print(f"  available CG maps: {available_maps}")
check(len(available_maps) > 0, "at least one CG map is available")
for expected_map in ("CA", "martini3"):
    check(expected_map in available_maps, f"expected map '{expected_map}' is available")

# ---------------------------------------------------------------------------
# 2. Load trajectory (will read from the synthetic NPZ)
# ---------------------------------------------------------------------------
print("\n=== 2. Loading trajectory (load_traj) ===")
ds.load_traj()

check(ds.dataset_X is not None, "dataset_X is populated after load_traj()")
check(ds.dataset_U is not None, "dataset_U (fractional) is populated after load_traj()")
check("training" in ds.dataset_X, "'training' split exists")
check("validation" in ds.dataset_X, "'validation' split exists")

for split in ds.splits:
    R = np.asarray(ds.dataset_X[split]["R"])
    F = np.asarray(ds.dataset_X[split]["F"])
    sp = np.asarray(ds.dataset_X[split]["species"])
    box_sp = np.asarray(ds.dataset_X[split]["box"])
    mask = np.asarray(ds.dataset_X[split]["mask"])

    n_frames, n_atoms, _ = R.shape
    print(f"\n  [{split}]")
    print(f"    R shape    : {R.shape}  dtype={R.dtype}")
    print(f"    F shape    : {F.shape}  dtype={F.dtype}")
    print(f"    species    : {sp.shape}")
    print(f"    box        : {box_sp.shape}")
    print(f"    mask       : {mask.shape}")

    check(n_frames > 0, f"{split}: n_frames > 0")
    check(n_atoms == N_ATOMS, f"{split}: trajectory n_atoms == {N_ATOMS}")
    check(R.shape == F.shape, f"{split}: R and F have same shape")
    check(R.dtype == np.float32, f"{split}: R is float32")
    check(F.dtype == np.float32, f"{split}: F is float32")
    check(sp.shape == (n_frames, N_ATOMS), f"{split}: species shape is (n_frames, n_atoms)")
    check(box_sp.shape == (n_frames, 3, 3), f"{split}: box shape is (n_frames,3,3)")
    check(mask.shape == (n_frames, N_ATOMS), f"{split}: mask shape is (n_frames, n_atoms)")
    check(mask.all(), f"{split}: all mask entries are True")

    # Fractional coords
    R_frac = np.asarray(ds.dataset_U[split]["R"])
    check(
        R_frac.shape == R.shape,
        f"{split}: fractional dataset_U has same shape as dataset_X",
    )

n_train = ds.dataset_X["training"]["R"].shape[0]
n_val = ds.dataset_X["validation"]["R"].shape[0]
n_total_splits = n_train + n_val
if "testing" in ds.dataset_X:
    n_total_splits += ds.dataset_X["testing"]["R"].shape[0]
print(f"\n  n_train={n_train}  n_val={n_val}  n_total={n_total_splits}")
check(n_train > n_val, "training split is larger than validation split")
check(n_total_splits == N_FRAMES, "all frames are accounted for across splits")

# Displacement functions
check(ds.displacement_fn_U is not None, "displacement_fn_U is initialised")
check(ds.displacement_fn_X is not None, "displacement_fn_X is initialised")

# ---------------------------------------------------------------------------
# 3. Coarse-grain with the first available map
# ---------------------------------------------------------------------------
map_name = available_maps[0]
print(f"\n=== 3. Coarse-graining with map='{map_name}' ===")
ds.coarse_grain(map_name)

check(hasattr(ds, "cg_dataset_X"), "cg_dataset_X exists after coarse_grain()")
check(hasattr(ds, "cg_species"), "cg_species exists after coarse_grain()")
check(hasattr(ds, "cg_bond_index"), "cg_bond_index exists after coarse_grain()")

cg_species = np.asarray(ds.cg_species)
print(f"  n_cg_sites  : {ds.n_cg_sites}")
print(f"  n_cg_species: {ds.n_cg_species}")
print(f"  cg_species  : {cg_species}")
check(ds.n_cg_sites > 0, "n_cg_sites > 0")
check(ds.n_cg_sites < N_ATOMS, "n_cg_sites < n_atoms (reduction happened)")

for split in ds.splits:
    R_cg = np.asarray(ds.cg_dataset_X[split]["R"])
    F_cg = np.asarray(ds.cg_dataset_X[split]["F"])
    n_frames_cg, n_cg, _ = R_cg.shape
    print(f"\n  [{split}] cg R: {R_cg.shape}  cg F: {F_cg.shape}")
    check(n_cg == ds.n_cg_sites, f"{split}: CG site count matches n_cg_sites")
    check(n_frames_cg > 0, f"{split}: CG dataset has frames")
    check(R_cg.dtype == np.float32, f"{split}: CG R is float32")

# CG topology
cg_bonds = np.asarray(ds.cg_bond_index)
print(f"  cg_bond_index shape: {cg_bonds.shape}")
check(cg_bonds.ndim == 2, "cg_bond_index is 2-D")

# CG masses and weights
cg_masses = np.asarray(ds.cg_masses)
check(len(cg_masses) == ds.n_cg_sites, "cg_masses length matches n_cg_sites")
check(np.all(cg_masses > 0), "all CG masses are positive")

# ---------------------------------------------------------------------------
print("\n=== All checks passed ===\n")
