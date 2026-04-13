from chemtrain.data import preprocessing
import copy
from collections import defaultdict
from jax_md import space
import jax
from jax import numpy as jnp
import numpy as np
import json
import os
from ..utils import io
from .mapping import (
    Hexane_Map,
    BenzeneCrystal_Map,
    CappedPeptideMap,
    TIP3P_Water_Map,
    CATH_Map,
    UncappedProteinMap,
    get_map_weights,
    compute_cg_bond_types,
    map_dataset,
    _derive_cg_topology,
)
from .config import MD_DATASET_PATHS, STATIC_FRAME_DATASET_PATHS, SEED


def _guess_atomic_number_from_name(atom_name: str) -> int:
    """Best-effort atom-name -> atomic number conversion for GROMACS names."""
    clean = "".join([c for c in atom_name if not c.isdigit()]).upper()
    if not clean:
        return 6
    # Prefer explicit two-letter ions/elements first.
    table = {
        "CL": 17,
        "NA": 11,
        "MG": 12,
        "CA": 20,  # ion; alpha-carbon in proteins is handled by fallback below.
        "K": 19,
        "S": 16,
        "P": 15,
        "O": 8,
        "N": 7,
        "C": 6,
        "H": 1,
    }
    # Protein atom names (CA, CB, CG, ...) should map by first letter.
    if clean in ("CA", "CB", "CG", "CD", "CE", "CZ"):
        return 6
    if clean in table:
        return table[clean]
    return table.get(clean[0], 6)


def _load_bonded_from_mda(
    topology_path: str,
    config_path: str,
    selection: str = "all",
) -> tuple:
    """Load a GROMACS .top + .gro pair via MDAnalysis and extract bonded indices.

    Args:
        topology_path: Path to the GROMACS .top file.
        config_path:   Path to the .gro structure file.
        selection:     MDAnalysis selection string to restrict atoms (e.g. "protein").
                       Bonded indices are re-indexed to be local to the selection.

    Returns:
        ag:        MDAnalysis AtomGroup for the selection (full Universe if "all").
        bonds:     (N_bonds, 2)     int32 array of bonded atom-index pairs.
        angles:    (N_angles, 3)    int32 array of angle triples.
        dihedrals: (N_dihedrals, 4) int32 array of proper dihedral quadruples.
    """
    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise ImportError(
            "MDAnalysis is required for topology loading. "
            "Install with `pip install MDAnalysis`."
        ) from e

    u = mda.Universe(topology_path, config_path, topology_format="ITP")
    ag = u.select_atoms(selection) if selection != "all" else u.atoms

    # Build global-index -> local-index map for the selection
    global_to_local = {int(gidx): lidx for lidx, gidx in enumerate(ag.indices)}

    def _remap(indices_global: list[list[int]]) -> np.ndarray:
        return np.array(
            [[global_to_local[i] for i in row] for row in indices_global],
            dtype=np.int32,
        )

    bonds = (
        _remap([[b.atoms[0].index, b.atoms[1].index] for b in ag.bonds])
        if len(ag.bonds) > 0
        else np.zeros((0, 2), dtype=np.int32)
    )
    angles = (
        _remap([[a.atoms[0].index, a.atoms[1].index, a.atoms[2].index] for a in ag.angles])
        if len(ag.angles) > 0
        else np.zeros((0, 3), dtype=np.int32)
    )
    dihedrals = (
        _remap(
            [
                [d.atoms[0].index, d.atoms[1].index, d.atoms[2].index, d.atoms[3].index]
                for d in ag.dihedrals
            ]
        )
        if len(ag.dihedrals) > 0
        else np.zeros((0, 4), dtype=np.int32)
    )

    return ag, bonds, angles, dihedrals


def _load_gromacs_to_dataset(
    topology_path: str,
    trajectory_path: str,
    selection: str = "protein",
) -> dict:
    """Load a GROMACS trajectory into the in-project dataset dict format."""
    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise ImportError(
            "MDAnalysis is required to load 1UBQ from .gro/.xtc. "
            "Install with `pip install MDAnalysis`."
        ) from e

    u = mda.Universe(topology_path, trajectory_path)
    atoms = u.select_atoms(selection)
    n_frames = len(u.trajectory)
    n_atoms = len(atoms)

    if n_atoms == 0:
        raise ValueError(f"Selection '{selection}' returned zero atoms for {topology_path}.")

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    forces = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)  # xtc has no forces
    boxes = np.zeros((n_frames, 3, 3), dtype=np.float32)

    for fi, ts in enumerate(u.trajectory):
        coords[fi] = atoms.positions.astype(np.float32) * 0.1  # Å -> nm
        # Assume orthorhombic box from first three dimensions.
        Lx, Ly, Lz = ts.dimensions[:3]
        boxes[fi] = np.diag(np.array([Lx, Ly, Lz], dtype=np.float32) * 0.1)

    atom_names = np.asarray(atoms.names, dtype=object)
    residue_names = np.asarray(atoms.resnames, dtype=object)
    residue_ids = np.asarray(atoms.resids, dtype=np.int32)
    species = np.asarray(
        [_guess_atomic_number_from_name(n) for n in atom_names], dtype=np.int32
    )

    return {
        "R": coords,
        "F": forces,
        "box": boxes,
        "species": np.tile(species[None, :], (n_frames, 1)),
        "mask": np.ones((n_frames, n_atoms), dtype=bool),
        "atom_names": atom_names,
        "residue_names": residue_names,
        "residue_ids": residue_ids,
    }


def _read_last_gro_box_nm(topology_path: str) -> np.ndarray:
    """Read the final GRO line and return orthorhombic box lengths in nm."""
    with open(topology_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty GRO file: {topology_path}")

    parts = lines[-1].split()
    if len(parts) < 3:
        raise ValueError(
            f"Could not parse box from final GRO line in {topology_path}: '{lines[-1]}'"
        )
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _parse_gro_box_matrix_nm(box_line: str) -> np.ndarray:
    """Parse a GRO box line into a 3x3 box matrix in nm.

    Supports both orthorhombic (3 floats) and triclinic (9 floats) formats.
    """
    vals = [float(v) for v in box_line.split()]
    if len(vals) == 3:
        return np.diag(np.asarray(vals, dtype=np.float32))
    if len(vals) == 9:
        # GROMACS triclinic order:
        # v1x v2y v3z v1y v1z v2x v2z v3x v3y
        v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = vals
        return np.array(
            [
                [v1x, v1y, v1z],
                [v2x, v2y, v2z],
                [v3x, v3y, v3z],
            ],
            dtype=np.float32,
        )
    raise ValueError(
        "Unsupported GRO box format. Expected 3 or 9 floats, "
        f"got {len(vals)} in line: '{box_line.strip()}'."
    )


def _load_single_gro_snapshot_dataset(topology_path: str) -> dict:
    """Load a single-frame GRO file into the in-project dataset dict format."""
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Missing GRO file: {topology_path}")

    with open(topology_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file (too few lines): {topology_path}")

    try:
        n_atoms = int(lines[1].strip())
    except ValueError as e:
        raise ValueError(
            f"Could not parse atom count from second GRO line in {topology_path}: "
            f"'{lines[1]}'"
        ) from e

    expected_lines = n_atoms + 3
    if len(lines) < expected_lines:
        raise ValueError(
            f"Invalid GRO file: expected at least {expected_lines} lines for {n_atoms} atoms, "
            f"found {len(lines)} in {topology_path}."
        )

    atom_lines = lines[2 : 2 + n_atoms]
    box_line = lines[2 + n_atoms]
    box = _parse_gro_box_matrix_nm(box_line)

    coords = np.zeros((1, n_atoms, 3), dtype=np.float32)
    forces = np.zeros((1, n_atoms, 3), dtype=np.float32)
    atom_names: list[str] = []

    for i, ln in enumerate(atom_lines):
        # GRO fixed-width columns: atom name [10:15], coordinates [20:28],[28:36],[36:44].
        atom_name = ln[10:15].strip()
        if not atom_name:
            parts = ln.split()
            if len(parts) < 6:
                raise ValueError(f"Could not parse atom line {i + 3} in {topology_path}: '{ln}'")
            atom_name = parts[1]
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        else:
            x = float(ln[20:28])
            y = float(ln[28:36])
            z = float(ln[36:44])

        atom_names.append(atom_name)
        coords[0, i] = np.array([x, y, z], dtype=np.float32)

    species = np.asarray(
        [_guess_atomic_number_from_name(name) for name in atom_names], dtype=np.int32
    )

    return {
        "R": coords,
        "F": forces,
        "box": box[None, :, :],
        "species": species[None, :],
        "mask": np.ones((1, n_atoms), dtype=bool),
    }


class BaseDataset:
    """Base class for molecular dynamics datasets with coarse-graining support."""

    def __init__(
        self,
        dataset_name,
        map_class,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        map_kwargs=None,
        cache_cg: bool = True,
    ):
        """
        Initialize dataset with train/val/test splits.

        Args:
            dataset_name: Key for Dataset_paths dictionary
            map_class: Mapping class (e.g., CappedPeptideMap, Hexane_Map)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            shuffle: Whether to shuffle data during split
            map_kwargs: Keyword arguments for map_class initialization
        """
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)
        map_kwargs = map_kwargs or {}

        ds_cfg = MD_DATASET_PATHS[dataset_name]
        self.ds_cfg = ds_cfg
        self.npz_path = ds_cfg["path"]

        self.dataset_X = None
        self.dataset_U = None
        self.splits = ()

        self.box = None
        if "config" in ds_cfg and os.path.exists(ds_cfg["config"]):
            try:
                snap = _load_single_gro_snapshot_dataset(ds_cfg["config"])
                self.box = snap["box"][0]
            except Exception:
                self.box = None

        # Load topology via MDAnalysis if .gro + .top are registered
        self.mda_universe = None
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        if "config" in ds_cfg and "topology" in ds_cfg:
            print(f"Loading topology for {dataset_name} from: {ds_cfg['topology']}")
            selection = ds_cfg.get("selection", "all")
            self.mda_universe, self.bonds, self.angles, self.dihedrals = (
                _load_bonded_from_mda(ds_cfg["topology"], ds_cfg["config"], selection=selection)
            )

        self.species = None
        self.n_species = 0

        # Initialize mapping
        self.map_obj = map_class(**map_kwargs)
        self.masses = jnp.array(self.map_obj.at_masses)

        # Trajectory/forces are loaded lazily via load_traj().
        self.displacement_fn_U = None
        self.shift_fn_U = None
        self.displacement_fn_X = None
        self.shift_fn_X = None

    def load_traj(self) -> None:
        """Load atomistic trajectory/forces and build train/val/test splits."""
        if self.dataset_X is not None and self.dataset_U is not None:
            return

        print(f"Loading {self.dataset_name} trajectory from: {self.npz_path}")
        dataset = dict(np.load(self.npz_path, allow_pickle=True))

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=self.shuffle,
            shuffle_seed=SEED,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        dataset_ = {
            "training": train_data,
            "validation": val_data,
        }
        if self.train_ratio + self.val_ratio < 1.0 and test_data is not None:
            dataset_["testing"] = test_data

        for split in dataset_.keys():
            dataset_[split]["R"] = dataset_[split]["R"]
            dataset_[split]["F"] = dataset_[split]["F"]
            dataset_[split]["box"] = dataset_[split]["box"]
            dataset_[split]["species"] = dataset_[split]["species"]
            dataset_[split]["mask"] = dataset_[split]["mask"]

        self.dataset_X = copy.deepcopy(dataset_)

        dataset_frac = {}
        self.splits = dataset_.keys()
        for split in self.splits:
            dataset_frac[split] = io.scale_dataset_box_aware(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))
        self._setup_displacement_functions()

    def _cg_cache_path(self, map_name: str) -> str:
        if "traj" in self.ds_cfg:
            base_dir = os.path.dirname(self.ds_cfg["traj"])
        elif "config" in self.ds_cfg:
            base_dir = os.path.dirname(self.ds_cfg["config"])
        else:
            base_dir = os.path.dirname(self.npz_path)
        safe_map = str(map_name).replace("/", "_")
        return os.path.join(base_dir, f"{self.dataset_name}_cg_{safe_map}.npz")

    @staticmethod
    def _pack_split_dict_for_cache(split_dict: dict) -> dict:
        payload = {}
        for split, split_data in split_dict.items():
            for key, value in split_data.items():
                payload[f"{split}__{key}"] = np.asarray(value)
        return payload

    @staticmethod
    def _unpack_split_dict_from_cache(payload: dict) -> dict:
        out: dict[str, dict] = {}
        for full_key, value in payload.items():
            if "__" not in full_key:
                continue
            split, key = full_key.split("__", 1)
            out.setdefault(split, {})[key] = value
        return out

    def _setup_displacement_functions(self):
        """Set up displacement and shift functions for both coordinate systems."""
        displacement_fn_U, shift_fn_U = space.periodic_general(
            box=self.box, fractional_coordinates=True
        )
        self.displacement_fn_U = displacement_fn_U
        self.shift_fn_U = shift_fn_U

        displacement_fn_X, shift_fn_X = space.periodic_general(
            box=self.box, fractional_coordinates=False
        )
        self.displacement_fn_X = displacement_fn_X
        self.shift_fn_X = shift_fn_X

    def coarse_grain(self, map, cached_dataset_path: str | None = None):
        """Coarse grain the dataset using the specified mapping strategy.

        Args:
            map: Mapping strategy name (e.g. ``"coreBetaMap2"``).
        """
        map_indices, cg_species, cg_masses, weights = self.map_obj.get_map(map)
        cache_path = cached_dataset_path or self._cg_cache_path(map)

        if self.cache_cg and os.path.exists(cache_path):
            print(f"Loading cached coarse-grained dataset from {cache_path}")
            payload = dict(np.load(cache_path, allow_pickle=True))
            cg_dataset = self._unpack_split_dict_from_cache(payload)
            if cg_dataset and "training" in cg_dataset and "validation" in cg_dataset:
                self.cg_dataset_X = copy.deepcopy(cg_dataset)
                self.splits = self.cg_dataset_X.keys()

                cg_dataset_frac = {}
                for split in self.splits:
                    cg_dataset_frac[split] = io.scale_dataset_box_aware(
                        self.cg_dataset_X[split], scale_R=1, scale_U=1, fractional=True
                    )
                self.cg_dataset_U = cg_dataset_frac

                self.cg_species = np.asarray(cg_species)
                self.cg_masses = cg_masses
                self.cg_weights = weights.astype(jnp.float32)
                self.n_cg_sites = len(cg_species)
                self.n_cg_species = len(set(np.asarray(cg_species).tolist()))
                self.cg_map_name = map

                if "training" in self.cg_dataset_X and "box" in self.cg_dataset_X["training"]:
                    self.box = self.cg_dataset_X["training"]["box"][0]
                    self._setup_displacement_functions()
                return

        if self.dataset_X is None or self.dataset_U is None:
            self.load_traj()

        # CG topology: bonds, angles, dihedrals
        if hasattr(self.map_obj, "get_cg_topology"):
            self.cg_bond_index, self.cg_angle_index, self.cg_dihedral_index = (
                self.map_obj.get_cg_topology(map)
            )
        else:
            self.cg_bond_index = None
            self.cg_angle_index = None
            self.cg_dihedral_index = None

        # cg_bonds: (2, B) direct bond index (alias for cg_bond_index)
        self.cg_bonds = self.cg_bond_index

        # Legacy bond-type dict (bond_0 … bond_3) for backward compat
        if hasattr(self.map_obj, "get_bond_types"):
            bt = self.map_obj.get_bond_types(map)
            self.cg_bond_types = {
                k: jnp.array(v, dtype=jnp.int32) if len(v) > 0
                else jnp.empty((0, 2), dtype=jnp.int32)
                for k, v in bt.items()
            }
            # Nonbonded 1-4 / 1-5 pairs
            nb_parts = [
                np.asarray(bt[k], dtype=np.int32)
                for k in ("bond_2", "bond_3")
                if k in bt and len(bt[k]) > 0
            ]
            self.cg_nonbonded_index = (
                np.concatenate(nb_parts, axis=0).T if nb_parts else None
            )
        else:
            self.cg_bond_types = None
            self.cg_nonbonded_index = None

        n_cg_sites = len(cg_species)
        n_cg_species = len(set(cg_species))
        weights = weights.astype(jnp.float32)  # (M,N)

        cg_dataset = {}
        for split in self.splits:
            cg_coords, cg_forces = map_dataset(
                self.dataset_X[split]["R"],
                self.displacement_fn_X,
                self.shift_fn_X,
                weights,
                weights,
                self.dataset_X[split]["F"],
            )
            n_frames, n_cg_sites, _ = cg_coords.shape

            cg_dataset[split] = {
                "R": cg_coords.astype(jnp.float32),
                "F": cg_forces.astype(jnp.float32),
                "species": jnp.tile(jnp.array(cg_species), (n_frames, 1)),
                "box": jnp.tile(self.box, (n_frames, 1, 1)),
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        # Create fractional coordinate versions
        cg_dataset_frac = {}
        for split in self.splits:
            out = io.scale_dataset_box_aware(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights
        self.cg_map_name = map  # remember which strategy was applied

        if self.cache_cg:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            payload = self._pack_split_dict_for_cache(self.cg_dataset_X)
            payload["cg_map"] = np.asarray(map)
            np.savez_compressed(cache_path, **payload)
            print(f"Saved coarse-grained cache to {cache_path}")

    def save_xyz(
        self,
        output_path: str,
        split: str = "training",
        cg: bool = False,
        n_frames: int | None = None,
        workers: int = 1,
        unwrap: bool = True,
    ) -> None:
        """Save a trajectory split as a multi-frame XYZ file (coordinates in Å).

        Positions stored in nm are multiplied by 10 before writing.

        Args:
            output_path: Destination ``.xyz`` file path.
            split:       Dataset split: ``"training"``, ``"validation"``, or
                         ``"testing"``.
            cg:          ``True`` → write the CG trajectory (requires
                         :meth:`coarse_grain` to have been called first);
                         ``False`` → write the all-atom trajectory.
            n_frames:    If given, write only the first *n_frames* frames.
            workers:     Worker processes for XYZ formatting (default 1 to
                         avoid ``os.fork()`` deadlocks with JAX).
            unwrap:      If ``True`` (default), apply PBC make-whole unwrapping
                         via :attr:`displacement_fn_X` before writing.
        """
        from .topology import get_atom_info, write_xyz_trajectory, unwrap_trajectory

        if cg:
            if not hasattr(self, "cg_dataset_X"):
                raise RuntimeError("Call coarse_grain() before save_xyz(cg=True).")
            data = self.cg_dataset_X[split]
            map_name = getattr(self, "cg_map_name", None)
            atom_info = get_atom_info(self.map_obj, map_name=map_name, cg=True)
            bond_index = getattr(self, "cg_bond_index", None)
        else:
            data = self.dataset_X[split]
            atom_info = get_atom_info(self.map_obj, cg=False)
            bond_index = getattr(self, "bonds", None)

        R_nm = np.asarray(data["R"])
        if n_frames is not None:
            R_nm = R_nm[:n_frames]
        if unwrap:
            R_nm = unwrap_trajectory(R_nm, self.displacement_fn_X, bond_index)

        write_xyz_trajectory(output_path, R_nm, atom_info, workers=workers)

    def save_pdb(
        self,
        output_path: str,
        map_name: str | None = None,
        split: str = "training",
        cg: bool = False,
        n_frames: int | None = None,
        unwrap: bool = True,
    ) -> None:
        """Save a trajectory split as a multi-model PDB file (coordinates in Å).

        Positions stored in nm are multiplied by 10 before writing.
        Bond connectivity is written as CONECT records:

        - All-atom PDB: uses the MDAnalysis-derived bond list (``self.bonds``).
        - CG PDB: uses the CG bond index set by :meth:`coarse_grain`.

        For :class:`~cgbench.core.mapping.CappedPeptideMap` datasets every
        atom/bead is labelled with its GROMACS atom name (``CA``, ``CB``, …)
        and three-letter residue name (``ALA``, ``ACE``, …).

        Args:
            output_path: Destination ``.pdb`` file path.
            map_name:    CG strategy name (e.g. ``"coreBetaMap2"``).  Required
                         when *cg* is ``True`` and :attr:`cg_map_name` is not
                         set.
            split:       Dataset split: ``"training"``, ``"validation"``, or
                         ``"testing"``.
            cg:          ``True`` → write the CG trajectory; ``False`` →
                         all-atom.
            n_frames:    If given, write only the first *n_frames* frames.
            unwrap:      If ``True`` (default), apply PBC make-whole unwrapping
                         via :attr:`displacement_fn_X` before writing.
        """
        from .topology import get_atom_info, write_pdb_trajectory_with_bonds, unwrap_trajectory

        if cg:
            if not hasattr(self, "cg_dataset_X"):
                raise RuntimeError("Call coarse_grain() before save_pdb(cg=True).")
            _map = map_name or getattr(self, "cg_map_name", None)
            if _map is None:
                raise ValueError(
                    "map_name is required for CG PDB saving. "
                    "Pass it explicitly or call coarse_grain() first."
                )
            data = self.cg_dataset_X[split]
            bond_index = getattr(self, "cg_bond_index", None)
            atom_info = get_atom_info(self.map_obj, map_name=_map, cg=True)
        else:
            data = self.dataset_X[split]
            bond_index = getattr(self, "bonds", None)
            atom_info = get_atom_info(self.map_obj, cg=False)

        R_nm = np.asarray(data["R"])
        if n_frames is not None:
            R_nm = R_nm[:n_frames]
        if unwrap:
            R_nm = unwrap_trajectory(R_nm, self.displacement_fn_X, bond_index)

        write_pdb_trajectory_with_bonds(
            output_path,
            R_nm,
            atom_info,
            bond_index=bond_index,
            box_nm=np.asarray(self.box),
        )


class Hexane_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1, cache_cg: bool = True):
        super().__init__(
            dataset_name="hexane",
            map_class=Hexane_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=True,
            map_kwargs={"nmol": 100},
            cache_cg=cache_cg,
        )
        self.hexane_map = self.map_obj


class TIP3P_water_Dataset(BaseDataset):
    """TIP3P liquid water dataset.

    Each water molecule (O, H, H) maps to a single CG bead.

    Maps:
    - ``"UnitedAtom"``: all three atoms contribute (mass-weighted COM).
    - ``"HeavyAtom"``:  only the oxygen atom contributes.
    """

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        cfg_path = MD_DATASET_PATHS["tip3p-water"]["config"]
        snap = _load_single_gro_snapshot_dataset(cfg_path)
        n_atoms = int(snap["R"].shape[1])
        if n_atoms % 3 != 0:
            raise ValueError(
                f"TIP3P water expects atom count divisible by 3, got {n_atoms}."
            )
        n_mols = n_atoms // 3
        super().__init__(
            dataset_name="tip3p-water",
            map_class=TIP3P_Water_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"n_mols": n_mols},
            cache_cg=cache_cg,
        )
        self.water_map = self.map_obj


class BenzeneCrystal_Dataset(BaseDataset):
    """Benzene crystal dataset wrapper."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        cfg_path = MD_DATASET_PATHS["benzene_crystal"]["config"]
        snap = _load_single_gro_snapshot_dataset(cfg_path)
        n_atoms = int(snap["R"].shape[1])
        if n_atoms % 12 != 0:
            raise ValueError(
                f"Benzene crystal expects atom count divisible by 12, got {n_atoms}."
            )
        nmol = n_atoms // 12

        super().__init__(
            dataset_name="benzene_crystal",
            map_class=BenzeneCrystal_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"nmol": nmol},
            cache_cg=cache_cg,
        )
        self.benzene_map = self.map_obj


class BenzeneCrystal288_Dataset(BaseDataset):
    """Benzene crystal 288-molecule trajectory dataset wrapper."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cg_map: str | None = None,
        prefer_cg_cache: bool = False,
        cache_cg: bool = True,
    ):
        self._cached_cg_loaded = False
        self._cached_cg_map = None

        if self.cache_cg and prefer_cg_cache and cg_map is not None:
            cache_path = self._cache_path_for_map(cg_map)
            if os.path.exists(cache_path):
                print(f"Initializing benzene_crystal_288 from cached CG dataset: {cache_path}")
                payload = dict(np.load(cache_path, allow_pickle=True))
                cg_dataset = self._unpack_split_dict_from_cache(payload)

                if cg_dataset and "training" in cg_dataset and "validation" in cg_dataset:
                    self.cg_dataset_X = copy.deepcopy(cg_dataset)
                    self.splits = self.cg_dataset_X.keys()

                    cg_dataset_frac = {}
                    for split in self.splits:
                        cg_dataset_frac[split] = io.scale_dataset_box_aware(
                            self.cg_dataset_X[split], scale_R=1, scale_U=1, fractional=True
                        )
                    self.cg_dataset_U = cg_dataset_frac

                    n_cg_sites = int(self.cg_dataset_X["training"]["R"].shape[1])
                    if n_cg_sites % 3 != 0:
                        raise ValueError(
                            "Invalid cached benzene_crystal_288 CG dataset: "
                            f"training sites ({n_cg_sites}) not divisible by 3."
                        )

                    self.nmol = n_cg_sites // 3
                    self.map_obj = BenzeneCrystal_Map(nmol=self.nmol)
                    self.benzene_map = self.map_obj
                    self.masses = jnp.array(self.map_obj.at_masses)

                    self.box = self.cg_dataset_X["training"]["box"][0]
                    self.species = self.cg_dataset_X["training"]["species"][0]
                    self.n_species = len(set(np.asarray(self.species).tolist()))
                    self._setup_displacement_functions()

                    self._set_cached_topology_metadata(cg_map)
                    self._cached_cg_loaded = True
                    self._cached_cg_map = cg_map
                    return

        cfg_path = MD_DATASET_PATHS["benzene_crystal_288"]["config"]
        snap = _load_single_gro_snapshot_dataset(cfg_path)
        n_atoms = int(snap["R"].shape[1])
        if n_atoms % 12 != 0:
            raise ValueError(
                f"Benzene crystal expects atom count divisible by 12, got {n_atoms}."
            )
        nmol = n_atoms // 12
        self.nmol = nmol

        super().__init__(
            dataset_name="benzene_crystal_288",
            map_class=BenzeneCrystal_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"nmol": nmol},
            cache_cg=cache_cg,
        )
        self.benzene_map = self.map_obj

    @staticmethod
    def _cache_path_for_map(cg_map: str) -> str:
        base_npz = MD_DATASET_PATHS["benzene_crystal_288"]["path"]
        data_dir = os.path.dirname(base_npz)
        return os.path.join(data_dir, f"benzene_crystal_288_{cg_map}.npz")

    @staticmethod
    def _pack_split_dict_for_cache(split_dict: dict) -> dict:
        payload = {}
        for split, split_data in split_dict.items():
            for key, value in split_data.items():
                payload[f"{split}__{key}"] = np.asarray(value)
        return payload

    @staticmethod
    def _unpack_split_dict_from_cache(payload: dict) -> dict:
        out: dict[str, dict] = {}
        for full_key, value in payload.items():
            if "__" not in full_key:
                continue
            split, key = full_key.split("__", 1)
            out.setdefault(split, {})[key] = value
        return out

    def _set_cached_topology_metadata(self, map_name: str) -> None:
        _, cg_species, cg_masses, weights = self.map_obj.get_map(map_name)

        if hasattr(self.map_obj, "get_cg_topology"):
            self.cg_bond_index, self.cg_angle_index, self.cg_dihedral_index = (
                self.map_obj.get_cg_topology(map_name)
            )
            self.cg_bonds = self.cg_bond_index
        else:
            self.cg_bond_index = None
            self.cg_angle_index = None
            self.cg_dihedral_index = None
            self.cg_bonds = None

        if hasattr(self.map_obj, "get_bond_types"):
            bt = self.map_obj.get_bond_types(map_name)
            self.cg_bond_types = {
                k: jnp.array(v, dtype=jnp.int32) if len(v) > 0 else jnp.empty((0, 2), dtype=jnp.int32)
                for k, v in bt.items()
            }
            nb_parts = [
                np.asarray(bt[k], dtype=np.int32)
                for k in ("bond_2", "bond_3")
                if k in bt and len(bt[k]) > 0
            ]
            self.cg_nonbonded_index = (
                np.concatenate(nb_parts, axis=0).T if nb_parts else None
            )
        else:
            self.cg_bond_types = None
            self.cg_nonbonded_index = None

        self.cg_species = np.asarray(cg_species)
        self.cg_masses = cg_masses
        self.cg_weights = weights
        self.n_cg_sites = len(cg_species)
        self.n_cg_species = len(set(np.asarray(cg_species).tolist()))

    def coarse_grain(self, map, cached_dataset_path: str | None = None):
        if self._cached_cg_loaded and self._cached_cg_map == map:
            print(f"Using already-loaded cached coarse-grained dataset for map '{map}'.")
            return

        cache_path = cached_dataset_path or self._cache_path_for_map(map)

        if self.cache_cg and os.path.exists(cache_path):
            print(f"Loading cached coarse-grained dataset from {cache_path}")
            payload = dict(np.load(cache_path, allow_pickle=True))
            cg_dataset = self._unpack_split_dict_from_cache(payload)

            if not cg_dataset or "training" not in cg_dataset or "validation" not in cg_dataset:
                raise ValueError(
                    f"Invalid benzene_crystal_288 CG cache format in {cache_path}."
                )

            self.cg_dataset_X = copy.deepcopy(cg_dataset)

            cg_dataset_frac = {}
            for split in self.cg_dataset_X.keys():
                cg_dataset_frac[split] = io.scale_dataset_box_aware(
                    self.cg_dataset_X[split], scale_R=1, scale_U=1, fractional=True
                )
            self.cg_dataset_U = cg_dataset_frac

            self._set_cached_topology_metadata(map)
            self._cached_cg_loaded = True
            self._cached_cg_map = map
            return

        super().coarse_grain(map)

        if self.cache_cg:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            payload = self._pack_split_dict_for_cache(self.cg_dataset_X)
            payload["cg_map"] = np.asarray(map)
            np.savez_compressed(cache_path, **payload)
            print(f"Saved coarse-grained cache to {cache_path}")
        self._cached_cg_loaded = False
        self._cached_cg_map = map


class Capped_Ala_Dataset(BaseDataset):
    """ACE-ALA-NME capped alanine dipeptide dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_ala",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "ALA", "NME"]},
            cache_cg=cache_cg,
        )


class Capped_Ala2_Dataset(BaseDataset):
    """ACE-ALA-ALA-NME capped alanine tripeptide dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_ala2",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "ALA", "ALA", "NME"]},
            cache_cg=cache_cg,
        )


class Capped_Ala15_Dataset(BaseDataset):
    """ACE + 15×ALA + NME capped poly-alanine dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_ala15",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE"] + ["ALA"] * 15 + ["NME"]},
            cache_cg=cache_cg,
        )


class Capped_Pro_Dataset(BaseDataset):
    """ACE-PRO-NME capped proline dipeptide dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_pro",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "PRO", "NME"]},
            cache_cg=cache_cg,
        )


class Capped_Thr_Dataset(BaseDataset):
    """ACE-THR-NME capped threonine dipeptide dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_thr",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "THR", "NME"]},
            cache_cg=cache_cg,
        )


class Capped_Gly_Dataset(BaseDataset):
    """ACE-GLY-NME capped glycine dipeptide dataset."""

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        super().__init__(
            dataset_name="capped_gly",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "GLY", "NME"]},
            cache_cg=cache_cg,
        )


class _ProSolUncappedProteinDataset(BaseDataset):
    """Base class for uncapped ProSol proteins using CATH-style residue mapping."""

    def __init__(
        self,
        dataset_name: str,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        self.dataset_name = dataset_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)
        ds_cfg = STATIC_FRAME_DATASET_PATHS[self.dataset_name]
        self.config_path = ds_cfg["config"]
        self.top_path = topology_path or ds_cfg["topology"]
        self.traj_path = trajectory_path or ds_cfg["traj"]
        self.selection = selection
        self.ds_cfg = {"config": self.config_path, "topology": self.top_path, "traj": self.traj_path}
        self.npz_path = self.traj_path

        # Topology-only initialisation
        self.dataset_X = None
        self.dataset_U = None
        self.splits = ()
        self.species = None
        self.box = None
        self.n_species = 0
        self.displacement_fn_U = None
        self.shift_fn_U = None
        self.displacement_fn_X = None
        self.shift_fn_X = None

        self.mda_universe, self.bonds, self.angles, self.dihedrals = _load_bonded_from_mda(
            self.top_path,
            self.config_path,
            selection=self.selection,
        )

        # Build mapping metadata from the topology frame.
        ag = self.mda_universe
        residue_ids = np.asarray(ag.resids, dtype=np.int32)
        residue_names = np.asarray(ag.resnames, dtype=object)
        atom_names = np.asarray(ag.names, dtype=object)
        self.map_obj = UncappedProteinMap(
            residue_ids=residue_ids,
            residue_names=residue_names,
            atom_names=atom_names,
        )
        self.masses = jnp.array(self.map_obj.at_masses)

    def load_traj(self) -> None:
        if self.dataset_X is not None and self.dataset_U is not None:
            return

        print(f"Loading {self.dataset_name} dataset from: {self.top_path} + {self.traj_path}")
        dataset_full = _load_gromacs_to_dataset(self.top_path, self.traj_path, selection=self.selection)

        split_src = {
            "R": dataset_full["R"],
            "F": dataset_full["F"],
            "box": dataset_full["box"],
            "species": dataset_full["species"],
            "mask": dataset_full["mask"],
        }
        train_data, val_data, test_data = preprocessing.train_val_test_split(
            split_src,
            shuffle=self.shuffle,
            shuffle_seed=SEED,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        dataset_ = {"training": train_data, "validation": val_data}
        if self.train_ratio + self.val_ratio < 1.0 and test_data is not None:
            dataset_["testing"] = test_data

        self.dataset_X = copy.deepcopy(dataset_)
        self.splits = dataset_.keys()
        self.dataset_U = {
            split: io.scale_dataset_box_aware(dataset_[split], scale_R=1, scale_U=1, fractional=True)
            for split in self.splits
        }

        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))
        self._setup_displacement_functions()


class UBQ1_Dataset(_ProSolUncappedProteinDataset):
    """Uncapped 1UBQ dataset with CATH-style per-residue mapping."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        super().__init__(
            dataset_name="1UBQ",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            selection=selection,
            cache_cg=cache_cg,
        )
        self.ubq1_map = self.map_obj


class IFC1_Dataset(_ProSolUncappedProteinDataset):
    """Uncapped 1IFC dataset with CATH-style per-residue mapping."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        super().__init__(
            dataset_name="1IFC",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            selection=selection,
            cache_cg=cache_cg,
        )
        self.ifc1_map = self.map_obj


class MJC1_Dataset(_ProSolUncappedProteinDataset):
    """Uncapped 1MJC dataset with CATH-style per-residue mapping."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        super().__init__(
            dataset_name="1MJC",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            selection=selection,
            cache_cg=cache_cg,
        )
        self.mjc1_map = self.map_obj


class QX5_1_Dataset(_ProSolUncappedProteinDataset):
    """Uncapped 1QX5 dataset with CATH-style per-residue mapping."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        super().__init__(
            dataset_name="1QX5",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            selection=selection,
            cache_cg=cache_cg,
        )
        self.qx5_1_map = self.map_obj


class LYT6_Dataset(_ProSolUncappedProteinDataset):
    """Uncapped 6LYT dataset with CATH-style per-residue mapping."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        topology_path: str | None = None,
        trajectory_path: str | None = None,
        selection: str = "protein",
        cache_cg: bool = True,
    ):
        super().__init__(
            dataset_name="6LYT",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            topology_path=topology_path,
            trajectory_path=trajectory_path,
            selection=selection,
            cache_cg=cache_cg,
        )
        self.lyt6_map = self.map_obj


class SPICE_Dipeptides(BaseDataset):
    """Dataset wrapper for pre-mapped SPICE dipeptides (coreBetaMap2 only).

    The SPICE dipeptides file is already coarse-grained, therefore this class
    loads the NPZ directly and exposes a ``coarse_grain`` method that validates
    the requested mapping strategy and forwards the already-loaded data.
    """

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        self.dataset_name = "spice_dipeptides"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)

        print(
            f"Loading {self.dataset_name} dataset from:",
            MD_DATASET_PATHS[self.dataset_name]["path"],
        )
        dataset = dict(np.load(MD_DATASET_PATHS[self.dataset_name]["path"], allow_pickle=True))


        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=shuffle,
            shuffle_seed=SEED,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        dataset_ = {
            "training": train_data,
            "validation": val_data,
        }
        if test_data not in (None, {}):
            dataset_["testing"] = test_data

        for split in dataset_.keys():
            dataset_[split]["R"] = dataset_[split]["R"]
            dataset_[split]["F"] = dataset_[split]["F"]
            dataset_[split]["species"] = dataset_[split]["species"]
            dataset_[split]["mask"] = dataset_[split]["mask"]

        self.dataset_X = copy.deepcopy(dataset_)
        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.species = dataset_["training"]["species"][0]
        self.box = None
        self.n_species = len(np.unique(self.species))
        self.masses = None
        self.map_obj = None
        self.displacement_fn_X, _ = space.free()


    def coarse_grain(self, map):
        if map != "coreBetaMap2":
            raise ValueError(
                "SPICE_Dipeptides is only available for map='coreBetaMap2'. "
                f"Received map='{map}'."
            )

        # Dataset is already coarse-grained in coreBetaMap2 mapping.
        self.cg_dataset_X = copy.deepcopy(self.dataset_X)
        self.cg_species = self.species
        self.n_cg_sites = self.cg_dataset_X["training"]["R"].shape[1]
        self.n_cg_species = len(np.unique(self.cg_species))
        self.cg_masses = None
        self.cg_weights = None
        self.cg_bonds = None
        self.cg_bond_types = None


class PepsolDimers:
    """Pre-mapped PepSol dipeptide dimer dataset (coreBetaMap2 only).

    The NPZ produced by ``preprocess_pepsol_dimers.py`` is already coarse-
    grained.  This class loads it, splits into train/val/test, and exposes a
    ``coarse_grain`` method that validates the requested strategy and forwards
    the already-loaded data.  Metadata fields ``id``, ``r0``, and ``window``
    are preserved in every split dict alongside ``R``, ``F``, ``species``,
    and ``mask``.

    Uses ``space.periodic_general`` with the reference 6×6×6 nm box.
    Trajectory coordinates are unwrapped (positions can lie outside the box),
    so the dataset wraps them into ``[0, L)`` on load using
    ``jax_md.space``'s shift function (``shift_fn(R, 0)``).
    """

    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True, cache_cg: bool = True):
        self.dataset_name = "pepsol_dimers"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)

        print(
            f"Loading {self.dataset_name} dataset from:",
            MD_DATASET_PATHS[self.dataset_name]["path"],
        )
        dataset = dict(np.load(MD_DATASET_PATHS[self.dataset_name]["path"], allow_pickle=True))

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=shuffle,
            shuffle_seed=SEED,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        dataset_ = {
            "training": train_data,
            "validation": val_data,
        }
        if test_data not in (None, {}):
            dataset_["testing"] = test_data

        # Box is fixed 6×6×6 nm for all PepSol dimer systems.
        self.box = dataset_["training"]["box"][0]

        # Build displacement / shift functions before wrapping so we can use
        # shift_fn to fold positions back into [0, L) (the JAX-MD canonical way).
        displacement_fn_X, shift_fn_X = space.periodic_general(
            self.box, fractional_coordinates=False
        )
        self.displacement_fn_X = displacement_fn_X
        self.shift_fn_X = shift_fn_X
        displacement_fn_U, shift_fn_U = space.periodic_general(
            self.box, fractional_coordinates=True
        )
        self.displacement_fn_U = displacement_fn_U
        self.shift_fn_U = shift_fn_U

        # Wrap unwrapped coordinates into the box: shift_fn(R, 0) folds any
        # position back into [0, L) for each periodic dimension.
        # Then move padded beads (mask=False) to box centre so they are always
        # far (>2.5 nm) from any real bead, preventing NaN in MACE's radial
        # basis functions caused by near-zero distances to the padded origin.
        box_center = np.diag(self.box) / 2.0  # [3.0, 3.0, 3.0] nm
        for split in dataset_.keys():
            R = dataset_[split]["R"].astype(np.float32)
            R = np.asarray(shift_fn_X(jnp.asarray(R), jnp.zeros_like(R)))
            mask = dataset_[split]["mask"]  # (N, max_cg) bool
            dataset_[split]["R"] = np.where(mask[:, :, None], R, box_center[None, None, :])
            dataset_[split]["F"] = dataset_[split]["F"]
            dataset_[split]["species"] = dataset_[split]["species"]
            dataset_[split]["mask"] = dataset_[split]["mask"]

        self.dataset_X = copy.deepcopy(dataset_)
        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.species = dataset_["training"]["species"][0]
        self.n_species = len(np.unique(self.species[dataset_["training"]["mask"][0]]))
        self.masses = None
        self.map_obj = None

        # Fractional coordinate version
        self.splits = dataset_.keys()
        dataset_frac = {}
        for split in self.splits:
            dataset_frac[split] = io.scale_dataset_box_aware(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
        self.dataset_U = dataset_frac

    def coarse_grain(self, map):
        if map != "coreBetaMap2":
            raise ValueError(
                "PepsolDimers is only available for map='coreBetaMap2'. "
                f"Received map='{map}'."
            )

        self.cg_dataset_X = copy.deepcopy(self.dataset_X)
        self.cg_species = self.species
        self.n_cg_sites = self.cg_dataset_X["training"]["R"].shape[1]
        self.n_cg_species = len(np.unique(self.cg_species[self.cg_dataset_X["training"]["mask"][0]]))
        self.cg_masses = None
        self.cg_weights = None
        self.cg_bonds = None
        self.cg_bond_types = None

        cg_dataset_frac = {}
        for split in self.splits:
            cg_dataset_frac[split] = io.scale_dataset_box_aware(
                self.cg_dataset_X[split], scale_R=1, scale_U=1, fractional=True
            )
        self.cg_dataset_U = cg_dataset_frac


class CATH_Dataset:
    """Dataset for ALL CATH protein domains with coarse-graining support.

    Loads the entire combined padded npz (all 41 domains) at once.  The
    ``subset`` field (shape ``(n_frames,)``) is an integer index into
    ``domain_residue_index.json``'s ``subsets`` list and identifies which
    domain each frame belongs to.

    Call :meth:`coarse_grain_all` to map every domain's frames with the
    appropriate per-domain :class:`CATH_Map` and accumulate the results into a
    single padded dataset whose ``species`` array reflects the CG species
    (replacing the original all-atom species).

    Args:
        cg_strategy: CG mapping strategy (``'CA'``, ``'heavyOnly'``,
            ``'coreBetaMap2'``, ``'conway'``, ``'martini3'``).
        dataset_key: Key into :data:`Dataset_paths` (``'cath'`` or
            ``'cath_test'``).
        domain_index_path: Optional alternative path to
            ``domain_residue_index.json``.
    """

    def __init__(
        self,
        dataset_key: str = "cath_full",
        cg_strategy: str = "coreBetaMap2",
        domain_index_path: str | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        cached_dataset_path: str | None = None,
        cache_cg: bool = True,
    ):
        import json
        from .mapping import _DEFAULT_DOMAIN_INDEX

        self.cg_strategy = cg_strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)
        self.cached_dataset_path = cached_dataset_path

        if self.cached_dataset_path is None and self.cache_cg:
            raw_path = MD_DATASET_PATHS[dataset_key]["path"]
            raw_dir = os.path.dirname(raw_path)
            self.cached_dataset_path = os.path.join(
                raw_dir,
                f"{dataset_key}_{self.cg_strategy}_cg.npz",
            )
        
        # ------------------------------------------------------------------
        # load domain residue index
        # ------------------------------------------------------------------
        index_path = domain_index_path or _DEFAULT_DOMAIN_INDEX
        with open(index_path) as f:
            domain_index = json.load(f)

        self.subsets_list = domain_index["subsets"]   # ordered list of 41 domain names
        self.domain_info = domain_index["domains"]    # {domain_name: {n_atoms, ...}}
        self._index_path = index_path

        # ------------------------------------------------------------------
        # load the full multi-domain npz (all frames, all domains)
        # ------------------------------------------------------------------
        import os
        if self.cached_dataset_path is not None and os.path.exists(self.cached_dataset_path):
            print(f"Skipping loading of full CATH dataset, will use cached CG dataset: {self.cached_dataset_path}")
            self.raw = None
        else:
            npz_path = MD_DATASET_PATHS[dataset_key]["path"]
            print(f"Loading full CATH dataset ({len(self.subsets_list)} domains) from: {npz_path}")
            raw = np.load(npz_path, allow_pickle=True)
            self.raw = dict(raw)

            print(f"  Total frames: {self.raw['R'].shape[0]}, "
                  f"padded atoms: {self.raw['R'].shape[1]}")
            print(f"  Domains in subset field: "
                  f"{len(set(self.raw['subset'].flatten().tolist()))}")

    # ------------------------------------------------------------------
    def coarse_grain_all(self) -> dict:
        """Coarse-grain all domains and return a flat padded dataset dict.

        For each domain the method:
          1. Selects frames whose ``subset`` value matches the domain index.
          2. Slices atoms to the domain's real atom count (removes padding).
          3. Applies :class:`CATH_Map` to obtain CG positions and forces.
          4. Pads CG arrays to ``max_cg_sites`` across all domains.

        The returned dict has the same keys as the original npz but with
        ``species`` replaced by ``cg_species`` (per-frame, padded).

        Returns:
            dict with keys ``R``, ``F``, ``box``, ``mask``, ``species``
            (CG species, replaces original), ``subset``, ``cv``, ``U``,
            ``n_cg_sites`` – all shape ``(n_frames_total, ...)``.  Also sets
            ``self.per_domain_results`` (list of per-domain dicts) and
            ``self.max_cg_sites``.
        """
        raw = self.raw
        subset_flat = raw["subset"].flatten()   # shape (n_frames,), dtype float or int

        per_domain: list[dict] = []

        for domain_idx, domain_name in enumerate(self.subsets_list):
            if domain_name not in self.domain_info:
                print(f"  [SKIP] domain '{domain_name}' not in domain_residue_index")
                continue

            n_atoms = self.domain_info[domain_name]["n_atoms"]

            # Select frames for this domain
            frame_mask = subset_flat == float(domain_idx)
            n_frames = int(frame_mask.sum())
            if n_frames == 0:
                print(f"  [WARN] No frames found for domain '{domain_name}' "
                      f"(index {domain_idx})")
                continue

            # Slice to real atom count
            R = raw["R"][frame_mask, :n_atoms, :]   # (n_frames, n_atoms, 3)
            F = raw["F"][frame_mask, :n_atoms, :]   # (n_frames, n_atoms, 3)
            box = raw["box"][frame_mask]             # (n_frames, 3, 3)

            # Keep ancillary per-frame arrays
            extra = {}
            for key in ("subset", "cv", "U"):
                if key in raw:
                    extra[key] = raw[key][frame_mask]

            # Build per-domain CATH_Map
            try:
                cath_map = CATH_Map(
                    domain_name=domain_name,
                    cg_strategy=self.cg_strategy,
                    domain_index_path=self._index_path,
                )
            except (ValueError, KeyError) as e:
                print(f"  [SKIP] domain '{domain_name}': CG mapping failed — {e}", flush=True)
                continue
            _, cg_species, cg_masses, weights = cath_map.get_map()
            weights = weights.astype(jnp.float32)
            bond_types = cath_map.get_bond_types()

            # displacement / shift for this domain's box (first frame)
            box0 = jnp.array(box[0])
            displacement_fn_X, shift_fn_X = space.periodic_general(
                box=box0, fractional_coordinates=False
            )

            cg_coords, cg_forces = map_dataset(
                R,
                displacement_fn_X,
                shift_fn_X,
                weights,
                weights,
                F,
            )
            n_cg = cg_coords.shape[1]

            # CG species tiled to (n_frames, n_cg)
            cg_species_2d = np.tile(
                np.array(cg_species, dtype=np.int32), (n_frames, 1)
            )
            # Real CG mask (all True – padding happens below)
            cg_mask = np.ones((n_frames, n_cg), dtype=bool)

            print(f"  [OK] {domain_name}: {n_frames} frames, {n_cg} CG sites")

            per_domain.append({
                "domain_name":  domain_name,
                "n_cg_sites":   n_cg,
                "n_frames":     n_frames,
                "R":            np.array(cg_coords, dtype=np.float32),
                "F":            np.array(cg_forces, dtype=np.float32),
                "box":          box,
                "mask":         cg_mask,
                "species":      cg_species_2d,
                "cg_species":   np.array(cg_species, dtype=np.int32),
                "bond_types":   bond_types,
                **extra,
            })

        if not per_domain:
            raise RuntimeError("No domain frames were processed.")

        # ------------------------------------------------------------------
        # Pad all domains to max_cg_sites and concatenate
        # ------------------------------------------------------------------
        max_cg = max(d["n_cg_sites"] for d in per_domain)
        self.max_cg_sites = max_cg
        print(f"\nMax CG sites across all domains: {max_cg}")

        all_R, all_F, all_box, all_mask, all_species = [], [], [], [], []
        all_n_cg = []
        extra_accum: dict[str, list] = {}

        for d in per_domain:
            n = d["n_frames"]
            pad = max_cg - d["n_cg_sites"]

            all_R.append(np.pad(d["R"], ((0, 0), (0, pad), (0, 0))))
            all_F.append(np.pad(d["F"], ((0, 0), (0, pad), (0, 0))))
            all_box.append(d["box"])
            all_mask.append(
                np.pad(d["mask"], ((0, 0), (0, pad)), constant_values=False)
            )
            sp_padded = np.pad(d["species"], ((0, 0), (0, pad)))
            all_species.append(sp_padded)
            all_n_cg.append(np.full(n, d["n_cg_sites"], dtype=np.int32))

            for key in ("subset", "cv", "U"):
                if key in d:
                    extra_accum.setdefault(key, []).append(d[key])

        combined = {
            "R":          np.concatenate(all_R, axis=0),
            "F":          np.concatenate(all_F, axis=0),
            "box":        np.concatenate(all_box, axis=0),
            "mask":       np.concatenate(all_mask, axis=0),
            "species":    np.concatenate(all_species, axis=0),  # CG species
            "n_cg_sites": np.concatenate(all_n_cg, axis=0),
        }
        for key, arrays in extra_accum.items():
            val = np.concatenate(arrays, axis=0)
            if key == "subset":
                val = val.astype(np.int32)
            combined[key] = val

        self.per_domain_results = per_domain
        # Collect per-domain CG bond types (indexed by domain_name)
        self.cg_bond_types_per_domain = {
            d["domain_name"]: {
                k: jnp.array(v, dtype=jnp.int32) if len(v) > 0
                else jnp.empty((0, 2), dtype=jnp.int32)
                for k, v in d["bond_types"].items()
            }
            for d in per_domain
        }
        self.cg_dataset = combined
        print(f"Total frames mapped: {combined['R'].shape[0]}")
        return combined

    def coarse_grain(self, map=None, cached_dataset_path=None):
        """
        Coarse grain the dataset and split into train/val/test splits like BaseDataset.
        If cached_dataset_path is provided and exists, load the dataset from there instead of
        coarse-graining from scratch.
        """
        import os
        if map:
            self.cg_strategy = map
        
        cached_path = cached_dataset_path or self.cached_dataset_path
        
        if cached_path is not None and os.path.exists(cached_path):
            print(f"Loading cached coarse-grained dataset from {cached_path}")
            combined = dict(np.load(cached_path, allow_pickle=True))
            if not hasattr(self, "max_cg_sites"):
                self.max_cg_sites = combined["R"].shape[1]
        else:
            combined = self.coarse_grain_all()
            if self.cache_cg and cached_path is not None:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                np.savez_compressed(cached_path, **combined)
                print(f"Saved coarse-grained CATH cache to {cached_path}")

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            combined,
            shuffle=self.shuffle,
            shuffle_seed=SEED,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
        )

        cg_dataset = {
            "training": train_data,
            "validation": val_data,
        }
        if test_data is not None and len(test_data["R"]) > 0:
            cg_dataset["testing"] = test_data

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        for split, data in cg_dataset.items():
            cg_dataset_frac[split] = io.scale_dataset_box_aware(
                copy.deepcopy(data), scale_R=1, scale_U=1, fractional=True
            )
        self.cg_dataset_U = cg_dataset_frac

        self.cg_species = combined["species"][0]
        self.cg_masses = None
        self.n_cg_species = len(np.unique(combined["species"]))

        self.n_cg_sites = self.max_cg_sites
        self.dataset_U = self.cg_dataset_U
        self.dataset_X = self.cg_dataset_X
        self.box = combined["box"][0]

        # Build per-domain bond types if not already populated (e.g. cached path)
        if not hasattr(self, "cg_bond_types_per_domain"):
            self._build_bond_types_per_domain()

    def _build_bond_types_per_domain(self) -> None:
        """Instantiate CATH_Map per domain to compute CG bond types.

        Used when loading from a cached coarse-grained dataset so that bond
        types are always available without re-running the full CG pipeline.
        """
        self.cg_bond_types_per_domain = {}
        for domain_name in self.subsets_list:
            if domain_name not in self.domain_info:
                continue
            try:
                cath_map = CATH_Map(
                    domain_name=domain_name,
                    cg_strategy=self.cg_strategy,
                    domain_index_path=self._index_path,
                )
                bt = cath_map.get_bond_types()
            except (ValueError, KeyError) as e:
                print(f"  [SKIP] domain '{domain_name}': bond building failed — {e}", flush=True)
                continue
            self.cg_bond_types_per_domain[domain_name] = {
                k: jnp.array(v, dtype=jnp.int32) if len(v) > 0
                else jnp.empty((0, 2), dtype=jnp.int32)
                for k, v in bt.items()
            }


def _process_domain_worker(args: tuple) -> dict | None:
    """Process a single CATH domain.  Module-level for multiprocessing pickling.

    When called from a worker process, JAX is forced onto CPU to avoid GPU
    memory contention between processes.  All returned arrays are numpy so
    they survive inter-process serialisation.

    Args:
        args: ``(data_dir, domain_name, domain_info_entry, index_path,
                  cg_strategy, n_frames)``

    Returns:
        Per-domain result dict, or ``None`` if the domain is skipped.
    """
    import os
    import h5py
    import numpy as np
    from jax import numpy as jnp
    from jax_md import space

    # Avoid circular import by importing lazily inside the function.
    from cgbench.core.mapping import CATH_Map, map_dataset

    data_dir, domain_name, domain_info_entry, index_path, cg_strategy, n_frames = args

    hdf5_path = os.path.join(data_dir, domain_name, f"{domain_name}_traj_all.hdf5")
    subset_path = os.path.join(data_dir, domain_name, "subset_1000_cg.npy")

    if not os.path.exists(hdf5_path):
        print(f"  [SKIP] HDF5 not found for '{domain_name}': {hdf5_path}", flush=True)
        return None

    # Skip uncapped domains
    res_names = domain_info_entry["residue_names"]
    if res_names[0] != "ACE" or res_names[-1] != "NME":
        print(
            f"  [SKIP] '{domain_name}' not capped "
            f"(first={res_names[0]}, last={res_names[-1]})",
            flush=True,
        )
        return None

    # Load frame indices
    if os.path.exists(subset_path):
        frame_indices = np.unique(np.load(subset_path).astype(int))
    else:
        print(
            f"  [WARN] subset_1000_cg.npy not found for '{domain_name}', "
            f"using all frames",
            flush=True,
        )
        frame_indices = None

    if n_frames is not None:
        frame_indices = (
            frame_indices[:n_frames] if frame_indices is not None else np.array([0])
        )

    with h5py.File(hdf5_path, "r") as hf:
        if frame_indices is not None:
            R  = hf["positions"][frame_indices]
            F  = hf["forces"][frame_indices]
            pe = hf["pe"][frame_indices]
            ke = hf["ke"][frame_indices]
        else:
            R  = hf["positions"][:]
            F  = hf["forces"][:]
            pe = hf["pe"][:]
            ke = hf["ke"][:]

    # Convert from HDF5 native units (Å, kcal/mol) to project units (nm, kJ/mol)
    R  = R  * 0.1    # Å  → nm
    F  = F  * 41.84  # kcal/mol/Å → kJ/mol/nm
    pe = pe * 4.184  # kcal/mol   → kJ/mol
    ke = ke * 4.184  # kcal/mol   → kJ/mol

    n_loaded = R.shape[0]
    displacement_fn, shift_fn = space.free()

    try:
        cath_map = CATH_Map(
            domain_name=domain_name,
            cg_strategy=cg_strategy,
            domain_index_path=index_path,
        )
        _, cg_species, cg_masses, weights = cath_map.get_map()
        weights = weights.astype(jnp.float32)
        bond_types = cath_map.get_bond_types()
        cg_coords, cg_forces = map_dataset(
            R, displacement_fn, shift_fn, weights, weights, F
        )
    except (ValueError, KeyError) as e:
        print(f"  [SKIP] {domain_name}: CG mapping failed — {e}", flush=True)
        return None

    n_cg = cg_coords.shape[1]
    print(f"  [OK] {domain_name}: {n_loaded} frames, {n_cg} CG sites", flush=True)

    return {
        "domain_name": domain_name,
        "n_cg_sites":  n_cg,
        "n_frames":    n_loaded,
        "R":           np.array(cg_coords, dtype=np.float32),
        "F":           np.array(cg_forces, dtype=np.float32),
        "mask":        np.ones((n_loaded, n_cg), dtype=bool),
        "species":     np.tile(np.array(cg_species, dtype=np.int32), (n_loaded, 1)),
        "cg_species":  np.array(cg_species, dtype=np.int32),
        "bond_types":  bond_types,
        "pe":          pe.astype(np.float32),
        "ke":          ke.astype(np.float32),
    }

