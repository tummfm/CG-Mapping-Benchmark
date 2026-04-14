from chemtrain.data import preprocessing
import copy
from collections import defaultdict
from jax_md import space
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
    _compute_bond_types_from_cg_bond_index,
    map_dataset,
)
from .config import MD_DATASET_PATHS, SEED


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

        topology_path = ds_cfg.get("topology")
        if topology_path is None:
            raise ValueError(
                f"Dataset '{dataset_name}' must define a .tpr topology path in MD_DATASET_PATHS."
            )
        if os.path.splitext(topology_path)[1].lower() != ".tpr":
            raise ValueError(
                f"Dataset '{dataset_name}' requires .tpr topology, got: {topology_path}"
            )

        selection = ds_cfg.get("selection", "all")
        topo_meta = io.load_tpr_topology_metadata(
            topology_path=topology_path,
            config_path=ds_cfg.get("config"),
            selection=selection,
        )

        self.box = topo_meta["box"]
        self.species = topo_meta["species"]
        self.n_species = len(set(np.asarray(self.species).tolist()))

        # Topology metadata
        self.mda_universe = None
        self.bonds = topo_meta["bonds"]
        self.angles = topo_meta["angles"]
        self.dihedrals = topo_meta["dihedrals"]
        self.topology_atom_names = topo_meta.get("atom_names")
        self.topology_residue_names = topo_meta.get("residue_names")
        self.topology_residue_ids = topo_meta.get("residue_ids")

        # Inject topology metadata for map classes that accept it
        import inspect
        sig = inspect.signature(map_class.__init__)
        if "residue_names" in sig.parameters:
            map_kwargs.setdefault("residue_names", topo_meta.get("residue_names"))
            map_kwargs.setdefault("residue_ids", topo_meta.get("residue_ids"))
            map_kwargs.setdefault("n_atoms", topo_meta.get("n_atoms"))

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
            R = np.asarray(dataset_[split]["R"])
            F = np.asarray(dataset_[split]["F"])
            n_frames = int(R.shape[0])
            n_atoms = int(R.shape[1])

            if int(np.asarray(self.species).shape[0]) != n_atoms:
                raise ValueError(
                    f"Topology/data atom-count mismatch for '{self.dataset_name}' split '{split}': "
                    f"topology species count={int(np.asarray(self.species).shape[0])}, "
                    f"trajectory atom count={n_atoms}. Adjust topology selection to match data."
                )

            dataset_[split]["R"] = R
            dataset_[split]["F"] = F

            if "box" in dataset_[split] and dataset_[split]["box"] is not None:
                box = np.asarray(dataset_[split]["box"])
                dataset_[split]["box"] = box
            else:
                dataset_[split]["box"] = np.tile(np.asarray(self.box), (n_frames, 1, 1))

            dataset_[split]["species"] = np.tile(np.asarray(self.species)[None, :], (n_frames, 1))
            dataset_[split]["mask"] = np.ones((n_frames, n_atoms), dtype=bool)

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
        self.box = dataset_["training"]["box"][0]
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

    def _map_get_map(self, map_name):
        """Dispatch mapping retrieval for map classes with different signatures."""
        return self.map_obj.get_map(map_name)

    def _map_get_cg_topology(self, map_name):
        """Dispatch CG topology retrieval for map classes with different signatures."""
        return self.map_obj.get_cg_topology(map_name)

    def coarse_grain(self, map, cached_dataset_path: str | None = None):
        """Coarse grain the dataset using the specified mapping strategy.

        Args:
            map: Mapping strategy name (e.g. ``"coreBetaMap2"``).
        """
        map_indices, cg_species, cg_masses, weights = self._map_get_map(map)
        cache_path = cached_dataset_path or self._cg_cache_path(map)

        # Retrieve CG topology (bonds/angles/dihedrals) for the specified map.
        self.cg_bond_index, self.cg_angle_index, self.cg_dihedral_index = (
            self._map_get_cg_topology(map)
        )

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
                        copy.deepcopy(self.cg_dataset_X[split]),
                        scale_R=1,
                        scale_U=1,
                        fractional=True,
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

        # cg_bonds: (2, B) direct bond index (alias for cg_bond_index)
        self.cg_bonds = self.cg_bond_index

        bt = _compute_bond_types_from_cg_bond_index(self.cg_bond_index)
        self.cg_bond_types = {
            k: jnp.array(v, dtype=jnp.int32) if len(v) > 0
            else jnp.empty((0, 2), dtype=jnp.int32)
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
                copy.deepcopy(cg_dataset[split]),
                scale_R=1,
                scale_U=1,
                fractional=True,
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


class MixedDataset:
    """Compose multiple BaseDataset objects into one subset-labeled dataset."""

    def __init__(
        self,
        dataset_name: str,
        children: list[tuple[str, BaseDataset]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        cache_cg: bool = True,
    ):
        self.dataset_name = dataset_name
        self.children = children
        self.child_names = [name for name, _ in children]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.shuffle = shuffle
        self.cache_cg = bool(cache_cg)

        self.dataset_X = None
        self.dataset_U = None
        self.cg_dataset_X = None
        self.cg_dataset_U = None
        self.splits = ()

        self.box = None
        self.species = None
        self.n_species = 0
        self.n_subsets = len(children)

        self.displacement_fn_U = None
        self.shift_fn_U = None
        self.displacement_fn_X = None
        self.shift_fn_X = None

    @staticmethod
    def _merge_split_entries(entries: list[dict], subset_id: int) -> dict:
        out = copy.deepcopy(entries)
        for item in out:
            n_frames = int(item["R"].shape[0])
            item["subset"] = np.full((n_frames,), subset_id, dtype=np.int32)
        return out

    @staticmethod
    def _pad_and_concat(entries: list[dict], cg: bool = False) -> dict:
        if not entries:
            raise ValueError("No entries to merge for mixed dataset.")

        max_sites = max(int(e["R"].shape[1]) for e in entries)
        all_R, all_F, all_box, all_species, all_mask, all_subset = [], [], [], [], [], []
        all_n_sites = []

        for e in entries:
            n_frames = int(e["R"].shape[0])
            n_sites = int(e["R"].shape[1])
            pad = max_sites - n_sites

            all_R.append(np.pad(np.asarray(e["R"]), ((0, 0), (0, pad), (0, 0))))
            all_F.append(np.pad(np.asarray(e["F"]), ((0, 0), (0, pad), (0, 0))))
            all_box.append(np.asarray(e["box"]))

            species = np.asarray(e["species"]).astype(np.int32)
            all_species.append(np.pad(species, ((0, 0), (0, pad)), constant_values=0))

            mask = np.asarray(e["mask"]).astype(bool)
            all_mask.append(np.pad(mask, ((0, 0), (0, pad)), constant_values=False))

            all_subset.append(np.asarray(e["subset"]).astype(np.int32))
            if cg:
                all_n_sites.append(np.full((n_frames,), n_sites, dtype=np.int32))

        merged = {
            "R": np.concatenate(all_R, axis=0),
            "F": np.concatenate(all_F, axis=0),
            "box": np.concatenate(all_box, axis=0),
            "species": np.concatenate(all_species, axis=0),
            "mask": np.concatenate(all_mask, axis=0),
            "subset": np.concatenate(all_subset, axis=0),
        }
        if cg:
            merged["n_cg_sites"] = np.concatenate(all_n_sites, axis=0)
        return merged

    def _setup_displacement_functions(self):
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

    def load_traj(self, workers: int = 1) -> None:
        if self.dataset_X is not None and self.dataset_U is not None:
            return

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(child.load_traj): (subset_id, name, child)
                           for subset_id, (name, child) in enumerate(self.children)}
                for future in as_completed(futures):
                    future.result()  # re-raise any exception

        per_split_entries: dict[str, list[dict]] = defaultdict(list)

        for subset_id, (name, child) in enumerate(self.children):
            if workers <= 1:
                child.load_traj()
            for split, split_data in child.dataset_X.items():
                merged_entry = copy.deepcopy(split_data)
                n_frames = int(merged_entry["R"].shape[0])
                merged_entry["subset"] = np.full((n_frames,), subset_id, dtype=np.int32)
                per_split_entries[split].append(merged_entry)

        out_X: dict[str, dict] = {}
        for split, entries in per_split_entries.items():
            out_X[split] = self._pad_and_concat(entries, cg=False)

        self.dataset_X = out_X
        self.splits = out_X.keys()

        out_U: dict[str, dict] = {}
        for split in self.splits:
            out_U[split] = io.scale_dataset_box_aware(
                copy.deepcopy(self.dataset_X[split]), scale_R=1, scale_U=1, fractional=True
            )
        self.dataset_U = out_U

        self.box = self.dataset_X["training"]["box"][0]
        self.species = self.dataset_X["training"]["species"][0]
        train_mask = self.dataset_X["training"]["mask"][0]
        self.n_species = len(np.unique(self.species[train_mask]))
        self._setup_displacement_functions()

    def coarse_grain(self, map, cached_dataset_path: str | None = None, workers: int = 1):
        if self.dataset_X is None or self.dataset_U is None:
            self.load_traj(workers=workers)

        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(child.coarse_grain, map): (subset_id, name, child)
                           for subset_id, (name, child) in enumerate(self.children)}
                for future in as_completed(futures):
                    future.result()  # re-raise any exception

        per_split_entries: dict[str, list[dict]] = defaultdict(list)
        cg_species_per_subset: dict[int, np.ndarray] = {}
        cg_bond_types_per_subset: dict[int, dict] = {}

        for subset_id, (name, child) in enumerate(self.children):
            if workers <= 1:
                child.coarse_grain(map)
            cg_species_per_subset[subset_id] = np.asarray(child.cg_species)
            cg_bond_types_per_subset[subset_id] = getattr(child, "cg_bond_types", None)

            for split, split_data in child.cg_dataset_X.items():
                merged_entry = copy.deepcopy(split_data)
                n_frames = int(merged_entry["R"].shape[0])
                merged_entry["subset"] = np.full((n_frames,), subset_id, dtype=np.int32)
                per_split_entries[split].append(merged_entry)

        out_X: dict[str, dict] = {}
        for split, entries in per_split_entries.items():
            out_X[split] = self._pad_and_concat(entries, cg=True)

        self.cg_dataset_X = out_X
        self.splits = out_X.keys()

        out_U: dict[str, dict] = {}
        for split in self.splits:
            out_U[split] = io.scale_dataset_box_aware(
                copy.deepcopy(self.cg_dataset_X[split]), scale_R=1, scale_U=1, fractional=True
            )
        self.cg_dataset_U = out_U

        self.cg_species_per_subset = cg_species_per_subset
        self.cg_bond_types_per_subset = cg_bond_types_per_subset

        self.box = self.cg_dataset_X["training"]["box"][0]
        self.cg_species = self.cg_dataset_X["training"]["species"][0]
        self.n_cg_sites = int(self.cg_dataset_X["training"]["R"].shape[1])
        train_mask = self.cg_dataset_X["training"]["mask"][0]
        self.n_cg_species = len(np.unique(self.cg_species[train_mask]))
        self.cg_masses = None
        self.cg_weights = None

        self._setup_displacement_functions()


class CATHDomain_Dataset(BaseDataset):
    """Single CATH domain dataset with strict .tpr topology loading."""

    def __init__(
        self,
        domain_name: str,
        cg_strategy: str = "coreBetaMap2",
        mapping_type: str = "slice",
        domain_index_path: str | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        cache_cg: bool = True,
    ):
        cath_root = MD_DATASET_PATHS["CATH"]["path"]
        domain_dir = os.path.join(cath_root, domain_name)
        ds_key = f"CATH::{domain_name}"

        MD_DATASET_PATHS[ds_key] = {
            "path": os.path.join(domain_dir, "dataset.npz"),
            "config": os.path.join(domain_dir, "md.gro"),
            "topology": os.path.join(domain_dir, "md.tpr"),
            "selection": "protein",
        }

        self.domain_name = domain_name
        self.cg_strategy = cg_strategy
        self.mapping_type = mapping_type
        self.domain_index_path = domain_index_path

        super().__init__(
            dataset_name=ds_key,
            map_class=CATH_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={
                "domain_name": domain_name,
                "cg_strategy": cg_strategy,
                "mapping_type": mapping_type,
                "domain_index_path": domain_index_path,
            },
            cache_cg=cache_cg,
        )

    def coarse_grain(self, map, cached_dataset_path: str | None = None):
        if map != self.cg_strategy:
            self.cg_strategy = map
            self.map_obj = CATH_Map(
                domain_name=self.domain_name,
                cg_strategy=map,
                mapping_type=self.mapping_type,
                domain_index_path=self.domain_index_path,
                residue_names=self.topology_residue_names,
                residue_ids=self.topology_residue_ids,
                n_atoms=len(self.topology_residue_ids) if self.topology_residue_ids is not None else None,
            )
            self.masses = jnp.array(self.map_obj.at_masses)
        return super().coarse_grain(map, cached_dataset_path=cached_dataset_path)

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
        if cg:
            if not hasattr(self, "cg_dataset_X"):
                raise RuntimeError("Call coarse_grain() before save_xyz(cg=True).")
            data = self.cg_dataset_X[split]
            map_name = getattr(self, "cg_map_name", None)
            atom_info = io.get_atom_info(self.map_obj, map_name=map_name, cg=True)
            bond_index = getattr(self, "cg_bond_index", None)
        else:
            data = self.dataset_X[split]
            atom_info = io.get_atom_info(self.map_obj, cg=False)
            bond_index = getattr(self, "bonds", None)

        R_nm = np.asarray(data["R"])
        if n_frames is not None:
            R_nm = R_nm[:n_frames]
        if unwrap:
            R_nm = io.unwrap_trajectory(R_nm, self.displacement_fn_X, bond_index)

        io.write_xyz_trajectory(output_path, R_nm, atom_info, workers=workers)

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
            atom_info = io.get_atom_info(self.map_obj, map_name=_map, cg=True)
        else:
            data = self.dataset_X[split]
            bond_index = getattr(self, "bonds", None)
            atom_info = io.get_atom_info(self.map_obj, cg=False)

        R_nm = np.asarray(data["R"])
        if n_frames is not None:
            R_nm = R_nm[:n_frames]
        if unwrap:
            R_nm = io.unwrap_trajectory(R_nm, self.displacement_fn_X, bond_index)

        io.write_pdb_trajectory_with_bonds(
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
        snap = io.load_single_gro_snapshot_dataset(cfg_path)
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
        snap = io.load_single_gro_snapshot_dataset(cfg_path)
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


class Capped_Ala_Dataset(BaseDataset):
    """ACE-ALA-NME capped alanine dipeptide dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_ala",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "ALA", "NME"], "mapping_type": mapping_type},
            cache_cg=cache_cg,
        )


class Capped_Ala2_Dataset(BaseDataset):
    """ACE-ALA-ALA-NME capped alanine tripeptide dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_ala2",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={
                "residue_sequence": ["ACE", "ALA", "ALA", "NME"],
                "mapping_type": mapping_type,
            },
            cache_cg=cache_cg,
        )


class Capped_Ala15_Dataset(BaseDataset):
    """ACE + 15xALA + NME capped poly-alanine dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_ala15",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={
                "residue_sequence": ["ACE"] + ["ALA"] * 15 + ["NME"],
                "mapping_type": mapping_type,
            },
            cache_cg=cache_cg,
        )


class Capped_Pro_Dataset(BaseDataset):
    """ACE-PRO-NME capped proline dipeptide dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_pro",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "PRO", "NME"], "mapping_type": mapping_type},
            cache_cg=cache_cg,
        )


class Capped_Thr_Dataset(BaseDataset):
    """ACE-THR-NME capped threonine dipeptide dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_thr",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "THR", "NME"], "mapping_type": mapping_type},
            cache_cg=cache_cg,
        )


class Capped_Gly_Dataset(BaseDataset):
    """ACE-GLY-NME capped glycine dipeptide dataset."""

    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
        shuffle=True,
        cache_cg: bool = True,
        mapping_type: str = "slice",
    ):
        super().__init__(
            dataset_name="capped_gly",
            map_class=CappedPeptideMap,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            map_kwargs={"residue_sequence": ["ACE", "GLY", "NME"], "mapping_type": mapping_type},
            cache_cg=cache_cg,
        )


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

    Uses ``space.periodic_general`` with the reference 6x6x6 nm box.
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


class CATH_Dataset(MixedDataset):
    """CATH dataset as a MixedDataset of per-domain BaseDataset children."""

    def __init__(
        self,
        dataset_key: str = "cath_full",
        cg_strategy: str = "coreBetaMap2",
        mapping_type: str = "slice",
        domain_index_path: str | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        cached_dataset_path: str | None = None,
        cache_cg: bool = True,
    ):
        from .mapping import _DEFAULT_DOMAIN_INDEX

        key_alias = {
            "cath": "CATH",
            "cath_full": "CATH",
            "CATH_full": "CATH",
            "cath_quarter": "CATH",
            "cath_test": "CATH",
        }
        normalized_key = key_alias.get(dataset_key, dataset_key)
        if normalized_key != "CATH":
            raise ValueError(
                "CATH_Dataset now uses per-domain composition from the full CATH folder only. "
                f"Received dataset_key='{dataset_key}'."
            )
        if dataset_key in ("cath_quarter", "cath_test"):
            print(
                f"[WARN] dataset_key='{dataset_key}' is mapped to full per-domain CATH composition."
            )

        index_candidates = []
        if domain_index_path is not None:
            index_candidates.append(domain_index_path)
        index_candidates.append(_DEFAULT_DOMAIN_INDEX)
        index_candidates.append("/ds/project/franz/projects/CG-Bench/data/domain_residue_index.json")

        index_path = None
        for cand in index_candidates:
            if cand is not None and os.path.exists(cand):
                index_path = cand
                break
        if index_path is None:
            raise FileNotFoundError(
                "Could not locate domain_residue_index.json. Tried: "
                + ", ".join([str(c) for c in index_candidates if c is not None])
            )

        with open(index_path) as f:
            domain_index = json.load(f)

        self.dataset_key = normalized_key
        self.cg_strategy = cg_strategy
        self.mapping_type = mapping_type
        self._index_path = index_path
        self.subsets_list = list(domain_index["subsets"])
        self.domain_info = dict(domain_index["domains"])
        self.cached_dataset_path = cached_dataset_path

        cath_root = MD_DATASET_PATHS["CATH"]["path"]
        children: list[tuple[str, BaseDataset]] = []
        missing_domains: list[str] = []

        for domain_name in self.subsets_list:
            domain_dir = os.path.join(cath_root, domain_name)
            npz_path = os.path.join(domain_dir, "dataset.npz")
            gro_path = os.path.join(domain_dir, "md.gro")
            tpr_path = os.path.join(domain_dir, "md.tpr")

            if not (os.path.exists(npz_path) and os.path.exists(gro_path) and os.path.exists(tpr_path)):
                missing_domains.append(domain_name)
                continue

            child = CATHDomain_Dataset(
                domain_name=domain_name,
                cg_strategy=cg_strategy,
                mapping_type=mapping_type,
                domain_index_path=index_path,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                shuffle=shuffle,
                cache_cg=cache_cg,
            )
            children.append((domain_name, child))

        if not children:
            raise RuntimeError("No CATH domains with complete dataset.npz/md.gro/md.tpr were found.")

        if missing_domains:
            print(
                f"[WARN] Skipping {len(missing_domains)} CATH domains with incomplete files."
            )

        super().__init__(
            dataset_name="CATH",
            children=children,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle,
            cache_cg=cache_cg,
        )

    def coarse_grain(self, map=None, cached_dataset_path=None, workers: int = 1):
        effective_map = map or self.cg_strategy
        self.cg_strategy = effective_map
        return super().coarse_grain(effective_map, cached_dataset_path=cached_dataset_path, workers=workers)

