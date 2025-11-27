from chemtrain.data import preprocessing
import copy
from jax_md import space
from jax import numpy as jnp
import numpy as np
import utils 
from mapping import Ala2_Map, Hexane_Map, map_dataset, Ala15_Map, Pro2_Map, Thr2_Map, Gly2_Map
from constants import Dataset_paths


class Hexane_Dataset:
    def __init__(
        self,
        train_ratio=0.7,
        val_ratio=0.1,
    ):
        print("Loading hexane dataset from:", Dataset_paths["hexane"])

        dataset = np.load(Dataset_paths["hexane"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=True,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.hexane_map = Hexane_Map(nmol=100)
        self.masses = jnp.array(self.hexane_map.at_masses)
        self.at_map_species = self.hexane_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.hexane_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights


class Ala2_Dataset:
    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        print("Loading Ala2 dataset from:", Dataset_paths["ala2"])
        dataset = np.load(Dataset_paths["ala2"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=shuffle,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.ala2_map = Ala2_Map()
        self.masses = jnp.array(self.ala2_map.at_masses)
        self.at_map_species = self.ala2_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.ala2_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights


class Ala15_Dataset:
    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        print("Loading Ala15 dataset from:", Dataset_paths["ala15"])
        dataset = np.load(Dataset_paths["ala15"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=shuffle,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.ala15_map = Ala15_Map()
        self.masses = jnp.array(self.ala15_map.at_masses)
        self.at_map_species = self.ala15_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.ala15_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights


class Pro2_Dataset:
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        print("Loading pro2 dataset from:", Dataset_paths["pro2"])
        dataset = np.load(Dataset_paths["pro2"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=True,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.pro2_map = Pro2_Map()
        self.masses = jnp.array(self.pro2_map.at_masses)
        self.at_map_species = self.pro2_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.pro2_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights


class Thr2_Dataset:
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        print("Loading thr2 dataset from:", Dataset_paths["thr2"])
        dataset = np.load(Dataset_paths["thr2"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=True,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.thr2_map = Thr2_Map()
        self.masses = jnp.array(self.thr2_map.at_masses)
        self.at_map_species = self.thr2_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.thr2_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights


class Gly2_Dataset:
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        print("Loading gly2 dataset from:", Dataset_paths["gly2"])
        dataset = np.load(Dataset_paths["gly2"], allow_pickle=True)
        dataset = dict(dataset)

        train_data, val_data, test_data = preprocessing.train_val_test_split(
            dataset,
            shuffle=True,
            shuffle_seed=11,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        dataset_ = {
            "training": train_data,
            "validation": val_data,
            "testing": test_data,
        }

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
            out = utils.scale_dataset(
                dataset_[split], scale_R=1, scale_U=1, fractional=True
            )
            dataset_frac[split] = out

        print("Training set size:", dataset_["training"]["R"].shape[0])
        print("Validation set size:", dataset_["validation"]["R"].shape[0])

        self.dataset_U = dataset_frac
        self.species = dataset_["training"]["species"][0]
        self.box = dataset_["training"]["box"][0]
        self.n_species = len(set(self.species))

        self.gly2_map = Gly2_Map()
        self.masses = jnp.array(self.gly2_map.at_masses)
        self.at_map_species = self.gly2_map.map_species

        # Displacement and shift functions for periodic boundary conditions
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

    def coarse_grain(self, map):
        """Coarse grain the dataset using the mapping defined in the class."""
        map, cg_species, cg_masses, weights = self.gly2_map.get_map(map)
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
                "box": jnp.tile(self.box, (n_frames, 1, 1)),  # (T, 3, 3) or (T, 3)
                "mask": jnp.ones((n_frames, n_cg_sites), dtype=bool),
            }

        self.cg_dataset_X = copy.deepcopy(cg_dataset)

        cg_dataset_frac = {}
        self.splits = cg_dataset.keys()
        for split in self.splits:
            out = utils.scale_dataset(
                cg_dataset[split], scale_R=1, scale_U=1, fractional=True
            )
            cg_dataset_frac[split] = out

        self.cg_dataset_U = cg_dataset_frac
        self.cg_species = cg_species
        self.n_cg_sites = n_cg_sites
        self.n_cg_species = n_cg_species
        self.cg_masses = cg_masses
        self.cg_weights = weights
