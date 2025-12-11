from chemtrain.data import preprocessing
import copy
from jax_md import space
from jax import numpy as jnp
import numpy as np
from ..utils import helpers as utils
from .mapping import Ala2_Map, Hexane_Map, Ala15_Map, Pro2_Map, Thr2_Map, Gly2_Map, map_dataset
from .config import Dataset_paths, SEED


class BaseDataset:
    """Base class for molecular dynamics datasets with coarse-graining support."""
    
    def __init__(self, dataset_name, map_class, train_ratio=0.7, val_ratio=0.1, 
                 shuffle=True, map_kwargs=None):
        """
        Initialize dataset with train/val/test splits.
        
        Args:
            dataset_name: Key for Dataset_paths dictionary
            map_class: Mapping class (e.g., Ala2_Map, Hexane_Map)
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            shuffle: Whether to shuffle data during split
            map_kwargs: Keyword arguments for map_class initialization
        """
        self.dataset_name = dataset_name
        map_kwargs = map_kwargs or {}
        
        print(f"Loading {dataset_name} dataset from:", Dataset_paths[dataset_name])
        dataset = np.load(Dataset_paths[dataset_name], allow_pickle=True)
        dataset = dict(dataset)

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
            "testing": test_data,
        }

        # Ensure all required fields are present
        for split in dataset_.keys():
            dataset_[split]["R"] = dataset_[split]["R"]
            dataset_[split]["F"] = dataset_[split]["F"]
            dataset_[split]["box"] = dataset_[split]["box"]
            dataset_[split]["species"] = dataset_[split]["species"]
            dataset_[split]["mask"] = dataset_[split]["mask"]

        self.dataset_X = copy.deepcopy(dataset_)

        # Create fractional coordinate versions
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

        # Initialize mapping
        self.map_obj = map_class(**map_kwargs)
        self.masses = jnp.array(self.map_obj.at_masses)
        self.at_map_species = self.map_obj.map_species

        # Set up displacement and shift functions for periodic boundary conditions
        self._setup_displacement_functions()

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

    def coarse_grain(self, map):
        """
        Coarse grain the dataset using the mapping defined in the class.
        
        Args:
            map: Mapping specification to use
        """
        map, cg_species, cg_masses, weights = self.map_obj.get_map(map)
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


class Hexane_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        super().__init__(
            dataset_name="hexane",
            map_class=Hexane_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=True,
            map_kwargs={"nmol": 100}
        )
        self.hexane_map = self.map_obj


class Ala2_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        super().__init__(
            dataset_name="ala2",
            map_class=Ala2_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle
        )
        self.ala2_map = self.map_obj


class Ala15_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1, shuffle=True):
        super().__init__(
            dataset_name="ala15",
            map_class=Ala15_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=shuffle
        )
        self.ala15_map = self.map_obj


class Pro2_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        super().__init__(
            dataset_name="pro2",
            map_class=Pro2_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=True
        )
        self.pro2_map = self.map_obj


class Thr2_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        super().__init__(
            dataset_name="thr2",
            map_class=Thr2_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=True
        )
        self.thr2_map = self.map_obj


class Gly2_Dataset(BaseDataset):
    def __init__(self, train_ratio=0.7, val_ratio=0.1):
        super().__init__(
            dataset_name="gly2",
            map_class=Gly2_Map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            shuffle=True
        )
        self.gly2_map = self.map_obj