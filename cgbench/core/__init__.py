"""
Core functionality for CG mapping and datasets.
"""

from .mapping import (
    map_dataset,
    get_map_weights,
    Hexane_Map,
    Ala2_Map,
    Ala15_Map,
    Pro2_Map,
    Gly2_Map,
    Thr2_Map,
)
from .dataset import (
    BaseDataset,
    Hexane_Dataset,
    Ala2_Dataset,
    Ala15_Dataset,
    Pro2_Dataset,
    Thr2_Dataset,
    Gly2_Dataset,
)
from .config import (
    Dataset_paths,
    SEED,
    DEFAULT_MACE_CONFIG,
    DEFAULT_NEQUIP_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_SIM_CONFIG,
    _get_available_datasets,
)

__all__ = [
    # Mapping
    "map_dataset",
    "get_map_weights",
    "Hexane_Map",
    "Ala2_Map",
    "Ala15_Map",
    "Pro2_Map",
    "Gly2_Map",
    "Thr2_Map",
    # Dataset
    "BaseDataset",
    "Hexane_Dataset",
    "Ala2_Dataset",
    "Ala15_Dataset",
    "Pro2_Dataset",
    "Thr2_Dataset",
    "Gly2_Dataset",
    # Config
    "Dataset_paths",
    "SEED",
    "DEFAULT_MACE_CONFIG",
    "DEFAULT_NEQUIP_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "DEFAULT_SIM_CONFIG",
    "_get_available_datasets",
]

