"""
Core functionality for CG mapping and datasets.
"""

from .prior import BoltzmannPrior
from .mapping import (
    map_dataset,
    get_map_weights,
    Hexane_Map,
    BenzeneCrystal_Map,
    CappedPeptideMap,
    TIP3P_Water_Map,
    CATH_Map,
)
from .dataset import (
    BaseDataset,
    Hexane_Dataset,
    TIP3P_water_Dataset,
    BenzeneCrystal_Dataset,
    Capped_Ala_Dataset,
    Capped_Ala2_Dataset,
    Capped_Ala15_Dataset,
    Capped_Pro_Dataset,
    Capped_Thr_Dataset,
    Capped_Gly_Dataset,
    SPICE_Dipeptides,
)
from .config import (
    MD_DATASET_PATHS,
    SEED,
    DEFAULT_MACE_CONFIG,
    DEFAULT_NEQUIP_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_SIM_CONFIG,
    _get_available_datasets,
)
__all__ = [
    # Prior
    "BoltzmannPrior",
    # Mapping
    "map_dataset",
    "get_map_weights",
    "Hexane_Map",
    "BenzeneCrystal_Map",
    "CappedPeptideMap",
    "TIP3P_Water_Map",
    "CATH_Map",
    # Dataset
    "BaseDataset",
    "Hexane_Dataset",
    "TIP3P_water_Dataset",
    "BenzeneCrystal_Dataset",
    "Capped_Ala_Dataset",
    "Capped_Ala2_Dataset",
    "Capped_Ala15_Dataset",
    "Capped_Pro_Dataset",
    "Capped_Thr_Dataset",
    "Capped_Gly_Dataset",
    "SPICE_Dipeptides",
    # Config
    "MD_DATASET_PATHS",
    "SEED",
    "DEFAULT_MACE_CONFIG",
    "DEFAULT_NEQUIP_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "DEFAULT_SIM_CONFIG",
    "_get_available_datasets",
]
