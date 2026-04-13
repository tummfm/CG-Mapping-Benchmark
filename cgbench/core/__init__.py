"""
Core functionality for CG mapping and datasets.
"""

from .mapping import (
    map_dataset,
    get_map_weights,
    Hexane_Map,
    BenzeneCrystal_Map,
    CappedPeptideMap,
    TIP3P_Water_Map,
    CATH_Map,
    UncappedProteinMap,
)
from .dataset import (
    BaseDataset,
    Hexane_Dataset,
    TIP3P_water_Dataset,
    BenzeneCrystal_Dataset,
    BenzeneCrystal288_Dataset,
    Capped_Ala_Dataset,
    Capped_Ala2_Dataset,
    Capped_Ala15_Dataset,
    Capped_Pro_Dataset,
    Capped_Thr_Dataset,
    Capped_Gly_Dataset,
    UBQ1_Dataset,
    IFC1_Dataset,
    MJC1_Dataset,
    QX5_1_Dataset,
    LYT6_Dataset,
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
from .topology import (
    TOPOLOGY_FILES,
    parse_gromacs_top,
    get_cg_indices,
)

__all__ = [
    # Mapping
    "map_dataset",
    "get_map_weights",
    "Hexane_Map",
    "BenzeneCrystal_Map",
    "CappedPeptideMap",
    "TIP3P_Water_Map",
    "CATH_Map",
    "UncappedProteinMap",
    # Dataset
    "BaseDataset",
    "Hexane_Dataset",
    "TIP3P_water_Dataset",
    "BenzeneCrystal_Dataset",
    "BenzeneCrystal288_Dataset",
    "Capped_Ala_Dataset",
    "Capped_Ala2_Dataset",
    "Capped_Ala15_Dataset",
    "Capped_Pro_Dataset",
    "Capped_Thr_Dataset",
    "Capped_Gly_Dataset",
    "UBQ1_Dataset",
    "IFC1_Dataset",
    "MJC1_Dataset",
    "QX5_1_Dataset",
    "LYT6_Dataset",
    "SPICE_Dipeptides",
    # Config
    "MD_DATASET_PATHS",
    "SEED",
    "DEFAULT_MACE_CONFIG",
    "DEFAULT_NEQUIP_CONFIG",
    "DEFAULT_TRAIN_CONFIG",
    "DEFAULT_SIM_CONFIG",
    "_get_available_datasets",
    # Topology
    "TOPOLOGY_FILES",
    "parse_gromacs_top",
    "get_cg_indices",
]
