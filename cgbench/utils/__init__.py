"""
Utility functions for analysis and data processing.

Submodules:
- io: File I/O utilities (trajectory loading, XYZ writing)
- chains: Chain splitting and filtering utilities
- geometry: Geometry calculations (dihedrals, angles, distances)
- structural: Structural analysis (RDF, radius of gyration, helicity)
"""

from . import io
from . import chains
from . import geometry
from . import structural

from .io import (
    load_trajectory,
    prepare_output_dir,
    save_xyz_frames_parallel,
)
from .chains import (
    get_line_locations,
    compute_line_locations,
    split_into_chains,
    setup_distance_filter_fn,
    mark_nan,
    calculate_stability,
    compute_bond_metrics,
)
from .geometry import (
    init_dihedral_fn,
    init_angle_fn,
    compute_atom_distance,
    calculate_dihedral,
    calc_mse_dihedrals,
    periodic_displacement,
)
from .structural import (
    radius_of_gyration_vectorized,
    helicity_vectorized,
    xi_norm_vectorized,
    calculate_rdf,
    calculate_rdf_mse,
    calculate_rdf_mse_from_dict,
)

__all__ = [
    "io",
    "chains",
    "geometry",
    "structural",
    # io functions
    "load_trajectory",
    "prepare_output_dir",
    "save_xyz_frames_parallel",
    # chains functions
    "get_line_locations",
    "compute_line_locations",
    "split_into_chains",
    "setup_distance_filter_fn",
    "mark_nan",
    "calculate_stability",
    "compute_bond_metrics",
    # geometry functions
    "init_dihedral_fn",
    "init_angle_fn",
    "compute_atom_distance",
    "calculate_dihedral",
    "calc_mse_dihedrals",
    "periodic_displacement",
    # structural functions
    "radius_of_gyration_vectorized",
    "helicity_vectorized",
    "xi_norm_vectorized",
    "calculate_rdf",
    "calculate_rdf_mse",
    "calculate_rdf_mse_from_dict",
]
