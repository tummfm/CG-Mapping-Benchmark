"""
Plotting utilities for CG-Mapping-Benchmark.

Submodules:
- style: Shared styling constants and setup utilities
- distributions: 1D histogram plotting for structural distributions
- structural: Structural visualization (RDF, Ramachandran, helicity)
- timeseries: Time series plotting
- training: Training diagnostics
- molecules: High-level molecule visualization routines
"""

from . import style
from . import distributions
from . import structural
from . import timeseries
from . import training
from . import molecules
from . import priors

__all__ = [
    "style",
    "distributions",
    "structural",
    "timeseries",
    "training",
    "molecules",
    "priors",
]
