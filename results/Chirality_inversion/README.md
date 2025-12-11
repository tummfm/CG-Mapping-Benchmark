# WTMetaD Simulations for Chiral Inversion Barrier

This repo contains the setup and evaluation for the well-tempered metadynamics (WTMetaD) simulations along the improper CA dihedral.

## Important files

- `chirality_metadynamics.ipynb` contains the analysis of the results, starting from the `HILLS` output from PLUMED.
- `/structures/` contains the initial starting configurations. In all cases a specific frame from the reference atomistic simulation was chosen and mapped to the CG space. The frame was chosen so it represents the capped alanine in the C7eq minima of the backbone dihedrals.
- `/metadynamics/` contains the scripts for the PLUMED+LAMMPS runs. Output files were omitted due to size constraints.
- `/model/` contains the exported models from chemtrain deploy (Omitted due to size). 