# Setup for reference simulations

This folder contains the GROMACS code for the atomistic reference data.

## Liquid hexane
The liquid hexane example follows the protocol of Ruehle et al, as outlined in the [VOTCA Coarse-graining example](https://gitlab.mpcdf.mpg.de/votca/votca/-/tree/master/csg-tutorials/hexane).

## Peptides
The `/peptides/` folder contains the workflow that was used for all capped amino acids and the alanine 15-mer. As an example, we only show the files of the Alanine 15-mer run. To run other peptides, change the initial PDB structure in `/coords/`. Since we use capped peptides, the ion addition can also be removed.