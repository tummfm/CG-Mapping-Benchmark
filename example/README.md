# Example Outputs
This folder contains some example outputs (models + simulations).

## Capped Alanine (Alanine dipeptide)
### Training
`python scripts/run_fm_training.py --mol capped_ala --cgmap coreBetaMap2 --stride 10 --epochs 20 --model mace --device 1` 
With `stride=10`, this amounts to 50k frames, of which 45k are used for training and 5k for validation (model selection).

### Simulation
`python scripts/run_simulation.py --model outputs/Model=mace/Capped_ala_map=coreBetaMap2_tr=0.9_rcut=0.5_epochs=20_seed=22_prior=None_stride=10_int=2_corr=2_maxL=3_eq=O3/best_params.pkl --t-total 1000 --n-chains 10 --mol capped_ala --device 1` 
Aka 10 parallel simulations each with a length of 1000 ps. By default, the starting frames of each simulation are randomly selected from the validation set.