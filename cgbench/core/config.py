from chemtrain import quantity

Dataset_paths = {
    'hexane': 'data/reference_simulations/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.npz',
    'ala2': '/home/franz/l-ala2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'pro2': '/home/franz/l-pro2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'thr2': '/home/franz/l-thr2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'gly2': '/home/franz/l-gly2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'ala15': '/home/franz/Lala15_ttot=500ns_dt=0.5fs.npz',
}
def _get_available_datasets():
    return list(Dataset_paths.keys())

# Global configurations
SEED = 22

DEFAULT_MACE_CONFIG = {
    "hidden_irreps": "32x0e + 32x1o",
    "readout_mlp_irreps": "16x0e",
    "output_irreps": "1x0e",
    "max_ell": 3,
    "num_interactions": 2,
    "correlation": 2,
    "n_radial_basis": 8,
    "train_ratio": 0.9,  
    "val_ratio": 0.1,  # Ratio of validation data, (Train_ratio + Val_ratio <= 1.0, rest is test data)
    "PRNGKey_seed": SEED,
}

DEFAULT_NEQUIP_CONFIG = {
    "train_ratio": 0.9,  
    "val_ratio": 0.1,
    "PRNGKey_seed": SEED,
}

DEFAULT_TRAIN_CONFIG = {
    "batch_size": 32,
    "init_lr": 0.001,
    "num_epochs": 50,
    "decay_rate": 0.90,
    "optimizer": "adam+decay",
}

DEFAULT_SIM_CONFIG = {
    "gamma": 100.0, # Friction coefficient in 1/ps (for NVT Langevin)
    "dt_values_fs": [2],  # Add more dt values as needed
    "print_every": 0.5,  # Save frame every 0.5 ps
    "sim_mode": "sampling",  # simulation mode: 'sampling', 'stability', 'helix', 'speed'
    "ensemble": "NVT", # NVT or NVE
    "t_eq": 0,  # Equlibration time in ps
    "t_total": 1000,  # Total simulation time in ps (- t_eq)
    "n_chains": 10, # Number of simulations (parallel)
    "kT": 300.0 * quantity.kb,  # Temperature in energy units
    "T": 300.0,
    "PRNGKey_seed": SEED,
}