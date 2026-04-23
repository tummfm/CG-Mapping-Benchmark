from chemtrain import quantity

BASE_DATASET_PATH = "/ds/project/franz/Datasets" # default "./data/reference_simulations/"

MD_DATASET_PATHS = {
    "hexane": {
        "path": f"{BASE_DATASET_PATH}/liquid_hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.npz",
        "config": f"{BASE_DATASET_PATH}/liquid_hexane/md.gro",
        "topology": f"{BASE_DATASET_PATH}/liquid_hexane/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/liquid_hexane/md.xtc",
    },
    "benzene_crystal": {
        "path": f"{BASE_DATASET_PATH}/BenzeneCrystal/benzene_crystal.npz",
        "config": f"{BASE_DATASET_PATH}/BenzeneCrystal/crys_elong_nvt_1.gro",
        "topology": f"{BASE_DATASET_PATH}/BenzeneCrystal/crys_prod_1fs.tpr",
        "traj": f"{BASE_DATASET_PATH}/BenzeneCrystal/crys_prod_1fs.trr", # if no xtc is given, we assume the trr contains both coordinates and forces, and use it for traj as well
    },
    "capped_ala": { # also referred to as alanine dipeptide
        "path": f"{BASE_DATASET_PATH}/Capped_L-Ala/l-ala2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_ala2": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Ala/l-ala2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Ala/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_ala3": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Ala2/L-ala3_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Ala2/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Ala2/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Ala2/md.trr",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Ala2/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_pro": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Pro/l-pro2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Pro/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Pro/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Pro/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Pro/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_thr": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Thr/l-thr2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Thr/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Thr/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Thr/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Thr/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_gly": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Gly/l-gly2_ttot=500ns_dt=0.5fs_nstxout=2000.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Gly/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Gly/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Gly/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Gly/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "capped_ala15": {
        "path": f"{BASE_DATASET_PATH}/Capped_L-Ala15/l-ala15_ttot=500ns_dt=0.5fs.npz",
        "config": f"{BASE_DATASET_PATH}/Capped_L-Ala15/md.gro",
        "topology": f"{BASE_DATASET_PATH}/Capped_L-Ala15/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/Capped_L-Ala15/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/Capped_L-Ala15/md.trr",
        "selection": "not resname SOL WAT HOH",
    },
    "tip3p-water": {
        "path": f"{BASE_DATASET_PATH}/TIP3P-water/tip3p-water_ttot=500ns_dt=0.5fs_nstxout=2000_stride=1_nframes=500001_nmol=901.npz",
        "config": f"{BASE_DATASET_PATH}/TIP3P-water/md.gro",
        "topology": f"{BASE_DATASET_PATH}/TIP3P-water/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/TIP3P-water/md.xtc",
        "traj_forces": f"{BASE_DATASET_PATH}/TIP3P-water/md.trr",
    },   
    "CATH": {
        "path": f"{BASE_DATASET_PATH}/CATH/",
        "config_pattern": f"{BASE_DATASET_PATH}/CATH/*/md.gro", 
        "topology_pattern": f"{BASE_DATASET_PATH}/CATH/*md.tpr",
        "npz_pattern": f"{BASE_DATASET_PATH}/CATH/*/dataset.npz",
        "domains": ['1b43A02','1bl0A02','1d3yA01','1iz6A02','1mpgA03','1neqA00','1on2A01','1r3fA02','1s6lA01','1skyB01','1sxjE02','1wfxA01','1xovA03','1z1vA00','2au3A04','2bh1X00','2ckkA01','2dl0A01','2e8oA01','2f48A03','2ga1A02','2hbpA00','2heoA00','2htjA01','2hyvA01','2nttA02','2v0cA03','2wg5F02','3a5zD02','3ethA03','3g7lA00','3luyA02','3ossC00','3tj8A02','3udcA02','3vseA01','4a53A01','4hwiB01','4kdiD00','4npsA02','4o96A01']
    },
    "spice_dipeptides": {
        "path": f"{BASE_DATASET_PATH}/SPICE/spice_dipeptides_coreBetaMap2.npz",
    },
    "cath_test": {
        "path": f"{BASE_DATASET_PATH}/cath_10.npz"
    },
    "3bpa": { # unbiased
        "path": f"{BASE_DATASET_PATH}/3BPA/3bpa.npz",
        "config": f"{BASE_DATASET_PATH}/3BPA/bpa_prod_new.gro",
        "topology": f"{BASE_DATASET_PATH}/3BPA/bpa_prod_new.tpr",
        "traj": f"{BASE_DATASET_PATH}/3BPA/bpa_prod_new.trr",  # trr contains both coordinates and forces
    },
    "3bpa_biased": {
        "path": f"{BASE_DATASET_PATH}/3BPA_biased/3bpa_biased.npz",
        "config": f"{BASE_DATASET_PATH}/3BPA_biased/opes_bpa.gro",
        "topology": f"{BASE_DATASET_PATH}/3BPA_biased/opes_bpa.tpr",
        "traj": f"{BASE_DATASET_PATH}/3BPA_biased/recomputed_forces.trr",  # trr contains both coordinates and forces
    },
    "azobenzene_biased": {
        "path": f"{BASE_DATASET_PATH}/Azobenzene_biased/azobenzene_biased.npz",
        "config": f"{BASE_DATASET_PATH}/Azobenzene_biased/azo_prod.gro",
        "topology": f"{BASE_DATASET_PATH}/Azobenzene_biased/opes_azo.tpr",
        "traj": f"{BASE_DATASET_PATH}/Azobenzene_biased/recomputed_forces.trr",  # trr contains both coordinates and forces
    },
}
STATIC_FRAME_DATASET_PATHS = {
    "1UBQ": {
        "config": f"{BASE_DATASET_PATH}/StaticFrame/1ubq/coords.gro",
        "topology": f"{BASE_DATASET_PATH}/StaticFrame/1ubq/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/StaticFrame/1ubq/simulation.xtc", # coords only, no forces
    },
    "1IFC": {
        "config": f"{BASE_DATASET_PATH}/StaticFrame/1ifc/coords.gro",
        "topology": f"{BASE_DATASET_PATH}/StaticFrame/1ifc/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/StaticFrame/1ifc/simulation.xtc",
    },
    "1MJC": {
        "config": f"{BASE_DATASET_PATH}/StaticFrame/1mjc/coords.gro",
        "topology": f"{BASE_DATASET_PATH}/StaticFrame/1mjc/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/StaticFrame/1mjc/simulation.xtc",
    },
    "1QX5": {
        "config": f"{BASE_DATASET_PATH}/StaticFrame/1qx5/coords.gro",
        "topology": f"{BASE_DATASET_PATH}/StaticFrame/1qx5/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/StaticFrame/1qx5/simulation.xtc",
    },
    "6LYT": {
        "config": f"{BASE_DATASET_PATH}/StaticFrame/6lyt/coords.gro",
        "topology": f"{BASE_DATASET_PATH}/StaticFrame/6lyt/md.tpr",
        "traj": f"{BASE_DATASET_PATH}/StaticFrame/6lyt/simulation.xtc",
    },
}


def _get_available_datasets():
    return list(MD_DATASET_PATHS.keys())

# Global configurations
SEED = 22

DEFAULT_MACE_CONFIG = {
    "hidden_irreps": "32x0e+32x1o",
    "readout_mlp_irreps": "16x0e",  # MLPirreps in MACEJAX
    "output_irreps": "1x0e",
    "max_ell": 3,
    "num_interactions": 2,
    "correlation": 3,
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

DEFAULT_SPLINE_CONFIG = {
    "n_knots_nb": 20,
    "n_knots_bond": 20,
    "n_knots_angle": 20,
    "n_knots_dihedral": 20,
    "r_onset_fraction": 0.9,   # r_onset = r_onset_fraction * r_cutoff
    "train_ratio": 0.9,
    "val_ratio": 0.1,
    "PRNGKey_seed": SEED,
}

DEFAULT_TRAIN_CONFIG = {
    "batch_size": 32,
    "init_lr": 0.001,
    "num_epochs": 5,
    "decay_rate": 0.95,
    "optimizer": "adam+decay",
    "cache": 100,
    "lr_schedule": "exponential",
    "weight_decay": 0.0,
    "optimizer_kwargs": {"b1": 0.9, "b2": 0.999, "eps": 1e-8},
    "gamma_F": 1.0,
    "grad_clip": 10.0,
}

DEFAULT_FINETUNE_CONFIG = {
    "batch_size": 32,
    "init_lr": 0.001,
    "num_epochs": 100,
    "decay_rate": 0.95,
    "optimizer": "adam+decay",
    "cache": 100,
    "lr_schedule": "exponential",
    "weight_decay": 0.0,
    "optimizer_kwargs": {"b1": 0.9, "b2": 0.999, "eps": 1e-8},
    "gamma_F": 1.0,
    "grad_clip": 10.0,
}


DEFAULT_RE_CONFIG = {
    # Simulation parameters
    "kT": 300.0 * quantity.kb,   # Temperature in energy units
    "T": 300.0,                  # Temperature in K
    "gamma": 100.0,              # Friction coefficient in 1/ps
    "dt": 0.002,                 # Timestep in ps (2 fs)
    "t_total": 100.0,             # Total simulation time per RE step in ps
    "t_eq": 1.0,                 # Equilibration time in ps
    "t_sample": 0.1,             # Sample every t_sample ps (for re_timings)
    "n_chains": 10,               # Number of parallel simulation chains (vmap_batch)
    # RE training parameters
    "num_epochs": 10,
    "init_lr": 1e-4,
    "lr_decay_rate": 0.95,
    "reweight_ratio": 0.9,       # Fraction of simulation samples used for reweighting
    "sim_batch_size": 1,         # Number of statepoints processed per RE step
    "PRNGKey_seed": SEED,
}

DEFAULT_SIM_CONFIG = {
    "gamma": 100.0,  # Friction coefficient in 1/ps (for NVT Langevin)
    "dt_values_fs": [2],  # Add more dt values as needed
    "print_every": 0.5,  # Save frame every 0.5 ps
    "sim_mode": "sampling",  # simulation mode: 'sampling', 'stability', 'helix', 'speed'
    "ensemble": "NVT",  # NVT or NVE
    # "t_eq": 0,  # Equlibration time in ps
    "t_total": 100,  # Total simulation time in ps (- t_eq)
    "n_chains": 3,  # Number of simulations (parallel)
    "T": 300.0,
    "PRNGKey_seed": SEED,
}