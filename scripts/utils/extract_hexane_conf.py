import os
import sys

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import jax.numpy as jnp
from jax import random
from cgbench.core.config import SEED

def write_gro_for_chain(positions, mapping, n_mol, box_length,
                        out_path, title='cg hexane', filename='cg_hexane.gro'):
    """
    positions: ndarray (n_atoms, 3) for one chain
    mapping: 'two-site' | 'three-site' | 'four-site'
    n_mol: number of molecules per chain
    box_length: cubic box size (float)
    """
    beads_per_mol = positions.shape[0] // n_mol
    mapping_names = {
        'two-site': ['A1', 'A2'],
        'three-site': ['A1', 'B', 'A2'],
        'four-site': ['A1', 'B1', 'B2', 'A2'],
    }
    if mapping not in mapping_names:
        raise ValueError(f"Unknown mapping '{mapping}'. Choose from {list(mapping_names.keys())}")

    atom_names = mapping_names[mapping]

    if len(atom_names) != beads_per_mol:
        raise ValueError(
            f"Mapping '{mapping}' expects {len(atom_names)} beads per mol, "
            f"but positions indicate {beads_per_mol} beads per mol."
        )

    coords = np.asarray(positions)

    n_atoms_total = n_mol * beads_per_mol
    out_file = os.path.join(out_path, filename)

    # Use a fixed residue name
    resname = "HEX"   # 3 characters = safe

    with open(out_file, 'w') as fh:
        fh.write(f"{title}\n")
        fh.write(f"{n_atoms_total:5d}\n")

        atom_idx = 1
        for mol_idx in range(n_mol):
            resid = mol_idx + 1
            for bead_idx in range(beads_per_mol):
                x, y, z = coords[mol_idx * beads_per_mol + bead_idx]
                aname = atom_names[bead_idx]
                vx = vy = vz = 0.0

                # Standard .gro formatting
                line = (f"{resid:5d}{resname:>5s}{aname:>5s}"
                        f"{atom_idx:5d}"
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"{vx:8.4f}{vy:8.4f}{vz:8.4f}\n")
                fh.write(line)
                atom_idx += 1

        # Write cubic simulation box
        fh.write(f"{box_length:10.5f}{box_length:10.5f}{box_length:10.5f}\n")


def write_chains_from_rinit(r_init, mapping, n_mol, box_length,
                            target_dir, prefix='chain', gro_name='cg_hexane.gro'):

    r_init = np.asarray(r_init)
    n_chains = r_init.shape[0]
    os.makedirs(target_dir, exist_ok=True)

    for i in range(n_chains):
        chain_dir = os.path.join(target_dir, f"{prefix}_{i}")
        os.makedirs(chain_dir, exist_ok=True)
        write_gro_for_chain(
            r_init[i], mapping, n_mol, box_length,
            chain_dir, filename=gro_name
        )
        print(f"Wrote {os.path.join(chain_dir, gro_name)}")


## CHANGE THESE TO ACTUAL PATHS 
mapping = 'three-site'              
target_dir = f'Classical_potential/model/map={mapping}'  
path = f"../../data/reference_simulations/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200_CG={mapping}.npz"

n_mol = 100
n_chains = 10

data = np.load(path, allow_pickle=True)
data = data['arr_0'].item()

key = random.PRNGKey(SEED)
key, split = random.split(key)
selection = random.choice(
    split,
    jnp.arange(data["validation"]["R"].shape[0]),
    shape=(n_chains,),
    replace=False,
)
r_init = data["validation"]["R"][selection]
box_length = float(data["validation"]["box"][0][0, 0])
    
write_chains_from_rinit(r_init, mapping, n_mol, box_length, target_dir)