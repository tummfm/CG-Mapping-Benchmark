"""Export a trained force-matching model to a LAMMPS-compatible .ptb file.

Usage:
    python export_lammps_model.py --model <PATH_TO_PARAMS_PKL>

The exported .ptb file is saved next to the .pkl file.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

parser = argparse.ArgumentParser(
    description="Export trained model to LAMMPS .ptb format."
)
parser.add_argument("--model", type=str, required=True, help="Path to params .pkl file")
parser.add_argument("--device", type=str, default=None, help="GPU or MIG UUID")
args = parser.parse_args()

from utils import (
    build_mace_config,
    configure_runtime_environment,
    init_mace_model_and_template,
    init_nequip_model,
    load_model_artifacts,
    load_simulation_dataset,
    load_training_dataset
    )

configure_runtime_environment(device=args.device, xla_mem_fraction=0.5)

import cloudpickle as pickle
from jax import numpy as jnp, tree_util
from jax_md import partition, space
from chemtrain.data import preprocessing
from chemtrain.deploy import exporter as deploy_exporter, graphs as export_graphs

base_dir, MODEL_CONFIG, _ = load_model_artifacts(args.model)
model_type = MODEL_CONFIG.get("model", "mace")
r_cut_nm = MODEL_CONFIG["r_cutoff"]

_, data, _ = load_training_dataset(
    mol=MODEL_CONFIG["mol"],
    train_ratio=MODEL_CONFIG["train_ratio"],
    val_ratio=MODEL_CONFIG["val_ratio"],
    cg_map=MODEL_CONFIG["CG_map"],
)

if MODEL_CONFIG["CG_map"] == "AT":
    data.load_traj()
    dataset_split = data.dataset_U["training"]
else:
    data.coarse_grain(MODEL_CONFIG["CG_map"])
    dataset_split = data.cg_dataset_X["training"]

# LAMMPS ghost atoms already represent periodic images, so free-space
# displacement is correct for inference.
displacement_free, _ = space.free()

_, (_, max_edges, avg_num_neighbors) = preprocessing.allocate_neighborlist(
    dataset_split,
    displacement_free,
    r_cutoff=r_cut_nm,
    mask_key="mask",
    box=None,
    format=partition.Sparse,
    batch_size=100,
    capacity_multiplier=1.0,
)

species_init = jnp.asarray(dataset_split["species"][0])

with open(args.model, "rb") as f:
    params = pickle.load(f)
params = tree_util.tree_map(jnp.asarray, params)

if model_type == "mace":
    from chemtrain.compose import mace_jax as mace_jax_compose

    use_so3 = MODEL_CONFIG.get("use_so3", False)
    mace_cfg = build_mace_config(MODEL_CONFIG, use_so3=use_so3)

    _init_params, _model_energy_fn_template, model_config = init_mace_model_and_template(
        displacement_free,
        MODEL_CONFIG["r_cutoff"],
        None,
        species_init,
        avg_num_neighbors,
        mace_cfg=mace_cfg,
        n_species=100, # hardcoded n_species
        per_particle=False,
        use_so3=MODEL_CONFIG.get("use_so3", False),
        enable_cueq=None,
    )

    num_interactions = int(model_config.get(
        "num_interactions", MODEL_CONFIG.get("num_interactions", 2)
    ))
    atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
    atomic_energies = jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)

    _nbr_order = [num_interactions, 2 * num_interactions]
    _r_cutoff_ang = r_cut_nm * 10.0

    gnn_fn = _model_energy_fn_template(params)

    class LammpsExporter(deploy_exporter.Exporter):
        graph_type = export_graphs.SimpleSparseNeighborList
        unit_style = "real"
        nbr_order = _nbr_order
        r_cutoff = _r_cutoff_ang

        def energy_fn(self, position, species, graph):
            neighbor = graph.to_neighborlist()
            species = species + 1  # Shift to 1-based indexing for LAMMPS

            pots = gnn_fn(position / 10.0, neighbor, species=species)
            mapped = jnp.argmax(species[:, None] == atomic_numbers[None, :], axis=-1)
            pots -= atomic_energies[species]
            
            return pots / 4.184

else:
    raise ValueError(f"Unknown model type: {model_type!r}. Expected 'mace' or 'nequip'.")

model_exporter = LammpsExporter()
model_exporter.export()

output_path = os.path.splitext(args.model)[0] + ".ptb"
model_exporter.save(output_path)
                                                                                                                                                                                         
print(f"Saved exported model to {output_path}")
