import argparse
import os
import sys

# Add parent directory to path to import cgbench
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="GPU or MIG UUID")
parser.add_argument("--cgmap", type=str, help="CG mapping to use", required=True)
parser.add_argument("--mol", type=str, help="Molecule to use", required=True)
parser.add_argument(
    "--rcut", type=float, help="Cutoff radius for neighbor list", default=0.5
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output", default=False
)
args = parser.parse_args()

if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import numpy as onp
import optax
from chemutils.models import allegro, nequip
from chemtrain import trainers
from chemtrain.data import preprocessing
from jax_md import partition, space
from chemtrain.trainers import trainers
import json
from jax import numpy as jnp, random, tree_util
from cgbench.core import dataset
from cgbench.core.config import DEFAULT_NEQUIP_CONFIG, DEFAULT_TRAIN_CONFIG as TRAIN_CONFIG

NEQUIP_CONFIG = DEFAULT_NEQUIP_CONFIG.copy()
NEQUIP_CONFIG.update({
    "r_cutoff": args.rcut,
    "mol": args.mol, 
    "CG_map": args.cgmap,
})
NEQUIP_CONFIG["type"] = "CG" if NEQUIP_CONFIG["CG_map"] != "AT" else "AT"

# -------------------------
# Load dataset
# -------------------------
if NEQUIP_CONFIG["mol"] == "ala2":
    data = dataset.Ala2_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"], val_ratio=NEQUIP_CONFIG["val_ratio"]
    )
elif NEQUIP_CONFIG["mol"] == "ala15":
    data = dataset.Ala15_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"], val_ratio=NEQUIP_CONFIG["val_ratio"]
    )
elif NEQUIP_CONFIG["mol"] == "hexane":
    data = dataset.Hexane_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"],
        val_ratio=NEQUIP_CONFIG["val_ratio"],
    )
elif NEQUIP_CONFIG["mol"] == "pro2":
    data = dataset.Pro2_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"], val_ratio=NEQUIP_CONFIG["val_ratio"]
    )
elif NEQUIP_CONFIG["mol"] == "thr2":
    data = dataset.Thr2_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"], val_ratio=NEQUIP_CONFIG["val_ratio"]
    )
elif NEQUIP_CONFIG["mol"] == "gly2":
    data = dataset.Gly2_Dataset(
        train_ratio=NEQUIP_CONFIG["train_ratio"], val_ratio=NEQUIP_CONFIG["val_ratio"]
    )
else:
    raise ValueError(
        "Invalid molecule. Use 'ala2', 'ala15', 'hexane', 'pro2', 'thr2', or 'gly2'."
    )
    
# AT
if NEQUIP_CONFIG["type"] == "AT":
    dataset = data.dataset_U
    species = data.species
    masses = data.masses
    n_species = data.n_species
    output_dir = f"outputs/MLP_train_Nequip/{NEQUIP_CONFIG['mol'].capitalize()}_map={NEQUIP_CONFIG['CG_map']}_tr={NEQUIP_CONFIG['train_ratio']}_epochs={TRAIN_CONFIG['num_epochs']}"
    if NEQUIP_CONFIG["mol"] == "hexane":
        output_dir += f"_nstxout={NEQUIP_CONFIG.get('nstxout', '')}/"
# CG
elif NEQUIP_CONFIG["type"] == "CG":
    data.coarse_grain(map=NEQUIP_CONFIG["CG_map"])
    dataset = data.cg_dataset_U
    species = data.cg_species
    masses = data.cg_masses
    n_species = data.n_cg_species
    output_dir = f"outputs/MLP_train_nequip/{NEQUIP_CONFIG['mol'].capitalize()}_map={NEQUIP_CONFIG['CG_map']}_tr={NEQUIP_CONFIG['train_ratio']}_epochs={TRAIN_CONFIG['num_epochs']}"
else:
    raise ValueError("Invalid simulation type. Use 'AT' or 'CG'.")
os.makedirs(output_dir, exist_ok=True)


# -------------------------
# Setup neighbor list and MACE model
# -------------------------
box = data.box
displacement_fn, _ = space.periodic_general(box=box, fractional_coordinates=True)

nbrs_init, (max_neighbors, max_edges, avg_num_neighbors) = (
    preprocessing.allocate_neighborlist(
        dataset["training"],
        displacement_fn,
        box,
        r_cutoff=NEQUIP_CONFIG["r_cutoff"],
        mask_key="mask",
        box_key="box",
        format=partition.Sparse,
        batch_size=100,
    )
)

if args.verbose:
    print(
        f"Max neighbors: {max_neighbors}, Max edges: {max_edges}, Avg neighbors: {avg_num_neighbors}"
    )

init_fn, gnn_energy_fn = nequip.nequip_neighborlist_pp(
    displacement_fn,
    NEQUIP_CONFIG["r_cutoff"],
    n_species,
    max_edges=max_edges,
    per_particle=False,
    avg_num_neighbors=avg_num_neighbors,
    mode="energy",
    positive_species=True,
)


def energy_fn_template(energy_params):
    def energy_fn(pos, neighbor, mode=None, **dynamic_kwargs):
        assert "species" in dynamic_kwargs.keys(), "species not in dynamic_kwargs"

        if "mask" not in dynamic_kwargs:
            print(f"Add defaul all-positive mask.")
            dynamic_kwargs["mask"] = jnp.ones(pos.shape[0], dtype=jnp.bool_)

        if "box" in dynamic_kwargs:
            print(f"Found box in energy kwargs")

        return gnn_energy_fn(energy_params, pos, neighbor, **dynamic_kwargs)

    return energy_fn


key = random.PRNGKey(NEQUIP_CONFIG["PRNGKey_seed"])
r_init = jnp.asarray(dataset["training"]["R"][0])
species_init = jnp.asarray(dataset["training"]["species"][0])
mask_init = jnp.asarray(dataset["training"]["mask"][0])
init_params = init_fn(key, r_init, nbrs_init, species=species_init, mask=mask_init)
nbrs_init = nbrs_init.update(r_init, mask=mask_init)

# -------------------------
# Setup optimizer
# -------------------------
batch_size = TRAIN_CONFIG["batch_size"]
num_samples = dataset["training"]["R"].shape[0]
epochs = TRAIN_CONFIG["num_epochs"]
total_steps = (epochs * num_samples) // batch_size
transition_steps = total_steps

scheduler = optax.exponential_decay(
    init_value=TRAIN_CONFIG["init_lr"],
    transition_steps=transition_steps,
    decay_rate=TRAIN_CONFIG["decay_rate"],
)

optimizer_fm = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0),
)
if args.verbose:
    print(f"Total steps: {total_steps}")
    print(f"Training on {num_samples} samples.")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")


# -------------------------
# Setup trainer
# -------------------------
trainer_fm = trainers.ForceMatching(
    init_params,
    optimizer_fm,
    energy_fn_template,
    nbrs_init,
    log_file=f"{output_dir}/force_matching.log",
    batch_per_device=int(batch_size),
)
trainer_fm.set_dataset(dataset["training"], stage="training")
trainer_fm.set_dataset(dataset["validation"], stage="validation", include_all=True)
if "testing" in dataset:
    trainer_fm.set_dataset(dataset["testing"], stage="testing", include_all=True)

# -------------------------
# Run training and save results
# -------------------------
# Train and save the results to a new folder
trainer_fm.train(epochs)
trainer_fm.save_trainer(f"{output_dir}/trainer.pkl", format=".pkl")
trainer_fm.save_energy_params(f"{output_dir}/best_params.pkl", ".pkl", best=True)
trainer_fm.save_energy_params(f"{output_dir}/final_params.pkl", ".pkl", best=False)

# Save configs as json
with open(f"{output_dir}/config.json", "w") as f:
    json.dump(NEQUIP_CONFIG, f, indent=4)
# Save training config as json
with open(f"{output_dir}/train_config.json", "w") as f:
    json.dump(TRAIN_CONFIG, f, indent=4)

from cgbench.utils.helpers import plot_predictions, plot_convergence

# Plot training convergence
plot_convergence(trainer_fm, output_dir)

predictions_val = trainer_fm.predict(
    dataset["validation"],
    trainer_fm.best_params,
    batch_size=batch_size,
)
predictions_val = tree_util.tree_map(onp.asarray, predictions_val)
plot_predictions(
    predictions_val, dataset["validation"], output_dir, name="preds_validation"
)

if "testing" in dataset:
    predictions_test = trainer_fm.predict(
        dataset["testing"],
        trainer_fm.best_params,
        batch_size=batch_size,
    )
    predictions_test = tree_util.tree_map(onp.asarray, predictions_test)
    onp.savez(f"{output_dir}/predictions_test.npz", **predictions_test)
    plot_predictions(
        predictions_test, dataset["testing"], output_dir, name="preds_testing"
    )
