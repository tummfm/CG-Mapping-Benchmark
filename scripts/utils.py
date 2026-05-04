"""Shared utilities for unified force-matching training and simulation scripts."""

from __future__ import annotations

import json
import os
from typing import Any
from jax import tree_util
from collections import OrderedDict

MOL_ALIASES = {
    "BenzeneCrystal": "benzene_crystal",
    "CATH": "cath_full",
    "cath": "cath_full",
    "ThreeBPA_biased": "3bpa_biased",
    "Azobenzene_biased": "azobenzene_biased",
}


def configure_runtime_environment(
    device: str | None = "",
    xla_mem_fraction: float = 0.96,
) -> None:
    """Configure CUDA device visibility and XLA memory behavior before JAX import."""
    device_str = "" if device is None else str(device)
    print(
        "[DEVICE] Running on device(s):",
        device_str if device_str else "[DEVICE] No device specified, running on CPU",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(xla_mem_fraction)


def normalize_molecule_name(mol: str) -> str:
    return MOL_ALIASES.get(mol, mol)


def load_json_file(path: str, label: str = "JSON") -> dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file {path} not found.")
    with open(path, "r") as f:
        return json.load(f)


def load_model_artifacts(model_path: str) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Load config.json and (optionally) train_config.json from a trained model directory."""
    base_dir = os.path.dirname(os.path.abspath(model_path))
    model_config = load_json_file(os.path.join(base_dir, "config.json"), "Config")
    train_config_path = os.path.join(base_dir, "train_config.json")
    train_config = load_json_file(train_config_path, "Train config") if os.path.exists(train_config_path) else {}
    return base_dir, model_config, train_config


def load_training_dataset(
    mol: str,
    train_ratio: float,
    val_ratio: float,
    cg_map: str,
    stride: int | None = None,
    verbose: bool = False,
) -> tuple[str, Any, bool]:
    """Load dataset object for training; returns (normalized_mol, data_obj, used_stride_cache)."""
    from cgbench.core import dataset

    mol_normalized = normalize_molecule_name(mol)

    basic_loaders = {
        "capped_ala": dataset.Capped_Ala_Dataset,
        "capped_ala2": dataset.Capped_Ala2_Dataset,
        "capped_ala15": dataset.Capped_Ala15_Dataset,
        "hexane": dataset.Hexane_Dataset,
        "benzene_crystal": dataset.BenzeneCrystal_Dataset,
        "capped_pro": dataset.Capped_Pro_Dataset,
        "capped_thr": dataset.Capped_Thr_Dataset,
        "capped_gly": dataset.Capped_Gly_Dataset,
        "spice_dipeptides": dataset.SPICE_Dipeptides,
        "tip3p": dataset.TIP3P_water_Dataset,
        "3bpa": dataset.ThreeBPA_Dataset,
        "3bpa_biased": dataset.ThreeBPA_Biased_Dataset,
        "azobenzene_biased": dataset.Azobenzene_Biased_Dataset,
    }

    if mol_normalized in basic_loaders:
        data_obj = basic_loaders[mol_normalized](
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return mol_normalized, data_obj, False

    if mol_normalized in ("cath_full", "cath_quarter", "cath_test"):
        mol_size = mol_normalized.split("_", 1)[1]
        cached_path = f"/ds/project/franz/Datasets/CATH_{mol_size}_{cg_map}.npz"
        used_stride_cache = False
        if stride == 10:
            stride_cached_path = (
                f"/ds/project/franz/Datasets/CATH_{mol_size}_{cg_map}_stride=10.npz"
            )
            if os.path.exists(stride_cached_path):
                cached_path = stride_cached_path
                used_stride_cache = True
                if verbose:
                    print(f"Using pre-strided cache: {cached_path}")

        data_obj = dataset.CATH_Dataset(
            dataset_key=mol_normalized,
            cg_strategy=cg_map,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            cached_dataset_path=cached_path,
        )
        return mol_normalized, data_obj, used_stride_cache

    raise ValueError(
        "Invalid molecule. Use 'capped_ala', 'capped_ala2', 'capped_ala15', "
        "'hexane', 'benzene_crystal', 'capped_pro', 'capped_thr', 'capped_gly', "
        "'spice_dipeptides', 'tip3p', '3bpa', '3bpa_biased', 'ThreeBPA_biased', "
        "'azobenzene_biased', 'Azobenzene_biased', "
        "'cath_full', 'cath_quarter', 'cath_test', or 'CATH'."
    )


def _infer_nmol_from_map(data_obj: Any, default: int) -> int:
    map_obj = getattr(data_obj, "map_obj", None)
    if map_obj is not None:
        if hasattr(map_obj, "n_mols"):
            return int(getattr(map_obj, "n_mols"))
        if hasattr(map_obj, "n_replicas"):
            return int(getattr(map_obj, "n_replicas"))
    return int(default)


def load_simulation_dataset(
    mol: str,
    train_ratio: float,
    val_ratio: float,
    cg_map: str | None = None,
    verbose: bool = False,
) -> tuple[str, Any, int]:
    """Load dataset object for simulation; returns (normalized_mol, data_obj, nmol)."""
    from cgbench.core import dataset

    mol = normalize_molecule_name(mol)

    basic_loaders = {
        "capped_ala": (dataset.Capped_Ala_Dataset, 1),
        "capped_ala2": (dataset.Capped_Ala2_Dataset, 1),
        "hexane": (dataset.Hexane_Dataset, 100),
        "capped_ala15": (dataset.Capped_Ala15_Dataset, 1),
        "capped_pro": (dataset.Capped_Pro_Dataset, 1),
        "capped_thr": (dataset.Capped_Thr_Dataset, 1),
        "capped_gly": (dataset.Capped_Gly_Dataset, 1),
        "3bpa": (dataset.ThreeBPA_Dataset, 1),
        "3bpa_biased": (dataset.ThreeBPA_Biased_Dataset, 1),
        "azobenzene_biased": (dataset.Azobenzene_Biased_Dataset, 1),
    }

    if mol in basic_loaders:
        cls, nmol = basic_loaders[mol]
        data_obj = cls(train_ratio=train_ratio, val_ratio=val_ratio)
        return mol, data_obj, nmol

    if mol == "benzene_crystal":
        data_obj = dataset.BenzeneCrystal_Dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return mol, data_obj, _infer_nmol_from_map(data_obj, default=288)

    if mol in ("tip3p", "tip3p-water"):
        data_obj = dataset.TIP3P_water_Dataset(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        return mol, data_obj, _infer_nmol_from_map(data_obj, default=901)

    if mol in ("cath_full", "cath_quarter", "cath_test"):
        map_name = cg_map or "coreBetaMap2"
        cache_candidates = [
            f"/ds/project/franz/Datasets/CATH_{mol.split('_', 1)[1]}_{map_name}.npz"
        ]
        cached_path = next((p for p in cache_candidates if os.path.exists(p)), None)
        if verbose and cached_path is not None:
            print(f"Using CATH cached dataset: {cached_path}")

        data_obj = dataset.CATH_Dataset(
            dataset_key=mol,
            cg_strategy=map_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            cached_dataset_path=cached_path,
        )
        return mol, data_obj, 1

    raise ValueError(
        "Invalid molecule. Use 'capped_ala', 'capped_ala2', 'capped_ala15', "
        "'hexane', 'benzene_crystal', 'tip3p', 'tip3p-water', 'capped_pro', "
        "'capped_thr', 'capped_gly', '3bpa', '3bpa_biased', 'ThreeBPA_biased', "
        "'azobenzene_biased', 'Azobenzene_biased', 'cath_full', 'cath_quarter', "
        "or 'cath_test'."
    )

def drop_energy_targets(dataset_dict: dict[str, dict[str, Any]]) -> None:
    """Drop supervised energies for force-only force-matching runs."""
    for split in ("training", "validation", "testing"):
        if split in dataset_dict and "U" in dataset_dict[split]:
            del dataset_dict[split]["U"]


def apply_stride(dataset_dict: dict[str, dict[str, Any]], stride: int) -> None:
    for split in dataset_dict:
        for key in list(dataset_dict[split].keys()):
            dataset_dict[split][key] = dataset_dict[split][key][::stride]


def allocate_neighborlist(
    dataset_split: dict[str, Any],
    displacement_fn: Any,
    box: Any,
    r_cutoff: float,
    *,
    format_name: str = "sparse",
    batch_size: int = 100,
    capacity_multiplier: float | None = None,
) -> tuple[Any, tuple[int, int, float]]:
    """Allocate neighbor list with consistent defaults across scripts."""
    from chemtrain.data import preprocessing
    from jax_md import partition

    fmt = partition.Sparse if str(format_name).lower() == "sparse" else partition.Dense

    kwargs: dict[str, Any] = {
        "r_cutoff": r_cutoff,
        "mask_key": "mask",
        "box_key": "box" if box is not None else None,
        "format": fmt,
        "batch_size": batch_size,
    }
    if capacity_multiplier is not None:
        kwargs["capacity_multiplier"] = float(capacity_multiplier)

    return preprocessing.allocate_neighborlist(
        dataset_split,
        displacement_fn,
        box,
        **kwargs,
    )


def build_mace_config(source: dict[str, Any], use_so3: bool) -> dict[str, Any]:
    """Build the compact mace-jax configuration dict from run config."""
    from cgbench.core.config import DEFAULT_MACE_CONFIG

    return {
        "r_cutoff": source["r_cutoff"],
        "hidden_irreps": source.get(
            "hidden_irreps", DEFAULT_MACE_CONFIG["hidden_irreps"]
        ),
        "MLP_irreps": source.get("readout_mlp_irreps", "16x0e"),
        "num_interactions": source.get(
            "num_interactions", DEFAULT_MACE_CONFIG["num_interactions"]
        ),
        "max_ell": source.get("max_ell", DEFAULT_MACE_CONFIG["max_ell"]),
        "correlation": source.get("correlation", DEFAULT_MACE_CONFIG["correlation"]),
        "n_radial_basis": source.get(
            "n_radial_basis", DEFAULT_MACE_CONFIG["n_radial_basis"]
        ),
        "output_irreps": source.get("output_irreps", "1x0e"),
        "use_so3": bool(use_so3),
    }


def init_mace_model_and_template(
    displacement_fn: Any,
    r_cutoff: float,
    box: Any,
    species_init: Any,
    avg_num_neighbors: float,
    mace_cfg: dict[str, Any],
    *,
    n_species: int = 100,
    per_particle: bool = False,
    use_so3: bool = False,
    enable_cueq: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Initialize MACE model and return (init_params, energy_fn_template, model_config)."""
    from jax import numpy as jnp
    from chemtrain.compose import mace_jax as mace_jax_compose
    from mace_jax.modules.wrapper_ops import CuEquivarianceConfig

    cueq_config = None
    if enable_cueq and not use_so3:
        cueq_config = CuEquivarianceConfig(
            enabled=True,
            layout=("mul_ir"),
            group=("O3"),
            optimize_all=True,
            conv_fusion=True,
        )

    template_vars, gnn_energy_fn, model_config = mace_jax_compose.mace_jax_neighborlist(
        displacement=displacement_fn,
        r_cutoff=r_cutoff,
        n_species=n_species,
        per_particle=per_particle,
        avg_num_neighbors=avg_num_neighbors,
        mode="energy",
        mace_config=mace_cfg,
        cueq_config=cueq_config,
    )

    init_params = template_vars["params"]
    variables = template_vars

    def energy_fn_template(params):
        vars = {**variables}
        vars["params"] = params

        def energy_fn(pos, neighbor, species=species_init, **dynamic_kwargs):
            dynamic_kwargs.setdefault("box", box)
            pots = gnn_energy_fn(vars, pos, neighbor, species=species, **dynamic_kwargs)

            # Subtract the provided atomic energies
            atomic_numbers = jnp.asarray(model_config["atomic_numbers"], dtype=jnp.int32)
            mapped_species = jnp.argmax(
                species[:, None] == atomic_numbers[None, :], axis=-1
            )
            pots -= jnp.asarray(model_config["atomic_energies"], dtype=jnp.float32)[
                mapped_species
            ] * dynamic_kwargs.get("mask", 1.0)

            return jnp.sum(pots)

        return energy_fn

    return init_params, energy_fn_template, model_config




def init_nequip_model(
    displacement_fn: Any,
    r_cutoff: float,
    n_species: int,
    max_edges: int,
    avg_num_neighbors: float,
) -> tuple[Any, Any]:
    """NequIP initialization retained from legacy scripts for compatibility experiments."""
    from external.models import nequip

    return nequip.nequip_neighborlist_pp(
        displacement_fn,
        r_cutoff,
        n_species,
        max_edges=max_edges,
        per_particle=False,
        avg_num_neighbors=avg_num_neighbors,
        mode="energy",
        positive_species=True,
    )




class SWAManager:
    """Collect epoch snapshots and maintain a running SWA average."""

    def __init__(self, enabled, start_epoch, every, min_snapshots, prefer):
        self.enabled = enabled
        self.start_epoch = int(start_epoch)
        self.every = max(1, int(every))
        self.min_snapshots = max(1, int(min_snapshots))
        self.prefer = prefer
        self.snapshot_count = 0
        self.swa_params = None
        self.active = False

    def _should_collect(self, epoch):
        if not self.enabled:
            return False
        if epoch < self.start_epoch:
            return False
        return ((epoch - self.start_epoch) % self.every) == 0

    def _average(self, params):
        if self.swa_params is None:
            self.swa_params = tree_util.tree_map(lambda x: x, params)
            self.snapshot_count = 1
            return

        self.snapshot_count += 1
        count = float(self.snapshot_count)
        self.swa_params = tree_util.tree_map(
            lambda avg, new: avg + (new - avg) / count,
            self.swa_params,
            params,
        )

    def post_epoch(self, trainer, *_, **__):
        epoch = int(trainer._epoch)
        if not self._should_collect(epoch):
            return
        self._average(trainer.params)
        print(
            f"[SWA] Collected snapshot {self.snapshot_count} at epoch {epoch} "
            f"(start={self.start_epoch}, every={self.every})"
        )

    def post_training(self, *_args, **_kwargs):
        self.active = self.enabled and (self.snapshot_count >= self.min_snapshots)
        if self.enabled:
            print(
                f"[SWA] Training finished with {self.snapshot_count} snapshots. "
                f"Active={self.active} (min_snapshots={self.min_snapshots}, prefer={self.prefer})"
            )


def get_train_config(train_defaults, swa_overrides=None):
    swa_overrides = swa_overrides or {}
    return OrderedDict(
        optimizer=OrderedDict(
            init_lr=train_defaults["init_lr"],
            lr_decay=train_defaults["decay_rate"],
            epochs=train_defaults["num_epochs"],
            batch=train_defaults["batch_size"],
            cache=100,
            power="exponential",
            weight_decay=0.0,
            type="ADAM",
            optimizer_kwargs=OrderedDict(b1=0.9, b2=0.999, eps=1e-8),
        ),
        gammas=OrderedDict(F=1.0),
        swa=OrderedDict(
            enabled=bool(swa_overrides.get("enabled", False)),
            start_epoch=swa_overrides.get("start_epoch"),
            every=max(1, int(swa_overrides.get("every", 1))),
            min_snapshots=max(1, int(swa_overrides.get("min_snapshots", 2))),
            prefer=bool(swa_overrides.get("prefer", True)),
        ),
    )