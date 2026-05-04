"""
Training diagnostics plotting (predictions, convergence, distances).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler

def plot_predictions(
    predictions: dict, reference_data: dict, out_dir: str, name: str
) -> None:
    """
    Plot force predictions vs reference data with scatter plot and compute MAE.

    Parameters
    ----------
    predictions : dict
        Dictionary containing predicted values with 'F' key for forces
    reference_data : dict
        Dictionary containing reference values with 'F' key for forces
    out_dir : str
        Output directory to save the figure
    name : str
        Name for the output file
    """
    # Simplifies comparison: convert units
    scale_energy = 96.485  # [eV] -> [kJ/mol]
    scale_pos = 0.1  # [Å] -> [nm]

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), layout="constrained")
    fig.suptitle("Predictions")

    # Reshape forces and scale units
    pred_F = predictions["F"].reshape(-1, 3) / scale_energy * scale_pos
    ref_F = reference_data["F"].reshape(-1, 3) / scale_energy * scale_pos

    # Ensure pred_F has same number of entries as ref_F by dropping extra entries
    if len(pred_F) > len(ref_F):
        pred_F = pred_F[: len(ref_F)]
    elif len(ref_F) > len(pred_F):
        ref_F = ref_F[: len(pred_F)]

    # Verify shapes match
    assert (
        pred_F.shape == ref_F.shape
    ), f"Shape mismatch: pred_F {pred_F.shape}, ref_F {ref_F.shape}"

    # Compute MAE
    mae = np.mean(np.abs(pred_F - ref_F))
    ax.set_title(f"Force (MAE: {mae * 1000:.1f} meV/A)")

    # 45-degree reference line
    ax.axline((0, 0), slope=1, color="black", linestyle=(0, (3, 5, 1, 5)), linewidth=1)

    # Scatter plot
    ax.set_prop_cycle(cycler(color=plt.get_cmap("tab20c").colors))
    ax.scatter(ref_F.ravel(), pred_F.ravel(), s=5, edgecolors="none", alpha=0.2)

    ax.set_xlabel("Ref. F [eV/A]")
    ax.set_ylabel("Pred. F [eV/A]")
    ax.legend().remove()  # no legend needed

    # Save figure
    fig.savefig(f"{out_dir}/{name}.png", bbox_inches="tight", dpi=1200)


def plot_convergence(trainer, out_dir: str) -> None:
    """
    Plot training and validation loss convergence.

    Parameters
    ----------
    trainer : object
        Trainer object with train_losses and val_losses attributes
    out_dir : str
        Output directory to save the figure
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(f"{out_dir}/convergence.pdf", bbox_inches="tight")


def plot_atom_distance(
    ax: Axes,
    distances: np.ndarray | list[np.ndarray],
    labels: list[str] | None = None,
    bins: int = 60,
    xlabel: str = "Distance",
    ylabel: str = "Frequency",
) -> Axes:
    """
    Plot histogram of atom distances.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    distances : np.ndarray | list[np.ndarray]
        Distance data - single array or list of arrays for multiple models
    labels : list[str] | None, optional
        List of labels for each set of distances
    bins : int, optional
        Number of bins for the histogram
    xlabel : str, optional
        Label for the x-axis
    ylabel : str, optional
        Label for the y-axis

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = ["#368274", "#0C7CBA", "#C92D39", "k"]
    line = ["-", "-", "-", "--"]

    if isinstance(distances, (list, tuple)) and hasattr(distances[0], "__len__"):
        n_models = len(distances)
        for i in range(n_models):
            ax.hist(
                distances[i],
                bins=bins,
                alpha=0.6,
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                histtype="step",
                linewidth=2.0,
                linestyle=line[i % len(line)],
            )
    else:
        ax.hist(
            distances,
            bins=bins,
            alpha=0.6,
            color=color[0],
            histtype="step",
            linewidth=2.0,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def compare_atom_distances(
    AT_distances: list[np.ndarray],
    Traj_distances: list[np.ndarray],
    dist_labels: list[str],
    outpath: str,
    name: str,
    at_label: str = "Reference",
    traj_label: str = "Simulation",
    bins: int = 60,
    at_color: str = "#368274",
    traj_color: str = "#C92D39",
    xlabel: str = "Distance",
    ylabel: str = "Normalized frequency",
) -> str:
    """
    Plot reference vs simulation atom-distance histograms side by side.

    Parameters
    ----------
    AT_distances : list[np.ndarray]
        List of 1D arrays of reference distances
    Traj_distances : list[np.ndarray]
        List of 1D arrays of simulation distances
    dist_labels : list[str]
        List of titles for each subplot
    outpath : str
        Directory to save the figure in
    name : str
        Basename for the output file
    at_label : str, optional
        Legend label for reference data
    traj_label : str, optional
        Legend label for simulation data
    bins : int, optional
        Number of bins
    at_color : str, optional
        Color for reference histograms
    traj_color : str, optional
        Color for simulation histograms
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label

    Returns
    -------
    str
        Full path to the saved figure file
    """
    n = len(dist_labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=True)

    for i, title in enumerate(dist_labels):
        ax = axes[i] if n > 1 else axes
        # AT
        ax.hist(
            AT_distances[i],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            linestyle="-",
            color=at_color,
            label=at_label,
        )
        # Simulation
        ax.hist(
            Traj_distances[i],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            linestyle="-",
            color=traj_color,
            label=traj_label,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

    plt.tight_layout()
    fname = f"{outpath}/Atom_distances_{name}_vs_Reference.png"
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    return fname


def plot_bond_potentials(
    extract_fn,
    full_vars,
    dataset,
    nbrs,
    bond_index,
    output_dir,
    species=None,
    n_sample=20,
):
    """Plot fitted harmonic bond potentials for a MACEBond model.

    Since bond parameters come from a species-pair lookup table (same k and r0
    for every bond of the same type), bonds are grouped by their species pair
    and one curve is drawn per unique pair.

    Saves to ``{output_dir}/bond_potentials.png``.

    Args:
        extract_fn: Callable returning ``{"bond_k": [B], "bond_r0": [B]}``.
        full_vars:  Flax variable dict.
        dataset:    Dict with keys ``"R"`` and ``"mask"``.
        nbrs:       Initialised neighbour list.
        bond_index: ``[2, B]`` integer array of bonded pairs.
        output_dir: Directory to save the figure.
        species:    Global species array (single-molecule); ignored for CATH
                    (per-frame species read from dataset).
        n_sample:   Number of frames to sample (used to confirm consistency).
    """
    import math
    from jax import numpy as jnp, tree_util

    R     = np.asarray(dataset["R"])
    masks = np.asarray(dataset["mask"])
    n_total  = R.shape[0]
    n_sample = min(n_sample, n_total)
    sample_idx = np.linspace(0, n_total - 1, n_sample, dtype=int)

    bond_index = np.asarray(bond_index)   # [2, B]
    n_bonds = bond_index.shape[1]

    per_frame_species = dataset.get("species", None)
    per_frame_subset  = dataset.get("subset", None)

    # Resolve species for the first sampled frame — used to label bond types.
    if per_frame_species is not None:
        frame_species = np.asarray(per_frame_species[sample_idx[0]])
    elif species is not None:
        frame_species = np.asarray(species)
    else:
        frame_species = None

    # Group bond indices by canonical species pair (s_lo, s_hi).
    pair_to_bonds: dict = {}
    for b in range(n_bonds):
        if frame_species is not None:
            s_i = int(frame_species[bond_index[0, b]])
            s_j = int(frame_species[bond_index[1, b]])
            pair = (min(s_i, s_j), max(s_i, s_j))
        else:
            pair = (int(bond_index[0, b]), int(bond_index[1, b]))
        pair_to_bonds.setdefault(pair, []).append(b)

    # Extract params from sampled frames.
    all_bond_k, all_bond_r0 = [], []
    print(f"Extracting bond parameters from {n_sample} validation frames …")
    for i in sample_idx:
        r = jnp.asarray(R[i])
        m = jnp.asarray(masks[i])
        nbrs_i = nbrs.update(r, mask=m)
        extra = {}
        if per_frame_species is not None:
            extra["species"] = jnp.asarray(per_frame_species[i])
        elif species is not None:
            extra["species"] = jnp.asarray(species)
        if per_frame_subset is not None:
            extra["subset"] = int(per_frame_subset[i])
        params = extract_fn(full_vars, r, nbrs_i, mask=m, **extra)
        params = tree_util.tree_map(np.asarray, params)
        all_bond_k.append(params["bond_k"])
        all_bond_r0.append(params["bond_r0"])

    bond_k  = np.stack(all_bond_k)   # [n_sample, B]
    bond_r0 = np.stack(all_bond_r0)  # [n_sample, B]

    # Per-bond mean across frames (should be constant for table-based model).
    k_mean  = bond_k.mean(axis=0)   # [B]
    r0_mean = bond_r0.mean(axis=0)  # [B]

    # Aggregate to unique species pairs.
    pair_k:  dict = {}
    pair_r0: dict = {}
    for pair, bond_ids in pair_to_bonds.items():
        pair_k[pair]  = float(k_mean[bond_ids].mean())
        pair_r0[pair] = float(r0_mean[bond_ids].mean())

    r_vals = list(pair_r0.values())
    r_lo = max(0.01, min(r_vals) - 0.35)
    r_hi = max(r_vals) + 0.35
    r_arr = np.linspace(r_lo, r_hi, 300)

    n_pairs = len(pair_to_bonds)
    fig, ax = plt.subplots(figsize=(6, 4.8), layout="constrained")
    fig.suptitle(
        f"MACEBond — Fitted Bond Potentials  ({n_pairs} unique species pairs)",
        fontsize=11,
    )
    ax.set_title("U(r) = ½ k (r − r₀)²")
    ax.set_xlabel("r  (nm)")
    ax.set_ylabel("U  (kJ mol⁻¹)")

    cmap = plt.cm.tab10
    for color_idx, (pair, bond_ids) in enumerate(sorted(pair_to_bonds.items())):
        s_i, s_j = pair
        k  = pair_k[pair]
        r0 = pair_r0[pair]
        color = cmap(color_idx % 10)
        U = 0.5 * k * (r_arr - r0) ** 2
        n_bonds_of_type = len(bond_ids)
        label = (
            f"Species {s_i}–{s_j}  "
            f"k={k:.1f}  r₀={r0:.3f} nm"
            + (f"  (×{n_bonds_of_type})" if n_bonds_of_type > 1 else "")
        )
        ax.plot(r_arr, U, color=color, label=label)
        ax.axvline(r0, color=color, linestyle=":", linewidth=0.8, alpha=0.6)

    u_max = max(
        0.5 * pair_k[p] * max(abs(r_lo - pair_r0[p]), abs(r_hi - pair_r0[p])) ** 2
        for p in pair_to_bonds
    ) if pair_to_bonds else 1.0
    if not math.isfinite(u_max) or u_max <= 0:
        u_max = 1.0
    ax.set_ylim(0, min(u_max * 1.1, u_max * 4))
    ax.legend(fontsize=7, frameon=False, loc="upper center")

    fname = f"{output_dir}/bond_potentials.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Bond potential plot saved to {fname}")
    return fname
