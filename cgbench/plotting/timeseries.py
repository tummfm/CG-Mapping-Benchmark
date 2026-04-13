"""
Time series plotting (energy, temperature, coordinates over time).
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from typing import Union

from chemtrain import quantity


def add_chain_lines(ax: plt.Axes, line_locations: list[int]) -> None:
    """
    Draw vertical lines to indicate the start of each chain segment.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes object to annotate.
    line_locations : list[int]
        List of time-step indices where new chains begin.
    """
    for loc in line_locations:
        ax.axvline(x=loc, color="r", linestyle="-", alpha=0.5)


def overlay_chains(
    ax: plt.Axes,
    data: np.ndarray,
    line_locations: list[int],
    y_label: str,
    title: str,
    relative: bool = False,
) -> None:
    """
    Overlay multiple chain segments on a single plot.

    Splits a time-series into segments defined by line_locations, and plots
    each segment either in absolute or relative x-axis.

    Parameters
    ----------
    ax : plt.Axes
        Axes on which to draw the overlay.
    data : np.ndarray
        1D array of values to plot.
    line_locations : list[int]
        Indices delimiting chain boundaries.
    y_label : str
        Label for the Y-axis.
    title : str
        Plot title.
    relative : bool, optional
        If True, each segment is plotted from zero, by default False.
    """
    locs = [0] + list(line_locations) + [len(data)]
    for i in range(len(locs) - 1):
        segment = data[locs[i] : locs[i + 1]]
        x_vals = range(len(segment)) if relative else range(locs[i], locs[i + 1])
        ax.plot(x_vals, segment, alpha=0.7, label=f"Chain {i+1}")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_chain_lines(ax, line_locations)
    ax.legend()




def plot_dist_series(
    pairs: list[tuple[int, int]],
    ref_dists: list[np.ndarray],
    traj_dists: list[np.ndarray],
    outpath: str,
    name: str,
    line_locations: list[int],
) -> None:
    """
    Plot distance time-series for atom pairs in reference and trajectory.

    Generates two figures: one for reference distances and one for simulation,
    each showing distances for specified atom-pair indices over time.

    Parameters
    ----------
    pairs : list[tuple[int, int]]
        Atom index pairs for distance calculation.
    ref_dists : list[np.ndarray]
        Reference distances arrays per pair.
    traj_dists : list[np.ndarray]
        Simulation distances arrays per pair.
    outpath : str
        Directory for saving plots.
    name : str
        Label for simulation plots.
    line_locations : list[int]
        Frame indices where chains restart.
    """
    print("Plotting atom pair distance series..")
    # Reference distances
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, dist in enumerate(ref_dists):
        ax.plot(dist, label=f"Dist {i} {pairs[i]}")
    add_chain_lines(ax, line_locations)
    ax.set_title("Reference Atom Pair Distances")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Distance")
    ax.legend(loc="upper right")
    fig.savefig(os.path.join(outpath, "Reference_atom_pair_distances.png"), dpi=300)
    plt.close(fig)

    # Simulation distances
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, dist in enumerate(traj_dists):
        ax2.plot(dist, label=f"Dist {i} {pairs[i]}")
    add_chain_lines(ax2, line_locations)
    ax2.set_title(f"{name} Atom Pair Distances")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Distance")
    ax2.legend(loc="upper right")
    fig2.savefig(os.path.join(outpath, f"{name}_atom_pair_distances.png"), dpi=300)
    plt.close(fig2)


def plot_dihedrals(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    outpath: str,
    line_locations: list[int],
) -> None:
    """
    Plot dihedral angle distributions and chain-averaged statistics.

    First panel compares histograms of phi/psi for reference vs simulation.
    Second panel overlays per-chain mean±std for simulation.

    Parameters
    ----------
    AT_phi : np.ndarray
        Reference phi angles per frame.
    AT_psi : np.ndarray
        Reference psi angles per frame.
    Traj_phi : np.ndarray
        Simulation phi angles per frame.
    Traj_psi : np.ndarray
        Simulation psi angles per frame.
    outpath : str
        Directory for saving figures.
    line_locations : list[int]
        Chain boundary indices.
    """
    from cgbench.plotting import distributions
    from cgbench.utils.chains import split_into_chains

    print("Plotting dihedrals..")
    # 1D dihedral distributions
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].set_title("Dihedral angle phi")
    distributions.plot_1d_dihedral(
        axs[1], [AT_psi, Traj_psi], ["Reference", "Simulation"], bins=60, degrees=True
    )
    axs[1].set_title("Dihedral angle psi")
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "Dihedrals.png"), dpi=300)
    plt.close(fig)

    # Per-chain mean/std overlay
    Traj_phi_chains = split_into_chains(Traj_phi, line_locations)
    Traj_psi_chains = split_into_chains(Traj_psi, line_locations)
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    distributions.plot_1d_dihedral_mean_std(
        axs2[0], [AT_phi], ["Reference"], bins=60, degrees=True, color="blue"
    )
    distributions.plot_1d_dihedral_mean_std(
        axs2[0],
        [Traj_phi_chains],
        ["Simulation"],
        bins=60,
        degrees=True,
        color="orange",
    )
    axs2[0].set_title("Dihedral angle phi")
    distributions.plot_1d_dihedral_mean_std(
        axs2[1], [AT_psi], ["Reference"], bins=60, degrees=True, color="blue"
    )
    distributions.plot_1d_dihedral_mean_std(
        axs2[1],
        [Traj_psi_chains],
        ["Simulation"],
        bins=60,
        degrees=True,
        color="orange",
    )
    axs2[1].set_title("Dihedral angle psi")
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, "Dihedrals_mean_std.png"), dpi=300)
    plt.close(fig2)


def plot_energy(
    ax: Axes,
    energy: Union[np.ndarray, list[np.ndarray]],
    labels: list[str] = None,
    xlabel: str = "Time",
    ylabel: str = "Energy [kJ/mol]",
) -> Axes:
    """
    Plot energy values over time.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    energy : Union[np.ndarray, list[np.ndarray]]
        Energy data - single array or list of arrays for multiple models
    labels : list[str], optional
        List of labels for each set of energy values
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

    if isinstance(energy, (list, tuple)) and hasattr(energy[0], "__len__"):
        n_models = len(energy)
        for i in range(n_models):
            ax.plot(
                range(len(energy[i])),
                energy[i],
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0,
            )
    else:
        ax.plot(range(len(energy)), energy, color=color[0], linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_kT(
    ax: Axes,
    kT: Union[np.ndarray, list[np.ndarray]],
    labels: list[str] = None,
    xlabel: str = "Time",
    ylabel: str = "kT [kJ/mol]",
) -> Axes:
    """
    Plot kT values over time.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    kT : Union[np.ndarray, list[np.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str], optional
        List of labels for each set of kT values
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

    if isinstance(kT, (list, tuple)) and hasattr(kT[0], "__len__"):
        n_models = len(kT)
        for i in range(n_models):
            ax.plot(
                range(len(kT[i])),
                kT[i],
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0,
            )
    else:
        ax.plot(range(len(kT)), kT, color=color[0], linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_T(
    ax: Axes,
    kT: Union[np.ndarray, list[np.ndarray]],
    labels: list[str] = None,
    xlabel: str = "Time",
    ylabel: str = "T [K]",
) -> Axes:
    """
    Plot temperature values over time by converting from kT.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    kT : Union[np.ndarray, list[np.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str], optional
        List of labels for each set of kT values
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

    if isinstance(kT, (list, tuple)) and hasattr(kT[0], "__len__"):
        n_models = len(kT)
        for i in range(n_models):
            ax.plot(
                range(len(kT[i])),
                kT[i] / quantity.kb,
                label=labels[i] if labels else None,
                color=color[i % len(color)],
                linestyle=line[i % len(line)],
                linewidth=2.0,
            )
    else:
        ax.plot(range(len(kT)), kT / quantity.kb, color=color[0], linewidth=2.0)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    return ax


def plot_energy_and_kT(
    aux: dict,
    line_locations: list[int],
    outpath: str,
) -> None:
    """
    Plot energy and temperature time-series and overlay chains.

    For each available key in aux ('epot', 'kT', 'etot', 'Temperature'),
    creates two plots: a standard time series and an overlaid chains plot.
    Highlights any chains that "explode" (values >10000).

    Parameters
    ----------
    aux : dict
        Dictionary containing arrays for 'epot', 'kT', 'etot', etc.
    line_locations : list[int]
        Indices delineating chain boundaries.
    outpath : str
        Directory to save plots.
    """
    mapping = [
        ("Epot", aux.get("epot"), plot_energy),
        ("kT", aux.get("kT"), plot_kT),
        ("Etotal", aux.get("etot"), plot_energy),
        ("Temperature", aux.get("kT"), plot_T),
    ]
    if "eprior" in aux:
        mapping.append(("Eprior", aux.get("eprior"), plot_energy))

    for label, data, plot_fn in mapping:
        if data is None:
            continue
        # Standard time-series
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_fn(ax, data)
        add_chain_lines(ax, line_locations)
        ax.set_title(label)
        plt.tight_layout()
        fig.savefig(os.path.join(outpath, f"{label}.png"), dpi=300)
        plt.close(fig)
        # Overlaid chains
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        boundaries = [0] + list(line_locations) + [len(data)]
        exploded = 0
        threshold = 5 if label == "kT" else 10000
        for i in range(len(boundaries) - 1):
            seg = np.array(data[boundaries[i] : boundaries[i + 1]])
            if np.any(seg > threshold):
                exploded += 1
            mask = seg <= threshold
            if mask.any():
                ax2.plot(
                    np.where(mask)[0], seg[mask], alpha=0.7, label=f"Chain {i+1}"
                )
        if exploded:
            ax2.text(
                0.02,
                0.98,
                f"Chains exploded: {exploded}",
                transform=ax2.transAxes,
                color="red",
                fontweight="bold",
                verticalalignment="top",
            )
        ax2.set_title(f"{label} - Overlaid chains")
        ax2.set_xlabel("Time step (0.5 ps)")
        ax2.set_ylabel(label)
        plt.tight_layout()
        fig2.savefig(os.path.join(outpath, f"{label}_overlaid.png"), dpi=300)
        plt.close(fig2)
