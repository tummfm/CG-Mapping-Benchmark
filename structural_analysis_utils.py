import jax.numpy as jnp
from jax import vmap
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import utils
from chemtrain import quantity
from jax import jit

import numpy as np
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

# Color, font size and line width variables
color_ref = "#332288"
color_mace = "#88CCEE"
color_fm = "#44AA99"
colors = ["#88CCEE", "#44AA99", "#DDCC77", "#CC6677"]
tick_font_size = 16
axis_label_font_size = 16
legend_font_size = 16
line_width = 3


def radius_of_gyration_vectorized(
    coords: jnp.ndarray,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Vectorized version of radius of gyration calculation with PBC correction.
    """
    n_frames, n_atoms, _ = coords.shape

    # Use first atom as reference for each frame
    ref_atoms = coords[:, 0, :]  # Shape: (n_frames, 3)

    # Expand reference atoms to match all atom positions
    ref_expanded = jnp.expand_dims(ref_atoms, axis=1)  # Shape: (n_frames, 1, 3)
    ref_tiled = jnp.tile(ref_expanded, (1, n_atoms, 1))  # Shape: (n_frames, n_atoms, 3)

    # Compute PBC-corrected displacements for all atoms relative to reference
    # Flatten to apply displacement_fn
    coords_flat = coords.reshape(-1, 3)  # Shape: (n_frames * n_atoms, 3)
    ref_flat = ref_tiled.reshape(-1, 3)  # Shape: (n_frames * n_atoms, 3)

    # Apply displacement function to all pairs
    displacements_flat = vmap(displacement_fn)(coords_flat, ref_flat)
    displacements = displacements_flat.reshape(n_frames, n_atoms, 3)

    # Corrected positions = reference + displacements
    corrected_positions = ref_tiled + displacements

    # Calculate center of mass for each frame
    center_of_mass = jnp.mean(
        corrected_positions, axis=1, keepdims=True
    )  # Shape: (n_frames, 1, 3)

    # Calculate squared distances from COM
    diff = corrected_positions - center_of_mass
    squared_distances = jnp.sum(diff**2, axis=2)  # Shape: (n_frames, n_atoms)

    # Calculate Rg for each frame
    mean_squared_dist = jnp.mean(squared_distances, axis=1)  # Shape: (n_frames,)
    rg_values = jnp.sqrt(mean_squared_dist)

    return rg_values


def helicity_vectorized(
    coords: jnp.ndarray,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Calculate the helicity content using vectorized distance computation.

    Parameters:
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    displacement_fn : Callable
        Function that computes PBC-corrected displacement between two position vectors

    Returns:
    jnp.ndarray
        Array of helicity content values with shape (n_frames,)

    Based on:
    Rudzinski, Joseph F., and William G. Noid. "Bottom-up coarse-graining of peptide ensembles and helix–coil transitions." 
    Journal of chemical theory and computation 11.3 (2015): 1278-1291.
    https://pubs.acs.org/doi/10.1021/ct5009922
    """
    n_frames, n_atoms, _ = coords.shape

    # Parameters from the formula
    R0 = 0.5  # nm
    sigma_squared = 0.02  # nm^2

    if n_atoms < 4:
        return jnp.zeros(n_frames)

    # Calculate all 1-4 distances at once
    all_distances = []
    for i in range(n_atoms - 3):
        j = i + 3
        distances = utils.compute_atom_distance(coords, i, j, displacement_fn)
        all_distances.append(distances)

    # Stack all distances: shape (n_hel_pairs, n_frames)
    distance_matrix = jnp.stack(all_distances, axis=0)

    # Calculate exponential terms for all pairs and frames
    exp_terms = jnp.exp(-(1.0 / (2 * sigma_squared)) * (distance_matrix - R0) ** 2)

    # Average over all 1-4 pairs for each frame
    helicity_values = jnp.mean(exp_terms, axis=0)

    return helicity_values


def plot_helicity_gyration(
    coords,
    displacement,
    starting_frames=None,
    save_pdf=False,
    prefix="",
    suffix="",
    scale_used=None,
):
    # Font and line settings
    tick_font_size = 16
    axis_label_font_size = 16
    legend_font_size = 16
    line_width = 3

    # Set DPI and font sizes consistently for all plots
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = tick_font_size
    plt.rcParams["axes.labelsize"] = axis_label_font_size
    plt.rcParams["axes.titlesize"] = axis_label_font_size
    plt.rcParams["xtick.labelsize"] = tick_font_size
    plt.rcParams["ytick.labelsize"] = tick_font_size
    plt.rcParams["legend.fontsize"] = legend_font_size
    plt.rcParams["lines.linewidth"] = line_width

    print(f"Input coords shape: {coords.shape}")

    # Create mask for valid (non-NaN) frames
    valid_mask = ~jnp.isnan(coords).any(axis=(1, 2))
    valid_indices = jnp.where(valid_mask)[0]  # Original frame indices of valid frames
    coords_valid = coords[valid_mask]  # Only valid coordinates

    print(
        f"Valid coords shape: {coords_valid.shape}, Number of valid frames: {coords_valid.shape[0]}"
    )

    # Calculate values for valid frames only
    rg_values = radius_of_gyration_vectorized(coords_valid, displacement)
    helicity_values = helicity_vectorized(coords_valid, displacement)

    xi_norm_ref_starting_frames = None
    helicity_values_starting_frames = None
    rg_values_starting_frames = None
    if starting_frames is not None:
        xi_norm_ref_starting_frames = xi_norm_vectorized(
            starting_frames, displacement
        ).flatten()
        helicity_values_starting_frames = helicity_vectorized(
            starting_frames, displacement
        )
        rg_values_starting_frames = radius_of_gyration_vectorized(
            starting_frames, displacement
        )

    # Convert to numpy (1D) to be safe for plotting / utils
    rg_values = np.asarray(rg_values).ravel()
    helicity_values = np.asarray(helicity_values).ravel()

    # Find extrema in the valid data
    max_helicity_idx_in_valid = np.argmax(helicity_values)
    min_helicity_idx_in_valid = np.argmin(helicity_values)
    max_rg_idx_in_valid = np.argmax(rg_values)
    min_rg_idx_in_valid = np.argmin(rg_values)

    # Convert back to original frame indices
    max_idx = int(valid_indices[max_helicity_idx_in_valid])
    min_idx = int(valid_indices[min_helicity_idx_in_valid])
    max_rg_idx = int(valid_indices[max_rg_idx_in_valid])
    min_rg_idx = int(valid_indices[min_rg_idx_in_valid])

    print(
        f"Frame with max helicity: {max_idx}, value: {helicity_values[max_helicity_idx_in_valid]}"
    )
    print(
        f"Frame with min helicity: {min_idx}, value: {helicity_values[min_helicity_idx_in_valid]}"
    )
    print(f"Frame with max Rg: {max_rg_idx}, value: {rg_values[max_rg_idx_in_valid]}")
    print(f"Frame with min Rg: {min_rg_idx}, value: {rg_values[min_rg_idx_in_valid]}")

    # Find the frame with the lowest sum of rg_values and helicity_values
    sum_rg_hel = rg_values + helicity_values
    min_sum_idx_in_valid = np.argmin(sum_rg_hel)
    min_sum_idx = int(
        valid_indices[min_sum_idx_in_valid]
    )  # Convert to original frame index

    print(
        f"Frame with lowest rg + helicity: {min_sum_idx}, value: {sum_rg_hel[min_sum_idx_in_valid]} "
        f"(Rg: {rg_values[min_sum_idx_in_valid]}, Helicity: {helicity_values[min_sum_idx_in_valid]})"
    )

    # Plot helicity_values over original frame indices
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(
        np.asarray(valid_indices),
        helicity_values,
        label="Helicity Content",
        color="blue",
        linewidth=line_width,
    )
    ax1.set_ylabel("Helicity Content (Q_hel)")
    ax1.set_xlabel("Frame")
    ax2 = ax1.twinx()
    ax2.plot(
        np.asarray(valid_indices),
        rg_values,
        label="Radius of Gyration",
        color="orange",
        linewidth=line_width,
    )
    ax2.set_ylabel("Radius of Gyration (nm)")
    ax1.set_title("Helicity Content Over Time")

    # Build combined legend (handles from both axes)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    if save_pdf:
        plt.savefig(f"{prefix}helicity_over_time{suffix}.pdf")

    plt.show()

    # Plot rg against helicity with free-energy histogram background
    fig, ax = plt.subplots(figsize=(8, 5))

    # Call with legend=False to avoid colorbar, we'll add our own legend for scatter points
    utils.plot_histogram_free_energy(
        ax,
        np.asarray(rg_values),
        np.asarray(helicity_values),
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$Rg (nm)$",
        ylabel_text="$Q_{hel}$",
        show_ylabel=True,
        ylim=(-0.001, 1),
        xlim=(0.4, 2.5),
        legend=True,  # Changed to False to avoid colorbar
        show_yticks=True,
        scale=scale_used,
        bins=200,
    )

    # Add scatter points
    ax.scatter(
        rg_values[min_sum_idx_in_valid],
        helicity_values[min_sum_idx_in_valid],
        color="black",
        s=60,
        label="min(Rg+Helicity)",
        zorder=5,
    )
    ax.scatter(
        rg_values[max_helicity_idx_in_valid],
        helicity_values[max_helicity_idx_in_valid],
        color="magenta",
        s=60,
        marker="x",
        label="max Helicity",
        zorder=5,
    )

    if starting_frames is not None:
        ax.scatter(
            np.asarray(rg_values_starting_frames).ravel(),
            np.asarray(helicity_values_starting_frames).ravel(),
            color="green",
            s=40,
            marker="x",
            label="starting frames",
            zorder=5,
        )

    ax.legend(loc="best")  # Add legend only for scatter points
    plt.tight_layout()
    if save_pdf:
        plt.savefig(f"{prefix}helicity_vs_rg{suffix}.pdf")

    plt.show()

    # xi_norm reference vs helicity (third plot)
    xi_norm_ref = xi_norm_vectorized(coords_valid, displacement).flatten()
    xi_norm_ref_np = np.asarray(xi_norm_ref).ravel()
    helicity_values_np = helicity_values  # already numpy

    fig, ax = plt.subplots(figsize=(8, 5))
    utils.plot_histogram_free_energy(
        ax,
        xi_norm_ref_np,
        helicity_values_np,
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$\\chi_{hel}$",
        ylabel_text="$Q_{hel}$",
        show_ylabel=True,
        ylim=(-0.001, 1),
        xlim=(-0.06, 0.06),
        legend=True,  # Changed to False
        show_yticks=True,
        scale=scale_used,
        bins=200,
    )

    # Add scatter points for starting frames if provided
    if starting_frames is not None and xi_norm_ref_starting_frames is not None:
        ax.scatter(
            np.asarray(xi_norm_ref_starting_frames).ravel(),
            np.asarray(helicity_values_starting_frames).ravel(),
            color="green",
            s=40,
            marker="x",
            label="starting frames",
            zorder=5,
        )

    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.tight_layout()
    if save_pdf:
        plt.savefig(f"{prefix}helicity_vs_xi_norm{suffix}.pdf")

    plt.show()

    return max_idx, min_idx, min_sum_idx, rg_values, helicity_values_np, xi_norm_ref_np


def _calculate_xi_norm_single_frame(
    coords_f: jnp.ndarray,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Calculates the normalized helical chirality index (xi_norm) for a single frame.

    Based on the idea from:
    Sidorova, Alla E., et al. "Protein helical structures: Defining handedness and localization features." Symmetry 13.5 (2021): 879.
    https://www.mdpi.com/2073-8994/13/5/879
    """
    n_atoms, _ = coords_f.shape
    assert n_atoms >= 4, "Number of atoms must be at least 4 to calculate xi_norm."

    v_i = vmap(displacement_fn)(coords_f[1:], coords_f[:-1])

    # i runs from 0 to n_atoms - 4. Number of triplets = n_atoms - 3
    v1 = v_i[:-2]  # v_i
    v2 = v_i[1:-1]  # v_{i+1}
    v3 = v_i[2:]  # v_{i+2}

    # cross_product shape: (n_triplets, 3)
    cross_product = jnp.cross(v2, v3)

    # Calculate the mixed product (scalar triple product): (v_i x v_{i+1}) . v_{i+2}
    mixed_products = jnp.sum(cross_product * v1, axis=1)

    chi_total = jnp.sum(mixed_products)

    # Normalize by the number of triplets to get the normalized index (xi_norm)
    n_triplets = n_atoms - 3
    xi_norm_value = chi_total / n_triplets

    return xi_norm_value


def xi_norm_vectorized(
    coords: jnp.ndarray,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Vmap wrapper of _calculate_xi_norm_single_frame.

    Parameters:
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_CA, 3)
    displacement_fn : Callable
        Function that computes PBC-corrected displacement between two position vectors

    Returns:
    jnp.ndarray
        Array of xi_norm values with shape (n_frames, 1)
    """
    xi_norm_values = vmap(_calculate_xi_norm_single_frame, in_axes=(0, None))(
        coords, displacement_fn
    )
    return jnp.expand_dims(xi_norm_values, axis=1)


def plot_1d_dihedral(
    ax,
    angles: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = "$\phi$ (deg)",
    plot_legend: bool = True,
    ylabel: bool = True,
    tick_bin: float = 90,
    mode: str = "single",
    n_std: int = 1,
):
    """
    Plot 1D dihedral angle distributions with support for single or multiple chains.

    Parameters
    ----------
    ax : matplotlib axis
        The axis to plot on
    angles : list[np.ndarray]
        List of angle arrays. Each can be:
        - 1D array (n_frames) for single chain
        - 2D array (n_chains, n_frames) for multiple chains
    labels : list[str]
        Labels for each dataset
    bins : int
        Number of histogram bins
    degrees : bool
        If True, angles are in degrees; if False, in radians
    xlabel : str
        X-axis label
    plot_legend : bool
        Whether to show legend
    ylabel : bool
        Whether to show y-axis label
    tick_bin : float
        Spacing for x-axis ticks
    mode : str
        'single' for original behavior, 'multi' for multi-chain with std
    n_std : int
        Number of standard deviations for fill_between in multi mode
    """
    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    n_models = len(angles)

    for i in range(n_models):
        ang = angles[i]

        # -----------------------------------
        # mode="single" (original behavior)
        # -----------------------------------
        if mode == "single":
            if degrees:
                # Convert angles to [-180, 180] range
                angles_conv = ((ang + 180) % 360) - 180
                hist_range = [-180, 180]
            else:
                # Convert angles to [-π, π] range
                angles_conv = ((ang + np.pi) % (2 * np.pi)) - np.pi
                hist_range = [-np.pi, np.pi]

            hist, x_bins = np.histogram(
                angles_conv, bins=bins, density=True, range=hist_range
            )
            width = x_bins[1] - x_bins[0]
            bin_center = x_bins[:-1] + width / 2

            ax.plot(
                bin_center,
                hist,
                label=labels[i],
                color=color[i % len(color)],
                linewidth=line_width,
            )

        # -----------------------------------
        # mode="multi" (single or multi-chain)
        # -----------------------------------
        elif mode == "multi":
            # Set histogram range based on units
            if degrees:
                hist_range = [-180, 180]
            else:
                hist_range = [-np.pi, np.pi]

            # Handle single chain (1D array)
            if ang.ndim == 1:
                if degrees:
                    angles_conv = ((ang + 180) % 360) - 180
                else:
                    angles_conv = ((ang + np.pi) % (2 * np.pi)) - np.pi

                hist, x_bins = np.histogram(
                    angles_conv, bins=bins, density=True, range=hist_range
                )
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                ax.plot(
                    bin_center,
                    hist,
                    color=color[i % len(color)],
                    linewidth=line_width,
                    label=labels[i],
                )

            # Handle multiple chains (2D array)
            elif ang.ndim == 2:
                n_chains = ang.shape[0]

                # Create bin edges once
                _, x_bins = np.histogram([], bins=bins, range=hist_range)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                chain_hists = []
                for c in range(n_chains):
                    if degrees:
                        angles_conv = ((ang[c] + 180) % 360) - 180
                    else:
                        angles_conv = ((ang[c] + np.pi) % (2 * np.pi)) - np.pi

                    hist_c, _ = np.histogram(
                        angles_conv, bins=x_bins, density=True, range=hist_range
                    )
                    chain_hists.append(hist_c)

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                col = color[i % len(color)]

                # Plot mean curve
                ax.plot(
                    bin_center,
                    hist_mean,
                    color=col,
                    linewidth=line_width,
                    label=labels[i],
                )

                # Plot ± n_std fill
                ax.fill_between(
                    bin_center,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.4,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    # Decorations
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    if ylabel:
        ax.set_ylabel("Density", fontsize=axis_label_font_size)

    # Set x-axis ticks
    if degrees:
        ax.set_xticks(np.arange(-180, 181, tick_bin))
        ax.set_xlim(-180, 180)
    else:
        ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, tick_bin))
        ax.set_xlim(-np.pi, np.pi)

    ax.tick_params(direction="in", labelsize=tick_font_size)
    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)

    ax.set_ylim(bottom=0)

    return ax


def plot_1d_angle(
    ax,
    angles: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    xlabel: str = "$\\Theta$ (deg)",
    plot_legend: bool = True,
    ylabel: bool = True,
    degrees=True,
    mode="single",
    n_std=1,
):
    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    n_models = len(angles)

    for i in range(n_models):

        ang = angles[i]

        # -----------------------------------
        # mode="single" (old behavior)
        # -----------------------------------
        if mode == "single":
            if degrees:
                ang = np.rad2deg(ang)

            hist_range = [ang.min(), ang.max()]
            hist, x_bins = np.histogram(ang, bins=bins, range=hist_range)
            hist = hist / np.sum(hist)

            width = x_bins[1] - x_bins[0]
            bin_center = x_bins[:-1] + width / 2

            ax.plot(
                bin_center,
                hist,
                color=color[i % len(color)],
                linewidth=line_width,
                label=labels[i],
            )

        # -----------------------------------
        # mode="multi" (single or multi-chain)
        # -----------------------------------
        elif mode == "multi":

            # Convert angle units
            if degrees:
                ang = np.rad2deg(ang)

            # Handle single chain (1D array)
            if ang.ndim == 1:
                hist_range = [ang.min(), ang.max()]
                hist, x_bins = np.histogram(ang, bins=bins, range=hist_range)
                hist = hist / np.sum(hist)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                ax.plot(
                    bin_center,
                    hist,
                    color=color[i % len(color)],
                    linewidth=line_width,
                    label=labels[i],
                )

            # Handle multiple chains (2D array)
            elif ang.ndim == 2:
                n_chains = ang.shape[0]

                hist_range = [ang.min(), ang.max()]
                _, x_bins = np.histogram(ang[0], bins=bins, range=hist_range)
                width = x_bins[1] - x_bins[0]
                bin_center = x_bins[:-1] + width / 2

                chain_hists = []
                for c in range(n_chains):
                    hist_c, _ = np.histogram(ang[c], bins=x_bins, range=hist_range)
                    hist_c = hist_c / np.sum(hist_c)
                    chain_hists.append(hist_c)

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                col = color[i % len(color)]

                ax.plot(
                    bin_center,
                    hist_mean,
                    color=col,
                    linewidth=line_width,
                    label=labels[i],
                )
                ax.fill_between(
                    bin_center,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.4,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    # decorations
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    ax.tick_params(direction="in", labelsize=tick_font_size)
    if ylabel:
        ax.set_ylabel("Density", fontsize=axis_label_font_size)
    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)
    ax.set_ylim(bottom=0)

    return ax


def plot_1d_bond(
    ax,
    bonds: list[np.ndarray],
    labels: list[str],
    bins: int = 120,
    xlabel: str = "b (nm)",
    ylabel: bool = True,
    plot_legend: bool = True,
    tick_bin: float = 0.01,
    mode="single",
    n_std=1,
):
    colors = [
        color_ref,
        color_mace,
        color_fm,
        "#FFB347",
        "#7851A9",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#F4A261",
    ]

    # -------------------------------------------------------
    # Shared bin edges across ALL datasets (matching your original)
    # -------------------------------------------------------
    gmin = min(a.min() for a in bonds)
    gmax = max(a.max() for a in bonds)

    if np.isclose(gmin, gmax):
        gmin -= 1e-6
        gmax += 1e-6

    bin_edges = np.linspace(gmin, gmax, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # -------------------------------------------------------
    # Loop through datasets
    # -------------------------------------------------------
    for i, (bond_data, lab) in enumerate(zip(bonds, labels)):
        col = colors[i % len(colors)]

        print(f"Processing bond data for '{lab}': {bond_data.shape}")

        # ===================================================
        # MODE: Single chain — original behavior
        # ===================================================
        if mode == "single":

            counts, _ = np.histogram(bond_data, bins=bin_edges)
            total = counts.sum()
            if total == 0:
                continue

            frac = counts / total

            ax.plot(bin_centers, frac, label=lab, color=col, linewidth=line_width)

        # ===================================================
        # MODE: Multi chain — std across multiple chains
        # ===================================================
        elif mode == "multi":

            if bond_data.ndim == 1:
                # treat as single-chain
                counts, _ = np.histogram(bond_data, bins=bin_edges)
                total = counts.sum()
                if total == 0:
                    continue
                frac = counts / total

                ax.plot(bin_centers, frac, label=lab, color=col, linewidth=line_width)

            elif bond_data.ndim == 2:
                # Multiple chains: shape (n_chains, n_frames)
                n_chains = bond_data.shape[0]

                chain_hists = []
                for c in range(n_chains):
                    bc = bond_data[c]
                    bc = bc[np.isfinite(bc)]
                    if bc.size == 0:
                        continue

                    counts_c, _ = np.histogram(bc, bins=bin_edges)
                    total_c = counts_c.sum()
                    if total_c == 0:
                        continue

                    chain_hists.append(counts_c / total_c)

                if len(chain_hists) == 0:
                    continue

                chain_hists = np.stack(chain_hists, axis=0)
                hist_mean = chain_hists.mean(axis=0)
                hist_std = chain_hists.std(axis=0)

                print(
                    f"Plotting bond distribution for '{lab}': {n_chains} chains, mean={hist_mean.sum()}, std sum={hist_std.sum()}"
                )

                # mean curve
                ax.plot(
                    bin_centers, hist_mean, color=col, linewidth=line_width, label=lab
                )

                # ± n_std fill
                ax.fill_between(
                    bin_centers,
                    hist_mean - n_std * hist_std,
                    hist_mean + n_std * hist_std,
                    color=col,
                    alpha=0.5,
                )

            else:
                raise ValueError(
                    "mode='multi' requires 1D (n_frames) or 2D (n_chains, n_frames) arrays."
                )

        else:
            raise ValueError("mode must be 'single' or 'multi'.")

    # -------------------------------------------------------
    # Decorations
    # -------------------------------------------------------
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)

    if ylabel:
        ax.set_ylabel("Probability per bin", fontsize=axis_label_font_size)

    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size)

    ax.tick_params(direction="in", labelsize=tick_font_size)
    ax.set_ylim(bottom=0)

    # tick spacing
    if tick_bin and tick_bin > 0:
        start = np.floor(gmin / tick_bin) * tick_bin
        stop = np.ceil(gmax / tick_bin) * tick_bin
        ticks = np.arange(start, stop + tick_bin * 0.5, tick_bin)
        ax.set_xticks(ticks)

    return ax


def calculate_rdf(
    trajectories,
    bead_types,
    sites_per_mol=2,
    box_length=2.79573,
    dr=0.01,
    pair_batch_size=900_000,
    frame_batch_size=100000,
    dtype=jnp.float32,
):
    """
    Calculate radial distribution functions for intermolecular pairs using JAX.

    Args:
        trajectories: List of arrays of shape (n_frames, n_particles, 3) or (n_particles, 3)
        bead_types: List of bead types for each site in a molecule
        sites_per_mol: Number of sites per molecule
        box_length: Simulation box length
        dr: Bin width for RDF histogram
        pair_batch_size: Number of pairs to process in a batch
        frame_batch_size: Number of frames to process in a batch
        dtype: JAX data type for computations

    Returns:
        dict: Dictionary with structure {bead_combo: {traj_idx: (r, g_r)}}
        list: List of bead combinations
    """
    trajectories = [jnp.asarray(traj, dtype=dtype) for traj in trajectories]

    n_particles = trajectories[0].shape[-2]

    # Validate bead_types input
    if len(bead_types) != sites_per_mol:
        raise ValueError(
            f"bead_types length ({len(bead_types)}) must match sites_per_mol ({sites_per_mol})"
        )

    # Get unique bead types and create all combinations
    unique_types = sorted(list(set(bead_types)))
    bead_combinations = [
        (t1, t2) for i, t1 in enumerate(unique_types) for t2 in unique_types[i:]
    ]

    # Create full bead type array for all particles
    n_molecules = n_particles // sites_per_mol
    full_bead_types = np.tile(bead_types, n_molecules)

    # Build intermolecular pair indices for each bead combination
    pair_indices = {}
    i_all, j_all = np.triu_indices(n_particles, k=1)
    mol_i = i_all // sites_per_mol
    mol_j = j_all // sites_per_mol
    inter_mask = mol_i != mol_j
    i_inter = i_all[inter_mask]
    j_inter = j_all[inter_mask]

    for type1, type2 in bead_combinations:
        if type1 == type2:
            type_mask = (full_bead_types[i_inter] == type1) & (
                full_bead_types[j_inter] == type1
            )
        else:
            type_mask = (
                (full_bead_types[i_inter] == type1)
                & (full_bead_types[j_inter] == type2)
            ) | (
                (full_bead_types[i_inter] == type2)
                & (full_bead_types[j_inter] == type1)
            )
        pair_indices[(type1, type2)] = (i_inter[type_mask], j_inter[type_mask])

    # JAX-optimized distance calculation
    @jit
    def compute_distances(positions, i_idx, j_idx, L):
        """Compute distances for a batch of frames and pairs."""
        pos_i = positions[:, i_idx, :]  # (F, P, 3)
        pos_j = positions[:, j_idx, :]  # (F, P, 3)
        dr = pos_j - pos_i
        dr = dr - L * jnp.round(dr / L)  # Minimum image convention
        return jnp.sqrt(jnp.sum(dr * dr, axis=-1))  # (F, P)

    # Histogram computation
    @jit
    def compute_histogram(dists, bins):
        """Compute histogram of distances."""
        return jnp.histogram(dists.ravel(), bins=bins)[0]

    volume = box_length**3
    r_max = box_length / 2
    bins_arr = jnp.arange(0.0, r_max + dr, dr)
    shell_volumes = (4.0 / 3.0) * jnp.pi * (bins_arr[1:] ** 3 - bins_arr[:-1] ** 3)

    L = jnp.asarray(box_length, dtype=dtype)
    rdf_data = {}

    for traj_idx, traj in enumerate(trajectories):

        # remove nan
        traj = traj[~jnp.isnan(traj).any(axis=(1, 2))]

        # Ensure trajectory is 3D
        traj = jnp.asarray(traj[None, ...] if traj.ndim == 2 else traj, dtype=dtype)
        n_frames = traj.shape[0]

        for bead_combo in bead_combinations:
            i_combo, j_combo = pair_indices[bead_combo]
            n_combo_pairs = len(i_combo)

            if n_combo_pairs == 0:
                continue

            hist = jnp.zeros(bins_arr.size - 1, dtype=jnp.float32)

            # Process pairs in batches to manage memory
            p0 = 0
            while p0 < n_combo_pairs:
                p1 = min(p0 + pair_batch_size, n_combo_pairs)
                i_batch = jnp.asarray(i_combo[p0:p1], dtype=jnp.int32)
                j_batch = jnp.asarray(j_combo[p0:p1], dtype=jnp.int32)

                # Process frames in batches
                f0 = 0
                while f0 < n_frames:
                    f1 = min(f0 + frame_batch_size, n_frames)
                    positions_f = traj[f0:f1]

                    # Compute distances for this batch
                    dists = compute_distances(positions_f, i_batch, j_batch, L)
                    hist = hist + compute_histogram(dists, bins_arr)

                    f0 = f1
                p0 = p1

            total_pairs = n_combo_pairs
            ideal_counts = (total_pairs * shell_volumes / volume) * n_frames
            g_r = jnp.where(ideal_counts > 0, hist / ideal_counts, 0.0)
            r = 0.5 * (bins_arr[1:] + bins_arr[:-1])

            if bead_combo not in rdf_data:
                rdf_data[bead_combo] = {}
            rdf_data[bead_combo][traj_idx] = (np.asarray(r), np.asarray(g_r))

    return rdf_data, bead_combinations


def plot_rdf(
    rdf_data,
    bead_combinations,
    labels,
    output_prefix="rdf",
    box_length=2.79573,
    mode="single",
    n_std=1.0,
    show_legend=True,
    save_pdf=True,
):
    """
    Plot RDF data with optional multi-chain mean and std shading.

    rdf_data structure:
        rdf_data[bead_combo][traj_idx] = (r_vals, g_vals)
            r_vals shape: (n_chains, n_bins)
            g_vals shape: (n_chains, n_bins)
        If only 1 chain exists → shapes become (1, n_bins) or (n_bins) (both allowed)
    """

    import matplotlib.pyplot as plt
    import numpy as np

    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    r_max = box_length / 2

    for bead_combo in bead_combinations:
        if bead_combo not in rdf_data:
            continue

        type1, type2 = bead_combo
        combo_label = f"{type1}-{type2}"

        fig, ax = plt.subplots(figsize=(6, 5))

        for traj_idx in rdf_data[bead_combo]:

            r_vals, g_vals = rdf_data[bead_combo][traj_idx]

            # --- Ensure shapes ----------------------------------------------------
            r_vals = np.asarray(r_vals)
            g_vals = np.asarray(g_vals)

            # if input is (n_bins,) → convert to (1, n_bins)
            if r_vals.ndim == 1:
                r_vals = r_vals[None, :]
            if g_vals.ndim == 1:
                g_vals = g_vals[None, :]

            n_chains = r_vals.shape[0]
            col = color[traj_idx % len(color)]
            label = labels[traj_idx]

            # --- Mode: SINGLE (old behavior) -------------------------------------
            if mode == "single" or n_chains == 1:
                ax.plot(
                    r_vals[0], g_vals[0], color=col, label=label, linewidth=line_width
                )
                continue

            # --- Mode: MULTI (mean ± std) ----------------------------------------
            r_mean = np.mean(r_vals, axis=0)
            g_mean = np.mean(g_vals, axis=0)
            g_std = np.std(g_vals, axis=0)

            ax.plot(r_mean, g_mean, color=col, label=label, linewidth=line_width)

            ax.fill_between(
                r_mean,
                g_mean - n_std * g_std,
                g_mean + n_std * g_std,
                color=col,
                alpha=0.4,
            )

        # Formatting
        ax.set_xlabel("r (nm)", fontsize=axis_label_font_size)
        ax.set_xlim(0.3, r_max)
        ax.set_ylabel(f"g$_{{{type1}-{type2}}}$(r)", fontsize=axis_label_font_size)
        if show_legend:
            ax.legend(frameon=False, fontsize=legend_font_size, loc="lower right")
        ax.tick_params(direction="in", labelsize=tick_font_size)

        filename = f"{output_prefix}_{combo_label}.pdf"
        plt.tight_layout()
        if save_pdf:
            plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
            print(f"Saved RDF plot for {combo_label} pairs to {filename}")

    print(f"Generated {len(bead_combinations)} RDF plots saved as PDF files.")


def calculate_rdf_mse(r1, g_r1, r2, g_r2, r_min=None, r_max=None):
    """
    Calculate the mean squared error between two radial distribution functions.

    This function handles RDFs that may have different r-value grids by interpolating
    them onto a common grid before computing the MSE.

    Args:
        r1: Array of r values for the first RDF
        g_r1: Array of g(r) values for the first RDF
        r2: Array of r values for the second RDF
        g_r2: Array of g(r) values for the second RDF
        r_min: Minimum r value to consider (default: max of both r minimums)
        r_max: Maximum r value to consider (default: min of both r maximums)

    Returns:
        float: Mean squared error between the two RDFs
    """
    # Convert to numpy arrays if needed
    r1 = np.asarray(r1)
    g_r1 = np.asarray(g_r1)
    r2 = np.asarray(r2)
    g_r2 = np.asarray(g_r2)

    # Determine the common r range
    if r_min is None:
        r_min = max(r1.min(), r2.min())
    if r_max is None:
        r_max = min(r1.max(), r2.max())

    # Create interpolation functions
    interp1 = interp1d(r1, g_r1, kind="linear", bounds_error=False, fill_value=0.0)
    interp2 = interp1d(r2, g_r2, kind="linear", bounds_error=False, fill_value=0.0)

    # Use the finer grid of the two for comparison
    dr1 = np.mean(np.diff(r1))
    dr2 = np.mean(np.diff(r2))
    dr_common = min(dr1, dr2)

    # Create common r grid
    r_common = np.arange(r_min, r_max, dr_common)

    # Interpolate both RDFs onto common grid
    g_r1_interp = interp1(r_common)
    g_r2_interp = interp2(r_common)

    # Calculate MSE
    mse = np.mean((g_r1_interp - g_r2_interp) ** 2)

    return mse


def calculate_rdf_mse_from_dict(
    rdf_data, bead_combo, traj_idx1, traj_idx2, r_min=None, r_max=None
):
    """
    Calculate MSE between two RDFs directly from the rdf_data dictionary output
    of calculate_rdf function.

    Args:
        rdf_data: Dictionary output from calculate_rdf function
        bead_combo: Tuple of bead types (e.g., ('A', 'B'))
        traj_idx1: Index of first trajectory
        traj_idx2: Index of second trajectory
        r_min: Minimum r value to consider (optional)
        r_max: Maximum r value to consider (optional)

    Returns:
        float: Mean squared error between the two RDFs
    """
    if bead_combo not in rdf_data:
        raise ValueError(f"Bead combination {bead_combo} not found in rdf_data")

    if traj_idx1 not in rdf_data[bead_combo]:
        raise ValueError(f"Trajectory index {traj_idx1} not found for {bead_combo}")

    if traj_idx2 not in rdf_data[bead_combo]:
        raise ValueError(f"Trajectory index {traj_idx2} not found for {bead_combo}")

    r1, g_r1 = rdf_data[bead_combo][traj_idx1]
    r2, g_r2 = rdf_data[bead_combo][traj_idx2]

    return calculate_rdf_mse(r1, g_r1, r2, g_r2, r_min, r_max)


def plot_1d_rdf(
    ax,
    rdf_data: dict,
    bead_combo: tuple,
    labels: list[str],
    xlabel: str = "r (nm)",
    ylabel: bool = True,
    tick_spacing: float = 0.2,
    xlim: tuple = None,
    plot_legend: bool = True,
    n_std: float = 1.0,  # <-- new argument
    mode: str = "single",  # "single" or "multi"
):
    """
    Plot RDF data for a specific bead combination on a single axis.
    Supports multi-chain RDF (mean ± std) similar to plot_rdf.

    rdf_data structure:
        rdf_data[bead_combo][traj_idx] = (r_vals, g_vals)
        r_vals and g_vals can be either:
            - shape (n_bins,)
            - shape (n_chains, n_bins)
    """

    import numpy as np

    color = [
        color_ref,
        color_mace,
        color_fm,
        "#DDCC77",
        "#CC6677",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]

    type1, type2 = bead_combo

    # Check this bead combination exists
    if bead_combo not in rdf_data:
        print(f"Warning: Bead combination {bead_combo} not found in RDF data")
        return ax

    # Plot each trajectory
    for traj_idx in rdf_data[bead_combo]:

        r_vals, g_vals = rdf_data[bead_combo][traj_idx]

        # Convert to np arrays
        r_vals = np.asarray(r_vals)
        g_vals = np.asarray(g_vals)

        # Convert shape (n_bins,) --> (1, n_bins)
        if r_vals.ndim == 1:
            r_vals = r_vals[None, :]
        if g_vals.ndim == 1:
            g_vals = g_vals[None, :]

        n_chains = r_vals.shape[0]
        col = color[traj_idx % len(color)]
        label = labels[traj_idx]

        # -------------------------------
        # Single-chain or mode="single"
        # -------------------------------
        if mode == "single" or n_chains == 1:
            ax.plot(r_vals[0], g_vals[0], color=col, label=label, linewidth=line_width)
            continue

        # -------------------------------
        # Multi-chain mean ± std shading
        # -------------------------------
        r_mean = np.mean(r_vals, axis=0)
        g_mean = np.mean(g_vals, axis=0)
        g_std = np.std(g_vals, axis=0)

        ax.plot(r_mean, g_mean, color=col, linewidth=line_width, label=label)

        ax.fill_between(
            r_mean, g_mean - n_std * g_std, g_mean + n_std * g_std, color=col, alpha=0.4
        )

    # Labels
    ax.set_xlabel(xlabel, fontsize=axis_label_font_size)
    if ylabel:
        ax.set_ylabel(f"g$_{{{type1}-{type2}}}$(r)", fontsize=axis_label_font_size)

    # Legend
    if plot_legend:
        ax.legend(frameon=False, fontsize=legend_font_size, loc="lower right")

    ax.tick_params(direction="in", labelsize=tick_font_size)

    # X-limits
    if xlim:
        ax.set_xlim(xlim)

    # Calculate x-ticks
    if xlim:
        min_r, max_r = xlim
    else:
        # extract range across all chains/trajectories
        all_r = []
        for traj_idx in rdf_data[bead_combo]:
            r_vals, _ = rdf_data[bead_combo][traj_idx]
            r_vals = np.asarray(r_vals)
            if r_vals.ndim == 1:
                all_r.extend(r_vals)
            else:
                for r in r_vals:
                    all_r.extend(r)

        if len(all_r) == 0:
            min_r, max_r = 0, 1
        else:
            min_r, max_r = min(all_r), max(all_r)

    ticks = np.arange(
        np.floor(min_r / tick_spacing) * tick_spacing,
        np.ceil(max_r / tick_spacing) * tick_spacing + tick_spacing,
        tick_spacing,
    )
    ax.set_xticks(ticks)

    return ax
