"""
Structural visualization (RDF, Ramachandran, helicity, free energy surfaces).
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from jax import numpy as jnp

from .style import (
    color_ref,
    color_mace,
    color_fm,
    tick_font_size,
    axis_label_font_size,
    legend_font_size,
    line_width,
)


def plot_rdf(
    rdf_data,
    bead_combinations,
    labels,
    output_prefix="rdf",
    box_length=2.79573,
    mode="single",
    n_std=1.0,
    show_legend=True,
    legend_loc="lower right",
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

        if len(bead_combo) == 3:
            type1, type2, type3 = bead_combo
            combo_label = f"{type1}-{type2}-{type3}"
        else:
            type1, type2 = bead_combo
            combo_label = f"{type1}-{type2}"

        fig, ax = plt.subplots(figsize=(6, 5))

        for traj_idx in rdf_data[bead_combo]:
            r_vals, g_vals = rdf_data[bead_combo][traj_idx]

            # Ensure shapes
            r_vals = np.asarray(r_vals)
            g_vals = np.asarray(g_vals)

            if r_vals.ndim == 1:
                r_vals = r_vals[None, :]
            if g_vals.ndim == 1:
                g_vals = g_vals[None, :]

            n_chains = r_vals.shape[0]
            col = color[traj_idx % len(color)]
            label = labels[traj_idx]

            if mode == "single" or n_chains == 1:
                ax.plot(
                    r_vals[0], g_vals[0], color=col, label=label, linewidth=line_width
                )
                continue

            # Mode: MULTI (mean ± std)
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
        ax.set_ylabel(f"g$_{combo_label}$(r)", fontsize=axis_label_font_size)
        if show_legend:
            ax.legend(frameon=False, fontsize=legend_font_size, loc=legend_loc)
        ax.tick_params(direction="in", labelsize=tick_font_size)

        filename = f"{output_prefix}_{combo_label}.pdf"
        plt.tight_layout()
        if save_pdf:
            plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
            print(f"Saved RDF plot for {combo_label} pairs to {filename}")

    print(f"Generated {len(bead_combinations)} RDF plots saved as PDF files.")


def plot_ramachandran(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    kT: float,
    outpath: str,
) -> None:
    """
    Plot free-energy surfaces (Ramachandran plots) for phi vs psi.

    Generates side-by-side free-energy contour plots for reference and simulation
    on the same phi-psi grid, colored by kcal/mol.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1, c1 = plot_histogram_free_energy_simple(ax1, AT_phi, AT_psi, kT)
    ax1.set_title("Reference")
    plt.colorbar(c1, ax=ax1, label="Free energy [kcal/mol]")
    ax2, c2 = plot_histogram_free_energy_simple(ax2, Traj_phi, Traj_psi, kT)
    ax2.set_title("Simulation")
    plt.colorbar(c2, ax=ax2, label="Free energy [kcal/mol]")
    plt.tight_layout()
    fig.savefig(f"{outpath}/Ramachandran.png", dpi=300)
    plt.close(fig)


def determine_free_energy_scale(
    x_list, y_list, kbt: float, bins: int = 100, min_count_for_color: int = 0
):
    """
    Determine a common free-energy scale (vmin, vmax) across multiple datasets.
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length.")

    F_all_min, F_all_max = np.inf, -np.inf
    any_valid = False

    for x, y in zip(x_list, y_list):
        x_np = np.asarray(x)
        y_np = np.asarray(y)

        if x_np.size == 0 or y_np.size == 0:
            continue

        mask = np.isfinite(x_np) & np.isfinite(y_np)
        if np.sum(mask) == 0:
            continue

        x_f = x_np[mask]
        y_f = y_np[mask]

        counts, x_edges, y_edges = np.histogram2d(x_f, y_f, bins=bins, density=False)
        density, _, _ = np.histogram2d(x_f, y_f, bins=bins, density=True)

        mask_occ = counts > min_count_for_color

        density_jnp = jnp.asarray(density)
        mask_occ_jnp = jnp.asarray(mask_occ.astype(bool))

        with np.errstate(invalid="ignore"):
            F_jnp = jnp.where(
                density_jnp > 0, jnp.log(density_jnp) * (-(kbt / 4.184)), jnp.nan
            )

        F_jnp = jnp.where(mask_occ_jnp, F_jnp, jnp.nan)
        F_np = np.asarray(F_jnp)

        if np.any(np.isfinite(F_np)):
            any_valid = True
            F_valid = F_np[np.isfinite(F_np)]

            current_min = float(np.min(F_valid))
            current_max = float(np.max(F_valid))

            F_all_min = min(F_all_min, current_min)
            F_all_max = max(F_all_max, current_max)

    if not any_valid:
        raise ValueError(
            "No valid free-energy data found across datasets (all inputs empty or NaN)."
        )

    F_range = F_all_max - F_all_min
    return (0, float(F_range))


def plot_histogram_free_energy(
    ax,
    x,
    y,
    kbt: float,
    is_angular: bool = True,
    degrees: bool = True,
    xlabel: str | None = None,
    ylabel_text: str | None = None,
    show_ylabel: bool = False,
    show_yticks: bool = True,
    title: str = "",
    bins: int = 100,
    min_count_for_color: int = 0,
    edge_feather_bins: float = 1.0,
    alpha_min: float = 0.3,
    legend: bool = False,
    scale: tuple | None = None,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    shift_scale_to_zero: bool = True,
) -> tuple:
    """
    Plot a free-energy surface from x/y samples.

    Returns:
      (ax, cax, cbar, scale_used)
    """
    # Styling
    sns.set_style("white")
    sns.set_palette(sns.color_palette("Dark2", n_colors=6), n_colors=6)
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.markersize": 3,
            "lines.linewidth": 3.0,
            "figure.dpi": 300,
        }
    )

    # Process input data
    x_np = np.asarray(x)
    y_np = np.asarray(y)

    if is_angular:
        if degrees:
            x_plot = np.deg2rad(x_np)
            y_plot = np.deg2rad(y_np)
            default_xlabel = r"$\phi\ (°)$"
            default_ylabel = r"$\psi\ (°)$"
        else:
            x_plot = x_np
            y_plot = y_np
            default_xlabel = r"$\phi$ [rad]"
            default_ylabel = r"$\psi$ [rad]"

        if xlim is None:
            xlim = (-np.pi, np.pi)
        if ylim is None:
            ylim = (-np.pi, np.pi)

        if degrees:
            deg_ticks_rad = np.deg2rad(np.array([-180, -90, 0, 90, 180]))
            deg_tick_labels = ["-180", "-90", "0", "90", "180"]
            use_angular_ticks = True
        else:
            use_angular_ticks = False
    else:
        x_plot = x_np
        y_plot = y_np
        default_xlabel = "x"
        default_ylabel = "y"
        use_angular_ticks = False

    xlabel_final = xlabel if xlabel is not None else default_xlabel
    ylabel_final = ylabel_text if ylabel_text is not None else default_ylabel

    # Histogram
    counts, x_edges, y_edges = np.histogram2d(x_plot, y_plot, bins=bins, density=False)
    density, _, _ = np.histogram2d(x_plot, y_plot, bins=bins, density=True)

    mask_occ = counts > min_count_for_color

    # Free-energy calculation
    density_jnp = jnp.asarray(density)
    with np.errstate(invalid="ignore"):
        F_jnp = jnp.where(
            density_jnp > 0, jnp.log(density_jnp) * (-(kbt / 4.184)), jnp.nan
        )

    mask_occ_jnp = jnp.asarray(mask_occ.astype(bool))
    F_jnp = jnp.where(mask_occ_jnp, F_jnp, jnp.nan)

    F = np.asarray(F_jnp)

    valid = np.isfinite(F).astype(float)
    feather_sigma = float(edge_feather_bins)
    support_blur = gaussian_filter(valid, sigma=feather_sigma)

    alpha = np.clip((support_blur - alpha_min) / max(1e-9, (1.0 - alpha_min)), 0.0, 1.0)
    F_out = np.where(alpha > 0, F, np.nan)

    # Colormap
    colors = [
        (0.00, "#001060"),
        (0.05, "#0030b0"),
        (0.25, "#00b0ff"),
        (0.40, "#80ff80"),
        (0.55, "#ffff80"),
        (0.70, "#ffb000"),
        (0.85, "#ff0000"),
        (1.00, "#800000"),
    ]
    cmap = LinearSegmentedColormap.from_list("fes_blue_to_red", colors, N=512)
    cmap.set_bad(color="white")

    Xe, Ye = np.meshgrid(x_edges, y_edges)

    # Determine color scale
    finite_mask = np.isfinite(F_out)
    if np.any(finite_mask):
        data_min = float(np.nanmin(F_out))
        data_max = float(np.nanmax(F_out))
    else:
        data_min = 0.0
        data_max = 1.0

    if scale is not None:
        if not (isinstance(scale, (tuple, list)) and len(scale) == 2):
            raise ValueError("`scale` must be a tuple (vmin, vmax) or None.")
        vmin, vmax = float(scale[0]), float(scale[1])
        scale_used = (vmin, vmax)
        F_out = F_out - np.nanmin(F_out)
    elif shift_scale_to_zero:
        F_out = F_out - data_min
        scale_used = (0.0, data_max - data_min)
        vmin, vmax = scale_used
    else:
        scale_used = (data_min, data_max)
        vmin, vmax = scale_used

    # Plotting
    cax = ax.pcolormesh(
        Xe, Ye, F_out.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )

    ax.set_xlabel(xlabel_final)
    if show_ylabel:
        ax.set_ylabel(ylabel_final)
    ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if is_angular and use_angular_ticks:
        ax.set_xticks(deg_ticks_rad)
        ax.set_xticklabels(deg_tick_labels)
        if show_yticks:
            ax.set_yticks(deg_ticks_rad)
            ax.set_yticklabels(deg_tick_labels)
        else:
            ax.set_yticks([])
    elif not show_yticks:
        ax.set_yticks([])

    ax.tick_params(direction="out", which="both")

    # Optional colorbar
    cbar = None
    if legend:
        divider = make_axes_locatable(ax)
        cb_ax = divider.append_axes("right", size="4%", pad=0.03)
        cbar = plt.colorbar(cax, cax=cb_ax, orientation="vertical")
        cbar.set_label("Free energy (kcal/mol)")

    return ax, cax, cbar, scale_used


def plot_histogram_free_energy_simple(
    ax: Axes,
    phi: jnp.ndarray,
    psi: jnp.ndarray,
    kbt: float,
    degrees: bool = True,
    ylabel: bool = False,
    title: str = "",
    bins: int = 60,
):
    """
    Plot 2D free energy histogram for alanine from the dihedral angles.
    """
    cmap = plt.get_cmap("viridis")

    if degrees:
        phi = jnp.deg2rad(phi)
        psi = jnp.deg2rad(psi)

    h, x_edges, y_edges = jnp.histogram2d(phi, psi, bins=bins, density=True)

    h = jnp.log(h) * -(kbt / 4.184)
    x, y = np.meshgrid(x_edges, y_edges)

    cax = ax.pcolormesh(x, y, h.T, cmap=cmap, vmax=5.25)
    ax.set_xlabel("$\\phi$ [rad]")
    if ylabel:
        ax.set_ylabel("$\\psi$ [rad]")
    ax.set_title(title)

    ax.set_ylim([-jnp.pi, jnp.pi])
    ax.set_xlim([-jnp.pi, jnp.pi])

    return ax, cax


def plot_helicity_gyration(
    coords,
    displacement,
    starting_frames=None,
    save_pdf=False,
    prefix="",
    suffix="",
    scale_used=None,
):
    """
    Plot helicity content and radius of gyration analysis.

    Note: This function requires cgbench.utils.structural for the calculation
    functions (radius_of_gyration_vectorized, helicity_vectorized, xi_norm_vectorized).
    """
    from cgbench.utils import structural as struct_utils
    from chemtrain import quantity

    # Font and line settings
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
    valid_indices = jnp.where(valid_mask)[0]
    coords_valid = coords[valid_mask]

    print(
        f"Valid coords shape: {coords_valid.shape}, Number of valid frames: {coords_valid.shape[0]}"
    )

    # Calculate values for valid frames only
    rg_values = struct_utils.radius_of_gyration_vectorized(coords_valid, displacement)
    helicity_values = struct_utils.helicity_vectorized(coords_valid, displacement)

    xi_norm_ref_starting_frames = None
    helicity_values_starting_frames = None
    rg_values_starting_frames = None
    if starting_frames is not None:
        xi_norm_ref_starting_frames = struct_utils.xi_norm_vectorized(
            starting_frames, displacement
        ).flatten()
        helicity_values_starting_frames = struct_utils.helicity_vectorized(
            starting_frames, displacement
        )
        rg_values_starting_frames = struct_utils.radius_of_gyration_vectorized(
            starting_frames, displacement
        )

    # Convert to numpy
    rg_values = np.asarray(rg_values).ravel()
    helicity_values = np.asarray(helicity_values).ravel()

    # Find extrema
    max_helicity_idx_in_valid = np.argmax(helicity_values)
    min_helicity_idx_in_valid = np.argmin(helicity_values)
    max_rg_idx_in_valid = np.argmax(rg_values)
    min_rg_idx_in_valid = np.argmin(rg_values)

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

    sum_rg_hel = rg_values + helicity_values
    min_sum_idx_in_valid = np.argmin(sum_rg_hel)
    min_sum_idx = int(valid_indices[min_sum_idx_in_valid])

    print(
        f"Frame with lowest rg + helicity: {min_sum_idx}, value: {sum_rg_hel[min_sum_idx_in_valid]} "
        f"(Rg: {rg_values[min_sum_idx_in_valid]}, Helicity: {helicity_values[min_sum_idx_in_valid]})"
    )

    # Plot 1: helicity over time
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

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    if save_pdf:
        plt.savefig(f"{prefix}helicity_over_time{suffix}.pdf")

    plt.show()

    # Plot 2: rg vs helicity
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_histogram_free_energy(
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
        legend=True,
        show_yticks=True,
        scale=scale_used,
        bins=200,
    )

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

    ax.legend(loc="best")
    plt.tight_layout()
    if save_pdf:
        plt.savefig(f"{prefix}helicity_vs_rg{suffix}.pdf")

    plt.show()

    # Plot 3: xi_norm vs helicity
    xi_norm_ref = struct_utils.xi_norm_vectorized(coords_valid, displacement).flatten()
    xi_norm_ref_np = np.asarray(xi_norm_ref).ravel()
    helicity_values_np = helicity_values

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_histogram_free_energy(
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
        legend=True,
        show_yticks=True,
        scale=scale_used,
        bins=200,
    )

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
