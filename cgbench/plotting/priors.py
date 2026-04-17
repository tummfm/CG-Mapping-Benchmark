"""Plotting utilities for Boltzmann-inverted bonded priors."""

import math
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .style import setup_plot_style, colors_extended


def _safe_label(key: tuple) -> str:
    return "(" + ",".join(str(x) for x in key) + ")"


def _plot_type_subplot(
    ax: plt.Axes,
    data: dict,
    x_key: str,
    xlabel: str,
    title: str,
) -> None:
    """Fill one subplot with PMF curves for a single bonded type."""
    n = len(data)
    colors = (colors_extended * math.ceil(n / len(colors_extended)))[:n]

    for idx, (key, val) in enumerate(data.items()):
        xgrid = val[x_key]
        U = val["U"]
        # Replace NaN with nanmax so the line is continuous
        finite = ~np.isnan(U)
        if not np.any(finite):
            continue
        U_plot = np.where(finite, U, np.nanmax(U[finite]))
        ax.plot(xgrid, U_plot, color=colors[idx], label=_safe_label(key), linewidth=1.8)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("$U$ (kJ/mol)", fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.tick_params(labelsize=11)

    # Compact legend that doesn't overflow even with many bonds
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = max(1, math.ceil(len(handles) / 8))
        ax.legend(
            handles,
            labels,
            fontsize=max(6, 10 - ncol),
            ncol=ncol,
            loc="upper right",
            framealpha=0.7,
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=1.2,
        )


def _eval_harmonic(x_grid: np.ndarray, x0: float, k: float) -> np.ndarray:
    U = 0.5 * k * (x_grid - x0) ** 2
    return U - U.min()


def _eval_fourier(phi_grid: np.ndarray, coeffs: np.ndarray, n_fourier: int) -> np.ndarray:
    U = coeffs[0] * np.ones_like(phi_grid)
    for n in range(1, n_fourier + 1):
        U = U + coeffs[2 * n - 1] * np.cos(n * phi_grid) + coeffs[2 * n] * np.sin(n * phi_grid)
    return U - U.min()


def plot_bonded_priors(
    all_priors: dict,
    output_dir: str,
    filename: str = "priors_bi.png",
) -> str:
    """Plot Boltzmann-inversion PMFs and save to *output_dir*.

    Creates one subplot per non-empty bonded type (bonds, angles, dihedrals).
    Legends are compact (multi-column, smaller font) to avoid overflow when
    many bonded terms are present.

    Args:
        all_priors:  Output of :meth:`~cgbench.core.prior.BoltzmannPrior.compute_all_priors`.
        output_dir:  Directory to save the figure.
        filename:    Output filename (default ``"priors_bi.png"``).

    Returns:
        Absolute path to the saved figure.
    """
    setup_plot_style()

    type_specs = [
        ("bonds",     "r_grid",     "$r$ (nm)",        "Bond PMFs"),
        ("angles",    "theta_grid", r"$\theta$ (rad)", "Angle PMFs"),
        ("dihedrals", "phi_grid",   r"$\phi$ (rad)",   "Dihedral PMFs"),
    ]

    # Only include non-empty types
    active = [(xk, xlab, title, all_priors[tp])
              for tp, xk, xlab, title in type_specs
              if all_priors.get(tp)]

    if not active:
        return ""

    n_panels = len(active)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.5),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (x_key, xlabel, title, data) in zip(axes, active):
        _plot_type_subplot(ax, data, x_key, xlabel, title)

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_1dfunc_priors(
    all_priors: dict,
    fitted_priors: dict,
    output_dir: str,
    filename: str = "priors_1dfunc.png",
) -> str:
    """Plot BI PMFs with fitted analytical functions overlaid.

    Each bonded type gets one subplot.  For each term, the raw BI potential
    (dashed, translucent) is shown behind the fitted analytical curve (solid).
    This makes fit quality immediately visible.

    Args:
        all_priors:    Output of :meth:`~cgbench.core.prior.BoltzmannPrior.compute_all_priors`.
        fitted_priors: Output of :func:`~cgbench.core.prior.fit_1dfunc_priors`.
        output_dir:    Directory to save the figure.
        filename:      Output filename (default ``"priors_1dfunc.png"``).

    Returns:
        Absolute path to the saved figure.
    """
    setup_plot_style()
    n_fourier = int(fitted_priors.get("n_fourier", 5))

    type_specs = [
        ("bonds",     "r_grid",     "$r$ (nm)",        "Bond priors"),
        ("angles",    "theta_grid", r"$\theta$ (rad)", "Angle priors"),
        ("dihedrals", "phi_grid",   r"$\phi$ (rad)",   "Dihedral priors"),
    ]

    active = [
        (tp, xk, xlab, title)
        for tp, xk, xlab, title in type_specs
        if all_priors.get(tp) and fitted_priors.get(tp)
    ]
    if not active:
        return ""

    n_panels = len(active)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(5.5 * n_panels, 4.5),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes = [axes]

    for ax, (tp, x_key, xlabel, title) in zip(axes, active):
        bi_data = all_priors[tp]
        fit_data = fitted_priors[tp]
        n = len(bi_data)
        color_cycle = (colors_extended * math.ceil(n / len(colors_extended)))[:n]

        for idx, (key, bval) in enumerate(bi_data.items()):
            col = color_cycle[idx]
            x_grid = bval[x_key]
            U_bi = bval["U"]

            # BI PMF — dashed background
            finite = ~np.isnan(U_bi)
            if np.any(finite):
                U_plot = np.where(finite, U_bi, np.nanmax(U_bi[finite]))
                ax.plot(x_grid, U_plot, color=col, alpha=0.35,
                        linestyle="--", linewidth=1.5)

            # Fitted function — solid foreground
            fval = fit_data.get(key)
            if fval is None:
                continue
            if tp == "dihedrals":
                U_fit = _eval_fourier(x_grid, fval["coeffs"], n_fourier)
            else:
                x0_key = "r0" if tp == "bonds" else "theta0"
                U_fit = _eval_harmonic(x_grid, fval[x0_key], fval["k"])

            ax.plot(x_grid, U_fit, color=col, linewidth=1.8,
                    label=_safe_label(key))

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("$U$ (kJ/mol)", fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=11)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = max(1, math.ceil(len(handles) / 8))
            ax.legend(
                handles, labels,
                fontsize=max(6, 10 - ncol),
                ncol=ncol,
                loc="upper right",
                framealpha=0.7,
                borderpad=0.4,
                labelspacing=0.3,
                handlelength=1.2,
            )

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
