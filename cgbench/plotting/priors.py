"""Plotting utilities for Boltzmann-inverted bonded priors and learned splines."""

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


# ---------------------------------------------------------------------------
# Learned spline visualisation
# ---------------------------------------------------------------------------

def _reconstruct_type_labels(terms, species, n_atom_cols):
    """Return a dict mapping type_id → label string from bonded terms list."""
    labels = {}
    for entry in terms:
        tid = int(entry[-1])
        if tid not in labels:
            sp = tuple(int(species[entry[i]]) for i in range(n_atom_cols))
            labels[tid] = "(" + ",".join(str(s) for s in sp) + ")"
    return labels


def _eval_spline(x_grid: np.ndarray, knot_vals: np.ndarray, n_fine: int = 300) -> tuple:
    """Evaluate learned spline on a fine grid using PCHIP interpolation.

    Returns (fine_x, fine_U) with U shifted to zero at its minimum.
    """
    from scipy.interpolate import PchipInterpolator

    fine_x = np.linspace(float(x_grid[0]), float(x_grid[-1]), n_fine)
    fine_U = PchipInterpolator(x_grid, knot_vals)(fine_x)
    fine_U = fine_U - fine_U.min()
    return fine_x, fine_U


def _spline_panel_specs(spline_model):
    """Return the fixed panel specification list for a SplineModel."""
    return [
        ("$r$ (nm)",        "Non-bonded splines", spline_model._nb_x_grids,       "non_bonded"),
        ("$r$ (nm)",        "Bond splines",        spline_model._bond_x_grids,     "bonds"),
        (r"$\theta$ (rad)", "Angle splines",       spline_model._angle_x_grids,    "angles"),
        (r"$\phi$ (rad)",   "Dihedral splines",    spline_model._dihedral_x_grids, "dihedrals"),
    ]


def _spline_type_labels(spline_model):
    """Return per-category label dicts for a SplineModel."""
    species = spline_model._cg_species
    return {
        "bonds":      _reconstruct_type_labels(spline_model._bond_terms,     species, 2),
        "angles":     _reconstruct_type_labels(spline_model._angle_terms,    species, 3),
        "dihedrals":  _reconstruct_type_labels(spline_model._dihedral_terms, species, 4),
        "non_bonded": {
            tid: "(" + ",".join(str(s) for s in pair) + ")"
            for tid, pair in enumerate(spline_model._nb_species_pairs)
        },
    }


def plot_splines(
    spline_model,
    params: dict,
    output_dir: str,
    filename: str = "splines_learned.png",
    bi_priors: dict | None = None,
) -> str:
    """Plot all learned splines after training.

    Creates one subplot per non-empty interaction category (non-bonded, bonds,
    angles, dihedrals).  Each type within a category is drawn as a separate
    curve with knot positions marked as small dots.  Style mirrors
    :func:`plot_1dfunc_priors`.

    Args:
        spline_model:  :class:`~cgbench.core.prior.SplineModel` instance used
                       during training (provides grids and topology).
        params:        Trained parameter pytree (keys ``bonds``, ``angles``,
                       ``dihedrals``, ``non_bonded``).
        output_dir:    Directory to save the figure.
        filename:      Output filename (default ``"splines_learned.png"``).

    Returns:
        Absolute path to the saved figure, or ``""`` if nothing to plot.
    """
    setup_plot_style()

    type_labels = _spline_type_labels(spline_model)

    active = [
        (xlabel, title, x_grids, key)
        for xlabel, title, x_grids, key in _spline_panel_specs(spline_model)
        if len(type_labels[key]) > 0 and np.asarray(params[key]).shape[0] > 0
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

    for ax, (xlabel, title, x_grids, key) in zip(axes, active):
        p_arr = np.asarray(params[key])
        labels = type_labels[key]
        n_types = p_arr.shape[0]
        color_cycle = (colors_extended * math.ceil(n_types / max(1, len(colors_extended))))[:n_types]

        _overlay_bi_reference(ax, key, bi_priors, labels, color_cycle, n_types)

        for tid in range(n_types):
            x_grid = np.asarray(x_grids[tid])
            fine_x, fine_U = _eval_spline(x_grid, p_arr[tid])
            ax.plot(fine_x, fine_U, color=color_cycle[tid], linewidth=1.8,
                    label=labels.get(tid, str(tid)))
            knot_U = p_arr[tid] - p_arr[tid].min()
            ax.scatter(x_grid, knot_U, color=color_cycle[tid], s=15, zorder=3, alpha=0.6)

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("$U$ (kJ/mol)", fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=11)

        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ncol = max(1, math.ceil(len(handles) / 8))
            ax.legend(
                handles, labels_,
                fontsize=max(6, 10 - ncol),
                ncol=ncol,
                loc="best",
                framealpha=0.7,
                borderpad=0.4,
                labelspacing=0.3,
                handlelength=1.2,
            )

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_MULTI_LINESTYLES = ["-", "--", "-."]

_BI_X_KEY = {"bonds": "r_grid", "angles": "theta_grid", "dihedrals": "phi_grid",
             "non_bonded": "r_grid"}


def _overlay_bi_reference(ax, key, bi_priors, labels, color_cycle, n_types):
    """Overlay Boltzmann-inverted PMF curves on a spline panel (dotted lines)."""
    if bi_priors is None or key not in bi_priors or key not in _BI_X_KEY:
        return
    bi_data = bi_priors[key]
    x_key_bi = _BI_X_KEY[key]
    bi_lookup = {_safe_label(k): v for k, v in bi_data.items()}
    for tid in range(n_types):
        type_label = labels.get(tid, str(tid))
        bval = bi_lookup.get(type_label)
        if bval is None:
            continue
        x_bi = np.asarray(bval[x_key_bi])
        U_bi = np.asarray(bval["U"])
        finite = ~np.isnan(U_bi)
        if not np.any(finite):
            continue
        U_bi_plot = np.where(finite, U_bi, np.nanmax(U_bi[finite]))
        U_bi_plot -= np.nanmin(U_bi_plot[finite])
        ax.plot(x_bi, U_bi_plot, color=color_cycle[tid], linestyle=":",
                linewidth=1.5, alpha=0.7, label=f"{type_label} [BI ref]")


def plot_splines_multi(
    spline_model,
    params_list: list,
    output_dir: str,
    filename: str = "splines_learned.png",
    bi_priors: dict | None = None,
) -> str:
    """Plot learned splines for multiple parameter sets overlaid in each panel.

    Useful for comparing e.g. best FM params, best NVE params, and final params
    after a combined force-matching + NVE smoothing run.

    Within each panel (one per non-empty interaction category), every interaction
    type gets one color.  The different parameter sets are distinguished by
    linestyle (solid / dashed / dash-dot), so divergence between training
    variants is immediately visible.  Legend entries are labelled
    ``"<type> [<params_name>]"``.

    Args:
        spline_model:  :class:`~cgbench.core.prior.SplineModel` used during
                       training (provides grids and topology).
        params_list:   List of ``(label, params_dict)`` tuples, one per param
                       set to overlay.  ``label`` appears in the legend.
        output_dir:    Directory to save the figure.
        filename:      Output filename (default ``"splines_learned.png"``).

    Returns:
        Absolute path to the saved figure, or ``""`` if nothing to plot.
    """
    if not params_list:
        return ""

    setup_plot_style()

    type_labels = _spline_type_labels(spline_model)
    first_p = params_list[0][1]

    active = [
        (xlabel, title, x_grids, key)
        for xlabel, title, x_grids, key in _spline_panel_specs(spline_model)
        if len(type_labels[key]) > 0 and np.asarray(first_p[key]).shape[0] > 0
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

    for ax, (xlabel, title, x_grids, key) in zip(axes, active):
        labels = type_labels[key]
        n_types = np.asarray(first_p[key]).shape[0]
        color_cycle = (colors_extended * math.ceil(n_types / max(1, len(colors_extended))))[:n_types]

        _overlay_bi_reference(ax, key, bi_priors, labels, color_cycle, n_types)

        for tid in range(n_types):
            x_grid = np.asarray(x_grids[tid])
            type_label = labels.get(tid, str(tid))

            for pi, (pname, params) in enumerate(params_list):
                p_arr = np.asarray(params[key])
                ls = _MULTI_LINESTYLES[pi % len(_MULTI_LINESTYLES)]
                fine_x, fine_U = _eval_spline(x_grid, p_arr[tid])
                ax.plot(
                    fine_x, fine_U,
                    color=color_cycle[tid],
                    linestyle=ls,
                    linewidth=1.8,
                    label=f"{type_label} [{pname}]",
                )

        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("$U$ (kJ/mol)", fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.tick_params(labelsize=11)

        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ncol = max(1, math.ceil(len(handles) / 8))
            ax.legend(
                handles, labels_,
                fontsize=max(6, 10 - ncol),
                ncol=ncol,
                loc="best",
                framealpha=0.7,
                borderpad=0.4,
                labelspacing=0.3,
                handlelength=1.8,
            )

    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
