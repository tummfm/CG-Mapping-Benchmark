"""Test Boltzmann-inversion bonded priors for Ala2 (ACE-ALA-NME).

Loads the capped alanine dipeptide dataset and derives bonded priors for:
  1. Atomistic representation (all-atom trajectory).
  2. Coarse-grained representation with the ``coreBetaMap2`` mapping.

Prints the derived harmonic parameters and saves distribution / PMF plots to
  outputs/prior_test/
"""

import os
import sys

# Force JAX to use CPU only — must happen before any JAX/jax_md import.
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.98'

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Make the package importable when running the script directly.
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cgbench.core.dataset import Capped_Ala_Dataset
from cgbench.core.prior import BoltzmannPrior

# ---------------------------------------------------------------------------
# Constants / settings
# ---------------------------------------------------------------------------
T = 300.0            # simulation temperature [K]
N_BINS = 80          # histogram bins for Boltzmann inversion
SPLIT = "training"   # dataset split to use
OUT_DIR = os.path.join(_ROOT, "outputs", "prior_test")

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _hline(char="─", n=70):
    print(char * n)


def print_bond_priors(priors: dict, label: str = "") -> None:
    _hline()
    print(f"  Bond priors  {label}")
    _hline()
    if not priors:
        print("  (none)")
        return
    print(f"  {'pair':>12s}   {'r0 [nm]':>10s}   {'r0 [Å]':>8s}   {'k [kJ/mol/nm²]':>16s}")
    _hline("-", 70)
    for (i, j), p in priors.items():
        print(
            f"  ({i:4d},{j:4d})   {p['r0']:10.4f}   {p['r0']*10:8.4f}   {p['k']:16.1f}"
        )


def print_angle_priors(priors: dict, label: str = "") -> None:
    _hline()
    print(f"  Angle priors  {label}")
    _hline()
    if not priors:
        print("  (none)")
        return
    print(f"  {'triple':>16s}   {'θ0 [°]':>9s}   {'k [kJ/mol/rad²]':>18s}")
    _hline("-", 70)
    for (i, j, k), p in priors.items():
        print(
            f"  ({i:3d},{j:3d},{k:3d})   {np.degrees(p['theta0']):9.2f}   {p['k']:18.1f}"
        )


def print_dihedral_priors(priors: dict, label: str = "") -> None:
    _hline()
    print(f"  Dihedral priors  {label}")
    _hline()
    if not priors:
        print("  (none)")
        return
    print(f"  {'quadruple':>20s}   {'φ0 [°]':>10s}")
    _hline("-", 70)
    for (i, j, k, l), p in priors.items():
        print(f"  ({i:3d},{j:3d},{k:3d},{l:3d})   {np.degrees(p['phi0']):10.2f}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_prior_grid(
    priors: dict,
    x_key: str,
    label_fn,
    x_label: str,
    out_path: str,
    to_degrees: bool = False,
    max_panels: int = 40,
) -> None:
    """Generic 2-column figure: distribution (left) + BI potential (right)."""
    items = list(priors.items())[:max_panels]
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(10, 2.6 * n), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    for row, (key, p) in enumerate(items):
        ax_dist = axes[row, 0]
        ax_pmf = axes[row, 1]

        x = p[x_key]
        P = p["P"]
        U = p["U"]
        samples = p["samples"]

        if to_degrees:
            x = np.degrees(x)
            samples_plot = np.degrees(samples)
        else:
            samples_plot = samples

        title = label_fn(key)

        # --- Distribution ---
        ax_dist.hist(
            samples_plot, bins=N_BINS, density=True,
            color="#4C72B0", alpha=0.75, edgecolor="none", label="P(x)"
        )
        ax_dist.set_xlabel(x_label)
        ax_dist.set_ylabel("Density")
        ax_dist.set_title(f"{title}\nDistribution", fontsize=8)
        ax_dist.tick_params(labelsize=7)

        # --- Boltzmann-inverted PMF ---
        valid = ~np.isnan(U)
        ax_pmf.plot(x[valid], U[valid], color="#C44E52", lw=1.5)
        ax_pmf.set_xlabel(x_label)
        ax_pmf.set_ylabel("U  [kJ/mol]")
        ax_pmf.set_title(f"{title}\nBoltzmann inversion", fontsize=8)
        ax_pmf.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_bond_priors(priors: dict, label: str, out_path: str) -> None:
    _plot_prior_grid(
        priors,
        x_key="r_grid",
        label_fn=lambda k: f"bond {k}",
        x_label="r  [nm]",
        out_path=out_path,
    )


def plot_angle_priors(priors: dict, label: str, out_path: str) -> None:
    _plot_prior_grid(
        priors,
        x_key="theta_grid",
        label_fn=lambda k: f"angle {k}",
        x_label="θ  [°]",
        out_path=out_path,
        to_degrees=True,
    )


def plot_dihedral_priors(priors: dict, label: str, out_path: str) -> None:
    _plot_prior_grid(
        priors,
        x_key="phi_grid",
        label_fn=lambda k: f"dihedral {k}",
        x_label="φ  [°]",
        out_path=out_path,
        to_degrees=True,
    )


def plot_summary(at_priors: dict, cg_priors: dict, out_path: str) -> None:
    """Side-by-side summary: number of terms and mean parameters."""
    categories = ["bonds", "angles", "dihedrals"]
    at_counts = [len(at_priors[c]) for c in categories]
    cg_counts = [len(cg_priors[c]) for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars_at = ax.bar(x - width / 2, at_counts, width, label="Atomistic", color="#4C72B0")
    bars_cg = ax.bar(x + width / 2, cg_counts, width, label="CG coreBetaMap2", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.set_ylabel("Number of terms")
    ax.set_title("Bonded interaction terms: atomistic vs. CG")
    ax.legend()

    for bar in list(bars_at) + list(bars_cg):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.1,
            str(int(h)), ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("  Boltzmann-inversion prior test — Ala2 (ACE-ALA-NME, 300 K)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("\n[1/4] Loading Capped_Ala_Dataset (ACE-ALA-NME) …")
    ds = Capped_Ala_Dataset(train_ratio=0.7, val_ratio=0.1, shuffle=False)
    ds.load_traj()
    n_frames = ds.dataset_X[SPLIT]["R"].shape[0]
    n_atoms = ds.dataset_X[SPLIT]["R"].shape[1]
    print(f"  Frames in '{SPLIT}' split : {n_frames}")
    print(f"  Atoms per frame           : {n_atoms}")
    box_nm = np.diagonal(np.asarray(ds.box))
    print(f"  Box (nm)                  : {box_nm}")

    # ------------------------------------------------------------------
    # 2. Atomistic priors
    # ------------------------------------------------------------------
    print("\n[2/4] Computing atomistic priors …")
    prior_at = BoltzmannPrior(ds, T=T)
    at_priors = prior_at.compute_all_priors(split=SPLIT, cg=False, n_bins=N_BINS)

    print_bond_priors(at_priors["bonds"],      label="(atomistic)")
    print_angle_priors(at_priors["angles"],    label="(atomistic)")
    print_dihedral_priors(at_priors["dihedrals"], label="(atomistic)")

    print(f"\n  Summary: {len(at_priors['bonds'])} bonds,"
          f" {len(at_priors['angles'])} angles,"
          f" {len(at_priors['dihedrals'])} dihedrals")

    # ------------------------------------------------------------------
    # 3. CG priors (coreBetaMap2)
    # ------------------------------------------------------------------
    print("\n[3/4] Applying coreBetaMap2 mapping and computing CG priors …")
    ds.coarse_grain("coreBetaMap2")
    n_cg = ds.cg_dataset_X[SPLIT]["R"].shape[1]
    print(f"  CG beads                  : {n_cg}")

    prior_cg = BoltzmannPrior(ds, T=T)
    cg_priors = prior_cg.compute_all_priors(split=SPLIT, cg=True, n_bins=N_BINS)

    print_bond_priors(cg_priors["bonds"],      label="(CG coreBetaMap2)")
    print_angle_priors(cg_priors["angles"],    label="(CG coreBetaMap2)")
    print_dihedral_priors(cg_priors["dihedrals"], label="(CG coreBetaMap2)")

    print(f"\n  Summary: {len(cg_priors['bonds'])} bonds,"
          f" {len(cg_priors['angles'])} angles,"
          f" {len(cg_priors['dihedrals'])} dihedrals")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    print("\n[4/4] Saving plots …")
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_bond_priors(
        at_priors["bonds"], "Atomistic",
        os.path.join(OUT_DIR, "bonds_atomistic.png"),
    )
    plot_bond_priors(
        cg_priors["bonds"], "CG coreBetaMap2",
        os.path.join(OUT_DIR, "bonds_cg_coreBetaMap2.png"),
    )
    plot_angle_priors(
        at_priors["angles"], "Atomistic",
        os.path.join(OUT_DIR, "angles_atomistic.png"),
    )
    plot_angle_priors(
        cg_priors["angles"], "CG coreBetaMap2",
        os.path.join(OUT_DIR, "angles_cg_coreBetaMap2.png"),
    )
    plot_dihedral_priors(
        at_priors["dihedrals"], "Atomistic",
        os.path.join(OUT_DIR, "dihedrals_atomistic.png"),
    )
    plot_dihedral_priors(
        cg_priors["dihedrals"], "CG coreBetaMap2",
        os.path.join(OUT_DIR, "dihedrals_cg_coreBetaMap2.png"),
    )
    plot_summary(
        at_priors, cg_priors,
        os.path.join(OUT_DIR, "summary_counts.png"),
    )

    print("\n" + "=" * 70)
    print(f"  All plots saved to  {OUT_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
