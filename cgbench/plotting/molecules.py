"""
High-level molecule visualization routines.
"""

import os
import json
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
from jax import vmap
from jax import numpy as jnp
from chemtrain import quantity

from cgbench.utils.io import load_trajectory, prepare_output_dir
from cgbench.utils.geometry import (
    init_dihedral_fn,
    init_angle_fn,
    compute_atom_distance,
    periodic_displacement,
)
from cgbench.utils.chains import compute_line_locations, split_into_chains
from cgbench.plotting.timeseries import (
    plot_energy_and_kT,
    plot_dist_series,
    plot_dihedrals,
)
from cgbench.plotting.structural import (
    plot_rdf,
    plot_ramachandran,
    plot_helicity_gyration,
    plot_histogram_free_energy,
    determine_free_energy_scale,
)
from cgbench.plotting.distributions import plot_1d_dihedral, plot_1d_angle, plot_1d_bond
from cgbench.utils.structural import calculate_rdf


def plot_hexane_angle(
    angle_indices_all: list[tuple[int, int, int]],
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    outpath: str,
    disp_fn: callable,
) -> None:
    """
    Plot KDE of bond angles across all hexane molecules.

    Calculates angle values for each frame and molecule, then produces two
    density plots: full range [0, π] and zoomed [1.6, π].
    """
    angle_fn = init_angle_fn(disp_fn, angle_indices_all)
    angles_ref = angle_fn(ref_coords)
    angles_traj = angle_fn(traj_coords)
    ref_flat = np.radians(np.concatenate(angles_ref))
    traj_flat = np.radians(np.concatenate(angles_traj))
    ref_clean = ref_flat[np.isfinite(ref_flat)]
    traj_clean = traj_flat[np.isfinite(traj_flat)]

    # Full-range KDE
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        xs = np.linspace(traj_clean.min(), traj_clean.max(), 1000)
        ax1.plot(xs, kde_t(xs), label="Trajectory KDE")
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        xsr = np.linspace(
            min(ref_clean.min(), traj_clean.min()),
            max(ref_clean.max(), traj_clean.max()),
            1000,
        )
        ax1.plot(xsr, kde_r(xsr), "--", label="Reference KDE")
    ax1.set_xlim(0, np.pi)
    ax1.set_xlabel("Angle (radians)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Bond Angle KDE: Trajectory vs Reference (Full Range)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(outpath, "bond_angles_density.png"), dpi=300)
    plt.close(fig1)

    # Zoomed KDE
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        ax2.plot(xs, kde_t(xs), label="Trajectory KDE")
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        ax2.plot(xsr, kde_r(xsr), "--", label="Reference KDE")
    ax2.set_xlim(1.6, np.pi)
    ax2.set_xlabel("Angle (radians)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("Bond Angle KDE: Trajectory vs Reference (Zoomed: 1.6 to π)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, "bond_angles_density_zoomed.png"), dpi=300)
    plt.close(fig2)


def plot_hex_dihedral(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    dihedral_indices_all: list[tuple[int, int, int, int]],
    outpath: str,
) -> None:
    """
    Plot dihedral angle distributions for all hexane CG dihedrals.

    Computes dihedral angles for every molecule and frame, then overlays
    reference vs simulation histograms on a single panel.
    """
    hex_fn = init_dihedral_fn(disp_fn, dihedral_indices_all)
    CG_angles = np.concatenate(hex_fn(traj_coords))
    AT_angles = np.concatenate(hex_fn(ref_coords))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plot_1d_dihedral(ax, [AT_angles, CG_angles], ["AT", "Simulation"], bins=60, degrees=True)
    ax.set_title("Dihedral angle (all molecules)")
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "dihedral_angle.png"), dpi=300)
    plt.close(fig)


def plot_hexane_two_site_bond_distribution(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    bond_indices_all: list[tuple[int, int]],
    outpath: str,
) -> None:
    """Plot two-site hexane bond-length distribution (reference vs simulation)."""
    ref_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in bond_indices_all]
    traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in bond_indices_all]

    ref_flat = np.concatenate(ref_dists)
    traj_flat = np.concatenate(traj_dists)
    ref_flat = ref_flat[np.isfinite(ref_flat)]
    traj_flat = traj_flat[np.isfinite(traj_flat)]

    if ref_flat.size == 0 or traj_flat.size == 0:
        print("Skipping two-site bond distribution: no finite bond values available.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_1d_bond(
        ax,
        [ref_flat, traj_flat],
        ["Reference", "Simulation"],
        bins=120,
        xlabel="Bond length (nm)",
        mode="single",
    )
    ax.set_title("Hexane Two-Site Bond Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "hexane_two_site_bond_distribution.png"), dpi=300)
    plt.close(fig)


def plot_bond_angle_correlation(
    ref_coords, traj_coords, angle_idcs, bond_idcs, disp_fn, outpath
):
    """Plot 2D histogram of bond angles vs bond distances."""
    hex_angle_fn = init_angle_fn(disp_fn, angle_idcs)
    angles_ref = hex_angle_fn(ref_coords)
    angles_traj = hex_angle_fn(traj_coords)

    dists_ref = [compute_atom_distance(ref_coords, a, b, disp_fn) for a, b in bond_idcs]
    dists_traj = [
        compute_atom_distance(traj_coords, a, b, disp_fn) for a, b in bond_idcs
    ]

    angles_ref_flat = np.radians(np.concatenate(angles_ref))
    angles_traj_flat = np.radians(np.concatenate(angles_traj))
    dists_ref_flat = np.concatenate(dists_ref)
    dists_traj_flat = np.concatenate(dists_traj)

    # determine how many bonds per angle
    n_angles = len(angle_idcs)
    n_distances = len(bond_idcs)
    if n_angles == 0 or (n_distances % n_angles) != 0:
        raise ValueError(
            f"Expected number of distances ({n_distances}) to be a multiple of number of angles ({n_angles})"
        )
    repeat_factor = n_distances // n_angles

    # repeat angles to align with distances
    angles_ref_rep = np.repeat(angles_ref_flat, repeat_factor)
    angles_traj_rep = np.repeat(angles_traj_flat, repeat_factor)

    # drop any pairs where either is NaN
    mask_ref = np.isfinite(angles_ref_rep) & np.isfinite(dists_ref_flat)
    mask_traj = np.isfinite(angles_traj_rep) & np.isfinite(dists_traj_flat)

    angles_ref_final = angles_ref_rep[mask_ref]
    dists_ref_final = dists_ref_flat[mask_ref]
    angles_traj_final = angles_traj_rep[mask_traj]
    dists_traj_final = dists_traj_flat[mask_traj]

    # make 2D histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hist_ref, xedges_ref, yedges_ref = np.histogram2d(
        angles_ref_final, dists_ref_final, bins=50, density=True
    )
    hist_traj, xedges_traj, yedges_traj = np.histogram2d(
        angles_traj_final, dists_traj_final, bins=50, density=True
    )
    # Plot reference histogram
    extent_ref = [xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]]
    im1 = ax1.imshow(
        hist_ref.T, origin="lower", extent=extent_ref, aspect="auto", cmap="plasma"
    )
    ax1.set_xlabel("Bond Angle (radians)")
    ax1.set_ylabel("Bond Distance (nm)")
    ax1.set_title("Reference: Bond vs Angle")

    extent_traj = [xedges_traj[0], xedges_traj[-1], yedges_traj[0], yedges_traj[-1]]
    im2 = ax2.imshow(
        hist_traj.T, origin="lower", extent=extent_traj, aspect="auto", cmap="plasma"
    )
    ax2.set_xlabel("Bond Angle (radians)")
    ax2.set_ylabel("Bond Distance (nm)")
    ax2.set_title("Trajectory: Bond vs Angle")
    plt.colorbar(im2, ax=ax2, label="Density")

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "bond_angle_correlation_heatmap.png"), dpi=300)
    plt.close(fig)


def _plot_martini3_capped_angles(
    angle_defs: list[tuple],
    angle_labels: list[str],
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    outpath: str,
) -> None:
    """
    Plot angle distributions for martini3 capped dipeptides.

    Produces one subplot per angle definition, overlaying reference and
    simulation KDE distributions.
    """
    n = len(angle_defs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, triple, label in zip(axes, angle_defs, angle_labels):
        angle_fn = init_angle_fn(disp_fn, [list(triple)])
        # init_angle_fn returns degrees; squeeze (1, n_frames) → (n_frames,)
        ref_ang = np.asarray(angle_fn(ref_coords)).ravel()
        traj_ang = np.asarray(angle_fn(traj_coords)).ravel()
        ref_ang = ref_ang[np.isfinite(ref_ang)]
        traj_ang = traj_ang[np.isfinite(traj_ang)]
        # degrees=False: values already in degrees, no rad2deg conversion
        plot_1d_angle(
            ax,
            [ref_ang, traj_ang],
            ["Reference", "Simulation"],
            degrees=False,
            xlabel=f"$\\Theta$ (deg)\n{label}",
        )
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "martini3_angles.png"), dpi=300)
    plt.close(fig)


def vis_capped_ala(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize alanine dipeptide trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 8, 14]
        psi_indices = [6, 8, 14, 16]
        pairs = [(4, 6), (6, 8), (8, 14)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "heavyOnly": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "heavyOnlyMap2": ([1, 3, 4, 6], [3, 4, 6, 8], [(1, 3), (3, 4), (4, 6)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreSingle": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
            "coreBetaMap2": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
            "coreBetaSingle": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 4)]),
            # martini3: 4 beads ACE(0)-BB(1)-SC1(2)-NME(3)
            # phi=psi=improper dihedral NME-BB-SC1-ACE; pairs are BB-centred bonds
            "martini3": ([3, 1, 2, 0], [3, 1, 2, 0], [(0, 1), (1, 2), (1, 3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    if cg_map == "martini3":
        # Angles around the central BB bead: ACE-BB-SC1, ACE-BB-NME, SC1-BB-NME
        _plot_martini3_capped_angles(
            [(0, 1, 2), (0, 1, 3), (2, 1, 3)],
            ["ACE-BB-SC1", "ACE-BB-NME", "SC1-BB-NME"],
            ref_coords, traj_coords, disp_fn, outpath,
        )
    else:
        plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def vis_hexane(
    traj_path,
    config,
    type="AT",
    name="Simulation",
    dataset=None,
    cg_map="six-site",
    nmol=100,
):
    """Visualize hexane trajectory."""
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    config = json.load(
        open(os.path.join(os.path.dirname(traj_path), "traj_config.json"), "r")
    )
    line_locs = compute_line_locations(config)

    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    # Initialize variables
    cg_dihedral_idcs = None
    CG_angle_idcs = None

    # mapping
    if type == "AT":
        sites_per_mol = 20
        at_dihedral_idcs = [4, 7, 10, 13]
        CC_pairs = [(0, 4), (4, 7), (7, 10), (10, 13), (13, 16)]
        CH_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (4, 5),
            (4, 6),
            (7, 8),
            (7, 9),
            (10, 11),
            (10, 12),
            (13, 14),
            (13, 15),
            (16, 17),
            (16, 18),
            (16, 19),
        ]
        ref_coords = np.concatenate(
            [
                dataset.dataset_X["training"]["R"],
                dataset.dataset_X["validation"]["R"],
                dataset.dataset_X["testing"]["R"],
            ],
            axis=0,
        )

    else:
        definitions = {
            "two-site": ([(0, 1)], 2, None, None),
            "two-site-Map2": ([(0, 1)], 2, None, None),
            "three-site": ([(0, 1), (1, 2)], 3, [(0, 1, 2)], None),
            "three-site-Map1": ([(0, 1), (1, 2)], 3, [(0, 1, 2)], None),
            "four-site": ([(0, 1), (1, 2), (2, 3)], 4, None, [0, 1, 2, 3]),
            "six-site": ([(1, 2), (2, 3), (3, 4)], 6, None, [1, 2, 3, 4]),
            "six-site-Map2": ([(1, 2), (2, 3), (3, 4)], 6, None, [1, 2, 3, 4]),
        }

        if cg_map not in definitions:
            raise ValueError(
                f"Unknown cg_map: {cg_map}. Available options: {list(definitions.keys())}"
            )

        CC_pairs, sites_per_mol, CG_angle_idcs, cg_dihedral_idcs = definitions[cg_map]
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )

    actual_nmol = config.get("nmol", nmol)
    plot_energy_and_kT(aux, line_locs, outpath)

    if "epot" in aux:
        epot = aux["epot"]
        if np.any(epot > 1000):
            first_explosion = np.where(epot > 1000)[0][0]
            traj_coords = traj_coords[:first_explosion]
            aux = {
                k: v[:first_explosion]
                for k, v in aux.items()
                if isinstance(v, (np.ndarray, list))
            }
            print(f"Energy exceeded 10^4 at frame {first_explosion}, truncating trajectory.")

    CC_all = []
    Dihedrals_idcs_all = []
    Angles_idcs_all = []

    for m in range(actual_nmol):
        offset = m * sites_per_mol
        CC_all.extend([(a + offset, b + offset) for a, b in CC_pairs])

        if cg_dihedral_idcs is not None:
            Dihedrals_idcs_all.extend(
                [
                    (a + offset, b + offset, c + offset, d + offset)
                    for a, b, c, d in [cg_dihedral_idcs]
                ]
            )

        if CG_angle_idcs is not None:
            Angles_idcs_all.extend(
                [(a + offset, b + offset, c + offset) for a, b, c in CG_angle_idcs]
            )

    if type == "AT":
        CH_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            CH_all.extend([(a + offset, b + offset) for a, b in CH_pairs])

        fig_ch, ax_ch = plt.subplots(figsize=(10, 5))
        for a, b in CH_all:
            d = compute_atom_distance(traj_coords, a, b, disp_fn)
            ax_ch.plot(d, alpha=0.1)
        ax_ch.set_title("AT CH distances (all molecules)")
        ax_ch.set_xlabel("Time step")
        ax_ch.set_ylabel("Distance")
        plt.tight_layout()
        fig_ch.savefig(os.path.join(outpath, "AT_CH_distances_all.png"), dpi=300)
        plt.close(fig_ch)

        Dihedral_AT_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            Dihedral_AT_all.extend(
                [
                    (a + offset, b + offset, c + offset, d + offset)
                    for a, b, c, d in [at_dihedral_idcs]
                ]
            )

        plot_hex_dihedral(ref_coords, traj_coords, disp_fn, Dihedral_AT_all, outpath)

    elif "two-site" in cg_map:
        plot_hexane_two_site_bond_distribution(
            ref_coords, traj_coords, disp_fn, CC_all, outpath
        )

    elif cg_map == "three-site":
        plot_hexane_angle(Angles_idcs_all, ref_coords, traj_coords, outpath, disp_fn)
        plot_bond_angle_correlation(
            ref_coords, traj_coords, Angles_idcs_all, CC_all, disp_fn, outpath
        )

    elif "six-site" in cg_map or "four-site" in cg_map:
        if cg_dihedral_idcs is not None:
            plot_hex_dihedral(
                ref_coords, traj_coords, disp_fn, Dihedrals_idcs_all, outpath
            )


def plot_helicity_gyration_ala15(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    ca_indices: list[int],
    line_locs: list[int],
    outpath: str,
    name: str = "Simulation",
) -> tuple:
    """
    Plot helicity and gyration radius analysis for ALA15.

    Creates three comparison plots:
    1. Helicity vs Radius of Gyration (free energy surface, ref and sim side by side)
    2. Handedness vs Helicity (free energy surface, ref and sim side by side)
    3. Helicity and Gyration radius over time (overlaid with concatenated chains)
    """
    from cgbench.utils import structural as struct_utils

    # Extract CA atoms from reference and simulation (mapping-dependent indices).
    ref_ca = ref_coords[:, ca_indices, :]
    traj_ca = traj_coords[:, ca_indices, :]

    def _compute_valid_metrics(coords_ca: np.ndarray) -> tuple[np.ndarray, ...]:
        valid_mask = np.isfinite(coords_ca).all(axis=(1, 2))
        coords_valid = coords_ca[valid_mask]
        rg_valid = np.asarray(
            struct_utils.radius_of_gyration_vectorized(coords_valid, disp_fn)
        ).ravel()
        helicity_valid = np.asarray(
            struct_utils.helicity_vectorized(coords_valid, disp_fn)
        ).ravel()
        handedness_valid = np.asarray(
            struct_utils.xi_norm_vectorized(coords_valid, disp_fn)
        ).ravel()

        rg_full = np.full(coords_ca.shape[0], np.nan, dtype=float)
        helicity_full = np.full(coords_ca.shape[0], np.nan, dtype=float)
        handedness_full = np.full(coords_ca.shape[0], np.nan, dtype=float)
        rg_full[valid_mask] = rg_valid
        helicity_full[valid_mask] = helicity_valid
        handedness_full[valid_mask] = handedness_valid

        return (
            rg_valid,
            helicity_valid,
            handedness_valid,
            rg_full,
            helicity_full,
            handedness_full,
            valid_mask,
        )

    # Calculate metrics for valid frames in each dataset.
    (
        ref_rg,
        ref_helicity,
        ref_handedness,
        _,
        _,
        _,
        _,
    ) = _compute_valid_metrics(ref_ca)
    (
        traj_rg,
        traj_helicity,
        traj_handedness,
        traj_rg_full,
        traj_helicity_full,
        _,
        _,
    ) = _compute_valid_metrics(traj_ca)

    # Plot 1: Helicity vs Radius of Gyration (reference and simulation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Determine common free energy scale
    scale = determine_free_energy_scale(
        [ref_rg, traj_rg],
        [ref_helicity, traj_helicity],
        300.0 * quantity.kb,
        bins=200
    )

    # Reference
    plot_histogram_free_energy(
        ax1,
        ref_rg,
        ref_helicity,
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$R_g$ (nm)",
        ylabel_text="$Q_{hel}$",
        show_ylabel=True,
        ylim=(-0.001, 1.0),
        xlim=(0.4, 2.5),
        scale=scale,
        show_yticks=True,
        bins=200,
        title="Reference",
    )

    # Simulation
    plot_histogram_free_energy(
        ax2,
        traj_rg,
        traj_helicity,
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$R_g$ (nm)",
        ylabel_text="$Q_{hel}$",
        show_ylabel=False,
        ylim=(-0.001, 1.0),
        xlim=(0.4, 2.5),
        scale=scale,
        show_yticks=False,
        bins=200,
        title=name,
        legend=True,
    )

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "helicity_vs_radius_gyration.png"), dpi=300)
    plt.close(fig)

    # Plot 2: Handedness vs Helicity (reference and simulation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    scale = determine_free_energy_scale(
        [ref_handedness, traj_handedness],
        [ref_helicity, traj_helicity],
        300.0 * quantity.kb,
        bins=200
    )

    # Reference
    plot_histogram_free_energy(
        ax1,
        ref_handedness,
        ref_helicity,
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$\\chi_{hel}$",
        ylabel_text="$Q_{hel}$",
        show_ylabel=True,
        ylim=(-0.001, 1.0),
        xlim=(-0.06, 0.06),
        scale=scale,
        show_yticks=True,
        bins=200,
        title="Reference",
    )
    ax1.axvline(0, color='k', linestyle='--', linewidth=1)

    # Simulation
    plot_histogram_free_energy(
        ax2,
        traj_handedness,
        traj_helicity,
        kbt=300.0 * quantity.kb,
        is_angular=False,
        xlabel="$\\chi_{hel}$",
        ylabel_text="$Q_{hel}$",
        show_ylabel=False,
        ylim=(-0.001, 1.0),
        xlim=(-0.06, 0.06),
        scale=scale,
        show_yticks=False,
        bins=200,
        title=name,
        legend=True,
    )
    ax2.axvline(0, color='k', linestyle='--', linewidth=1)

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "handedness_vs_helicity.png"), dpi=300)
    plt.close(fig)

    # Plot 3: Helicity and Gyration radius over time (concatenated chains)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    frames = np.arange(traj_helicity_full.shape[0])
    ax1.plot(
        frames,
        traj_helicity_full,
        label="Helicity Content",
        color="blue",
        linewidth=2,
        alpha=0.7,
    )
    ax1.set_ylabel("Helicity Content ($Q_{hel}$)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_xlabel("Frame")
    ax1.set_title(f"Helicity and Radius of Gyration Over Time ({name})")

    ax2 = ax1.twinx()
    ax2.plot(
        frames,
        traj_rg_full,
        label="Radius of Gyration",
        color="orange",
        linewidth=2,
        alpha=0.7,
    )
    ax2.set_ylabel("Radius of Gyration (nm)", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    # Mark chain boundaries with red vertical lines (same style as kT plot).
    for loc in line_locs:
        ax1.axvline(x=loc, color="r", linestyle="-", alpha=0.5)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "helicity_gyration_timeseries.png"), dpi=300)
    plt.close(fig)

    # Plot 4: overlaid chains (relative frame) for trajectory helicity and gyration.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    traj_helicity_chains = split_into_chains(traj_helicity_full, line_locs)
    traj_rg_chains = split_into_chains(traj_rg_full, line_locs)

    for i, chain_vals in enumerate(traj_helicity_chains):
        valid = np.isfinite(chain_vals)
        if valid.any():
            ax1.plot(
                np.where(valid)[0],
                chain_vals[valid],
                alpha=0.7,
                label=f"Chain {i+1}",
            )
    ax1.set_title(f"{name} Helicity - Overlaid chains")
    ax1.set_xlabel("Relative frame")
    ax1.set_ylabel("$Q_{hel}$")

    for i, chain_vals in enumerate(traj_rg_chains):
        valid = np.isfinite(chain_vals)
        if valid.any():
            ax2.plot(
                np.where(valid)[0],
                chain_vals[valid],
                alpha=0.7,
                label=f"Chain {i+1}",
            )
    ax2.set_title(f"{name} $R_g$ - Overlaid chains")
    ax2.set_xlabel("Relative frame")
    ax2.set_ylabel("$R_g$ (nm)")

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "helicity_gyration_overlaid_chains.png"), dpi=300)
    plt.close(fig)

    return ref_rg, ref_helicity, ref_handedness, traj_rg, traj_helicity, traj_handedness


def vis_capped_ala15(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize ALA15 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    plot_energy_and_kT(aux, line_locs, outpath)

    if type == "AT":
        raise NotImplementedError("AT visualization for ALA15 is not implemented yet.")
    else:
        maps = {
            "CA": {
                "phi_indices": [0, 1, 2, 3],
                "psi_indices": [1, 2, 3, 4],
                "pairs": [(0, 1), (1, 2), (2, 3)],
                "ca_indices": list(range(15)),
            },
            "CA-Map2": {
                "phi_indices": [0, 1, 2, 3],
                "psi_indices": [1, 2, 3, 4],
                "pairs": [(0, 1), (1, 2), (2, 3)],
                "ca_indices": list(range(15)),
            },
            "CA-Map3": {
                "phi_indices": [0, 1, 2, 3],
                "psi_indices": [1, 2, 3, 4],
                "pairs": [(0, 1), (1, 2), (2, 3)],
                "ca_indices": list(range(15)),
            },
            "CA-Map4": {
                "phi_indices": [0, 1, 2, 3],
                "psi_indices": [1, 2, 3, 4],
                "pairs": [(0, 1), (1, 2), (2, 3)],
                "ca_indices": list(range(15)),
            },
            "coreMap2": {
                "phi_indices": [3, 4, 5, 6],
                "psi_indices": [4, 5, 6, 7],
                "pairs": [(4, 5), (5, 6), (6, 7)],
                "ca_indices": [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44],
            },
            "coreBetaMap2": {
                "phi_indices": [4, 5, 6, 8],
                "psi_indices": [5, 6, 8, 9],
                "pairs": [(4, 5), (5, 6), (6, 8)],
                "ca_indices": [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58],
            },
            # martini3: 32 beads — ACE(0), then [BB_i(1+2i), SC1_i(2+2i)] x15, NME(31)
            # BB beads at indices 1,3,5,...,29; use them as the "CA" proxy for helicity
            "martini3": {
                "phi_indices": [1, 3, 5, 7],
                "psi_indices": [3, 5, 7, 9],
                "pairs": [(1, 3), (3, 5), (5, 7)],
                "ca_indices": [1 + 2 * i for i in range(15)],
            },
        }
        if cg_map not in maps:
            raise ValueError(f"Unknown cg_map: {cg_map}. Available options: {list(maps.keys())}")

        mapping = maps[cg_map]
        phi_indices = mapping["phi_indices"]
        psi_indices = mapping["psi_indices"]
        pairs = mapping["pairs"]
        ca_indices = mapping["ca_indices"]

        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )

    if len(phi_indices) > 0:
        ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
        AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
        Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

        plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
        plot_ramachandran(
            AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath
        )

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)

    # Plot helicity and gyration radius analysis
    plot_helicity_gyration_ala15(
        ref_coords, traj_coords, disp_fn, ca_indices, line_locs, outpath, name
    )


def vis_capped_pro(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize PRO2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "heavyOnly": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "heavyOnlyMap2": ([1, 3, 7, 8], [3, 7, 8, 10], [(1, 3), (3, 7), (7, 8)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 3, 4], [1, 3, 4, 5], [(0, 1), (1, 3), (3, 4)]),
            "coreBetaMap2": ([0, 1, 3, 4], [1, 3, 4, 5], [(0, 1), (1, 3), (3, 4)]),
            # martini3: 4 beads ACE(0)-BB_PRO(1)-SC1_PRO(2)-NME(3)
            "martini3": ([3, 1, 2, 0], [3, 1, 2, 0], [(0, 1), (1, 2), (1, 3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    if cg_map == "martini3":
        # Angles around the central BB bead: ACE-BB-SC1, ACE-BB-NME, SC1-BB-NME
        _plot_martini3_capped_angles(
            [(0, 1, 2), (0, 1, 3), (2, 1, 3)],
            ["ACE-BB-SC1", "ACE-BB-NME", "SC1-BB-NME"],
            ref_coords, traj_coords, disp_fn, outpath,
        )
    else:
        plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def vis_capped_gly(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize GLY2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 8, 11]
        psi_indices = [6, 8, 11, 13]
        pairs = [(4, 6), (6, 8), (8, 11)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "heavyOnly": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "heavyOnlyMap2": ([1, 3, 4, 5], [3, 4, 5, 7], [(1, 3), (3, 4), (4, 5)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            # martini3: 3 beads ACE(0)-BB_GLY(1)-NME(2) — no sidechain, no dihedral
            "martini3": ([], [], [(0, 1), (1, 2)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)

    if phi_indices:
        ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
        AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
        Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)
        plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
        plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)
    elif cg_map == "martini3":
        # Gly2 martini3: only backbone angle ACE-BB-NME
        _plot_martini3_capped_angles(
            [(0, 1, 2)],
            ["ACE-BB-NME"],
            ref_coords, traj_coords, disp_fn, outpath,
        )


def vis_capped_thr(
    traj_path, config, type="AT", name="Simulation", dataset=None, cg_map="hmerged"
):
    """Visualize THR2 trajectory."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == "AT":
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate(
            [
                dataset.dataset_U["training"]["R"],
                dataset.dataset_U["validation"]["R"],
                dataset.dataset_U["testing"]["R"],
            ],
            axis=0,
        )
    else:
        maps = {
            "hmerged": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "heavyOnly": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "heavyOnlyMap2": ([1, 3, 5, 8], [3, 5, 8, 10], [(1, 3), (3, 5), (5, 8)]),
            "core": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreMap2": ([0, 1, 2, 3], [1, 2, 3, 4], [(0, 1), (1, 2), (2, 3)]),
            "coreBeta": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 3)]),
            "coreBetaMap2": ([0, 1, 2, 4], [1, 2, 4, 5], [(0, 1), (1, 2), (2, 3)]),
            # martini3: 4 beads ACE(0)-BB_THR(1)-SC1_THR(2)-NME(3)
            "martini3": ([3, 1, 2, 0], [3, 1, 2, 0], [(0, 1), (1, 2), (1, 3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate(
            [dataset.cg_dataset_U[s]["R"] for s in splits],
            axis=0,
        )
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pairs]
    Traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)

    if cg_map == "martini3":
        # Angles around the central BB bead: ACE-BB-SC1, ACE-BB-NME, SC1-BB-NME
        _plot_martini3_capped_angles(
            [(0, 1, 2), (0, 1, 3), (2, 1, 3)],
            ["ACE-BB-SC1", "ACE-BB-NME", "SC1-BB-NME"],
            ref_coords, traj_coords, disp_fn, outpath,
        )
    else:
        plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.0 * quantity.kb, outpath)


def _compute_rmsd_to_ref0(
    coords: np.ndarray,
    ref0: np.ndarray,
    ca_indices: list[int],
    displacement_fn,
) -> np.ndarray:
    """Compute per-frame CA RMSD to a fixed reference frame."""
    if len(ca_indices) == 0:
        raise ValueError("ca_indices must not be empty.")

    ca = jnp.asarray(coords[:, ca_indices, :])
    ref = jnp.asarray(ref0[ca_indices, :])

    def _frame_rmsd(frame_ca):
        disp = vmap(displacement_fn)(frame_ca, ref)
        sq = jnp.sum(disp * disp, axis=-1)
        return jnp.sqrt(jnp.mean(sq))

    return np.asarray(vmap(_frame_rmsd)(ca))


def _plot_ca_rmsd_and_pair_distances(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn,
    outpath: str,
    line_locs: list[int],
    name: str = "Simulation",
    ca_indices: list[int] | None = None,
    pair_indices: list[tuple[int, int]] | None = None,
) -> None:
    """Plot CA RMSD (to reference frame 0) and first three pair distances."""
    if ca_indices is None:
        ca_indices = list(range(min(ref_coords.shape[1], traj_coords.shape[1])))
    if pair_indices is None:
        pair_indices = [(0, 1), (1, 2), (2, 3)]

    ref0 = ref_coords[0]
    ref_rmsd = _compute_rmsd_to_ref0(ref_coords, ref0, ca_indices, disp_fn)
    traj_rmsd = _compute_rmsd_to_ref0(traj_coords, ref0, ca_indices, disp_fn)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ref_rmsd, label="Reference CA RMSD")
    ax.plot(traj_rmsd, label=f"{name} CA RMSD")
    for loc in line_locs:
        ax.axvline(x=loc, color="r", linestyle="-", alpha=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("RMSD (nm)")
    ax.set_title("CA RMSD to reference frame 0")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "ca_rmsd_ref0.png"), dpi=300)
    plt.close(fig)

    ref_dists = [compute_atom_distance(ref_coords, i, j, disp_fn) for i, j in pair_indices]
    traj_dists = [compute_atom_distance(traj_coords, i, j, disp_fn) for i, j in pair_indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (pair, d_ref, d_traj) in enumerate(zip(pair_indices, ref_dists, traj_dists), 1):
        ax.plot(d_ref, label=f"Ref d{k} {pair}", alpha=0.9)
        ax.plot(d_traj, label=f"{name} d{k} {pair}", alpha=0.9, linestyle="--")
    for loc in line_locs:
        ax.axvline(x=loc, color="r", linestyle="-", alpha=0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Distance (nm)")
    ax.set_title("First three atom-pair distances")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "first_three_pair_distances.png"), dpi=300)
    plt.close(fig)


def vis_staticframe_protein(
    traj_path, config, type="CG", name="Simulation", dataset=None, cg_map="CA"
):
    """Visualize StaticFrame protein trajectory with CA RMSD and pair distances."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    splits = ["training", "validation"]
    if "testing" in dataset.cg_dataset_U:
        splits.append("testing")
    ref_coords = np.concatenate([dataset.cg_dataset_U[s]["R"] for s in splits], axis=0)

    # For CA map each residue contributes one bead, so all indices are CA beads.
    n_sites = min(ref_coords.shape[1], traj_coords.shape[1])
    ca_indices = list(range(n_sites))
    pair_indices = [(0, 1), (1, 2), (2, 3)] if n_sites >= 4 else [
        (i, i + 1) for i in range(max(0, min(3, n_sites - 1)))
    ]

    plot_energy_and_kT(aux, line_locs, outpath)
    _plot_ca_rmsd_and_pair_distances(
        ref_coords=ref_coords,
        traj_coords=traj_coords,
        disp_fn=disp_fn,
        outpath=outpath,
        line_locs=line_locs,
        name=name,
        ca_indices=ca_indices,
        pair_indices=pair_indices,
    )


def vis_tip3p_water(
    traj_path,
    config,
    type="CG",
    name="Simulation",
    dataset=None,
    cg_map="UnitedAtom",
):
    """Visualize TIP3P water with timeseries and RDF for both water mappings."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    if type != "CG":
        raise ValueError("TIP3P-water visualisation currently supports CG trajectories only.")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)

    plot_energy_and_kT(aux, line_locs, outpath)

    traj_coords = np.asarray(traj_coords)
    if traj_coords.ndim == 4:
        traj_coords = traj_coords.reshape(-1, traj_coords.shape[-2], traj_coords.shape[-1])

    box_arr = np.asarray(box)
    box_len = float(box_arr[0, 0]) if box_arr.ndim == 2 else float(box_arr[0])

    # Compute reference RDFs for both map definitions and compare to trajectory when possible.
    map_variants = ["UnitedAtom", "HeavyAtom"]
    original_map = cg_map
    for map_name in map_variants:
        dataset.coarse_grain(map=map_name)
        splits = ["training", "validation"]
        if "testing" in dataset.cg_dataset_U:
            splits.append("testing")
        ref_coords = np.concatenate([dataset.cg_dataset_U[s]["R"] for s in splits], axis=0)
        ref_coords = np.asarray(ref_coords)
        if ref_coords.ndim == 4:
            ref_coords = ref_coords.reshape(-1, ref_coords.shape[-2], ref_coords.shape[-1])

        ref_max_frames = 20000
        traj_max_frames = 20000
        ref_stride = max(1, int(np.ceil(ref_coords.shape[0] / ref_max_frames)))
        traj_stride = max(1, int(np.ceil(traj_coords.shape[0] / traj_max_frames)))

        ref_nm = ref_coords[::ref_stride] * box_len
        trajectories = [ref_nm]
        labels = [f"Reference ({map_name})"]

        if map_name == cg_map and traj_coords.shape[1] == ref_coords.shape[1]:
            trajectories.append(traj_coords[::traj_stride] * box_len)
            labels.append(f"{name} ({map_name})")

        rdf_data, bead_combinations = calculate_rdf(
            trajectories=trajectories,
            bead_types=[1],
            sites_per_mol=1,
            box_length=box_len,
            dr=0.01,
            pair_batch_size=20_000,
            frame_batch_size=512,
        )
        plot_rdf(
            rdf_data=rdf_data,
            bead_combinations=bead_combinations,
            labels=labels,
            output_prefix=os.path.join(outpath, f"tip3p_rdf_{map_name.lower()}"),
            box_length=box_len,
            mode="single",
            save_pdf=True,
        )

    # Restore the originally requested map in the dataset object for consistency.
    dataset.coarse_grain(map=original_map)


def vis_benzene_crystal(
    traj_path,
    config,
    type="CG",
    name="Simulation",
    dataset=None,
    cg_map="three-site-adjacent",
):
    """Visualize benzene crystal trajectory with 1-1 distances and RDF."""
    print(f"Visualizing {name} trajectory at {traj_path}")

    if type != "CG":
        raise ValueError("Benzene crystal visualisation currently supports CG trajectories only.")

    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, shift_fn = periodic_displacement(box, True)

    splits = ["training", "validation"]
    if "testing" in dataset.cg_dataset_U:
        splits.append("testing")
    ref_coords = np.concatenate([dataset.cg_dataset_U[s]["R"] for s in splits], axis=0)

    # Flatten potential (n_chains, n_frames, n_atoms, 3) trajectories to frame-major 3D arrays.
    traj_coords = np.asarray(traj_coords)
    if traj_coords.ndim == 4:
        traj_coords = traj_coords.reshape(-1, traj_coords.shape[-2], traj_coords.shape[-1])

    ref_coords = np.asarray(ref_coords)
    if ref_coords.ndim == 4:
        ref_coords = ref_coords.reshape(-1, ref_coords.shape[-2], ref_coords.shape[-1])

    sites_per_mol = 3
    nmol_cfg = int(config.get("nmol", min(ref_coords.shape[1], traj_coords.shape[1]) // sites_per_mol))
    nmol = min(nmol_cfg, ref_coords.shape[1] // sites_per_mol, traj_coords.shape[1] // sites_per_mol)
    n_sites = nmol * sites_per_mol

    ref_coords = ref_coords[:, :n_sites, :]
    traj_coords = traj_coords[:, :n_sites, :]

    # Intramolecular 1-1 bead pairs (all three edges of each 3-site benzene triangle).
    pairs_11 = []
    for m in range(nmol):
        off = m * sites_per_mol
        pairs_11.extend([(off, off + 1), (off + 1, off + 2), (off, off + 2)])

    ref_dists = [np.asarray(compute_atom_distance(ref_coords, i, j, disp_fn)) for i, j in pairs_11]
    traj_dists = [np.asarray(compute_atom_distance(traj_coords, i, j, disp_fn)) for i, j in pairs_11]

    plot_energy_and_kT(aux, line_locs, outpath)

    # Plot all 1-1 bond distance traces with mean overlays.
    ref_mat = np.vstack(ref_dists)
    traj_mat = np.vstack(traj_dists)

    fig, (ax_ref, ax_traj) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for row in ref_mat:
        ax_ref.plot(row, color="tab:blue", alpha=0.05, linewidth=0.6)
    ax_ref.plot(ref_mat.mean(axis=0), color="tab:blue", linewidth=2.0, label="Mean")
    ax_ref.set_title("Reference 1-1 Bead Distances")
    ax_ref.set_xlabel("Frame")
    ax_ref.set_ylabel("Distance (nm)")
    ax_ref.legend(loc="upper right")

    for row in traj_mat:
        ax_traj.plot(row, color="tab:orange", alpha=0.05, linewidth=0.6)
    ax_traj.plot(traj_mat.mean(axis=0), color="tab:orange", linewidth=2.0, label="Mean")
    for loc in line_locs:
        if loc < traj_mat.shape[1]:
            ax_traj.axvline(x=loc, color="r", linestyle="-", alpha=0.5)
    ax_traj.set_title(f"{name} 1-1 Bead Distances")
    ax_traj.set_xlabel("Frame")
    ax_traj.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, "benzene_11_bond_distances.png"), dpi=300)
    plt.close(fig)

    # RDF for 1-1 bead pairs.
    # Use bounded frame counts and conservative batch sizes to avoid GPU OOM.
    rdf_max_frames = 20000
    rdf_pair_batch_size = 20_000
    rdf_frame_batch_size = 512
    rdf_dr = 0.01

    ref_stride = max(1, int(np.ceil(ref_coords.shape[0] / rdf_max_frames)))
    traj_stride = max(1, int(np.ceil(traj_coords.shape[0] / rdf_max_frames)))
    ref_coords_rdf = ref_coords[::ref_stride]
    traj_coords_rdf = traj_coords[::traj_stride]

    print(
        "RDF settings: "
        f"ref_frames={ref_coords_rdf.shape[0]} (stride={ref_stride}), "
        f"traj_frames={traj_coords_rdf.shape[0]} (stride={traj_stride}), "
        f"pair_batch_size={rdf_pair_batch_size}, frame_batch_size={rdf_frame_batch_size}, "
        f"dr={rdf_dr}"
    )

    box_arr = np.asarray(box)
    box_len = float(box_arr[0, 0]) if box_arr.ndim == 2 else float(box_arr[0])

    # Coordinates are fractional; convert to nm for calculate_rdf.
    ref_coords_nm = ref_coords_rdf * box_len
    traj_coords_nm = traj_coords_rdf * box_len

    rdf_data, bead_combinations = calculate_rdf(
        trajectories=[ref_coords_nm, traj_coords_nm],
        bead_types=[1, 1, 1],
        sites_per_mol=sites_per_mol,
        box_length=box_len,
        dr=rdf_dr,
        pair_batch_size=rdf_pair_batch_size,
        frame_batch_size=rdf_frame_batch_size,
    )
    plot_rdf(
        rdf_data=rdf_data,
        bead_combinations=bead_combinations,
        labels=["Reference", name],
        output_prefix=os.path.join(outpath, "benzene_rdf"),
        box_length=box_len,
        mode="single",
        save_pdf=True,
    )

    # Additional RDF: molecule COMs (one bead per benzene) for three-site-adjacent mapping.
    if cg_map == "three-site-adjacent":
        from jax import numpy as jnp
        from cgbench.core.mapping import map_dataset as pbc_map_dataset

        # Build a block-diagonal mapping: each benzene's 3 CG sites -> 1 COM site.
        com_map = np.zeros((nmol, n_sites), dtype=np.float32)
        for m in range(nmol):
            start = m * sites_per_mol
            com_map[m, start : start + sites_per_mol] = 1.0 / sites_per_mol

        ref_dummy_forces = np.zeros_like(ref_coords_rdf, dtype=np.float32)
        traj_dummy_forces = np.zeros_like(traj_coords_rdf, dtype=np.float32)

        ref_com_frac, _ = pbc_map_dataset(
            ref_coords_rdf,
            disp_fn,
            shift_fn,
            jnp.asarray(com_map),
            jnp.asarray(com_map),
            ref_dummy_forces,
        )
        traj_com_frac, _ = pbc_map_dataset(
            traj_coords_rdf,
            disp_fn,
            shift_fn,
            jnp.asarray(com_map),
            jnp.asarray(com_map),
            traj_dummy_forces,
        )

        ref_com_nm = np.asarray(ref_com_frac) * box_len
        traj_com_nm = np.asarray(traj_com_frac) * box_len

        print(
            "COM RDF settings: "
            f"ref_frames={ref_com_nm.shape[0]}, traj_frames={traj_com_nm.shape[0]}, "
            f"n_benzenes={ref_com_nm.shape[1]}"
        )

        rdf_data_com, bead_combinations_com = calculate_rdf(
            trajectories=[ref_com_nm, traj_com_nm],
            bead_types=[1],
            sites_per_mol=1,
            box_length=box_len,
            dr=rdf_dr,
            pair_batch_size=rdf_pair_batch_size,
            frame_batch_size=rdf_frame_batch_size,
        )
        plot_rdf(
            rdf_data=rdf_data_com,
            bead_combinations=bead_combinations_com,
            labels=["Reference COM", f"{name} COM"],
            output_prefix=os.path.join(outpath, "benzene_rdf_com"),
            box_length=box_len,
            mode="single",
            legend_loc="upper right",
            save_pdf=True,
        )


# Backward-compatible aliases for older call sites.
vis_ala2 = vis_capped_ala
vis_ala15 = vis_capped_ala15
vis_pro2 = vis_capped_pro
vis_gly2 = vis_capped_gly
vis_thr2 = vis_capped_thr
vis_1ubq = vis_staticframe_protein
