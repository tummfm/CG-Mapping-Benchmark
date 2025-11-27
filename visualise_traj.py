import os
import json
import pickle as pkl
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp, jit
from jax_md import space
from chemtrain import quantity
import utils
from scipy.stats import gaussian_kde
import jax

def prepare_output_dir(traj_path: str) -> str:
    """
    Create an output directory named 'plots' next to a trajectory file.

    Ensures that a directory called 'plots' exists alongside the given
    trajectory file path. If it does not exist, it is created.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory file.

    Returns
    -------
    str
        Path to the 'plots' directory where outputs will be saved.
    """
    outdir = os.path.join(os.path.dirname(traj_path), 'plots')
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_trajectory(traj_path: str) -> tuple[jnp.ndarray, dict]:
    """
    Load trajectory coordinates and auxiliary state from pickle files.

    Opens 'trajectory.pkl' and 'traj_state_aux.pkl' in the same directory as
    the provided path, and returns the trajectory as a JAX array along with
    auxiliary simulation data.

    Parameters
    ----------
    traj_path : str
        Path to one of the trajectory pickle files.

    Returns
    -------
    tuple[jnp.ndarray, dict]
        traj : JAX array of shape (n_frames, n_particles, 3)
            Simulation trajectory coordinates.
        aux : dict
            Auxiliary state information (energy, temperature, etc.).
    """
    base = os.path.dirname(traj_path)
    traj = pkl.load(open(os.path.join(base, 'trajectory.pkl'), 'rb'))
    aux = pkl.load(open(os.path.join(base, 'traj_state_aux.pkl'), 'rb'))
    return jnp.array(traj), aux


def periodic_displacement(box: np.ndarray, fractional: bool = False) -> tuple[callable, None]:
    """
    Create a periodic displacement function for simulating boundary conditions.

    Uses JAX MD's periodic_general to produce a function that calculates
    displacement vectors under periodic boundary conditions for a given box.

    Parameters
    ----------
    box : np.ndarray
        Array or matrix defining the simulation box.
    fractional : bool, optional
        Whether input coordinates are in fractional units, by default False.

    Returns
    -------
    tuple[callable, None]
        A function to compute periodic displacements and a placeholder None.
    """
    return space.periodic_general(box=box, fractional_coordinates=fractional)


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
        ax.axvline(x=loc, color='r', linestyle='-', alpha=0.5)


def overlay_chains(
    ax: plt.Axes,
    data: np.ndarray,
    line_locations: list[int],
    y_label: str,
    title: str,
    relative: bool = False
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
        segment = data[locs[i]:locs[i+1]]
        x_vals = range(len(segment)) if relative else range(locs[i], locs[i+1])
        ax.plot(x_vals, segment, alpha=0.7, label=f'Chain {i+1}')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_chain_lines(ax, line_locations)
    ax.legend()


def compute_line_locations(config: dict[str, float]) -> np.ndarray:
    """
    Compute chain boundary indices from simulation configuration.

    Based on total simulation time, equilibration time, output interval,
    and number of chains, returns the indices where each chain restarts.

    Parameters
    ----------
    config : dict[str, float]
        Simulation parameters including:
        - 't_total': total time (float)
        - 't_eq': equilibration time (float)
        - 'print_every': output interval (float, default 0.5)
        - 'n_chains': number of chains (int, default 1)

    Returns
    -------
    np.ndarray
        1D integer array of frame indices marking chain starts.
    """
    t_total = config['t_total']
    t_eq = config['t_eq']
    print_every = config.get('print_every', 0.5)
    n_chains = config.get('n_chains', 1)
    steps = int((t_total - t_eq) / print_every)
    arr = np.arange(0, steps * n_chains, steps)
    return arr[1:]


def split_into_chains(data: np.ndarray, line_locations: list[int]) -> np.ndarray:
    """
    Split array data into separate chains using boundary indices.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n, ...) to split along first axis.
    line_locations : list[int]
        Indices at which to split the array.

    Returns
    -------
    np.ndarray
        Array of shape (n_chains, segment_length, ...) after splitting.
    """
    segments: list[np.ndarray] = []
    start = 0
    for loc in line_locations:
        segments.append(data[start:loc])
        start = loc
    segments.append(data[start:])
    return np.array(segments)


def plot_time_series(
    traj_coords: np.ndarray,
    ref_coords: np.ndarray,
    indices: list[int],
    outpath: str,
    name: str,
    line_locations: list[int]
) -> None:
    """
    Plot Cartesian coordinates over time for selected atoms.

    Creates a two-panel figure showing reference and simulation trajectories
    for specified atom indices, with x/y/z as separate line styles.

    Parameters
    ----------
    traj_coords : np.ndarray
        Simulation coordinates, shape (n_frames, n_atoms, 3).
    ref_coords : np.ndarray
        Reference coordinates, same shape.
    indices : list[int]
        Atom indices to visualize.
    outpath : str
        Directory to save output images.
    name : str
        Label used for simulation plots.
    line_locations : list[int]
        Frame indices indicating chain breaks.
    """
    print("Plotting atom coordinate time series..")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, data, title in zip(axes, [ref_coords, traj_coords], ['Reference', name]):
        for idx in indices:
            coord = data[:, idx]
            ax.plot(coord[:, 0], label=f'Atom {idx} x')
            ax.plot(coord[:, 1], linestyle='--', label=f'Atom {idx} y')
            ax.plot(coord[:, 2], linestyle=':', label=f'Atom {idx} z')
        ax.set_title(f'{title} Atom Coordinates (indices {indices})')
        ax.set_ylabel('Coordinate')
        ax.legend(loc='upper right')
        if title == name:
            add_chain_lines(ax, line_locations)
            ax.set_xlabel('Time step')
    plt.tight_layout()
    fname = f"Atom_coords_{'_'.join(map(str, indices))}.png"
    fig.savefig(os.path.join(outpath, fname), dpi=300)
    plt.close(fig)


def plot_dist_series(
    pairs: list[tuple[int, int]],
    ref_dists: list[np.ndarray],
    traj_dists: list[np.ndarray],
    outpath: str,
    name: str,
    line_locations: list[int]
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
        ax.plot(dist, label=f'Dist {i} {pairs[i]}')
    add_chain_lines(ax, line_locations)
    ax.set_title('Reference Atom Pair Distances')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Distance')
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(outpath, 'Reference_atom_pair_distances.png'), dpi=300)
    plt.close(fig)

    # Simulation distances
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, dist in enumerate(traj_dists):
        ax2.plot(dist, label=f'Dist {i} {pairs[i]}')
    add_chain_lines(ax2, line_locations)
    ax2.set_title(f'{name} Atom Pair Distances')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Distance')
    ax2.legend(loc='upper right')
    fig2.savefig(os.path.join(outpath, f'{name}_atom_pair_distances.png'), dpi=300)
    plt.close(fig2)


def plot_dihedrals(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    outpath: str,
    line_locations: list[int]
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
    print("Plotting dihedrals..")
    # 1D dihedral distributions
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].set_title('Dihedral angle phi')
    utils.plot_1d_dihedral(axs[1], [AT_psi, Traj_psi], ['Reference', 'Simulation'], bins=60, degrees=True)
    axs[1].set_title('Dihedral angle psi')
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'Dihedrals.png'), dpi=300)
    plt.close(fig)

    # Per-chain mean/std overlay
    Traj_phi_chains = split_into_chains(Traj_phi, line_locations)
    Traj_psi_chains = split_into_chains(Traj_psi, line_locations)
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 4))
    utils.plot_1d_dihedral(axs2[0], [AT_phi], ['Reference'], bins=60, degrees=True)
    utils.plot_1d_dihedral_mean_std(axs2[0], [Traj_phi_chains], ['Simulation'], bins=60, degrees=True)
    axs2[0].set_title('Dihedral angle phi')
    utils.plot_1d_dihedral(axs2[1], [AT_psi], ['Reference'], bins=60, degrees=True)
    utils.plot_1d_dihedral_mean_std(axs2[1], [Traj_psi_chains], ['Simulation'], bins=60, degrees=True)
    axs2[1].set_title('Dihedral angle psi')
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, 'Dihedrals_mean_std.png'), dpi=300)
    plt.close(fig2)


def plot_ramachandran(
    AT_phi: np.ndarray,
    AT_psi: np.ndarray,
    Traj_phi: np.ndarray,
    Traj_psi: np.ndarray,
    kT: float,
    outpath: str
) -> None:
    """
    Plot free-energy surfaces (Ramachandran plots) for phi vs psi.

    Generates side-by-side free-energy contour plots for reference and simulation
    on the same phi-psi grid, colored by kcal/mol.

    Parameters
    ----------
    AT_phi : np.ndarray
        Reference phi angles.
    AT_psi : np.ndarray
        Reference psi angles.
    Traj_phi : np.ndarray
        Simulation phi angles.
    Traj_psi : np.ndarray
        Simulation psi angles.
    kT : float
        Thermal energy in internal units (e.g. 300*kb).
    outpath : str
        Directory to save the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1, c1 = utils.plot_histogram_free_energy(ax1, AT_phi, AT_psi, kT, degrees=True, ylabel=True)
    ax1.set_title('Reference')
    plt.colorbar(c1, ax=ax1, label='Free energy [kcal/mol]')
    ax2, c2 = utils.plot_histogram_free_energy(ax2, Traj_phi, Traj_psi, kT, degrees=True)
    ax2.set_title('Simulation')
    plt.colorbar(c2, ax=ax2, label='Free energy [kcal/mol]')
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'Ramachandran.png'), dpi=300)
    plt.close(fig)


def plot_energy_and_kT(
    aux: dict,
    line_locations: list[int],
    outpath: str
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
        ('Epot', aux.get('epot'), utils.plot_energy),
        ('kT', aux.get('kT'), utils.plot_kT),
        ('Etotal', aux.get('etot'), utils.plot_kT),
        ('Temperature', aux.get('kT'), utils.plot_T),
    ]
    for label, data, plot_fn in mapping:
        if data is None:
            continue
        # Standard time-series
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_fn(ax, data)
        add_chain_lines(ax, line_locations)
        ax.set_title(label)
        plt.tight_layout()
        fig.savefig(os.path.join(outpath, f'{label}.png'), dpi=300)
        plt.close(fig)
        # Overlaid chains
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        boundaries = [0] + list(line_locations) + [len(data)]
        exploded = 0
        for i in range(len(boundaries)-1):
            seg = np.array(data[boundaries[i]:boundaries[i+1]])
            if np.any(seg > 10000):
                exploded += 1
            mask = seg <= 10000
            if mask.any():
                ax2.plot(np.where(mask)[0], seg[mask], alpha=0.7, label=f'Chain {i+1}')
        if exploded:
            ax2.text(0.02, 0.98, f'Chains exploded: {exploded}', transform=ax2.transAxes,
                     color='red', fontweight='bold', verticalalignment='top')
        ax2.set_title(f'{label} - Overlaid chains')
        ax2.set_xlabel('Time step (0.5 ps)')
        ax2.set_ylabel(label)
        plt.tight_layout()
        fig2.savefig(os.path.join(outpath, f'{label}_overlaid.png'), dpi=300)
        plt.close(fig2)


# --------------------------------------
# Main visualization functions
# --------------------------------------
def vis_ala2(traj_path, config, type='AT', name='Simulation', dataset=None, cg_map='hmerged'):
    print(f"Visualizing {name} trajectory at {traj_path}")
    
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == 'AT':
        phi_indices = [4, 6, 7, 8]
        psi_indices = [6, 7, 8, 16]
        pairs = [(4, 6), (6, 7), (7, 8)]
        ref_coords = np.concatenate([
            dataset.dataset_X['training']['R'],
            dataset.dataset_X['validation']['R'],
            dataset.dataset_X['testing']['R']
        ], axis=0)
    else:
        maps = {
            'hmerged': ([1, 3, 4, 5], [3, 4, 5, 8], [(1,3),(3,4),(4,5)]),
            'heavyOnly': ([1, 3, 4, 5], [3, 4, 5, 8], [(1,3),(3,4),(4,5)]),
            'heavyOnlyMap2': ([1, 3, 4, 5], [3, 4, 5, 8], [(1,3),(3,4),(4,5)]),
            'core': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreMap2': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreBeta': ([0,1,2,3], [1,2,3,5], [(0,1),(1,2),(2,3)]),
            'coreBetaMap2': ([0,1,2,3], [1,2,3,5], [(0,1),(1,2),(2,3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate([
                    dataset.cg_dataset_X['training']['R'],
                    dataset.cg_dataset_X['validation']['R'],
                    dataset.cg_dataset_X['testing']['R']
                ], axis=0)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)

    ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i,j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i,j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
    
    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.*quantity.kb, outpath)

def plot_hexane_angle(
    angle_indices_all: list[tuple[int, int, int]],
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    outpath: str,
    disp_fn: callable
) -> None:
    """
    Plot KDE of bond angles across all hexane molecules.

    Calculates angle values for each frame and molecule, then produces two
    density plots: full range [0, π] and zoomed [1.6, π].

    Parameters
    ----------
    angle_indices_all : list[tuple[int, int, int]]
        Atom index triplets defining angles.
    ref_coords : np.ndarray
        Reference coordinates.
    traj_coords : np.ndarray
        Simulation coordinates.
    outpath : str
        Directory to save plots.
    disp_fn : callable
        Periodic displacement function.
    """
    angle_fn = utils.init_angle_fn(disp_fn, angle_indices_all)
    angles_ref = angle_fn(ref_coords)
    angles_traj = angle_fn(traj_coords)
    ref_flat = np.radians(np.concatenate(angles_ref))
    traj_flat = np.radians(np.concatenate(angles_traj))
    ref_clean = ref_flat[np.isfinite(ref_flat)]
    traj_clean = traj_flat[np.isfinite(traj_flat)]

    # Full-range KDE
    fig1, ax1 = plt.subplots(figsize=(8,6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        xs = np.linspace(traj_clean.min(), traj_clean.max(), 1000)
        ax1.plot(xs, kde_t(xs), label='Trajectory KDE')
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        xsr = np.linspace(min(ref_clean.min(), traj_clean.min()),
                          max(ref_clean.max(), traj_clean.max()), 1000)
        ax1.plot(xsr, kde_r(xsr), '--', label='Reference KDE')
    ax1.set_xlim(0, np.pi)
    ax1.set_xlabel('Angle (radians)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Bond Angle KDE: Trajectory vs Reference (Full Range)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(os.path.join(outpath, 'bond_angles_density.png'), dpi=300)
    plt.close(fig1)

    # Zoomed KDE
    fig2, ax2 = plt.subplots(figsize=(8,6))
    if traj_clean.size > 1:
        kde_t = gaussian_kde(traj_clean)
        ax2.plot(xs, kde_t(xs), label='Trajectory KDE')
    if ref_clean.size > 1:
        kde_r = gaussian_kde(ref_clean)
        ax2.plot(xsr, kde_r(xsr), '--', label='Reference KDE')
    ax2.set_xlim(1.6, np.pi)
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Bond Angle KDE: Trajectory vs Reference (Zoomed: 1.6 to π)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(outpath, 'bond_angles_density_zoomed.png'), dpi=300)
    plt.close(fig2)


def plot_hex_dihedral(
    ref_coords: np.ndarray,
    traj_coords: np.ndarray,
    disp_fn: callable,
    dihedral_indices_all: list[tuple[int, int, int, int]],
    outpath: str
) -> None:
    """
    Plot dihedral angle distributions for all hexane CG dihedrals.

    Computes dihedral angles for every molecule and frame, then overlays
    reference vs simulation histograms on a single panel.

    Parameters
    ----------
    ref_coords : np.ndarray
        Reference coordinates.
    traj_coords : np.ndarray
        Simulation coordinates.
    disp_fn : callable
        Periodic displacement function.
    dihedral_indices_all : list[tuple[int,int,int,int]]
        Lists of atom quartets defining dihedrals.
    outpath : str
        Directory to save the plot.
    """
    hex_fn = utils.init_dihedral_fn(disp_fn, dihedral_indices_all)
    CG_angles = np.concatenate(hex_fn(traj_coords))
    AT_angles = np.concatenate(hex_fn(ref_coords))
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    utils.plot_1d_dihedral(ax, [AT_angles, CG_angles], ['AT', 'Simulation'], bins=60, degrees=True)
    ax.set_title('Dihedral angle (all molecules)')
    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'dihedral_angle.png'), dpi=300)
    plt.close(fig)

def plot_bond_angle_correlation(ref_coords, traj_coords, angle_idcs, bond_idcs, disp_fn, outpath):
    # --- compute raw lists ---
    hex_angle_fn = utils.init_angle_fn(disp_fn, angle_idcs)  
    angles_ref  = hex_angle_fn(ref_coords)   # [N_angle_idcs, n_frames]
    angles_traj = hex_angle_fn(traj_coords)  # [N_angle_idcs, n_frames]
    
    dists_ref   = [utils.compute_atom_distance(ref_coords, a, b, disp_fn) for a,b in bond_idcs]
    dists_traj  = [utils.compute_atom_distance(traj_coords, a, b, disp_fn) for a,b in bond_idcs]

    angles_ref_flat  = np.radians(np.concatenate(angles_ref))
    angles_traj_flat = np.radians(np.concatenate(angles_traj))
    dists_ref_flat   = np.concatenate(dists_ref)
    dists_traj_flat  = np.concatenate(dists_traj)

    # determine how many bonds per angle
    n_angles    = len(angle_idcs)
    n_distances = len(bond_idcs)
    if n_angles == 0 or (n_distances % n_angles) != 0:
        raise ValueError(f"Expected number of distances ({n_distances}) to be a multiple of number of angles ({n_angles})")
    repeat_factor = n_distances // n_angles

    # repeat angles to align with distances
    angles_ref_rep  = np.repeat(angles_ref_flat,  repeat_factor)
    angles_traj_rep = np.repeat(angles_traj_flat, repeat_factor)

    # drop any pairs where either is NaN
    mask_ref  = np.isfinite(angles_ref_rep) & np.isfinite(dists_ref_flat)
    mask_traj = np.isfinite(angles_traj_rep) & np.isfinite(dists_traj_flat)

    angles_ref_final  = angles_ref_rep[mask_ref]
    dists_ref_final   = dists_ref_flat[mask_ref]
    angles_traj_final = angles_traj_rep[mask_traj]
    dists_traj_final  = dists_traj_flat[mask_traj]
    
    # --- finally make your 2D histograms ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    hist_ref,  xedges_ref,  yedges_ref  = np.histogram2d(
        angles_ref_final,  dists_ref_final,  bins=50, density=True
    )
    hist_traj, xedges_traj, yedges_traj = np.histogram2d(
        angles_traj_final, dists_traj_final, bins=50, density=True
    )
    # Plot reference histogram
    extent_ref = [xedges_ref[0], xedges_ref[-1], yedges_ref[0], yedges_ref[-1]]
    im1 = ax1.imshow(hist_ref.T, origin='lower', extent=extent_ref, 
                        aspect='auto', cmap='plasma')
    ax1.set_xlabel('Bond Angle (radians)')
    ax1.set_ylabel('Bond Distance (nm)')
    ax1.set_title('Reference: Bond vs Angle')   
    
    
    extent_traj = [xedges_traj[0], xedges_traj[-1], yedges_traj[0], yedges_traj[-1]]
    im2 = ax2.imshow(hist_traj.T, origin='lower', extent=extent_traj, 
                        aspect='auto', cmap='plasma')
    ax2.set_xlabel('Bond Angle (radians)')
    ax2.set_ylabel('Bond Distance (nm)')
    ax2.set_title('Trajectory: Bond vs Angle')
    plt.colorbar(im2, ax=ax2, label='Density')

    plt.tight_layout()
    fig.savefig(os.path.join(outpath, 'bond_angle_correlation_heatmap.png'), dpi=300)
    plt.close(fig)




    
def vis_hexane(traj_path, type='AT', name='Simulation', dataset=None, cg_map='six-site', nmol=100):
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    config = json.load(open(os.path.join(os.path.dirname(traj_path), 'traj_config.json'), 'r'))
    line_locs = compute_line_locations(config)

    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)
    
    # Initialize variables that might not be defined in all cases
    cg_dihedral_idcs = None
    CG_angle_idcs = None
    
    # mapping
    if type == 'AT':
        sites_per_mol = 20
        at_dihedral_idcs = [4, 7, 10, 13] 
        CC_pairs = [(0,4),(4,7),(7,10),(10,13),(13,16)]
        CH_pairs = [(0,1),(0,2),(0,3),(4,5),(4,6),(7,8),(7,9),(10,11),(10,12),(13,14),(13,15),(16,17),(16,18),(16,19)]
        ref_coords = np.concatenate([
            dataset.dataset_X['training']['R'], 
            dataset.dataset_X['validation']['R'], 
            dataset.dataset_X['testing']['R']
        ], axis=0)

    else:
        definitions = {
            'two-site':  ([(0,1)], 2, None, None),
            'two-site-Map2':  ([(0,1)], 2, None, None),
            'three-site': ([(0,1),(1,2)], 3, [(0,1,2)], None),
            'three-site-Map1': ([(0,1),(1,2)], 3, [(0,1,2)], None),
            'four-site': ([(0,1),(1,2),(2,3)], 4, None, [0, 1, 2, 3]),
            'six-site':  ([(1,2),(2,3),(3,4)], 6, None, [1, 2, 3, 4]),
            'six-site-Map2':  ([(1,2),(2,3),(3,4)], 6, None, [1, 2, 3, 4]),
        }
        
        if cg_map not in definitions:
            raise ValueError(f"Unknown cg_map: {cg_map}. Available options: {list(definitions.keys())}")
            
        CC_pairs, sites_per_mol, CG_angle_idcs, cg_dihedral_idcs = definitions[cg_map]
        ref_coords = np.concatenate([
            dataset.cg_dataset_U['training']['R'], 
            dataset.cg_dataset_U['validation']['R'], 
            dataset.cg_dataset_U['testing']['R']
        ], axis=0)

    actual_nmol = config.get('nmol', nmol)
    plot_energy_and_kT(aux, line_locs, outpath)

    if 'epot' in aux:
        epot = aux['epot']
        if np.any(epot > 1000):
            first_explosion = np.where(epot > 1000)[0][0]
            traj_coords = traj_coords[:first_explosion]
            aux = {k: v[:first_explosion] for k, v in aux.items() if isinstance(v, (np.ndarray, list))}
            print(f"Energy exceeded 10^4 at frame {first_explosion}, truncating trajectory.")
    

    CC_all = []
    Dihedrals_idcs_all = []
    Angles_idcs_all = []
    
    for m in range(actual_nmol):
        offset = m * sites_per_mol
        CC_all.extend([(a+offset, b+offset) for a, b in CC_pairs])
        
        if cg_dihedral_idcs is not None:
            Dihedrals_idcs_all.extend([(a+offset, b+offset, c+offset, d+offset) for a, b, c, d in [cg_dihedral_idcs]])
            
        if CG_angle_idcs is not None:
            Angles_idcs_all.extend([(a+offset, b+offset, c+offset) for a, b, c in CG_angle_idcs])
        
    if type == 'AT':
        CH_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            CH_all.extend([(a+offset, b+offset) for a, b in CH_pairs])
            
        fig_ch, ax_ch = plt.subplots(figsize=(10, 5))
        for a, b in CH_all:
            d = utils.compute_atom_distance(traj_coords, a, b, disp_fn)
            ax_ch.plot(d, alpha=0.1)
        ax_ch.set_title('AT CH distances (all molecules)')
        ax_ch.set_xlabel('Time step')
        ax_ch.set_ylabel('Distance')
        plt.tight_layout()
        fig_ch.savefig(os.path.join(outpath, 'AT_CH_distances_all.png'), dpi=300)
        plt.close(fig_ch)
        
        Dihedral_AT_all = []
        for m in range(actual_nmol):
            offset = m * sites_per_mol
            Dihedral_AT_all.extend([(a+offset, b+offset, c+offset, d+offset) for a, b, c, d in [at_dihedral_idcs]])
        
        plot_hex_dihedral(ref_coords, traj_coords, disp_fn, Dihedral_AT_all, outpath)
        
    elif cg_map == 'three-site':
        plot_hexane_angle(Angles_idcs_all, ref_coords, traj_coords, outpath, disp_fn)
        plot_bond_angle_correlation(ref_coords, traj_coords, Angles_idcs_all, CC_all, disp_fn, outpath)
    
    elif 'six-site' in cg_map or 'four-site' in cg_map:
        if cg_dihedral_idcs is not None:
            plot_hex_dihedral(ref_coords, traj_coords, disp_fn, Dihedrals_idcs_all, outpath)


    
def vis_ala15(traj_path, config, type='AT', name='Simulation', dataset=None, cg_map='hmerged'):
    print(f"Visualizing {name} trajectory at {traj_path}")
    
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)
    
    plot_energy_and_kT(aux, line_locs, outpath)
    
    if type == 'AT':
        raise NotImplementedError("AT visualization for ALA15 is not implemented yet.")
    else:
        maps = {
            'CA': ([0,1,2,3],[1,2,3,4], [(0,1),(1,2),(2,3)]),
            'CA-Map2': ([0,1,2,3],[1,2,3,4], [(0,1),(1,2),(2,3)]),
            'CA-Map3': ([0,1,2,3],[1,2,3,4], [(0,1),(1,2),(2,3)]),
            'CA-Map4': ([0,1,2,3],[1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreMap2': ([3,4,5,6], [4,5,6,7], [(4,5),(5,6),(6,7)]),
            'coreBetaMap2': ([4,5,6,8], [5,6,8,9], [(4,5),(5,6),(6,8)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate([
                    dataset.cg_dataset_U['training']['R'],
                    dataset.cg_dataset_U['validation']['R'],
                    dataset.cg_dataset_U['testing']['R']
                ], axis=0)

    
    if len(phi_indices)>0:
        ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
        AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
        Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

        plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
        plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.*quantity.kb, outpath)
    
    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i,j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i,j in pairs]

    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)


def vis_pro2(traj_path, config, type='AT', name='Simulation', dataset=None, cg_map='hmerged'):
    print(f"Visualizing {name} trajectory at {traj_path}")
    
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == 'AT':
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate([
            dataset.dataset_U['training']['R'],
            dataset.dataset_U['validation']['R'],
            dataset.dataset_U['testing']['R']
        ], axis=0)
    else:
        maps = {
            'hmerged': ([1,3,7,8],[3,7,8,10], [(1,3),(3,7),(7,8)]),
            'heavyOnly':  ([1,3,7,8],[3,7,8,10], [(1,3),(3,7),(7,8)]),
            'heavyOnlyMap2': ([1,3,7,8],[3,7,8,10], [(1,3),(3,7),(7,8)]),
            'core': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreMap2': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreBeta': ([0,1,3,4], [1,3,4,5], [(0,1),(1,3),(3,4)]),
            'coreBetaMap2': ([0,1,3,4], [1,3,4,5], [(0,1),(1,3),(3,4)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate([
                    dataset.cg_dataset_U['training']['R'],
                    dataset.cg_dataset_U['validation']['R'],
                    dataset.cg_dataset_U['testing']['R']
                ], axis=0)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)
    
    ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i,j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i,j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
    
    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.*quantity.kb, outpath)
    


def vis_gly2(traj_path, config, type='AT', name='Simulation', dataset=None, cg_map='hmerged'):
    print(f"Visualizing {name} trajectory at {traj_path}")
    
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)

    # selection
    if type == 'AT':
        phi_indices = [4,6,8,11]
        psi_indices = [6, 8, 11, 13]
        pairs = [(4, 6), (6, 8), (8, 11)]
        ref_coords = np.concatenate([
            dataset.dataset_U['training']['R'],
            dataset.dataset_U['validation']['R'],
            dataset.dataset_U['testing']['R']
        ], axis=0)
    else:
        maps = {
            'hmerged': ([1,3,4,5],[3,4,5,7], [(1,3),(3,4),(4,5)]),
            'heavyOnly': ([1,3,4,5],[3,4,5,7], [(1,3),(3,4),(4,5)]),
            'heavyOnlyMap2': ([1,3,4,5],[3,4,5,7], [(1,3),(3,4),(4,5)]),
            'core': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreMap2': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate([
                    dataset.cg_dataset_U['training']['R'],
                    dataset.cg_dataset_U['validation']['R'],
                    dataset.cg_dataset_U['testing']['R']
                ], axis=0)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)
    
    ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i,j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i,j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
    
    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.*quantity.kb, outpath)
    


def vis_thr2(traj_path, config, type='AT', name='Simulation', dataset=None, cg_map='hmerged'):
    print(f"Visualizing {name} trajectory at {traj_path}")
    
    box = dataset.box
    outpath = prepare_output_dir(traj_path)
    line_locs = compute_line_locations(config)
    
    # selection
    if type == 'AT':
        phi_indices = [4, 6, 16, 18]
        psi_indices = [6, 16, 18, 20]
        pairs = [(4, 6), (6, 16), (16, 18)]
        ref_coords = np.concatenate([
            dataset.dataset_U['training']['R'],
            dataset.dataset_U['validation']['R'],
            dataset.dataset_U['testing']['R']
        ], axis=0)
    else:
        maps = {
            'hmerged': ([1,3,5,8],[3,5,8,10], [(1,3),(3,5),(5,8)]),
            'heavyOnly': ([1,3,5,8],[3,5,8,10], [(1,3),(3,5),(5,8)]),
            'heavyOnlyMap2': ([1,3,5,8],[3,5,8,10], [(1,3),(3,5),(5,8)]),
            'core': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreMap2': ([0,1,2,3], [1,2,3,4], [(0,1),(1,2),(2,3)]),
            'coreBeta': ([0,1,2,4], [1,2,4,5], [(0,1),(1,2),(2,3)]),
            'coreBetaMap2': ([0,1,2,4], [1,2,4,5], [(0,1),(1,2),(2,3)]),
        }
        phi_indices, psi_indices, pairs = maps[cg_map]
        ref_coords = np.concatenate([
                    dataset.cg_dataset_U['training']['R'],
                    dataset.cg_dataset_U['validation']['R'],
                    dataset.cg_dataset_U['testing']['R']
                ], axis=0)
    traj_coords, aux = load_trajectory(traj_path)
    disp_fn, _ = periodic_displacement(box, True)
    
    ala2_dihedral_fn = utils.init_dihedral_fn(disp_fn, [phi_indices, psi_indices])
    AT_phi, AT_psi = ala2_dihedral_fn(ref_coords)
    Traj_phi, Traj_psi = ala2_dihedral_fn(traj_coords)

    AT_dists = [utils.compute_atom_distance(ref_coords, i, j, disp_fn) for i,j in pairs]
    Traj_dists = [utils.compute_atom_distance(traj_coords, i, j, disp_fn) for i,j in pairs]

    plot_energy_and_kT(aux, line_locs, outpath)
    plot_time_series(traj_coords, ref_coords, phi_indices, outpath, name, line_locs)
    plot_dist_series(pairs, AT_dists, Traj_dists, outpath, name, line_locs)
    plot_dihedrals(AT_phi, AT_psi, Traj_phi, Traj_psi, outpath, line_locs)
    
    plot_ramachandran(AT_phi, AT_psi, Traj_phi, Traj_psi, 300.*quantity.kb, outpath)
    