import numpy as onp
from jax import numpy as jnp
import jax
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from jax import vmap
from cycler import cycler
from chemtrain import quantity
from jax_md_mod import custom_quantity
from typing import Callable, Union
import numpy.typing as npt
import io
from concurrent.futures import ProcessPoolExecutor
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def setup_distance_filter_fn(ref_means, disp_fn, delta=0.05):
    """
    Creates a distance filter function to validate and filter trajectory frames 
    based on distance constraints between specified atom pairs.
        ref_means (list of float): Reference mean distances for each atom pair.
        disp_fn (callable): Function to compute displacement vectors.
        delta (float, optional): Tolerance for distance constraints. Defaults to 0.05 nm.
        callable: A function that filters trajectory frames based on distance constraints.
        
    The returned function, `apply_distance_filter`, has the following signature:
        apply_distance_filter(traj, indices, verbose=True, print_every=0.5)
            traj (jax.numpy.ndarray): Trajectory array of shape 
                (n_chains, n_frames, n_atoms, 3).
            indices (list of tuple): List of atom index pairs to compute distances.
            verbose (bool, optional): If True, prints shape and processing information. 
                Defaults to True.
            print_every (float, optional): Time interval for printing statistics. 
                Defaults to 0.5.
            tuple: 
                - traj (jax.numpy.ndarray): Filtered trajectory with NaN padding 
                  for invalid frames. Shape: (n_chains, n_frames, n_atoms, 3).
                - combined_mask (jax.numpy.ndarray): Boolean mask indicating valid 
                  frames. Shape: (n_chains, n_frames).
    """
    def apply_distance_filter(traj, indices, verbose=True, print_every=0.5):
        """
        Filter multi-chain trajectory frames based on distance constraints.
        
        Args:
            traj: JAX array of shape (n_chains, n_frames, n_atoms, 3).
            indices: List of atom index pairs to form distances.
            verbose: Print shape information.
            
        Returns:
            traj: Shape (n_chains, n_frames, n_atoms, 3) with NaN padding after violations.
            combined_mask: Boolean mask of shape (n_chains, n_frames).
        """
        # Ensure indices is a list of pairs
        assert all(len(pair) == 2 for pair in indices), "indices elements must be pairs."
        
        if verbose:
            print('Input shape:', traj.shape)

        # --- Inner function to process a single chain (T, N, 3) ---
        def filter_single_chain(chain_traj):
            # Initialize mask (T,)
            # We start with True (1) for all frames
            chain_mask = jnp.ones(chain_traj.shape[0], dtype=bool)
            
            # 1. Calculate validity based on geometry
            for (i, j), mean_dist in zip(indices, ref_means):
                # utils.compute_atom_distance is assumed to work on (T, N, 3)
                # based on your previous implementation
                distances = compute_atom_distance(chain_traj, i, j, disp_fn)
                
                # Check constraints
                distance_mask = (distances > (mean_dist - delta)) & \
                                (distances < (mean_dist + delta))
                
                chain_mask = chain_mask & distance_mask

            # 2. Apply "First Violation" Logic
            # We want to invalidate all frames AFTER the first False.
            # jnp.cumprod on booleans: [1, 1, 0, 1, 1] -> [1, 1, 0, 0, 0]
            cumulative_mask = jnp.cumprod(chain_mask, axis=0).astype(bool)
            
            # 3. Apply NaN masking
            # Broadcast mask (T,) to (T, N, 3)
            masked_chain_traj = jnp.where(
                cumulative_mask[:, None, None], 
                chain_traj, 
                jnp.nan
            )
            
            return masked_chain_traj, cumulative_mask

        # --- Apply vmap over the chain dimension (axis 0) ---
        # In: (n_chains, n_frames, n_atoms, 3)
        # Out: (n_chains, n_frames, n_atoms, 3), (n_chains, n_frames)
        mapped_fn = jax.vmap(filter_single_chain)
        filtered_traj, final_mask = mapped_fn(traj)

        if verbose:
            # Calculate how many valid frames remain per chain for info
            valid_counts = jnp.sum(final_mask, axis=1)
            print(f'>> Processed shape: {filtered_traj.shape}')
            avg_length = jnp.mean(valid_counts)*print_every
            std_length = jnp.std(valid_counts)*print_every
            print(f'>> Valid frames per chain ns {avg_length:.1f} ± {std_length:.1f}')
                    
            # Calculate how many chains have at least one invalid frame
            chains_with_invalid_frames = jnp.sum(jnp.any(~final_mask, axis=1))
            print(f'>> Chains with at least one invalid frame: {chains_with_invalid_frames}/{traj.shape[0]}')
        
        return filtered_traj, final_mask

    return apply_distance_filter


def get_line_locations(t_eq, t_tot, n_chains, print_every=0.5):
    steps = int((t_tot - t_eq) / print_every)
    arr = onp.arange(0, steps * n_chains, steps)
    return arr[1:]


def split_into_chains(data: onp.ndarray, line_locations: list[int]) -> onp.ndarray:
    segments: list[onp.ndarray] = []
    start = 0
    for loc in line_locations:
        segments.append(data[start:loc])
        start = loc
    segments.append(data[start:])
    return onp.array(segments)


def _format_xyz_frame(args):
    """Worker: format one frame to an XYZ string."""
    frame_idx, positions_frame, species_col = args
    n_atoms = positions_frame.shape[0]
    buf = io.StringIO()
    buf.write(f"{n_atoms}\nFrame {frame_idx + 1}\n")
    # mix species (object/str) + floats; onp.savetxt handles mixed fmt nicely
    data = onp.c_[species_col, positions_frame]
    onp.savetxt(buf, data, fmt="%s %.6f %.6f %.6f")
    return buf.getvalue()


def save_xyz_frames_parallel(
    positions, species_list, filename, workers=None, chunksize=8, buffer_bytes=1_048_576
):
    """
    Parallel XYZ writer.
    - Parallelizes CPU-bound text formatting per frame with processes.
    - Preserves frame order in the output file.
    - Streams results to disk to avoid large memory spikes.

    positions: (n_frames, n_atoms, 3) float array
    species_list: list[str] length n_atoms
    """
    positions = onp.asarray(positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions.shape
    if len(species_list) != n_atoms:
        raise ValueError(
            f"Species list length ({len(species_list)}) must match number of atoms ({n_atoms})"
        )

    # Cache species column once; small and cheap to pickle
    species_col = onp.asarray(species_list, dtype=object).reshape(-1, 1)

    # Small datasets don't benefit from process spin-up—fall back to fast single-process path
    if workers == 1 or n_frames < 4:
        with open(filename, "w", buffering=buffer_bytes) as f:
            for frame_idx in range(n_frames):
                f.write(
                    _format_xyz_frame((frame_idx, positions[frame_idx], species_col))
                )
        return

    # Parallel formatting; .map preserves input order so the file stays ordered
    with open(filename, "w", buffering=buffer_bytes) as f, ProcessPoolExecutor(
        max_workers=workers
    ) as ex:
        # Build an iterable of work items without materializing all frames at once
        iterable = ((i, positions[i], species_col) for i in range(n_frames))
        for frame_str in ex.map(_format_xyz_frame, iterable, chunksize=chunksize):
            f.write(frame_str)


def compute_angle(coords: jnp.ndarray, idcs: list[int]) -> jnp.ndarray:
    """
    Compute bond angles for every frame of a trajectory.

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    idcs : list[int]
        Three atom indices [i, j, k] where j is the central atom

    Returns
    -------
    jnp.ndarray
        Array of bond angles in radians with shape (n_frames,)
    """
    i0, i1, i2 = idcs

    @jax.jit
    def angle_of_frame(frame: jnp.ndarray) -> jnp.ndarray:
        p0 = frame[i0]
        p1 = frame[i1]
        p2 = frame[i2]
        return calculate_angle(p0, p1, p2)

    # Vectorize over frames
    return jax.vmap(angle_of_frame)(coords)


def calculate_angle(p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the bond angle between three points p0-p1-p2.

    Parameters
    ----------
    p0 : jnp.ndarray
        Position vector of the first atom with shape (3,)
    p1 : jnp.ndarray
        Position vector of the central atom with shape (3,)
    p2 : jnp.ndarray
        Position vector of the third atom with shape (3,)

    Returns
    -------
    jnp.ndarray
        Bond angle in radians (scalar)
    """
    # Vectors from central atom to the other two atoms
    v1 = p0 - p1  # Vector from p1 to p0
    v2 = p2 - p1  # Vector from p1 to p2

    # Normalize vectors
    v1_norm = jnp.linalg.norm(v1)
    v2_norm = jnp.linalg.norm(v2)

    # Compute cosine of angle using dot product
    cos_angle = jnp.dot(v1, v2) / (v1_norm * v2_norm)

    # Clamp to avoid numerical issues with arccos
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)

    # Calculate angle in radians
    angle = jnp.arccos(cos_angle)

    return angle


def init_dihedral_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute dihedral angles from trajectory positions.

    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Four atom indices defining the dihedral angle

    Returns
    -------
    Callable
        Function that takes positions and returns dihedral angles
    """
    idcs = jnp.array(idcs)

    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_dihedrals = jax.vmap(
            custom_quantity.dihedral_displacement, (0, None, None)
        )
        dihedral_angles = batched_dihedrals(positions, displacement_fn, idcs)
        return dihedral_angles.T

    return postprocess_fn


def init_angle_fn(displacement_fn: Callable, idcs: list[int]) -> Callable:
    """
    Initialize a function to compute bond angles from trajectory positions.

    Parameters
    ----------
    displacement_fn : Callable
        Function to compute displacement vectors between atoms
    idcs : list[int]
        Three atom indices defining the bond angle

    Returns
    -------
    Callable
        Function that takes positions and returns bond angles
    """
    idcs = jnp.array(idcs)

    def postprocess_fn(positions: jnp.ndarray) -> jnp.ndarray:
        batched_angles = jax.vmap(custom_quantity.angular_displacement, (0, None, None))
        dihedral_angles = batched_angles(positions, displacement_fn, idcs)
        return dihedral_angles.T

    return postprocess_fn


def compute_atom_distance(
    coords: jnp.ndarray,
    idx1: int,
    idx2: int,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Compute the PBC-aware distance between two atoms over all trajectory frames.

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    idx1 : int
        Index of the first atom
    idx2 : int
        Index of the second atom
    displacement_fn : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Function that computes PBC-corrected displacement between two position vectors

    Returns
    -------
    jnp.ndarray
        Array of scalar distances with shape (n_frames,)
    """
    # Slice out the trajectories of atom idx1 and idx2: each is [n_frames, 3]
    r1 = coords[:, idx1, :]
    r2 = coords[:, idx2, :]

    # Compute the PBC‐corrected displacement for each frame
    # displacement_fn is assumed to take two [3]-vectors → [3]-vector
    disp = vmap(displacement_fn)(r1, r2)  # → [n_frames, 3]
    # Finally, L2‐norm each displacement vector → [n_frames]
    distances = jnp.linalg.norm(disp, axis=-1)

    return distances


def calculate_dihedral(
    p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate dihedral angle between four points in degrees.

    Parameters
    ----------
    p0 : jnp.ndarray
        Position vector of the first atom with shape (3,)
    p1 : jnp.ndarray
        Position vector of the second atom with shape (3,)
    p2 : jnp.ndarray
        Position vector of the third atom with shape (3,)
    p3 : jnp.ndarray
        Position vector of the fourth atom with shape (3,)

    Returns
    -------
    jnp.ndarray
        Dihedral angle in degrees (scalar)
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # Normalize b1 so it does not influence magnitude of vector rejections
    b1 /= jnp.linalg.norm(b1)

    # Compute cross products
    v = b0 - jnp.dot(b0, b1) * b1
    w = b2 - jnp.dot(b2, b1) * b1

    x = jnp.dot(v, w)
    y = jnp.dot(jnp.cross(b1, v), w)

    return jnp.degrees(jnp.arctan2(y, x))


def plot_1d_dihedral(
    ax: Axes,
    angles: list[jnp.ndarray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = "$\phi$ in deg",
    ylabel: bool = True,
) -> Axes:
    """
    Plot 1D histogram splines for dihedral angles.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[jnp.ndarray]
        list of angle arrays, one for each model/dataset
    labels : list[str]
        list of labels for each angle dataset
    bins : int, optional
        Number of histogram bins (default: 120)
    degrees : bool, optional
        Whether angles are in degrees (default: True)
    xlabel : str, optional
        Label for x-axis (default: '$\phi$ in deg')
    ylabel : bool, optional
        Whether to add y-axis label (default: True)

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = [
        "#368274",
        "#0C7CBA",
        "#C92D39",
        "#FFB347",
        "#7851A9",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]
    line = ["-", "-", "-"]

    n_models = len(angles)
    for i in range(n_models):
        if degrees:
            angles_conv = angles[i]
            hist_range = [-180, 180]
        else:
            angles_conv = onp.rad2deg(angles[i])
            hist_range = [-onp.pi, onp.pi]

        # Compute the histogram
        hist, x_bins = jnp.histogram(
            angles_conv, bins=bins, density=True, range=hist_range
        )
        width = x_bins[1] - x_bins[0]
        bin_center = x_bins + width / 2

        ax.plot(
            bin_center[:-1],
            hist,
            label=labels[i],  # , color=color[i],
            # linestyle=line[i],
            linewidth=2.0,
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Density")

    # ax.legend()  # Add legend to the plot
    return ax


def plot_1d_dihedral_mean_std(
    ax: Axes,
    angles: list[npt.NDArray],
    labels: list[str],
    bins: int = 120,
    degrees: bool = True,
    xlabel: str = "$\phi$ in deg",
    ylabel: bool = True,
) -> Axes:
    """
    Plot 1D histogram with mean and standard deviation as shaded area for each dihedral angle set.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object to plot on
    angles : list[npt.NDArray]
        list of angle arrays, one for each model/dataset
    labels : list[str]
        list of labels for each angle dataset
    bins : int, optional
        Number of histogram bins (default: 120)
    degrees : bool, optional
        Whether angles are in degrees (default: True)
    xlabel : str, optional
        Label for x-axis (default: '$\phi$ in deg')
    ylabel : bool, optional
        Whether to add y-axis label (default: True)

    Returns
    -------
    Axes
        The modified matplotlib axes object
    """
    color = [
        "#0C7CBA",
        "#C92D39",
        "#FFB347",
        "#7851A9",
        "#66CC99",
        "#FF6B6B",
        "#4A90E2",
        "#50514F",
        "#F4A261",
    ]
    n_models = len(angles)
    for i in range(n_models):
        data = onp.array(angles[i])
        if degrees:
            data_conv = data
            hist_range = [-180, 180]
        else:
            data_conv = onp.rad2deg(data)
            hist_range = [-onp.pi, onp.pi]

        # Compute histogram for each sample, then mean/std over samples
        # If data_conv is 2D: [n_samples, n_points]
        if data_conv.ndim == 2:
            hists = []
            for sample in data_conv:
                hist, x_bins = onp.histogram(
                    sample, bins=bins, density=True, range=hist_range
                )
                hists.append(hist)
            hists = onp.stack(hists)
            hist_mean = hists.mean(axis=0)
            hist_std = hists.std(axis=0)
        else:
            hist_mean, x_bins = onp.histogram(
                data_conv, bins=bins, density=True, range=hist_range
            )
            hist_std = onp.zeros_like(hist_mean)

        width = x_bins[1] - x_bins[0]
        bin_center = x_bins[:-1] + width / 2

        ax.plot(bin_center, hist_mean, label=labels[i], color=color[i], linewidth=2.0)
        ax.fill_between(
            bin_center,
            hist_mean - hist_std,
            hist_mean + hist_std,
            color=color[i],
            alpha=0.2,
        )

    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel("Density")
    ax.legend()
    return ax


def determine_free_energy_scale(
    x_list, y_list, kbt: float, bins: int = 100, min_count_for_color: int = 0
):
    """
    Determine a common free-energy scale (vmin, vmax) across multiple datasets.

    Parameters
    ----------
    x_list : list of array-like
        List of x-coordinate arrays.
    y_list : list of array-like
        List of y-coordinate arrays.
    kbt : float
        Thermal energy for free energy calculation.
    bins : int or [int, int], optional
        Number of histogram bins along each dimension.
    min_count_for_color : int, optional
        Minimum number of counts per bin to be considered valid.

    Returns
    -------
    (vmin, vmax) : tuple of floats
        Suggested global free-energy color scale limits (in kcal/mol).

    Notes
    -----
    - Works with any x/y data (angular or Cartesian)
    - Free energy computed as: F = -kbt/4.184 * ln(density)
    - Only finite values are considered
    - Returns the range after shifting so minimum is at 0
    """
    import numpy as onp

    try:
        import jax.numpy as jnp
    except ImportError:
        # Fallback to numpy if jax not available
        jnp = onp

    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length.")

    F_all_min, F_all_max = onp.inf, -onp.inf
    any_valid = False

    for x, y in zip(x_list, y_list):
        # Convert to numpy arrays
        x_np = onp.asarray(x)
        y_np = onp.asarray(y)

        if x_np.size == 0 or y_np.size == 0:
            continue

        # Keep only entries where both are finite
        mask = onp.isfinite(x_np) & onp.isfinite(y_np)
        if onp.sum(mask) == 0:
            continue

        x_f = x_np[mask]
        y_f = y_np[mask]

        # Compute histograms - exactly as in plot_histogram_free_energy3
        counts, x_edges, y_edges = onp.histogram2d(x_f, y_f, bins=bins, density=False)
        density, _, _ = onp.histogram2d(x_f, y_f, bins=bins, density=True)

        # Occupancy mask
        mask_occ = counts > min_count_for_color

        # Compute free energy using same formula as plot function
        density_jnp = jnp.asarray(density)
        mask_occ_jnp = jnp.asarray(mask_occ.astype(bool))

        with onp.errstate(invalid="ignore"):
            F_jnp = jnp.where(
                density_jnp > 0, jnp.log(density_jnp) * (-(kbt / 4.184)), jnp.nan
            )

        F_jnp = jnp.where(mask_occ_jnp, F_jnp, jnp.nan)
        F_np = onp.asarray(F_jnp)

        # If this dataset has any finite F, update global min/max
        if onp.any(onp.isfinite(F_np)):
            any_valid = True
            F_valid = F_np[onp.isfinite(F_np)]

            # Find the actual min/max for this dataset
            current_min = float(onp.min(F_valid))
            current_max = float(onp.max(F_valid))

            # Update global min/max
            F_all_min = min(F_all_min, current_min)
            F_all_max = max(F_all_max, current_max)

    if not any_valid:
        raise ValueError(
            "No valid free-energy data found across datasets (all inputs empty or NaN)."
        )

    # Shift the scale so minimum is at 0
    F_range = F_all_max - F_all_min
    return (
        0,
        float(F_range),
    )  # F_all_min, F_all_max #(0, float(F_range))#F_all_min, F_all_max #, float(F_range)


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

    Parameters:
      ax: matplotlib axis object
      x, y: input data arrays
      kbt: thermal energy for free energy calculation
      is_angular: if True, treat data as angular (wrap to [-π, π]). Default True.
      degrees: if True and is_angular=True, interpret input as degrees. Default True.
      xlabel: custom x-axis label (default: φ for angular, "x" otherwise)
      ylabel_text: custom y-axis label (default: ψ for angular, "y" otherwise)
      show_ylabel: whether to display y-axis label
      show_yticks: whether to display y-axis tick labels. Default True.
      title: plot title
      bins: number of histogram bins
      min_count_for_color: minimum counts to show color
      edge_feather_bins: feather width for smooth edges
      alpha_min: minimum blurred support to show color
      legend: if True, add colorbar
      scale: optional (vmin, vmax) for color scale
      xlim: optional (xmin, xmax) for x-axis limits
      ylim: optional (ymin, ymax) for y-axis limits

    Returns:
      (ax, cax, cbar, scale_used)
      - cbar is None when legend=False
      - scale_used is the (vmin, vmax) tuple actually applied

    Notes:
      - Free-energy calculation: F = -kbt/4.184 * ln(density)
      - Colormap: blue (low F) -> red (high F)
      - Empty bins fade to white
    """

    # ----------------- Styling -----------------
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

    # ----------------- Process input data -----------------
    x_np = onp.asarray(x)
    y_np = onp.asarray(y)

    if is_angular:
        # Angular data handling
        if degrees:
            x_plot = onp.deg2rad(x_np)
            y_plot = onp.deg2rad(y_np)
            default_xlabel = r"$\phi\ (°)$"
            default_ylabel = r"$\psi\ (°)$"
        else:
            x_plot = x_np
            y_plot = y_np
            default_xlabel = r"$\phi$ [rad]"
            default_ylabel = r"$\psi$ [rad]"

        # Set default limits for angular data
        if xlim is None:
            xlim = (-onp.pi, onp.pi)
        if ylim is None:
            ylim = (-onp.pi, onp.pi)

        # Prepare tick labels for angular data
        if degrees:
            deg_ticks_rad = onp.deg2rad(onp.array([-180, -90, 0, 90, 180]))
            deg_tick_labels = ["-180", "-90", "0", "90", "180"]
            use_angular_ticks = True
        else:
            use_angular_ticks = False
    else:
        # Non-angular data - use as-is
        x_plot = x_np
        y_plot = y_np
        default_xlabel = "x"
        default_ylabel = "y"
        use_angular_ticks = False

    # Use custom labels if provided, otherwise use defaults
    xlabel_final = xlabel if xlabel is not None else default_xlabel
    ylabel_final = ylabel_text if ylabel_text is not None else default_ylabel

    # ----------------- Histogram: counts and density -----------------
    counts, x_edges, y_edges = onp.histogram2d(x_plot, y_plot, bins=bins, density=False)
    density, _, _ = onp.histogram2d(x_plot, y_plot, bins=bins, density=True)

    # Base occupancy mask
    mask_occ = counts > min_count_for_color

    # ----------------- Free-energy calculation -----------------
    density_jnp = jnp.asarray(density)
    with onp.errstate(invalid="ignore"):
        F_jnp = jnp.where(
            density_jnp > 0, jnp.log(density_jnp) * (-(kbt / 4.184)), jnp.nan
        )

    mask_occ_jnp = jnp.asarray(mask_occ.astype(bool))
    F_jnp = jnp.where(mask_occ_jnp, F_jnp, jnp.nan)

    F = onp.asarray(F_jnp)

    # Valid mask
    valid = onp.isfinite(F).astype(float)

    # Soft edge feathering
    feather_sigma = float(edge_feather_bins)
    support_blur = gaussian_filter(valid, sigma=feather_sigma)

    alpha = onp.clip(
        (support_blur - alpha_min) / max(1e-9, (1.0 - alpha_min)), 0.0, 1.0
    )
    F_out = onp.where(alpha > 0, F, onp.nan)

    # ----------------- Colormap -----------------
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

    # ----------------- Coordinates -----------------
    Xe, Ye = onp.meshgrid(x_edges, y_edges)

    # ----------------- Determine color scale -----------------
    finite_mask = onp.isfinite(F_out)
    if onp.any(finite_mask):
        data_min = float(onp.nanmin(F_out))
        data_max = float(onp.nanmax(F_out))
    else:
        data_min = 0.0
        data_max = 1.0

    if scale is not None:
        if not (isinstance(scale, (tuple, list)) and len(scale) == 2):
            raise ValueError("`scale` must be a tuple (vmin, vmax) or None.")
        vmin, vmax = float(scale[0]), float(scale[1])
        # Just use the provided scale without validation
        # This allows for a common scale across multiple plots
        scale_used = (vmin, vmax)
        F_out = F_out - onp.nanmin(F_out)
    elif shift_scale_to_zero:
        # Shift so minimum is at 0
        F_out = F_out - data_min
        scale_used = (0.0, data_max - data_min)
        vmin, vmax = scale_used
    else:
        scale_used = (data_min, data_max)
        vmin, vmax = scale_used

    # ----------------- Plotting -----------------
    cax = ax.pcolormesh(
        Xe, Ye, F_out.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )

    # Axis labels / title
    ax.set_xlabel(xlabel_final)
    if show_ylabel:
        ax.set_ylabel(ylabel_final)
    ax.set_title(title)

    # Set limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Special tick handling for angular data
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

    # ----------------- Optional colorbar -----------------
    cbar = None
    if legend:
        divider = make_axes_locatable(ax)
        cb_ax = divider.append_axes("right", size="4%", pad=0.03)
        cbar = plt.colorbar(cax, cax=cb_ax, orientation="vertical")
        cbar.set_label("Free energy (kcal/mol)")

    return ax, cax, cbar, scale_used


def plot_atom_distance(
    ax: Axes,
    distances: jnp.ndarray | list[jnp.ndarray],
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
    distances : jnp.ndarray | list[jnp.ndarray]
        Distance data - single array or list of arrays for multiple models
    labels : list[str] | None, optional
        list of labels for each set of distances (default: None)
    bins : int, optional
        Number of bins for the histogram (default: 60)
    xlabel : str, optional
        Label for the x-axis (default: 'Distance')
    ylabel : str, optional
        Label for the y-axis (default: 'Frequency')

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
    AT_distances: list[jnp.ndarray],
    Traj_distances: list[jnp.ndarray],
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
    AT_distances : list[jnp.ndarray]
        list of 1D arrays of reference distances
    Traj_distances : list[jnp.ndarray]
        list of 1D arrays of simulation distances
    dist_labels : list[str]
        list of titles for each subplot (same length as distances)
    outpath : str
        Directory to save the figure in
    name : str
        Basename for the output file
    at_label : str, optional
        Legend label for reference data (default: "Reference")
    traj_label : str, optional
        Legend label for simulation data (default: "Simulation")
    bins : int, optional
        Number of bins (default: 60)
    at_color : str, optional
        Color for reference histograms (default: "#368274")
    traj_color : str, optional
        Color for simulation histograms (default: "#C92D39")
    xlabel : str, optional
        X-axis label (default: "Distance")
    ylabel : str, optional
        Y-axis label (default: "Normalized frequency")

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


def plot_energy(
    ax: Axes,
    energy: Union[jnp.ndarray, list[jnp.ndarray]],
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
    energy : Union[jnp.ndarray, list[jnp.ndarray]]
        Energy data - single array or list of arrays for multiple models
    labels : list[str]], optional
        list of labels for each set of energy values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'Energy [kJ/mol]')

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
    kT: Union[jnp.ndarray, list[jnp.ndarray]],
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
    kT : Union[jnp.ndarray, list[jnp.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str]], optional
        list of labels for each set of kT values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'kT [kJ/mol]')

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
    kT: Union[jnp.ndarray, list[jnp.ndarray]],
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
    kT : Union[jnp.ndarray, list[jnp.ndarray]]
        kT data - single array or list of arrays for multiple models
    labels : list[str], optional
        list of labels for each set of kT values (default: None)
    xlabel : str, optional
        Label for the x-axis (default: 'Time')
    ylabel : str, optional
        Label for the y-axis (default: 'T [K]')

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
    mae = onp.mean(onp.abs(pred_F - ref_F))
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


def calc_mse_dihedrals(
    phi_ref: jnp.ndarray,
    psi_ref: jnp.ndarray,
    phi_sim: jnp.ndarray,
    psi_sim: jnp.ndarray,
    nbins: int = 60,
) -> float:
    """
    Calculate mean squared error between reference and simulation dihedral angle distributions.

    Parameters
    ----------
    phi_ref : jnp.ndarray
        Reference phi dihedral angles in degrees
    psi_ref : jnp.ndarray
        Reference psi dihedral angles in degrees
    phi_sim : jnp.ndarray
        Simulation phi dihedral angles in degrees
    psi_sim : jnp.ndarray
        Simulation psi dihedral angles in degrees
    nbins : int, optional
        Number of bins for 2D histogram (default: 60)

    Returns
    -------
    float
        Mean squared error between the two 2D density histograms
    """
    # convert to radians
    phi_ref_rad, psi_ref_rad = jnp.deg2rad(phi_ref), jnp.deg2rad(psi_ref)
    phi_sim_rad, psi_sim_rad = jnp.deg2rad(phi_sim), jnp.deg2rad(psi_sim)

    h_ref, _, _ = onp.histogram2d(phi_ref_rad, psi_ref_rad, bins=nbins, density=True)
    h_sim, _, _ = onp.histogram2d(phi_sim_rad, psi_sim_rad, bins=nbins, density=True)

    mse = onp.mean((h_ref - h_sim) ** 2)
    print("MSE of the phi-psi dihedral density histogram:", mse)
    return mse


def plot_convergence(trainer, out_dir):
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5), layout="constrained")

    ax1.set_title("Loss")
    ax1.semilogy(trainer.train_losses, label="Training")
    ax1.semilogy(trainer.val_losses, label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    fig.savefig(f"{out_dir}/convergence.pdf", bbox_inches="tight")




def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""

    # print(f"Original positions: {dataset['R']}")
    print(f"Original positions: {dataset['R'].min()} to {dataset['R'].max()}")


    if fractional:
        box = dataset['box'][0, 0, 0]
        dataset['R'] = dataset['R'] / box
    else:
        dataset['R'] = dataset['R'] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset['box'] = scale_R * dataset['box']
    dataset['F'] *= scale_F

    return dataset

