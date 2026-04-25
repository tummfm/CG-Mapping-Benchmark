"""
Structural analysis calculations (RDF, radius of gyration, helicity).
"""

import numpy as np
from jax import numpy as jnp
from jax import vmap, jit
from scipy.interpolate import interp1d
from typing import Callable


def radius_of_gyration_vectorized(
    coords: jnp.ndarray,
    displacement_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """
    Vectorized version of radius of gyration calculation with PBC correction.

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    displacement_fn : Callable
        Function that computes PBC-corrected displacement between two position vectors

    Returns
    -------
    jnp.ndarray
        Array of Rg values with shape (n_frames,)
    """
    n_frames, n_atoms, _ = coords.shape

    # Use first atom as reference for each frame
    ref_atoms = coords[:, 0, :]  # Shape: (n_frames, 3)

    # Expand reference atoms to match all atom positions
    ref_expanded = jnp.expand_dims(ref_atoms, axis=1)  # Shape: (n_frames, 1, 3)
    ref_tiled = jnp.tile(ref_expanded, (1, n_atoms, 1))  # Shape: (n_frames, n_atoms, 3)

    # Compute PBC-corrected displacements for all atoms relative to reference
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

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_atoms, 3)
    displacement_fn : Callable
        Function that computes PBC-corrected displacement between two position vectors

    Returns
    -------
    jnp.ndarray
        Array of helicity content values with shape (n_frames,)

    Based on:
    Rudzinski, Joseph F., and William G. Noid. "Bottom-up coarse-graining of peptide ensembles and helix–coil transitions."
    Journal of chemical theory and computation 11.3 (2015): 1278-1291.
    https://pubs.acs.org/doi/10.1021/ct5009922
    """
    from cgbench.utils.geometry import compute_atom_distance

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
        distances = compute_atom_distance(coords, i, j, displacement_fn)
        all_distances.append(distances)

    # Stack all distances: shape (n_hel_pairs, n_frames)
    distance_matrix = jnp.stack(all_distances, axis=0)

    # Calculate exponential terms for all pairs and frames
    exp_terms = jnp.exp(-(1.0 / (2 * sigma_squared)) * (distance_matrix - R0) ** 2)

    # Average over all 1-4 pairs for each frame
    helicity_values = jnp.mean(exp_terms, axis=0)

    return helicity_values


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

    Parameters
    ----------
    coords : jnp.ndarray
        Trajectory coordinates with shape (n_frames, n_CA, 3)
    displacement_fn : Callable
        Function that computes PBC-corrected displacement between two position vectors

    Returns
    -------
    jnp.ndarray
        Array of xi_norm values with shape (n_frames, 1)
    """
    xi_norm_values = vmap(_calculate_xi_norm_single_frame, in_axes=(0, None))(
        coords, displacement_fn
    )
    return jnp.expand_dims(xi_norm_values, axis=1)


def calculate_rdf(
    trajectories,
    bead_types,
    displacement_fn,
    sites_per_mol=2,
    box_length=2.79573,
    box_volume=None,
    dr=0.01,
    pair_batch_size=900_000,
    frame_batch_size=100000,
    dtype=jnp.float32,
):
    """
    Calculate radial distribution functions for intermolecular pairs using JAX.

    Args:
        trajectories: List of arrays of shape (n_frames, n_particles, 3) or (n_particles, 3).
            May be in any coordinate system (fractional or real-space); displacement_fn
            handles the PBC and unit conversion.
        bead_types: List of bead types for each site in a molecule
        displacement_fn: JAX-MD displacement function. Used for all pairwise distance
            calculations so the coordinate system of trajectories does not matter.
        sites_per_mol: Number of sites per molecule
        box_length: Smallest real-space box side length (nm); sets r_max = box_length / 2.
        box_volume: True box volume (nm^3) for shell-volume normalisation. Defaults to
            box_length**3, which is exact for cubic boxes.
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

    # JAX-optimized distance calculation via displacement_fn (handles any coord system)
    @jit
    def compute_distances(positions, i_idx, j_idx):
        pos_i = positions[:, i_idx, :]  # (F, P, 3)
        pos_j = positions[:, j_idx, :]  # (F, P, 3)
        def _frame(pi, pj):
            disps = vmap(displacement_fn)(pj, pi)  # (P, 3)
            return jnp.sqrt(jnp.sum(disps ** 2, axis=-1))  # (P,)
        return vmap(_frame)(pos_i, pos_j)  # (F, P)

    # Histogram computation
    @jit
    def compute_histogram(dists, bins):
        """Compute histogram of distances."""
        return jnp.histogram(dists.ravel(), bins=bins)[0]

    volume = box_volume if box_volume is not None else box_length**3
    r_max = box_length / 2
    bins_arr = jnp.arange(0.0, r_max + dr, dr)
    shell_volumes = (4.0 / 3.0) * jnp.pi * (bins_arr[1:] ** 3 - bins_arr[:-1] ** 3)

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

            # Process pairs in batches
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
                    dists = compute_distances(positions_f, i_batch, j_batch)
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
