"""
File I/O utilities for trajectory loading, XYZ writing, and output directory preparation.
"""

import os
import io
import pickle as pkl
import numpy as np
from jax import numpy as jnp
from concurrent.futures import ProcessPoolExecutor


def prepare_output_dir(traj_path: str) -> str:
    """
    Create an output directory named 'plots' next to a trajectory file or inside
    a trajectory directory.

    Ensures that a directory called 'plots' exists alongside the given
    trajectory file path (or inside the directory). If it does not exist, it is created.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory file or a trajectory directory.

    Returns
    -------
    str
        Path to the 'plots' directory where outputs will be saved.
    """
    base = traj_path if os.path.isdir(traj_path) else os.path.dirname(traj_path)
    outdir = os.path.join(base, "plots")
    os.makedirs(outdir, exist_ok=True)
    return outdir


def load_trajectory(traj_path: str) -> tuple[jnp.ndarray, dict]:
    """
    Load trajectory coordinates and auxiliary state from pickle files.

    Opens 'trajectory.pkl' and 'traj_state_aux.pkl' either in the given directory
    or in the directory containing the given file path.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory directory or to one of the trajectory pickle files.

    Returns
    -------
    tuple[jnp.ndarray, dict]
        traj : JAX array of shape (n_frames, n_particles, 3)
            Simulation trajectory coordinates.
        aux : dict
            Auxiliary state information (energy, temperature, etc.).
    """
    base = traj_path if os.path.isdir(traj_path) else os.path.dirname(traj_path)
    traj = pkl.load(open(os.path.join(base, "trajectory.pkl"), "rb"))
    aux = pkl.load(open(os.path.join(base, "traj_state_aux.pkl"), "rb"))
    return jnp.array(traj), aux


def _format_xyz_frame(args):
    """Worker: format one frame to an XYZ string."""
    frame_idx, positions_frame, species_col = args
    n_atoms = positions_frame.shape[0]
    buf = io.StringIO()
    buf.write(f"{n_atoms}\nFrame {frame_idx + 1}\n")
    data = np.c_[species_col, positions_frame]
    np.savetxt(buf, data, fmt="%s %.6f %.6f %.6f")
    return buf.getvalue()


def save_xyz_frames_parallel(
    positions,
    species_list,
    filename,
    workers=None,
    chunksize=8,
    buffer_bytes=1_048_576,
):
    """
    Parallel XYZ writer.
    - Parallelizes CPU-bound text formatting per frame with processes.
    - Preserves frame order in the output file.
    - Streams results to disk to avoid large memory spikes.

    positions: (n_frames, n_atoms, 3) float array
    species_list: list[str] length n_atoms
    """
    positions = np.asarray(positions)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3)")

    n_frames, n_atoms, _ = positions.shape
    if len(species_list) != n_atoms:
        raise ValueError(
            f"Species list length ({len(species_list)}) must match number of atoms ({n_atoms})"
        )

    # Cache species column once; small and cheap to pickle
    species_col = np.asarray(species_list, dtype=object).reshape(-1, 1)

    # Small datasets don't benefit from process spin-up
    if workers == 1 or n_frames < 4:
        with open(filename, "w", buffering=buffer_bytes) as f:
            for frame_idx in range(n_frames):
                f.write(
                    _format_xyz_frame((frame_idx, positions[frame_idx], species_col))
                )
        return

    # Parallel formatting
    with open(filename, "w", buffering=buffer_bytes) as f, ProcessPoolExecutor(
        max_workers=workers
    ) as ex:
        iterable = ((i, positions[i], species_col) for i in range(n_frames))
        for frame_str in ex.map(_format_xyz_frame, iterable, chunksize=chunksize):
            f.write(frame_str)


def scale_dataset(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm."""
    print(f"Original positions: {dataset['R'].min():.4f} to {dataset['R'].max():.4f}")

    if fractional:
        box = dataset["box"][0, 0, 0]
        dataset["R"] = dataset["R"] / box
    else:
        dataset["R"] = dataset["R"] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset["box"] = scale_R * dataset["box"]
    dataset["F"] *= scale_F

    return dataset


def scale_dataset_non_cubic(dataset, scale_R, scale_U, fractional=True):
    """Scales the dataset to kJ/mol and to nm.
    
    Handles arbitrary triclinic boxes via fractional coordinate transform.
    box shape assumed: (n_frames, 3, 3) where rows are lattice vectors.
    """
    print(f"Original positions: {dataset['R'].min():.4f} to {dataset['R'].max():.4f}")

    if fractional:
        # box: (n_frames, 3, 3) — each frame has a 3x3 matrix of lattice vectors
        # R:   (n_frames, n_atoms, 3)
        box = dataset["box"]  # (F, 3, 3)

        # Convert to fractional: s = R @ box^{-1}
        # box_inv: (F, 3, 3), R: (F, N, 3)
        box_inv = np.linalg.inv(box)  # (F, 3, 3)
        # einsum: for each frame f, atom n: s[f,n,:] = R[f,n,:] @ box_inv[f,:,:]
        dataset["R"] = np.einsum("fni,fij->fnj", dataset["R"], box_inv)
    else:
        dataset["R"] = dataset["R"] * scale_R

    print(f"Scale dataset by {scale_R} for R and {scale_U} for U.")

    scale_F = scale_U / scale_R
    dataset["box"] = scale_R * dataset["box"]   # scales all lattice vector components
    dataset["F"] *= scale_F

    return dataset


def _is_cubic_box(box, atol=1e-8):
    """Return True when box is orthorhombic with equal side lengths."""
    box = np.asarray(box)
    if box.shape[-2:] != (3, 3):
        return False

    diag = np.diagonal(box, axis1=-2, axis2=-1)
    eye = np.eye(3, dtype=box.dtype)
    offdiag = box - np.einsum("...i,ij->...ij", diag, eye)

    return bool(
        np.allclose(offdiag, 0.0, atol=atol)
        and np.allclose(diag, diag[..., :1], atol=atol)
    )


def scale_dataset_box_aware(dataset, scale_R, scale_U, fractional=True):
    """Dispatch scaling based on whether the box is cubic or non-cubic."""
    if not fractional:
        print("scale_dataset_box_aware: fractional=False")
        return scale_dataset(dataset, scale_R, scale_U, fractional=False)

    if _is_cubic_box(dataset["box"]):
        print("scale_dataset_box_aware: detected cubic box")
        return scale_dataset(dataset, scale_R, scale_U, fractional=True)

    print("scale_dataset_box_aware: detected non-cubic box")
    return scale_dataset_non_cubic(dataset, scale_R, scale_U, fractional=True)