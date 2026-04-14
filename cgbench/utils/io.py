"""
File I/O utilities for trajectory loading, XYZ writing, and output directory preparation.
"""

import os
import io
import pickle as pkl
import numpy as np
from jax import numpy as jnp
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict, deque


_ATOMIC_NUMBER_BY_SYMBOL = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "SC": 21,
    "TI": 22,
    "V": 23,
    "CR": 24,
    "MN": 25,
    "FE": 26,
    "CO": 27,
    "NI": 28,
    "CU": 29,
    "ZN": 30,
    "BR": 35,
    "I": 53,
}


def _normalize_symbol(token: str | None) -> str | None:
    if token is None:
        return None
    clean = str(token).strip().upper()
    return clean if clean else None


def atomic_number_from_mda_atom(atom) -> int:
    """Resolve atomic number from MDAnalysis atom metadata.

    This function is strict by design: it only accepts exact element symbols
    provided by MDAnalysis atom attributes (``type`` or ``element``).
    """
    tokens = [
        _normalize_symbol(getattr(atom, "type", None)),
        _normalize_symbol(getattr(atom, "element", None)),
    ]

    for token in tokens:
        if token is not None and token in _ATOMIC_NUMBER_BY_SYMBOL:
            return _ATOMIC_NUMBER_BY_SYMBOL[token]

    raise ValueError(
        "Could not determine atomic number from MDAnalysis atom metadata. "
        f"atom.index={getattr(atom, 'index', '?')}, "
        f"atom.name={getattr(atom, 'name', '?')}, "
        f"atom.type={getattr(atom, 'type', None)}, "
        f"atom.element={getattr(atom, 'element', None)}"
    )


def parse_gro_box_matrix_nm(box_line: str) -> np.ndarray:
    """Parse a GRO box line into a 3x3 box matrix in nm."""
    vals = [float(v) for v in box_line.split()]
    if len(vals) == 3:
        return np.diag(np.asarray(vals, dtype=np.float32))
    if len(vals) == 9:
        # GROMACS triclinic order: v1x v2y v3z v1y v1z v2x v2z v3x v3y
        v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = vals
        return np.array(
            [
                [v1x, v1y, v1z],
                [v2x, v2y, v2z],
                [v3x, v3y, v3z],
            ],
            dtype=np.float32,
        )
    raise ValueError(
        "Unsupported GRO box format. Expected 3 or 9 floats, "
        f"got {len(vals)} in line: '{box_line.strip()}'."
    )


def read_last_gro_box_nm(config_path: str) -> np.ndarray:
    """Read final GRO line and return orthorhombic box lengths in nm."""
    with open(config_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"Empty GRO file: {config_path}")

    parts = lines[-1].split()
    if len(parts) < 3:
        raise ValueError(
            f"Could not parse box from final GRO line in {config_path}: '{lines[-1]}'"
        )
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def load_single_gro_snapshot_dataset(config_path: str) -> dict:
    """Load a single-frame GRO file into the project dataset dict format.

    Species are intentionally not guessed here; they are set to zeros and
    should be sourced from strict topology metadata loaders.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing GRO file: {config_path}")

    with open(config_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file (too few lines): {config_path}")

    try:
        n_atoms = int(lines[1].strip())
    except ValueError as e:
        raise ValueError(
            f"Could not parse atom count from second GRO line in {config_path}: '{lines[1]}'"
        ) from e

    expected_lines = n_atoms + 3
    if len(lines) < expected_lines:
        raise ValueError(
            f"Invalid GRO file: expected at least {expected_lines} lines for {n_atoms} atoms, "
            f"found {len(lines)} in {config_path}."
        )

    atom_lines = lines[2 : 2 + n_atoms]
    box_line = lines[2 + n_atoms]
    box = parse_gro_box_matrix_nm(box_line)

    coords = np.zeros((1, n_atoms, 3), dtype=np.float32)
    forces = np.zeros((1, n_atoms, 3), dtype=np.float32)

    for i, ln in enumerate(atom_lines):
        atom_name = ln[10:15].strip()
        if not atom_name:
            parts = ln.split()
            if len(parts) < 6:
                raise ValueError(
                    f"Could not parse atom line {i + 3} in {config_path}: '{ln}'"
                )
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
        else:
            x = float(ln[20:28])
            y = float(ln[28:36])
            z = float(ln[36:44])

        coords[0, i] = np.array([x, y, z], dtype=np.float32)

    return {
        "R": coords,
        "F": forces,
        "box": box[None, :, :],
        "species": np.zeros((1, n_atoms), dtype=np.int32),
        "mask": np.ones((1, n_atoms), dtype=bool),
    }


def _extract_local_bonded_indices(ag) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_to_local = {int(gidx): lidx for lidx, gidx in enumerate(ag.indices)}

    def _remap(indices_global: list[list[int]]) -> np.ndarray:
        return np.array(
            [[global_to_local[i] for i in row] for row in indices_global],
            dtype=np.int32,
        )

    try:
        bonds = (
            _remap([[b.atoms[0].index, b.atoms[1].index] for b in ag.bonds])
            if len(ag.bonds) > 0
            else np.zeros((0, 2), dtype=np.int32)
        )
    except Exception:
        bonds = np.zeros((0, 2), dtype=np.int32)

    try:
        angles = (
            _remap([[a.atoms[0].index, a.atoms[1].index, a.atoms[2].index] for a in ag.angles])
            if len(ag.angles) > 0
            else np.zeros((0, 3), dtype=np.int32)
        )
    except Exception:
        angles = np.zeros((0, 3), dtype=np.int32)

    try:
        dihedrals = (
            _remap(
                [
                    [d.atoms[0].index, d.atoms[1].index, d.atoms[2].index, d.atoms[3].index]
                    for d in ag.dihedrals
                ]
            )
            if len(ag.dihedrals) > 0
            else np.zeros((0, 4), dtype=np.int32)
        )
    except Exception:
        dihedrals = np.zeros((0, 4), dtype=np.int32)

    return bonds, angles, dihedrals


def load_tpr_topology_metadata(
    topology_path: str,
    config_path: str | None = None,
    selection: str = "all",
) -> dict:
    """Load strict atomistic metadata from a GROMACS .tpr topology.

    Returns a dict with atomistic species, box, names, and bonded indices.
    """
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Missing topology file: {topology_path}")
    if os.path.splitext(topology_path)[1].lower() != ".tpr":
        raise ValueError(
            f"Strict topology loading requires a .tpr file, got: {topology_path}"
        )

    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise ImportError(
            "MDAnalysis is required for strict topology loading. "
            "Install with `pip install MDAnalysis`."
        ) from e

    tried_errors: list[str] = []
    universe_loaders = [lambda: mda.Universe(topology_path, topology_format="TPR")]
    if config_path is not None:
        universe_loaders.append(
            lambda: mda.Universe(topology_path, config_path, topology_format="TPR")
        )

    u = None
    for loader in universe_loaders:
        try:
            u = loader()
            break
        except Exception as e:
            tried_errors.append(str(e))

    if u is None:
        raise ValueError(
            "Failed to load .tpr topology with MDAnalysis. "
            f"Errors: {' | '.join(tried_errors)}"
        )

    ag = u.select_atoms(selection) if selection != "all" else u.atoms
    if len(ag) == 0:
        raise ValueError(
            f"Selection '{selection}' returned zero atoms for topology {topology_path}."
        )

    species = np.asarray([atomic_number_from_mda_atom(a) for a in ag], dtype=np.int32)

    dims = getattr(u.trajectory.ts, "dimensions", None)
    if dims is not None and len(dims) >= 3:
        box = np.diag(np.asarray(dims[:3], dtype=np.float32) * 0.1)
    elif config_path is not None:
        box = load_single_gro_snapshot_dataset(config_path)["box"][0]
    else:
        raise ValueError(
            f"Could not determine simulation box from {topology_path}."
        )

    bonds, angles, dihedrals = _extract_local_bonded_indices(ag)

    return {
        "species": species,
        "box": np.asarray(box, dtype=np.float32),
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "atom_names": np.asarray(ag.names, dtype=object),
        "residue_names": np.asarray(ag.resnames, dtype=object),
        "residue_ids": np.asarray(ag.resids, dtype=np.int32),
        "n_atoms": int(len(ag)),
    }


def load_gromacs_to_dataset(
    topology_path: str,
    trajectory_path: str,
    selection: str = "protein",
) -> dict:
    """Load a GROMACS trajectory into the in-project dataset dict format."""
    try:
        import MDAnalysis as mda
    except ImportError as e:
        raise ImportError(
            "MDAnalysis is required for topology loading. "
            "Install with `pip install MDAnalysis`."
        ) from e

    u = mda.Universe(topology_path, trajectory_path)
    atoms = u.select_atoms(selection)
    n_frames = len(u.trajectory)
    n_atoms = len(atoms)

    if n_atoms == 0:
        raise ValueError(f"Selection '{selection}' returned zero atoms for {topology_path}.")

    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    forces = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    boxes = np.zeros((n_frames, 3, 3), dtype=np.float32)

    for fi, ts in enumerate(u.trajectory):
        coords[fi] = atoms.positions.astype(np.float32) * 0.1
        lx, ly, lz = ts.dimensions[:3]
        boxes[fi] = np.diag(np.array([lx, ly, lz], dtype=np.float32) * 0.1)

    species = np.asarray([atomic_number_from_mda_atom(a) for a in atoms], dtype=np.int32)

    return {
        "R": coords,
        "F": forces,
        "box": boxes,
        "species": np.tile(species[None, :], (n_frames, 1)),
        "mask": np.ones((n_frames, n_atoms), dtype=bool),
        "atom_names": np.asarray(atoms.names, dtype=object),
        "residue_names": np.asarray(atoms.resnames, dtype=object),
        "residue_ids": np.asarray(atoms.resids, dtype=np.int32),
    }


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


_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S",
    17: "Cl", 11: "Na", 12: "Mg", 19: "K", 20: "Ca",
}


def element_symbol(atomic_number: int) -> str:
    """Return the element symbol for a given atomic number."""
    return _ELEMENT_SYMBOLS.get(int(atomic_number), "X")


def _element_from_atom_name(name: str) -> str:
    clean = name.strip()
    i = 0
    while i < len(clean) and clean[i].isdigit():
        i += 1
    clean = clean[i:].upper()
    if not clean:
        return " C"
    first = clean[0]
    return f" {first}" if first in "HCNOSP" else " C"


def _format_atom_name_pdb(name: str) -> str:
    name = name.strip()
    if len(name) < 4:
        return f" {name:<3s}"
    return f"{name:<4s}"


def _cryst1_record(box_nm: np.ndarray) -> str:
    box = np.asarray(box_nm, dtype=float) * 10.0
    if box.shape == (3, 3):
        diag = np.diag(box)
        offdiag = box - np.diag(diag)
        if np.allclose(offdiag, 0.0, atol=1e-4):
            a, b, c = diag
            return (
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"
            )
        v1, v2, v3 = box
        a = float(np.linalg.norm(v1))
        b = float(np.linalg.norm(v2))
        c = float(np.linalg.norm(v3))
        cos_alpha = np.dot(v2, v3) / (b * c)
        cos_beta = np.dot(v1, v3) / (a * c)
        cos_gamma = np.dot(v1, v2) / (a * b)
        alpha = float(np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0))))
        beta = float(np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0))))
        gamma = float(np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0))))
        return (
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}"
            f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1           1\n"
        )

    flat = np.asarray(box_nm).ravel() * 10.0
    a, b, c = float(flat[0]), float(flat[1]), float(flat[2])
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"


def _load_residue_maps_topology() -> dict:
    import json

    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "residue_maps.json")
    with open(path) as f:
        return json.load(f)


def get_atom_info(
    map_obj,
    map_name: str | None = None,
    cg: bool = False,
) -> list[dict]:
    """Build per-atom/bead metadata dicts for PDB and XYZ writing."""
    if hasattr(map_obj, "_residue_sequence"):
        residue_maps = _load_residue_maps_topology()
        seq = map_obj._residue_sequence
        atom_info: list[dict] = []

        for ri, res in enumerate(seq):
            rd = residue_maps[res]
            gro_syms: list[str] = rd["gro_symbols"]

            if not cg:
                for sym in gro_syms:
                    atom_info.append(
                        {
                            "res_id": ri + 1,
                            "res_name": res,
                            "atom_name": sym,
                            "element": _element_from_atom_name(sym).strip(),
                        }
                    )
            else:
                if map_name is None:
                    raise ValueError("map_name is required when cg=True")
                cg_map = rd["cg_maps"][map_name]
                local_indices: list[int] = cg_map["indices"]
                n_local_cg: int = len(cg_map["cg_species"])

                bead_src: dict[int, str] = {}
                for ai, lcg in enumerate(local_indices):
                    if lcg >= 0 and lcg not in bead_src and ai < len(gro_syms):
                        bead_src[lcg] = gro_syms[ai]

                for lcg in range(n_local_cg):
                    name = bead_src.get(lcg, f"B{lcg + 1}")
                    atom_info.append(
                        {
                            "res_id": ri + 1,
                            "res_name": res,
                            "atom_name": name,
                            "element": _element_from_atom_name(name).strip(),
                        }
                    )
        return atom_info

    if not cg:
        n_at = len(map_obj.at_masses)
        if hasattr(map_obj, "_at_numbers"):
            return [
                {
                    "res_id": 1,
                    "res_name": "MOL",
                    "atom_name": element_symbol(int(z)),
                    "element": element_symbol(int(z)),
                }
                for z in map_obj._at_numbers
            ]
        return [
            {"res_id": 1, "res_name": "MOL", "atom_name": f"A{i + 1}", "element": "C"}
            for i in range(n_at)
        ]

    if map_name is None:
        raise ValueError("map_name is required when cg=True")
    if hasattr(map_obj, "_maps") and map_name in map_obj._maps:
        n_cg = int(len(map_obj._maps[map_name]["cg_species"]))
    else:
        _, cg_species, _, _ = map_obj.get_map(map_name)
        n_cg = int(len(cg_species))
    return [
        {"res_id": 1, "res_name": "MOL", "atom_name": f"B{i + 1}", "element": "C"}
        for i in range(n_cg)
    ]


def _normalise_bond_index(bond_index) -> np.ndarray | None:
    if bond_index is None:
        return None
    b = np.asarray(bond_index, dtype=np.int32)
    if b.ndim != 2 or b.size == 0:
        return None
    if b.shape[0] == 2 and b.shape[1] != 2:
        b = b.T
    return b


def write_pdb_trajectory_with_bonds(
    path: str,
    positions_nm: np.ndarray,
    atom_info: list[dict],
    bond_index=None,
    box_nm: np.ndarray | None = None,
    n_frames: int | None = None,
) -> None:
    positions_nm = np.asarray(positions_nm, dtype=float)
    if n_frames is not None:
        positions_nm = positions_nm[:n_frames]
    total_frames, n_atoms, _ = positions_nm.shape

    bonds = _normalise_bond_index(bond_index)

    with open(path, "w") as f:
        if box_nm is not None:
            f.write(_cryst1_record(np.asarray(box_nm)))

        for fi in range(total_frames):
            f.write(f"MODEL     {fi + 1:4d}\n")
            for ai, info in enumerate(atom_info):
                x, y, z = positions_nm[fi, ai] * 10.0
                name_field = _format_atom_name_pdb(info["atom_name"])
                res_name_3 = f"{info['res_name'][:3]:3s}"
                res_id = int(info["res_id"])
                elem = info.get("element", _element_from_atom_name(info["atom_name"]))
                f.write(
                    f"ATOM  {ai + 1:5d} {name_field} {res_name_3} A{res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            f.write("ENDMDL\n")

        if bonds is not None and bonds.shape[0] > 0:
            adj: dict[int, list[int]] = defaultdict(list)
            for i, j in bonds:
                adj[int(i)].append(int(j))
                adj[int(j)].append(int(i))
            for atom_idx in sorted(adj.keys()):
                bonded = sorted(set(adj[atom_idx]))
                serial = atom_idx + 1
                for start in range(0, len(bonded), 4):
                    chunk = bonded[start : start + 4]
                    bonded_str = "".join(f"{b + 1:5d}" for b in chunk)
                    f.write(f"CONECT{serial:5d}{bonded_str}\n")

        f.write("END\n")

    print(f"  Wrote {total_frames} frames x {n_atoms} atoms/beads  ->  {path}")


def unwrap_trajectory(
    positions_nm: np.ndarray,
    displacement_fn,
    bond_index=None,
) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    positions = np.asarray(positions_nm, dtype=np.float64)
    t_frames, n_atoms, _ = positions.shape

    bonds = _normalise_bond_index(bond_index)
    _batch_disp = jax.jit(jax.vmap(displacement_fn))

    if bonds is None or bonds.shape[0] == 0:
        _batch_disp_n = jax.jit(jax.vmap(displacement_fn))
        unwrapped = np.empty_like(positions)
        unwrapped[0] = positions[0]
        for t in range(1, t_frames):
            disp = np.asarray(_batch_disp_n(jnp.asarray(positions[t]), jnp.asarray(positions[t - 1])))
            unwrapped[t] = unwrapped[t - 1] + disp
        return unwrapped

    adj: dict[int, list[int]] = defaultdict(list)
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))

    bfs_order: list[int] = []
    bfs_parent: list[int] = []
    visited = np.zeros(n_atoms, dtype=bool)
    for start in range(n_atoms):
        if not visited[start]:
            q: deque[int] = deque([start])
            visited[start] = True
            bfs_order.append(start)
            bfs_parent.append(start)
            while q:
                node = q.popleft()
                for nbr in adj[node]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        bfs_order.append(nbr)
                        bfs_parent.append(node)
                        q.append(nbr)

    unwrapped = np.copy(positions)
    for k in range(1, len(bfs_order)):
        i = bfs_order[k]
        p = bfs_parent[k]
        disp = np.asarray(
            _batch_disp(
                jnp.asarray(positions[:, i, :]),
                jnp.asarray(unwrapped[:, p, :]),
            )
        )
        unwrapped[:, i, :] = unwrapped[:, p, :] + disp

    return unwrapped


def write_xyz_trajectory(
    path: str,
    positions_nm: np.ndarray,
    atom_info: list[dict],
    n_frames: int | None = None,
    workers: int = 1,
) -> None:
    positions_nm = np.asarray(positions_nm, dtype=float)
    if n_frames is not None:
        positions_nm = positions_nm[:n_frames]
    positions_ang = positions_nm * 10.0
    species_list = [info["element"] for info in atom_info]

    save_xyz_frames_parallel(positions_ang, species_list, path, workers=workers)
    print(
        f"  Wrote {positions_ang.shape[0]} frames x {positions_ang.shape[1]} "
        f"atoms/beads  ->  {path}"
    )