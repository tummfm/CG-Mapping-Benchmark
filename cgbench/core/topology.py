"""CG topology index generation from GROMACS topology files.

This module provides utilities to parse atomistic GROMACS topology files and,
combined with a CG mapping, derive the bond, angle, dihedral, and nonbonded
index arrays required by EspalomaCG and related models.

Typical usage::

    from cgbench.core.topology import get_cg_indices
    from cgbench.core.mapping import Ala2_Map

    map_obj = Ala2_Map()
    bond_idx, angle_idx, dihedral_idx, nonbonded_idx = get_cg_indices(
        "ala2", map_obj, "coreBeta"
    )
"""

from __future__ import annotations

import os
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from .mapping import compute_cg_bond_types


# ---------------------------------------------------------------------------
# Registry of topology files (one per dataset, excluding CATH)
# ---------------------------------------------------------------------------

TOPOLOGY_FILES: dict[str, str] = {
    "ala2":  "/ds/project/franz/gro/ala2/top/topol.top",
    "ala15": "/ds/project/franz/gro/L-Ala15_500ns_unconstrained/top/topol.top",
    "gly2":  "/ds/project/franz/gro/gly2/top/topol.top",
    "pro2":  "/ds/project/franz/gro/pro2/top/topol.top",
    "thr2":  "/ds/project/franz/gro/thr2/top/topol.top",
    "hexane": "/ds/project/franz/gro/hexane_ttot=100ns_dt=1fs_nstxout=200/topol.top",
}


# ---------------------------------------------------------------------------
# GROMACS topology parser
# ---------------------------------------------------------------------------

def parse_gromacs_top(
    top_file: str | Path,
) -> tuple[list[tuple[int, int]], int]:
    """Parse a GROMACS ``.top`` file and return atomistic bonds.

    Only the *first* ``[ bonds ]`` section found in the file is parsed
    (i.e., the bonds of the primary solute molecule).  Solvent / ion
    definitions pulled in via ``#include`` directives are not resolved
    and therefore do not appear.

    Args:
        top_file: Path to the ``.top`` file.

    Returns:
        bonds: 0-indexed bond pairs ``[(i, j), ...]``.
        n_atoms: Number of atoms in the first molecule type (= highest
            atom index seen in the bonds section, since GROMACS numbers
            atoms starting at 1 and contiguously within a moleculetype).
    """
    top_file = Path(top_file)
    if not top_file.exists():
        raise FileNotFoundError(f"Topology file not found: {top_file}")

    bonds: list[tuple[int, int]] = []
    n_atoms: int = 0
    in_atoms_section = False
    in_bonds_section = False
    bonds_done = False  # stop after the first [bonds] block

    with open(top_file) as fh:
        for raw_line in fh:
            line = raw_line.split(";")[0].strip()  # strip inline comments

            if not line:
                continue

            # Detect section headers
            section_match = re.match(r"^\[\s*(\w+)\s*\]", line)
            if section_match:
                section = section_match.group(1).lower()
                in_atoms_section = section == "atoms" and not bonds_done
                in_bonds_section = section == "bonds" and not bonds_done
                if section == "bonds" and bonds_done:
                    break  # second [bonds] → different molecule type, stop
                if bonds_done and section not in ("bonds",):
                    pass  # keep scanning until we'd hit a second bonds
                continue

            if in_atoms_section:
                tokens = line.split()
                if tokens and tokens[0].isdigit():
                    n_atoms = max(n_atoms, int(tokens[0]))
                continue

            if in_bonds_section:
                tokens = line.split()
                if len(tokens) >= 2 and tokens[0].isdigit():
                    ai, aj = int(tokens[0]) - 1, int(tokens[1]) - 1  # 0-indexed
                    bonds.append((ai, aj))
                    n_atoms = max(n_atoms, ai + 1, aj + 1)
                continue

    if not bonds:
        raise ValueError(f"No bonds found in topology file: {top_file}")

    return bonds, n_atoms


# ---------------------------------------------------------------------------
# CG index derivation
# ---------------------------------------------------------------------------

def _derive_cg_dihedrals(
    bond_0: np.ndarray,
) -> np.ndarray:
    """Derive proper dihedral index quadruples ``(4, D)`` from bond_0.

    All connected 4-atom chains ``(i–j–k–l)`` in the CG bond graph are
    enumerated.  The canonical form keeps ``i < l`` to avoid storing both
    ``(i, j, k, l)`` and its reverse ``(l, k, j, i)``.

    Args:
        bond_0: ``(B, 2)`` array of direct CG bonds.

    Returns:
        ``(4, D)`` int32 array of dihedral quadruples, or ``(4, 0)`` if
        the graph has fewer than 4 nodes or no valid chains.
    """
    if bond_0.shape[0] == 0:
        return np.zeros((4, 0), dtype=np.int32)

    neighbors: dict[int, list[int]] = defaultdict(list)
    for i, j in bond_0:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    seen: set[tuple[int, int, int, int]] = set()
    dihedrals: list[tuple[int, int, int, int]] = []

    for j, k in bond_0:
        j, k = int(j), int(k)
        for i in neighbors[j]:
            if i == k:
                continue
            for l in neighbors[k]:
                if l == j or l == i:
                    continue
                # canonical: smallest endpoint first
                canon = (i, j, k, l) if i < l else (l, k, j, i)
                if canon not in seen:
                    seen.add(canon)
                    dihedrals.append(canon)

    if not dihedrals:
        return np.zeros((4, 0), dtype=np.int32)

    return np.array(dihedrals, dtype=np.int32).T  # (4, D)


def get_cg_indices(
    dataset_name: str,
    map_obj,
    map_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Derive EspalomaCG-ready CG topology index arrays.

    Reads atomistic bonds from the GROMACS topology for *dataset_name*,
    tiles them to match the full system size described by *map_obj*, calls
    :func:`~cgbench.core.mapping.compute_cg_bond_types` with the
    mapping indices for *map_name*, and returns four index arrays.

    Args:
        dataset_name: Key in :data:`TOPOLOGY_FILES` (e.g. ``"ala2"``).
        map_obj: An instantiated mapping class (e.g. :class:`Ala2_Map`).
            Must expose an ``at_masses`` attribute and a
            ``get_map(map_name)`` method whose first return value is the
            per-atom CG-site index list.
        map_name: Name of the CG strategy (e.g. ``"coreBeta"``).

    Returns:
        cg_bond_index:      ``(2, B)``  direct CG bonds.
        cg_angle_index:     ``(3, A)``  valence angle triples
                            ``[i, j_central, k]``, ``i < k``.
        cg_dihedral_index:  ``(4, D)``  proper dihedral quadruples
                            ``[i, j, k, l]``, canonical form.
        cg_nonbonded_index: ``(2, NB)`` 1-4 and 1-5 CG pairs, or
                            ``None`` when there are none.

    Raises:
        KeyError: If *dataset_name* is not in :data:`TOPOLOGY_FILES`.
        FileNotFoundError: If the topology file does not exist.
        ValueError: If *map_name* is not available in *map_obj*.
    """
    if dataset_name not in TOPOLOGY_FILES:
        raise KeyError(
            f"No topology registered for '{dataset_name}'. "
            f"Available: {sorted(TOPOLOGY_FILES)}"
        )

    # ---- 1. Parse topology ------------------------------------------------
    top_file = TOPOLOGY_FILES[dataset_name]
    bonds_single, n_atoms_per_mol = parse_gromacs_top(top_file)

    # ---- 2. Tile to full system size -------------------------------------
    n_total_atoms = len(map_obj.at_masses)
    if n_total_atoms % n_atoms_per_mol != 0:
        raise ValueError(
            f"Total atom count ({n_total_atoms}) is not a multiple of the "
            f"per-molecule atom count ({n_atoms_per_mol}) from {top_file}. "
            "Check that the topology file matches the mapping class."
        )
    n_replicas = n_total_atoms // n_atoms_per_mol

    at_bonds: list[tuple[int, int]] = []
    for rep in range(n_replicas):
        offset = rep * n_atoms_per_mol
        for a, b in bonds_single:
            at_bonds.append((a + offset, b + offset))

    # ---- 3. Get mapping indices -------------------------------------------
    map_result = map_obj.get_map(map_name)
    mapping_indices: list[int] = map_result[0]  # first return value

    # ---- 4. Compute CG bond types via BFS --------------------------------
    bond_types = compute_cg_bond_types(at_bonds, mapping_indices)

    # ---- 5. Derive index arrays ------------------------------------------
    bond_0 = np.asarray(bond_types.get("bond_0", []), dtype=np.int32)  # (B, 2)

    # Bond index [2, B]
    cg_bond_index = (
        bond_0.T if bond_0.shape[0] > 0 else np.zeros((2, 0), dtype=np.int32)
    )

    # Angle index [3, A]
    neighbors: dict[int, list[int]] = defaultdict(list)
    for i, j in bond_0:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    angles: list[tuple[int, int, int]] = []
    for j, nbrs in neighbors.items():
        nbrs_sorted = sorted(nbrs)
        for a in range(len(nbrs_sorted)):
            for b in range(a + 1, len(nbrs_sorted)):
                angles.append((nbrs_sorted[a], j, nbrs_sorted[b]))

    cg_angle_index = (
        np.array(angles, dtype=np.int32).T
        if angles
        else np.zeros((3, 0), dtype=np.int32)
    )

    # Dihedral index [4, D]
    cg_dihedral_index = _derive_cg_dihedrals(bond_0)

    # Nonbonded index [2, NB]: 1-4 (bond_2) and 1-5 (bond_3) CG pairs
    nb_parts = [
        np.asarray(bond_types[k], dtype=np.int32)
        for k in ("bond_2", "bond_3")
        if k in bond_types and len(bond_types[k]) > 0
    ]
    cg_nonbonded_index = (
        np.concatenate(nb_parts, axis=0).T if nb_parts else None
    )

    return cg_bond_index, cg_angle_index, cg_dihedral_index, cg_nonbonded_index


# ---------------------------------------------------------------------------
# Element / atom-name utilities
# ---------------------------------------------------------------------------

_ELEMENT_SYMBOLS: dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S",
    17: "Cl", 11: "Na", 12: "Mg", 19: "K", 20: "Ca",
}


def element_symbol(atomic_number: int) -> str:
    """Return the element symbol for a given atomic number (e.g. 6 → ``'C'``)."""
    return _ELEMENT_SYMBOLS.get(int(atomic_number), "X")


def _element_from_atom_name(name: str) -> str:
    """Derive a PDB element symbol from a GROMACS atom name (e.g. ``'CA'`` → ``' C'``)."""
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
    """Format atom name to fit PDB columns 13–16 (4 characters)."""
    name = name.strip()
    if len(name) < 4:
        return f" {name:<3s}"
    return f"{name:<4s}"


def _cryst1_record(box_nm: np.ndarray) -> str:
    """Return a CRYST1 line from a 3×3 box matrix in nm (converted to Å internally)."""
    box = np.asarray(box_nm, dtype=float) * 10.0  # nm → Å
    if box.shape == (3, 3):
        diag = np.diag(box)
        offdiag = box - np.diag(diag)
        if np.allclose(offdiag, 0.0, atol=1e-4):
            a, b, c = diag
            return (
                f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"
            )
        # Triclinic: derive crystallographic parameters from row vectors
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
    # 1-D box lengths
    flat = np.asarray(box_nm).ravel() * 10.0
    a, b, c = float(flat[0]), float(flat[1]), float(flat[2])
    return f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}  90.00  90.00  90.00 P 1           1\n"


# ---------------------------------------------------------------------------
# Per-atom / per-bead metadata builder
# ---------------------------------------------------------------------------

def _load_residue_maps_topology() -> dict:
    """Load residue_maps.json from the project data directory."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "residue_maps.json")
    with open(path) as f:
        return json.load(f)


def get_atom_info(
    map_obj,
    map_name: str | None = None,
    cg: bool = False,
) -> list[dict]:
    """Build per-atom/bead metadata dicts for PDB and XYZ writing.

    For :class:`~cgbench.core.mapping.CappedPeptideMap` instances (detected
    via the ``_residue_sequence`` attribute), uses ``residue_maps.json`` to
    assign correct GROMACS atom names (e.g. ``CA``, ``CB``) and three-letter
    residue names (e.g. ``ALA``, ``ACE``).  For other map types a generic
    fallback is used.

    Args:
        map_obj:  Instantiated mapping class.
        map_name: CG strategy name (e.g. ``"coreBetaMap2"``).  Required when
                  *cg* is ``True``.
        cg:       ``True`` → one entry per CG bead; ``False`` → one entry per
                  all-atom site.

    Returns:
        List of dicts with keys ``res_id`` (1-based ``int``), ``res_name``
        (``str``), ``atom_name`` (``str``), and ``element`` (``str``).
    """
    # ---- CappedPeptideMap: has _residue_sequence ----
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

                # First contributing atom determines the bead name / element
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

    # ---- Generic fallback for other map types ----
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
    else:
        if map_name is None:
            raise ValueError("map_name is required when cg=True")
        # Infer CG count from the map's stored data
        if hasattr(map_obj, "_maps") and map_name in map_obj._maps:
            n_cg = int(len(map_obj._maps[map_name]["cg_species"]))
        else:
            indices, cg_species, _, _ = map_obj.get_map(map_name)
            n_cg = int(len(cg_species))
        return [
            {"res_id": 1, "res_name": "MOL", "atom_name": f"B{i + 1}", "element": "C"}
            for i in range(n_cg)
        ]


# ---------------------------------------------------------------------------
# Bond-index normalisation helper
# ---------------------------------------------------------------------------

def _normalise_bond_index(bond_index) -> np.ndarray | None:
    """Return bond connectivity as a ``(B, 2)`` int32 array, or ``None``."""
    if bond_index is None:
        return None
    b = np.asarray(bond_index, dtype=np.int32)
    if b.ndim != 2 or b.size == 0:
        return None
    # Detect (2, B) layout where B ≠ 2
    if b.shape[0] == 2 and b.shape[1] != 2:
        b = b.T  # (2, B) → (B, 2)
    return b


# ---------------------------------------------------------------------------
# PDB trajectory writer with CONECT records
# ---------------------------------------------------------------------------

def write_pdb_trajectory_with_bonds(
    path: str,
    positions_nm: np.ndarray,
    atom_info: list[dict],
    bond_index=None,
    box_nm: np.ndarray | None = None,
    n_frames: int | None = None,
) -> None:
    """Write a multi-model PDB trajectory including CONECT records for bonds.

    Positions are expected in **nm** and are converted to Å internally
    (multiplied by 10).

    Args:
        path:          Output ``.pdb`` file path.
        positions_nm:  Coordinate array ``(n_frames, n_atoms, 3)`` in nm.
        atom_info:     Per-atom metadata from :func:`get_atom_info`.
        bond_index:    Bond connectivity in ``(B, 2)`` or ``(2, B)`` format
                       (0-indexed).  ``None`` omits CONECT records.
        box_nm:        3×3 box matrix (nm) used for the CRYST1 header.
        n_frames:      Write only the first *n_frames* frames when given.
    """
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
                x, y, z = positions_nm[fi, ai] * 10.0  # nm → Å
                name_field = _format_atom_name_pdb(info["atom_name"])
                res_name_3 = f"{info['res_name'][:3]:3s}"
                res_id = int(info["res_id"])
                elem = info.get("element", _element_from_atom_name(info["atom_name"]))
                f.write(
                    f"ATOM  {ai + 1:5d} {name_field} {res_name_3} A{res_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
                )
            f.write("ENDMDL\n")

        # CONECT records written once after all MODEL blocks
        if bonds is not None and bonds.shape[0] > 0:
            adj: dict[int, list[int]] = defaultdict(list)
            for i, j in bonds:
                adj[int(i)].append(int(j))
                adj[int(j)].append(int(i))
            for atom_idx in sorted(adj.keys()):
                bonded = sorted(set(adj[atom_idx]))
                serial = atom_idx + 1  # PDB is 1-indexed
                for start in range(0, len(bonded), 4):
                    chunk = bonded[start : start + 4]
                    bonded_str = "".join(f"{b + 1:5d}" for b in chunk)
                    f.write(f"CONECT{serial:5d}{bonded_str}\n")

        f.write("END\n")

    print(f"  Wrote {total_frames} frames × {n_atoms} atoms/beads  →  {path}")


# ---------------------------------------------------------------------------
# PBC unwrapping
# ---------------------------------------------------------------------------

def unwrap_trajectory(
    positions_nm: np.ndarray,
    displacement_fn,
    bond_index=None,
) -> np.ndarray:
    """Remove PBC artifacts from a trajectory.

    Two modes are supported depending on whether a bond graph is supplied:

    **Make-whole** (``bond_index`` given, recommended):
        For each frame independently, atoms are traversed in BFS order
        starting from atom 0.  Each atom is placed at its minimum-image
        position relative to its already-placed bonded neighbour, so the
        molecule is made "whole" regardless of box crossings.
        Displacements for all frames are batched via ``jax.vmap``, so only
        *N_atoms* JAX calls are made (not N_frames × N_atoms).

    **Temporal unwrapping** (fallback when ``bond_index`` is ``None``):
        Consecutive frames are linked by accumulating minimum-image
        displacements.  Meaningful only when the supplied frames are
        temporally consecutive.

    Args:
        positions_nm:  ``(T, N, 3)`` float array in nm (Cartesian).
        displacement_fn: jax_md displacement function
                         ``dr = fn(ra, rb)`` for single ``(3,)`` vectors.
        bond_index:    Bond connectivity ``(B, 2)`` or ``(2, B)`` (0-indexed).
                       ``None`` → temporal fallback.

    Returns:
        ``(T, N, 3)`` numpy array in nm with PBC artifacts removed.
    """
    import jax
    import jax.numpy as jnp

    positions = np.asarray(positions_nm, dtype=np.float64)
    T, N, _ = positions.shape

    bonds = _normalise_bond_index(bond_index)

    # Displacement batched over T frames: (T,3),(T,3) -> (T,3)
    _batch_disp = jax.jit(jax.vmap(displacement_fn))

    if bonds is None or bonds.shape[0] == 0:
        # ---- Temporal unwrapping (no bond graph) ----
        # Batch over N atoms: (N,3),(N,3) -> (N,3)
        _batch_disp_N = jax.jit(jax.vmap(displacement_fn))
        unwrapped = np.empty_like(positions)
        unwrapped[0] = positions[0]
        for t in range(1, T):
            disp = np.asarray(
                _batch_disp_N(jnp.asarray(positions[t]), jnp.asarray(positions[t - 1]))
            )
            unwrapped[t] = unwrapped[t - 1] + disp
        return unwrapped

    # ---- Make-whole (BFS per frame, batched over T) ----

    # Build adjacency list
    adj: dict[int, list[int]] = defaultdict(list)
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))

    # BFS traversal order + parent (computed once, topology-fixed)
    bfs_order: list[int] = []
    bfs_parent: list[int] = []
    visited = np.zeros(N, dtype=bool)

    for start in range(N):
        if not visited[start]:
            q: deque[int] = deque([start])
            visited[start] = True
            bfs_order.append(start)
            bfs_parent.append(start)  # root points to itself
            while q:
                node = q.popleft()
                for nbr in adj[node]:
                    if not visited[nbr]:
                        visited[nbr] = True
                        bfs_order.append(nbr)
                        bfs_parent.append(node)
                        q.append(nbr)

    # Process atoms in BFS order; for each atom batch displacement over T frames
    unwrapped = np.copy(positions)  # roots keep their original positions

    for k in range(1, len(bfs_order)):
        i = bfs_order[k]
        p = bfs_parent[k]
        # pos[i] = original wrapped positions; unwrapped[p] = already-placed parent
        disp = np.asarray(
            _batch_disp(
                jnp.asarray(positions[:, i, :]),   # (T, 3) wrapped
                jnp.asarray(unwrapped[:, p, :]),   # (T, 3) already placed
            )
        )
        unwrapped[:, i, :] = unwrapped[:, p, :] + disp

    return unwrapped


# ---------------------------------------------------------------------------
# XYZ trajectory writer
# ---------------------------------------------------------------------------

def write_xyz_trajectory(
    path: str,
    positions_nm: np.ndarray,
    atom_info: list[dict],
    n_frames: int | None = None,
    workers: int = 1,
) -> None:
    """Write a multi-frame XYZ trajectory file.

    Positions are expected in **nm** and are converted to Å internally
    (multiplied by 10).

    Args:
        path:          Output ``.xyz`` file path.
        positions_nm:  Coordinate array ``(n_frames, n_atoms, 3)`` in nm.
        atom_info:     Per-atom metadata from :func:`get_atom_info`.
        n_frames:      Write only the first *n_frames* frames when given.
        workers:       Number of worker processes for parallel formatting.
                       Defaults to 1 (sequential) to avoid ``os.fork()``
                       deadlocks when JAX is already initialised.
    """
    from ..utils.io import save_xyz_frames_parallel

    positions_nm = np.asarray(positions_nm, dtype=float)
    if n_frames is not None:
        positions_nm = positions_nm[:n_frames]
    positions_ang = positions_nm * 10.0  # nm → Å
    species_list = [info["element"] for info in atom_info]

    save_xyz_frames_parallel(positions_ang, species_list, path, workers=workers)
    print(
        f"  Wrote {positions_ang.shape[0]} frames × {positions_ang.shape[1]} "
        f"atoms/beads  →  {path}"
    )
