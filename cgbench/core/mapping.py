from jax import numpy as jnp, lax
import jax
import functools
import numpy as np
import json
import os
from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Core map_dataset
# ---------------------------------------------------------------------------


def map_dataset(
    position_dataset, displacement_fn, shift_fn, c_map, d_map=None, force_dataset=None
):
    """Maps fine-scaled positions and forces to a coarser scale.

    Uses the linear mapping from [Noid2008]_ to map fine-scaled positions and
    forces to coarse grained positions and forces via the relations:

    .. math::

        \\mathbf R_I = \\sum_{i \\in \\mathcal I_I} c_{Ii} \\mathbf r_i,\\quad \\text{and}

        \\mathbf{F}_I = \\sum_{i \\in \\mathcal I_I} \\frac{d_{Ii}}{c_{Ii}} \\mathbf f_i.

    Args:
        position_dataset: Dataset of fine-scaled positions.
        displacement_fn: Function to compute displacement (handles boundary conditions).
        shift_fn: Ensures produced coordinates remain in the box.
        c_map: Matrix defining the linear mapping of positions.
        d_map: Matrix defining the linear mapping of forces.
        force_dataset: Dataset of fine-scaled forces.

    Returns:
        Coarse-grained positions and (if provided) coarse-grained forces.

    References:
        .. [Noid2008] W. G. Noid et al.; J. Chem. Phys. 128 (24): 244114 (2008).
    """

    def _map_single(ipt, shift_fn, displacement_fn, c_map, d_map):
        pos, forc = ipt
        c_map /= jnp.sum(c_map, axis=1, keepdims=True)
        d_map /= jnp.sum(d_map, axis=1, keepdims=True)
        mask = c_map > 0.0

        ref_idx = jnp.argmax(c_map, axis=1)
        ref_positions = pos[ref_idx, :]

        disp = jax.vmap(lambda r: jax.vmap(lambda p: displacement_fn(p, r))(pos))(
            ref_positions
        )

        cg_disp = jnp.einsum("Ii,Iid->Id", c_map, disp)
        cg_pos = jax.vmap(shift_fn)(ref_positions, cg_disp)

        cg_forces = jnp.einsum("Ii, id ->Id", mask, forc)
        return cg_pos, cg_forces

    _map_single = functools.partial(
        _map_single,
        shift_fn=shift_fn,
        displacement_fn=displacement_fn,
        c_map=c_map,
        d_map=d_map,
    )

    return lax.map(_map_single, (position_dataset, force_dataset))


# ---------------------------------------------------------------------------
# Atomic look-up tables
# ---------------------------------------------------------------------------

mass_map = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}
atomic_number_map = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
atomic_number_map_reverse = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


# ---------------------------------------------------------------------------
# residue_maps.json loading
# ---------------------------------------------------------------------------


def _default_residue_maps_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "residue_maps.json"
    )


@functools.lru_cache(maxsize=1)
def _load_residue_maps_json() -> dict:
    with open(_default_residue_maps_path(), "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mapping weight computation
# ---------------------------------------------------------------------------


@jax.jit
def get_map_weights(
    map_arr: jnp.ndarray,
    at_masses_arr: jnp.ndarray,
    cg_masses: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mass-weighted mapping matrix from atom→CG-site assignment.

    Args:
        map_arr:      (n_atoms,) int32 - CG site index per atom; -1 = excluded.
        at_masses_arr:(n_atoms,) float32 - atomic masses.
        cg_masses:    (n_cg,)   float32 - total mass per CG site.

    Returns:
        weights: (n_cg, n_atoms) float32 - row-normalised (rows sum to 1).
    """
    valid = map_arr >= 0
    clipped = jnp.where(valid, map_arr, 0)
    onehot = jax.nn.one_hot(clipped, cg_masses.shape[0], dtype=jnp.float32).T
    onehot *= valid[None, :]
    per_atom_w = jnp.where(
        cg_masses[:, None] > 0, at_masses_arr[None, :] / cg_masses[:, None], 0.0
    )
    return onehot * per_atom_w


# ---------------------------------------------------------------------------
# CG topology helpers
# ---------------------------------------------------------------------------


def _derive_cg_topology(
    bond_0_pairs: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive CG topology arrays from direct CG bonds.

    Args:
        bond_0_pairs: List of [i, j] direct CG bond pairs.

    Returns:
        cg_bond_index:     (2, B) int32 array of direct bonds.
        cg_angle_index:    (3, A) int32 array of angle triples [i, j_central, k].
        cg_dihedral_index: (4, D) int32 array of dihedral quadruples.
    """
    neighbors: dict[int, list[int]] = defaultdict(list)
    for i, j in bond_0_pairs:
        neighbors[int(i)].append(int(j))
        neighbors[int(j)].append(int(i))

    bond_arr = (
        np.array(bond_0_pairs, dtype=np.int32).T
        if bond_0_pairs
        else np.zeros((2, 0), dtype=np.int32)
    )

    # Angles: for each central j, all pairs of its neighbours
    angles: list[list[int]] = []
    for j in sorted(neighbors):
        nbrs = sorted(set(neighbors[j]))
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append([nbrs[a], j, nbrs[b]])
    angle_arr = (
        np.array(angles, dtype=np.int32).T
        if angles
        else np.zeros((3, 0), dtype=np.int32)
    )

    # Dihedrals: extend each bond i-j to i_ext-i-j-j_ext
    seen: set[tuple] = set()
    dihedrals: list[list[int]] = []
    for i, j in bond_0_pairs:
        for i_ext in neighbors[i]:
            if i_ext == j:
                continue
            for j_ext in neighbors[j]:
                if j_ext == i or j_ext == i_ext:
                    continue
                fwd = (i_ext, i, j, j_ext)
                rev = (j_ext, j, i, i_ext)
                if rev not in seen:
                    seen.add(fwd)
                    dihedrals.append(list(fwd))
    dihedrals.sort()
    dihedral_arr = (
        np.array(dihedrals, dtype=np.int32).T
        if dihedrals
        else np.zeros((4, 0), dtype=np.int32)
    )

    return bond_arr, angle_arr, dihedral_arr


def _derive_cg_topology_from_atomistic_graph(
    at_bonds: list[tuple[int, int]] | np.ndarray,
    mapping_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive CG topology by contracting an atomistic bond graph.

    Steps:
    1) Build CG bond graph by mapping each atomistic bond (a, b) -> (I, J).
       If either endpoint is excluded (index < 0) or I == J, the edge is skipped.
    2) Deduplicate undirected CG bonds.
    3) Derive CG angles/dihedrals by iterating the resulting CG bond graph.

    Args:
        at_bonds: Atomistic undirected bonds as (N, 2) or list[(i, j)].
        mapping_indices: Atom->CG index assignment; -1 means excluded atom.

    Returns:
        (cg_bond_index, cg_angle_index, cg_dihedral_index) as column-major arrays.
    """
    print(f"[MAPPING] Deriving CG topology via Graph contraction from atomistic graph with {len(at_bonds)} bonds.")
    if at_bonds is None:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((3, 0), dtype=np.int32),
            np.zeros((4, 0), dtype=np.int32),
        )

    bond_arr = np.asarray(at_bonds, dtype=np.int32)
    if bond_arr.size == 0:
        return (
            np.zeros((2, 0), dtype=np.int32),
            np.zeros((3, 0), dtype=np.int32),
            np.zeros((4, 0), dtype=np.int32),
        )
    if bond_arr.ndim != 2 or bond_arr.shape[1] != 2:
        raise ValueError(
            f"Expected atomistic bonds with shape (N, 2), got {bond_arr.shape}."
        )

    n_atoms = len(mapping_indices)
    cg_bond_set: set[tuple[int, int]] = set()

    for ai, aj in bond_arr:
        i, j = int(ai), int(aj)
        if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms:
            raise ValueError(
                "Atomistic bond index out of range for provided mapping_indices: "
                f"({i}, {j}) with n_atoms={n_atoms}."
            )

        cg_i = int(mapping_indices[i])
        cg_j = int(mapping_indices[j])

        if cg_i < 0 or cg_j < 0 or cg_i == cg_j:
            continue

        cg_bond_set.add((min(cg_i, cg_j), max(cg_i, cg_j)))

    cg_bond_pairs = [list(p) for p in sorted(cg_bond_set)]
    return _derive_cg_topology(cg_bond_pairs)


def _tile_topology(
    single_bonds: list[list[int]],
    single_angles: list[list[int]],
    single_dihedrals: list[list[int]],
    n_sites_per_mol: int,
    n_mols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile per-molecule topology across n_mols replicas."""
    bonds, angles, dihedrals = [], [], []
    for m in range(n_mols):
        off = m * n_sites_per_mol
        bonds.extend([[i + off, j + off] for i, j in single_bonds])
        angles.extend([[i + off, j + off, k + off] for i, j, k in single_angles])
        dihedrals.extend(
            [[i + off, j + off, k + off, l + off] for i, j, k, l in single_dihedrals]
        )

    def _arr(lst, ncols):
        return (
            np.array(lst, dtype=np.int32).T
            if lst
            else np.zeros((ncols, 0), dtype=np.int32)
        )

    return _arr(bonds, 2), _arr(angles, 3), _arr(dihedrals, 4)


def _arr_topology(terms: list[list[int]], ncols: int) -> np.ndarray:
    return (
        np.array(terms, dtype=np.int32).T
        if terms
        else np.zeros((ncols, 0), dtype=np.int32)
    )


def _atomistic_angles_from_bonds(at_bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Build unique atomistic angles [i, j, k] from undirected bonds."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i, j in at_bonds:
        neighbors[int(i)].add(int(j))
        neighbors[int(j)].add(int(i))

    angles: list[list[int]] = []
    for j in sorted(neighbors):
        nbrs = sorted(neighbors[j])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.append([nbrs[a], j, nbrs[b]])
    return angles


def _atomistic_dihedrals_from_bonds(at_bonds: list[tuple[int, int]]) -> list[list[int]]:
    """Build unique atomistic dihedrals [i, j, k, l] from undirected bonds."""
    neighbors: dict[int, set[int]] = defaultdict(set)
    for i, j in at_bonds:
        neighbors[int(i)].add(int(j))
        neighbors[int(j)].add(int(i))

    seen: set[tuple[int, int, int, int]] = set()
    dihedrals: list[list[int]] = []
    for j in sorted(neighbors):
        for k in sorted(neighbors[j]):
            if j >= k:
                continue
            for i in sorted(neighbors[j]):
                if i == k:
                    continue
                for l in sorted(neighbors[k]):
                    if l == j or l == i:
                        continue
                    fwd = (i, j, k, l)
                    rev = (l, k, j, i)
                    canon = min(fwd, rev)
                    if canon not in seen:
                        seen.add(canon)
                        dihedrals.append(list(canon))
    dihedrals.sort()
    return dihedrals


def _project_atomistic_terms_to_cg(
    atom_terms: list[list[int]] | list[tuple[int, ...]],
    mapping_indices: list[int],
) -> list[list[int]]:
    """Project atomistic bonds/angles/dihedrals to CG indices and deduplicate."""
    if not atom_terms:
        return []

    size = len(atom_terms[0])
    if size not in (2, 3, 4):
        raise ValueError(f"Unsupported topology term size: {size}")

    uniq: set[tuple[int, ...]] = set()
    for term in atom_terms:
        cg_term = [int(mapping_indices[int(a)]) for a in term]
        if any(v < 0 for v in cg_term):
            continue
        if len(set(cg_term)) != size:
            continue

        tup = tuple(cg_term)
        if size == 2:
            canon = (min(tup[0], tup[1]), max(tup[0], tup[1]))
        else:
            canon = min(tup, tuple(reversed(tup)))
        uniq.add(canon)

    return [list(t) for t in sorted(uniq)]


def _slice_cg_topology_from_atomistic(
    at_bonds: list[tuple[int, int]],
    mapping_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project atomistic topology to CG by dropping removed atoms."""
    at_angles = _atomistic_angles_from_bonds(at_bonds)
    at_dihedrals = _atomistic_dihedrals_from_bonds(at_bonds)

    cg_bonds = _project_atomistic_terms_to_cg(at_bonds, mapping_indices)
    cg_angles = _project_atomistic_terms_to_cg(at_angles, mapping_indices)
    cg_dihedrals = _project_atomistic_terms_to_cg(at_dihedrals, mapping_indices)

    return (
        _arr_topology(cg_bonds, 2),
        _arr_topology(cg_angles, 3),
        _arr_topology(cg_dihedrals, 4),
    )


def _compute_bond_types_from_cg_bond_index(
    cg_bond_index: np.ndarray,
    max_sep: int = 4,
) -> dict[str, list[list[int]]]:
    """Compute bond_0..bond_3 buckets from direct CG bonds."""
    if cg_bond_index.size == 0:
        return {"bond_0": [], "bond_1": [], "bond_2": [], "bond_3": []}

    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in np.asarray(cg_bond_index, dtype=np.int32).T:
        ii, jj = int(i), int(j)
        if ii == jj:
            continue
        adj[ii].add(jj)
        adj[jj].add(ii)

    pair_min: dict[tuple[int, int], int] = {}
    for start in sorted(adj):
        dist: dict[int, int] = {start: 0}
        queue: deque[int] = deque([start])
        while queue:
            cur = queue.popleft()
            if dist[cur] >= max_sep:
                continue
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)

        for other, d in dist.items():
            if d == 0:
                continue
            p = (min(start, other), max(start, other))
            if p not in pair_min or d < pair_min[p]:
                pair_min[p] = d

    buckets: dict[int, list[list[int]]] = {0: [], 1: [], 2: [], 3: []}
    for (i, j), d in sorted(pair_min.items()):
        if 1 <= d <= 4:
            buckets[d - 1].append([i, j])

    return {
        "bond_0": buckets[0],
        "bond_1": buckets[1],
        "bond_2": buckets[2],
        "bond_3": buckets[3],
    }


def _build_explicit_residue_topology(
    residue_name: str,
    strategy: str,
    cg_map: dict,
    cg_offset: int,
    n_local_cg: int,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """Read explicit per-residue CG topology and offset indices to global space."""
    missing = [k for k in ("bonds", "angles", "dihedrals") if k not in cg_map]
    if missing:
        raise ValueError(
            f"mapping_type='com' requires explicit topology in residue_maps.json for "
            f"residue '{residue_name}', strategy '{strategy}'. Missing keys: {missing}"
        )

    def _shift_terms(key: str, size: int) -> list[list[int]]:
        shifted: list[list[int]] = []
        for term in cg_map.get(key, []):
            if len(term) != size:
                raise ValueError(
                    f"Invalid {key} term length for residue '{residue_name}', "
                    f"strategy '{strategy}': expected {size}, got {len(term)}"
                )
            vals = [int(v) for v in term]
            if any(v < 0 or v >= n_local_cg for v in vals):
                raise ValueError(
                    f"Out-of-range {key} term for residue '{residue_name}', "
                    f"strategy '{strategy}': {vals} (n_local_cg={n_local_cg})"
                )
            shifted.append([v + cg_offset for v in vals])
        return shifted

    return (
        _shift_terms("bonds", 2),
        _shift_terms("angles", 3),
        _shift_terms("dihedrals", 4),
    )


# ---------------------------------------------------------------------------
# Hexane_Map
# ---------------------------------------------------------------------------


class Hexane_Map:
    """CG mapping for liquid hexane (multi-molecule system).

    Each hexane molecule (20 atoms: CH3-CH2-CH2-CH2-CH2-CH3) can be mapped
    to 2-6 CG sites.  All CG topologies are linear chains.

    Available maps: six-site, four-site, three-site, two-site,
                    two-site-Map2, A3, A4.
    """

    _base_species = [
        "C",
        "H",
        "H",
        "H",  # CH3
        "C",
        "H",
        "H",  # CH2
        "C",
        "H",
        "H",  # CH2
        "C",
        "H",
        "H",  # CH2
        "C",
        "H",
        "H",  # CH2
        "C",
        "H",
        "H",
        "H",  # CH3
    ]

    # Atomistic bonds for a single hexane (20 atoms, 0-indexed)
    _at_bonds_single: list[tuple[int, int]] = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (7, 8),
        (7, 9),
        (7, 10),
        (10, 11),
        (10, 12),
        (10, 13),
        (13, 14),
        (13, 15),
        (13, 16),
        (16, 17),
        (16, 18),
        (16, 19),
    ]

    # (n_cg_per_mol, single-molecule index pattern, cg_species_single)
    _map_specs: dict[str, tuple] = {
        "six-site": (
            6,
            [0, -1, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5, -1, -1, -1],
            [2, 1, 1, 1, 1, 2],
        ),
        "four-site": (
            4,
            [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,
                -1,
                1,
                -1,
                -1,
                2,
                -1,
                -1,
                3,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [1, 2, 2, 1],
        ),
        "three-site": (
            3,
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 1],
        ),
        "three-site-noh": (
            3,
            [0, -1, -1, -1, 0, -1, -1, 1, -1, -1, 1, -1, -1, 2, -1, -1, 2, -1, -1, -1],
            [1, 2, 1],
        ),
        "two-site": (
            2,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1],
        ),
        "two-site-noh": (
            2,
            [0, -1, -1, -1, 0, -1, -1, 0, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1],
            [1, 1],
        ),
        "A3": (
            2,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 2],
        ),
        "A4": (
            3,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 3],
        ),
    }

    def __init__(self, nmol: int = 100):
        self.n_replicas = nmol
        self.at_masses = [mass_map[s] for s in self._base_species] * nmol
        at_bonds_global = self._tile_atomistic_bonds(
            self._at_bonds_single,
            n_atoms_per_mol=len(self._base_species),
        )

        self._maps: dict[str, dict] = {}
        for name, (n_cg, single_idx, cg_sp) in self._map_specs.items():
            indices = self._tile_indices(single_idx, n_cg)
            cg_species = np.array(cg_sp * nmol, dtype=np.int32)

            cg_bonds, cg_angles, cg_dihedrals = (
                _derive_cg_topology_from_atomistic_graph(at_bonds_global, indices)
            )

            self._maps[name] = {
                "indices": indices,
                "cg_species": cg_species,
                "cg_bonds": cg_bonds,
                "cg_angles": cg_angles,
                "cg_dihedrals": cg_dihedrals,
            }

    def _tile_indices(self, single: list[int], block_size: int) -> list[int]:
        result = []
        for block in range(self.n_replicas):
            offset = block * block_size
            for v in single:
                result.append(-1 if v < 0 else v + offset)
        return result

    def _tile_atomistic_bonds(
        self,
        single_bonds: list[tuple[int, int]],
        n_atoms_per_mol: int,
    ) -> list[tuple[int, int]]:
        bonds: list[tuple[int, int]] = []
        for m in range(self.n_replicas):
            off = m * n_atoms_per_mol
            bonds.extend([(i + off, j + off) for i, j in single_bonds])
        return bonds

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        assert jnp.allclose(jnp.sum(weights, axis=1), 1.0, atol=1e-6)
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]


# ---------------------------------------------------------------------------
# BenzeneCrystal_Map
# ---------------------------------------------------------------------------


class BenzeneCrystal_Map:
    """CG mapping for benzene crystal (multi-molecule system).

    Three CG beads per benzene molecule in a triangular arrangement:
    - bead 0: C1, C2, H1, H2
    - bead 1: C3, C4, H3, H4
    - bead 2: C5, C6, H5, H6
    """

    _base_species = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]

    # Single benzene atomistic bonds (12 atoms, 0-indexed)
    _at_bonds_single: list[tuple[int, int]] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 6),
        (1, 7),
        (2, 8),
        (3, 9),
        (4, 10),
        (5, 11),
    ]

    def __init__(self, nmol: int = 128):
        self.n_replicas = nmol
        self.at_masses = [mass_map[s] for s in self._base_species] * nmol
        at_bonds_global = self._tile_atomistic_bonds(
            self._at_bonds_single,
            n_atoms_per_mol=len(self._base_species),
        )

        single_indices = [0, 0, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1]
        indices = self._tile_indices(single_indices, block_size=3)
        cg_species = np.array([1, 1, 1] * nmol, dtype=np.int32)
        cg_bonds, cg_angles, cg_dihedrals = _derive_cg_topology_from_atomistic_graph(
            at_bonds_global,
            indices,
        )

        self._maps = {
            "three-site-adjacent": {
                "indices": indices,
                "cg_species": cg_species,
                "cg_bonds": cg_bonds,
                "cg_angles": cg_angles,
                "cg_dihedrals": cg_dihedrals,
            }
        }

    def _tile_indices(self, single: list[int], block_size: int) -> list[int]:
        result: list[int] = []
        for block in range(self.n_replicas):
            offset = block * block_size
            for v in single:
                result.append(-1 if v < 0 else v + offset)
        return result

    def _tile_atomistic_bonds(
        self,
        single_bonds: list[tuple[int, int]],
        n_atoms_per_mol: int,
    ) -> list[tuple[int, int]]:
        bonds: list[tuple[int, int]] = []
        for m in range(self.n_replicas):
            off = m * n_atoms_per_mol
            bonds.extend([(i + off, j + off) for i, j in single_bonds])
        return bonds

    @staticmethod
    def _normalize_map_name(name: str) -> str:
        if name == "three-side-adjacent":
            return "three-site-adjacent"
        return name

    def get_available_maps(self) -> list[str]:
        return list(self._maps) + ["three-side-adjacent"]

    def get_map(self, name: str = "three-site-adjacent") -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        name = self._normalize_map_name(name)
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str = "three-site-adjacent") -> tuple:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        name = self._normalize_map_name(name)
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]


# ---------------------------------------------------------------------------
# CappedPeptideMap
# ---------------------------------------------------------------------------


class CappedPeptideMap:
    """CG mapping for ACE/NME-capped peptides using residue_maps.json.

    Automatically builds all CG maps that are available for every residue in
    the given sequence (intersection of per-residue available strategies).

    The atomistic topology (masses, bonds) is loaded from residue_maps.json.

    Args:
        residue_sequence: Ordered list of residue names,
            e.g. ``["ACE", "ALA", "NME"]``.
    """

    def __init__(self, residue_sequence: list[str], mapping_type: str = "slice"):
        residue_maps = _load_residue_maps_json()
        if mapping_type not in ("slice", "com"):
            raise ValueError(
                f"Invalid mapping_type '{mapping_type}'. Choose one of ['slice', 'com']."
            )
        self.mapping_type = mapping_type

        for res in residue_sequence:
            if res not in residue_maps:
                raise ValueError(
                    f"Residue '{res}' not found in residue_maps.json. "
                    f"Available: {list(residue_maps.keys())}"
                )

        self._residue_sequence = residue_sequence

        # Per-residue atom counts and cumulative offsets
        atom_offsets: list[int] = [0]
        at_masses: list[float] = []
        at_numbers: list[int] = []
        for res in residue_sequence:
            rd = residue_maps[res]
            at_masses.extend(rd["masses"])
            at_numbers.extend(rd["atomic_numbers"])
            atom_offsets.append(atom_offsets[-1] + len(rd["masses"]))

        self.at_masses = at_masses
        self._at_numbers = at_numbers
        self._n_atoms = atom_offsets[-1]

        # Atomistic bonds: intra-residue + backbone inter-residue C→N
        at_bonds: list[tuple[int, int]] = []
        for ri, res in enumerate(residue_sequence):
            off = atom_offsets[ri]
            for li, lj in residue_maps[res].get("at_bonds", []):
                at_bonds.append((off + li, off + lj))
        for ri in range(len(residue_sequence) - 1):
            res_k = residue_sequence[ri]
            res_k1 = residue_sequence[ri + 1]
            c_idx = residue_maps[res_k].get("backbone_C_idx")
            n_idx = residue_maps[res_k1].get("backbone_N_idx")
            if c_idx is not None and n_idx is not None:
                at_bonds.append(
                    (atom_offsets[ri] + c_idx, atom_offsets[ri + 1] + n_idx)
                )
        self._at_bonds = at_bonds

        # Available strategies = intersection across all residues
        all_strategies = set.intersection(
            *[set(residue_maps[res]["cg_maps"].keys()) for res in residue_sequence]
        )

        self._maps: dict[str, dict] = {}
        for strategy in sorted(all_strategies):
            indices: list[int] = []
            cg_species: list[int] = []
            explicit_bonds: list[list[int]] = []
            explicit_angles: list[list[int]] = []
            explicit_dihedrals: list[list[int]] = []
            cg_offset = 0

            for ri, res in enumerate(residue_sequence):
                rd = residue_maps[res]
                cg_map = rd["cg_maps"][strategy]
                local_indices = cg_map["indices"]
                local_species = cg_map["cg_species"]
                for local_idx in local_indices:
                    indices.append(-1 if local_idx < 0 else int(local_idx) + cg_offset)
                cg_species.extend(int(s) for s in local_species)

                if self.mapping_type == "com":
                    b, a, d = _build_explicit_residue_topology(
                        residue_name=res,
                        strategy=strategy,
                        cg_map=cg_map,
                        cg_offset=cg_offset,
                        n_local_cg=len(local_species),
                    )
                    explicit_bonds.extend(b)
                    explicit_angles.extend(a)
                    explicit_dihedrals.extend(d)

                cg_offset += len(local_species)

            if self.mapping_type == "slice":
                cg_bonds, cg_angles, cg_dihedrals = _slice_cg_topology_from_atomistic(
                    at_bonds=at_bonds,
                    mapping_indices=indices,
                )
            else:
                cg_bonds = _arr_topology(explicit_bonds, 2)
                cg_angles = _arr_topology(explicit_angles, 3)
                cg_dihedrals = _arr_topology(explicit_dihedrals, 4)

            self._maps[strategy] = {
                "indices": indices,
                "cg_species": np.array(cg_species, dtype=np.int32),
                "cg_bonds": cg_bonds,
                "cg_angles": cg_angles,
                "cg_dihedrals": cg_dihedrals,
            }

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(
            row_sums, 1.0, atol=1e-5
        ), f"Weights don't sum to 1 (max err {float(jnp.max(jnp.abs(row_sums - 1))):.2e})"
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]


# ---------------------------------------------------------------------------
# TIP3P_Water_Map
# ---------------------------------------------------------------------------


class TIP3P_Water_Map:
    """CG mapping for TIP3P water: one bead per water molecule, no bonds.

    Maps:
    - ``"UnitedAtom"``: all three atoms (O, H, H) contribute to the bead.
    - ``"HeavyAtom"``:  only the oxygen atom contributes.
    """

    _at_masses_single = [15.999, 1.008, 1.008]  # O, H, H

    def __init__(self, n_mols: int):
        self.n_mols = n_mols
        self.at_masses = self._at_masses_single * n_mols

        empty_bonds = np.zeros((2, 0), dtype=np.int32)
        empty_angles = np.zeros((3, 0), dtype=np.int32)
        empty_dihedrals = np.zeros((4, 0), dtype=np.int32)

        ua_indices = []
        ha_indices = []
        for m in range(n_mols):
            ua_indices.extend([m, m, m])  # O, H1, H2 → bead m
            ha_indices.extend([m, -1, -1])  # O → bead m; H1, H2 excluded

        self._maps = {
            "UnitedAtom": {
                "indices": ua_indices,
                "cg_species": np.ones(n_mols, dtype=np.int32),
                "cg_bonds": empty_bonds,
                "cg_angles": empty_angles,
                "cg_dihedrals": empty_dihedrals,
            },
            "HeavyAtom": {
                "indices": ha_indices,
                "cg_species": np.ones(n_mols, dtype=np.int32),
                "cg_bonds": empty_bonds,
                "cg_angles": empty_angles,
                "cg_dihedrals": empty_dihedrals,
            },
        }

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return empty topology arrays (no bonds for single-bead water)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]


# ---------------------------------------------------------------------------
# CATH protein-domain mapping
# ---------------------------------------------------------------------------

import json as _json
import os as _os

_DEFAULT_RESIDUE_MAPS = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "data", "residue_maps.json"
)
_DEFAULT_DOMAIN_INDEX = _os.path.join(
    _os.path.dirname(__file__), "..", "..", "data", "domain_residue_index.json"
)

_AN_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}
_SYM_TO_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06}


class CATH_Map:
    """Residue-aware CG mapping for a single ACE/NME-capped CATH domain.

    Reads atom→residue assignment from ``domain_residue_index.json`` and
    per-residue CG maps from ``residue_maps.json``.

    Args:
        domain_name:        CATH domain identifier, e.g. ``"1neqA00"``.
        cg_strategy:        Strategy key in ``residue_maps.json``
                            (e.g. ``"CA"``, ``"coreBetaMap2"``).
        residue_maps_path:  Optional override for ``residue_maps.json`` path.
        domain_index_path:  Optional override for ``domain_residue_index.json`` path.
    """

    def __init__(
        self,
        domain_name: str,
        cg_strategy: str = "CA",
        mapping_type: str = "slice",
        residue_maps_path: str | None = None,
        domain_index_path: str | None = None,
        residue_names=None,
        residue_ids=None,
        n_atoms: int | None = None,
    ):
        self.domain_name = domain_name
        self.cg_strategy = cg_strategy
        if mapping_type not in ("slice", "com"):
            raise ValueError(
                f"Invalid mapping_type '{mapping_type}'. Choose one of ['slice', 'com']."
            )
        self.mapping_type = mapping_type

        residue_maps_path = residue_maps_path or _DEFAULT_RESIDUE_MAPS

        with open(residue_maps_path) as f:
            residue_maps = _json.load(f)

        if residue_names is None or residue_ids is None:
            domain_index_path = domain_index_path or _DEFAULT_DOMAIN_INDEX
            with open(domain_index_path) as f:
                domain_index = _json.load(f)

            if domain_name not in domain_index["domains"]:
                avail = list(domain_index["domains"].keys())
                raise ValueError(
                    f"Domain '{domain_name}' not found in domain_residue_index.json. "
                    f"Available (first 5): {avail[:5]}"
                )

            domain_info = domain_index["domains"][domain_name]
            residue_names = domain_info["residue_names"]
            residue_ids = domain_info["residue_ids"]
            self.n_atoms = domain_info["n_atoms"]
        else:
            residue_names = list(residue_names)
            residue_ids = list(residue_ids)
            self.n_atoms = n_atoms if n_atoms is not None else len(residue_ids)

        first_res = residue_names[0] if residue_names else ""
        last_res = residue_names[-1] if residue_names else ""
        if first_res != "ACE" or last_res != "NME":
            raise ValueError(
                f"Domain '{domain_name}' is not ACE/NME-capped "
                f"(first='{first_res}', last='{last_res}')"
            )

        # Group atoms by (residue_id, residue_name)
        from collections import OrderedDict

        residues: "OrderedDict[tuple, list[int]]" = OrderedDict()
        for atom_idx, (res_id, res_name) in enumerate(zip(residue_ids, residue_names)):
            residues.setdefault((res_id, res_name), []).append(atom_idx)

        global_indices: list[int] = [-1] * self.n_atoms
        global_cg_species: list[int] = []
        at_masses: list[float] = []
        explicit_bonds: list[list[int]] = []
        explicit_angles: list[list[int]] = []
        explicit_dihedrals: list[list[int]] = []
        cg_offset = 0

        for (res_id, res_name), atom_idxs in residues.items():
            n_res_atoms = len(atom_idxs)
            if res_name not in residue_maps:
                print(
                    f"  [WARN] residue '{res_name}' not in residue_maps.json - skipped"
                )
                at_masses.extend([12.011] * n_res_atoms)
                continue

            res_data = residue_maps[res_name]
            cg_maps = res_data.get("cg_maps", {})
            if cg_strategy not in cg_maps:
                raise ValueError(
                    f"Strategy '{cg_strategy}' not found for residue '{res_name}'. "
                    f"Available: {list(cg_maps.keys())}"
                )

            local_indices = cg_maps[cg_strategy]["indices"]
            local_species = cg_maps[cg_strategy]["cg_species"]

            if len(local_indices) != n_res_atoms:
                raise ValueError(
                    f"Residue '{res_name}' (id={res_id}): expected {n_res_atoms} "
                    f"atoms, residue_maps has {len(local_indices)}"
                )

            for local_pos, ai in enumerate(atom_idxs):
                local_cg = local_indices[local_pos]
                if local_cg >= 0:
                    global_indices[ai] = cg_offset + int(local_cg)

            global_cg_species.extend(int(s) for s in local_species)

            if self.mapping_type == "com":
                b, a, d = _build_explicit_residue_topology(
                    residue_name=res_name,
                    strategy=cg_strategy,
                    cg_map=cg_maps[cg_strategy],
                    cg_offset=cg_offset,
                    n_local_cg=len(local_species),
                )
                explicit_bonds.extend(b)
                explicit_angles.extend(a)
                explicit_dihedrals.extend(d)

            cg_offset += len(local_species)

            # Masses
            masses = res_data.get("masses")
            if masses and len(masses) == n_res_atoms:
                at_masses.extend(masses)
            else:
                syms = res_data.get("symbols", [])
                for s in syms[:n_res_atoms]:
                    at_masses.append(_SYM_TO_MASS.get(str(s), 12.011))

        self.at_masses = at_masses
        self._indices = global_indices
        self._cg_species = np.array(global_cg_species, dtype=np.int32)
        self._n_cg = len(global_cg_species)

        # Atomistic bonds
        at_bonds: list[tuple[int, int]] = []
        res_items = list(residues.items())
        for (_, res_name), atom_idxs in res_items:
            if res_name not in residue_maps:
                continue
            for li, lj in residue_maps[res_name].get("at_bonds", []):
                at_bonds.append((atom_idxs[li], atom_idxs[lj]))

        for k in range(len(res_items) - 1):
            (_, rname_k), atoms_k = res_items[k]
            (_, rname_k1), atoms_k1 = res_items[k + 1]
            if rname_k not in residue_maps or rname_k1 not in residue_maps:
                continue
            c_idx = residue_maps[rname_k].get("backbone_C_idx")
            n_idx = residue_maps[rname_k1].get("backbone_N_idx")
            if c_idx is not None and n_idx is not None:
                at_bonds.append((atoms_k[c_idx], atoms_k1[n_idx]))

        self._at_bonds = at_bonds

        # Pre-compute CG topology
        if self.mapping_type == "slice":
            self._cg_bonds, self._cg_angles, self._cg_dihedrals = (
                _slice_cg_topology_from_atomistic(
                    at_bonds=self._at_bonds,
                    mapping_indices=self._indices,
                )
            )
        else:
            self._cg_bonds = _arr_topology(explicit_bonds, 2)
            self._cg_angles = _arr_topology(explicit_angles, 3)
            self._cg_dihedrals = _arr_topology(explicit_dihedrals, 4)

    def get_available_maps(self) -> list[str]:
        return [self.cg_strategy]

    def get_map(self, name: str | None = None) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        indices = self._indices
        cg_species = self._cg_species
        n_cg = self._n_cg

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(
            row_sums, 1.0, atol=1e-5
        ), f"Weights don't sum to 1 (max err {float(jnp.max(jnp.abs(row_sums - 1))):.2e})"
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str | None = None) -> tuple:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        return self._cg_bonds, self._cg_angles, self._cg_dihedrals


# ---------------------------------------------------------------------------
# 3BPA mapping
# ---------------------------------------------------------------------------


class ThreeBPA_Map:
    """CG mapping for 3-bromopropionic acid (3BPA).

    Available maps:
        - ``"LVC=0.6"``: LVC-based mapping
    """

    def __init__(self, at_bonds: list[tuple[int, int]] | np.ndarray | None = None):
        # Placeholder: LVC=0.6
        self.base_species = [
            "C",
            "C",
            "C",
            "H",
            "C",
            "O",
            "N",
            "N",
            "H",
            "H",
            "C",
            "H",
            "H",
            "C",
            "C",
            "H",
            "H",
            "C",
            "C",
            "C",
            "H",
            "C",
            "H",
            "C",
            "H",
            "H",
            "H",
        ]
        self.at_masses = [mass_map[s] for s in self.base_species]

        lvc_indices = [
            0,  # 1BPA               1LIG     C1
            1,  # 1BPA               1LIG     C2
            0,  # 1BPA               1LIG     C3
            -1,  # 1BPA              1LIG     H1
            1,  # 1BPA               1LIG     C4
            4,  # 1BPA       O       1LIG     O5
            3,  # 1BPA       N       1LIG     N1
            2,  # 1BPA               1LIG     N2
            -1,  # 1BPA              1LIG     H2
            -1,  # 1BPA              1LIG     H3
            2,  # 1BPA               1LIG     C5
            -1,  # 1BPA              1LIG     H4
            -1,  # 1BPA              1LIG     H5
            5,  # 1BPA               1LIG     C6
            6,  # 1BPA               1LIG     C7
            -1,  # 1BPA              1LIG     H6
            -1,  # 1BPA              1LIG     H7
            7,  # 1BPA               1LIG     C8
            6,  # 1BPA               1LIG     C9
            7,  # 1BPA               1LIG    C10
            -1,  # 1BPA              1LIG    H10
            8,  # 1BPA               1LIG    C11
            -1,  # 1BPA              1LIG    H11
            8,  # 1BPA               1LIG    C12
            -1,  # 1BPA              1LIG    H12
            -1,  # 1BPA              1LIG    H13
            -1,  # 1BPA              1LIG    H14
        ]

        try:
            bonds, angles, dihedrals = (
                _derive_cg_topology_from_atomistic_graph(at_bonds, lvc_indices)
            )
        except Exception as e:
            print(f"Error deriving CG topology for 3BPA: {e}")
            bonds = np.zeros((2, 0), dtype=np.int32)
            angles = np.zeros((3, 0), dtype=np.int32)
            dihedrals = np.zeros((4, 0), dtype=np.int32)

        self._maps = {
            "LVC=0.6": {
                "indices": list(lvc_indices),
                "cg_species": [1, 1, 2, 3, 4, 5, 1, 1, 1],
                "cg_bonds": bonds,
                "cg_angles": angles,
                "cg_dihedrals": dihedrals,
            },
        }

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]


# ---------------------------------------------------------------------------
# Azobenzene mapping
# ---------------------------------------------------------------------------


class Azobenzene_Map:
    """ """

    def __init__(self, at_bonds: list[tuple[int, int]] | np.ndarray | None = None):
        self.base_species = [
            "N",
            "N",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "C",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
            "H",
        ]
        self.at_masses = [mass_map[s] for s in self.base_species]

        lvc_indices = [
            0,  # 1LIG     N1
            1,  # 1LIG     N2
            2,  # 1LIG     C1
            2,  # 1LIG     C2
            3,  # 1LIG     C3
            3,  # 1LIG     C4
            4,  # 1LIG     C5
            4,  # 1LIG     C6
            5,  # 1LIG     C7
            5,  # 1LIG     C8
            6,  # 1LIG     C9
            6,  # 1LIG    C10
            7,  # 1LIG    C11
            7,  # 1LIG    C12
            -1,  # 1LIG     H1
            -1,  # 1LIG     H2
            -1,  # 1LIG     H3
            -1,  # 1LIG     H4
            -1,  # 1LIG     H5
            -1,  # 1LIG     H6
            -1,  # 1LIG     H7
            -1,  # 1LIG     H8
            -1,  # 1LIG     H9
            -1,  # 1LIG    H10
        ]

        try:
            bonds, angles, dihedrals = (
                _derive_cg_topology_from_atomistic_graph(at_bonds, lvc_indices)
            )
        except Exception as e:
            print(f"Error deriving CG topology for Azobenzene: {e}")
            bonds = np.zeros((2, 0), dtype=np.int32)
            angles = np.zeros((3, 0), dtype=np.int32)
            dihedrals = np.zeros((4, 0), dtype=np.int32)

        self._maps = {
            "LVC=0.45": {
                "indices": list(lvc_indices),
                "cg_species": [1, 1, 2, 2, 2, 2, 2, 2],  # 1 = N, 2 = C-C
                "cg_bonds": bonds,
                "cg_angles": angles,
                "cg_dihedrals": dihedrals,
            },
        }

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. Choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        valid_mask = indices_arr >= 0
        clipped = jnp.where(valid_mask, indices_arr, 0)
        cg_masses = jax.ops.segment_sum(
            jnp.where(valid_mask, at_masses_arr, 0.0), clipped, n_cg
        )
        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]
