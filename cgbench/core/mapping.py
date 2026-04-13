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
    return os.path.join(os.path.dirname(__file__), "..", "..", "data", "residue_maps.json")


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
        map_arr:      (n_atoms,) int32 – CG site index per atom; -1 = excluded.
        at_masses_arr:(n_atoms,) float32 – atomic masses.
        cg_masses:    (n_cg,)   float32 – total mass per CG site.

    Returns:
        weights: (n_cg, n_atoms) float32 – row-normalised (rows sum to 1).
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
# CG bond-type computation from the atomistic graph
# ---------------------------------------------------------------------------

def compute_cg_bond_types(
    at_bonds: list[tuple[int, int]],
    mapping_indices: list[int],
    max_sep: int = 4,
) -> dict[str, list[list[int]]]:
    """Compute CG-site bond types from the atomistic bond graph.

    Uses a two-step approach:
    1. Build a CG adjacency graph: two CG sites are adjacent if any atom in
       one can reach any atom in the other through only unmapped intermediates.
    2. BFS on the CG graph to get CG-level distances, then bucket:
       - bond_0: CG distance 1 (directly bonded, 1-2)
       - bond_1: CG distance 2 (1-3 in CG graph)
       - bond_2: CG distance 3 (1-4 in CG graph)
       - bond_3: CG distance 4 (1-5 in CG graph)

    Args:
        at_bonds:        List of 0-indexed (i, j) atomistic bond pairs.
        mapping_indices: Per-atom CG-site assignment (-1 = excluded).
        max_sep:         Maximum CG-graph path length to consider (default 4).

    Returns:
        Dict with keys bond_0 … bond_3, each a sorted list of [cg_i, cg_j] pairs.
    """
    adj: dict[int, set[int]] = defaultdict(set)
    for i, j in at_bonds:
        adj[i].add(j)
        adj[j].add(i)

    mapped_atoms = [i for i, cg in enumerate(mapping_indices) if cg >= 0]

    # Step 1: build CG adjacency (traverse through unmapped intermediates).
    cg_adj: dict[int, set[int]] = defaultdict(set)
    for start in mapped_atoms:
        cg_start = mapping_indices[start]
        visited: set[int] = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb in visited:
                    continue
                visited.add(nb)
                nb_cg = mapping_indices[nb]
                if nb_cg >= 0:
                    if nb_cg != cg_start:
                        cg_adj[cg_start].add(nb_cg)
                else:
                    queue.append(nb)

    # Step 2: BFS on CG graph for CG-level distances.
    cg_sites = sorted(set(mapping_indices[i] for i in mapped_atoms))
    cg_pair_min: dict[tuple[int, int], int] = {}

    for start_cg in cg_sites:
        dist_cg: dict[int, int] = {start_cg: 0}
        queue_cg: deque[int] = deque([start_cg])
        while queue_cg:
            curr = queue_cg.popleft()
            if dist_cg[curr] >= max_sep:
                continue
            for nb_cg in cg_adj[curr]:
                if nb_cg not in dist_cg:
                    dist_cg[nb_cg] = dist_cg[curr] + 1
                    queue_cg.append(nb_cg)

        for other_cg, d in dist_cg.items():
            if d == 0:
                continue
            pair = (min(start_cg, other_cg), max(start_cg, other_cg))
            if pair not in cg_pair_min or cg_pair_min[pair] > d:
                cg_pair_min[pair] = d

    buckets: dict[int, list[list[int]]] = {0: [], 1: [], 2: [], 3: []}
    for (i, j), d in sorted(cg_pair_min.items()):
        if 1 <= d <= 4:
            buckets[d - 1].append([i, j])

    return {
        "bond_0": buckets[0],
        "bond_1": buckets[1],
        "bond_2": buckets[2],
        "bond_3": buckets[3],
    }


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
        np.array(angles, dtype=np.int32).T if angles else np.zeros((3, 0), dtype=np.int32)
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
        dihedrals.extend([[i + off, j + off, k + off, l + off] for i, j, k, l in single_dihedrals])

    def _arr(lst, ncols):
        return np.array(lst, dtype=np.int32).T if lst else np.zeros((ncols, 0), dtype=np.int32)

    return _arr(bonds, 2), _arr(angles, 3), _arr(dihedrals, 4)


# ---------------------------------------------------------------------------
# Hexane_Map
# ---------------------------------------------------------------------------

class Hexane_Map:
    """CG mapping for liquid hexane (multi-molecule system).

    Each hexane molecule (20 atoms: CH3-CH2-CH2-CH2-CH2-CH3) can be mapped
    to 2–6 CG sites.  All CG topologies are linear chains.

    Available maps: six-site, four-site, three-site, two-site,
                    two-site-Map2, A3, A4.
    """

    _base_species = [
        "C", "H", "H", "H",   # CH3
        "C", "H", "H",         # CH2
        "C", "H", "H",         # CH2
        "C", "H", "H",         # CH2
        "C", "H", "H",         # CH2
        "C", "H", "H", "H",   # CH3
    ]

    # Atomistic bonds for a single hexane (20 atoms, 0-indexed)
    _at_bonds_single: list[tuple[int, int]] = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (4, 5), (4, 6), (4, 7),
        (7, 8), (7, 9), (7, 10),
        (10, 11), (10, 12), (10, 13),
        (13, 14), (13, 15), (13, 16),
        (16, 17), (16, 18), (16, 19),
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
            [-1, -1, -1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, -1, -1, -1, -1],
            [1, 2, 2, 1],
        ),
        "three-site": (
            3,
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
            [1, 2, 1],
        ),
        "two-site": (
            2,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1],
        ),
        "two-site-Map2": (
            2,
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2],
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

        self._maps: dict[str, dict] = {}
        for name, (n_cg, single_idx, cg_sp) in self._map_specs.items():
            indices = self._tile_indices(single_idx, n_cg)
            cg_species = np.array(cg_sp * nmol, dtype=np.int32)

            # Linear CG chain topology, tiled across molecules
            s_bonds = [[i, i + 1] for i in range(n_cg - 1)]
            s_angles = [[i, i + 1, i + 2] for i in range(n_cg - 2)]
            s_dihedrals = [[i, i + 1, i + 2, i + 3] for i in range(n_cg - 3)]
            cg_bonds, cg_angles, cg_dihedrals = _tile_topology(
                s_bonds, s_angles, s_dihedrals, n_cg, nmol
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

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(self, name: str) -> tuple:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'. Choose one of {self.get_available_maps()}")
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

    def get_bond_types(self, name: str) -> dict[str, list[list[int]]]:
        """Derive CG bond types from the atomistic graph (backward compat)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        indices = self._maps[name]["indices"]
        tiled_bonds: list[tuple[int, int]] = []
        mol_size = 20
        for mol in range(self.n_replicas):
            off = mol * mol_size
            for a, b in self._at_bonds_single:
                tiled_bonds.append((a + off, b + off))
        return compute_cg_bond_types(tiled_bonds, indices)


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
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),
    ]

    # Triangle topology for one molecule
    _single_bonds = [[0, 1], [1, 2], [0, 2]]
    _single_angles = [[2, 0, 1], [0, 1, 2], [0, 2, 1]]
    _single_dihedrals: list[list[int]] = []  # no 4-site dihedrals in a triangle

    def __init__(self, nmol: int = 128):
        self.n_replicas = nmol
        self.at_masses = [mass_map[s] for s in self._base_species] * nmol

        single_indices = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]
        indices = self._tile_indices(single_indices, block_size=3)
        cg_species = np.array([1, 1, 1] * nmol, dtype=np.int32)
        cg_bonds, cg_angles, cg_dihedrals = _tile_topology(
            self._single_bonds, self._single_angles, self._single_dihedrals, 3, nmol
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
            raise ValueError(f"Invalid map '{name}'. Choose one of {self.get_available_maps()}")
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

    def get_bond_types(self, name: str = "three-site-adjacent") -> dict:
        """Derive CG bond types from the atomistic graph (backward compat)."""
        name = self._normalize_map_name(name)
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        indices = self._maps[name]["indices"]
        tiled_bonds: list[tuple[int, int]] = []
        mol_size = 12
        for mol in range(self.n_replicas):
            off = mol * mol_size
            for a, b in self._at_bonds_single:
                tiled_bonds.append((a + off, b + off))
        return compute_cg_bond_types(tiled_bonds, indices)


# ---------------------------------------------------------------------------
# CappedPeptideMap
# ---------------------------------------------------------------------------

class CappedPeptideMap:
    """CG mapping for ACE/NME-capped peptides using residue_maps.json.

    Automatically builds all CG maps that are available for every residue in
    the given sequence (intersection of per-residue available strategies).

    The atomistic topology (masses, bonds) is loaded from residue_maps.json;
    the CG topology (bonds, angles, dihedrals) is derived via
    :func:`compute_cg_bond_types` and :func:`_derive_cg_topology`.

    Args:
        residue_sequence: Ordered list of residue names,
            e.g. ``["ACE", "ALA", "NME"]``.
    """

    def __init__(self, residue_sequence: list[str]):
        residue_maps = _load_residue_maps_json()

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
                at_bonds.append((atom_offsets[ri] + c_idx, atom_offsets[ri + 1] + n_idx))
        self._at_bonds = at_bonds

        # Available strategies = intersection across all residues
        all_strategies = set.intersection(
            *[set(residue_maps[res]["cg_maps"].keys()) for res in residue_sequence]
        )

        self._maps: dict[str, dict] = {}
        for strategy in sorted(all_strategies):
            indices: list[int] = []
            cg_species: list[int] = []
            cg_offset = 0

            for ri, res in enumerate(residue_sequence):
                rd = residue_maps[res]
                cg_map = rd["cg_maps"][strategy]
                local_indices = cg_map["indices"]
                local_species = cg_map["cg_species"]
                for local_idx in local_indices:
                    indices.append(-1 if local_idx < 0 else int(local_idx) + cg_offset)
                cg_species.extend(int(s) for s in local_species)
                cg_offset += len(local_species)

            # CG topology from atomistic bond graph
            bond_types = compute_cg_bond_types(at_bonds, indices)
            cg_bonds, cg_angles, cg_dihedrals = _derive_cg_topology(bond_types["bond_0"])

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
        assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
            f"Weights don't sum to 1 (max err {float(jnp.max(jnp.abs(row_sums - 1))):.2e})"
        )
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        d = self._maps[name]
        return d["cg_bonds"], d["cg_angles"], d["cg_dihedrals"]

    def get_bond_types(self, name: str) -> dict[str, list[list[int]]]:
        """Return CG bond types dict (backward compat)."""
        if name not in self._maps:
            raise ValueError(f"Invalid map '{name}'.")
        return compute_cg_bond_types(self._at_bonds, self._maps[name]["indices"])


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
            ua_indices.extend([m, m, m])       # O, H1, H2 → bead m
            ha_indices.extend([m, -1, -1])     # O → bead m; H1, H2 excluded

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
            raise ValueError(f"Invalid map '{name}'. Choose one of {self.get_available_maps()}")
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
        residue_maps_path: str | None = None,
        domain_index_path: str | None = None,
    ):
        self.domain_name = domain_name
        self.cg_strategy = cg_strategy

        residue_maps_path = residue_maps_path or _DEFAULT_RESIDUE_MAPS
        domain_index_path = domain_index_path or _DEFAULT_DOMAIN_INDEX

        with open(residue_maps_path) as f:
            residue_maps = _json.load(f)
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
        cg_offset = 0

        for (res_id, res_name), atom_idxs in residues.items():
            n_res_atoms = len(atom_idxs)
            if res_name not in residue_maps:
                print(f"  [WARN] residue '{res_name}' not in residue_maps.json – skipped")
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
        bond_types = compute_cg_bond_types(self._at_bonds, self._indices)
        self._cg_bonds, self._cg_angles, self._cg_dihedrals = _derive_cg_topology(
            bond_types["bond_0"]
        )

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
        assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
            f"Weights don't sum to 1 (max err {float(jnp.max(jnp.abs(row_sums - 1))):.2e})"
        )
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str | None = None) -> tuple:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        return self._cg_bonds, self._cg_angles, self._cg_dihedrals

    def get_bond_types(self, name: str | None = None) -> dict:
        """Return CG bond types dict (backward compat)."""
        return compute_cg_bond_types(self._at_bonds, self._indices, max_sep=4)


# ---------------------------------------------------------------------------
# UncappedProteinMap  (for STATIC_FRAME datasets)
# ---------------------------------------------------------------------------

class UncappedProteinMap:
    """CG mapping for proteins loaded via MDAnalysis (e.g. STATIC_FRAME datasets).

    Unlike :class:`CATH_Map`, this class:
    - Accepts per-atom residue metadata directly (from MDAnalysis).
    - Does not require ACE/NME caps.
    - Gracefully skips residues absent from ``residue_maps.json`` or whose
      atom count mismatches the reference (warning is printed).

    Args:
        residue_ids:    (n_atoms,) per-atom residue IDs (1-based int array).
        residue_names:  (n_atoms,) per-atom residue name strings.
        atom_names:     (n_atoms,) per-atom GRO/PDB atom name strings.
        cg_strategy:    Strategy key in residue_maps.json (default ``"coreBetaMap2"``).
        residue_maps_path: Optional override path.
    """

    def __init__(
        self,
        residue_ids: np.ndarray,
        residue_names: np.ndarray,
        atom_names: np.ndarray,
        cg_strategy: str = "coreBetaMap2",
        residue_maps_path: str | None = None,
    ):
        self.cg_strategy = cg_strategy
        n_atoms = len(residue_ids)

        rmap_path = residue_maps_path or _DEFAULT_RESIDUE_MAPS
        with open(rmap_path) as f:
            residue_maps = _json.load(f)

        # Group atoms by (residue_id, residue_name), preserving insertion order
        from collections import OrderedDict
        residues: "OrderedDict[tuple, list[int]]" = OrderedDict()
        for ai, (res_id, res_name) in enumerate(zip(residue_ids, residue_names)):
            key = (int(res_id), str(res_name))
            residues.setdefault(key, []).append(ai)

        global_indices: list[int] = [-1] * n_atoms
        global_cg_species: list[int] = []
        at_masses: list[float] = []
        cg_offset = 0

        for (res_id, res_name), atom_idxs in residues.items():
            n_res = len(atom_idxs)
            if res_name not in residue_maps:
                print(f"  [WARN] residue '{res_name}' not in residue_maps.json – skipped")
                at_masses.extend([12.011] * n_res)
                continue

            res_data = residue_maps[res_name]
            cg_maps = res_data.get("cg_maps", {})
            if cg_strategy not in cg_maps:
                print(
                    f"  [WARN] strategy '{cg_strategy}' missing for '{res_name}' – skipped"
                )
                masses = res_data.get("masses", [12.011] * n_res)
                at_masses.extend(masses[:n_res])
                continue

            local_indices = cg_maps[cg_strategy]["indices"]
            local_species = cg_maps[cg_strategy]["cg_species"]

            if len(local_indices) != n_res:
                print(
                    f"  [WARN] residue '{res_name}' (id={res_id}): atom count mismatch "
                    f"({n_res} in structure, {len(local_indices)} in residue_maps) – skipped"
                )
                masses = res_data.get("masses", [12.011] * n_res)
                at_masses.extend(masses[:n_res])
                continue

            for local_pos, ai in enumerate(atom_idxs):
                local_cg = local_indices[local_pos]
                if local_cg >= 0:
                    global_indices[ai] = cg_offset + int(local_cg)

            global_cg_species.extend(int(s) for s in local_species)
            cg_offset += len(local_species)

            masses = res_data.get("masses")
            if masses and len(masses) == n_res:
                at_masses.extend(masses)
            else:
                syms = res_data.get("symbols", [])
                at_masses.extend(_SYM_TO_MASS.get(str(s), 12.011) for s in syms[:n_res])

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
                if li < len(atom_idxs) and lj < len(atom_idxs):
                    at_bonds.append((atom_idxs[li], atom_idxs[lj]))

        for k in range(len(res_items) - 1):
            (_, rname_k), atoms_k = res_items[k]
            (_, rname_k1), atoms_k1 = res_items[k + 1]
            if rname_k not in residue_maps or rname_k1 not in residue_maps:
                continue
            c_idx = residue_maps[rname_k].get("backbone_C_idx")
            n_idx = residue_maps[rname_k1].get("backbone_N_idx")
            if c_idx is not None and n_idx is not None:
                if c_idx < len(atoms_k) and n_idx < len(atoms_k1):
                    at_bonds.append((atoms_k[c_idx], atoms_k1[n_idx]))

        self._at_bonds = at_bonds

        # Pre-compute CG topology
        bond_types = compute_cg_bond_types(self._at_bonds, self._indices)
        self._cg_bonds, self._cg_angles, self._cg_dihedrals = _derive_cg_topology(
            bond_types["bond_0"]
        )

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
        return indices, cg_species, cg_masses, weights

    def get_cg_topology(self, name: str | None = None) -> tuple:
        """Return (cg_bond_index, cg_angle_index, cg_dihedral_index)."""
        return self._cg_bonds, self._cg_angles, self._cg_dihedrals

    def get_bond_types(self, name: str | None = None) -> dict:
        """Return CG bond types dict (backward compat)."""
        return compute_cg_bond_types(self._at_bonds, self._indices, max_sep=4)


# Alias for backward compatibility with _ProSolUncappedProteinDataset
UncappedCATHLike_Map = UncappedProteinMap
