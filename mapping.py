from jax import numpy as jnp, lax
import jax
import functools
import numpy as np


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
        displacement_fn: Function to compute the displacement between two
            sets of coordinates. Necessary to handle boundary conditions.
        shift_fn: Ensures that the produced coordinates remain in the
            box.
        c_map: Matrix $c_{Ii}$ defining the linear mapping of positions.
        d_map: Matrix $d_{Ii}$ defining the linear mapping of forces in combination
            with $c_{Ii}$.
        force_dataset: Dataset of fine-scaled forces.

    Returns:
        Returns the coarse-grained positions and, if provided, coarse-grained
        forces.

    References:
        .. [Noid2008] W. G. Noid, Jhih-Wei Chu, Gary S. Ayton, Vinod Krishna,
           Sergei Izvekov, Gregory A. Voth, Avisek Das, Hans C. Andersen;
           *The multiscale coarse-graining method. I. A rigorous bridge between
           atomistic and coarse-grained models*. J. Chem. Phys. 28 June 2008;
           128 (24): 244114. https://doi-org.eaccess.tum.edu/10.1063/1.2938860


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

    debug = False
    if debug:
        first_frame = position_dataset[3]
        first_force = force_dataset[3]
        return _map_single((first_frame, first_force))

    return lax.map(_map_single, (position_dataset, force_dataset))


# atomic‐mass lookup by symbol
mass_map = {"H": 1.01, "C": 12.011, "N": 14.007, "O": 15.999}
# atomic number to symbol mapping
atomic_number_map = {1: "H", 6: "C", 7: "N", 8: "O"}
atomic_number_map_reverse = {"H": 1, "C": 6, "N": 7, "O": 8}


@jax.jit
def get_map_weights(
    map_arr: jnp.ndarray,
    at_masses_arr: jnp.ndarray,
    cg_masses: jnp.ndarray,
) -> jnp.ndarray:
    valid = map_arr >= 0
    clipped = jnp.where(valid, map_arr, 0)
    onehot = jax.nn.one_hot(clipped, cg_masses.shape[0], dtype=jnp.float32).T
    onehot *= valid[None, :]
    per_atom_w = at_masses_arr[None, :] / cg_masses[:, None]

    per_atom_w = jnp.where(
        cg_masses[:, None] > 0, at_masses_arr[None, :] / cg_masses[:, None], 0.0
    )

    weights = onehot * per_atom_w
    return weights


class Hexane_Map:
    """Class for hexane mapping with dynamic number of molecules."""

    # Base species for a single hexane
    _base_species = [
        "C",
        "H",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "H",
    ]

    def __init__(self, nmol=100):
        self.mass_map = {"H": 1.01, "C": 12.011, "N": 14.007, "O": 15.999}

        self.n_replicas = nmol
        # Build full atom species list
        self.map_species = self._base_species * self.n_replicas
        # Map to atomic masses
        self.at_masses = [self.mass_map[s] for s in self.map_species]
        # Prepare map definitions
        self._maps = {
            "six-site": {
                "indices": self._get_six_site_indices(),
                "cg_species": np.array([2, 1, 1, 1, 1, 2] * self.n_replicas),
            },
            "four-site": {
                "indices": self._get_four_site_indices(),
                "cg_species": np.array([1, 2, 2, 1] * self.n_replicas),
            },
            "three-site": {
                "indices": self._get_three_site_indices(),
                "cg_species": np.array([1, 2, 1] * self.n_replicas),
            },
            "two-site": {
                "indices": self._get_two_site_indices(),
                "cg_species": np.array([1] * 2 * self.n_replicas),
            },
            "two-site-Map2": {
                "indices": self._get_two_site_indices(),
                "cg_species": np.array([1, 2] * self.n_replicas),
            },
            "A3": {
                "indices": self._get_A3_site_indices(),
                "cg_species": np.array([1, 2] * self.n_replicas),
            },
            "A4": {
                "indices": self._get_A4_site_indices(),
                "cg_species": np.array([1, 2, 3] * self.n_replicas),
            },
        }

    def _get_six_site_indices(self) -> list[int]:
        single = [
            0,
            -1,
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
            4,
            -1,
            -1,
            5,
            -1,
            -1,
            -1,
        ]
        return self._tile_indices(single, block_size=6)

    def _get_four_site_indices(self) -> list[int]:
        single = [
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
        ]
        return self._tile_indices(single, block_size=4)

    def _get_three_site_indices(self) -> list[int]:
        single = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        return self._tile_indices(single, block_size=3)

    def _get_two_site_indices(self) -> list[int]:
        single = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        return self._tile_indices(single, block_size=2)

    def _get_A3_site_indices(self) -> list[int]:
        single = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        return self._tile_indices(single, block_size=2)

    def _get_A4_site_indices(self) -> list[int]:
        single = [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
        ]
        return self._tile_indices(single, block_size=3)

    def _tile_indices(self, single: list[int], block_size: int) -> list[int]:
        """
        Tile a single-map pattern across all replicas.
        """
        result = []
        for block in range(self.n_replicas):
            offset = block * block_size
            for v in single:
                result.append(v if v < 0 else v + offset)
        return result

    def get_available_maps(self) -> list[str]:
        return list(self._maps.keys())

    def get_map(
        self, name: str
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights)."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )
        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(indices_arr, at_masses_arr, cg_masses)

        # Ensure weights sum to 1 per cg-site
        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights


class Ala2_Map:
    """Class for Ala2 mapping. ASSUMES PEPSOL ATOM ORDERING."""

    # Base species list
    map_species: list[str] = [
        "C",
        "H",
        "H",
        "H",  # ACE CH3
        "C",  # ACE C
        "O",  # ACE O
        "N",  # ALA N
        "H",  # ALA H
        "C",  # ALA CA
        "H",  # ALA CA-H
        "C",
        "H",
        "H",
        "H",  # ALA CB
        "C",  # ALA C
        "O",  # ALA O
        "N",
        "H",  # NME NH
        "C",
        "H",
        "H",
        "H",  # NME CH3
    ]

    # Per‐map definitions
    _maps: dict[str, dict] = {
        "hmerged": {
            "indices": [
                0,
                0,
                0,
                0,  # ACE CH3
                1,
                2,  # CO
                3,
                3,  # N,H
                4,
                4,  # CA,HA
                5,
                5,
                5,
                5,  # CB,HB1,HB2,HB3
                6,
                7,  # C,O
                8,
                8,  # NME NH
                9,
                9,
                9,
                9,  # NME CH3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 1, 2, 3, 4, 1]
            ),  # 1=CH3, 2=C, 3=O, 4=NH, 5=CH
        },
        "core": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                -1,
                -1,
                -1,
                -1,  # CB
                3,
                -1,  # C
                4,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 2, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreSingle": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                -1,
                -1,
                -1,
                -1,  # CB
                3,
                -1,  # C
                4,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 1, 1, 1, 1]),  # 1=C, 2=N
        },
        "coreMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                -1,
                -1,
                -1,
                -1,  # CB
                3,
                -1,  # C
                4,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 2, 3, 4, 5]),  # 𝐶𝑂𝐶𝐻3, 𝑁𝐻, 𝐶𝐻𝐶𝐻3, 𝐶𝑂, 𝑁𝐻𝐶𝐻3
        },
        "heavyOnly": {  #
            "indices": [
                0,
                -1,
                -1,
                -1,  # ACE CH3
                1,
                2,  # CO
                3,
                -1,  # N,H
                4,
                -1,  # CA,HA
                5,
                -1,
                -1,
                -1,  # CB,HB1,HB2,HB3
                6,
                7,  # C,O
                8,
                -1,  # NME NH
                9,
                -1,
                -1,
                -1,  # NME CH3
            ],
            "cg_species": np.array([1, 1, 2, 3, 1, 1, 1, 2, 3, 1]),  # 1=C, 2=O, 3=N
        },
        "heavyOnlyMap2": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # ACE CH3
                1,
                2,  # CO
                3,
                -1,  # N,H
                4,
                -1,  # CA,HA
                5,
                -1,
                -1,
                -1,  # CB,HB1,HB2,HB3
                6,
                7,  # C,O
                8,
                -1,  # NME NH
                9,
                -1,
                -1,
                -1,  # NME CH3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 1, 2, 3, 4, 1]
            ),  # 1=CH3, 2=C, 3=O, 4=NH, 5=CH
        },
        "coreBeta": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0, # C_ACE
                -1,  
                1, # N_ALA
                -1,  
                2, # CA
                -1,  
                3, # CB
                -1,
                -1,
                -1,  
                4,# C_ALA
                -1,  
                5, # N_NME
                -1,  
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 2, 1, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreBetaMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                3,
                -1,
                -1,
                -1,  # CB
                4,
                -1,  # C
                5,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 2, 3, 4, 5, 6]),
        },
        "coreBetaMap3": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                3,
                -1,
                -1,
                -1,  # CB
                4,
                -1,  # C
                5,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array([1, 2, 3, 3, 2, 1]),
        },
        "coreBetaMap4": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                3,
                -1,
                -1,
                -1,  # CB
                4,
                -1,  # C
                5,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array(
                [1, 2, 1, 1, 1, 3]
            ),  # 1=COCH3, 2=NH, 3=CA, 4=CO, 5=CB, 6=NHCH3
        },
        "coreBetaMap5": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                3,
                -1,
                -1,
                -1,  # CB
                4,
                -1,  # C
                5,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array(
                [2, 1, 1, 1, 1, 2]
            ),  # 1=COCH3, 2=NH, 3=CA, 4=CO, 5=CB, 6=NHCH3
        },
        "coreBetaSingle": {
            "indices": [
                -1,
                -1,
                -1,
                -1,
                0,
                -1,  # C
                1,
                -1,  # N
                2,
                -1,  # Ca
                3,
                -1,
                -1,
                -1,  # CB
                4,
                -1,  # C
                5,
                -1,  # N
                -1,
                -1,
                -1,
                -1,
            ],
            "cg_species": np.array(
                [1, 1, 1, 1, 1, 1]
            ),  # 1=COCH3, 2=NH, 3=CA, 4=CO, 5=CB, 6=NHCH3
        },
    }

    at_masses = [mass_map[s] for s in map_species]

    def __init__(self):
        self.n_atoms = len(self.map_species)

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(
        self, name: str = "hmerged"
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights) for 'hmerged' or 'core'."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )

        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)  # shape (n_atoms,)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)  # shape (n_atoms,)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(
            indices_arr,  # map
            at_masses_arr,  # per-atom masses
            cg_masses,  # computed via bincount or segment_sum
        )

        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights


class Ala15_Map:
    """Class for Ala15 mapping. ASSUMES PEPSOL ATOM ORDERING."""

    map_species: list[str] = [
        "H",  # 1ACE HH31   - Hydrogen (methyl hydrogen 1)
        "C",  # 1ACE CH3    - Carbon (methyl group in acetyl cap)
        "H",  # 1ACE HH32   - Hydrogen (methyl hydrogen 2)
        "H",  # 1ACE HH33   - Hydrogen (methyl hydrogen 3)
        "C",  # 1ACE C      - Carbon (carbonyl carbon in acetyl)
        "O",  # 1ACE O      - Oxygen (carbonyl oxygen in acetyl)
        "N",  # 2ALA N      - Nitrogen (amino nitrogen in alanine 2)
        "H",  # 2ALA H      - Hydrogen (amino hydrogen)
        "C",  # 2ALA CA     - Carbon (alpha carbon)
        "H",  # 2ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 2ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 2ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 2ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 2ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 2ALA C      - Carbon (carbonyl carbon)
        "O",  # 2ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 3ALA N      - Nitrogen (amino nitrogen in alanine 3)
        "H",  # 3ALA H      - Hydrogen (amino hydrogen)
        "C",  # 3ALA CA     - Carbon (alpha carbon)
        "H",  # 3ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 3ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 3ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 3ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 3ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 3ALA C      - Carbon (carbonyl carbon)
        "O",  # 3ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 4ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 4ALA H      - Hydrogen (amino hydrogen)
        "C",  # 4ALA CA     - Carbon (alpha carbon)
        "H",  # 4ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 4ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 4ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 4ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 4ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 4ALA C      - Carbon (carbonyl carbon)
        "O",  # 4ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 5ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 5ALA H      - Hydrogen (amino hydrogen)
        "C",  # 5ALA CA     - Carbon (alpha carbon)
        "H",  # 5ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 5ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 5ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 5ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 5ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 5ALA C      - Carbon (carbonyl carbon)
        "O",  # 5ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 6ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 6ALA H      - Hydrogen (amino hydrogen)
        "C",  # 6ALA CA     - Carbon (alpha carbon)
        "H",  # 6ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 6ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 6ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 6ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 6ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 6ALA C      - Carbon (carbonyl carbon)
        "O",  # 6ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 7ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 7ALA H      - Hydrogen (amino hydrogen)
        "C",  # 7ALA CA     - Carbon (alpha carbon)
        "H",  # 7ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 7ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 7ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 7ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 7ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 7ALA C      - Carbon (carbonyl carbon)
        "O",  # 7ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 8ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 8ALA H      - Hydrogen (amino hydrogen)
        "C",  # 8ALA CA     - Carbon (alpha carbon)
        "H",  # 8ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 8ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 8ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 8ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 8ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 8ALA C      - Carbon (carbonyl carbon)
        "O",  # 8ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 9ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 9ALA H      - Hydrogen (amino hydrogen)
        "C",  # 9ALA CA     - Carbon (alpha carbon)
        "H",  # 9ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 9ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 9ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 9ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 9ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 9ALA C      - Carbon (carbonyl carbon)
        "O",  # 9ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 10ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 10ALA H      - Hydrogen (amino hydrogen)
        "C",  # 10ALA CA     - Carbon (alpha carbon)
        "H",  # 10ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 10ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 10ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 10ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 10ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 10ALA C      - Carbon (carbonyl carbon)
        "O",  # 10ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 11ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 11ALA H      - Hydrogen (amino hydrogen)
        "C",  # 11ALA CA     - Carbon (alpha carbon)
        "H",  # 11ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 11ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 11ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 11ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 11ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 11ALA C      - Carbon (carbonyl carbon)
        "O",  # 11ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 12ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 12ALA H      - Hydrogen (amino hydrogen)
        "C",  # 12ALA CA     - Carbon (alpha carbon)
        "H",  # 12ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 12ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 12ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 12ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 12ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 12ALA C      - Carbon (carbonyl carbon)
        "O",  # 12ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 13ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 13ALA H      - Hydrogen (amino hydrogen)
        "C",  # 13ALA CA     - Carbon (alpha carbon)
        "H",  # 13ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 13ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 13ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 13ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 13ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 13ALA C      - Carbon (carbonyl carbon)
        "O",  # 13ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 14ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 14ALA H      - Hydrogen (amino hydrogen)
        "C",  # 14ALA CA     - Carbon (alpha carbon)
        "H",  # 14ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 14ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 14ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 14ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 14ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 14ALA C      - Carbon (carbonyl carbon)
        "O",  # 14ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 15ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 15ALA H      - Hydrogen (amino hydrogen)
        "C",  # 15ALA CA     - Carbon (alpha carbon)
        "H",  # 15ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 15ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 15ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 15ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 15ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 15ALA C      - Carbon (carbonyl carbon)
        "O",  # 15ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 16ALA N      - Nitrogen (amino nitrogen in alanine 4)
        "H",  # 16ALA H      - Hydrogen (amino hydrogen)
        "C",  # 16ALA CA     - Carbon (alpha carbon)
        "H",  # 16ALA HA     - Hydrogen (alpha hydrogen)
        "C",  # 16ALA CB     - Carbon (beta carbon, methyl side chain)
        "H",  # 16ALA HB1    - Hydrogen (beta hydrogen 1)
        "H",  # 16ALA HB2    - Hydrogen (beta hydrogen 2)
        "H",  # 16ALA HB3    - Hydrogen (beta hydrogen 3)
        "C",  # 16ALA C      - Carbon (carbonyl carbon)
        "O",  # 16ALA O      - Oxygen (carbonyl oxygen)
        "N",  # 17NME N      - Nitrogen (N-methyl cap)
        "H",  # 17NME H      - Hydrogen (amino hydrogen)
        "C",  # 17NME CH3    - Carbon (methyl carbon in N-methyl cap)
        "H",  # 17NME HH31   - Hydrogen (methyl hydrogen 1)
        "H",  # 17NME HH32   - Hydrogen (methyl hydrogen 2)
        "H",  # 17NME HH33   - Hydrogen (methyl hydrogen 3)
    ]

    # Per‐map definitions
    _maps: dict[str, dict] = {
        "core": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # H1, CH3, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                3,  # C
                -1,  # O
                4,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                6,  # C
                -1,  # O
                7,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                9,  # C
                -1,  # O
                10,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                12,  # C
                -1,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                15,  # C
                -1,  # O
                16,
                -1,  # N H
                17,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                18,  # C
                -1,  # O
                19,
                -1,  # N H
                20,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                21,  # C
                -1,  # O
                22,
                -1,  # N H
                23,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                24,  # C
                -1,  # O
                25,
                -1,  # N H
                26,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                27,  # C
                -1,  # O
                28,
                -1,  # N H
                29,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                30,  # C
                -1,  # O
                31,
                -1,  # N H
                32,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                33,  # C
                -1,  # O
                34,
                -1,  # N H
                35,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                36,  # C
                -1,  # O
                37,
                -1,  # N H
                38,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                39,  # C
                -1,  # O
                40,
                -1,  # N H
                41,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                42,  # C
                -1,  # O
                43,
                -1,  # N H
                44,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                45,  # C
                -1,  # O
                46,  # N
                -1,  # H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                    1,
                    1,
                    2,
                ]
            ),  # 1=C, 2=N
            # Ala2: "cg_species": np.array([1,2,3,4,5]),  # 𝐶𝑂𝐶𝐻3, 𝑁𝐻, 𝐶𝐻𝐶𝐻3, 𝐶𝑂, 𝑁𝐻𝐶𝐻3
        },
        "coreMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                3,  # C
                -1,  # O
                4,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                6,  # C
                -1,  # O
                7,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                9,  # C
                -1,  # O
                10,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                12,  # C
                -1,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                15,  # C
                -1,  # O
                16,
                -1,  # N H
                17,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                18,  # C
                -1,  # O
                19,
                -1,  # N H
                20,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                21,  # C
                -1,  # O
                22,
                -1,  # N H
                23,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                24,  # C
                -1,  # O
                25,
                -1,  # N H
                26,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                27,  # C
                -1,  # O
                28,
                -1,  # N H
                29,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                30,  # C
                -1,  # O
                31,
                -1,  # N H
                32,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                33,  # C
                -1,  # O
                34,
                -1,  # N H
                35,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                36,  # C
                -1,  # O
                37,
                -1,  # N H
                38,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                39,  # C
                -1,  # O
                40,
                -1,  # N H
                41,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                42,  # C
                -1,  # O
                43,
                -1,  # N H
                44,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                45,  # C
                -1,  # O
                46,  # N
                -1,  # H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    2,
                    3,
                    4,
                    5,
                ]
            ),  # 𝐶𝑂𝐶𝐻3, 𝑁𝐻, 𝐶𝐻𝐶𝐻3, 𝐶𝑂, 𝑁𝐻𝐶𝐻3
            # Ala2: "cg_species": np.array([1,2,3,4,5]),  # 𝐶𝑂𝐶𝐻3, 𝑁𝐻, 𝐶𝐻𝐶𝐻3, 𝐶𝑂, 𝑁𝐻𝐶𝐻3
        },
        "coreBeta": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N H
                2,
                -1,  # CA HA
                3,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                4,  # C
                -1,  # O
                5,
                -1,  # N H
                6,
                -1,  # CA HA
                7,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                8,  # C
                -1,  # O
                9,
                -1,  # N H
                10,
                -1,  # CA HA
                11,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                12,  # C
                -1,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                15,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                16,  # C
                -1,  # O
                17,
                -1,  # N H
                18,
                -1,  # CA HA
                19,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                20,  # C
                -1,  # O
                21,
                -1,  # N H
                22,
                -1,  # CA HA
                23,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                24,  # C
                -1,  # O
                25,
                -1,  # N H
                26,
                -1,  # CA HA
                27,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                28,  # C
                -1,  # O
                29,
                -1,  # N H
                30,
                -1,  # CA HA
                31,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                32,  # C
                -1,  # O
                33,
                -1,  # N H
                34,
                -1,  # CA HA
                35,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                36,  # C
                -1,  # O
                37,
                -1,  # N H
                38,
                -1,  # CA HA
                39,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                40,  # C
                -1,  # O
                41,
                -1,  # N H
                42,
                -1,  # CA HA
                43,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                44,  # C
                -1,  # O
                45,
                -1,  # N H
                46,
                -1,  # CA HA
                47,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                48,  # C
                -1,  # O
                49,
                -1,  # N H
                50,
                -1,  # CA HA
                51,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                52,  # C
                -1,  # O
                53,
                -1,  # N H
                54,
                -1,  # CA HA
                55,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                56,  # C
                -1,  # O
                57,
                -1,  # N H
                58,
                -1,  # CA HA
                59,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                60,  # C
                -1,  # O
                61,  # N
                -1,  # H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                ]
            ),  # 1=C, 2=N
        },
        "heavyOnlyMap2": {
            "indices": [
                -1,
                0,
                -1,
                -1,  # H1,CH3,  H2, H3
                1,  # C
                2,  # O
                3,
                -1,  # N H
                4,
                -1,  # CA HA
                5,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                6,  # C
                7,  # O
                8,
                -1,  # N H
                9,
                -1,  # CA HA
                10,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                11,  # C
                12,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                15,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                16,  # C
                17,  # O
                18,
                -1,  # N H
                19,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH
            # Ala2 hmerged: "cg_species": np.array([1,2,3,4,5,2,3,1,4,1]),  # 1=CH3,2=C,3=O, 4=NH, 5=CH
        },
        "coreBeta": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N H
                2,
                -1,  # CA HA
                3,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                4,  # C
                -1,  # O
                5,
                -1,  # N H
                6,
                -1,  # CA HA
                7,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                8,  # C
                -1,  # O
                9,
                -1,  # N H
                10,
                -1,  # CA HA
                11,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                12,  # C
                -1,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                15,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                16,  # C
                -1,  # O
                17,
                -1,  # N H
                18,
                -1,  # CA HA
                19,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                20,  # C
                -1,  # O
                21,
                -1,  # N H
                22,
                -1,  # CA HA
                23,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                24,  # C
                -1,  # O
                25,
                -1,  # N H
                26,
                -1,  # CA HA
                27,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                28,  # C
                -1,  # O
                29,
                -1,  # N H
                30,
                -1,  # CA HA
                31,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                32,  # C
                -1,  # O
                33,
                -1,  # N H
                34,
                -1,  # CA HA
                35,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                36,  # C
                -1,  # O
                37,
                -1,  # N H
                38,
                -1,  # CA HA
                39,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                40,  # C
                -1,  # O
                41,
                -1,  # N H
                42,
                -1,  # CA HA
                43,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                44,  # C
                -1,  # O
                45,
                -1,  # N H
                46,
                -1,  # CA HA
                47,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                48,  # C
                -1,  # O
                49,
                -1,  # N H
                50,
                -1,  # CA HA
                51,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                52,  # C
                -1,  # O
                53,
                -1,  # N H
                54,
                -1,  # CA HA
                55,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                56,  # C
                -1,  # O
                57,
                -1,  # N H
                58,
                -1,  # CA HA
                59,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                60,  # C
                -1,  # O
                61,  # N
                -1,  # H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                    1,
                    1,
                    1,
                    2,
                ]
            ),  # 1=C, 2=N
        },
        "coreBetaMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N H
                2,
                -1,  # CA HA
                3,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                4,  # C
                -1,  # O
                5,
                -1,  # N H
                6,
                -1,  # CA HA
                7,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                8,  # C
                -1,  # O
                9,
                -1,  # N H
                10,
                -1,  # CA HA
                11,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                12,  # C
                -1,  # O
                13,
                -1,  # N H
                14,
                -1,  # CA HA
                15,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                16,  # C
                -1,  # O
                17,
                -1,  # N H
                18,
                -1,  # CA HA
                19,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                20,  # C
                -1,  # O
                21,
                -1,  # N H
                22,
                -1,  # CA HA
                23,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                24,  # C
                -1,  # O
                25,
                -1,  # N H
                26,
                -1,  # CA HA
                27,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                28,  # C
                -1,  # O
                29,
                -1,  # N H
                30,
                -1,  # CA HA
                31,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                32,  # C
                -1,  # O
                33,
                -1,  # N H
                34,
                -1,  # CA HA
                35,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                36,  # C
                -1,  # O
                37,
                -1,  # N H
                38,
                -1,  # CA HA
                39,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                40,  # C
                -1,  # O
                41,
                -1,  # N H
                42,
                -1,  # CA HA
                43,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                44,  # C
                -1,  # O
                45,
                -1,  # N H
                46,
                -1,  # CA HA
                47,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                48,  # C
                -1,  # O
                49,
                -1,  # N H
                50,
                -1,  # CA HA
                51,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                52,  # C
                -1,  # O
                53,
                -1,  # N H
                54,
                -1,  # CA HA
                55,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                56,  # C
                -1,  # O
                57,
                -1,  # N H
                58,
                -1,  # CA HA
                59,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                60,  # C
                -1,  # O
                61,  # N
                -1,  # H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    2,
                    3,
                    5,
                    4,
                    6,
                ]
            ),
            # Ala2: "cg_species": np.array([1,2,3,4,5,6]),  # 1=COCH3, 2=NH, 3=CA, 4=CO, 5=CB, 6=NHCH3
        },
        "conway": {
            "indices": [
                0,
                0,
                0,
                0,  # CH3, H1, H2, H3
                0,  # C
                0,  # O
                1,
                1,  # N H
                1,
                1,  # CA HA
                1,
                1,
                1,
                1,  # CB HB1 HB2 HB3
                2,  # C
                2,  # O
                3,
                3,  # N H
                3,
                3,  # CA HA
                3,
                3,
                3,
                3,  # CB HB1 HB2 HB3
                4,  # C
                4,  # O
                5,
                5,  # N H
                5,
                5,  # CA HA
                5,
                5,
                5,
                5,  # CB HB1 HB2 HB3
                6,  # C
                6,  # O
                7,
                7,  # N H
                7,
                7,  # CA HA
                7,
                7,
                7,
                7,  # CB HB1 HB2 HB3
                8,  # C
                8,  # O
                9,
                9,  # N H
                9,
                9,  # CA HA
                9,
                9,
                9,
                9,  # CB HB1 HB2 HB3
                10,  # C
                10,  # O
                11,
                11,  # N H
                11,
                11,  # CA HA
                11,
                11,
                11,
                11,  # CB HB1 HB2 HB3
                12,  # C
                12,  # O
                13,
                13,  # N H
                13,
                13,  # CA HA
                13,
                13,
                13,
                13,  # CB HB1 HB2 HB3
                14,  # C
                14,  # O
                15,
                15,  # N H
                15,
                15,  # CA HA
                15,
                15,
                15,
                15,  # CB HB1 HB2 HB3
                16,  # C
                16,  # O
                17,
                17,  # N H
                17,
                17,  # CA HA
                17,
                17,
                17,
                17,  # CB HB1 HB2 HB3
                18,  # C
                18,  # O
                19,
                19,  # N H
                19,
                19,  # CA HA
                19,
                19,
                19,
                19,  # CB HB1 HB2 HB3
                20,  # C
                20,  # O
                21,
                21,  # N H
                21,
                21,  # CA HA
                21,
                21,
                21,
                21,  # CB HB1 HB2 HB3
                22,  # C
                22,  # O
                23,
                23,  # N H
                23,
                23,  # CA HA
                23,
                23,
                23,
                23,  # CB HB1 HB2 HB3
                24,  # C
                24,  # O
                25,
                25,  # N H
                25,
                25,  # CA HA
                25,
                25,
                25,
                25,  # CB HB1 HB2 HB3
                26,  # C
                26,  # O
                27,
                27,  # N H
                27,
                27,  # CA HA
                27,
                27,
                27,
                27,  # CB HB1 HB2 HB3
                28,  # C
                28,  # O
                29,
                29,  # N H
                29,
                29,  # CA HA
                29,
                29,
                29,
                29,  # CB HB1 HB2 HB3
                30,  # C
                30,  # O
                31,
                31,  # N H
                31,
                31,
                31,
                31,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [
                    1,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    2,
                    3,
                    4,
                ]
            ),  # 1=CH3CO, 2=NHCACB, 3=CO, 4=NHCH3
        },
        "CA": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                0,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                1,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                3,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                4,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                6,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                7,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                9,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                10,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                12,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                13,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ),  # 1=CA
        },
        "CA-Map2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                0,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                1,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                3,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                4,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                6,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                7,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                9,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                10,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                12,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                13,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        },
        "CA-Map3": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                0,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                1,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                3,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                4,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                6,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                7,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                9,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                10,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                12,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                13,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]),
        },
        "CA-Map4": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                0,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                1,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                2,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                3,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                4,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                5,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                6,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                7,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                8,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                9,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                10,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                11,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                12,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                13,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                14,
                -1,  # CA HA
                -1,
                -1,
                -1,
                -1,  # CB HB1 HB2 HB3
                -1,  # C
                -1,  # O
                -1,
                -1,  # N H
                -1,
                -1,
                -1,
                -1,  # CH3 HH31 HH32 HH33
            ],
            "cg_species": np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]),
        },
    }

    at_masses = [mass_map[s] for s in map_species]

    def __init__(self):
        self.n_atoms = len(self.map_species)

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(
        self, name: str = "hmerged"
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights) for 'hmerged' or 'core'."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )

        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)  # shape (n_atoms,)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)  # shape (n_atoms,)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(
            indices_arr,  # map
            at_masses_arr,  # per-atom masses
            cg_masses,  # computed via bincount or segment_sum
        )

        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights


class Pro2_Map:
    """Class for Ala2 mapping. ASSUMES PEPSOL ATOM ORDERING."""

    map_species: list[str] = [
        "C",  # CH3
        "H",  # H1
        "H",  # H2
        "H",  # H3
        "C",  # C
        "O",  # O
        "N",  # N
        "C",  # CD
        "H",  # HD3
        "H",  # HD2
        "C",  # CG
        "H",  # HG3
        "H",  # HG2
        "C",  # CB
        "H",  # HB3
        "H",  # HB2
        "C",  # CA
        "H",  # HA
        "C",  # C
        "O",  # O
        "N",  # N
        "H",  # H
        "C",  # C
        "H",  # H1
        "H",  # H2
        "H",  # H3
    ]

    # Per‐map definitions
    _maps: dict[str, dict] = {
        "hmerged": {
            "indices": [
                0,
                0,
                0,
                0,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,  # N
                4,
                4,
                4,  # CD, HD3, HD2
                5,
                5,
                5,  # CG, HG3, HG2
                6,
                6,
                6,  # CB, HB3, HB2
                7,
                7,  # CA, HA
                8,  # C
                9,  # O
                10,
                10,  # N, H
                11,
                11,
                11,
                11,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 5, 5, 6, 2, 3, 7, 1]
            ),  # 1=CH3, 2=C, 3=O, 4=N, 5=CH2, 6=CH, 7=NH
        },
        "core": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,  # N
                -1,
                -1,
                -1,  # CD, HD3, HD2
                -1,
                -1,
                -1,  # CG, HG3, HG2
                -1,
                -1,
                -1,  # CB, HB3, HB2
                2,
                -1,  # CA, HA
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array([1, 2, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,  # N
                -1,
                -1,
                -1,  # CD, HD3, HD2
                -1,
                -1,
                -1,  # CG, HG3, HG2
                -1,
                -1,
                -1,  # CB, HB3, HB2
                2,
                -1,  # CA, HA
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array([1, 2, 3, 4, 5]),  # 𝐶𝑂𝐶𝐻3, 𝑁𝐻, CA, 𝐶𝑂, 𝑁𝐻𝐶𝐻3
        },
        "heavyOnly": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,  # N
                4,
                -1,
                -1,  # CD, HD3, HD2
                5,
                -1,
                -1,  # CG, HG3, HG2
                6,
                -1,
                -1,  # CB, HB3, HB2
                7,
                -1,  # CA, HA
                8,  # C
                9,  # O
                10,
                -1,  # N, H
                11,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 1, 2, 3, 1, 1, 1, 1, 1, 2, 3, 1]
            ),  # 1=C, 2=O, 3=N
        },
        "heavyOnlyMap2": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,  # N
                4,
                -1,
                -1,  # CD, HD3, HD2
                5,
                -1,
                -1,  # CG, HG3, HG2
                6,
                -1,
                -1,  # CB, HB3, HB2
                7,
                -1,  # CA, HA
                8,  # C
                9,  # O
                10,
                -1,  # N, H
                11,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 5, 5, 6, 2, 3, 7, 1]
            ),  # 1=CH3, 2=C, 3=O, 4=N, 5=CH2, 6=CH, 7=NH
        },
        "coreBeta": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,  # N
                -1,
                -1,
                -1,  # CD, HD3, HD2
                -1,
                -1,
                -1,  # CG, HG3, HG2
                2,
                -1,
                -1,  # CB, HB3, HB2
                3,
                -1,  # CA, HA
                4,  # C
                -1,  # O
                5,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array([1, 2, 1, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreBetaMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,  # N
                -1,
                -1,
                -1,  # CD, HD3, HD2
                -1,
                -1,
                -1,  # CG, HG3, HG2
                2,
                -1,
                -1,  # CB, HB3, HB2
                3,
                -1,  # CA, HA
                4,  # C
                -1,  # O
                5,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 6]
            ),  # 1=COCH3, 2=N, 3=CB, 4=CA, 5=CO, 6=NH
        },
    }

    at_masses = [mass_map[s] for s in map_species]

    def __init__(self):
        self.n_atoms = len(self.map_species)

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(
        self, name: str = "hmerged"
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights) for 'hmerged' or 'core'."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )

        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)  # shape (n_atoms,)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)  # shape (n_atoms,)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(
            indices_arr,  # map
            at_masses_arr,  # per-atom masses
            cg_masses,  # computed via bincount or segment_sum
        )

        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights


class Gly2_Map:
    """Class for Ala2 mapping. ASSUMES PEPSOL ATOM ORDERING."""

    map_species: list[str] = [
        "C",  #    ACE1-CH3 CH3
        "H",  #    ACE1-H1 H1
        "H",  #    ACE1-H2 H2
        "H",  #    ACE1-H3 H3
        "C",  #    ACE1-C C
        "O",  #    ACE1-O O
        "N",  #    GLY2-N N
        "H",  #    GLY2-H H
        "C",  #    GLY2-CA CA
        "H",  #    GLY2-HA3 HA3
        "H",  #    GLY2-HA2 HA2
        "C",  #   GLY2-C C
        "O",  #   GLY2-O O
        "N",  #   NME3-N N
        "H",  #   NME3-H H
        "C",  #   NME3-C C
        "H",  #   NME3-H1 H1
        "H",  #   NME3-H2 H2
        "H",  #   NME3-H3 H3
    ]

    # Per‐map definitions
    _maps: dict[str, dict] = {
        "hmerged": {
            "indices": [
                0,
                0,
                0,
                0,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                3,  # N, H
                4,
                4,
                4,  # CA, HA3, HA2
                5,  # C
                6,  # O
                7,
                7,  # N, H
                8,
                8,
                8,
                8,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 2, 3, 4, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH2
        },
        "core": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,
                -1,  # CA, HA3, HA2
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array([1, 2, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,
                -1,  # CA, HA3, HA2
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5]
            ),  # 1=COCH3, 2=N, 3=CA, 4=CO, 5=NHCH3
        },
        "heavyOnly": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                -1,  # N, H
                4,
                -1,
                -1,  # CA, HA3, HA2
                5,  # C
                6,  # O
                7,
                -1,  # N, H
                8,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array([1, 1, 2, 3, 1, 1, 2, 3, 1]),  # 1=C, 2=O, 3=N
        },
        "heavyOnlyMap2": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                -1,  # N, H
                4,
                -1,
                -1,  # CA, HA3, HA2
                5,  # C
                6,  # O
                7,
                -1,  # N, H
                8,
                -1,
                -1,
                -1,  # C, H1, H2, H3
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 2, 3, 4, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH2
        },
    }

    at_masses = [mass_map[s] for s in map_species]

    def __init__(self):
        self.n_atoms = len(self.map_species)

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(
        self, name: str = "hmerged"
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights) for 'hmerged' or 'core'."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )

        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)  # shape (n_atoms,)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)  # shape (n_atoms,)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(
            indices_arr,  # map
            at_masses_arr,  # per-atom masses
            cg_masses,  # computed via bincount or segment_sum
        )

        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights


class Thr2_Map:
    """Class for Ala2 mapping. ASSUMES PEPSOL ATOM ORDERING."""

    map_species: list[str] = [
        "C",  #    1ACE-CH3 CH3
        "H",  #    1ACE HH31
        "H",  #    1ACE HH32
        "H",  #    1ACE HH33
        "C",  #    1ACE-C C
        "O",  #    1ACE-O O
        "N",  #    2THR-N N
        "H",  #    2THR-H H
        "C",  #    2THR-CA CA
        "H",  #    2THR-HA HA
        "C",  #    2THR-CB CB
        "H",  #    2THR-HB HB
        "C",  #    2THR-CG2 CG2
        "H",  #    2THR-HG21 HG21
        "H",  #    2THR-HG22 HG22
        "H",  #    2THR-HG23 HG23
        "C",  #    2THR-OG1 OG1
        "H",  #    2THR-HG1 HG1
        "C",  #    2THR-C C
        "O",  #    2THR-O O
        "N",  #    3NME-N N
        "H",  #    3NME-H H
        "C",  #    3NME-CH3 CH3
        "H",  #    3NME HH31
        "H",  #    3NME HH32
        "H",  #    3NME HH33
    ]

    # Per‐map definitions
    _maps: dict[str, dict] = {
        "hmerged": {
            "indices": [
                0,
                0,
                0,
                0,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                3,  # N, H
                4,
                4,  # CA, HA
                5,
                5,  # CB, HB
                6,
                6,
                6,
                6,  # CG2, HG21, HG22, HG23
                7,
                7,  # OG1, HG1
                8,  # C
                9,  # O
                10,
                10,  # N, H
                11,
                11,
                11,
                11,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 5, 1, 6, 2, 3, 4, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH2, 6=OH
        },
        "core": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,  # CA, HA
                -1,
                -1,  # CB, HB
                -1,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                -1,
                -1,  # OG1, HG1
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array([1, 2, 1, 1, 2]),  # 1=C, 2=N
        },
        "coreMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,  # CA, HA
                -1,
                -1,  # CB, HB
                -1,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                -1,
                -1,  # OG1, HG1
                3,  # C
                -1,  # O
                4,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5]
            ),  # 1=COCH3, 2=N, 3=CA, 4=CO, 5=NHCH3
        },
        "heavyOnly": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                -1,  # N, H
                4,
                -1,  # CA, HA
                5,
                -1,  # CB, HB
                6,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                7,
                -1,  # OG1, HG1
                8,  # C
                9,  # O
                10,
                -1,  # N, H
                11,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array(
                [1, 1, 2, 3, 1, 1, 1, 2, 1, 2, 3, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH2, 6=OH
        },
        "heavyOnlyMap2": {
            "indices": [
                0,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                1,  # C
                2,  # O
                3,
                -1,  # N, H
                4,
                -1,  # CA, HA
                5,
                -1,  # CB, HB
                6,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                7,
                -1,  # OG1, HG1
                8,  # C
                9,  # O
                10,
                -1,  # N, H
                11,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 5, 1, 6, 2, 3, 4, 1]
            ),  # 1=CH3,2=C,3=O, 4=NH, 5=CH2, 6=OH
        },
        "coreBeta": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,  # CA, HA
                3,
                -1,  # CB, HB
                -1,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                -1,
                -1,  # OG1, HG1
                4,  # C
                -1,  # O
                5,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array([1, 2, 1, 1, 1, 2]),
        },
        "coreBetaMap2": {
            "indices": [
                -1,
                -1,
                -1,
                -1,  # CH3, H1, H2, H3
                0,  # C
                -1,  # O
                1,
                -1,  # N, H
                2,
                -1,  # CA, HA
                3,
                -1,  # CB, HB
                -1,
                -1,
                -1,
                -1,  # CG2, HG21, HG22, HG23
                -1,
                -1,  # OG1, HG1
                4,  # C
                -1,  # O
                5,
                -1,  # N, H
                -1,
                -1,
                -1,
                -1,  # CH3, HH31, HH32, HH33
            ],
            "cg_species": np.array(
                [1, 2, 3, 4, 5, 6]
            ),  # 1=COCH3, 2=N, 3=CA, 4=CB, 5=CO, 6=OH
        },
    }

    at_masses = [mass_map[s] for s in map_species]

    def __init__(self):
        self.n_atoms = len(self.map_species)

    def get_available_maps(self) -> list[str]:
        return list(self._maps)

    def get_map(
        self, name: str = "hmerged"
    ) -> tuple[list[int], np.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return (map_indices, cg_species, cg_masses, weights) for 'hmerged' or 'core'."""
        if name not in self._maps:
            raise ValueError(
                f"Invalid map '{name}'. choose one of {self.get_available_maps()}"
            )

        data = self._maps[name]
        indices = data["indices"]
        cg_species = data["cg_species"]
        n_cg = len(cg_species)

        indices_arr = jnp.array(indices, dtype=jnp.int32)  # shape (n_atoms,)
        at_masses_arr = jnp.array(self.at_masses, dtype=jnp.float32)  # shape (n_atoms,)
        cg_masses = jax.ops.segment_sum(at_masses_arr, indices_arr, n_cg)

        weights = get_map_weights(
            indices_arr,  # map
            at_masses_arr,  # per-atom masses
            cg_masses,  # computed via bincount or segment_sum
        )

        row_sums = jnp.sum(weights, axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6)

        return indices, cg_species, cg_masses, weights
