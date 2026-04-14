"""Bonded priors via Boltzmann inversion.

Derives effective bonded interaction parameters (bonds, angles, dihedrals)
from MD or CG trajectory data stored in a :class:`BaseDataset` instance,
using the Boltzmann inversion method:

    U(x) = -k_B T ln P(x) + const

Harmonic parameters (equilibrium value and spring constant) are additionally
estimated from the first two moments of the observed distribution, following
the equipartition theorem:

    x_0 = <x>,    k = k_B T / Var(x)

All internal coordinates are computed using the dataset's PBC-aware
``displacement_fn_X`` (via ``jax_md.space.periodic_general``), so minimum-
image convention is applied correctly for every geometry measurement.

Units throughout: positions in nm, energies in kJ/mol, angles in radians.
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax_md_mod import custom_quantity

# Boltzmann constant in kJ/(mol·K)
KB: float = 8.314462618e-3


# ---------------------------------------------------------------------------
# PBC-aware geometry (using jax_md displacement functions)
# ---------------------------------------------------------------------------

def _bond_lengths_pbc(
    positions: jnp.ndarray,
    bonds: jnp.ndarray,
    displacement_fn,
) -> np.ndarray:
    """PBC-aware bond lengths over all trajectory frames.

    Args:
        positions:      (F, N, 3) Cartesian positions in nm (JAX array).
        bonds:          (B, 2) integer atom-index pairs (JAX array).
        displacement_fn: jax_md displacement function (handles PBC).

    Returns:
        (F, B) float32 array of bond lengths in nm.
    """
    def _single_frame(pos):
        # pos: (N, 3)
        r1 = pos[bonds[:, 0]]                        # (B, 3)
        r2 = pos[bonds[:, 1]]                        # (B, 3)
        disps = jax.vmap(displacement_fn)(r1, r2)    # (B, 3)
        return jnp.linalg.norm(disps, axis=-1)       # (B,)

    batched = jax.vmap(_single_frame)                # (F, B)
    return np.asarray(batched(positions))


def _angles_pbc(
    positions: jnp.ndarray,
    angle_idxs: jnp.ndarray,
    displacement_fn,
) -> np.ndarray:
    """PBC-aware bond angles over all trajectory frames.

    Delegates to ``custom_quantity.angular_displacement`` which internally
    applies ``displacement_fn`` for each bond vector.

    Args:
        positions:      (F, N, 3) Cartesian positions in nm (JAX array).
        angle_idxs:     (A, 3) integer index triples; column 1 is central atom.
        displacement_fn: jax_md displacement function.

    Returns:
        (F, A) float32 array of angles in radians.
    """
    _fn = functools.partial(
        custom_quantity.angular_displacement,
        displacement_fn=displacement_fn,
        angle_idxs=angle_idxs,
        degrees=False,
    )
    batched = jax.vmap(_fn)   # maps over axis-0 of positions → (F, A)
    return np.asarray(batched(positions))


def _dihedrals_pbc(
    positions: jnp.ndarray,
    dihedral_idxs: jnp.ndarray,
    displacement_fn,
) -> np.ndarray:
    """PBC-aware dihedral angles over all trajectory frames.

    Delegates to ``custom_quantity.dihedral_displacement``.

    Args:
        positions:      (F, N, 3) Cartesian positions in nm (JAX array).
        dihedral_idxs:  (D, 4) integer index quadruples.
        displacement_fn: jax_md displacement function.

    Returns:
        (F, D) float32 array of dihedral angles in radians ∈ (−π, π].
    """
    _fn = functools.partial(
        custom_quantity.dihedral_displacement,
        displacement_fn=displacement_fn,
        dihedral_idxs=dihedral_idxs,
        degrees=False,
    )
    batched = jax.vmap(_fn)   # maps over axis-0 of positions → (F, D)
    return np.asarray(batched(positions))


# ---------------------------------------------------------------------------
# Boltzmann inversion
# ---------------------------------------------------------------------------

def boltzmann_inversion(
    samples: np.ndarray,
    kT: float,
    n_bins: int = 100,
    jacobian: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Boltzmann inversion to a 1-D distribution.

    Computes the effective potential of mean force:

        U(x) = -kT · ln[ P_obs(x) / J(x) ] + const

    where J(x) is an optional Jacobian factor:

    * ``'r2'``:  J(r) = r²    — bond length in 3-D spherical shell.
    * ``'sin'``: J(θ) = sin θ — bond angle in 3-D spherical coordinates.
    * ``None``:  J = 1        — dihedral angles, or direct 1-D inversion.

    Args:
        samples:  1-D array of coordinate samples.
        kT:       Thermal energy k_B T in kJ/mol.
        n_bins:   Number of histogram bins.
        jacobian: Jacobian correction: ``'r2'``, ``'sin'``, or ``None``.

    Returns:
        bin_centers: (n_bins,) bin centre values.
        U:           (n_bins,) effective potential in kJ/mol, shifted so
                     min(U) = 0.  NaN where the histogram count is zero.
        P:           (n_bins,) corrected probability density.
    """
    counts, edges = np.histogram(samples, bins=n_bins, density=True)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    P = counts.copy().astype(float)
    if jacobian == "r2":
        P = np.where(bin_centers > 0, P / bin_centers**2, 0.0)
    elif jacobian == "sin":
        sin_vals = np.abs(np.sin(bin_centers))
        P = np.where(sin_vals > 1e-10, P / sin_vals, 0.0)

    valid = P > 0.0
    U = np.full_like(P, np.nan)
    U[valid] = -kT * np.log(P[valid])
    U -= np.nanmin(U)

    return bin_centers, U, P


def _fit_harmonic(samples: np.ndarray, kT: float) -> tuple[float, float]:
    """Estimate harmonic parameters from the sample mean and variance.

    Applies the equipartition theorem: Var(x) = kT / k, hence k = kT / Var(x).

    Args:
        samples: 1-D array of coordinate samples.
        kT:      Thermal energy k_B T in kJ/mol.

    Returns:
        x0: Equilibrium value (mean of samples).
        k:  Spring constant in kJ/(mol · unit²).
    """
    x0 = float(np.mean(samples))
    var = float(np.var(samples))
    k = kT / var if var > 0.0 else np.inf
    return x0, k


# ---------------------------------------------------------------------------
# BoltzmannPrior
# ---------------------------------------------------------------------------

class BoltzmannPrior:
    """Derives bonded priors via Boltzmann inversion from MD trajectory data.

    Supports both atomistic and coarse-grained trajectories stored in a
    :class:`~cgbench.core.dataset.BaseDataset` instance.

    All internal coordinates (bond lengths, angles, dihedrals) are computed
    using the dataset's PBC-aware ``displacement_fn_X`` so that minimum-image
    convention is respected throughout.

    Bond, angle, and dihedral distributions are then inverted via:

        U(x) = -k_B T ln P(x) + const

    Harmonic parameters (x_0, k) are additionally estimated from the first
    two moments of the observed distribution.

    Args:
        dataset: :class:`~cgbench.core.dataset.BaseDataset` instance.
                 :meth:`~cgbench.core.dataset.BaseDataset.load_traj` must have
                 been called before instantiation.  For CG priors,
                 :meth:`~cgbench.core.dataset.BaseDataset.coarse_grain` must
                 additionally have been called.
        T:       Simulation temperature in Kelvin (default 300 K).
    """

    def __init__(self, dataset, T: float = 300.0) -> None:
        self.dataset = dataset
        self.T = T
        self.kT: float = KB * T  # kJ/mol

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _displacement_fn(self):
        """Return the dataset's Cartesian PBC displacement function."""
        fn = getattr(self.dataset, "displacement_fn_X", None)
        if fn is None:
            raise RuntimeError(
                "dataset.displacement_fn_X is not set. "
                "Call load_traj() (and coarse_grain() for CG priors) first."
            )
        return fn

    def _positions(self, split: str, cg: bool) -> jnp.ndarray:
        """Return (F, N, 3) Cartesian positions in nm as a JAX array."""
        if cg:
            if not hasattr(self.dataset, "cg_dataset_X"):
                raise RuntimeError(
                    "Call dataset.coarse_grain(map_name) before requesting CG priors."
                )
            return jnp.array(self.dataset.cg_dataset_X[split]["R"])
        if self.dataset.dataset_X is None:
            raise RuntimeError(
                "Call dataset.load_traj() before requesting atomistic priors."
            )
        return jnp.array(self.dataset.dataset_X[split]["R"])

    def _ensure_cg_topology(self) -> None:
        """Populate cg_bond/angle/dihedral_index on the dataset if missing.

        When ``coarse_grain()`` hits its cached-dataset early-return path it
        skips the topology derivation block.  We recover by calling the map
        object directly using the stored ``cg_map_name``.
        """
        if getattr(self.dataset, "cg_bond_index", None) is not None:
            return

        map_name = getattr(self.dataset, "cg_map_name", None)
        map_obj = getattr(self.dataset, "map_obj", None)
        if map_name is None or map_obj is None:
            return
        if not hasattr(map_obj, "get_cg_topology"):
            return

        bonds, angles, dihedrals = map_obj.get_cg_topology(map_name)
        self.dataset.cg_bond_index = bonds
        self.dataset.cg_angle_index = angles
        self.dataset.cg_dihedral_index = dihedrals

    @staticmethod
    def _as_rows(arr, n_cols: int) -> np.ndarray | None:
        """Return topology terms as shape (N, n_cols).

        Handles both conventions used in this codebase:
        - row-major: (N, n_cols)
        - column-major: (n_cols, N)

        For square ambiguous cases (e.g. 4x4 dihedrals), select the
        orientation with the larger number of rows containing distinct
        indices, which is the physically valid form for bonded terms.
        """
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.ndim != 2:
            return a

        if a.shape[1] == n_cols and a.shape[0] != n_cols:
            return a
        if a.shape[0] == n_cols and a.shape[1] != n_cols:
            return a.T

        if a.shape[0] == n_cols and a.shape[1] == n_cols:
            rows_a = a
            rows_t = a.T

            def _distinct_row_count(x: np.ndarray) -> int:
                return int(sum(len(set(map(int, row.tolist()))) == n_cols for row in x))

            if _distinct_row_count(rows_t) > _distinct_row_count(rows_a):
                return rows_t
            return rows_a

        return a

    def _bonds(self, cg: bool) -> np.ndarray | None:
        """Return (B, 2) bond index array or None."""
        if cg:
            self._ensure_cg_topology()
            return self._as_rows(getattr(self.dataset, "cg_bond_index", None), 2)
        return getattr(self.dataset, "bonds", None)

    def _angles(self, cg: bool) -> np.ndarray | None:
        """Return (A, 3) angle index array or None."""
        if cg:
            self._ensure_cg_topology()
            return self._as_rows(getattr(self.dataset, "cg_angle_index", None), 3)
        return getattr(self.dataset, "angles", None)

    def _dihedrals(self, cg: bool) -> np.ndarray | None:
        """Return (D, 4) dihedral index array or None."""
        if cg:
            self._ensure_cg_topology()
            return self._as_rows(getattr(self.dataset, "cg_dihedral_index", None), 4)
        return getattr(self.dataset, "dihedrals", None)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def compute_bond_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        jacobian: str | None = None,
    ) -> dict:
        """Derive bond priors via Boltzmann inversion.

        Bond lengths are computed with full PBC awareness via the dataset's
        ``displacement_fn_X``.

        Args:
            split:    Dataset split (``"training"``, ``"validation"``, …).
            cg:       Use CG positions and CG bond topology when ``True``.
            n_bins:   Number of histogram bins.
            jacobian: Jacobian correction for Boltzmann inversion.
                      ``'r2'`` applies a 1/r² factor for the 3-D spherical
                      shell volume element; ``None`` performs direct 1-D
                      inversion.

        Returns:
            Dict keyed by ``(i, j)`` atom/bead index pairs.  Each value is
            a dict with:

            * ``'r0'``     - equilibrium bond length in nm (distribution mean).
            * ``'k'``      - harmonic spring constant in kJ/(mol·nm²).
            * ``'r_grid'`` - histogram bin centres in nm.
            * ``'U'``      - effective potential in kJ/mol (min-shifted to 0).
            * ``'P'``      - corrected probability density.
            * ``'samples'``- raw bond-length samples in nm.
        """
        bonds = self._bonds(cg)
        if bonds is None or len(bonds) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        bonds_jnp = jnp.array(bonds, dtype=jnp.int32)

        lengths = _bond_lengths_pbc(positions, bonds_jnp, displacement_fn)  # (F, B)

        results: dict = {}
        for b in range(len(bonds)):
            i, j = int(bonds[b, 0]), int(bonds[b, 1])
            samples = np.asarray(lengths[:, b])
            r_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian)
            r0, k = _fit_harmonic(samples, self.kT)
            results[(i, j)] = {
                "r0": r0,
                "k": k,
                "r_grid": r_grid,
                "U": U,
                "P": P,
                "samples": samples,
            }
        return results

    def compute_angle_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        jacobian: str | None = None,
    ) -> dict:
        """Derive angle priors via Boltzmann inversion.

        Angles are computed via ``custom_quantity.angular_displacement`` which
        applies ``displacement_fn_X`` for each bond vector.

        Args:
            split:    Dataset split.
            cg:       Use CG positions and CG angle topology when ``True``.
            n_bins:   Number of histogram bins.
            jacobian: Jacobian correction: ``'sin'`` divides by sin(θ) to
                      remove the spherical-coordinate volume factor;
                      ``None`` performs direct 1-D inversion.

        Returns:
            Dict keyed by ``(i, j, k)`` index triples (j is the central atom).
            Each value is a dict with:

            * ``'theta0'``    - equilibrium angle in radians (distribution mean).
            * ``'k'``         - harmonic spring constant in kJ/(mol·rad²).
            * ``'theta_grid'``- bin centres in radians.
            * ``'U'``         - effective potential in kJ/mol.
            * ``'P'``         - corrected probability density.
            * ``'samples'``   - raw angle samples in radians.
        """
        angle_indices = self._angles(cg)
        if angle_indices is None or len(angle_indices) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        angle_idxs_jnp = jnp.array(angle_indices, dtype=jnp.int32)

        ang = _angles_pbc(positions, angle_idxs_jnp, displacement_fn)  # (F, A)

        results: dict = {}
        for a in range(len(angle_indices)):
            i, j, k = (
                int(angle_indices[a, 0]),
                int(angle_indices[a, 1]),
                int(angle_indices[a, 2]),
            )
            samples = np.asarray(ang[:, a])
            theta_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian)
            theta0, k_val = _fit_harmonic(samples, self.kT)
            results[(i, j, k)] = {
                "theta0": theta0,
                "k": k_val,
                "theta_grid": theta_grid,
                "U": U,
                "P": P,
                "samples": samples,
            }
        return results

    def compute_dihedral_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
    ) -> dict:
        """Derive dihedral priors via Boltzmann inversion.

        Dihedrals are computed via ``custom_quantity.dihedral_displacement``
        which applies ``displacement_fn_X`` for each bond vector.
        No Jacobian correction is applied (uniform measure on the circle).

        Args:
            split:  Dataset split.
            cg:     Use CG positions and CG dihedral topology when ``True``.
            n_bins: Number of histogram bins.

        Returns:
            Dict keyed by ``(i, j, k, l)`` index quadruples.  Each value is a
            dict with:

            * ``'phi0'``      - mean dihedral angle in radians.
            * ``'phi_grid'``  - bin centres in radians.
            * ``'U'``         - effective potential in kJ/mol.
            * ``'P'``         - probability density.
            * ``'samples'``   - raw dihedral angle samples in radians.
        """
        dihedral_indices = self._dihedrals(cg)
        if dihedral_indices is None or len(dihedral_indices) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        dihedral_idxs_jnp = jnp.array(dihedral_indices, dtype=jnp.int32)

        dih = _dihedrals_pbc(positions, dihedral_idxs_jnp, displacement_fn)  # (F, D)

        results: dict = {}
        for d in range(len(dihedral_indices)):
            i, j, k, l = (
                int(dihedral_indices[d, 0]),
                int(dihedral_indices[d, 1]),
                int(dihedral_indices[d, 2]),
                int(dihedral_indices[d, 3]),
            )
            samples = np.asarray(dih[:, d])
            phi_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian=None)
            results[(i, j, k, l)] = {
                "phi0": float(np.mean(samples)),
                "phi_grid": phi_grid,
                "U": U,
                "P": P,
                "samples": samples,
            }
        return results

    def compute_all_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        bond_jacobian: str | None = None,
        angle_jacobian: str | None = None,
    ) -> dict:
        """Compute all bonded priors (bonds, angles, dihedrals) at once.

        Args:
            split:          Dataset split.
            cg:             Use CG topology when ``True``.
            n_bins:         Number of histogram bins for all terms.
            bond_jacobian:  Jacobian correction for bonds (default ``None``).
            angle_jacobian: Jacobian correction for angles (default ``None``).

        Returns:
            Dict with keys ``'bonds'``, ``'angles'``, ``'dihedrals'``.
        """
        return {
            "bonds": self.compute_bond_priors(split, cg, n_bins, bond_jacobian),
            "angles": self.compute_angle_priors(split, cg, n_bins, angle_jacobian),
            "dihedrals": self.compute_dihedral_priors(split, cg, n_bins),
        }
