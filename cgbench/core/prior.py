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

By default, interactions with the same species combination are pooled into a
single distribution (``group_by_species=True``).  For example, in a two-site
hexane system all 100 identical 1-1 bonds contribute to one pooled histogram
and one set of Boltzmann-inversion parameters.  Results are keyed by the
canonical species tuple; the contributing atom-index pairs are stored under
the ``'indices'`` key.

Units throughout: positions in nm, energies in kJ/mol, angles in radians.
"""

import functools
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax_md_mod import custom_quantity
from jax_md_mod.custom_interpolate import MonotonicInterpolate

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
# Canonical species keys
# ---------------------------------------------------------------------------

def _canonical_bond_key(s_i: int, s_j: int) -> tuple[int, int]:
    """Canonical (unordered) species key for a bond."""
    return (min(s_i, s_j), max(s_i, s_j))


def _canonical_angle_key(s_i: int, s_j: int, s_k: int) -> tuple[int, int, int]:
    """Canonical species key for an angle; s_j is the central species.

    The two end species may be swapped while the central one is held fixed.
    """
    fwd = (s_i, s_j, s_k)
    rev = (s_k, s_j, s_i)
    return min(fwd, rev)


def _canonical_dihedral_key(
    s_i: int, s_j: int, s_k: int, s_l: int
) -> tuple[int, int, int, int]:
    """Canonical species key for a dihedral (forward/reverse symmetry)."""
    fwd = (s_i, s_j, s_k, s_l)
    rev = (s_l, s_k, s_j, s_i)
    return min(fwd, rev)


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

    By default (``group_by_species=True``) all interactions that share the
    same canonical species combination are pooled into a single distribution.
    Results are then keyed by the species tuple, e.g. ``(0, 0)`` for a
    homo-species bond.  The full list of contributing atom-index tuples is
    stored under the ``'indices'`` key of each entry.

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

    def _species_array(self, cg: bool) -> np.ndarray | None:
        """Return per-bead/atom species index array, or None if unavailable."""
        if cg:
            s = getattr(self.dataset, "cg_species", None)
        else:
            s = getattr(self.dataset, "species", None)
        return None if s is None else np.asarray(s).ravel()

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def compute_bond_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        jacobian: str | None = None,
        group_by_species: bool = True,
    ) -> dict:
        """Derive bond priors via Boltzmann inversion.

        Bond lengths are computed with full PBC awareness via the dataset's
        ``displacement_fn_X``.

        Args:
            split:            Dataset split (``"training"``, ``"validation"``, …).
            cg:               Use CG positions and CG bond topology when ``True``.
            n_bins:           Number of histogram bins.
            jacobian:         Jacobian correction for Boltzmann inversion.
                              ``'r2'`` applies a 1/r² factor for the 3-D spherical
                              shell volume element; ``None`` performs direct 1-D
                              inversion.
            group_by_species: When ``True`` (default), pool all bonds that share
                              the same canonical species pair into one distribution.
                              Results are keyed by ``(s_i, s_j)`` with ``s_i ≤ s_j``.
                              When ``False``, one entry per bond, keyed by
                              ``(atom_i, atom_j)``.

        Returns:
            Dict whose keys are either species pairs ``(s_i, s_j)`` (when
            ``group_by_species=True``) or atom-index pairs ``(i, j)`` (when
            ``False``).  Each value is a dict with:

            * ``'r0'``      - equilibrium bond length in nm (distribution mean).
            * ``'k'``       - harmonic spring constant in kJ/(mol·nm²).
            * ``'r_grid'``  - histogram bin centres in nm.
            * ``'U'``       - effective potential in kJ/mol (min-shifted to 0).
            * ``'P'``       - corrected probability density.
            * ``'samples'`` - pooled bond-length samples in nm.
            * ``'indices'`` - list of ``(i, j)`` atom-index pairs that contributed
                              (only present when ``group_by_species=True``).
        """
        bonds = self._bonds(cg)
        if bonds is None or len(bonds) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        bonds_jnp = jnp.array(bonds, dtype=jnp.int32)

        lengths = _bond_lengths_pbc(positions, bonds_jnp, displacement_fn)  # (F, B)

        if not group_by_species:
            results: dict = {}
            for b in range(len(bonds)):
                i, j = int(bonds[b, 0]), int(bonds[b, 1])
                samples = np.asarray(lengths[:, b])
                r_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian)
                r0, k = _fit_harmonic(samples, self.kT)
                results[(i, j)] = {
                    "r0": r0, "k": k,
                    "r_grid": r_grid, "U": U, "P": P,
                    "samples": samples,
                }
            return results

        # --- group by canonical species pair ---
        species = self._species_array(cg)
        groups: dict[tuple, list[np.ndarray]] = {}
        group_indices: dict[tuple, list[tuple[int, int]]] = {}

        for b in range(len(bonds)):
            i, j = int(bonds[b, 0]), int(bonds[b, 1])
            if species is not None:
                key = _canonical_bond_key(int(species[i]), int(species[j]))
            else:
                key = (i, j)
            groups.setdefault(key, []).append(np.asarray(lengths[:, b]))
            group_indices.setdefault(key, []).append((i, j))

        results = {}
        for key, sample_list in sorted(groups.items()):
            samples = np.concatenate(sample_list)
            r_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian)
            r0, k = _fit_harmonic(samples, self.kT)
            results[key] = {
                "r0": r0, "k": k,
                "r_grid": r_grid, "U": U, "P": P,
                "samples": samples,
                "indices": group_indices[key],
            }
        return results

    def compute_angle_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        jacobian: str | None = None,
        group_by_species: bool = True,
    ) -> dict:
        """Derive angle priors via Boltzmann inversion.

        Angles are computed via ``custom_quantity.angular_displacement`` which
        applies ``displacement_fn_X`` for each bond vector.

        Args:
            split:            Dataset split.
            cg:               Use CG positions and CG angle topology when ``True``.
            n_bins:           Number of histogram bins.
            jacobian:         Jacobian correction: ``'sin'`` divides by sin(θ) to
                              remove the spherical-coordinate volume factor;
                              ``None`` performs direct 1-D inversion.
            group_by_species: When ``True`` (default), pool all angles that share
                              the same canonical species triple (end species may be
                              swapped, central species is held fixed) into one
                              distribution.  Results are keyed by the canonical
                              ``(s_i, s_j, s_k)`` triple.

        Returns:
            Dict keyed by ``(s_i, s_j, s_k)`` species triples (j central) or
            ``(i, j, k)`` atom-index triples when ``group_by_species=False``.
            Each value is a dict with:

            * ``'theta0'``    - equilibrium angle in radians (distribution mean).
            * ``'k'``         - harmonic spring constant in kJ/(mol·rad²).
            * ``'theta_grid'``- bin centres in radians.
            * ``'U'``         - effective potential in kJ/mol.
            * ``'P'``         - corrected probability density.
            * ``'samples'``   - pooled angle samples in radians.
            * ``'indices'``   - list of ``(i, j, k)`` atom-index triples that
                              contributed (only when ``group_by_species=True``).
        """
        angle_indices = self._angles(cg)
        if angle_indices is None or len(angle_indices) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        angle_idxs_jnp = jnp.array(angle_indices, dtype=jnp.int32)

        ang = _angles_pbc(positions, angle_idxs_jnp, displacement_fn)  # (F, A)

        if not group_by_species:
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
                    "theta0": theta0, "k": k_val,
                    "theta_grid": theta_grid, "U": U, "P": P,
                    "samples": samples,
                }
            return results

        # --- group by canonical species triple ---
        species = self._species_array(cg)
        groups: dict[tuple, list[np.ndarray]] = {}
        group_indices: dict[tuple, list[tuple[int, int, int]]] = {}

        for a in range(len(angle_indices)):
            i, j, k = (
                int(angle_indices[a, 0]),
                int(angle_indices[a, 1]),
                int(angle_indices[a, 2]),
            )
            if species is not None:
                key = _canonical_angle_key(int(species[i]), int(species[j]), int(species[k]))
            else:
                key = (i, j, k)
            groups.setdefault(key, []).append(np.asarray(ang[:, a]))
            group_indices.setdefault(key, []).append((i, j, k))

        results = {}
        for key, sample_list in sorted(groups.items()):
            samples = np.concatenate(sample_list)
            theta_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian)
            theta0, k_val = _fit_harmonic(samples, self.kT)
            results[key] = {
                "theta0": theta0, "k": k_val,
                "theta_grid": theta_grid, "U": U, "P": P,
                "samples": samples,
                "indices": group_indices[key],
            }
        return results

    def compute_dihedral_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        group_by_species: bool = True,
    ) -> dict:
        """Derive dihedral priors via Boltzmann inversion.

        Dihedrals are computed via ``custom_quantity.dihedral_displacement``
        which applies ``displacement_fn_X`` for each bond vector.
        No Jacobian correction is applied (uniform measure on the circle).

        Args:
            split:            Dataset split.
            cg:               Use CG positions and CG dihedral topology when ``True``.
            n_bins:           Number of histogram bins.
            group_by_species: When ``True`` (default), pool all dihedrals that share
                              the same canonical species quadruple (forward/reverse
                              symmetry) into one distribution.

        Returns:
            Dict keyed by ``(s_i, s_j, s_k, s_l)`` species quadruples or
            ``(i, j, k, l)`` atom-index quadruples when ``group_by_species=False``.
            Each value is a dict with:

            * ``'phi0'``     - mean dihedral angle in radians.
            * ``'phi_grid'`` - bin centres in radians.
            * ``'U'``        - effective potential in kJ/mol.
            * ``'P'``        - probability density.
            * ``'samples'``  - pooled dihedral angle samples in radians.
            * ``'indices'``  - list of ``(i, j, k, l)`` atom-index quadruples that
                              contributed (only when ``group_by_species=True``).
        """
        dihedral_indices = self._dihedrals(cg)
        if dihedral_indices is None or len(dihedral_indices) == 0:
            return {}

        positions = self._positions(split, cg)
        displacement_fn = self._displacement_fn()
        dihedral_idxs_jnp = jnp.array(dihedral_indices, dtype=jnp.int32)

        dih = _dihedrals_pbc(positions, dihedral_idxs_jnp, displacement_fn)  # (F, D)

        if not group_by_species:
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
                    "phi_grid": phi_grid, "U": U, "P": P,
                    "samples": samples,
                }
            return results

        # --- group by canonical species quadruple ---
        species = self._species_array(cg)
        groups: dict[tuple, list[np.ndarray]] = {}
        group_indices: dict[tuple, list[tuple[int, int, int, int]]] = {}

        for d in range(len(dihedral_indices)):
            i, j, k, l = (
                int(dihedral_indices[d, 0]),
                int(dihedral_indices[d, 1]),
                int(dihedral_indices[d, 2]),
                int(dihedral_indices[d, 3]),
            )
            if species is not None:
                key = _canonical_dihedral_key(
                    int(species[i]), int(species[j]),
                    int(species[k]), int(species[l]),
                )
            else:
                key = (i, j, k, l)
            groups.setdefault(key, []).append(np.asarray(dih[:, d]))
            group_indices.setdefault(key, []).append((i, j, k, l))

        results = {}
        for key, sample_list in sorted(groups.items()):
            samples = np.concatenate(sample_list)
            phi_grid, U, P = boltzmann_inversion(samples, self.kT, n_bins, jacobian=None)
            results[key] = {
                "phi0": float(np.mean(samples)),
                "phi_grid": phi_grid, "U": U, "P": P,
                "samples": samples,
                "indices": group_indices[key],
            }
        return results

    def compute_all_priors(
        self,
        split: str = "training",
        cg: bool = False,
        n_bins: int = 100,
        bond_jacobian: str | None = None,
        angle_jacobian: str | None = None,
        group_by_species: bool = True,
    ) -> dict:
        """Compute all bonded priors (bonds, angles, dihedrals) at once.

        Args:
            split:            Dataset split.
            cg:               Use CG topology when ``True``.
            n_bins:           Number of histogram bins for all terms.
            bond_jacobian:    Jacobian correction for bonds (default ``None``).
            angle_jacobian:   Jacobian correction for angles (default ``None``).
            group_by_species: Pool interactions of the same species combination
                              (default ``True``).

        Returns:
            Dict with keys ``'bonds'``, ``'angles'``, ``'dihedrals'``.
        """
        return {
            "bonds": self.compute_bond_priors(
                split, cg, n_bins, bond_jacobian, group_by_species
            ),
            "angles": self.compute_angle_priors(
                split, cg, n_bins, angle_jacobian, group_by_species
            ),
            "dihedrals": self.compute_dihedral_priors(
                split, cg, n_bins, group_by_species
            ),
        }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fill_nan_u(U: np.ndarray) -> np.ndarray:
    """Replace NaN in a PMF array with the maximum finite value."""
    valid = ~np.isnan(U)
    if not np.any(valid):
        return np.zeros_like(U)
    return np.where(valid, U, np.nanmax(U))


# ---------------------------------------------------------------------------
# Single-frame geometry helpers (used by the energy-function template)
# ---------------------------------------------------------------------------

def _make_geometry_fns(displacement_fn):
    """Return PBC-aware single-frame geometry evaluation functions.

    Args:
        displacement_fn: jax_md displacement function (handles PBC).

    Returns:
        ``(_bond_length, _angle, _dihedral)`` — each takes ``(pos, *indices)``
        where ``pos`` is ``(N, 3)`` and indices are integer atom indices.
    """
    def _bond_length(pos, i, j):
        return jnp.linalg.norm(displacement_fn(pos[i], pos[j]))

    def _angle(pos, i, j, k):
        # j is central
        u = displacement_fn(pos[i], pos[j])   # vector j→i
        v = displacement_fn(pos[k], pos[j])   # vector j→k
        cos_t = jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-12)
        # Clip strictly inside (-1, 1): arccos'(x) = -1/sqrt(1-x^2) diverges at
        # x = ±1.  In float32, cos_t rounds to exactly ±1.0 for angles near 0°/180°
        # (e.g., the near-linear 3-site hexane chain), making the gradient NaN.
        # Keeping 1e-6 margin ensures 1-cos_t^2 >= ~2e-6, so the gradient stays finite.
        return jnp.arccos(jnp.clip(cos_t, -1.0 + 1e-6, 1.0 - 1e-6))

    def _dihedral(pos, i, j, k, l):
        b1 = displacement_fn(pos[j], pos[i])   # i→j
        b2 = displacement_fn(pos[k], pos[j])   # j→k
        b3 = displacement_fn(pos[l], pos[k])   # k→l
        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)
        b2_hat = b2 / (jnp.linalg.norm(b2) + 1e-12)
        m1 = jnp.cross(n1, b2_hat)
        # arctan2(y, x) has an undefined gradient at (0, 0).  When a constituent
        # angle (i-j-k or j-k-l) is exactly 180° in float32, the cross products
        # n1/n2 collapse to zero, making both arguments exactly zero.
        # Double-where pattern: replace (y, x) with a safe fallback (0, 1) so the
        # backward pass never evaluates 0 * NaN.  A tiny-but-nonzero eps2 for the
        # threshold avoids triggering on ordinary near-zero float32 noise.
        y = jnp.dot(m1, n2)
        x = jnp.dot(n1, n2)
        r2 = y ** 2 + x ** 2
        eps2 = jnp.float32(1e-12)
        degenerate = r2 < eps2
        y_s = jnp.where(degenerate, jnp.float32(0.), y)
        x_s = jnp.where(degenerate, jnp.float32(1.), x)
        return jnp.where(degenerate, jnp.float32(0.), jnp.arctan2(y_s, x_s))

    return _bond_length, _angle, _dihedral


# ---------------------------------------------------------------------------
# Prior "1dfunc": harmonic bonds/angles + truncated Fourier dihedrals
# ---------------------------------------------------------------------------

def _fit_harmonic_pmf(
    x_grid: np.ndarray,
    U: np.ndarray,
    x0_fallback: float | None = None,
    k_fallback: float | None = None,
) -> tuple[float, float]:
    """Fit U(x) ≈ ½ k (x − x₀)² to a BI PMF by non-linear least squares.

    Only non-NaN bins are used.  Falls back to equipartition estimates on
    failure.

    Args:
        x_grid:      Bin-centre coordinate array.
        U:           PMF values (kJ/mol); NaN where unsampled.
        x0_fallback: Equilibrium value to use on fitting failure.
        k_fallback:  Spring constant to use on fitting failure.

    Returns:
        ``(x0, k)`` in the natural units of *x_grid* and kJ/mol/unit².
    """
    from scipy.optimize import curve_fit

    valid = ~np.isnan(U)
    if np.sum(valid) < 3:
        return (x0_fallback or float(np.nanmean(x_grid)), k_fallback or 1000.0)

    x_v = x_grid[valid].astype(np.float64)
    U_v = U[valid].astype(np.float64)
    x0_init = float(x_v[np.argmin(U_v)])
    k_init = float(k_fallback or 500.0)

    try:
        def _harmonic(x, x0, k):
            return 0.5 * k * (x - x0) ** 2

        popt, _ = curve_fit(
            _harmonic,
            x_v,
            U_v,
            p0=[x0_init, k_init],
            bounds=([x_v.min(), 0.0], [x_v.max(), np.inf]),
            maxfev=5000,
        )
        x0_fit, k_fit = float(popt[0]), float(popt[1])
        if not (np.isfinite(x0_fit) and np.isfinite(k_fit) and k_fit > 0):
            raise ValueError("unphysical fit parameters")
        return x0_fit, k_fit
    except Exception:
        return (x0_fallback or x0_init, k_fallback or k_init)


def _fit_fourier_pmf(
    phi_grid: np.ndarray,
    U: np.ndarray,
    n_fourier: int = 5,
) -> np.ndarray:
    """Fit a truncated Fourier series to a dihedral BI PMF by linear least squares.

    The model is:

        U(φ) = a₀ + Σₙ₌₁ᴺ [ aₙ cos(n φ) + bₙ sin(n φ) ]

    giving ``1 + 2·n_fourier`` coefficients stored as
    ``[a₀, a₁, b₁, a₂, b₂, …, aₙ, bₙ]``.

    Args:
        phi_grid: Bin-centre angles in radians.
        U:        PMF values (kJ/mol); NaN where unsampled.
        n_fourier: Number of Fourier terms N.

    Returns:
        Float32 array of shape ``(1 + 2·n_fourier,)``.
    """
    n_params = 1 + 2 * n_fourier
    valid = ~np.isnan(U)
    if np.sum(valid) < n_params:
        return np.zeros(n_params, dtype=np.float32)

    phi_v = phi_grid[valid].astype(np.float64)
    U_v = U[valid].astype(np.float64)

    cols = [np.ones(len(phi_v))]
    for n in range(1, n_fourier + 1):
        cols.append(np.cos(n * phi_v))
        cols.append(np.sin(n * phi_v))
    X = np.column_stack(cols)

    coeffs, _, _, _ = np.linalg.lstsq(X, U_v, rcond=None)
    return coeffs.astype(np.float32)


def fit_1dfunc_priors(all_priors: dict, n_fourier: int = 5) -> dict:
    """Fit analytical bonded functions to Boltzmann-inverted PMFs.

    * **Bonds** and **angles**: harmonic U(x) = ½ k (x − x₀)², parameters
      from non-linear least squares on the BI PMF.
    * **Dihedrals**: truncated Fourier series fitted by linear least squares.

    Works with both species-grouped priors (keys are species tuples, entries
    contain an ``'indices'`` list) and legacy atom-index-keyed priors.
    The ``'indices'`` list is forwarded to the fitted dict so that
    :func:`get_1dfunc_prior_energy_fn_template` can expand each species type
    to all its contributing atom pairs.

    The returned dict is picklable and can be passed directly to
    :func:`get_1dfunc_prior_energy_fn_template`.

    Args:
        all_priors: Output of :meth:`BoltzmannPrior.compute_all_priors`.
        n_fourier:  Number of Fourier harmonics for dihedral terms (default 5).

    Returns:
        Dict with keys ``'bonds'``, ``'angles'``, ``'dihedrals'``,
        ``'n_fourier'``.
    """
    bonds: dict = {}
    for key, d in all_priors.get("bonds", {}).items():
        r0, k = _fit_harmonic_pmf(
            d["r_grid"], d["U"],
            x0_fallback=d.get("r0"), k_fallback=d.get("k"),
        )
        entry: dict = {"r0": r0, "k": k}
        if "indices" in d:
            entry["indices"] = d["indices"]
        bonds[key] = entry

    angles: dict = {}
    for key, d in all_priors.get("angles", {}).items():
        theta0, k = _fit_harmonic_pmf(
            d["theta_grid"], d["U"],
            x0_fallback=d.get("theta0"), k_fallback=d.get("k"),
        )
        entry = {"theta0": theta0, "k": k}
        if "indices" in d:
            entry["indices"] = d["indices"]
        angles[key] = entry

    dihedrals: dict = {}
    for key, d in all_priors.get("dihedrals", {}).items():
        coeffs = _fit_fourier_pmf(d["phi_grid"], d["U"], n_fourier)
        entry = {"coeffs": coeffs}
        if "indices" in d:
            entry["indices"] = d["indices"]
        dihedrals[key] = entry

    return {
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "n_fourier": n_fourier,
    }


def get_1dfunc_prior_energy_fn_template(
    fitted_priors: dict,
    displacement_fn: Callable,
) -> Callable:
    """Build a prior ``energy_fn_template`` from pre-fitted analytical functions.

    Uses harmonic potentials for bonds and angles, and a truncated Fourier
    series for dihedrals.  Parameters must have been obtained via
    :func:`fit_1dfunc_priors`.

    Supports both species-grouped priors (each entry has an ``'indices'`` key
    listing all contributing atom-index tuples) and legacy atom-index-keyed
    priors.  For species-grouped priors the same fitted parameters are applied
    to every atom pair of that species type.

    Args:
        fitted_priors:   Output of :func:`fit_1dfunc_priors`.
        displacement_fn: JAX-MD displacement function handling PBC.

    Returns:
        Callable with signature ``prior_energy_fn_template(params)``.
    """
    n_fourier = int(fitted_priors.get("n_fourier", 5))
    _bond_length, _angle, _dihedral = _make_geometry_fns(displacement_fn)

    # Expand species-grouped entries into per-atom-pair terms.
    bond_terms: list[tuple] = []
    for key, p in fitted_priors.get("bonds", {}).items():
        r0, k = float(p["r0"]), float(p["k"])
        if "indices" in p:
            for (i, j) in p["indices"]:
                bond_terms.append((int(i), int(j), r0, k))
        else:
            i, j = key
            bond_terms.append((int(i), int(j), r0, k))

    angle_terms: list[tuple] = []
    for key, p in fitted_priors.get("angles", {}).items():
        theta0, k = float(p["theta0"]), float(p["k"])
        if "indices" in p:
            for (i, j, k_idx) in p["indices"]:
                angle_terms.append((int(i), int(j), int(k_idx), theta0, k))
        else:
            i, j, k_idx = key
            angle_terms.append((int(i), int(j), int(k_idx), theta0, k))

    dihedral_terms: list[tuple] = []
    for key, p in fitted_priors.get("dihedrals", {}).items():
        coeffs = jnp.asarray(p["coeffs"], dtype=jnp.float32)
        if "indices" in p:
            for (i, j, k_idx, l) in p["indices"]:
                dihedral_terms.append((int(i), int(j), int(k_idx), int(l), coeffs))
        else:
            i, j, k_idx, l = key
            dihedral_terms.append((int(i), int(j), int(k_idx), int(l), coeffs))

    def _eval_fourier(phi: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
        # Unrolled at JAX trace time — n_fourier is static.
        total = coeffs[0]
        for n in range(1, n_fourier + 1):
            total = total + coeffs[2 * n - 1] * jnp.cos(n * phi) + coeffs[2 * n] * jnp.sin(n * phi)
        return total

    def prior_energy_fn_template(params):
        del params

        def prior_energy_fn(position, neighbor, **kwargs):
            del neighbor
            total = jnp.zeros((), dtype=jnp.float32)

            for (i, j, r0, k) in bond_terms:
                r = _bond_length(position, i, j)
                total = total + jnp.float32(0.5 * k) * (r - jnp.float32(r0)) ** 2

            for (i, j, k_idx, theta0, k) in angle_terms:
                theta = _angle(position, i, j, k_idx)
                total = total + jnp.float32(0.5 * k) * (theta - jnp.float32(theta0)) ** 2

            for (i, j, k_idx, l, coeffs) in dihedral_terms:
                phi = _dihedral(position, i, j, k_idx, l)
                total = total + _eval_fourier(phi, coeffs)

            return total.astype(jnp.float32)

        return prior_energy_fn

    return prior_energy_fn_template


def get_prior_energy_fn_template(
    all_priors: dict,
    displacement_fn: Callable,
) -> Callable:
    """Build a prior ``energy_fn_template`` using tabulated BI PMFs.

    Evaluates bond, angle, and dihedral energies by linear interpolation of
    the Boltzmann-inversion PMF grids at runtime.  NaN bins are replaced with
    the maximum finite PMF value before interpolation so that unsampled
    regions act as a soft wall.

    Supports both species-grouped priors (each entry has an ``'indices'`` key)
    and legacy atom-index-keyed priors.

    Args:
        all_priors:      Output of :meth:`BoltzmannPrior.compute_all_priors`.
        displacement_fn: JAX-MD displacement function handling PBC.

    Returns:
        Callable with signature ``prior_energy_fn_template(params)``.
    """
    _bond_length, _angle, _dihedral = _make_geometry_fns(displacement_fn)

    # Pre-process grids: replace NaN with nanmax and convert to JAX arrays.
    def _clean(U: np.ndarray) -> jnp.ndarray:
        return jnp.asarray(_fill_nan_u(U), dtype=jnp.float32)

    # Expand species-grouped entries into per-atom-pair terms, each carrying
    # its own (frozen) grid and PMF arrays.
    bond_terms: list[tuple] = []
    for key, d in all_priors.get("bonds", {}).items():
        r_grid = jnp.asarray(d["r_grid"], dtype=jnp.float32)
        U = _clean(d["U"])
        if "indices" in d:
            for (i, j) in d["indices"]:
                bond_terms.append((int(i), int(j), r_grid, U))
        else:
            i, j = key
            bond_terms.append((int(i), int(j), r_grid, U))

    angle_terms: list[tuple] = []
    for key, d in all_priors.get("angles", {}).items():
        theta_grid = jnp.asarray(d["theta_grid"], dtype=jnp.float32)
        U = _clean(d["U"])
        if "indices" in d:
            for (i, j, k_idx) in d["indices"]:
                angle_terms.append((int(i), int(j), int(k_idx), theta_grid, U))
        else:
            i, j, k_idx = key
            angle_terms.append((int(i), int(j), int(k_idx), theta_grid, U))

    dihedral_terms: list[tuple] = []
    for key, d in all_priors.get("dihedrals", {}).items():
        phi_grid = jnp.asarray(d["phi_grid"], dtype=jnp.float32)
        U = _clean(d["U"])
        if "indices" in d:
            for (i, j, k_idx, l) in d["indices"]:
                dihedral_terms.append((int(i), int(j), int(k_idx), int(l), phi_grid, U))
        else:
            i, j, k_idx, l = key
            dihedral_terms.append((int(i), int(j), int(k_idx), int(l), phi_grid, U))

    def prior_energy_fn_template(params):
        del params

        def prior_energy_fn(position, neighbor, **kwargs):
            del neighbor
            total = jnp.zeros((), dtype=jnp.float32)

            for (i, j, r_grid, U) in bond_terms:
                r = _bond_length(position, i, j)
                total = total + jnp.interp(r, r_grid, U)

            for (i, j, k_idx, theta_grid, U) in angle_terms:
                theta = _angle(position, i, j, k_idx)
                total = total + jnp.interp(theta, theta_grid, U)

            for (i, j, k_idx, l, phi_grid, U) in dihedral_terms:
                phi = _dihedral(position, i, j, k_idx, l)
                total = total + jnp.interp(phi, phi_grid, U)

            return total.astype(jnp.float32)

        return prior_energy_fn

    return prior_energy_fn_template


# ---------------------------------------------------------------------------
# Smooth cutoff helper (shared by SplineModel)
# ---------------------------------------------------------------------------

def _smooth_cutoff(
    r: jnp.ndarray, r_onset: float, r_cutoff: float
) -> jnp.ndarray:
    """5th-order polynomial switch: 1 for r ≤ r_onset, 0 for r ≥ r_cutoff."""
    t = jnp.clip((r - r_onset) / (r_cutoff - r_onset + 1e-12), 0.0, 1.0)
    poly = jnp.float32(1.0) - jnp.float32(6.0) * t**5 + jnp.float32(15.0) * t**4 - jnp.float32(10.0) * t**3
    return jnp.where(r >= jnp.float32(r_cutoff), jnp.float32(0.0), poly)


# ---------------------------------------------------------------------------
# SplineModel: standalone CG model with learnable cubic splines
# ---------------------------------------------------------------------------

class SplineModel:
    """Standalone CG force-matched model using learnable cubic splines.

    Analogous to VOTCA csg_fmatch: all interactions (bonds, angles, dihedrals,
    and non-bonded pairs) are represented as piecewise cubic splines with
    uniformly-spaced knots.  The energy values at the knot positions are the
    trainable parameters, optimised by force matching via the standard
    :class:`~chemtrain.trainers.ForceMatching` trainer.

    Uses :class:`~jax_md_mod.custom_interpolate.MonotonicInterpolate` for
    JAX-native differentiable cubic spline evaluation.

    Args:
        dataset:           :class:`~cgbench.core.dataset.BaseDataset` instance
                           with CG topology and trajectory already available
                           (``coarse_grain()`` must have been called first).
        rcut:              Non-bonded hard cutoff in nm.
        n_knots_nb:        Number of knots for non-bonded splines (≥ 4).
        n_knots_bond:      Number of knots for bond splines (≥ 4).
        n_knots_angle:     Number of knots for angle splines (≥ 4).
        n_knots_dihedral:  Number of knots for dihedral splines (≥ 4).
        T:                 Temperature in K (reserved for future use).
        split:             Dataset split used for knot-range estimation.
        r_onset:           Smooth-cutoff onset in nm.  Defaults to 0.9·rcut.
    """

    _MIN_KNOTS: int = 4  # MonotonicInterpolate requires len(x) > 3

    def __init__(
        self,
        dataset,
        rcut: float,
        n_knots_nb: int = 20,
        n_knots_bond: int = 20,
        n_knots_angle: int = 20,
        n_knots_dihedral: int = 20,
        T: float = 300.0,
        split: str = "training",
        r_onset: float | None = None,
    ):
        self.rcut = float(rcut)
        self.r_onset = float(r_onset if r_onset is not None else 0.9 * rcut)
        self.n_knots_nb = max(n_knots_nb, self._MIN_KNOTS)
        self.n_knots_bond = max(n_knots_bond, self._MIN_KNOTS)
        self.n_knots_angle = max(n_knots_angle, self._MIN_KNOTS)
        self.n_knots_dihedral = max(n_knots_dihedral, self._MIN_KNOTS)

        # ---- topology -------------------------------------------------------
        cg_species = np.asarray(dataset.cg_species)  # (N,) int
        n_particles = int(len(cg_species))

        # Normalise topology arrays to row-major (N, n_cols) using the same
        # helper as BoltzmannPrior, which handles both (n_cols, N) and (N, n_cols).
        bond_index = BoltzmannPrior._as_rows(
            getattr(dataset, "cg_bond_index", None), 2
        )
        angle_index = BoltzmannPrior._as_rows(
            getattr(dataset, "cg_angle_index", None), 3
        )
        dihedral_index = BoltzmannPrior._as_rows(
            getattr(dataset, "cg_dihedral_index", None), 4
        )

        bond_index = bond_index if bond_index is not None else np.empty((0, 2), dtype=int)
        angle_index = angle_index if angle_index is not None else np.empty((0, 3), dtype=int)
        dihedral_index = dihedral_index if dihedral_index is not None else np.empty((0, 4), dtype=int)

        displacement_fn = dataset.displacement_fn_X

        # ---- training data for range estimation ----------------------------
        avail = list(dataset.cg_dataset_X.keys())
        use_split = split if split in avail else avail[0]
        R_split = jnp.asarray(dataset.cg_dataset_X[use_split]["R"])  # (F, N, 3)

        # ---- bond terms ----------------------------------------------------
        bond_type_map: dict = {}
        bond_terms: list = []  # (i, j, type_id)

        if len(bond_index) > 0:
            bl_all = _bond_lengths_pbc(R_split, jnp.asarray(bond_index), displacement_fn)
            # bl_all: (F, n_bonds)

            # First pass: assign type ids
            bond_type_assign = []
            for i, j in bond_index:
                key = _canonical_bond_key(int(cg_species[i]), int(cg_species[j]))
                if key not in bond_type_map:
                    bond_type_map[key] = len(bond_type_map)
                bond_type_assign.append(bond_type_map[key])

            n_bond_types = len(bond_type_map)
            # Pool lengths per type and build grids
            bond_x_grids_np = np.zeros((n_bond_types, self.n_knots_bond))
            type_bond_col_indices: list = [[] for _ in range(n_bond_types)]
            for bi, tid in enumerate(bond_type_assign):
                type_bond_col_indices[tid].append(bi)

            for tid, cols in enumerate(type_bond_col_indices):
                bl_type = bl_all[:, cols].ravel()
                r_lo = float(np.percentile(bl_type, 1))
                r_hi = float(np.percentile(bl_type, 99))
                bond_x_grids_np[tid] = np.linspace(r_lo, r_hi, self.n_knots_bond)

            for bi, (i, j) in enumerate(bond_index):
                bond_terms.append((int(i), int(j), bond_type_assign[bi]))
        else:
            n_bond_types = 0
            bond_x_grids_np = np.empty((0, self.n_knots_bond))

        # ---- angle terms ---------------------------------------------------
        angle_type_map: dict = {}
        angle_terms: list = []  # (i, j, k, type_id)

        if len(angle_index) > 0:
            ang_all = _angles_pbc(R_split, jnp.asarray(angle_index), displacement_fn)
            # ang_all: (F, n_angles)

            angle_type_assign = []
            for i, j, k in angle_index:
                key = _canonical_angle_key(int(cg_species[i]), int(cg_species[j]), int(cg_species[k]))
                if key not in angle_type_map:
                    angle_type_map[key] = len(angle_type_map)
                angle_type_assign.append(angle_type_map[key])

            n_angle_types = len(angle_type_map)
            angle_x_grids_np = np.zeros((n_angle_types, self.n_knots_angle))
            type_angle_col_indices: list = [[] for _ in range(n_angle_types)]
            for ai, tid in enumerate(angle_type_assign):
                type_angle_col_indices[tid].append(ai)

            for tid, cols in enumerate(type_angle_col_indices):
                ang_type = ang_all[:, cols].ravel()
                a_lo = max(0.0, float(np.percentile(ang_type, 1)))
                a_hi = min(float(np.pi), float(np.percentile(ang_type, 99)))
                angle_x_grids_np[tid] = np.linspace(a_lo, a_hi, self.n_knots_angle)

            for ai, (i, j, k) in enumerate(angle_index):
                angle_terms.append((int(i), int(j), int(k), angle_type_assign[ai]))
        else:
            n_angle_types = 0
            angle_x_grids_np = np.empty((0, self.n_knots_angle))

        # ---- dihedral terms ------------------------------------------------
        dihedral_type_map: dict = {}
        dihedral_terms: list = []  # (i, j, k, l, type_id)

        if len(dihedral_index) > 0:
            for i, j, k, l in dihedral_index:
                key = _canonical_dihedral_key(
                    int(cg_species[i]), int(cg_species[j]),
                    int(cg_species[k]), int(cg_species[l]),
                )
                if key not in dihedral_type_map:
                    dihedral_type_map[key] = len(dihedral_type_map)

            n_dihedral_types = len(dihedral_type_map)
            dihedral_x_grids_np = np.zeros((n_dihedral_types, self.n_knots_dihedral))
            for tid in range(n_dihedral_types):
                dihedral_x_grids_np[tid] = np.linspace(-np.pi, np.pi, self.n_knots_dihedral)

            for i, j, k, l in dihedral_index:
                key = _canonical_dihedral_key(
                    int(cg_species[i]), int(cg_species[j]),
                    int(cg_species[k]), int(cg_species[l]),
                )
                dihedral_terms.append((int(i), int(j), int(k), int(l), dihedral_type_map[key]))
        else:
            n_dihedral_types = 0
            dihedral_x_grids_np = np.empty((0, self.n_knots_dihedral))

        # ---- non-bonded types (all unique canonical species pairs) ---------
        unique_species = sorted(set(int(s) for s in cg_species))
        nb_species_pairs: list = []
        nb_x_grids_list: list = []

        for idx_si, si in enumerate(unique_species):
            for sj in unique_species[idx_si:]:  # canonical: si <= sj
                nb_species_pairs.append((si, sj))
                nb_x_grids_list.append(np.linspace(0.1, self.rcut, self.n_knots_nb))

        n_nb_types = len(nb_species_pairs)
        nb_x_grids_np = (np.stack(nb_x_grids_list) if n_nb_types > 0
                         else np.empty((0, self.n_knots_nb)))

        # ---- store ----------------------------------------------------------
        self._cg_species = cg_species
        self._n_particles = n_particles
        self._bond_terms = bond_terms
        self._angle_terms = angle_terms
        self._dihedral_terms = dihedral_terms
        self._nb_species_pairs = nb_species_pairs
        self._bond_x_grids = bond_x_grids_np
        self._angle_x_grids = angle_x_grids_np
        self._dihedral_x_grids = dihedral_x_grids_np
        self._nb_x_grids = nb_x_grids_np
        self._n_bond_types = n_bond_types
        self._n_angle_types = n_angle_types
        self._n_dihedral_types = n_dihedral_types
        self._n_nb_types = n_nb_types

        n_b = len(bond_type_map)
        n_a = len(angle_type_map)
        n_d = len(dihedral_type_map)
        print(
            f"[SplineModel] {n_b} bond type(s), {n_a} angle type(s), "
            f"{n_d} dihedral type(s), {n_nb_types} non-bonded type(s). "
            f"rcut={self.rcut} nm, r_onset={self.r_onset:.3f} nm."
        )

    # -------------------------------------------------------------------------
    # Serialisation
    # -------------------------------------------------------------------------

    def save_data(self, path: str) -> None:
        """Pickle the fixed topology/grid data needed to reconstruct this model.

        Args:
            path: File path for the output pickle (``spline_model.pkl``).
        """
        import cloudpickle
        data = {
            "bond_terms": self._bond_terms,
            "angle_terms": self._angle_terms,
            "dihedral_terms": self._dihedral_terms,
            "bond_x_grids": self._bond_x_grids,
            "angle_x_grids": self._angle_x_grids,
            "dihedral_x_grids": self._dihedral_x_grids,
            "nb_x_grids": self._nb_x_grids,
            "nb_species_pairs": self._nb_species_pairs,
            "species": self._cg_species,
            "n_particles": self._n_particles,
            "rcut": self.rcut,
            "r_onset": self.r_onset,
            "n_knots_nb": self.n_knots_nb,
            "n_knots_bond": self.n_knots_bond,
            "n_knots_angle": self.n_knots_angle,
            "n_knots_dihedral": self.n_knots_dihedral,
        }
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)

    @classmethod
    def from_data(cls, data: dict) -> "SplineModel":
        """Reconstruct a SplineModel from a dict produced by :meth:`save_data`.

        Does not require a dataset or trajectory; all topology/grid information
        is read directly from *data*.

        Args:
            data: Dict as written by :meth:`save_data`.

        Returns:
            Fully initialised :class:`SplineModel` ready for
            :meth:`get_energy_fn_template`.
        """
        obj = cls.__new__(cls)
        obj.rcut = float(data["rcut"])
        obj.r_onset = float(data["r_onset"])
        obj.n_knots_nb = int(data["n_knots_nb"])
        obj.n_knots_bond = int(data["n_knots_bond"])
        obj.n_knots_angle = int(data["n_knots_angle"])
        obj.n_knots_dihedral = int(data["n_knots_dihedral"])
        obj._cg_species = np.asarray(data["species"])
        obj._n_particles = int(data["n_particles"])
        obj._bond_terms = list(data["bond_terms"])
        obj._angle_terms = list(data["angle_terms"])
        obj._dihedral_terms = list(data["dihedral_terms"])
        obj._nb_species_pairs = list(data["nb_species_pairs"])
        obj._bond_x_grids = np.asarray(data["bond_x_grids"])
        obj._angle_x_grids = np.asarray(data["angle_x_grids"])
        obj._dihedral_x_grids = np.asarray(data["dihedral_x_grids"])
        obj._nb_x_grids = np.asarray(data["nb_x_grids"])
        obj._n_bond_types = int(obj._bond_x_grids.shape[0])
        obj._n_angle_types = int(obj._angle_x_grids.shape[0])
        obj._n_dihedral_types = int(obj._dihedral_x_grids.shape[0])
        obj._n_nb_types = len(obj._nb_species_pairs)
        return obj

    # -------------------------------------------------------------------------
    # Parameter initialisation
    # -------------------------------------------------------------------------

    def init_params(self) -> dict:
        """Return zero-initialised trainable parameter pytree.

        The structure mirrors what :meth:`get_energy_fn_template` expects:

        .. code-block:: python

            {
                "bonds":      jnp.zeros((n_bond_types,     n_knots_bond)),
                "angles":     jnp.zeros((n_angle_types,    n_knots_angle)),
                "dihedrals":  jnp.zeros((n_dihedral_types, n_knots_dihedral)),
                "non_bonded": jnp.zeros((n_nb_types,       n_knots_nb)),
            }

        Returns:
            JAX pytree (dict of float32 arrays).
        """
        return {
            "bonds": jnp.zeros((self._n_bond_types, self.n_knots_bond), dtype=jnp.float32),
            "angles": jnp.zeros((self._n_angle_types, self.n_knots_angle), dtype=jnp.float32),
            "dihedrals": jnp.zeros((self._n_dihedral_types, self.n_knots_dihedral), dtype=jnp.float32),
            "non_bonded": jnp.zeros((self._n_nb_types, self.n_knots_nb), dtype=jnp.float32),
        }

    # -------------------------------------------------------------------------
    # Energy function template
    # -------------------------------------------------------------------------

    def get_energy_fn_template(self, displacement_fn: Callable) -> Callable:
        """Build the energy function template for this spline model.

        The returned template has the standard cgbench signature::

            spline_energy_fn_template(params) -> energy_fn(position, neighbor, **kwargs) -> float32

        where *params* is a pytree as returned by :meth:`init_params` (or
        equivalently, the trained parameters loaded from a checkpoint).

        Bonded terms (bonds, angles, dihedrals) are evaluated via a Python
        loop over interaction instances, which JAX unrolls at trace time —
        identical to the existing :func:`get_prior_energy_fn_template`.

        Non-bonded terms use the sparse neighbor list (``neighbor.idx`` of
        shape ``(n_particles, max_neighbors)`` with ``n_particles`` as the
        invalid-entry sentinel).

        Args:
            displacement_fn: JAX-MD displacement function (handles PBC).

        Returns:
            ``spline_energy_fn_template`` callable.
        """
        _bond_length, _angle, _dihedral = _make_geometry_fns(displacement_fn)

        # Close over fixed data as JAX arrays
        bond_x_grids_jax = [jnp.asarray(self._bond_x_grids[tid], dtype=jnp.float32)
                            for tid in range(self._n_bond_types)]
        angle_x_grids_jax = [jnp.asarray(self._angle_x_grids[tid], dtype=jnp.float32)
                             for tid in range(self._n_angle_types)]
        dihedral_x_grids_jax = [jnp.asarray(self._dihedral_x_grids[tid], dtype=jnp.float32)
                                for tid in range(self._n_dihedral_types)]
        nb_x_grids_jax = [jnp.asarray(self._nb_x_grids[tid], dtype=jnp.float32)
                          for tid in range(self._n_nb_types)]

        bond_terms = self._bond_terms        # list[(i, j, type_id)]
        angle_terms = self._angle_terms      # list[(i, j, k, type_id)]
        dihedral_terms = self._dihedral_terms  # list[(i, j, k, l, type_id)]
        nb_species_pairs = self._nb_species_pairs  # list[(si, sj)]

        species_jax = jnp.asarray(self._cg_species)
        n_particles = self._n_particles
        rcut = jnp.float32(self.rcut)
        r_onset = jnp.float32(self.r_onset)

        def spline_energy_fn_template(params: dict) -> Callable:
            bond_params = params["bonds"]       # (n_bond_types, n_knots_bond)
            angle_params = params["angles"]     # (n_angle_types, n_knots_angle)
            dihedral_params = params["dihedrals"]  # (n_dihedral_types, n_knots_dihedral)
            nb_params = params["non_bonded"]    # (n_nb_types, n_knots_nb)

            def energy_fn(position: jnp.ndarray, neighbor, **kwargs) -> jnp.ndarray:
                total = jnp.zeros((), dtype=jnp.float32)

                # ---- bonded: bonds -----------------------------------------
                for (i, j, tid) in bond_terms:
                    spline = MonotonicInterpolate(bond_x_grids_jax[tid], bond_params[tid])
                    r = _bond_length(position, i, j)
                    total = total + spline(r).astype(jnp.float32)

                # ---- bonded: angles ----------------------------------------
                for (i, j, k, tid) in angle_terms:
                    spline = MonotonicInterpolate(angle_x_grids_jax[tid], angle_params[tid])
                    theta = _angle(position, i, j, k)
                    total = total + spline(theta).astype(jnp.float32)

                # ---- bonded: dihedrals -------------------------------------
                for (i, j, k, l, tid) in dihedral_terms:
                    spline = MonotonicInterpolate(dihedral_x_grids_jax[tid], dihedral_params[tid])
                    phi = _dihedral(position, i, j, k, l)
                    total = total + spline(phi).astype(jnp.float32)

                # ---- non-bonded via sparse neighbor list (COO format) ------
                # neighbor.idx has shape (2, max_edges): row 0 = senders,
                # row 1 = receivers.  Invalid edges are padded with n_particles.
                if nb_species_pairs:
                    idx_i = neighbor.idx[0]  # (max_edges,)
                    idx_j = neighbor.idx[1]  # (max_edges,)
                    valid = idx_i < n_particles  # (max_edges,)
                    safe_idx_i = jnp.where(valid, idx_i, 0)
                    safe_idx_j = jnp.where(valid, idx_j, 0)

                    # Pairwise distances for every edge: (max_edges,)
                    xi = position[safe_idx_i]  # (max_edges, 3)
                    xj = position[safe_idx_j]  # (max_edges, 3)
                    dr = jax.vmap(displacement_fn)(xi, xj)  # (max_edges, 3)
                    # Use double-where to avoid NaN gradient of norm at r=0 for
                    # padding edges (safe_idx_i == safe_idx_j == 0 → dr == 0).
                    # Invalid edges get r=1.0 (masked out later by pair_mask).
                    r_sq = jnp.sum(dr**2, axis=-1)
                    r_all = jnp.sqrt(
                        jnp.where(valid, r_sq, jnp.float32(1.0))
                    ).astype(jnp.float32)  # (max_edges,)
                    cutoff_weights = _smooth_cutoff(r_all, r_onset, rcut)

                    species_i = species_jax[safe_idx_i]  # (max_edges,)
                    species_j = species_jax[safe_idx_j]  # (max_edges,)

                    for type_id, (si, sj) in enumerate(nb_species_pairs):
                        spline = MonotonicInterpolate(nb_x_grids_jax[type_id], nb_params[type_id])
                        u_ij = spline(r_all).astype(jnp.float32) * cutoff_weights

                        # Match both orderings; each physical pair (i,j) appears
                        # twice in the symmetric NL, so multiply by 0.5.
                        pair_mask = valid & (
                            ((species_i == si) & (species_j == sj)) |
                            ((species_i == sj) & (species_j == si))
                        )
                        total = total + jnp.sum(
                            jnp.where(pair_mask, u_ij, jnp.float32(0.0))
                        ) * jnp.float32(0.5)

                return total

            return energy_fn

        return spline_energy_fn_template
