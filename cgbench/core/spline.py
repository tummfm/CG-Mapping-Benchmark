"""One-shot linear force matching for CG spline potentials (OpenMSCG approach).

The spline potential is linear in its knot-value parameters::

    U(R; c) = c · φ(R)

so forces are also linear in c::

    F_iα(R; c) = Σ_k c_k · X_iα,k(R)

The force-matching objective min_c ||Xc − y||² is a convex quadratic with
a unique global minimum found by solving the normal equations X^T X c = X^T y
in one pass through the data — no iterative optimisation needed.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


class SplineForceMatcher:
    """One-shot force matching for a :class:`~cgbench.core.prior.SplineModel`.

    Accumulates the normal equations X^T X and X^T y frame-by-frame, then
    solves once with :func:`numpy.linalg.lstsq`.  The force loading matrix
    X (shape ``n_atoms*3 × n_params``) is computed per frame via
    ``jax.jacfwd`` over the flat parameter vector.  Since the spline energy
    is linear in its parameters, this Jacobian is independent of the current
    parameter values; evaluating at ``c = 0`` is exact.

    Args:
        spline_model:    Initialised :class:`~cgbench.core.prior.SplineModel`.
        displacement_fn: JAX-MD displacement function.
        ridge_alpha:     L2 regularisation added to the diagonal of X^T X
                         before solving (0 = no regularisation).
    """

    def __init__(
        self,
        spline_model,
        displacement_fn: Callable,
        ridge_alpha: float = 0.0,
    ) -> None:
        self.model = spline_model
        self.displacement_fn = displacement_fn
        self.ridge_alpha = float(ridge_alpha)

        self._setup_param_layout()
        self._build_loading_fn()

        n = self.n_params
        self.XtX = np.zeros((n, n), dtype=np.float64)
        self.XtY = np.zeros(n, dtype=np.float64)
        self.y_sumsq = 0.0
        self.n_frames = 0

    # ------------------------------------------------------------------
    # Parameter layout
    # ------------------------------------------------------------------

    def _setup_param_layout(self) -> None:
        m = self.model
        self._bond_offset = 0
        self._bond_size = m._n_bond_types * m.n_knots_bond

        self._angle_offset = self._bond_offset + self._bond_size
        self._angle_size = m._n_angle_types * m.n_knots_angle

        self._dihedral_offset = self._angle_offset + self._angle_size
        self._dihedral_size = m._n_dihedral_types * m.n_knots_dihedral

        self._nb_offset = self._dihedral_offset + self._dihedral_size
        self._nb_size = m._n_nb_types * m.n_knots_nb

        self.n_params = self._nb_offset + self._nb_size

    # ------------------------------------------------------------------
    # JAX loading function
    # ------------------------------------------------------------------

    def _build_loading_fn(self) -> None:
        """Compile the JIT-ted JAX function that returns the force loading matrix."""
        m = self.model
        energy_template = m.get_energy_fn_template(self.displacement_fn)

        bond_offset = self._bond_offset
        bond_size = self._bond_size
        angle_offset = self._angle_offset
        angle_size = self._angle_size
        dih_offset = self._dihedral_offset
        dih_size = self._dihedral_size
        nb_offset = self._nb_offset
        nb_size = self._nb_size

        n_bond_types = m._n_bond_types
        n_angle_types = m._n_angle_types
        n_dih_types = m._n_dihedral_types
        n_nb_types = m._n_nb_types
        n_knots_bond = m.n_knots_bond
        n_knots_angle = m.n_knots_angle
        n_knots_dih = m.n_knots_dihedral
        n_knots_nb = m.n_knots_nb
        n_params = self.n_params
        n_atoms = m._n_particles

        # Evaluate at c=0; exact because energy is linear in c.
        c_zero = jnp.zeros(n_params, dtype=jnp.float32)

        def flat_to_params(c_flat):
            return {
                "bonds": c_flat[bond_offset:bond_offset + bond_size].reshape(
                    n_bond_types, n_knots_bond),
                "angles": c_flat[angle_offset:angle_offset + angle_size].reshape(
                    n_angle_types, n_knots_angle),
                "dihedrals": c_flat[dih_offset:dih_offset + dih_size].reshape(
                    n_dih_types, n_knots_dih),
                "non_bonded": c_flat[nb_offset:nb_offset + nb_size].reshape(
                    n_nb_types, n_knots_nb),
            }

        def energy_flat(c_flat, R, nbrs, **kwargs):
            return energy_template(flat_to_params(c_flat))(R, nbrs, **kwargs)

        def force_fn(c_flat, R, nbrs, **kwargs):
            return -jax.grad(energy_flat, argnums=1)(c_flat, R, nbrs, **kwargs)

        n_dof = n_atoms * 3
        _use_jacrev = n_dof < n_params
        _jac_fn = jax.jacrev if _use_jacrev else jax.jacfwd
        print(
            f"[SplineLSQ] AD direction: {'jacrev' if _use_jacrev else 'jacfwd'} "
            f"(n_dof={n_dof}, n_params={n_params})"
        )

        def compute_loading(R, nbrs, **kwargs):
            jac = _jac_fn(force_fn, argnums=0)(c_zero, R, nbrs, **kwargs)
            return jac.reshape(n_dof, n_params)

        self._loading_fn = jax.jit(compute_loading)

        # Batched accumulation: vmap compute_loading over B frames, accumulate on GPU.
        def _accum_batch(R_batch, nbrs_batch, y_batch, mask_batch, species):
            def _one(R, nbrs, y, mask):
                X = compute_loading(R, nbrs, species=species, mask=mask)  # (n_dof, n_params)
                atom_mask = jnp.repeat(mask, 3).astype(X.dtype)
                X = X * atom_mask[:, None]
                y = y * atom_mask
                return X.T @ X, X.T @ y, jnp.dot(y, y)

            XtX_b, XtY_b, yy_b = jax.vmap(_one)(R_batch, nbrs_batch, y_batch, mask_batch)
            return XtX_b.sum(0), XtY_b.sum(0), yy_b.sum()

        self._accum_batch_fn = jax.jit(_accum_batch)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def accumulate(
        self,
        R: jnp.ndarray,
        nbrs,
        F_ref: np.ndarray,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """Process one frame and update the normal equations.

        Args:
            R:      CG positions, shape ``(n_atoms, 3)`` [nm].
            nbrs:   Neighbor list updated for this frame.
            F_ref:  Reference CG forces, shape ``(n_atoms, 3)`` [kJ/(mol·nm)].
            mask:   Boolean validity mask, shape ``(n_atoms,)``.  Padding atoms
                    (False entries) are excluded from both sides of the equations.
            **kwargs: Forwarded to the energy function (e.g. ``species``, ``mask``).
        """
        if mask is not None:
            kwargs.setdefault("mask", jnp.asarray(mask))

        X = np.asarray(self._loading_fn(R, nbrs, **kwargs), dtype=np.float64)
        y = np.asarray(F_ref, dtype=np.float64).reshape(-1)

        if mask is not None:
            atom_mask = np.repeat(np.asarray(mask, dtype=bool), 3)
            X = X[atom_mask]
            y = y[atom_mask]

        self.XtX += X.T @ X
        self.XtY += X.T @ y
        self.y_sumsq += float(np.dot(y, y))
        self.n_frames += 1

    def accumulate_batch(
        self,
        R_batch: jnp.ndarray,
        nbrs_batch,
        F_batch: np.ndarray,
        mask_batch: np.ndarray | None = None,
        species: jnp.ndarray | None = None,
    ) -> None:
        """Process a batch of frames in a single JIT'd call.

        Args:
            R_batch:    Positions, shape ``(B, n_atoms, 3)`` [nm].
            nbrs_batch: Stacked neighbor lists (one per frame, same capacity).
            F_batch:    Reference forces, shape ``(B, n_atoms, 3)``.
            mask_batch: Boolean validity masks, shape ``(B, n_atoms)``.
            species:    Atom types, shape ``(n_atoms,)``.
        """
        B = R_batch.shape[0]
        n_dof = self.model._n_particles * 3
        y_batch = jnp.asarray(F_batch, dtype=jnp.float32).reshape(B, n_dof)

        if mask_batch is None:
            mask_batch = jnp.ones((B, self.model._n_particles), dtype=bool)
        else:
            mask_batch = jnp.asarray(mask_batch)

        XtX_b, XtY_b, yy_b = self._accum_batch_fn(
            R_batch, nbrs_batch, y_batch, mask_batch, species
        )
        self.XtX += np.asarray(XtX_b, dtype=np.float64)
        self.XtY += np.asarray(XtY_b, dtype=np.float64)
        self.y_sumsq += float(yy_b)
        self.n_frames += B

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, alpha: float | None = None) -> tuple[dict, float, int]:
        """Solve the accumulated normal equations.

        Args:
            alpha: Ridge regularisation override.  ``None`` uses
                   ``self.ridge_alpha`` set at construction.

        Returns:
            ``(params_dict, chi2, rank)`` where *params_dict* is compatible
            with :meth:`~cgbench.core.prior.SplineModel.get_energy_fn_template`,
            *chi2* is the mean squared force residual per component per frame,
            and *rank* is the numerical rank of the system.
        """
        if alpha is None:
            alpha = self.ridge_alpha

        A = self.XtX.copy()
        b = self.XtY.copy()
        if alpha > 0.0:
            A += alpha * np.eye(self.n_params)

        c, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

        # chi2 = (||y||² - 2 c·X^T y + c·X^T X c) / (3·N·frames)
        denom = max(1, 3 * self.model._n_particles * self.n_frames)
        chi2 = (self.y_sumsq - 2.0 * (c @ self.XtY) + c @ (self.XtX @ c)) / denom

        return self._flat_to_params(c), float(chi2), int(rank)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flat_to_params(self, c_flat: np.ndarray) -> dict:
        m = self.model
        return {
            "bonds": jnp.asarray(
                c_flat[self._bond_offset:self._bond_offset + self._bond_size].reshape(
                    m._n_bond_types, m.n_knots_bond), dtype=jnp.float32),
            "angles": jnp.asarray(
                c_flat[self._angle_offset:self._angle_offset + self._angle_size].reshape(
                    m._n_angle_types, m.n_knots_angle), dtype=jnp.float32),
            "dihedrals": jnp.asarray(
                c_flat[self._dihedral_offset:self._dihedral_offset + self._dihedral_size].reshape(
                    m._n_dihedral_types, m.n_knots_dihedral), dtype=jnp.float32),
            "non_bonded": jnp.asarray(
                c_flat[self._nb_offset:self._nb_offset + self._nb_size].reshape(
                    m._n_nb_types, m.n_knots_nb), dtype=jnp.float32),
        }
