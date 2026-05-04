
"""Model wrappers exposed by the minimal vendored chemutils package."""

# Optional monkeypatch for older e3nn_jax behavior used by some chemutils paths.
# Keep it best-effort so importing model wrappers does not fail when optional
# dependencies from the full chemutils stack are not installed.
try:  # pragma: no cover - defensive import guard
	from jax_md_mod import uncache
	import e3nn_jax._src.scatter
	from . import e3nn_mod

	e3nn_jax._src.scatter._distinct_but_small = e3nn_mod._distinct_but_small
	uncache("e3nn_jax._src.scatter")
except Exception:
	pass
