"""Smoke tests for the scaffold: the package imports and basic invariants hold.

These are intentionally trivial — their job is to prove the package layout and CI
are wired correctly and green before any real logic lands. Real logic (the reservoir
core, dynamics metrics) gets its own test-driven suites.
"""
import reservoir


def test_package_imports_and_has_version():
    assert isinstance(reservoir.__version__, str)
    # semantic-ish version: at least "X.Y"
    parts = reservoir.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts[:2])


def test_numpy_available():
    # numpy is a core dependency; the reservoir core is built on it.
    import numpy as np

    x = np.tanh(np.zeros(4))
    assert x.shape == (4,)
    assert float(x.sum()) == 0.0
