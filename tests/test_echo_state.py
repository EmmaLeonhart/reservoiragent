"""Tests for the echo-state reservoir core (written before the implementation).

These pin the properties the dynamics study depends on: correct spectral-radius
scaling, the echo state / contraction property, bounded non-exploding state,
shapes/dtype, seed reproducibility, and the leak-rate semantics.
"""
import numpy as np
import pytest

from reservoir.echo_state import EchoStateReservoir


def test_spectral_radius_scaling_is_accurate():
    for target in (0.3, 0.9, 1.2):
        r = EchoStateReservoir(n_reservoir=120, n_input=4, spectral_radius=target,
                               sparsity=0.2, seed=1)
        assert r.spectral_radius_actual() == pytest.approx(target, rel=1e-6, abs=1e-6)


def test_shapes_and_dtype():
    r = EchoStateReservoir(n_reservoir=64, n_input=3, seed=0)
    s = r.step(np.ones(3))
    assert s.shape == (64,)
    assert s.dtype == np.float64
    traj = r.run(np.random.default_rng(0).standard_normal((10, 3)))
    assert traj.shape == (10, 64)


def test_echo_state_contraction_forgets_initial_condition():
    # rho < 1 with zero input: the null state is a stable fixed point, so two
    # different initial states must converge to (nearly) the same state.
    r = EchoStateReservoir(n_reservoir=100, n_input=2, spectral_radius=0.8,
                           leak_rate=1.0, seed=2)
    zero = np.zeros((300, 2))
    rng = np.random.default_rng(7)
    r.reset(rng.standard_normal(100))
    a = r.run(zero, reset=False)[-1]
    r.reset(rng.standard_normal(100))
    b = r.run(zero, reset=False)[-1]
    assert np.linalg.norm(a - b) < 1e-4
    assert np.linalg.norm(a) < 1e-2  # both decay toward the null state


def test_state_is_bounded_and_finite_even_above_unit_radius():
    r = EchoStateReservoir(n_reservoir=80, n_input=5, spectral_radius=1.5, seed=3)
    rng = np.random.default_rng(0)
    traj = r.run(rng.standard_normal((500, 5)))
    assert np.all(np.isfinite(traj))
    assert np.max(np.abs(traj)) <= 1.0  # tanh keeps state in [-1, 1]


def test_seed_reproducibility():
    kw = dict(n_reservoir=50, n_input=3, spectral_radius=0.9, seed=42)
    r1, r2 = EchoStateReservoir(**kw), EchoStateReservoir(**kw)
    assert np.array_equal(r1.W_r, r2.W_r)
    assert np.array_equal(r1.W_in, r2.W_in)
    x = np.random.default_rng(1).standard_normal((20, 3))
    assert np.allclose(r1.run(x), r2.run(x))
    r3 = EchoStateReservoir(**{**kw, "seed": 43})
    assert not np.array_equal(r1.W_r, r3.W_r)


def test_leak_rate_zero_freezes_state():
    r = EchoStateReservoir(n_reservoir=32, n_input=2, leak_rate=0.0, seed=0)
    r.reset(np.full(32, 0.25))
    out = r.step(np.ones(2))
    assert np.allclose(out, 0.25)  # leak 0 -> state never updates


def test_sparsity_controls_reservoir_density():
    r = EchoStateReservoir(n_reservoir=200, n_input=2, sparsity=0.1, seed=0)
    density = np.mean(r.W_r != 0.0)
    assert 0.05 < density < 0.15  # ~10% nonzero
