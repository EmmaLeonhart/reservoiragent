"""Tests for the H3 delay-memory task + trained readout (all numpy, runs in CI)."""
import numpy as np

from reservoir.tasks import fit_ridge, r2_score, delay_memory_curve, memory_capacity


def test_fit_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 5))
    W_true = rng.standard_normal((5, 2))
    Y = X @ W_true
    W = fit_ridge(X, Y, alpha=1e-8)
    assert np.allclose(W, W_true, atol=1e-3)


def test_r2_score_bounds():
    Y = np.linspace(0, 1, 50)
    assert r2_score(Y, Y) == 1.0                 # perfect
    assert abs(r2_score(Y, np.full_like(Y, Y.mean()))) < 1e-6   # mean predictor -> 0


def test_reservoir_has_memory_the_baseline_lacks():
    recs = delay_memory_curve(K=150, delays=range(0, 12), T=2500, seed=0)
    by_delay = {r["delay"]: r for r in recs}
    # tau=0: both recover the current input trivially
    assert by_delay[0]["reservoir_r2"] > 0.9
    # tau>=1: the reservoir reconstructs delayed input; the stateless baseline cannot
    assert by_delay[1]["reservoir_r2"] > 0.5
    assert by_delay[2]["baseline_r2"] < 0.1
    assert by_delay[5]["baseline_r2"] < 0.1
    # total memory capacity is substantial for the reservoir, ~0 for the baseline
    assert memory_capacity(recs, "reservoir_r2") > 2.0
    assert memory_capacity(recs, "baseline_r2") < 0.2
