"""Tests for the reservoir-state probe (written before wiring; thresholds grounded in a
measured exploratory run: reservoir-state R²≈0.995, stateless R²≈0.16, resilience 0.99→0.82
across input drift 0→0.8).

Checks the falsifiable claims from the Grok chat: a temporal process property is linearly
decodable from reservoir state (no SAE) while a stateless monitor cannot read it, and the
pre-drift probe stays usable under a fine-tuning-like input drift (graceful, not invariant).
"""
import numpy as np
import pytest

from reservoir.probe import (
    fit_ridge,
    predict_ridge,
    r2_score,
    run_probe_experiment,
)


def test_r2_score_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    # predicting the mean everywhere gives R² == 0
    assert r2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    # constant truth -> SS_tot 0 -> defined as 0.0
    assert r2_score(np.ones(4), np.array([1.0, 2.0, 0.0, 1.0])) == 0.0


def test_ridge_recovers_known_linear_map():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((500, 5))
    w_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w_true + 0.7
    w = fit_ridge(X, y, alpha=1e-6)
    assert np.allclose(w[:5], w_true, atol=1e-2)
    assert w[-1] == pytest.approx(0.7, abs=1e-2)        # the bias term
    assert r2_score(y, predict_ridge(X, w)) > 0.999


def test_elapsed_clock_is_linearly_decodable_from_reservoir_only():
    out = run_probe_experiment(seed=0)
    # the reservoir state carries the elapsed-since-trigger clock; a linear probe reads it.
    assert out["r2_state"] > 0.9
    # the instantaneous input does not contain elapsed time, so a stateless probe fails.
    assert out["r2_input"] < 0.3
    assert out["r2_state"] - out["r2_input"] > 0.5


def test_probe_resilient_to_moderate_drift_then_degrades():
    out = run_probe_experiment(seed=0)
    curve = {round(c["drift"], 3): c["r2_state"] for c in out["resilience_curve"]}
    assert curve[0.0] == pytest.approx(out["r2_state"])      # zero drift == clean R²
    assert curve[0.2] > 0.85                                 # resilient up to moderate drift
    assert curve[0.8] < curve[0.0]                           # large drift does degrade it
    # even under the largest drift the reservoir probe stays well above the stateless baseline.
    assert curve[0.8] > out["r2_input"]


def test_resilience_curve_is_monotone_non_increasing():
    out = run_probe_experiment(seed=0)
    ys = [c["r2_state"] for c in out["resilience_curve"]]
    assert all(b <= a + 1e-6 for a, b in zip(ys, ys[1:]))
