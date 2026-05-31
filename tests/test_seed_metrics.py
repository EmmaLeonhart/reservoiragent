"""Tests for per-seed reservoir metrics + the correlation analysis (pure numpy/scipy).

These are untrained, structural metrics of a seed's fixed reservoir, computed to hunt for
a predictor of which seeds train to high cross-pass recall. No GPU, no LLM.
"""
import numpy as np
import pytest

from reservoir.seed_metrics import reservoir_metrics, correlate_seed_metrics

# small reservoir keeps the tests instant
SMALL = dict(K=32, d_in=8)


def test_metrics_has_expected_keys():
    m = reservoir_metrics(0, **SMALL)
    for k in ("realized_rho", "mean_abs_eig", "std_abs_eig", "non_normality",
              "participation_ratio", "memory_capacity"):
        assert k in m


def test_realized_rho_matches_construction_target():
    # W is scaled to the target spectral radius, so realized rho ~= 0.9 for every seed
    m = reservoir_metrics(3, spectral_radius=0.9, **SMALL)
    assert abs(m["realized_rho"] - 0.9) < 1e-3


def test_metrics_are_deterministic_in_seed():
    assert reservoir_metrics(7, **SMALL) == reservoir_metrics(7, **SMALL)


def test_non_normality_nonnegative():
    assert reservoir_metrics(1, **SMALL)["non_normality"] >= 0.0


def test_correlate_returns_stats_per_metric():
    seed_recall = {0: 1.0, 1: 0.5, 2: 0.17, 3: 0.83}
    res = correlate_seed_metrics(seed_recall, **SMALL)
    assert "memory_capacity" in res and "non_normality" in res
    for stat in res.values():
        assert set(stat) >= {"spearman", "p_value", "n"}
        assert stat["n"] == 4


def test_correlate_empty_raises():
    with pytest.raises(ValueError):
        correlate_seed_metrics({}, **SMALL)
