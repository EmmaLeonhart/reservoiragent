"""Tests for reservoir dynamics metrics (written before the implementation).

Each metric is checked against a synthetic input with a known answer, so the metric
itself is trustworthy before it is used to judge real reservoir dynamics.
"""
import numpy as np
import pytest

from reservoir.metrics import (
    saturation_fraction,
    state_variance,
    participation_ratio,
    trajectory_distinguishability,
)


def test_saturation_fraction_known():
    traj = np.array([[0.0, 1.0], [0.995, -0.999]])  # 3 of 4 entries have |r| > 0.99
    assert saturation_fraction(traj, threshold=0.99) == pytest.approx(0.75)
    assert saturation_fraction(np.zeros((5, 3))) == 0.0
    assert saturation_fraction(np.ones((5, 3))) == 1.0


def test_state_variance_zero_for_constant_and_matches_numpy():
    assert state_variance(np.ones((10, 5))) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    traj = rng.standard_normal((200, 8))
    assert state_variance(traj) == pytest.approx(float(np.mean(np.var(traj, axis=0))))


def test_participation_ratio_rank_one_is_one():
    # every unit carries the same signal -> the trajectory is rank 1 -> PR ~ 1
    signal = np.sin(np.linspace(0, 10, 400))
    traj = np.outer(signal, np.ones(16))
    assert participation_ratio(traj) == pytest.approx(1.0, abs=1e-6)


def test_participation_ratio_full_for_independent_units():
    rng = np.random.default_rng(1)
    K = 20
    traj = rng.standard_normal((4000, K))  # independent equal-variance columns
    pr = participation_ratio(traj)
    assert 0.6 * K < pr <= K + 1e-6


def test_trajectory_distinguishability_known():
    T, K = 50, 9
    a = np.zeros((T, K))
    b = np.ones((T, K))
    # per-step RMS distance: ||0-1|| / sqrt(K) = sqrt(K)/sqrt(K) = 1
    assert trajectory_distinguishability(a, b) == pytest.approx(1.0)
    assert trajectory_distinguishability(a, a) == 0.0
