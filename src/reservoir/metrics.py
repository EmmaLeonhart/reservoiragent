"""Dynamics metrics for reservoir state trajectories.

These quantify whether a reservoir's state carries usable signal rather than noise —
the observables the spectral-radius sweep uses to locate a healthy regime:

- ``state_variance``        — how active the units are (dead vs. driven).
- ``saturation_fraction``   — fraction of state values pinned near ±1 (over-driven).
- ``participation_ratio``   — effective dimensionality of the explored state space.
- ``trajectory_distinguishability`` — how far apart two trajectories driven by
  different input histories stay (the precondition for the state to encode history).

A trajectory is an array of shape ``(T, K)`` — T time steps of a K-unit reservoir.
"""
from __future__ import annotations

import numpy as np


def _as_traj(traj) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2:
        raise ValueError("trajectory must have shape (T, K)")
    return traj


def state_variance(traj) -> float:
    """Mean over units of each unit's temporal variance (0 for a constant state)."""
    traj = _as_traj(traj)
    return float(np.mean(np.var(traj, axis=0)))


def saturation_fraction(traj, threshold: float = 0.99) -> float:
    """Fraction of state values with magnitude above ``threshold`` (near ±1)."""
    traj = _as_traj(traj)
    return float(np.mean(np.abs(traj) > threshold))


def participation_ratio(traj) -> float:
    """Effective dimensionality of the state covariance.

    PR = (Σ λ_i)² / Σ λ_i², where λ_i are the eigenvalues of the unit covariance.
    PR ≈ 1 when all units carry the same signal (rank-1); PR ≈ K when the K units
    vary independently with equal power.
    """
    traj = _as_traj(traj)
    # covariance across units (K×K); rowvar=False treats columns as variables
    cov = np.cov(traj, rowvar=False)
    cov = np.atleast_2d(cov)
    lam = np.linalg.eigvalsh(cov)
    lam = np.clip(lam, 0.0, None)  # tiny negatives from numerical error
    denom = float(np.sum(lam ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(lam) ** 2 / denom)


def trajectory_distinguishability(traj_a, traj_b) -> float:
    """Mean per-step RMS distance between two equal-length trajectories.

    0 when identical; for K units this is ``mean_t ||a_t - b_t|| / sqrt(K)``, so a
    constant unit-magnitude separation gives 1.0 regardless of K.
    """
    a, b = _as_traj(traj_a), _as_traj(traj_b)
    if a.shape != b.shape:
        raise ValueError("trajectories must have the same shape")
    K = a.shape[1]
    per_step = np.linalg.norm(a - b, axis=1) / np.sqrt(K)
    return float(np.mean(per_step))
