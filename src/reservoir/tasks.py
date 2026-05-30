"""Reservoir-requiring tasks + a trained readout (H3).

The H3 question: can a light, trained readout extract from the reservoir state
information about the model's *history* that a stateless model cannot recover from the
current input alone? The canonical demonstration is the **delay-memory task**: drive
the reservoir with an i.i.d. input u(t) and train a linear readout to reproduce the
input from τ steps ago, u(t−τ). The reservoir's fading memory lets it reconstruct
delayed inputs; a stateless readout on the *current* input u(t) cannot (the inputs are
independent across time), so any above-chance recovery at τ ≥ 1 is information that
lives in the carried state, not the input.

All numpy / CPU — the readout is closed-form ridge regression.
"""
from __future__ import annotations

import numpy as np

from .echo_state import EchoStateReservoir


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    """Closed-form ridge readout W minimizing ||X W − Y||² + alpha||W||²."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    A = X.T @ X + alpha * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Y)


def r2_score(Y: np.ndarray, P: np.ndarray) -> float:
    """Coefficient of determination, averaged over output columns."""
    Y = np.atleast_2d(np.asarray(Y, float))
    P = np.atleast_2d(np.asarray(P, float))
    if Y.shape[0] == 1:
        Y, P = Y.T, P.T
    ss_res = ((Y - P) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return float(np.mean(1.0 - ss_res / (ss_tot + 1e-12)))


def delay_memory_curve(K: int, *, delays=range(0, 26), T: int = 2500,
                       rho: float = 0.9, input_scaling: float = 0.5,
                       washout: int = 100, seed: int = 0, alpha: float = 1e-4) -> list[dict]:
    """For each delay τ, the test-set R² of recovering u(t−τ) from (a) the reservoir
    state and (b) the current input (stateless baseline). Returns one record per τ.
    """
    res = EchoStateReservoir(K, 1, spectral_radius=rho, input_scaling=input_scaling,
                             seed=seed)
    rng = np.random.default_rng(seed + 1)
    u = rng.uniform(-1.0, 1.0, size=(T, 1))
    R = res.run(u)  # (T, K)

    records = []
    for tau in delays:
        start = max(int(tau), washout)
        X = R[start:]                      # reservoir state at t
        Xb = u[start:]                     # current input at t (stateless baseline)
        Y = u[start - tau:T - tau]         # target: input from tau steps ago
        n = X.shape[0]
        h = n // 2                          # train / test split
        Wr = fit_ridge(X[:h], Y[:h], alpha)
        Wb = fit_ridge(Xb[:h], Y[:h], alpha)
        records.append({
            "delay": int(tau),
            "reservoir_r2": max(0.0, r2_score(Y[h:], X[h:] @ Wr)),
            "baseline_r2": max(0.0, r2_score(Y[h:], Xb[h:] @ Wb)),
        })
    return records


def memory_capacity(records: list[dict], key: str = "reservoir_r2") -> float:
    """Total linear short-term memory capacity = Σ_{τ≥1} R²(τ)."""
    return float(sum(r[key] for r in records if r["delay"] >= 1))


def plot_memory_curve(records: list[dict], path: str, *,
                      title: str = "Delay-memory: reservoir state vs stateless baseline") -> None:
    """Render the recovery R² vs delay for the reservoir and the stateless baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = [r["delay"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, [r["reservoir_r2"] for r in records], "-o", ms=4, color="#5b7a4a",
            label="readout on reservoir state")
    ax.plot(d, [r["baseline_r2"] for r in records], "-o", ms=4, color="#b8553a",
            label="stateless baseline (current input only)")
    ax.set_xlabel("delay τ  (recover input from τ steps ago)")
    ax.set_ylabel("test R²")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
