"""Reservoir-state probes: reading a process property off the fixed reservoir (safety / interp).

The imported Grok conversation argued that the reservoir is an unusually good *monitoring
surface*: "I don't think you'd need a sparse autoencoder for the reservoir state … it's much
more simple to have a learned representation of what is happening," and "because the reservoir
weights never change … you'll always be able to tell what's going on in this particular part"
— a representation "relatively resilient to fine-tuning."

This module tests the falsifiable parts of that on CPU:

1. **Decodable with a *linear* probe, no SAE.** A ridge-regression readout recovers a temporal
   *process property* — elapsed passes since the last trigger event, an internal clock — from
   the reservoir state with high R², while the same probe on the *instantaneous input* (a
   stateless monitor) cannot, because elapsed-time is simply not in the current input.

2. **Resilience to a fine-tuning-like drift (measured, not assumed).** Fine-tuning the readout
   /LoRA shifts the *activations that drive* the reservoir. We model that as a fixed drift added
   to the driving input and re-measure the *pre-drift* probe. Because the reservoir map itself
   is fixed and integrates over history, the probe stays usable up to moderate drift; we report
   the degradation curve rather than claiming invariance. (The reservoir *weights* never move;
   the driving activations still do, so very large fine-tunes would still degrade it — stated
   plainly in FINDINGS.)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .echo_state import EchoStateReservoir


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot (1.0 is perfect; <=0 is useless)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge regression with a bias column. Returns weights of shape (d+1,)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    d = Xb.shape[1]
    reg = alpha * np.eye(d)
    reg[-1, -1] = 0.0  # do not penalise the bias
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return w


def predict_ridge(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w


def _drive_sessions(
    res: EchoStateReservoir,
    n_sessions: int,
    horizon: int,
    n_input: int,
    *,
    max_elapsed: int,
    seed: int,
    drift: np.ndarray | None = None,
    baseline_noise: float = 0.05,
    trigger_scale: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ``n_sessions`` of length ``horizon``; return (states, inputs, elapsed) stacked over
    all passes. ``elapsed`` is passes since the most recent trigger, clamped to ``max_elapsed``.
    A trigger is a spike on input dim 0. ``drift`` (if given) is added to every input."""
    rng = np.random.default_rng(seed)
    trig = np.zeros(n_input)
    trig[0] = 1.0
    states_all, inputs_all, elapsed_all = [], [], []
    for s in range(n_sessions):
        inputs = baseline_noise * rng.standard_normal((horizon, n_input))
        # one trigger per session at a random early pass, then a long quiet tail
        t0 = int(rng.integers(2, max(3, horizon - max_elapsed - 1)))
        inputs[t0] += trig * trigger_scale
        if drift is not None:
            inputs = inputs + drift
        res.reset()
        states = res.run(inputs, reset=False)
        elapsed = np.full(horizon, max_elapsed, dtype=float)
        for t in range(t0, horizon):
            elapsed[t] = min(t - t0, max_elapsed)
        # only score the post-trigger tail (before the trigger there is no clock to read)
        keep = np.arange(t0, horizon)
        states_all.append(states[keep])
        inputs_all.append(inputs[keep])
        elapsed_all.append(elapsed[keep])
    return (np.vstack(states_all), np.vstack(inputs_all), np.concatenate(elapsed_all))


def run_probe_experiment(
    *,
    n_reservoir: int = 200,
    n_input: int = 16,
    n_sessions: int = 200,
    horizon: int = 40,
    max_elapsed: int = 15,
    spectral_radius: float = 0.9,
    leak_rate: float = 0.3,
    input_scaling: float = 0.5,
    ridge_alpha: float = 1.0,
    drift_grid: List[float] | None = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Train linear probes to read the elapsed-since-trigger clock, then measure resilience to
    a fine-tuning-like input drift. Returns R² for the reservoir-state and stateless probes and
    the reservoir probe's R²-vs-drift curve."""
    if drift_grid is None:
        drift_grid = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]
    res = EchoStateReservoir(n_reservoir, n_input, spectral_radius=spectral_radius,
                             leak_rate=leak_rate, input_scaling=input_scaling, seed=seed)

    # ---- train on a clean (pre-fine-tune) split ----
    Xs_tr, Xi_tr, y_tr = _drive_sessions(res, n_sessions, horizon, n_input,
                                         max_elapsed=max_elapsed, seed=seed)
    Xs_te, Xi_te, y_te = _drive_sessions(res, n_sessions // 2, horizon, n_input,
                                         max_elapsed=max_elapsed, seed=seed + 7)
    w_state = fit_ridge(Xs_tr, y_tr, alpha=ridge_alpha)
    w_input = fit_ridge(Xi_tr, y_tr, alpha=ridge_alpha)
    r2_state = r2_score(y_te, predict_ridge(Xs_te, w_state))
    r2_input = r2_score(y_te, predict_ridge(Xi_te, w_input))

    # ---- resilience: apply the SAME pre-drift state-probe to drifted driving inputs ----
    drift_rng = np.random.default_rng(seed + 99)
    drift_dir = drift_rng.standard_normal(n_input)
    drift_dir = drift_dir / (np.linalg.norm(drift_dir) + 1e-12)
    curve = []
    for a in drift_grid:
        Xs_d, _, y_d = _drive_sessions(res, n_sessions // 2, horizon, n_input,
                                       max_elapsed=max_elapsed, seed=seed + 7,
                                       drift=drift_dir * a)
        curve.append({"drift": float(a),
                      "r2_state": r2_score(y_d, predict_ridge(Xs_d, w_state))})

    return {
        "r2_state": r2_state,
        "r2_input": r2_input,
        "max_elapsed": max_elapsed,
        "resilience_curve": curve,
        "params": {"n_reservoir": n_reservoir, "n_input": n_input,
                   "n_sessions": n_sessions, "horizon": horizon,
                   "leak_rate": leak_rate, "spectral_radius": spectral_radius,
                   "ridge_alpha": ridge_alpha},
    }


def plot_probe(record: Dict[str, object], path: str, *,
               title: str = "Reservoir-state probe: an internal clock, linearly decodable") -> None:
    """Two panels: decodability (reservoir state vs. stateless input) and resilience-vs-drift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    ax1.bar(["reservoir\nstate", "stateless\ninput"],
            [record["r2_state"], record["r2_input"]],
            color=["#1f6f8b", "#c0392b"])
    ax1.set_ylabel("probe R²  (elapsed-since-trigger)")
    ax1.set_ylim(min(0, record["r2_input"]) - 0.05, 1.0)
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.set_title("Linear probe, no SAE")
    for i, v in enumerate([record["r2_state"], record["r2_input"]]):
        ty = min(v + 0.02, 0.93) if v >= 0 else v - 0.05
        ax1.text(i, ty, f"{v:.2f}", ha="center", fontsize=9)

    curve = record["resilience_curve"]
    xs = [c["drift"] for c in curve]
    ys = [c["r2_state"] for c in curve]
    ax2.plot(xs, ys, color="#1f6f8b", lw=2, marker="o", ms=4)
    ax2.set_xlabel("fine-tuning-like input drift (α)")
    ax2.set_ylabel("R² of the pre-drift probe")
    ax2.set_title("Resilience to driving-activation drift")
    ax2.set_ylim(min(0, min(ys)) - 0.05, 1.0)
    ax2.axhline(0, color="#999", lw=0.8)

    fig.suptitle(title)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
