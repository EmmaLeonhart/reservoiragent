"""Spectral-radius dynamics sweep.

Drives a fixed reservoir with synthetic (or supplied) input streams across a grid of
spectral radii and measures the dynamics observables, to locate the regime where the
state is non-saturating, non-exploding, high-dimensional, AND carries usable history
(forgets its initial condition — the echo state property — while still separating
different input histories).

Two distinguishability probes capture the echo state property directly:

- ``init_forgetting`` — SAME input, two different initial states. Low ⇒ the reservoir
  forgets where it started (ESP holds). It should rise sharply once ρ ≳ 1.
- ``input_separation`` — DIFFERENT inputs, same (zero) start. High ⇒ the reservoir
  actually responds to its input.

A healthy reservoir wants low ``init_forgetting`` and high ``input_separation``, with
``saturation`` low and ``participation_ratio`` high.
"""
from __future__ import annotations

import numpy as np

from .echo_state import EchoStateReservoir
from .metrics import (
    saturation_fraction,
    state_variance,
    participation_ratio,
    trajectory_distinguishability,
)


def synthetic_input(T: int, n_input: int, seed: int) -> np.ndarray:
    """A standard-normal input stream of shape (T, n_input)."""
    return np.random.default_rng(seed).standard_normal((T, n_input))


def measure_point(rho: float, K: int, *, n_input: int = 4, T: int = 600,
                  washout: int = 100, seed: int = 0, input_seed: int = 0,
                  **reservoir_kwargs) -> dict:
    """Measure dynamics observables for one (ρ, K) reservoir on synthetic input."""
    res = EchoStateReservoir(K, n_input, spectral_radius=rho, seed=seed,
                             **reservoir_kwargs)
    u = synthetic_input(T, n_input, input_seed)
    base = res.run(u)[washout:]

    # init forgetting: AUTONOMOUS probe (zero input), two different random initial
    # states. Under zero drive the null state is stable iff ρ < 1, so this isolates
    # the echo-state-property boundary at the spectral radius — the edge-of-chaos
    # transition — instead of letting strong input drive mask it (a unit-scale input
    # enforces the ESP across all ρ, hiding the boundary).
    zero = np.zeros((T, n_input))
    rng = np.random.default_rng(seed + 9991)
    res.reset(rng.standard_normal(K))
    a = res.run(zero, reset=False)[washout:]
    res.reset(rng.standard_normal(K))
    b = res.run(zero, reset=False)[washout:]
    init_forgetting = trajectory_distinguishability(a, b)

    # input separation: two different inputs from the zero state
    u2 = synthetic_input(T, n_input, input_seed + 1)
    sep = trajectory_distinguishability(res.run(u)[washout:], res.run(u2)[washout:])

    return {
        "rho": float(rho),
        "K": int(K),
        "variance": state_variance(base),
        "saturation": saturation_fraction(base),
        "participation_ratio": participation_ratio(base),
        "participation_ratio_frac": participation_ratio(base) / K,
        "init_forgetting": init_forgetting,
        "input_separation": sep,
    }


def run_sweep(rhos, K: int = 200, **kwargs) -> list[dict]:
    """Measure the dynamics observables across a grid of spectral radii."""
    return [measure_point(float(rho), K, **kwargs) for rho in rhos]


def _zscore(stream: np.ndarray) -> np.ndarray:
    """Per-dimension standardize an input stream (real activations have large, biased
    magnitudes that would otherwise saturate the reservoir)."""
    stream = np.asarray(stream, dtype=float)
    mu = stream.mean(axis=0, keepdims=True)
    sd = stream.std(axis=0, keepdims=True)
    return (stream - mu) / (sd + 1e-8)


def measure_point_stream(rho: float, stream_a, stream_b, K: int, *, washout: int = 100,
                         seed: int = 0, input_scaling: float = 1.0,
                         normalize: bool = True, **reservoir_kwargs) -> dict:
    """Like :func:`measure_point` but driven by supplied input streams (e.g. real
    transformer activations) instead of synthetic noise. ``stream_a`` drives the
    dynamics; ``stream_b`` (a different input history) is used for input separation.
    The init-forgetting probe is autonomous (zero input), so the ESP boundary it
    measures is independent of the stream.
    """
    a_in = _zscore(stream_a) if normalize else np.asarray(stream_a, dtype=float)
    b_in = _zscore(stream_b) if normalize else np.asarray(stream_b, dtype=float)
    n_input = a_in.shape[1]
    res = EchoStateReservoir(K, n_input, spectral_radius=rho, seed=seed,
                             input_scaling=input_scaling, **reservoir_kwargs)

    base = res.run(a_in)[washout:]

    zero = np.zeros((a_in.shape[0], n_input))
    rng = np.random.default_rng(seed + 9991)
    res.reset(rng.standard_normal(K))
    p = res.run(zero, reset=False)[washout:]
    res.reset(rng.standard_normal(K))
    q = res.run(zero, reset=False)[washout:]
    init_forgetting = trajectory_distinguishability(p, q)

    T = min(a_in.shape[0], b_in.shape[0])
    sep = trajectory_distinguishability(res.run(a_in[:T])[washout:],
                                        res.run(b_in[:T])[washout:])
    return {
        "rho": float(rho), "K": int(K),
        "variance": state_variance(base),
        "saturation": saturation_fraction(base),
        "participation_ratio": participation_ratio(base),
        "participation_ratio_frac": participation_ratio(base) / K,
        "init_forgetting": init_forgetting,
        "input_separation": sep,
    }


def run_sweep_stream(rhos, stream_a, stream_b, K: int = 200, **kwargs) -> list[dict]:
    """Spectral-radius sweep driven by supplied input streams."""
    return [measure_point_stream(float(rho), stream_a, stream_b, K, **kwargs)
            for rho in rhos]


def healthy_regime(records: list[dict], *, sat_max: float = 0.5,
                   forget_max: float = 0.2) -> list[dict]:
    """Records whose state forgets its init (ESP) and is not over-saturated."""
    return [r for r in records
            if r["init_forgetting"] <= forget_max and r["saturation"] <= sat_max]


def plot_sweep(records: list[dict], path: str, *, title: str = "Reservoir dynamics vs spectral radius") -> None:
    """Render the sweep curves to an image file (matplotlib, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho = [r["rho"] for r in records]
    series = [
        ("saturation", "saturation (|r|>0.99)", "#b8553a"),
        ("participation_ratio_frac", "participation ratio / K", "#5b7a4a"),
        ("init_forgetting", "init forgetting (ESP: lower=better)", "#a07a2b"),
        ("input_separation", "input separation (higher=better)", "#3f6487"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in series:
        ax.plot(rho, [r[key] for r in records], "-o", ms=4, color=color, label=label)
    ax.axvline(1.0, color="#999", ls="--", lw=1, label="ρ = 1 (echo-state boundary)")
    ax.set_xlabel("spectral radius ρ")
    ax.set_ylabel("metric")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
