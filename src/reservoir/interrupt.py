"""Interruptibility: how fast an urgent human "STOP" registers (a safety experiment).

The imported Grok conversation (`data_lake/transcripts/attention-reservoir-architecture-grok.md`)
makes a controllability argument: a turn-based harness agent "doesn't respond and often ends
up responding quite late" because it only reads input at turn boundaries — "I yell at it to
stop ... and it takes like ten minutes for it to respond." A Reservoir Agent, running every
tick with the reservoir always integrating input, "will be able to see frantic messages from
a human as indicating stop immediately." This module measures both halves of that claim on
CPU:

1. **Polling latency (structural).** A poller that only reads every ``period`` passes
   registers an arrival at the next boundary — latency uniform on ``0..period-1`` (mean
   ``(period-1)/2``, max ``period-1``). A per-tick poller reads every pass: latency 0.

2. **Signal persistence (dynamics).** A *one-shot* urgent burst (the user yells STOP once,
   then the channel goes quiet because the agent isn't answering) leaves a trace in the
   reservoir state that decays slowly via fading memory, so a matched-filter monitor on the
   reservoir state stays above threshold for several passes *after* arrival. A stateless
   monitor — which sees only the current input — catches the burst on the exact arrival pass
   and not one pass later. Combined: a turn-based + stateless agent can *miss a non-repeated
   burst entirely* when its poll period outruns the persistence window; the per-tick reservoir
   agent catches it on arrival and has a window besides.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .echo_state import EchoStateReservoir


def poll_latency_stats(horizon: int, period: int) -> Dict[str, float]:
    """Latency (in passes) to register an arrival for a poller reading every ``period`` passes.

    Boundaries are multiples of ``period``; an arrival at pass ``a`` is read at the smallest
    boundary ``>= a``. ``period <= 1`` reads every pass (latency 0)."""
    arrivals = np.arange(int(horizon))
    if period <= 1:
        lat = np.zeros(int(horizon))
    else:
        lat = np.ceil(arrivals / period) * period - arrivals
    return {"mean": float(lat.mean()) if horizon else 0.0,
            "max": int(lat.max()) if horizon else 0,
            "period": int(period)}


def reservoir_persistence(
    *,
    n_reservoir: int = 200,
    n_input: int = 16,
    t_star: int = 20,
    horizon: int = 60,
    urgent_scale: float = 6.0,
    seed: int = 0,
    spectral_radius: float = 0.9,
    leak_rate: float = 0.2,   # low leak = long fading memory: the regime where a one-shot
                              # burst lingers in state across passes (the persistence claim)
    input_scaling: float = 0.5,
    baseline_noise: float = 0.05,
    threshold_frac: float = 0.5,
) -> Dict[str, object]:
    """Drive a quiet input stream with a single urgent burst at ``t_star`` and measure how
    long the burst stays detectable in the reservoir state vs. in the raw input."""
    res = EchoStateReservoir(n_reservoir, n_input, spectral_radius=spectral_radius,
                             leak_rate=leak_rate, input_scaling=input_scaling, seed=seed)
    rng = np.random.default_rng(seed + 1000)

    u = np.zeros(n_input)
    u[0] = 1.0                                   # a fixed, deterministic "STOP" direction
    inputs = baseline_noise * rng.standard_normal((horizon, n_input))
    inputs[t_star] = u * urgent_scale            # the lone urgent burst

    # matched filter: the reservoir's state-response to a lone urgent impulse from rest.
    res.reset()
    r_ref = res.step(u * urgent_scale).copy()
    r_ref = r_ref / (np.linalg.norm(r_ref) + 1e-12)

    res.reset()
    states = res.run(inputs, reset=False)        # (horizon, n_reservoir)
    reservoir_score = states @ r_ref             # STOP signature present in state?
    stateless_score = inputs @ u                 # STOP component of the *current* input

    peak = float(reservoir_score[t_star])
    threshold = threshold_frac * peak

    def _window(score, thresh):                  # consecutive passes after t_star over thresh
        w = 0
        for t in range(t_star + 1, horizon):
            if score[t] > thresh:
                w += 1
            else:
                break
        return w

    return {
        "t_star": t_star,
        "horizon": horizon,
        "reservoir_score": reservoir_score.tolist(),
        "stateless_score": stateless_score.tolist(),
        "threshold": threshold,
        "reservoir_window": _window(reservoir_score, threshold),
        "stateless_window": _window(stateless_score, threshold_frac * float(stateless_score[t_star])),
    }


def run_interrupt_experiment(
    *,
    period: int = 8,
    n_reservoir: int = 200,
    n_input: int = 16,
    t_star: int = 20,
    horizon: int = 60,
    urgent_scale: float = 6.0,
    seed: int = 0,
    **persistence_kw,
) -> Dict[str, object]:
    """Combine the polling-latency and signal-persistence measurements into one record."""
    poll = poll_latency_stats(horizon=horizon, period=period)
    per_tick = poll_latency_stats(horizon=horizon, period=1)
    persist = reservoir_persistence(n_reservoir=n_reservoir, n_input=n_input, t_star=t_star,
                                    horizon=horizon, urgent_scale=urgent_scale, seed=seed,
                                    **persistence_kw)
    # a stateless turn-based agent sees the lone burst only if a poll boundary lands on t_star.
    burst_missed = (t_star % period != 0)
    summary = {
        "turn_based_mean_latency": poll["mean"],
        "turn_based_max_latency": poll["max"],
        "per_tick_latency": per_tick["max"],
        "reservoir_window": persist["reservoir_window"],
        "stateless_window": persist["stateless_window"],
        "burst_missed_by_turn_based": bool(burst_missed),
        "period": int(period),
        "t_star": int(t_star),
    }
    return {"summary": summary, "persistence": persist, "poll": poll}


def plot_interrupt(record: Dict[str, object], path: str, *,
                   title: str = "Interruptibility: a one-shot STOP in reservoir state") -> None:
    """Render the reservoir vs. stateless detection traces around the urgent burst."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = record["persistence"]
    t_star = p["t_star"]
    rscore = np.array(p["reservoir_score"])
    sscore = np.array(p["stateless_score"])
    thr = p["threshold"]
    passes = np.arange(len(rscore))

    # normalise both traces to their own peak so they share an axis (each is a detector score).
    rnorm = rscore / (rscore[t_star] + 1e-12)
    snorm = sscore / (sscore[t_star] + 1e-12)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axvline(t_star, color="#888", ls="--", lw=1, label="urgent STOP (one-shot)")
    ax.plot(passes, rnorm, color="#1f6f8b", lw=2,
            label="reservoir-state monitor (persists)")
    ax.plot(passes, snorm, color="#c0392b", lw=2, marker="o", ms=3,
            label="stateless monitor (this pass only)")
    ax.axhline(thr / (rscore[t_star] + 1e-12), color="#1f6f8b", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("forward pass")
    ax.set_ylabel("detection score (norm. to peak)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
