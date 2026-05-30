"""N-seed selection — train each seed's readout, keep the best; and test whether a
cheap *untrained* dynamics proxy predicts the *trained* performance.

This is the plan's N-seed selection mechanism at small scale: initialise N fixed
reservoirs with different seeds, train each one's readout on a reservoir-requiring task
(the delay-memory task, closed-form ridge — real training), evaluate, and keep the best
seed. It also answers an open question from the plan (§D, "seed pre-selection proxy"):
can a cheap dynamics metric measured *before* training predict which seed will train
best? If so, seeds can be pre-filtered before spending compute.

All numpy/CPU.
"""
from __future__ import annotations

import numpy as np

from .echo_state import EchoStateReservoir
from .metrics import participation_ratio, saturation_fraction
from .tasks import delay_memory_curve, memory_capacity


def seed_report(seed: int, *, K: int = 200, rho: float = 0.9,
                input_scaling: float = 0.5, max_delay: int = 30) -> dict:
    """For one seed: the trained-readout memory capacity (the selection criterion) plus
    cheap *untrained* dynamics metrics (the candidate pre-selection proxy)."""
    recs = delay_memory_curve(K, delays=range(0, max_delay + 1), rho=rho,
                              input_scaling=input_scaling, seed=seed)
    mc = memory_capacity(recs, "reservoir_r2")

    # untrained dynamics on a random drive (no readout fit)
    res = EchoStateReservoir(K, 1, spectral_radius=rho, input_scaling=input_scaling,
                             seed=seed)
    u = np.random.default_rng(seed + 5).uniform(-1.0, 1.0, size=(1500, 1))
    R = res.run(u)[200:]
    return {
        "seed": int(seed),
        "memory_capacity": float(mc),
        "pr_frac": float(participation_ratio(R) / K),
        "saturation": float(saturation_fraction(R)),
    }


def select_seeds(seeds, **kwargs) -> list[dict]:
    """Rank seeds best-first by trained-readout memory capacity."""
    return sorted((seed_report(int(s), **kwargs) for s in seeds),
                  key=lambda r: r["memory_capacity"], reverse=True)


def proxy_correlation(records: list[dict], proxy_key: str = "pr_frac") -> dict:
    """Spearman rank correlation between the untrained proxy and trained memory
    capacity across seeds — does the cheap proxy predict trained performance?"""
    from scipy.stats import spearmanr

    proxy = [r[proxy_key] for r in records]
    mc = [r["memory_capacity"] for r in records]
    rho_s, p = spearmanr(proxy, mc)
    return {"proxy": proxy_key, "spearman": float(rho_s), "p_value": float(p),
            "n": len(records)}
