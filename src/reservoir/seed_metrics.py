"""Per-seed reservoir metrics — hunting for a predictor of which reservoirs train well.

The N=12 + N=20 GPT-2 selection batches show cross-pass recall ranging 1.00 → chance
across seeds, yet the cheap dynamics proxy (participation ratio) is ~constant and does not
track recall. That is partly *by construction*: every reservoir is scaled to the same
target spectral radius and sparsity, so bulk metrics can't separate seeds — only the random
structure differs. This module computes a richer set of **untrained, structural** metrics
per seed and correlates them with the trained recall labels, to test whether *any* cheap
metric predicts a good reservoir (the "what makes a good reservoir" question the preserved
populations exist to answer). Pure numpy/scipy; no GPU, no LLM.
"""
from __future__ import annotations

import numpy as np

from .torch_inject import _build_reservoir_weights
from .tasks import delay_memory_curve, memory_capacity

METRIC_KEYS = ("realized_rho", "mean_abs_eig", "std_abs_eig", "non_normality",
               "participation_ratio", "memory_capacity")


def reservoir_metrics(seed: int, *, K: int = 512, d_in: int = 768,
                      spectral_radius: float = 0.9, input_scaling: float = 0.5,
                      sparsity: float = 0.1) -> dict:
    """Untrained structural metrics of the fixed reservoir built from ``seed`` (same
    construction the batches use). Deterministic in ``seed``."""
    W, W_in = _build_reservoir_weights(K, d_in, spectral_radius, input_scaling,
                                       sparsity, seed)
    W = np.asarray(W, float)
    eig = np.linalg.eigvals(W)
    abs_eig = np.abs(eig)

    # Henrici departure from normality, normalised by the Frobenius norm: how far W is from
    # a normal matrix (a structural property that varies with the random pattern even when
    # the spectral radius is fixed).
    fro2 = float(np.sum(W * W))
    departure = np.sqrt(max(fro2 - float(np.sum(abs_eig ** 2)), 0.0))
    non_normality = departure / np.sqrt(fro2) if fro2 > 0 else 0.0

    # participation ratio of reservoir states on a random drive (the existing cheap proxy)
    rng = np.random.default_rng(seed)
    state = np.zeros(K)
    states = []
    for _ in range(200):
        state = np.tanh(W @ state + W_in @ rng.standard_normal(d_in))
        states.append(state.copy())
    S = np.asarray(states[50:])
    ev = np.linalg.eigvalsh(np.cov(S, rowvar=False)).clip(min=0)
    pr = (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)

    mc = memory_capacity(delay_memory_curve(K, rho=spectral_radius,
                                            input_scaling=input_scaling, seed=seed))

    return {"realized_rho": float(abs_eig.max()),
            "mean_abs_eig": float(abs_eig.mean()),
            "std_abs_eig": float(abs_eig.std()),
            "non_normality": float(non_normality),
            "participation_ratio": float(pr / K),
            "memory_capacity": float(mc)}


def correlate_seed_metrics(seed_recall: dict, **metric_kwargs) -> dict:
    """Spearman rank correlation of each untrained metric vs trained recall, across seeds.

    ``seed_recall`` maps seed -> recall accuracy (from the batch manifests). Returns
    ``{metric: {spearman, p_value, n}}``. Raises on empty input.
    """
    if not seed_recall:
        raise ValueError("seed_recall is empty")
    from scipy.stats import spearmanr

    seeds = sorted(seed_recall)
    recalls = [seed_recall[s] for s in seeds]
    metrics = {s: reservoir_metrics(s, **metric_kwargs) for s in seeds}
    out = {}
    for key in METRIC_KEYS:
        vals = [metrics[s][key] for s in seeds]
        if len(set(vals)) <= 1:                       # constant metric -> undefined corr
            out[key] = {"spearman": float("nan"), "p_value": float("nan"),
                        "n": len(seeds), "note": "constant"}
            continue
        rho, p = spearmanr(vals, recalls)
        out[key] = {"spearman": float(rho), "p_value": float(p), "n": len(seeds)}
    return out
