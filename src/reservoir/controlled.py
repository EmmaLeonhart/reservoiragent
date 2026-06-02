"""Controlled N-seed selection: is reservoir quality real, or run-to-run noise?

Earlier the per-seed cross-pass recall spread (some seeds 1.0, some at chance) looked like
"reservoir selection" — but a confound check showed the same reservoir seed landed at very
different recall across runs, so at 250 steps the spread was training-noise, not reservoir
quality (devlog 2026-05-31). Phase I removes the noise sources: the trainable init and data
order are now seeded by ``train_seed`` (independent of the reservoir ``seed``) and the kernels
are made deterministic, so two runs of one reservoir with the same ``train_seed`` are identical.

The experiment: train **R runs per reservoir seed** (same R ``train_seed`` values across seeds),
record recall, and ask with a one-way ANOVA whether the **between-seed** variation exceeds the
**within-seed** (run-to-run) variation. If it does, some fixed reservoirs are durably better —
selection is real. If not, the spread is noise and "selection" was an artifact.

``selection_signal`` (the analysis) is pure and unit-tested; ``controlled_selection`` (the
trainer) is torch-gated and run locally.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence

import numpy as np


def selection_signal(records: List[Dict]) -> Dict[str, object]:
    """One-way ANOVA over recall grouped by reservoir seed.

    Each record needs ``reservoir_seed`` and ``recall_accuracy``. Returns the per-seed means,
    the between- vs within-seed mean squares, the F ratio + p-value, and a verdict
    ``selection_is_real`` (between-seed variation significant at p < 0.05)."""
    groups: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        groups[int(r["reservoir_seed"])].append(float(r["recall_accuracy"]))
    seeds = sorted(groups)
    if len(seeds) < 2:
        raise ValueError("need at least 2 reservoir seeds to compare")

    arrs = {s: np.asarray(groups[s], dtype=float) for s in seeds}
    means = {s: float(a.mean()) for s, a in arrs.items()}
    all_vals = np.concatenate([arrs[s] for s in seeds])
    grand = float(all_vals.mean())
    k, N = len(seeds), int(all_vals.size)

    ss_between = float(sum(arrs[s].size * (means[s] - grand) ** 2 for s in seeds))
    ss_within = float(sum(((arrs[s] - means[s]) ** 2).sum() for s in seeds))
    df_b, df_w = k - 1, N - k
    ms_b = ss_between / df_b if df_b > 0 else 0.0
    ms_w = ss_within / df_w if df_w > 0 else 0.0

    if ms_w == 0.0:
        f_ratio = float("inf") if ms_b > 0 else 0.0
    else:
        f_ratio = ms_b / ms_w

    p_value = None
    if df_b > 0 and df_w > 0:
        try:
            from scipy import stats
            if np.isfinite(f_ratio):
                p_value = float(stats.f.sf(f_ratio, df_b, df_w))
            else:
                p_value = 0.0   # infinite F (zero within-seed variance, nonzero between)
        except Exception:
            p_value = None

    if p_value is not None:
        selection_is_real = bool(p_value < 0.05)
    else:                                   # no scipy: fall back to the raw ratio
        selection_is_real = bool(f_ratio == float("inf") or f_ratio > 3.0)

    return {
        "n_seeds": k,
        "runs_total": N,
        "per_seed_mean": {str(s): means[s] for s in seeds},
        "per_seed_std": {str(s): float(arrs[s].std(ddof=1)) if arrs[s].size > 1 else 0.0
                         for s in seeds},
        "grand_mean": grand,
        "between_seed_ms": ms_b,
        "within_seed_ms": ms_w,
        "f_ratio": f_ratio,
        "df_between": df_b,
        "df_within": df_w,
        "p_value": p_value,
        "selection_is_real": selection_is_real,
        "best_seed": int(max(seeds, key=lambda s: means[s])),
        "worst_seed": int(min(seeds, key=lambda s: means[s])),
    }


def controlled_selection(
    reservoir_seeds: Sequence[int],
    runs_per_seed: int,
    *,
    model_name: str = "gpt2",
    steps: int = 600,
    n_keys: int = 6,
    lr: float = 1e-3,
    n_prefix: int = 8,
    input_scaling: float = 0.5,
    device: str | None = None,
    deterministic: bool = True,
    progress: Callable[[int, int, dict], None] | None = None,
) -> List[Dict]:
    """Train ``runs_per_seed`` runs for each reservoir seed (same train_seeds across seeds) and
    record recall per (seed, run). Torch-gated; the heavy local step."""
    from .crosspass import run_cross_pass_kv

    records: List[Dict] = []
    for s in reservoir_seeds:
        for run in range(runs_per_seed):
            r = run_cross_pass_kv(model_name, n_keys=n_keys, steps=steps, lr=lr, seed=int(s),
                                  train_seed=run, deterministic=deterministic, n_prefix=n_prefix,
                                  input_scaling=input_scaling, device=device)
            rec = {"reservoir_seed": int(s), "run": int(run), "train_seed": int(run),
                   "recall_accuracy": r["recall_accuracy"], "loss_end": r["loss_end"]}
            records.append(rec)
            if progress is not None:
                progress(int(s), int(run), r)
    return records


def plot_controlled(records: List[Dict], signal: Dict, path: str, *,
                    title: str = "Controlled N-seed selection: reservoir quality vs. run noise") -> None:
    """Per reservoir seed, plot each run's recall (points) + the seed mean, so between-seed
    spread is visually comparable to within-seed spread; annotate the ANOVA verdict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        groups[int(r["reservoir_seed"])].append(float(r["recall_accuracy"]))
    seeds = sorted(groups)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for i, s in enumerate(seeds):
        ys = groups[s]
        ax.scatter([i] * len(ys), ys, color="#1f6f8b", alpha=0.6, s=28, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.mean(ys)] * 2, color="#c0392b", lw=2.5, zorder=4)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("cross-pass recall")
    ax.set_ylim(-0.05, 1.05)
    f = signal["f_ratio"]
    p = signal["p_value"]
    verdict = "selection REAL" if signal["selection_is_real"] else "noise-dominated"
    ptxt = "n/a" if p is None else f"{p:.3g}"
    ax.set_title(f"{title}\nF={f:.2f} (df {signal['df_between']},{signal['df_within']}), "
                 f"p={ptxt} — {verdict}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
