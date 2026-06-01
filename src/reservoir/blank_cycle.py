"""Blank-cycle context-growth simulation (the Grok chat's concern, made measurable).

The imported conversation (`data_lake/transcripts/attention-reservoir-architecture-grok.md`)
identified the core operational pain point of an always-on Reservoir Agent: a *blank tick* —
an autonomous pass with no user input — still appends to the KV cache, so context burns
faster than a turn-based model that only runs when prompted. Left unmanaged the cache grows
linearly with the number of ticks and the agent eventually hits its context limit.

This module quantifies that, and the fix, with no GPU: it streams ``n_ticks`` blank ticks
through a cache and compares the size under two regimes — **no eviction** (vanilla growth)
versus the :class:`~reservoir.kv_evict.ReservoirEvictionPolicy`. The reservoir's K/V entries
are pinned, so the simulation also confirms that under heavy eviction the persistent
time-axis (every reservoir entry) survives on every tick — "a really long time of no
activity is signal", and that signal is never the thing evicted.

The reservoir contributes a *fixed* number of pseudo-tokens (it is re-prepended each pass in
``kv_live.py``, not accumulated), which is exactly why pinning it costs only a constant.
"""
from __future__ import annotations

from typing import Dict, List

from .kv_evict import NORMAL, RESERVOIR, SINK, ReservoirEvictionPolicy, gather


def simulate_blank_cycle(
    n_ticks: int,
    *,
    tokens_per_tick: int = 1,
    n_sink: int = 4,
    n_reservoir: int = 8,
    budget: int = 128,
    recent: int = 64,
) -> List[Dict[str, int]]:
    """Stream ``n_ticks`` blank ticks and record per-tick cache sizes.

    Each tick appends ``tokens_per_tick`` ordinary (NORMAL) tokens — a blank tick still emits
    at least a marker. ``vanilla_size`` is the cache size with no eviction; ``protected_size``
    is the size after applying the reservoir-protected policy each tick; ``reservoir_retained``
    counts how many of the ``n_reservoir`` reservoir entries survived (always all of them).
    """
    pol = ReservoirEvictionPolicy(budget=budget, recent=recent)
    # The cache opens with the attention sinks + the reservoir pseudo-tokens already pinned.
    tags = [SINK] * n_sink + [RESERVOIR] * n_reservoir
    vanilla = len(tags)
    records: List[Dict[str, int]] = []
    for t in range(1, n_ticks + 1):
        tags = tags + [NORMAL] * tokens_per_tick
        vanilla += tokens_per_tick
        tags = gather(tags, pol.retained_indices(tags))  # evict; this is the cache next tick
        records.append({
            "tick": t,
            "vanilla_size": vanilla,
            "protected_size": len(tags),
            "reservoir_retained": sum(1 for x in tags if x == RESERVOIR),
            "reservoir_expected": n_reservoir,
        })
    return records


def plot_blank_cycle(records: List[Dict[str, int]], path: str, *,
                     title: str = "Blank-cycle KV-cache growth") -> None:
    """Render vanilla vs reservoir-protected cache size over ticks to an image file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = [r["tick"] for r in records]
    vanilla = [r["vanilla_size"] for r in records]
    protected = [r["protected_size"] for r in records]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ticks, vanilla, color="#c0392b", lw=2,
            label="no eviction (vanilla) — linear burn")
    ax.plot(ticks, protected, color="#1f6f8b", lw=2,
            label="reservoir-protected policy — bounded")
    if records:
        budget = max(protected)
        ax.axhline(budget, color="#1f6f8b", ls=":", lw=1, alpha=0.7)
        ax.text(ticks[-1], budget, f"  budget={budget}", va="center",
                ha="left", color="#1f6f8b", fontsize=8)
    ax.set_xlabel("blank ticks")
    ax.set_ylabel("KV-cache positions")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
