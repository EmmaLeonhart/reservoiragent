"""Reservoir-protected KV-cache eviction (StreamingLLM with a pinned reservoir).

A Reservoir Agent runs *blank ticks*: silent passes with no user input that still append
to the KV cache, so an always-on agent burns its context window faster than a turn-based
one. The imported Grok conversation
(`data_lake/transcripts/attention-reservoir-architecture-grok.md`) points at the standard
remedy — StreamingLLM's "attention sink + recent window" eviction — with one project-specific
twist: the **reservoir's K/V entries are pinned** so the persistent time-axis is never the
thing that gets dropped. "A really long time of no activity is signal," and that signal must
survive eviction.

This module is the eviction *policy*, deliberately torch-free: it decides, from a list of
per-position tags, which positions to retain. Applying the selection to actual key/value
tensors (or token ids) is a one-liner with :func:`gather`, done by the caller. Keeping the
policy pure means its logic is unit-tested on CPU in CI, independent of any base model.

Tags per position are one of:

* ``SINK``       — an attention-sink token (StreamingLLM keeps the first few tokens; their
                   keys act as a stable attention anchor). Always retained.
* ``RESERVOIR``  — a reservoir-derived pseudo-token (see ``kv_live.py``). Always retained —
                   this is the pin that protects the time-axis.
* ``NORMAL``     — an ordinary conversation/tick token. The most recent ``recent`` of these
                   form the recent window and are retained; older ones are evictable,
                   newest-kept-first, only as far as the budget allows.

With no ``RESERVOIR`` tags present the policy degrades exactly to vanilla StreamingLLM
(sink + recent window). Passing per-position importance ``scores`` switches the normal-token
choice from recency to H2O-style heavy-hitter retention (keep the highest-scoring), with the
reservoir still pinned either way.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, TypeVar

SINK = "sink"
RESERVOIR = "reservoir"
NORMAL = "normal"
_VALID = frozenset({SINK, RESERVOIR, NORMAL})

T = TypeVar("T")


def gather(seq: Sequence[T], indices: Sequence[int]) -> List[T]:
    """Select ``seq`` positions named by ``indices`` (assumed ascending), in order."""
    return [seq[i] for i in indices]


@dataclass(frozen=True)
class ReservoirEvictionPolicy:
    """Decide which KV positions to keep so the cache stays within ``budget``.

    Parameters
    ----------
    budget:
        Target cap on the number of retained positions. Protected positions (sinks,
        reservoir entries, and the recent window) are a hard floor: if they alone exceed
        ``budget`` they are *all* still retained (and :meth:`over_budget` reports ``True``)
        — the policy never evicts the time-axis to satisfy an arithmetic cap.
    recent:
        Size of the recent window, counted in ``NORMAL`` positions only.
    """

    budget: int
    recent: int = 64

    def __post_init__(self) -> None:
        if self.budget < 0:
            raise ValueError(f"budget must be >= 0, got {self.budget}")
        if self.recent < 0:
            raise ValueError(f"recent must be >= 0, got {self.recent}")

    def _partition(self, tags: Sequence[str]):
        """Return (protected_set, evictable_normals_ascending)."""
        protected = set()
        normal_idx: List[int] = []
        for i, t in enumerate(tags):
            if t not in _VALID:
                raise ValueError(f"unknown tag {t!r}; expected one of {sorted(_VALID)}")
            if t == NORMAL:
                normal_idx.append(i)
            else:  # SINK or RESERVOIR
                protected.add(i)
        recent_set = set(normal_idx[-self.recent:]) if self.recent > 0 else set()
        protected |= recent_set
        evictable = [i for i in normal_idx if i not in recent_set]
        return protected, evictable

    def retained_indices(self, tags: Sequence[str],
                         scores: Sequence[float] | None = None) -> List[int]:
        """Positions to keep, ascending. Length is ``<= budget`` unless protected entries
        alone exceed the budget (see :meth:`over_budget`).

        ``scores`` (optional) gives a per-position importance (e.g. accumulated attention mass,
        H2O-style). When provided, the budget headroom is filled with the *highest-scoring*
        evictable normal tokens (heavy hitters) instead of the newest, ties broken by recency.
        When ``None`` (the default) headroom is filled by recency, i.e. plain StreamingLLM."""
        protected, evictable = self._partition(tags)
        kept = set(protected)
        headroom = self.budget - len(protected)
        if headroom > 0 and evictable:
            if scores is None:
                kept |= set(evictable[-headroom:])              # recency: newest tokens
            else:
                # heavy hitters: highest score first, ties broken toward more recent (larger idx)
                ranked = sorted(evictable, key=lambda i: (scores[i], i), reverse=True)
                kept |= set(ranked[:headroom])
        return sorted(kept)

    def evicted_indices(self, tags: Sequence[str],
                        scores: Sequence[float] | None = None) -> List[int]:
        """Positions to drop, ascending — the complement of :meth:`retained_indices`."""
        retained = set(self.retained_indices(tags, scores))
        return [i for i in range(len(tags)) if i not in retained]

    def over_budget(self, tags: Sequence[str]) -> bool:
        """True when the protected (un-evictable) positions alone exceed ``budget``."""
        protected, _ = self._partition(tags)
        return len(protected) > self.budget
