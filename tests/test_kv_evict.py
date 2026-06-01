"""Tests for the reservoir-protected KV eviction policy (written before the implementation).

The Grok chat (`data_lake/transcripts/attention-reservoir-architecture-grok.md`) raised the
concrete operational problem: a Reservoir Agent that processes *blank ticks* still appends to
the KV cache every silent pass, so context burns faster than a vanilla transformer. The fix it
points at is StreamingLLM-style eviction (attention sink + recent window) that *pins the
reservoir's K/V entries* so the persistent time-axis survives even as ordinary tokens are
dropped. This module is that policy, kept torch-free so the logic is CI-testable on CPU.
"""
import pytest

from reservoir.kv_evict import (
    SINK,
    RESERVOIR,
    NORMAL,
    ReservoirEvictionPolicy,
    gather,
)


def _tags(spec):
    """Expand a compact spec like "S R N N N" into the full tag list."""
    table = {"S": SINK, "R": RESERVOIR, "N": NORMAL}
    return [table[c] for c in spec.split()]


def test_reservoir_entries_are_never_evicted():
    # tiny budget that forces heavy eviction; every reservoir position must survive.
    tags = _tags("S N N R N N R N N N")
    pol = ReservoirEvictionPolicy(budget=4, recent=1)
    kept = set(pol.retained_indices(tags))
    reservoir_idx = {i for i, t in enumerate(tags) if t == RESERVOIR}
    assert reservoir_idx <= kept
    assert not (set(pol.evicted_indices(tags)) & reservoir_idx)


def test_sink_and_recent_window_always_retained():
    tags = _tags("S " + "N " * 10)  # 1 sink + 10 normals
    pol = ReservoirEvictionPolicy(budget=3, recent=2)
    kept = set(pol.retained_indices(tags))
    assert 0 in kept                       # the sink
    assert {9, 10} <= kept                 # the 2 most-recent normals (indices 9,10)


def test_oldest_normal_evicted_first():
    # 1 sink (idx 0) + normals 1..8; recent=2 protects 7,8; budget gives room for some middle.
    tags = _tags("S " + "N " * 8)
    pol = ReservoirEvictionPolicy(budget=5, recent=2)
    # protected = sink(0) + recent(7,8) = 3; headroom = 2 -> keep the 2 newest evictable (6,5)
    kept = pol.retained_indices(tags)
    assert kept == [0, 5, 6, 7, 8]
    # evicted are the OLDEST normals 1..4
    assert pol.evicted_indices(tags) == [1, 2, 3, 4]


def test_budget_respected_when_headroom_allows():
    tags = _tags("S " + "N " * 12)
    pol = ReservoirEvictionPolicy(budget=6, recent=2)
    kept = pol.retained_indices(tags)
    assert len(kept) == 6                  # exactly the budget, no more
    assert sorted(kept) == kept            # returned sorted ascending


def test_degrades_to_vanilla_sliding_window_without_reservoir_tags():
    # No reservoir tags: this is exactly StreamingLLM (first sink + recent window).
    tags = _tags("S " + "N " * 6)
    pol = ReservoirEvictionPolicy(budget=3, recent=2)
    kept = pol.retained_indices(tags)
    assert kept == [0, 5, 6]               # sink + 2 most-recent normals; middle all gone


def test_protected_entries_are_a_hard_floor_over_budget():
    # reservoir(3) + sink(1) + recent(2) = 6 protected, but budget is only 4.
    tags = _tags("S R R R N N N N")        # 1 sink, 3 reservoir, 4 normal
    pol = ReservoirEvictionPolicy(budget=4, recent=2)
    kept = set(pol.retained_indices(tags))
    # all protected survive even though that exceeds the budget; no extra middle normals added.
    assert {0, 1, 2, 3, 6, 7} <= kept
    assert pol.over_budget(tags) is True
    # only the two OLD normals (4,5) are evicted; nothing protected is touched.
    assert pol.evicted_indices(tags) == [4, 5]


def test_nothing_evicted_when_under_budget():
    tags = _tags("S R N N N")
    pol = ReservoirEvictionPolicy(budget=10, recent=2)
    assert pol.evicted_indices(tags) == []
    assert pol.retained_indices(tags) == [0, 1, 2, 3, 4]
    assert pol.over_budget(tags) is False


def test_retained_and_evicted_partition_the_sequence():
    tags = _tags("S R N N R N N N N N")
    pol = ReservoirEvictionPolicy(budget=5, recent=1)
    kept = set(pol.retained_indices(tags))
    dropped = set(pol.evicted_indices(tags))
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(range(len(tags)))


def test_gather_selects_positions_in_order():
    seq = ["a", "b", "c", "d", "e"]
    assert gather(seq, [0, 2, 4]) == ["a", "c", "e"]
    assert gather(seq, []) == []


def test_recent_zero_is_valid():
    # recent=0 means only sinks/reservoir are protected; all normals are evictable.
    tags = _tags("S N N N")
    pol = ReservoirEvictionPolicy(budget=2, recent=0)
    kept = pol.retained_indices(tags)
    assert 0 in kept                       # sink always kept
    assert len(kept) == 2                  # sink + 1 newest normal (the rest evicted)
    assert kept == [0, 3]


def test_invalid_tag_raises():
    pol = ReservoirEvictionPolicy(budget=4)
    with pytest.raises(ValueError):
        pol.retained_indices(["bogus"])


# --- importance-based (H2O heavy-hitter) eviction, with the reservoir still pinned ---

def test_importance_keeps_heavy_hitters_not_newest():
    # 1 sink (0) + normals 1..6; recent=1 protects idx 6; budget headroom = 2 middle normals.
    tags = _tags("S " + "N " * 6)
    # make an OLD normal a heavy hitter; recency alone would drop it.
    scores = [0.0, 0.9, 0.1, 0.2, 0.8, 0.3, 0.0]  # idx1 and idx4 are the heavy hitters
    pol = ReservoirEvictionPolicy(budget=4, recent=1)
    kept = pol.retained_indices(tags, scores=scores)
    # sink(0) + recent(6) pinned; the 2 highest-scoring evictable middles are idx1 (0.9) and idx4 (0.8)
    assert kept == [0, 1, 4, 6]
    assert pol.evicted_indices(tags, scores=scores) == [2, 3, 5]


def test_importance_never_evicts_reservoir_or_sink():
    tags = _tags("S R N N R N N N")
    scores = [0.0, 0.0, 0.1, 0.2, 0.0, 0.3, 0.4, 0.5]  # reservoir/sink scores are irrelevant
    pol = ReservoirEvictionPolicy(budget=5, recent=1)
    kept = set(pol.retained_indices(tags, scores=scores))
    assert {0, 1, 4} <= kept                       # sink + both reservoir entries always kept
    assert {i for i, t in enumerate(tags) if t == RESERVOIR} <= kept


def test_importance_ties_break_by_recency():
    tags = _tags("S " + "N " * 5)                  # sink 0, normals 1..5
    scores = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]        # all evictable normals tie
    pol = ReservoirEvictionPolicy(budget=3, recent=1)
    # recent=1 pins idx5; headroom 1 -> among tied {1,2,3,4} keep the newest (idx4).
    kept = pol.retained_indices(tags, scores=scores)
    assert kept == [0, 4, 5]


def test_scores_none_preserves_recency_behaviour():
    tags = _tags("S " + "N " * 8)
    pol = ReservoirEvictionPolicy(budget=5, recent=2)
    assert pol.retained_indices(tags, scores=None) == pol.retained_indices(tags)
    assert pol.retained_indices(tags) == [0, 5, 6, 7, 8]


def test_importance_respects_budget_and_partitions():
    tags = _tags("S R N N N N N N")
    scores = [0.0, 0.0, 0.7, 0.1, 0.9, 0.2, 0.4, 0.6]
    pol = ReservoirEvictionPolicy(budget=5, recent=1)
    kept = set(pol.retained_indices(tags, scores=scores))
    dropped = set(pol.evicted_indices(tags, scores=scores))
    assert len(kept) <= 5
    assert kept.isdisjoint(dropped)
    assert kept | dropped == set(range(len(tags)))
