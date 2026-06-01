"""Tests for the blank-cycle context-growth simulation (written before the implementation).

Demonstrates the Grok chat's concern made measurable: under repeated *blank ticks* a
Reservoir Agent's KV cache grows linearly without eviction, but the reservoir-protected
policy bounds it at the budget while never dropping a reservoir entry. Pure/CPU so it runs
in CI.
"""
import pytest

from reservoir.blank_cycle import simulate_blank_cycle


def test_vanilla_grows_linearly():
    recs = simulate_blank_cycle(20, tokens_per_tick=1, n_sink=4, n_reservoir=8,
                                budget=16, recent=4)
    base = 4 + 8  # sinks + reservoir present from tick 0
    for r in recs:
        assert r["vanilla_size"] == base + r["tick"] * 1


def test_protected_size_never_exceeds_budget():
    recs = simulate_blank_cycle(100, tokens_per_tick=2, n_sink=4, n_reservoir=8,
                                budget=32, recent=8)
    assert max(r["protected_size"] for r in recs) <= 32


def test_protected_size_plateaus():
    recs = simulate_blank_cycle(100, tokens_per_tick=1, n_sink=4, n_reservoir=8,
                                budget=24, recent=6)
    sizes = [r["protected_size"] for r in recs]
    # non-decreasing while filling, then flat at the cap
    assert all(b >= a for a, b in zip(sizes, sizes[1:]))
    assert sizes[-1] == 24
    assert sizes[-1] < recs[-1]["vanilla_size"]  # bounded well below the vanilla growth


def test_reservoir_signal_always_survives():
    recs = simulate_blank_cycle(200, tokens_per_tick=3, n_sink=2, n_reservoir=8,
                                budget=20, recent=4)
    # every reservoir entry is retained on every single tick, even under heavy eviction
    assert all(r["reservoir_retained"] == 8 for r in recs)
    assert all(r["reservoir_retained"] == r["reservoir_expected"] for r in recs)


def test_records_cover_all_ticks():
    recs = simulate_blank_cycle(7)
    assert [r["tick"] for r in recs] == [1, 2, 3, 4, 5, 6, 7]


def test_zero_ticks_is_empty():
    assert simulate_blank_cycle(0) == []
