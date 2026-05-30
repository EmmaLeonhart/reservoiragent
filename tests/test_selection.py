"""Tests for N-seed selection + the pre-selection proxy (numpy, runs in CI)."""
from reservoir.selection import seed_report, select_seeds, proxy_correlation


def test_seed_report_has_fields():
    r = seed_report(0, K=80, max_delay=12)
    assert {"seed", "memory_capacity", "pr_frac", "saturation"} <= set(r)
    assert r["memory_capacity"] > 0.0


def test_select_seeds_ranks_best_first_and_seeds_differ():
    ranked = select_seeds([1, 2, 3, 4, 5], K=80, max_delay=12)
    assert len(ranked) == 5
    mcs = [r["memory_capacity"] for r in ranked]
    assert mcs == sorted(mcs, reverse=True)            # best first
    assert len({round(m, 4) for m in mcs}) > 1          # seeds genuinely differ


def test_proxy_correlation_in_range():
    ranked = select_seeds([1, 2, 3, 4, 5, 6], K=80, max_delay=12)
    c = proxy_correlation(ranked, "pr_frac")
    assert -1.0 <= c["spearman"] <= 1.0
    assert c["n"] == 6
