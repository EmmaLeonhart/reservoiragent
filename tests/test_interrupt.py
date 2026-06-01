"""Tests for the interruptibility experiment (written before the implementation).

The Grok chat argued a Reservoir Agent can register an urgent human "STOP" with lower
latency than a turn-based harness agent (which is "deep in a long generation" and only reads
input at turn boundaries), and that because the reservoir continuously evolves, a *one-shot*
urgent burst persists in its state for several passes — so a per-tick monitor has a window to
catch it even if the user does not repeat themselves. Both parts are made measurable on CPU.
"""
import numpy as np
import pytest

from reservoir.interrupt import (
    poll_latency_stats,
    reservoir_persistence,
    run_interrupt_experiment,
)


def test_per_tick_poller_has_zero_latency():
    stats = poll_latency_stats(horizon=100, period=1)
    assert stats["max"] == 0
    assert stats["mean"] == 0.0


def test_turn_based_poller_latency_matches_period():
    # a poller that only reads every `period` passes registers an arrival at the next
    # boundary; latency is uniform on 0..period-1, so mean (period-1)/2, max period-1.
    stats = poll_latency_stats(horizon=8000, period=8)
    assert stats["max"] == 7
    assert stats["mean"] == pytest.approx(3.5, abs=0.05)


def test_reservoir_signal_persists_beyond_arrival():
    res = reservoir_persistence(n_reservoir=200, n_input=16, t_star=20, horizon=60,
                                urgent_scale=6.0, seed=0)
    # the reservoir monitor stays above threshold for at least 2 passes AFTER the burst...
    assert res["reservoir_window"] >= 2
    # ...while the stateless monitor (sees only the current input) catches it only on the
    # exact arrival pass — zero passes of post-arrival coverage.
    assert res["stateless_window"] == 0


def test_reservoir_score_peaks_at_arrival_then_decays():
    res = reservoir_persistence(n_reservoir=200, n_input=16, t_star=20, horizon=60,
                                urgent_scale=6.0, seed=0)
    scores = np.array(res["reservoir_score"])
    t = res["t_star"]
    assert scores[t] == pytest.approx(max(scores), rel=1e-6)   # arrival is the peak
    assert scores[t + 1] < scores[t]                            # then it decays
    assert scores[t + 1] > res["threshold"]                    # but is still detectable


def test_quiet_baseline_stays_below_threshold_before_arrival():
    res = reservoir_persistence(n_reservoir=200, n_input=16, t_star=30, horizon=60,
                                urgent_scale=6.0, seed=1)
    scores = np.array(res["reservoir_score"])
    # nothing urgent has happened yet, so the monitor should not false-fire (allow the
    # very first warm-up step slack by checking passes 2..t_star-1).
    assert np.all(scores[2:res["t_star"]] < res["threshold"])


def test_run_interrupt_experiment_summary_shape():
    out = run_interrupt_experiment(period=8, n_reservoir=150, n_input=16,
                                   t_star=20, horizon=60, urgent_scale=6.0, seed=0)
    assert set(out["summary"]) >= {
        "turn_based_mean_latency", "turn_based_max_latency", "per_tick_latency",
        "reservoir_window", "stateless_window", "burst_missed_by_turn_based",
    }
    # a single non-repeated burst: the turn-based+stateless agent can miss it entirely when
    # its poll period outruns the reservoir persistence window; the per-tick agent cannot.
    assert out["summary"]["per_tick_latency"] == 0
    assert isinstance(out["summary"]["burst_missed_by_turn_based"], bool)
