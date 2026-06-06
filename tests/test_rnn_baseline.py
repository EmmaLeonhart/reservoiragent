"""Test the trained-GRU cross-pass baseline (torch; local, skips in CI without torch).

The scientific point (a trained GRU solves cross-pass recall, situating the task difficulty)
is the real run in `scripts/run.py rnn-baseline`; this checks the pipeline and the ablation
property — carrying the hidden state must do at least as well as resetting it.
"""
import pytest


def test_rnn_baseline_stateful_beats_or_matches_stateless():
    pytest.importorskip("torch")
    from reservoir.rnn_baseline import run_rnn_baseline

    s = run_rnn_baseline(n_keys=4, steps=300, hidden_size=32, stateful=True, device="cpu")
    b = run_rnn_baseline(n_keys=4, steps=300, hidden_size=32, stateful=False, device="cpu")
    assert s["mode"] == "rnn-baseline"
    assert 0.0 <= s["recall_accuracy"] <= 1.0 and 0.0 <= b["recall_accuracy"] <= 1.0
    # the carried hidden state is the only bridge to the wiped key — it cannot do *worse*
    # than the reset control, and on this trivial task a trained GRU should reach perfect recall
    assert s["recall_accuracy"] >= b["recall_accuracy"]
    assert s["recall_accuracy"] == 1.0
