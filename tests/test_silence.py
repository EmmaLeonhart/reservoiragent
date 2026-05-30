"""Tests for the trained silence policy (numpy, runs in CI)."""
import numpy as np

from reservoir.silence import (
    make_silence_task,
    precision_recall_f1,
    evaluate_silence_gate,
)


def test_silence_task_labels_follow_triggers():
    u, labels, triggers = make_silence_task(200, speak_window=5, seed=0)
    assert u.shape == (200, 4)
    # every trigger opens a speak window; the trigger step itself is "speak"
    assert labels[triggers].all()
    # well before any trigger, the agent should be silent
    assert not labels[0]


def test_precision_recall_f1_known():
    yt = [True, True, False, False]
    yp = [True, False, False, True]
    m = precision_recall_f1(yt, yp)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
    perfect = precision_recall_f1(yt, yt)
    assert perfect["f1"] == 1.0


def test_reservoir_gate_beats_stateless_on_unresolved_thread():
    r = evaluate_silence_gate(K=300, T=4000, speak_window=5, seed=0)
    res_f1 = r["reservoir_gate"]["f1"]
    base_f1 = r["stateless_gate"]["f1"]
    # the reservoir gate sees the recent history -> high recall on the open thread;
    # the stateless gate sees only the current input -> it can only catch the trigger
    # pass itself, so it misses most of the window.
    assert res_f1 > 0.8
    assert r["reservoir_gate"]["recall"] > r["stateless_gate"]["recall"] + 0.3
    assert res_f1 > base_f1 + 0.2
