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
    # the pass right after a trigger opens an unresolved thread -> "speak"
    ti = np.where(triggers)[0]
    assert any(labels[t + 1] for t in ti if t + 1 < 200)
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
    r = evaluate_silence_gate(K=400, T=6000, speak_window=5, seed=0)
    res = r["reservoir_gate"]
    base = r["stateless_gate"]
    # the reservoir gate sees the recent history -> a real speak/silent policy.
    assert res["f1"] > 0.9 and res["precision"] > 0.8
    # the stateless gate is blind to the *past* trigger -> it cannot be selectively
    # silent; it degenerates to (near) always-speak (recall ~ 1, precision ~ base rate),
    # so its F1 is far below the reservoir gate's.
    assert res["f1"] > base["f1"] + 0.3
