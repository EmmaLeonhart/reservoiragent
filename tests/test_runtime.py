"""Tests for the always-alive runtime.

The pure logic (entropy gate, context buffer, state checkpoint) runs in CI. The
AliveAgent integration needs torch + transformers and runs locally.
"""
import numpy as np
import pytest

from reservoir.runtime import (
    ContextBuffer,
    topk_entropy,
    ConfidenceGate,
    checkpoint_state,
    restore_state,
)


def test_topk_entropy_bounds():
    uniform = np.zeros(100)                      # all-equal logits -> max entropy
    assert topk_entropy(uniform, k=50) == pytest.approx(1.0, abs=1e-6)
    onehot = np.full(100, -50.0); onehot[3] = 50.0   # one dominant token
    assert topk_entropy(onehot, k=50) < 1e-3
    # sharper distribution -> lower entropy
    soft = np.linspace(0, 1, 100)
    sharp = np.linspace(0, 10, 100)
    assert topk_entropy(sharp, k=50) < topk_entropy(soft, k=50)


def test_confidence_gate_emits_when_confident():
    gate = ConfidenceGate(threshold=0.85, k=50)
    onehot = np.full(100, -50.0); onehot[0] = 50.0
    emit, h = gate.decide(onehot)
    assert emit and h < 0.85
    emit2, h2 = gate.decide(np.zeros(100))       # uniform -> uncertain -> silent
    assert not emit2 and h2 > 0.85


def test_context_buffer_never_loses_turns():
    cb = ContextBuffer()
    cb.append("user", "hello")
    cb.append("agent", "hi")
    cb.append("user", "still there?")
    assert len(cb) == 3
    assert "user: hello" in cb.render()
    assert cb.render().count("\n") == 2


def test_state_checkpoint_round_trip(tmp_path):
    state = np.random.default_rng(0).standard_normal(128)
    p = str(tmp_path / "state.npy")
    checkpoint_state(state, p)
    assert np.array_equal(restore_state(p), state)


# --- model integration (local; skips in the light CI job) ---------------------

def _agent():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from reservoir.runtime import AliveAgent
    return AliveAgent("sshleifer/tiny-gpt2", n_reservoir=64, seed=0, device="cpu")


def test_unprompted_pass_updates_state_without_new_input():
    agent = _agent()
    agent.prompt("Two tasks: water the plants and call the bank.")
    before = agent.lm.reservoir_state().copy()
    rec = agent.idle()                            # no new input
    after = agent.lm.reservoir_state()
    assert rec["kind"] == "unprompted"
    assert not np.allclose(before, after)         # the reservoir ticked anyway


def test_agent_checkpoint_restore(tmp_path):
    agent = _agent()
    agent.prompt("hello")
    p = str(tmp_path / "res.npy")
    agent.checkpoint(p)
    saved = agent.lm.reservoir_state().copy()
    agent.idle(); agent.idle()                    # mutate state
    assert not np.allclose(agent.lm.reservoir_state(), saved)
    agent.restore(p)
    assert np.allclose(agent.lm.reservoir_state(), saved)
