"""Tests for the always-alive engine's pure loop logic (drain, gate, telemetry).

The token streaming (`stream_emit`) is torch-gated and exercised by the live app, not
here. These tests drive a fake LM so they need neither torch nor a model.
"""
import numpy as np

from reservoir.alive import AliveEngine
from reservoir.runtime import ConfidenceGate


class _Enc(dict):
    def __init__(self):
        super().__init__(input_ids=[[1, 2, 3]], attention_mask=[[1, 1, 1]])

    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors=None):
        return _Enc()


class _FakeLM:
    """Returns a fixed logits vector each pass and advances its reservoir state so the
    state-direction telemetry changes."""

    def __init__(self, logits):
        self.tokenizer = _FakeTok()
        self.device = "cpu"
        self.readout_scale = 1.0
        self._logits = np.asarray(logits, dtype=float)   # shape (1, 1, V)
        self._n = 0
        self._state = np.zeros(4)

    def reset_state(self):
        self._state = np.zeros(4)

    def forward_logits(self, input_ids, attention_mask):
        self._n += 1
        self._state = np.array([float(self._n), 1.0, 0.0, 0.0])
        return self._logits


def _peaked(v=5):
    a = np.full((1, 1, v), -10.0)
    a[0, 0, 2] = 10.0           # one dominant token -> low entropy -> confident
    return a


def _flat(v=5):
    return np.zeros((1, 1, v))  # uniform -> high entropy -> uncertain


def test_inject_yields_a_prompted_pass_and_enters_context():
    eng = AliveEngine(_FakeLM(_peaked()), system=None)
    eng.inject("hello there")
    info = eng.gating_pass()
    assert info["kind"] == "prompted"
    assert any(t["role"] == "user" and t["content"] == "hello there" for t in eng.turns)


def test_no_inject_is_an_idle_pass():
    eng = AliveEngine(_FakeLM(_peaked()), system=None)
    info = eng.gating_pass()
    assert info["kind"] == "idle"


def test_gate_emits_when_confident():
    eng = AliveEngine(_FakeLM(_peaked()), system=None,
                      gate=ConfidenceGate(threshold=0.85))
    assert eng.gating_pass()["emit"] is True


def test_gate_silent_when_uncertain():
    eng = AliveEngine(_FakeLM(_flat()), system=None,
                      gate=ConfidenceGate(threshold=0.85))
    assert eng.gating_pass()["emit"] is False


def test_tick_count_and_state_cos_are_reported():
    eng = AliveEngine(_FakeLM(_peaked()), system=None)
    a = eng.gating_pass()
    b = eng.gating_pass()
    assert a["tick"] == 1 and b["tick"] == 2
    assert -1.0 <= b["state_cos"] <= 1.0


def test_set_readout_scale_is_live_and_reported():
    eng = AliveEngine(_FakeLM(_peaked()), system=None)
    eng.set_readout_scale(0.25)
    assert eng.lm.readout_scale == 0.25
    assert eng.gating_pass()["readout_scale"] == 0.25
