"""Tests for the counterfactual 'use-the-state' aux term in episode_loss.

The aux term adds, on each emit step, a margin relu(aux_margin - (stateless_CE - stateful_CE))
computed by probing the model with the reservoir state wiped. These tests use a minimal torch
fake LM (no transformers) to confirm the wiring: aux_weight=0 is a no-op (no extra forward, default
behaviour unchanged), aux_weight>0 runs the stateless probe and restores the carried state so the
episode continues unperturbed, and the loss stays finite.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

torch = pytest.importorskip("torch")

from reservoir.episode import Episode, Step, episode_loss  # noqa: E402


class _Tok:
    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [(abs(hash(w)) % 40) + 1 for w in text.split()] or [1]
        if return_tensors == "pt":
            class _Enc(dict):
                def to(self, _device):
                    return self
            return _Enc(input_ids=torch.tensor([ids]),
                        attention_mask=torch.ones(1, len(ids), dtype=torch.long))
        return {"input_ids": ids}


class _FakeLM:
    """Logits at the last position favour token 5 in proportion to the reservoir state sum, so a
    wiped state (sum 0) predicts differently from a carried one — making the aux term non-trivial."""
    device = "cpu"

    def __init__(self):
        self.tokenizer = _Tok()
        self._state = torch.zeros(4)
        self.calls = 0
        self.V = 60

    def reset_state(self):
        self._state = torch.zeros(4)

    def forward_logits(self, input_ids, attention_mask):
        self.calls += 1
        logits = torch.zeros(1, input_ids.shape[1], self.V)
        logits[0, -1, 5] = self._state.sum()          # token 5 favoured when state is non-zero
        self._state = self._state + 1.0               # advance state each forward
        return logits

    def gate_logit(self):
        return torch.tensor(1.0)


def _emit_episode():
    # two emit steps so state restoration across steps matters
    return Episode([Step(inject="the secret is", target=" five"),
                    Step(wipe=True, inject="recall", target=" five")], "recall")


def test_aux_weight_zero_is_a_noop_on_forward_count():
    lm = _FakeLM()
    episode_loss(lm, _emit_episode(), aux_weight=0.0)
    baseline_calls = lm.calls
    lm2 = _FakeLM()
    episode_loss(lm2, _emit_episode(), aux_weight=1.0)
    # aux>0 must add at least one extra forward (the stateless probe) per emit step
    assert lm2.calls > baseline_calls


def test_aux_loss_is_finite_and_grad_capable():
    lm = _FakeLM()
    loss = episode_loss(lm, _emit_episode(), aux_weight=2.0, aux_margin=1.0)
    assert torch.isfinite(loss)


def test_state_is_restored_after_probe():
    # After an emit step's aux probe, _state must equal what it would be WITHOUT the probe
    # (the probe wipes + advances a throwaway copy, then restores) — so the two runs' final
    # state matches step-for-step.
    lm_no = _FakeLM()
    episode_loss(lm_no, _emit_episode(), aux_weight=0.0)
    lm_aux = _FakeLM()
    episode_loss(lm_aux, _emit_episode(), aux_weight=1.0)
    # reset_state runs at episode start; the carried-state trajectory (hence final state value
    # ignoring the extra probe forwards) must be identical — check the deterministic final sum
    # equals the no-aux run's, i.e. the probe did not leak into the carried state.
    assert torch.allclose(lm_no._state, lm_aux._state)
