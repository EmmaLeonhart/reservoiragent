"""Smoke test for the cross-pass recall pipeline (torch+peft; local, skips in CI).

Verifies the multi-pass differentiable loop runs end-to-end and returns a valid result.
The scientific claim (stateful recall beats the stateless baseline) is measured by the
real GPU run in `scripts/run.py crosspass`, not this tiny smoke test.
"""
import pytest


def test_cross_pass_pipeline_runs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass

    r = run_cross_pass("sshleifer/tiny-gpt2", n_keys=4, steps=6, lr=1e-2,
                       device="cpu", stateful=True)
    assert 0.0 <= r["recall_accuracy"] <= 1.0
    assert r["n_keys"] == 4
    assert "loss_end" in r


def test_cross_pass_kv_pipeline_runs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass_kv

    # the content-addressable (KV-prefix) path must run end-to-end too
    r = run_cross_pass_kv("sshleifer/tiny-gpt2", n_keys=4, steps=4, lr=1e-2,
                          device="cpu", stateful=True, n_prefix=4)
    assert r["mode"] == "kv-prefix"
    assert 0.0 <= r["recall_accuracy"] <= 1.0


import numpy as np
from reservoir.crosspass import eval_recall, recall_accuracy


class _FakeEnc(dict):
    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, return_tensors=None):
        return _FakeEnc(input_ids=text, attention_mask=None)


class _FakeReservoirLM:
    """Simulates a stateful kv reservoir: remembers the word from pass 1 and predicts it
    in pass 2 — unless the state was wiped between the passes (the baseline)."""

    def __init__(self, vocab):
        self.tokenizer = _FakeTok()
        self.device = "cpu"
        self.vocab = vocab            # stripped word -> tok_id
        self._mem = None

    def reset_state(self):
        self._mem = None

    def forward_logits(self, input_ids, attention_mask):
        text = input_ids
        prefix = "The secret word is "
        if text.startswith(prefix):
            self._mem = text[len(prefix):].rstrip(".")
        V = max(self.vocab.values()) + 2
        logits = np.full((1, 1, V), -1.0)
        idx = self.vocab.get(self._mem, V - 1)   # no memory -> sentinel id (a miss)
        logits[0, -1, idx] = 1.0
        return logits


def test_eval_recall_stateful_hits_baseline_misses():
    vocab = {"red": 1, "blue": 2, "green": 3}
    keys = [("red", 1), ("blue", 2), ("green", 3)]
    recs = eval_recall(_FakeReservoirLM(vocab), keys)
    assert all(r["stateful_ok"] for r in recs)
    assert not any(r["baseline_ok"] for r in recs)
    assert [r["word"] for r in recs] == ["red", "blue", "green"]
    assert recall_accuracy(recs, "stateful") == 1.0
    assert recall_accuracy(recs, "baseline") == 0.0


def test_recall_accuracy_empty_is_zero():
    assert recall_accuracy([], "stateful") == 0.0


def test_kv_forward_pair_curriculum_hint_controls_pass2_text():
    """The curriculum scaffold: hint=True leaves the key visible in pass 2; hint=False
    (the true cross-pass task, used at eval) does not."""
    from reservoir.crosspass import _kv_forward_pair

    seen = []

    class _RecLM:
        tokenizer = _FakeTok()
        device = "cpu"

        def reset_state(self):
            pass

        def forward_logits(self, input_ids, attention_mask):
            seen.append(input_ids)            # _FakeTok makes input_ids == the raw text
            return np.full((1, 1, 4), -1.0)

    lm = _RecLM()
    _kv_forward_pair(lm, "red", reset_between=False, hint=False)
    assert seen[-1] == "The secret word was"          # key not in pass 2 (hard task)

    seen.clear()
    _kv_forward_pair(lm, "red", reset_between=False, hint=True)
    assert "red" in seen[-1] and seen[-1].endswith("The secret word was")  # key visible
