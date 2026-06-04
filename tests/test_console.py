"""Tests for the installer console's pure logic (which directory to load).

A published repo may be a single model (config.json at the root) or a batch population
(seed_*/ + batch_manifest.json). The console must resolve a batch repo to its recommended
seed subdir. The model loading + REPL themselves are torch-gated and not tested here.
"""
import json

import pytest

from reservoir.installer.console import resolve_load_dir


def test_single_model_dir_resolves_to_itself(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_load_dir(str(tmp_path)) == str(tmp_path)


def test_batch_dir_resolves_to_recommended_seed(tmp_path):
    (tmp_path / "batch_manifest.json").write_text(json.dumps(
        {"best": {"seed": 3}, "population": [{"seed": 3, "recommended": True}]}))
    (tmp_path / "seed_3").mkdir()
    (tmp_path / "seed_3" / "config.json").write_text("{}")
    assert resolve_load_dir(str(tmp_path)) == str(tmp_path / "seed_3")


def test_batch_dir_missing_best_seed_raises(tmp_path):
    (tmp_path / "batch_manifest.json").write_text(json.dumps({"best": {"seed": 9}}))
    # the recommended seed_9 dir doesn't exist -> a clear error, not a silent wrong load
    with pytest.raises(FileNotFoundError):
        resolve_load_dir(str(tmp_path))


def test_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_load_dir(str(tmp_path))


import numpy as np

from reservoir.installer.console import recall_demo_session


class _FakeEnc(dict):
    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 999

    def __call__(self, text, return_tensors=None):
        return _FakeEnc(input_ids=text, attention_mask=None)


class _FakeReservoirLM:
    """Stateful kv reservoir stub: recalls pass-1 word in pass 2 unless state is wiped."""

    def __init__(self, vocab):
        self.tokenizer = _FakeTok()
        self.device = "cpu"
        self.vocab = vocab
        self._mem = None

    def reset_state(self):
        self._mem = None

    def forward_logits(self, input_ids, attention_mask):
        prefix = "The secret word is "
        if input_ids.startswith(prefix):
            self._mem = input_ids[len(prefix):].rstrip(".")
        V = max(self.vocab.values()) + 2
        logits = np.full((1, 1, V), -1.0)
        logits[0, -1, self.vocab.get(self._mem, V - 1)] = 1.0
        return logits


def test_recall_demo_session_prints_table_and_summary():
    keys = [("red", 1), ("blue", 2)]
    lines = []
    recs = recall_demo_session(_FakeReservoirLM({"red": 1, "blue": 2}), keys,
                               print_fn=lines.append)
    text = "\n".join(lines)
    assert "red" in text and "blue" in text
    assert "recall accuracy" in text
    assert "✓" in text          # at least one stateful hit marker (✓)
    assert "✗" in text          # at least one baseline miss marker (✗)
    assert len(recs) == 2
