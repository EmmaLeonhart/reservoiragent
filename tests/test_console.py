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
