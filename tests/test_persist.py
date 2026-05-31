"""Tests for reservoir-model persistence (the pure serialization layer).

The config (JSON) and the trained-array (npz) roundtrips are pure numpy/json and run in
CI. The full save_reservoir_model / load_reservoir_model glue is torch+models gated and
exercised locally (see scripts/run.py crosspass --save), not here.
"""
import numpy as np
import pytest

from reservoir.persist import (
    save_model_config, load_model_config, save_array_dict, load_array_dict,
)


def test_config_roundtrip(tmp_path):
    cfg = {"model_name": "gpt2", "n_reservoir": 512, "n_prefix": 8, "layer": 6,
           "spectral_radius": 0.9, "input_scaling": 0.5, "sparsity": 0.1, "leak": 1.0,
           "seed": 0, "lora_r": 8, "lora_alpha": 16, "summary": "last"}
    save_model_config(tmp_path, cfg)
    assert load_model_config(tmp_path) == cfg


def test_config_is_plain_json(tmp_path):
    # config.json must be human-readable plain JSON (it doubles as documentation)
    save_model_config(tmp_path, {"model_name": "gpt2", "seed": 0})
    import json
    with open(tmp_path / "config.json") as f:
        assert json.load(f)["model_name"] == "gpt2"


def test_array_dict_roundtrip(tmp_path):
    arrays = {"W_res.weight": np.random.default_rng(0).standard_normal((64, 512)),
              "W_res.bias": np.zeros(64)}
    save_array_dict(tmp_path, "reservoir_readout", arrays)
    loaded = load_array_dict(tmp_path, "reservoir_readout")
    assert set(loaded) == set(arrays)
    for k in arrays:
        np.testing.assert_allclose(loaded[k], arrays[k])


def test_array_dict_preserves_dtype_and_shape(tmp_path):
    arrays = {"a": np.ones((3, 4), dtype=np.float32)}
    save_array_dict(tmp_path, "x", arrays)
    loaded = load_array_dict(tmp_path, "x")
    assert loaded["a"].shape == (3, 4)
    assert loaded["a"].dtype == np.float32


def test_load_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_config(tmp_path)
