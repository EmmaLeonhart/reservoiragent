"""Persist and reload a trained reservoir model.

The experiments that produced the cross-pass result were written as *measurement*
functions that returned a metrics dict and discarded the trained model. This module is
the missing save/load layer so a trained reservoir can become a loadable, shippable
artifact (e.g. uploaded to the Hugging Face Hub).

What is saved (a directory):
- ``config.json``        — the constructor kwargs to rebuild the architecture. The
                           reservoir (``W_r``, ``W_in``) is fixed-random from ``seed``, so
                           it regenerates byte-identically and is NOT stored.
- ``reservoir_readout.npz`` — the trained ``W_res`` (reservoir -> prefix tokens) weights.
- ``adapter/``           — the trained LoRA adapter (peft ``save_pretrained``).

The config + array layers are pure numpy/json and unit-tested in CI; the full
``save_reservoir_model`` / ``load_reservoir_model`` glue needs the ``models`` extra
(torch + transformers + peft) and is exercised locally.
"""
from __future__ import annotations

import json
import os

import numpy as np

CONFIG_NAME = "config.json"


# --- pure serialization (CI-testable, numpy/json only) ----------------------------

def save_model_config(out_dir, config: dict) -> str:
    """Write the architecture config as plain, human-readable JSON. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def load_model_config(out_dir) -> dict:
    """Read the architecture config written by :func:`save_model_config`."""
    path = os.path.join(out_dir, CONFIG_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {CONFIG_NAME} in {out_dir}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_array_dict(out_dir, name: str, arrays: dict) -> str:
    """Save a name->ndarray mapping to ``<name>.npz``. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def load_array_dict(out_dir, name: str) -> dict:
    """Load the mapping saved by :func:`save_array_dict`."""
    path = os.path.join(out_dir, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no {name}.npz in {out_dir}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# --- full model glue (torch + transformers + peft; run locally) --------------------

def save_reservoir_model(out_dir, lm, *, extra_meta: dict | None = None) -> str:
    """Persist a trained ``TorchReservoirPrefixInjectedLM`` to ``out_dir``.

    Saves the reconstruction config, the trained ``W_res`` readout, and the LoRA adapter.
    """
    os.makedirs(out_dir, exist_ok=True)
    config = dict(lm._init_config)
    config["arch"] = "TorchReservoirPrefixInjectedLM"
    if extra_meta:
        config["meta"] = extra_meta
    save_model_config(out_dir, config)

    readout = {k: v.detach().cpu().numpy() for k, v in lm.W_res.state_dict().items()}
    save_array_dict(out_dir, "reservoir_readout", readout)

    lm.model.save_pretrained(os.path.join(out_dir, "adapter"))   # peft LoRA adapter
    return out_dir


def load_reservoir_model(out_dir, *, device: str | None = None):
    """Rebuild a ``TorchReservoirPrefixInjectedLM`` from a saved directory.

    Reconstructs the architecture from ``config.json`` (regenerating the fixed reservoir
    from ``seed``), then loads the trained ``W_res`` readout and the LoRA adapter weights.
    """
    import torch
    from peft import PeftModel
    from .kv_live import TorchReservoirPrefixInjectedLM

    config = load_model_config(out_dir)
    kwargs = {k: v for k, v in config.items() if k not in ("arch", "meta")}
    lm = TorchReservoirPrefixInjectedLM(device=device, **kwargs)

    readout = load_array_dict(out_dir, "reservoir_readout")
    lm.W_res.load_state_dict(
        {k: torch.tensor(v, device=lm.device) for k, v in readout.items()})

    # peft wraps a fresh adapter at construction; load the trained adapter weights over it
    adapter_dir = os.path.join(out_dir, "adapter")
    base = lm.model.get_base_model()
    lm.model = PeftModel.from_pretrained(base, adapter_dir)
    lm.model.to(lm.device)
    lm._register_read_hook()
    return lm
