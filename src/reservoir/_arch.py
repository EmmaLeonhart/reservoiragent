"""Architecture helpers so the injection works across decoder families.

GPT-2 keeps its decoder blocks at ``model.transformer.h`` and its width at
``config.n_embd``; Llama/Mistral/Qwen (and therefore Hermes, which is Llama-based) keep
blocks at ``model.model.layers`` and width at ``config.hidden_size``. peft wraps the
model one level deeper (``model.base_model.model``). These helpers locate the right
objects regardless, so one injection code path covers GPT-2 and Hermes.
"""
from __future__ import annotations


def _inner_model(model):
    """Unwrap a peft-wrapped model to the underlying HF model, if needed."""
    base = getattr(model, "base_model", None)
    if base is not None and hasattr(base, "model"):
        return base.model
    return model


def decoder_blocks(model):
    """Return the list/ModuleList of decoder blocks for a causal-LM model."""
    m = _inner_model(model)
    # GPT-2 / GPT-Neo family
    tr = getattr(m, "transformer", None)
    if tr is not None and hasattr(tr, "h"):
        return tr.h
    # Llama / Mistral / Qwen family (Hermes is Llama-based)
    inner = getattr(m, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    # some models expose .layers directly
    if hasattr(m, "layers"):
        return m.layers
    raise ValueError(
        f"could not locate decoder blocks on {type(m).__name__}; "
        "supported: GPT-2 (transformer.h) and Llama-family (model.layers)"
    )


def hidden_size(config) -> int:
    """Model width across config conventions (Llama hidden_size; GPT-2 n_embd)."""
    for attr in ("hidden_size", "n_embd", "d_model"):
        v = getattr(config, attr, None)
        if v is not None:
            return int(v)
    raise ValueError("could not determine hidden size from config")
