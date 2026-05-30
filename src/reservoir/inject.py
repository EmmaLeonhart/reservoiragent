"""Reservoir injection into a pretrained transformer (model surgery).

Hooks a mid-depth block of a pretrained GPT-2 so that, on every forward pass, the
block's hidden states drive a fixed reservoir (read via the reservoir's fixed
``W_in``) and the reservoir state is written back into the residual stream through a
readout ``W_out``:

    h'(x) = h(x) + W_out @ r(t),   r(t) = reservoir.step( summary(h(x)) )

This is the feasibility-scoped injection: writing the reservoir state into the
residual stream (rather than appending reservoir nodes to the attention key/value
sequence — the richer variant, left for the ambitious reach). It gives the property
the H1 regression checks: **with ``W_out = 0`` the model is byte-for-byte the base
model** (graceful degradation), while a nonzero ``W_out`` genuinely perturbs the
output — proving the injection point is live and the reservoir influences upper
layers. The reservoir state persists across forward passes (a real time axis).

Requires the optional ``models`` extra (torch + transformers).
"""
from __future__ import annotations

import numpy as np

from .echo_state import EchoStateReservoir


class ReservoirInjectedLM:
    """A pretrained GPT-2 with a fixed reservoir injected at a mid-depth block.

    Parameters
    ----------
    model_name : str
        HF model id (default "gpt2"; tests use a tiny one).
    layer : int | None
        Block index to inject at; defaults to the middle block.
    n_reservoir : int
        Reservoir size K.
    seed : int
        Seeds the reservoir and the readout.
    """

    def __init__(self, model_name: str = "gpt2", *, layer: int | None = None,
                 n_reservoir: int = 256, seed: int = 0, device: str | None = None,
                 load_in_4bit: bool = False, **reservoir_kwargs):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if load_in_4bit:
            # 4-bit base (bitsandbytes) so a multi-billion-parameter model (Hermes 3B)
            # fits the GPU. The injection still adds a full-precision readout on top.
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map={"": 0}).eval()
            self.device = "cuda"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()

        from ._arch import decoder_blocks, hidden_size
        self.blocks = decoder_blocks(self.model)
        n_layers = len(self.blocks)
        self.layer = n_layers // 2 if layer is None else int(layer)
        self.d_model = hidden_size(self.model.config)
        self.n_reservoir = int(n_reservoir)

        self.reservoir = EchoStateReservoir(self.n_reservoir, self.d_model,
                                            seed=seed, **reservoir_kwargs)
        # readout: (d_model, K). Zero by default => injection is a no-op (H1).
        self.W_out = np.zeros((self.d_model, self.n_reservoir), dtype=np.float64)
        self._hook = None
        self._register()

    # -- injection hook ---------------------------------------------------------
    def _register(self):
        torch = self.torch

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # summary of the block output drives the reservoir (mean over sequence;
            # batch size 1 in this feasibility setup).
            summary = hidden.detach().to("cpu", dtype=torch.float64).numpy()[0].mean(axis=0)
            r = self.reservoir.step(summary)               # (K,)
            add = self.W_out @ r                           # (d_model,)
            add_t = torch.as_tensor(add, dtype=hidden.dtype, device=hidden.device)
            hidden = hidden + add_t                        # broadcast over [B, S, D]
            if isinstance(output, tuple):
                return (hidden,) + tuple(output[1:])
            return hidden

        self._hook = self.blocks[self.layer].register_forward_hook(hook)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    # -- API --------------------------------------------------------------------
    def reset_reservoir(self):
        self.reservoir.reset()

    def set_readout(self, W_out):
        W_out = np.asarray(W_out, dtype=np.float64)
        if W_out.shape != (self.d_model, self.n_reservoir):
            raise ValueError(f"W_out must be ({self.d_model}, {self.n_reservoir})")
        self.W_out = W_out

    def logits(self, text: str):
        """Return next-token logits for ``text`` (a torch tensor on ``device``)."""
        torch = self.torch
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**ids)
        return out.logits[0, -1]

    def reservoir_state(self) -> np.ndarray:
        return self.reservoir.state.copy()


def extract_layer_stream(model_name: str, text: str, *, layer: int | None = None,
                         device: str | None = None) -> np.ndarray:
    """Return the per-token hidden-state stream at a mid-depth block of a pretrained
    transformer, shape ``(T, d_model)`` — the real activation stream the reservoir
    sees at the injection layer. Used to drive the sweep on real (not synthetic) input.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    from ._arch import decoder_blocks
    n_layers = len(decoder_blocks(model))
    layer = n_layers // 2 if layer is None else int(layer)
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    # hidden_states is (embeddings, block_0_out, ..., block_{n-1}_out); block `layer`
    # output is index layer+1.
    h = out.hidden_states[layer + 1][0]  # (T, d_model)
    return h.detach().to("cpu", dtype=torch.float64).numpy()


def base_logits(model_name: str, text: str, device: str | None = None):
    """Next-token logits from the unmodified base model (for regression checks)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    ids = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**ids)
    return out.logits[0, -1]
