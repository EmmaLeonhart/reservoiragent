"""Differentiable reservoir injection + LoRA fine-tuning (the compute-gated run).

`inject.py` ticks a *numpy* reservoir inside the forward hook — fine for inference and
the H1 check, but not differentiable. This module is the trainable version: a torch
reservoir (W_r, W_in fixed buffers; W_out a trainable Linear, zero-init) injected into
a GPT-2 block, so a real LoRA fine-tune can train W_out + upper-layer LoRA on the GPU.

IMPORTANT scope note, named plainly: the injection hook fires **once per forward pass**
(a transformer processes the whole sequence through each layer once), so the reservoir
ticks **per pass, not per token** — its distinctive value is the **cross-pass** time
axis. A single-forward LoRA fine-tune therefore exercises the *training pipeline on the
real architecture* (W_out + LoRA learn, loss decreases) but does NOT yet exercise the
reservoir's cross-pass value; that needs the multi-pass differentiable harness with
backprop-through-passes on a reservoir-requiring (cross-context) task — the next
compute step. With W_out zero-initialised the fine-tune starts exactly at the base
model (H1).

Requires the `models` extra (torch + transformers) and `peft`.
"""
from __future__ import annotations

import numpy as np


def _build_reservoir_weights(K, d_in, spectral_radius, input_scaling, sparsity, seed):
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-1, 1, (K, d_in)) * input_scaling
    mask = rng.random((K, K)) < sparsity
    W = rng.uniform(-1, 1, (K, K)) * mask
    rho = np.max(np.abs(np.linalg.eigvals(W)))
    if rho > 0:
        W = W * (spectral_radius / rho)
    return W.astype(np.float32), W_in.astype(np.float32)


class TorchReservoirInjectedLM:
    """GPT-2 with a differentiable reservoir injected at a mid-depth block, ready for a
    LoRA + W_out fine-tune. Use ``.model`` (a peft model) with a standard training loop.
    """

    def __init__(self, model_name: str = "gpt2", *, layer: int | None = None,
                 n_reservoir: int = 256, spectral_radius: float = 0.9,
                 input_scaling: float = 0.25, sparsity: float = 0.1, leak: float = 1.0,
                 seed: int = 0, device: str | None = None,
                 lora_r: int = 8, lora_alpha: int = 16):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(model_name)

        from ._arch import decoder_blocks, hidden_size
        d_model = hidden_size(base.config)
        n_layers = len(decoder_blocks(base))
        self.layer = n_layers // 2 if layer is None else int(layer)
        self.d_model, self.n_reservoir, self.leak = d_model, n_reservoir, float(leak)

        W_r, W_in = _build_reservoir_weights(n_reservoir, d_model, spectral_radius,
                                             input_scaling, sparsity, seed)
        # fixed reservoir weights as buffers (no grad); trainable readout W_out.
        self.W_r = torch.tensor(W_r)
        self.W_in = torch.tensor(W_in)
        self.W_out = nn.Linear(n_reservoir, d_model, bias=False)
        nn.init.zeros_(self.W_out.weight)          # H1: start exactly at the base model
        self._state = torch.zeros(n_reservoir)

        # LoRA on the upper-layer attention/MLP projections
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=["c_attn"],
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lcfg)
        self.model.to(self.device)
        self.W_out.to(self.device)
        self.W_r = self.W_r.to(self.device)
        self.W_in = self.W_in.to(self.device)
        self._state = self._state.to(self.device)

        self._register()

    def _register(self):
        torch = self.torch
        from ._arch import decoder_blocks
        block = decoder_blocks(self.model)[self.layer]

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            x = hidden.mean(dim=1).mean(dim=0).to(self.W_in.dtype)  # (d_model,)
            pre = self.W_r @ self._state + self.W_in @ x
            self._state = (1.0 - self.leak) * self._state + self.leak * torch.tanh(pre)
            add = self.W_out(self._state.to(self.W_out.weight.dtype))  # (d_model,)
            hidden = hidden + add
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        self._hook = block.register_forward_hook(hook)

    def reset_state(self):
        self._state = self.torch.zeros(self.n_reservoir, device=self.device)

    def trainable_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        params += list(self.W_out.parameters())
        return params


_CORPUS = [
    "The reservoir agent carries state across forward passes, giving it a time axis.",
    "A fixed random reservoir is injected into the transformer's mid-layer attention.",
    "Only the readout and a light LoRA adapter are trained; the reservoir stays fixed.",
    "Echo state networks train a readout while leaving the recurrent weights random.",
    "The agent can keep processing on an unprompted pass with no new input.",
    "Spectral radius near one places the reservoir at the edge of chaos.",
    "The context buffer is never wiped, and the reservoir state persists across passes.",
    "Standard transformers are stateless between calls; this one is not.",
]


def train_finetune(model_name: str = "gpt2", *, seeds=(0, 1, 2), steps: int = 60,
                   lr: float = 5e-4, device: str | None = None, max_len: int = 48,
                   corpus=None) -> list[dict]:
    """A real LoRA + W_out fine-tune on the GPU, per reservoir seed. Returns the start
    and end training loss per seed (the pipeline works iff loss decreases)."""
    import torch

    corpus = list(corpus) if corpus is not None else _CORPUS
    results = []
    for seed in seeds:
        lm = TorchReservoirInjectedLM(model_name, seed=int(seed), device=device)
        enc = lm.tokenizer(corpus, return_tensors="pt", padding=True, truncation=True,
                           max_length=max_len)
        ids = enc["input_ids"].to(lm.device)
        mask = enc["attention_mask"].to(lm.device)
        labels = ids.clone()
        labels[mask == 0] = -100
        opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)
        lm.model.train()
        losses = []
        for _ in range(steps):
            lm.reset_state()
            out = lm.model(input_ids=ids, attention_mask=mask, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            losses.append(float(out.loss.item()))
        n_train = int(sum(p.numel() for p in lm.trainable_parameters()))
        results.append({"seed": int(seed), "loss_start": losses[0],
                        "loss_end": losses[-1], "n_trainable_params": n_train,
                        "device": lm.device})
    return results
