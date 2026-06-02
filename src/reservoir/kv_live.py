"""Content-addressable reservoir injection into a *live* model (the KV-append fix).

The additive-readout injection (`torch_inject.py`) failed at cross-pass content recall:
a single bias vector can't carry *which* word appeared, and the model learns to ignore
it. The fix is to let the model **attend** to the reservoir — reservoir nodes as extra
keys/values — so the read is content-addressable.

Wiring extra K/V *inside* transformers-5.4 attention is fragile (Cache + dispatch). The
robust equivalent used here: project the reservoir state into a handful of **prefix
pseudo-tokens** prepended to the sequence (via ``inputs_embeds``). Attention then reads
them content-addressably at every layer, the standard causal mask / positions handle the
extended length with no mid-layer surgery, and the prefix positions are stripped from the
output. (This attends from all layers, not only upper layers — a benign difference from
strict mid-layer KV-append.)

A read hook still ticks the reservoir from a mid-layer hidden each pass, so state
accumulates across passes; the prefix written at pass t is computed from the state
carried out of the previous passes. Requires the `models` extra + peft.
"""
from __future__ import annotations

import numpy as np

from .torch_inject import _build_reservoir_weights


class TorchReservoirPrefixInjectedLM:
    """A live causal LM that attends to reservoir-derived prefix tokens.

    Trainable: ``W_res`` (reservoir → prefix tokens) + LoRA. Fixed: the reservoir
    (W_r, W_in) and the base model.
    """

    def __init__(self, model_name: str = "gpt2", *, n_reservoir: int = 512,
                 n_prefix: int = 8, layer: int | None = None, spectral_radius: float = 0.9,
                 input_scaling: float = 0.5, sparsity: float = 0.1, leak: float = 1.0,
                 seed: int = 0, device: str | None = None, lora_r: int = 8,
                 lora_alpha: int = 16, summary: str = "last", load_in_4bit: bool = False,
                 dtype: str | None = None, train_seed: int | None = None):
        import torch
        import torch.nn as nn
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        from ._arch import decoder_blocks, hidden_size

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)
            base = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb, device_map={"": 0})
            # gradient checkpointing is OFF on purpose: it re-runs block forwards during
            # backward, which would re-fire the reservoir read hook and double-tick the
            # state. The cross-pass prompts are tiny, so full activations fit anyway.
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=False)
            self.device = "cuda"
        else:
            kw = {}
            if dtype in ("float16", "bfloat16"):
                kw["torch_dtype"] = getattr(torch, dtype)   # fp16/bf16 so a 3B base fits
            base = AutoModelForCausalLM.from_pretrained(model_name, **kw)

        d_model = hidden_size(base.config)
        self.d_model, self.n_reservoir, self.n_prefix = d_model, n_reservoir, n_prefix
        self.leak, self.summary = float(leak), summary
        n_layers = len(decoder_blocks(base))
        self.layer = n_layers // 2 if layer is None else int(layer)

        W_r, W_in = _build_reservoir_weights(n_reservoir, d_model, spectral_radius,
                                             input_scaling, sparsity, seed)
        self.W_r = torch.tensor(W_r)
        self.W_in = torch.tensor(W_in)
        self._state = torch.zeros(n_reservoir)
        # Seed the trainable init (W_res + the LoRA adapter built just below) so it is a
        # *controlled* variable in the N-seed selection experiment rather than uncontrolled
        # noise. Without this, two runs of the SAME fixed reservoir start from different inits,
        # which dominated the per-seed recall spread (devlog 2026-05-31). train_seed=None keeps
        # the previous (unseeded) behaviour.
        self.train_seed = train_seed
        if train_seed is not None:
            torch.manual_seed(int(train_seed))
        # reservoir -> n_prefix prefix-token embeddings
        self.W_res = nn.Linear(n_reservoir, n_prefix * d_model)

        target = ["c_attn"] if hasattr(base.config, "n_embd") else ["q_proj", "v_proj"]
        lcfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target,
                          lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lcfg)
        if not load_in_4bit:
            self.model.to(self.device)
        self.W_res.to(self.device)
        self.W_r = self.W_r.to(self.device)
        self.W_in = self.W_in.to(self.device)
        self._state = self._state.to(self.device)
        self.embed = self.model.get_input_embeddings()
        # everything needed to reconstruct this model on load (device is chosen at load
        # time, not saved). The reservoir (W_r, W_in) regenerates from seed, so only the
        # trained W_res + LoRA adapter need to be persisted alongside this config.
        self._init_config = dict(
            model_name=model_name, n_reservoir=n_reservoir, n_prefix=n_prefix,
            layer=self.layer, spectral_radius=spectral_radius, input_scaling=input_scaling,
            sparsity=sparsity, leak=leak, seed=seed, lora_r=lora_r, lora_alpha=lora_alpha,
            summary=summary)
        self._register_read_hook()

    def _register_read_hook(self):
        torch = self.torch
        from ._arch import decoder_blocks
        block = decoder_blocks(self.model)[self.layer]

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.summary == "last":
                x = hidden[:, -1, :].mean(dim=0).to(self.W_in.dtype)
            else:
                x = hidden.mean(dim=1).mean(dim=0).to(self.W_in.dtype)
            pre = self.W_r @ self._state + self.W_in @ x
            self._state = (1.0 - self.leak) * self._state + self.leak * torch.tanh(pre)
            return output

        self._hook = block.register_forward_hook(hook)

    def reset_state(self):
        self._state = self.torch.zeros(self.n_reservoir, device=self.device)

    def forward_logits(self, input_ids, attention_mask):
        """Forward with the reservoir prefix prepended; returns logits over the real
        (non-prefix) token positions."""
        torch = self.torch
        B, T = input_ids.shape
        tok_emb = self.embed(input_ids)                              # (B, T, d)
        prefix = self.W_res(self._state.to(self.W_res.weight.dtype))
        prefix = prefix.view(self.n_prefix, self.d_model).unsqueeze(0).expand(B, -1, -1)
        inputs_embeds = torch.cat([prefix.to(tok_emb.dtype), tok_emb], dim=1)
        ext_mask = torch.cat(
            [torch.ones(B, self.n_prefix, device=self.device, dtype=attention_mask.dtype),
             attention_mask], dim=1)
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=ext_mask)
        return out.logits[:, self.n_prefix:, :]                     # strip the prefix

    def trainable_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        params += list(self.W_res.parameters())
        return params
