"""C: cross-pass training — make the reservoir's statefulness actually *do* something.

The load-bearing experiment. A task that a stateless model **structurally cannot** do:

- **Pass 1** shows a secret word: "The secret word is <KEY>."  (the reservoir ticks,
  encoding the key through the fixed W_in.)
- The context is then **wiped** — pass 2 does not contain the key.
- **Pass 2** asks for it: "The secret word was" → the model must output <KEY>.

The only place the key survives into pass 2 is the **reservoir state carried across the
two forward passes**. We train the readout W_out (+ LoRA) with backprop **through both
passes** so the injected model recalls the key, and compare against a **stateless
baseline** (identical, but the reservoir is reset between the passes) — which cannot.

This is differentiable end-to-end (the torch reservoir state carries the graph from
pass 1 to pass 2). Requires the `models` extra + peft.
"""
from __future__ import annotations

import numpy as np

# single-token-friendly secret words (verified at runtime against the tokenizer)
_CANDIDATE_KEYS = [" red", " blue", " green", " black", " white", " brown",
                   " gold", " silver", " orange", " purple"]


def _single_token_keys(tokenizer, n: int):
    keys = []
    for w in _CANDIDATE_KEYS:
        ids = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            keys.append((w.strip(), ids[0]))
        if len(keys) == n:
            break
    if len(keys) < n:
        raise RuntimeError("not enough single-token keys for this tokenizer")
    return keys


def run_cross_pass(model_name: str = "gpt2", *, n_keys: int = 6, steps: int = 200,
                   lr: float = 1e-3, seed: int = 0, device: str | None = None,
                   stateful: bool = True, layer: int | None = None,
                   summary: str = "last", n_reservoir: int = 512) -> dict:
    """Train cross-pass recall and return train losses + final recall accuracy.

    ``stateful=False`` is the baseline: the reservoir is reset between pass 1 and pass 2,
    so the carried state is destroyed and the key cannot survive into pass 2. ``summary``
    selects how the block output drives the reservoir ("last" token preserves the key
    better than "mean" pooling).
    """
    import torch
    from .torch_inject import TorchReservoirInjectedLM

    lm = TorchReservoirInjectedLM(model_name, seed=seed, device=device, layer=layer,
                                  summary=summary, n_reservoir=n_reservoir)
    tok = lm.tokenizer
    keys = _single_token_keys(tok, n_keys)
    rng = np.random.default_rng(seed)

    def passes(key_word):
        p1 = tok(f"The secret word is {key_word}.", return_tensors="pt").to(lm.device)
        p2 = tok("The secret word was", return_tensors="pt").to(lm.device)
        return p1, p2

    def forward_pair(key_word, *, reset_between):
        p1, p2 = passes(key_word)
        lm.reset_state()
        lm.model(**p1)                       # pass 1: reservoir encodes the key
        if reset_between:
            lm.reset_state()                 # baseline: destroy the carried state
        out2 = lm.model(**p2)                # pass 2: key is not in this context
        return out2.logits[0, -1]            # next-token logits at the recall point

    opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)
    lm.model.train()
    losses = []
    for _ in range(steps):
        word, tok_id = keys[int(rng.integers(n_keys))]
        logits = forward_pair(word, reset_between=not stateful)
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([tok_id], device=lm.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    # eval: recall accuracy over all keys (argmax == the key token)
    lm.model.eval()
    correct = 0
    with torch.no_grad():
        for word, tok_id in keys:
            logits = forward_pair(word, reset_between=not stateful)
            if int(logits.argmax().item()) == tok_id:
                correct += 1
    return {"model": model_name, "stateful": stateful, "n_keys": n_keys,
            "steps": steps, "loss_start": losses[0], "loss_end": losses[-1],
            "recall_accuracy": correct / n_keys, "device": lm.device, "mode": "additive"}


def run_cross_pass_kv(model_name: str = "gpt2", *, n_keys: int = 6, steps: int = 400,
                      lr: float = 1e-3, seed: int = 0, device: str | None = None,
                      stateful: bool = True, n_prefix: int = 8,
                      load_in_4bit: bool = False) -> dict:
    """Same cross-pass recall task, but with the **content-addressable** injection:
    the model attends to reservoir-derived prefix tokens (see ``kv_live``). This is the
    fix for the additive-injection negative result."""
    import torch
    from .kv_live import TorchReservoirPrefixInjectedLM

    lm = TorchReservoirPrefixInjectedLM(model_name, seed=seed, device=device,
                                        n_prefix=n_prefix, load_in_4bit=load_in_4bit)
    tok = lm.tokenizer
    keys = _single_token_keys(tok, n_keys)
    rng = np.random.default_rng(seed)

    def forward_pair(key_word, *, reset_between):
        p1 = tok(f"The secret word is {key_word}.", return_tensors="pt").to(lm.device)
        p2 = tok("The secret word was", return_tensors="pt").to(lm.device)
        lm.reset_state()
        lm.forward_logits(p1["input_ids"], p1["attention_mask"])   # pass 1: read key
        if reset_between:
            lm.reset_state()
        logits = lm.forward_logits(p2["input_ids"], p2["attention_mask"])
        return logits[0, -1]

    opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)
    lm.model.train()
    losses = []
    for _ in range(steps):
        word, tok_id = keys[int(rng.integers(n_keys))]
        logits = forward_pair(word, reset_between=not stateful)
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([tok_id], device=lm.device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    lm.model.eval()
    correct = 0
    with torch.no_grad():
        for word, tok_id in keys:
            if int(forward_pair(word, reset_between=not stateful).argmax().item()) == tok_id:
                correct += 1
    return {"model": model_name, "stateful": stateful, "n_keys": n_keys, "steps": steps,
            "loss_start": losses[0], "loss_end": losses[-1],
            "recall_accuracy": correct / n_keys, "device": lm.device, "mode": "kv-prefix"}
