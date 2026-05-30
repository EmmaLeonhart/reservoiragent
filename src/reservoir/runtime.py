"""The always-alive runtime — the stateful agent harness.

A standard transformer is a stateless request handler. The Reservoir Agent is a
persistent process: its reservoir state and context buffer are owned by the runtime
and never wiped between passes. This module is the substrate the fine-tuning will
later plug into; it runs on the *untrained* injected model so the whole loop can be
exercised before any training.

Pieces (the pure logic is unit-tested without torch; the model integration is
torch-gated):

- ``ContextBuffer``      — the running conversation/tool history, never wiped.
- ``topk_entropy`` / ``ConfidenceGate`` — the output gate: emit when the next-token
  distribution is confident (low normalized top-k entropy), else stay silent.
- ``checkpoint_state`` / ``restore_state`` — persist/restore the reservoir tensor.
- ``AliveAgent``        — the loop: ``prompt`` (prompted pass on new input) and
  ``idle`` (unprompted pass over context + reservoir only), with the reservoir state
  persisting across both.

NOTE (named plainly): on the untrained model the gate uses the *base model's*
next-token entropy as a proxy, so its emit/silence decisions are not yet meaningful —
the harness is the mechanism, not a trained policy. The gate threshold and the
self-initiation behaviour become meaningful only after the readout/LoRA are trained.
"""
from __future__ import annotations

import numpy as np


class ContextBuffer:
    """The running context, owned by the runtime and never wiped between passes."""

    def __init__(self):
        self.turns: list[dict] = []

    def append(self, role: str, text: str) -> None:
        self.turns.append({"role": role, "text": text})

    def render(self) -> str:
        return "\n".join(f"{t['role']}: {t['text']}" for t in self.turns)

    def __len__(self) -> int:
        return len(self.turns)


def topk_entropy(logits, k: int = 50) -> float:
    """Normalized Shannon entropy (in [0, 1]) of the softmax over the top-k logits.

    0 ⇒ a single dominant next token (confident); 1 ⇒ a uniform top-k (uncertain).
    """
    x = np.asarray(logits, dtype=float).ravel()
    k = min(k, x.size)
    idx = np.argpartition(x, -k)[-k:]
    z = x[idx] - np.max(x[idx])
    p = np.exp(z)
    p /= p.sum()
    H = -np.sum(p * np.log(p + 1e-12))
    return float(H / np.log(k)) if k > 1 else 0.0


class ConfidenceGate:
    """Decide whether to emit (confident) or stay silent (uncertain)."""

    def __init__(self, threshold: float = 0.85, k: int = 50):
        self.threshold = float(threshold)
        self.k = int(k)

    def decide(self, logits) -> tuple[bool, float]:
        """Return (emit?, normalized_entropy). Emit when entropy < threshold."""
        h = topk_entropy(logits, self.k)
        return (h < self.threshold, h)


def checkpoint_state(state: np.ndarray, path: str) -> None:
    """Persist the reservoir state tensor to disk."""
    np.save(path, np.asarray(state))


def restore_state(path: str) -> np.ndarray:
    """Load a reservoir state tensor from disk."""
    return np.load(path)


class AliveAgent:
    """An always-alive agent: a reservoir-injected LM + context buffer + gate.

    Requires the ``models`` extra (torch + transformers).
    """

    def __init__(self, model_name: str = "gpt2", *, gate: ConfidenceGate | None = None,
                 readout_scale: float = 0.0, seed: int = 0, n_reservoir: int = 256,
                 device: str | None = None, **reservoir_kwargs):
        from .inject import ReservoirInjectedLM

        self.lm = ReservoirInjectedLM(model_name, n_reservoir=n_reservoir, seed=seed,
                                      device=device, **reservoir_kwargs)
        if readout_scale:
            rng = np.random.default_rng(seed)
            self.lm.set_readout(rng.standard_normal((self.lm.d_model, self.lm.n_reservoir))
                                * readout_scale)
        self.ctx = ContextBuffer()
        self.gate = gate or ConfidenceGate()
        self.history: list[dict] = []

    # one pass = one reservoir tick (the gating forward); generation does not tick
    def _pass(self, kind: str, max_new_tokens: int = 12) -> dict:
        text = self.ctx.render() or "\n"
        logits = self.lm.logits(text).detach().to("cpu").numpy()
        emit, h = self.gate.decide(logits)
        rec = {"kind": kind, "emit": bool(emit), "entropy": round(h, 4),
               "state_norm": round(float(np.linalg.norm(self.lm.reservoir_state())), 4)}
        if emit:
            rec["said"] = self._generate(max_new_tokens)
            self.ctx.append("agent", rec["said"])
        self.history.append(rec)
        return rec

    def _generate(self, max_new_tokens: int) -> str:
        torch = self.lm.torch
        # pause the reservoir hook so generation doesn't keep ticking the reservoir
        self.lm.remove_hook()
        try:
            ids = self.lm.tokenizer(self.ctx.render() or "\n", return_tensors="pt").to(self.lm.device)
            with torch.no_grad():
                out = self.lm.model.generate(
                    **ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.lm.tokenizer.eos_token_id)
            new = out[0][ids["input_ids"].shape[1]:]
            return self.lm.tokenizer.decode(new, skip_special_tokens=True).strip()
        finally:
            self.lm._register()

    def prompt(self, text: str) -> dict:
        """A prompted pass: new input arrives, append it, run a pass."""
        self.ctx.append("user", text)
        return self._pass("prompted")

    def idle(self) -> dict:
        """An unprompted pass: an idle tick over context + reservoir, no new input."""
        return self._pass("unprompted")

    def checkpoint(self, path: str) -> None:
        checkpoint_state(self.lm.reservoir_state(), path)

    def restore(self, path: str) -> None:
        self.lm.reservoir.state = restore_state(path)
