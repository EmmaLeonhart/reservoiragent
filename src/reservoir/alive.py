"""The always-alive engine — the real-time stateful agent loop.

Drives a reservoir-injected causal LM as a *continuous process* rather than a request
handler: the reservoir state is never reset between passes, the context buffer is owned
by the engine, and the loop ticks on its own — a **prompted** pass when the user has
injected text, an unprompted **idle** tick otherwise. Each pass ticks the reservoir once
(through ``forward_logits``); when the output gate is confident, the agent streams a
bounded burst of tokens and appends them as its own turn.

The reservoir's influence on the output is a live gain (``lm.readout_scale``): an
untrained chat base stays coherent at low gain, and the user can fade the reservoir in
and out at runtime to *see* its effect — the point of the real-time harness.

The pure loop logic (drain injects, gate decision, telemetry including how far the
reservoir-state *direction* moved) is unit-tested with a fake LM. The token streaming
(``stream_emit``) is torch-gated / live.
"""
from __future__ import annotations

import contextlib

import numpy as np

from .runtime import ConfidenceGate  # reuse the entropy gate (model-agnostic)

# Minimal and NEUTRAL on purpose. The agent's stateful / "alive" behaviour must come
# from TRAINING the reservoir readout (+LoRA), not from describing it in the prompt —
# putting "you are a continuously running mind" here would just have the base model
# parrot it back, which demonstrates nothing about the reservoir. Keep this tiny or None.
DEFAULT_SYSTEM = "You are a helpful assistant."


def _to_numpy(x):
    """Accept a torch tensor or anything array-like; return a float numpy array."""
    if hasattr(x, "detach"):
        return x.detach().to("cpu").float().numpy()
    return np.asarray(x, dtype=float)


class AliveEngine:
    """A continuous reservoir-agent loop over a loaded reservoir-injected LM.

    ``lm`` must expose ``tokenizer`` (with ``apply_chat_template``), ``device``,
    ``forward_logits(input_ids, attention_mask)`` (ticks the reservoir, returns logits),
    ``reset_state()``, ``readout_scale`` (float gain), ``_state`` (the reservoir tensor),
    and (for streaming) ``torch``. ``TorchReservoirPrefixInjectedLM`` satisfies this.
    """

    def __init__(self, lm, *, system: str | None = DEFAULT_SYSTEM,
                 gate: ConfidenceGate | None = None, max_turns: int = 16,
                 burst_tokens: int = 24):
        self.lm = lm
        self.system = system
        self.gate = gate or ConfidenceGate(threshold=0.85)
        self.max_turns = int(max_turns)
        self.burst_tokens = int(burst_tokens)
        self.turns: list[dict] = []          # [{"role", "content"}], owned + never wiped
        self._pending: list[str] = []        # injected user text not yet folded into ctx
        self._prev_state = None
        self.tick_count = 0
        self.lm.reset_state()

    # ---- input (called from the server's IO thread; list.append is atomic under GIL) --
    def inject(self, text: str) -> None:
        """Queue user text to be folded into the context at the next pass."""
        if text and text.strip():
            self._pending.append(text.strip())

    def set_readout_scale(self, value: float) -> None:
        """Live-set the reservoir->output gain (0 = base model, higher = more reservoir)."""
        self.lm.readout_scale = float(value)

    def _inference(self):
        """No-grad context for the live loop. The reservoir state carries the autograd
        graph across passes (intended for *training* backprop-through-passes); in a
        continuous loop that graph would grow every tick until the GPU OOMs, so every
        forward here runs under no_grad. ``nullcontext`` keeps the fake-LM tests torch-free."""
        torch = getattr(self.lm, "torch", None)
        return torch.no_grad() if torch is not None else contextlib.nullcontext()

    # ---- context ----
    def _messages(self) -> list[dict]:
        head = [{"role": "system", "content": self.system}] if self.system else []
        return head + self.turns[-self.max_turns:]

    def _encode(self):
        tok = self.lm.tokenizer
        text = tok.apply_chat_template(self._messages(), tokenize=False,
                                       add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(self.lm.device)
        return enc["input_ids"], enc["attention_mask"]

    def _drain(self) -> bool:
        took = bool(self._pending)
        for t in self._pending:
            self.turns.append({"role": "user", "content": t})
        self._pending = []
        return took

    def _state_cos(self) -> float:
        """Cosine between the current and previous reservoir-state vectors — how aligned
        consecutive states are. Rises toward 1.0 as the state settles (fading memory)."""
        s = _to_numpy(self.lm._state).ravel()
        if self._prev_state is not None and s.size == self._prev_state.size:
            denom = float(np.linalg.norm(s) * np.linalg.norm(self._prev_state)) + 1e-9
            cos = float(np.dot(s, self._prev_state) / denom)
        else:
            cos = 0.0
        self._prev_state = s.copy()
        return cos

    # ---- one pass ----
    def gating_pass(self) -> dict:
        """Fold in any injected text, run one reservoir-ticking forward, decide whether to
        emit, and return telemetry. Stashes the rendered context + logits so a following
        ``stream_emit`` continues from this exact pass (no redundant forward)."""
        had_input = self._drain()
        ids, mask = self._encode()
        with self._inference():
            logits = self.lm.forward_logits(ids, mask)
        last = _to_numpy(logits[0, -1]).ravel()
        emit, h = self.gate.decide(last)
        cos = self._state_cos()
        self.tick_count += 1
        self._ids, self._mask, self._gate_logits = ids, mask, logits
        return {"kind": "prompted" if had_input else "idle", "emit": bool(emit),
                "entropy": round(float(h), 4), "state_cos": round(cos, 4),
                "tick": self.tick_count,
                "readout_scale": round(float(self.lm.readout_scale), 3)}

    def stream_emit(self):
        """Greedy-stream a bounded burst from the stashed pass, yielding decoded token
        pieces, then append the burst as the agent's own turn. Ticks the reservoir per
        token (generation is part of the live stream). Torch-gated."""
        torch = self.lm.torch
        tok = self.lm.tokenizer
        ids, mask, logits = self._ids, self._mask, self._gate_logits
        produced: list[int] = []
        for _ in range(self.burst_tokens):
            nid = int(_to_numpy(logits[0, -1]).argmax())
            if nid == tok.eos_token_id:
                break
            produced.append(nid)
            yield tok.decode([nid], skip_special_tokens=True)
            ids = torch.cat([ids, torch.tensor([[nid]], device=self.lm.device)], dim=1)
            mask = torch.cat(
                [mask, torch.ones((1, 1), dtype=mask.dtype, device=self.lm.device)], dim=1)
            with self._inference():
                logits = self.lm.forward_logits(ids, mask)
        text = tok.decode(produced, skip_special_tokens=True).strip()
        if text:
            self.turns.append({"role": "assistant", "content": text})

    def step(self):
        """One full pass as an event stream: a ``telemetry`` event, then ``token`` events
        if the gate opened. The server iterates this and broadcasts each event."""
        info = self.gating_pass()
        yield ("telemetry", info)
        if info["emit"]:
            for piece in self.stream_emit():
                yield ("token", piece)
