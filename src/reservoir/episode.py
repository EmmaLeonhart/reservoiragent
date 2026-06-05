"""Stateful episodes — the unit the agent's loss battery is built from.

An *episode* is a scripted sequence of forward passes through a reservoir-injected model
with the reservoir state carried across all of them and the context **wiped** at chosen
points, so the only information bridge between passes is the reservoir state. Each pass
(``Step``) may inject text, wipe the context first, and carry a supervision *target*:

- a **string**  -> the model should emit it here (teacher-forced cross-entropy);
- ``SILENCE``   -> the model should stay silent here (trained as "predict end-of-text");
- ``None``      -> a free tick: forward to advance the reservoir state, no loss.

``episode_loss`` runs an episode and returns the summed CE **with the autograd graph
spanning all passes** (backprop-through-passes, as the cross-pass recall trainer does).
``episode_eval`` greedy-decodes the targets under ``no_grad`` and returns per-target
correctness for metrics. The Episode/Step data model is pure and unit-tested; the
execution is torch-gated.
"""
from __future__ import annotations

from dataclasses import dataclass


class _Silence:
    def __repr__(self):
        return "SILENCE"


SILENCE = _Silence()   # sentinel target: "emit nothing this pass"


@dataclass
class Step:
    inject: str | None = None      # text folded into the context this pass
    wipe: bool = False             # clear the context (NOT the reservoir) before this pass
    target: object = None          # str (emit) | SILENCE | None (free tick)


@dataclass
class Episode:
    steps: list                    # list[Step]
    task: str = ""                 # task-family label, for metrics


def _ctx_add(ctx: str, text: str) -> str:
    return (ctx + " " + text).strip() if ctx else text


def _encode(lm, ctx: str):
    text = ctx if ctx else "\n"    # empty context -> a newline so tokenization is non-empty
    enc = lm.tokenizer(text, return_tensors="pt").to(lm.device)
    return enc["input_ids"], enc["attention_mask"]


def _target_ids(lm, step):
    if step.target is SILENCE:
        return [lm.tokenizer.eos_token_id]
    return lm.tokenizer(step.target, add_special_tokens=False)["input_ids"]


def episode_loss(lm, episode: Episode, *, gate_weight: float = 1.0):
    """Run ``episode`` and return the mean loss as a torch scalar whose graph spans every
    pass (so ``.backward()`` trains through the carried state).

    Each supervised pass supervises the **gate head** (speak vs silent, BCE) and — only on
    *emit* passes — the **content** (teacher-forced cross-entropy on the LM output). Silence
    is the gate head's job, not "predict eos", so it no longer competes with content.
    Requires a model exposing ``gate_logit()`` (``TorchReservoirPrefixInjectedLM``)."""
    import torch
    import torch.nn.functional as F

    lm.reset_state()
    ctx = ""
    total = None
    nterms = 0

    def add(x):
        nonlocal total, nterms
        total = x if total is None else total + x
        nterms += 1

    for step in episode.steps:
        if step.wipe:
            ctx = ""
        if step.inject:
            ctx = _ctx_add(ctx, step.inject)
        ids, mask = _encode(lm, ctx)
        if step.target is None:
            lm.forward_logits(ids, mask)               # free tick: advance state only
            continue
        logits = lm.forward_logits(ids, mask)          # ticks the state
        g = lm.gate_logit()                            # speak/silent from the updated state
        if step.target is SILENCE:
            add(gate_weight * F.binary_cross_entropy_with_logits(g, torch.zeros_like(g)))
            continue
        add(gate_weight * F.binary_cross_entropy_with_logits(g, torch.ones_like(g)))
        cur_ids, cur_mask = ids, mask
        for t in _target_ids(lm, step):                # teacher-forced content
            add(F.cross_entropy(logits[0, -1].unsqueeze(0),
                                torch.tensor([t], device=lm.device)))
            cur_ids = torch.cat([cur_ids, torch.tensor([[t]], device=lm.device)], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=lm.device)], dim=1)
            logits = lm.forward_logits(cur_ids, cur_mask)
        ctx = _ctx_add(ctx, step.target)
    if total is None:
        return torch.zeros((), device=lm.device, requires_grad=True)
    return total / max(nterms, 1)


def episode_eval(lm, episode: Episode) -> list[dict]:
    """Greedy-decode each target under no_grad using the gate head + content output; return
    one record per supervised step ``{"task", "target", "pred", "ok"}``. A SILENCE step is
    ``ok`` when the gate stays shut; an emit step is ``ok`` when the gate opens **and** the
    content decodes exactly."""
    import torch

    lm.reset_state()
    ctx = ""
    records = []
    with torch.no_grad():
        for step in episode.steps:
            if step.wipe:
                ctx = ""
            if step.inject:
                ctx = _ctx_add(ctx, step.inject)
            ids, mask = _encode(lm, ctx)
            if step.target is None:
                lm.forward_logits(ids, mask)
                continue
            logits = lm.forward_logits(ids, mask)
            speak = float(lm.gate_logit()) > 0.0
            if step.target is SILENCE:
                records.append({"task": episode.task, "target": "<silence>",
                                "pred": "<silent>" if not speak else "<spoke>",
                                "ok": (not speak)})
                continue
            tgt = _target_ids(lm, step)
            cur_ids, cur_mask = ids, mask
            pred = []
            for _ in tgt:                              # greedy-decode len(target) tokens
                nid = int(logits[0, -1].argmax())
                pred.append(nid)
                cur_ids = torch.cat([cur_ids, torch.tensor([[nid]], device=lm.device)], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=lm.device)], dim=1)
                logits = lm.forward_logits(cur_ids, cur_mask)
            shown = lm.tokenizer.decode(pred).strip()
            records.append({"task": episode.task, "target": step.target,
                            "pred": shown if speak else shown + " [gated-silent]",
                            "ok": bool(speak and pred == tgt)})
            ctx = _ctx_add(ctx, step.target)
    return records
