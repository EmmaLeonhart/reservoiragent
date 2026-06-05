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


def episode_loss(lm, episode: Episode):
    """Run ``episode`` and return the mean per-target-token cross-entropy as a torch scalar
    whose graph spans every pass (so ``.backward()`` trains through the carried state)."""
    import torch
    import torch.nn.functional as F

    lm.reset_state()
    ctx = ""
    total = None
    nterms = 0
    for step in episode.steps:
        if step.wipe:
            ctx = ""
        if step.inject:
            ctx = _ctx_add(ctx, step.inject)
        ids, mask = _encode(lm, ctx)
        if step.target is None:
            lm.forward_logits(ids, mask)               # free tick: advance state only
            continue
        tgt = _target_ids(lm, step)
        cur_ids, cur_mask = ids, mask
        for t in tgt:                                  # teacher-forced over the target tokens
            logits = lm.forward_logits(cur_ids, cur_mask)
            ce = F.cross_entropy(logits[0, -1].unsqueeze(0),
                                 torch.tensor([t], device=lm.device))
            total = ce if total is None else total + ce
            nterms += 1
            cur_ids = torch.cat([cur_ids, torch.tensor([[t]], device=lm.device)], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=lm.device)], dim=1)
        if step.target is not SILENCE:                 # teacher-force the true emit into context
            ctx = _ctx_add(ctx, step.target)
    if total is None:                                  # no supervised step
        return torch.zeros((), device=lm.device, requires_grad=True)
    return total / max(nterms, 1)


def episode_eval(lm, episode: Episode) -> list[dict]:
    """Greedy-decode each target under no_grad; return one record per supervised step:
    ``{"task", "target", "pred", "ok"}`` (SILENCE is ``ok`` when the model picks eos)."""
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
            tgt = _target_ids(lm, step)
            cur_ids, cur_mask = ids, mask
            pred = []
            for _ in tgt:                              # greedy-decode len(target) tokens
                logits = lm.forward_logits(cur_ids, cur_mask)
                nid = int(logits[0, -1].argmax())
                pred.append(nid)
                cur_ids = torch.cat([cur_ids, torch.tensor([[nid]], device=lm.device)], dim=1)
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=lm.device)], dim=1)
            ok = pred == tgt
            if step.target is SILENCE:
                shown = "<silent>" if ok else lm.tokenizer.decode(pred).strip()
            else:
                shown = lm.tokenizer.decode(pred).strip()
                ctx = _ctx_add(ctx, step.target)
            records.append({"task": episode.task,
                            "target": "<silence>" if step.target is SILENCE else step.target,
                            "pred": shown, "ok": bool(ok)})
    return records
