"""Train a reservoir agent on the composite stateful-task battery.

Samples episodes from the weighted mix of task generators, computes ``episode_loss``
(cross-entropy backpropagated *through the carried reservoir state* across all passes),
and steps AdamW on the trainable readout ``W_res`` + LoRA (the reservoir and base model
stay fixed). Periodically evaluates per-task accuracy on a fixed eval set. Optionally saves
the trained model via the existing artifact format so it loads straight into the live app.

Torch + the ``models`` extra required (GPU recommended). Entry point: ``scripts/run.py
battery``.
"""
from __future__ import annotations

from collections import defaultdict


def _evaluate(lm, eval_set) -> dict:
    """Per-task accuracy over the eval set (fraction of supervised steps decoded exactly)."""
    from .episode import episode_eval

    hits = defaultdict(int)
    tot = defaultdict(int)
    for ep in eval_set:
        for rec in episode_eval(lm, ep):
            tot[rec["task"]] += 1
            hits[rec["task"]] += int(rec["ok"])
    return {t: hits[t] / tot[t] for t in sorted(tot)}


def train_battery(model_name: str = "gpt2", *, steps: int = 400, lr: float = 1e-3,
                  seed: int = 0, weights: dict | None = None, device: str | None = None,
                  dtype: str | None = None, load_in_4bit: bool = False,
                  save_dir: str | None = None, eval_every: int = 100,
                  eval_n: int = 6, log=print) -> dict:
    import numpy as np
    import torch

    from .kv_live import TorchReservoirPrefixInjectedLM
    from .episode import episode_loss
    from .battery import sample_episode, make_eval_set, DEFAULT_WEIGHTS

    weights = weights or DEFAULT_WEIGHTS
    lm = TorchReservoirPrefixInjectedLM(model_name, seed=seed, device=device, dtype=dtype,
                                        load_in_4bit=load_in_4bit)
    rng = np.random.default_rng(seed)
    eval_set = make_eval_set(np.random.default_rng(seed + 9991), n_per_task=eval_n,
                             weights=weights)
    opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)

    log(f"battery training {model_name} on {sorted(weights)} for {steps} steps "
        f"(device {lm.device})…")
    lm.model.eval()
    base = _evaluate(lm, eval_set)
    log("  step    0 (untrained): " + "  ".join(f"{t}={base[t]:.2f}" for t in base))

    history = [{"step": 0, **base}]
    losses = []
    lm.model.train()
    for i in range(steps):
        ep = sample_episode(rng, weights)
        loss = episode_loss(lm, ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if (i + 1) % eval_every == 0 or i == steps - 1:
            lm.model.eval()
            m = _evaluate(lm, eval_set)
            lm.model.train()
            recent = sum(losses[-eval_every:]) / len(losses[-eval_every:])
            history.append({"step": i + 1, "loss": recent, **m})
            log(f"  step {i + 1:>4}: loss={recent:.3f}  "
                + "  ".join(f"{t}={m[t]:.2f}" for t in m))

    result = {"model": model_name, "steps": steps, "tasks": sorted(weights),
              "loss_start": losses[0], "loss_end": losses[-1],
              "final": history[-1], "history": history}
    if save_dir is not None:
        from .persist import save_reservoir_model
        save_reservoir_model(save_dir, lm, extra_meta=result)
        result["saved_to"] = save_dir
    return result
