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
                  eval_n: int = 16, n_reservoir: int = 512, n_prefix: int = 8,
                  lora_target: str = "attn", input_scaling: float = 0.5,
                  unfreeze_from: int | None = None, lora_r: int = 8, log=print) -> dict:
    import numpy as np
    import torch

    from .kv_live import TorchReservoirPrefixInjectedLM
    from .episode import episode_loss
    from .battery import sample_episode, make_eval_set, DEFAULT_WEIGHTS

    weights = weights or DEFAULT_WEIGHTS
    lm = TorchReservoirPrefixInjectedLM(model_name, seed=seed, device=device, dtype=dtype,
                                        load_in_4bit=load_in_4bit, n_reservoir=n_reservoir,
                                        n_prefix=n_prefix, lora_target=lora_target,
                                        input_scaling=input_scaling, unfreeze_from=unfreeze_from,
                                        lora_r=lora_r)
    rng = np.random.default_rng(seed)
    eval_set = make_eval_set(np.random.default_rng(seed + 9991), n_per_task=eval_n,
                             weights=weights)
    opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)
    # cosine decay to 0 over the run — the flat lr=1e-3 overshot and degraded past its peak.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def mean_acc(m):
        return sum(m.values()) / len(m) if m else 0.0

    log(f"battery training {model_name} on {sorted(weights)} for {steps} steps "
        f"(lr {lr} cosine, device {lm.device})…")
    lm.model.eval()
    base = _evaluate(lm, eval_set)
    log("  step    0 (untrained): " + "  ".join(f"{t}={base[t]:.2f}" for t in base))

    history = [{"step": 0, **base}]
    losses = []
    best = {"mean_acc": -1.0, "step": 0, "metrics": base}   # keep the BEST eval, not the last
    lm.model.train()
    for i in range(steps):
        ep = sample_episode(rng, weights)
        loss = episode_loss(lm, ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
        if (i + 1) % eval_every == 0 or i == steps - 1:
            lm.model.eval()
            m = _evaluate(lm, eval_set)
            lm.model.train()
            recent = sum(losses[-eval_every:]) / len(losses[-eval_every:])
            history.append({"step": i + 1, "loss": recent, **m})
            log(f"  step {i + 1:>4}: loss={recent:.3f}  mean={mean_acc(m):.2f}  "
                + "  ".join(f"{t}={m[t]:.2f}" for t in m))
            if mean_acc(m) > best["mean_acc"]:
                best = {"mean_acc": mean_acc(m), "step": i + 1, "metrics": m}
                if save_dir is not None:           # persist the best-so-far (not the last)
                    from .persist import save_reservoir_model
                    save_reservoir_model(save_dir, lm, extra_meta={"best": best,
                                                                   "model": model_name})
                    log(f"           ^ new best (mean {best['mean_acc']:.2f}) -> saved")

    result = {"model": model_name, "steps": steps, "tasks": sorted(weights),
              "loss_start": losses[0], "loss_end": losses[-1],
              "final": history[-1], "best": best, "history": history}
    if save_dir is not None:
        result["saved_to"] = save_dir
    return result


def train_battery_population(model_name: str, *, n_seeds: int = 4,
                             out_root: str = "artifacts/battery-pop", steps: int = 1500,
                             log=print, **kw) -> dict:
    """Train a **population of N reservoirs** (one per seed) on the battery and keep them
    ALL — the N-seed selection design (RESERVOIR_AGENTS.md / reservoir_agent_plan.md). Each
    seed is a different fixed-random reservoir, LoRA-fine-tuned on the battery; the best by
    mean per-task accuracy is *recommended* but the whole population is preserved (bad seeds
    are signal). Writes a ``batch_manifest.json`` so the installer/app can load the best.
    """
    import json
    import os

    population = []
    for seed in range(n_seeds):
        seed_dir = os.path.join(out_root, f"seed_{seed}")
        log(f"\n=== reservoir seed {seed}/{n_seeds - 1} -> {seed_dir} ===")
        res = train_battery(model_name, seed=seed, steps=steps, save_dir=seed_dir,
                            log=log, **kw)
        b = res["best"]
        population.append({"seed": seed, "mean_acc": b["mean_acc"],
                           "best_step": b["step"], "metrics": b["metrics"]})

    population.sort(key=lambda p: -p["mean_acc"])
    best_seed = population[0]["seed"]
    for rank, p in enumerate(population):
        p["rank"] = rank
        p["recommended"] = (p["seed"] == best_seed)

    manifest = {"model": model_name, "kind": "battery", "n": n_seeds,
                "best": {"seed": best_seed, "mean_acc": population[0]["mean_acc"]},
                "population": population}
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log(f"\npopulation of {n_seeds}; recommended seed {best_seed} "
        f"(mean {population[0]['mean_acc']:.2f}). saved under {out_root}/")
    return manifest
