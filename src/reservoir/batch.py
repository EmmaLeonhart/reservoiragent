"""N-seed batch training — the project's core method (reservoir selection via fine-tuning).

Reservoir performance is stochastic in the random seed and there is no gradient pulling a
bad seed toward a good solution (the recurrent weights are fixed). So we train a
**population** of N reservoir agents (different seeds), each fine-tuned on the task, and
**keep all of them** — privileging the best while retaining the rest as signal for learning
what makes a reservoir good. See ``RESERVOIR_AGENTS.md``.

The ranking + manifest logic here is pure and unit-tested; ``train_batch`` (which fine-tunes
each seed) is torch-gated and exercised locally / on GPU.
"""
from __future__ import annotations

import os


def rank_population(records: list[dict]) -> list[dict]:
    """Rank a batch best-first: higher cross-pass recall wins; ties broken by lower final
    loss. Pure — does not mutate the input records."""
    return sorted(records,
                  key=lambda r: (-float(r.get("recall_accuracy", 0.0)),
                                 float(r.get("loss_end", float("inf")))))


def select_best(records: list[dict]) -> dict:
    """The privileged 'recommended' model of the batch (top of the ranking)."""
    return rank_population(records)[0]


def build_batch_manifest(records: list[dict], *, model_name: str,
                         created: str | None = None) -> dict:
    """Build a batch manifest: the full ranked population (every seed retained — the bad
    ones are signal), with the best marked ``recommended`` and surfaced as ``best``.

    Each population entry is the seed's record plus its ``rank`` and a ``recommended`` flag.
    """
    if not records:
        raise ValueError("cannot build a manifest from an empty batch")
    ranked = rank_population(records)
    population = []
    for i, r in enumerate(ranked):
        entry = dict(r)
        entry["rank"] = i
        entry["recommended"] = (i == 0)
        population.append(entry)
    best = population[0]
    return {"model_name": model_name, "n": len(population), "created": created,
            "population": population,
            "best": {"seed": best["seed"], "save_dir": best.get("save_dir")}}


def build_batch_card(manifest: dict, *, repo_id: str) -> str:
    """Generate the Hugging Face model card (README.md) for a published batch.

    The card documents the WHOLE population — every seed, with its score and dynamics
    signal, the bad ones included — because preserving them is the point (learning what
    makes a reservoir good). The recommended best is flagged and is what to load by default.
    """
    model = manifest["model_name"]
    best = manifest["best"]["seed"]
    rows = ["| rank | seed | recall | loss_end | pr_frac | recommended |",
            "|---|---|---|---|---|---|"]
    for p in manifest["population"]:
        rows.append(
            f"| {p['rank']} | `seed_{p['seed']}` | {float(p.get('recall_accuracy', 0)):.2f} "
            f"| {float(p.get('loss_end', float('nan'))):.3f} "
            f"| {float(p.get('pr_frac', float('nan'))):.3f} "
            f"| {'**yes**' if p.get('recommended') else ''} |")
    table = "\n".join(rows)
    return f"""---
license: mit
base_model: {model}
tags:
  - reservoir-agent
  - reservoir-computing
  - echo-state-network
  - stateful-transformer
library_name: reservoir-agent
---

# Reservoir Agent batch — {model}

A **batch** of {manifest['n']} reservoir agents (different fixed-random reservoir seeds)
trained on the cross-pass recall task. A reservoir agent is a new model type: a pretrained
transformer with a fixed reservoir brain-surgeried in (attended, cross-pass-stateful,
RNN-like) — see the [project](https://github.com/EmmaLeonhart/reservoiragent) and
`RESERVOIR_AGENTS.md`.

**The whole population is published, not just the winner.** Reservoir performance is
stochastic in the seed; the suboptimal models are kept as **signal** for learning which
reservoir properties survive selection. The **recommended** model is `seed_{best}`.

## Population

{table}

## Use

Each `seed_<n>/` is a complete loadable reservoir agent. Load the recommended one:

```python
from huggingface_hub import snapshot_download
from reservoir.persist import load_reservoir_model
path = snapshot_download("{repo_id}")
lm = load_reservoir_model(f"{{path}}/seed_{best}")
```

`batch_manifest.json` records the ranking + each seed's score and reservoir-dynamics signal.
"""


def _reservoir_dynamics_proxy(seed: int, *, n_reservoir: int = 512,
                              spectral_radius: float = 0.9,
                              input_scaling: float = 0.5) -> dict:
    """Cheap, training-free dynamics signal for a seed's reservoir: participation-ratio
    fraction on a random drive. Retained per-model so we can later correlate reservoir
    structure with which seeds win selection (the whole reason we keep the bad ones)."""
    import numpy as np
    from .torch_inject import _build_reservoir_weights

    W_r, W_in = _build_reservoir_weights(n_reservoir, 64, spectral_radius,
                                         input_scaling, 0.1, seed)
    rng = np.random.default_rng(seed)
    state = np.zeros(n_reservoir)
    states = []
    for _ in range(200):
        x = rng.standard_normal(64)
        state = np.tanh(W_r @ state + W_in @ x)
        states.append(state.copy())
    S = np.asarray(states[50:])                       # drop washout
    cov = np.cov(S, rowvar=False)
    ev = np.linalg.eigvalsh(cov).clip(min=0)
    pr = (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)   # participation ratio
    return {"participation_ratio": float(pr),
            "pr_frac": float(pr / n_reservoir),
            "spectral_radius": spectral_radius}


def train_batch(model_name: str = "gpt2", *, seeds=range(4), steps: int = 600,
                out_root: str = "artifacts/batch", n_keys: int = 6,
                lr: float = 1e-3, n_prefix: int = 8, input_scaling: float = 0.5,
                load_in_4bit: bool = False, dtype: str | None = None) -> dict:
    """Train one reservoir agent per seed on the cross-pass task, SAVE ALL of them, record
    each one's benchmark score + reservoir-dynamics signal, and return the batch manifest.

    Torch-gated. Each seed's model is saved under ``<out_root>/<model-short>/seed_<n>`` and
    the manifest is written to ``<out_root>/<model-short>/batch_manifest.json``.
    """
    import json
    from .crosspass import run_cross_pass_kv

    short = model_name.split("/")[-1].lower()
    batch_dir = os.path.join(out_root, short)
    os.makedirs(batch_dir, exist_ok=True)

    records = []
    for seed in seeds:
        save_dir = os.path.join(batch_dir, f"seed_{seed}")
        r = run_cross_pass_kv(model_name, n_keys=n_keys, steps=steps, lr=lr, seed=seed,
                              stateful=True, n_prefix=n_prefix,
                              input_scaling=input_scaling, load_in_4bit=load_in_4bit,
                              dtype=dtype, save_dir=save_dir)
        try:
            r.update(_reservoir_dynamics_proxy(seed, input_scaling=input_scaling))
        except Exception:
            pass                                       # dynamics signal is best-effort
        r["seed"] = seed
        r["save_dir"] = save_dir
        records.append(r)

    manifest = build_batch_manifest(records, model_name=model_name)
    with open(os.path.join(batch_dir, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
