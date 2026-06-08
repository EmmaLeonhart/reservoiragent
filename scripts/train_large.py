"""Large-scale battery training for a reservoir agent — runs for hours, checkpointing and
uploading every epoch to the Hugging Face Hub.

This is the "make a real agent, not a 6-word toy" run: Qwen base, a LARGE single-token word
pool, a big reservoir + gate head, trained on the full 8-task stateful battery for a
wall-clock budget (default 8h). Every ~30 min it evaluates, saves a checkpoint, and uploads
it to the Hub as ``epoch_<n>/`` (plus a running ``index.json``). We expect the tail to
overtrain — that's fine, we keep all epochs and pick the best middle one (the Loka pattern).

Config via env vars (with defaults):
    RESERVOIR_MODEL   Qwen/Qwen2.5-1.5B-Instruct
    RESERVOIR_HF_REPO EmmaLeonhart/reservoir-agent-qwen-battery-large
    RESERVOIR_HOURS   8
    RESERVOIR_CKPT_MIN 30          checkpoint+upload cadence (minutes)
    RESERVOIR_VOCAB   1200         size of the single-token word pool
    RESERVOIR_NRES    1024         reservoir size
    RESERVOIR_NPREFIX 16           prefix tokens
    RESERVOIR_LR      5e-4
    RESERVOIR_EVALN   16
    RESERVOIR_EMIT_WEIGHT    3.0   up-weight the emit step
    RESERVOIR_SILENCE_WEIGHT 1.0   up-weight "stay shut" (fights gate over-firing)
    RESERVOIR_OUT     <repo name>  local artifact dir under artifacts/ (default: HF repo segment)
    RESERVOIR_TASKS   <mix>        override the task weights, e.g. "recall:1,deferred:1" (default: 8-task mix)

Run:  python scripts/train_large.py        (needs torch + GPU + HF write auth)
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


def main() -> int:
    import numpy as np
    import torch

    import reservoir.battery as B
    from reservoir.kv_live import TorchReservoirPrefixInjectedLM
    from reservoir.episode import episode_loss
    from reservoir.battery import sample_episode, make_eval_set
    from reservoir.train_battery import _evaluate
    from reservoir.persist import save_reservoir_model
    from huggingface_hub import HfApi, create_repo

    model = _env("RESERVOIR_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    repo = _env("RESERVOIR_HF_REPO", "EmmaLeonhart/reservoir-agent-qwen-battery-large")
    hours = _env("RESERVOIR_HOURS", "8", float)
    ckpt_min = _env("RESERVOIR_CKPT_MIN", "30", float)
    # Epoch-count mode (preferred): run exactly RESERVOIR_EPOCHS epochs of RESERVOIR_SPER
    # steps each, checkpoint+upload after each, then stop — NOT a wall-clock cap. Set
    # RESERVOIR_EPOCHS=0 to fall back to the time budget above.
    epochs_target = _env("RESERVOIR_EPOCHS", "0", int)
    steps_per_epoch = _env("RESERVOIR_SPER", "3000", int)
    vocab = _env("RESERVOIR_VOCAB", "1200", int)
    n_res = _env("RESERVOIR_NRES", "1024", int)
    n_prefix = _env("RESERVOIR_NPREFIX", "16", int)
    inscale = _env("RESERVOIR_INSCALE", "0.5", float)   # detune (lower) to avoid saturation
    lr = _env("RESERVOIR_LR", "5e-4", float)
    eval_n = _env("RESERVOIR_EVALN", "16", int)
    emit_weight = _env("RESERVOIR_EMIT_WEIGHT", "3.0", float)  # up-weight the emit step (not silence)
    silence_weight = _env("RESERVOIR_SILENCE_WEIGHT", "1.0", float)  # up-weight "stay shut" to fight gate over-firing
    proj_dim = _env("RESERVOIR_PROJ", "0", int) or None        # fixed down-projection for huge reservoirs
    lora_target = _env("RESERVOIR_LORA_TARGET", "all", str)    # adapt MLP too, not just attention
    # Local artifact dir, derived from the HF repo's last path segment by default so distinct
    # runs (e.g. different silence_weight) don't clobber each other's local index.json /
    # epoch_<n> dirs. Override with RESERVOIR_OUT.
    out_root = os.path.join(ROOT, "artifacts", _env("RESERVOIR_OUT", repo.split("/")[-1]))
    os.makedirs(out_root, exist_ok=True)

    # content-upweighted; silence down so it doesn't dominate
    weights = {"recall": 3, "deferred": 3, "accumulate": 2, "sequence": 2,
               "interrupt": 2, "timed": 1, "selfinit": 1, "silence": 0.5}
    # RESERVOIR_TASKS overrides the task mix, e.g. "recall:1,deferred:1" to isolate content tasks
    # from the gate/silence/temporal competition (probes whether the reservoir lift stabilizes).
    _tasks_env = os.environ.get("RESERVOIR_TASKS", "").strip()
    if _tasks_env:
        weights = {kv.split(":")[0].strip(): float(kv.split(":")[1])
                   for kv in _tasks_env.split(",") if ":" in kv}

    mode = (f"{epochs_target} epochs x {steps_per_epoch} steps" if epochs_target > 0
            else f"{hours}h budget, ckpt every {ckpt_min}m")
    print(f"[train_large] {model} | {mode} | vocab {vocab} | reservoir {n_res}/{n_prefix} "
          f"| input_scaling {inscale} | -> hf.co/{repo}", flush=True)

    lm = TorchReservoirPrefixInjectedLM(model, n_reservoir=n_res, n_prefix=n_prefix,
                                        input_scaling=inscale, dtype="bfloat16", seed=0,
                                        lora_target=lora_target, proj_dim=proj_dim)
    print(f"[train_large] emit_weight {emit_weight} | silence_weight {silence_weight} | "
          f"proj_dim {proj_dim} | lora_target {lora_target}", flush=True)
    pool = B.large_word_pool(lm.tokenizer, vocab)
    B.set_word_pool(pool)
    print(f"[train_large] word pool: {len(pool)} single-token words "
          f"(e.g. {pool[:8]})", flush=True)

    eval_set = make_eval_set(np.random.default_rng(123), n_per_task=eval_n, weights=weights)
    opt = torch.optim.AdamW(lm.trainable_parameters(), lr=lr)
    rng = np.random.default_rng(0)

    api = HfApi()
    create_repo(repo, repo_type="model", exist_ok=True)

    start = time.time()
    deadline = start + hours * 3600
    next_ckpt = start + ckpt_min * 60
    epoch = 0
    step = 0
    running = 0.0
    index = []
    lm.model.train()

    def checkpoint():
        nonlocal epoch, next_ckpt
        lm.model.eval()
        m = _evaluate(lm, eval_set)
        # CONTROL: same eval set with the reservoir reset every pass (no cross-pass carry).
        # Any capability above this stateless floor is attributable to the carried state.
        m_ctrl = _evaluate(lm, eval_set, stateless=True)
        lm.model.train()
        # capability mean: emit-requiring tasks only (exclude pure 'silence', which a quiet
        # gate passes for free) — headline number; per-task silence still reported.
        _keys = [t for t in m if t != "silence"]
        mean = (sum(m[t] for t in _keys) / len(_keys)) if _keys else 0.0
        ctrl_keys = [t for t in m_ctrl if t != "silence"]
        ctrl_mean = (sum(m_ctrl[t] for t in ctrl_keys) / len(ctrl_keys)) if ctrl_keys else 0.0
        meta = {"epoch": epoch, "step": step, "elapsed_min": round((time.time() - start) / 60, 1),
                "lr": lr, "vocab": len(pool), "mean": mean, "metrics": m,
                "stateless_mean": ctrl_mean, "stateless_metrics": m_ctrl,
                "lift": round(mean - ctrl_mean, 3), "model": model}
        d = os.path.join(out_root, f"epoch_{epoch}")
        save_reservoir_model(d, lm, extra_meta=meta)
        # Optimizer state per epoch (resumable training / analysis); lands in the epoch dir so
        # the upload_folder below ships it to HF alongside the model.
        torch.save({"optimizer": opt.state_dict(), "step": step, "epoch": epoch},
                   os.path.join(d, "optimizer.pt"))
        try:
            api.upload_folder(folder_path=d, repo_id=repo, path_in_repo=f"epoch_{epoch}",
                              commit_message=f"epoch {epoch} step {step} mean {mean:.3f}")
        except Exception as e:
            print(f"[train_large] epoch {epoch} upload failed: {e!r}", flush=True)
        index.append(meta)
        with open(os.path.join(out_root, "index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        try:
            api.upload_file(path_or_fileobj=os.path.join(out_root, "index.json"),
                            path_in_repo="index.json", repo_id=repo)
        except Exception as e:
            print(f"[train_large] index upload failed: {e!r}", flush=True)
        print(f"[train_large] epoch {epoch} step {step} ({meta['elapsed_min']}m) "
              f"mean {mean:.3f} (stateless ctrl {ctrl_mean:.3f}, lift {mean - ctrl_mean:+.3f}) :: "
              + "  ".join(f"{k}={v:.2f}" for k, v in m.items()), flush=True)
        epoch += 1
        next_ckpt = time.time() + ckpt_min * 60

    def train_step():
        nonlocal step, running
        ep = sample_episode(rng, weights)
        loss = episode_loss(lm, ep, emit_weight=emit_weight, silence_weight=silence_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        running += float(loss.item())

    if epochs_target > 0:                              # epoch-count mode (preferred)
        for _ in range(epochs_target):
            for _ in range(steps_per_epoch):
                train_step()
            print(f"[train_large] ~loss {running / max(step, 1):.3f} over {step} steps",
                  flush=True)
            checkpoint()
    else:                                              # wall-clock budget mode
        while time.time() < deadline:
            train_step()
            if time.time() >= next_ckpt:
                print(f"[train_large] ~loss {running / max(step, 1):.3f} over {step} steps",
                      flush=True)
                checkpoint()
        checkpoint()   # final

    print(f"[train_large] DONE: {epoch} epochs, {step} steps, "
          f"{(time.time() - start) / 3600:.2f}h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
