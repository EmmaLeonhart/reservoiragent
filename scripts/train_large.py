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
    vocab = _env("RESERVOIR_VOCAB", "1200", int)
    n_res = _env("RESERVOIR_NRES", "1024", int)
    n_prefix = _env("RESERVOIR_NPREFIX", "16", int)
    lr = _env("RESERVOIR_LR", "5e-4", float)
    eval_n = _env("RESERVOIR_EVALN", "16", int)
    out_root = os.path.join(ROOT, "artifacts", "qwen-large")
    os.makedirs(out_root, exist_ok=True)

    # content-upweighted; silence down so it doesn't dominate
    weights = {"recall": 3, "deferred": 3, "accumulate": 2, "sequence": 2,
               "interrupt": 2, "timed": 1, "selfinit": 1, "silence": 0.5}

    print(f"[train_large] {model} | {hours}h | ckpt every {ckpt_min}m | vocab {vocab} "
          f"| reservoir {n_res}/{n_prefix} | -> hf.co/{repo}", flush=True)

    lm = TorchReservoirPrefixInjectedLM(model, n_reservoir=n_res, n_prefix=n_prefix,
                                        dtype="bfloat16", seed=0)
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
        lm.model.train()
        mean = sum(m.values()) / len(m)
        meta = {"epoch": epoch, "step": step, "elapsed_min": round((time.time() - start) / 60, 1),
                "lr": lr, "vocab": len(pool), "mean": mean, "metrics": m, "model": model}
        d = os.path.join(out_root, f"epoch_{epoch}")
        save_reservoir_model(d, lm, extra_meta=meta)
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
              f"mean {mean:.3f} :: " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()),
              flush=True)
        epoch += 1
        next_ckpt = time.time() + ckpt_min * 60

    while time.time() < deadline:
        ep = sample_episode(rng, weights)
        loss = episode_loss(lm, ep)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        running += float(loss.item())
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
