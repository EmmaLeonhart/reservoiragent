#!/usr/bin/env python
"""Reproduce the headline Reservoir Attention Network results.

Runs the two load-bearing cross-pass recall experiments and checks the measured
recall against the values reported in the paper. Both need a CUDA GPU and the
``models`` extra (torch + transformers); GPT-2 fits in a few GB, Qwen2.5-1.5B in ~6 GB.

    python replication/reproduce.py             # run both headline experiments
    python replication/reproduce.py --gpt2-only # just the GPT-2 result (smaller/faster)

Each experiment trains the readout from scratch (a few minutes on GPU), then reports
the stateful recall vs a state-reset baseline. The result is a cross-pass *recall*:
a secret token is shown, the context is wiped, and the model must recall it from the
carried reservoir state alone — so any recall above the reset baseline is attributable
to the carried state, not to anything left in the context window.

Expected (from the paper; small variation across seeds/driver versions is normal):
  GPT-2,         6 keys : stateful recall 1.00  vs  ~0.17 reset baseline
  Qwen2.5-1.5B,  6 keys : stateful recall 0.83-1.00 (reservoir 2048, input scaling 0.10)

The logged JSON outputs we obtained are bundled under expected_results/ for inspection
without a GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, argv, expected stateful recall, tolerance) — argv is appended to `run.py crosspass`.
EXPERIMENTS = [
    (
        "GPT-2 cross-pass recall (6 keys)",
        ["--mode", "kv"],  # defaults: model gpt2, n_reservoir 512, input_scaling 0.5
        1.00, 0.10,
    ),
    (
        "Qwen2.5-1.5B cross-pass recall (6 keys, reservoir 2048, scaling 0.10)",
        ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--mode", "kv",
         "--n-reservoir", "2048", "--n-prefix", "16", "--input-scaling", "0.1",
         "--n-keys", "6", "--steps", "800"],
        0.83, 0.17,
    ),
]


def run_one(label: str, extra: list[str], expected: float, tol: float) -> bool:
    out = ROOT / "results" / "_replication_tmp.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "run.py"), "crosspass",
           "--out", str(out), *extra]
    print(f"\n=== {label} ===\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        print(f"  FAILED: run.py exited {proc.returncode}")
        return False
    data = json.loads(out.read_text(encoding="utf-8"))
    recall = data["results"]["stateful"]["recall_accuracy"]
    baseline = data["results"]["baseline"]["recall_accuracy"]
    ok = recall >= expected - tol
    print(f"  stateful recall = {recall:.3f}  (reset baseline {baseline:.3f}; "
          f"expected >= {expected - tol:.2f})  ->  {'OK' if ok else 'BELOW EXPECTED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt2-only", action="store_true", help="run only the GPT-2 experiment")
    args = ap.parse_args()
    experiments = EXPERIMENTS[:1] if args.gpt2_only else EXPERIMENTS
    results = [run_one(*e) for e in experiments]
    print("\n=== summary ===")
    for (label, *_), ok in zip(experiments, results):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
