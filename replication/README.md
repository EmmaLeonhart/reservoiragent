# Reservoir Attention Network — replication package

This package reproduces the **headline results** of *The Reservoir Attention Network: Cross-Pass
State in Pretrained Transformers via Content-Addressable Reservoir Injection*: that injecting a
fixed, randomly-initialized reservoir into a pretrained transformer's mid-layer attention as
content-addressable key/value prefix tokens gives the model **usable state across forward passes**,
demonstrated by cross-pass recall that survives a full context wipe.

The load-bearing result is a **cross-pass recall** task: a secret token is shown, the context window
is wiped, and the model must recall it on a later pass — so the only information bridge is the carried
reservoir state. Any recall above the state-reset baseline is attributable to that carried state.

## What this reproduces

| Experiment | Stateful recall | Reset baseline | Config |
|---|---|---|---|
| GPT-2, 6 keys | **1.00** | ~0.17 (chance) | reservoir 512, input scaling 0.5 (defaults) |
| Qwen2.5-1.5B, 6 keys | **0.83–1.00** | ~0.17 | reservoir 2048, input scaling 0.10 |
| GPT-2, 24 keys | ~0.92 | chance | capacity sweep (ceiling ~tens of items) |

The decisive levers for transferring the result to a larger model are **reservoir size**
(`--n-reservoir`) and **input scaling** (`--input-scaling`), not parameter count — GPT-2-medium fails
across a scaling sweep, while Qwen2.5-1.5B recovers once both are matched to the model.

## Requirements

- Python ≥ 3.9
- A CUDA GPU (GPT-2 needs a few GB; Qwen2.5-1.5B needs ~6 GB at bf16/4-bit)
- The `models` extra: `torch>=2.0`, `transformers>=4.30`

## Setup

```bash
git clone https://github.com/EmmaLeonhart/reservoiragent
cd reservoiragent
pip install -e ".[models]"      # installs the package + torch + transformers
```

## Reproduce

```bash
python replication/reproduce.py             # both headline experiments
python replication/reproduce.py --gpt2-only  # just GPT-2 (smaller / faster)
```

Each experiment trains the readout from scratch (a few minutes on GPU), then prints the measured
stateful recall against the state-reset baseline and checks it against the paper's value. Or run the
underlying entry point directly:

```bash
# GPT-2 headline result (1.00 vs ~0.17)
python scripts/run.py crosspass --mode kv

# Qwen2.5-1.5B scaling result (0.83–1.00 vs ~0.17)
python scripts/run.py crosspass --model Qwen/Qwen2.5-1.5B-Instruct --mode kv \
    --n-reservoir 2048 --n-prefix 16 --input-scaling 0.1 --n-keys 6 --steps 800
```

## Inspect the logged results without a GPU

The JSON outputs from our own runs are bundled under [`expected_results/`](expected_results/) — each
records the params, the start/end training loss, and the stateful-vs-baseline recall. These let you
verify the reported numbers without rerunning anything.

## Trained weights

A trained GPT-2 cross-pass reservoir is published at
[EmmaLeonhart/reservoir-agent-gpt2-crosspass](https://huggingface.co/EmmaLeonhart/reservoir-agent-gpt2-crosspass).

## What is *not* claimed

This is a feasibility + dynamics study, not an agent. The recall task is a minimal single-token probe;
the integrated 8-task "battery" does *not* stably retain a reservoir-driven content solution (a
well-controlled negative, documented in the paper). The reproducible, retained result is the
strict-wipe cross-pass recall above.

## Citation / provenance

Full method, results, controls, and limitations are in `FINDINGS.md` (the paper) and at
<https://reservoir.emmaleonhart.com>. The complete code, prompts, and commit history are public in the
repository, so the process is auditable end to end.
