# Reservoir Attention Network (RAN)

> A **research project** scaffolded with
> [cleanvibe](https://github.com/Immanuelle/cleanvibe) `research`.

**Research question:** Can the **Reservoir Attention Network (RAN)** architecture — 
which injects a fixed, randomly-initialized reservoir into a pretrained 
transformer's mid-layer attention — give the model genuine state between forward 
passes (a real time axis) without degrading its base capabilities?

The feasibility study is **complete**. The results confirm that a RAN can be 
successfully implemented (as a **Reservoir Agent**) using pretrained transformers 
(GPT-2, Hermes 3B) without breaking them, and that a content-addressable 
(attended) injection enables genuine cross-context recall.

**Key Findings:**
- **H1 (Non-destruction):** Injected models match vanilla performance when the readout is zeroed.
- **H2 (Dynamics):** The ρ ≈ 1 echo-state boundary holds on real transformer activations.
- **H3 (Recall):** 100% cross-pass recall on GPT-2; identified scale difficulty on Hermes 3B.

**Safety & runtime (Phase G — from the imported Grok conversation):** the *same* fixed
reservoir that gives the agent a time-axis also pays safety value back, each backed by a
measured result rather than asserted:
- **Interruptibility.** A per-tick Reservoir Agent registers an urgent "STOP" at latency 0
  (vs a turn-based agent's mean 3.57 passes), and a one-shot burst persists in reservoir
  state for ~3 passes (fading memory) where a stateless monitor sees it for 0.
- **A cheap, stable monitoring surface.** A *linear* probe (no SAE) reads an internal clock
  off the reservoir at R² ≈ 0.99 (vs 0.16 stateless), degrading gracefully under a
  fine-tuning-like drift — usable across moderate drift, not invariant.
- **Bounded idle context.** A reservoir-protected KV-eviction policy (StreamingLLM /
  H2O-style, with the reservoir pinned) keeps an always-on agent's cache from growing without
  limit on blank ticks while never dropping the time-axis.

The next phase moves toward a KV-efficient base (DeepSeek-V2-Lite, the small open MLA model)
and the full always-alive runtime. The originating spec is `data_lake/reservoir_agent_plan.md`;
the strategic conversation behind Phase G is `data_lake/transcripts/attention-reservoir-architecture-grok.md`.

## About

This is an original research project (not a replication). It poses a question,
surveys the prior literature, runs experiments / builds something to answer it,
and publishes the findings as a themed GitHub Pages report + a transportable PDF.

The distinctive first move is a **literature review** (agentic RAG) *before* any
building — see `literature/`.

## How it's organized

- `literature/` — the literature review (sources + `REVIEW.md`), built first.
- `data_lake/` — datasets and supplied material.
- `src/` — the research code; `scripts/run.py` — the run entry point.
- `results/` — run outputs (gitignored). `FINDINGS.md` — the write-up.
- `docs/` — the published GitHub Pages report site (themed) + built PDF.
- `queue.md` / `todo.md` / `devlog.md` — the cleanvibe work loop.

## Model / weights

The trained GPT-2 cross-pass reservoir is published on Hugging Face:
**[EmmaLeonhart/reservoir-agent-gpt2-crosspass](https://huggingface.co/EmmaLeonhart/reservoir-agent-gpt2-crosspass)**
(verified 100% cross-context recall vs 17% chance baseline). Reproduce and save your own
with `python scripts/run.py crosspass --mode kv --save <dir>`, then publish with
`python scripts/publish_hf.py --artifact-dir <dir> --repo-id <user>/<name>`.

The cross-pass recall result is **not GPT-2-specific** — it scales across the Qwen family once
the reservoir is sized up and the input scaling is matched to the model. To reproduce the
Qwen2.5-1.5B run (recall 0.83–1.00 vs 0.17 control):

```
python scripts/run.py crosspass --model Qwen/Qwen2.5-1.5B-Instruct --mode kv \
    --n-reservoir 2048 --n-prefix 16 --input-scaling 0.1 --n-keys 6 --steps 800
```

The defaults (`--model gpt2 --n-reservoir 512 --input-scaling 0.5`) reproduce the GPT-2 result;
the decisive levers for transfer are `--n-reservoir` and `--input-scaling` (see FINDINGS).

## Experiments

All run via `python scripts/run.py <cmd>` (metrics → `results/`, figures → `docs/`):

- `sweep` / `sweep-real` / `sweep-scaling` — reservoir dynamics vs spectral radius / input scaling.
- `crosspass` — cross-pass recall: stateful vs stateless (the headline result; GPT-2 at defaults,
  scales to the Qwen family with `--n-reservoir 2048 --input-scaling 0.1`).
- `silence` — a trained reservoir-state silence gate vs a stateless gate.
- `blankcycle` — blank-tick KV-cache growth: vanilla (linear) vs reservoir-protected (bounded).
- `interrupt` — interruptibility: STOP latency + reservoir signal persistence.
- `probe` — reservoir-state linear probe: decode an internal clock + drift resilience.
- `batch` / `nseed` / `nseed-select` — N-seed reservoir populations and selection.

The pure-logic modules (e.g. `reservoir.kv_evict`, `reservoir.blank_cycle`, `reservoir.probe`)
are unit-tested on CPU in CI; the model/GPU steps are local-only.

## Getting started

```
cd reservoiragent
claude
```

Then work `queue.md` top to bottom. The bootstrap sequence pins down the
research question with you, runs the literature review, plans the experiments,
takes the repo public, and keeps the report current as results land.

## Published report

Once the repo is public with Pages set to **Source: GitHub Actions**,
`.github/workflows/pages.yml` deploys `docs/` (the report site) and builds
`docs/report.pdf`. Site-shape inspiration: http://latent-space.emmaleonhart.com/
