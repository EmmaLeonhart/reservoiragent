# reservoiragent

> A **research project** scaffolded with
> [cleanvibe](https://github.com/Immanuelle/cleanvibe) `research`.

**Research question:** Can a fixed, randomly-initialized reservoir injected into a
pretrained transformer's mid-layer attention give the model genuine state between
forward passes — a real time axis — without degrading its base capabilities, and what
reservoir-dynamics regime (spectral radius, size, injection depth) makes that injected
state usable signal rather than noise?

The feasibility study is **complete**. The results confirm that a fixed, randomly-initialized
reservoir can be injected into a pretrained transformer (GPT-2, Hermes 3B) without
breaking it, and that a content-addressable (attended) injection enables genuine
cross-context recall.

**Key Findings:**
- **H1 (Non-destruction):** Injected models match vanilla performance when the readout is zeroed.
- **H2 (Dynamics):** The ρ ≈ 1 echo-state boundary holds on real transformer activations.
- **H3 (Recall):** 100% cross-pass recall on GPT-2; identified scale difficulty on Hermes 3B.

The next phase moves toward semantic agentic tasks and the full always-alive runtime.
The originating spec is `data_lake/reservoir_agent_plan.md`.

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
