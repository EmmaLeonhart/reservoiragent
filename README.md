# reservoiragent

> A **research project** scaffolded with
> [cleanvibe](https://github.com/Immanuelle/cleanvibe) `research`.

**Research question:** Can a fixed, randomly-initialized reservoir injected into a
pretrained transformer's mid-layer attention give the model genuine state between
forward passes — a real time axis — without degrading its base capabilities, and what
reservoir-dynamics regime (spectral radius, size, injection depth) makes that injected
state usable signal rather than noise?

This session scopes the question as a **feasibility + dynamics study** at small scale
(GPT-2-scale base, single machine): inject the reservoir, regression-test that base
behavior survives, characterize reservoir dynamics across spectral radius, and write up
the theory (genuine time dimension; Turing-completeness via recurrence). The full vision
— forking the Hermes harness into an always-alive runtime and N-seed LoRA selection at
agent scale — is the long-horizon target tracked in `todo.md`. The originating spec is
`data_lake/reservoir_agent_plan.md`.

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
