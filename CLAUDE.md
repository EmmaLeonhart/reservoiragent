# Reservoir Attention Network (RAN) — research project

## Skills

Workflow behaviors live as skills in `.claude/skills/` (auto-discovered by Claude Code):
`emergency-stop`, `cron-is-local`, `autonomous-loop`, `queue-driven-workflow`,
`writing-style`, `cleanvibe-update-check`. They are vendored into this repo and kept
current by the `cleanvibe-update-check` skill.

- **Last cleanvibe update check:** `2026-05-30` (cleanvibe v1.14.0 — all six skills current, no revisions)
- **Updates source:** <https://cleanvibe.emmaleonhart.com/updates.md>


## Project Description

This is a **research project** scaffolded by `cleanvibe research`. Unlike a
*replication* (which reproduces someone else's paper), this is **your own**
investigation: you pose a question, survey the prior literature, run experiments
or build something to answer it, and publish the findings.

The project investigates the **Reservoir Attention Network (RAN)**: a pretrained 
transformer with a fixed, randomly-initialized reservoir injected into its 
mid-layer attention so the model carries state *between* forward passes (a 
genuine time axis), rather than being stateless. We refer to a specific 
instantiation of this architecture as a **Reservoir Agent**. The originating 
spec is `data_lake/reservoir_agent_plan.md`; the originating conversations + 
architecture diagrams are preserved under `data_lake/transcripts/` and 
`data_lake/diagrams/`. **This session's scope is a feasibility + dynamics study 
at small scale** (GPT-2-scale base, single CUDA machine): inject the reservoir, 
regression-test that base behavior survives, characterize reservoir dynamics 
across spectral radius, and write up the theory. The full vision — forking the 
Hermes harness into an always-alive runtime and N-seed LoRA selection at agent 
scale — is the long-horizon target in `todo.md`.

> **Research question:** Can the **Reservoir Attention Network (RAN)** architecture — 
> which injects a fixed, randomly-initialized reservoir into a pretrained 
> transformer's mid-layer attention — give the model genuine state between forward 
> passes (a real time axis) without degrading its base capabilities?

Like a cleanvibe replication it produces a published, legible report — a themed
**GitHub Pages site** (`docs/`) plus a transportable PDF — but the content is
original research, grounded in a literature review rather than in one target
paper.

## Research workflow (the shape of this project)

1. **Question.** Pin down precisely what is being investigated / built and what a
   successful answer looks like. (Bootstrap step; the `> Research question` above
   gets filled in then.)
2. **Literature review (agentic RAG) — BEFORE building anything.** Survey the
   prior work: use whatever agentic search / RAG tooling is available (web
   search, `WebFetch`, and the `deep-research` skill if present) to find the
   relevant papers, posts, datasets, and code; read them; collect sources with
   citations into `literature/`; synthesize `literature/REVIEW.md` (what is
   already known, the gaps, and what *this* project adds). This grounds the work
   in the field instead of reinventing it, and it is what makes a `research`
   project different from a plain `new` one.
3. **Hypotheses & experiments.** Turn the identified gap into concrete, testable
   experiments / build steps. Plan them `todo.md` → `queue.md`.
4. **Build & run.** Implement under `src/`; entry point `scripts/run.py`;
   metrics → `results/`.
5. **Findings & report.** Write `FINDINGS.md`; keep the themed `docs/` site and
   the PDF report current as results land.

## Architecture and Conventions

- **`literature/`** — the literature review: source notes (one file per source,
  or a `sources.md`) and `REVIEW.md` (the synthesized survey, with citations).
  Committed; it is the evidentiary base of the project. Built in workflow step 2,
  before any implementation.
- **`data_lake/`** — datasets and other supplied/downloaded material (standard
  cleanvibe convention). Committed.
- **`src/`** — the research code. **`scripts/run.py`** — the entry point CI can
  invoke. **`results/`** — metrics JSON / run outputs (gitignored). **`FINDINGS.md`**
  — the write-up (question, method, results, limitations).
- **`docs/`** — the **published GitHub Pages site** (themed `index.html`, figures,
  and the built `report.pdf`). This is the legibility layer. The theme ships
  pre-styled (warm "paper" light theme + dark-mode variant); edit the content,
  keep the chrome. Site-shape inspiration: http://latent-space.emmaleonhart.com/
- **Go live early.** Create a **PUBLIC** GitHub repo and push near the start so
  every commit pushes and Pages/CI build as you go (public is required for free
  GitHub Pages).
- **Deliverables are built by GitHub Actions.** `.github/workflows/pages.yml`
  deploys `docs/` (the report site) and builds `docs/report.pdf` from
  `FINDINGS.md`. Make the repo public and set Settings -> Pages -> Source:
  GitHub Actions.

# currentDate
Today's date is 2026-05-29.

## Long command series run in strict order
When Emma gives a long series of commands, treat it as a long series of commands to be
executed in relatively STRICT ORDER, one after another, EVEN IF the order seems not to
make sense or seems inefficient. The sequencing is intentional — she organizes the steps
so states change in the order she wants. Do not reorder, merge, or skip steps.
