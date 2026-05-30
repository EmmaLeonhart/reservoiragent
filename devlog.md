# reservoiragent — Devlog

**This file is where "done" lives.** `queue.md` is delete-only: when a queue
item is finished, the item is **deleted from `queue.md`** and a dated entry
is **appended here**, in the same commit as the work, then pushed. Never
tick a box in place — a checked box left in `queue.md` is the failure mode
this file exists to prevent.

Also record releases (tag + a one-line note), notable milestones, and
anything else worth a chronological trail. Newest entries at the bottom.

This is the **same convention as the cleanvibe repo's own `devlog.md`** —
every cleanvibe-scaffolded project gets one for the same reason.

See `CLAUDE.md` § "Workflow Rules" and `queue.md`'s preamble.

---

## 2026-05-29 — Project scaffolded

Scaffolded with `cleanvibe new` (cleanvibe v1.13.0). Future entries
land here as queue items get deleted.

## 2026-05-29 — Bootstrap 1: autonomous loop started

Started the three session-local crons (`durable:false`): work-loop `3 * * * *`
(de9f5f85), auto-flush `15 * * * *` (42d311a4), status-report `42 * * * *`
(ab624872). Mirrored the bootstrap items into the task tool.

## 2026-05-29 — Bootstrap 2: data_lake triage + originating-context preservation

The project's originating context was three Claude chat HTML exports + a
user-authored `reservoir_agent_plan.md`. Preserved the durable, publishable parts
and protected privacy:

- `data_lake/extract_chat_context.py` distils each export into clean Markdown
  **transcripts** (`data_lake/transcripts/`, 24/6/32 turns) and recovers the
  **architecture diagrams** as SVG (`data_lake/diagrams/`: per-pass injection,
  always-alive runtime, joint attention layer + a transformer baseline) plus
  WEBP raster previews.
- The raw `*.html` exports + `*_files/` chrome are **gitignored** (kept local):
  they embed a private "Recents" sidebar of unrelated conversations and ~10 MB
  of web chrome each. Transcripts verified to contain zero sidebar leakage.
- Recovered SVGs use Claude's dark-theme CSS vars, so they render black-boxed
  standalone (labels/structure intact); they'll be re-themed for the `docs/`
  report. `data_lake/README.md` documents provenance.

## 2026-05-29 — Bootstrap 3: research question defined

After reading the plan + transcripts, pinned the question with the user (who chose
a **feasibility + dynamics study** scope for this session via a scoping checkpoint):

> Can a fixed, randomly-initialized reservoir injected into a pretrained transformer's
> mid-layer attention give the model genuine state between forward passes — a real time
> axis — without degrading its base capabilities, and what reservoir-dynamics regime
> (spectral radius, size, injection depth) makes that injected state usable signal
> rather than noise?

Scope this session: GPT-2-scale base on a single CUDA machine — inject the reservoir,
regression-test base behavior, characterize dynamics across spectral radius, write up
theory (time dimension; Turing-completeness via recurrence). Full Hermes-fork +
always-alive runtime + N-seed LoRA selection is the long-horizon `todo.md` target.
Written into `README.md`, `CLAUDE.md` (Project Description + Research question), and
`docs/index.html` (lede, question block, pillar 1).

## 2026-05-29 — Bootstrap 4: literature review (agentic RAG)

Ran a multi-agent deep-research pass (103 agents; 5 angles → 21 sources → 96 claims →
25 adversarially verified, 24 confirmed / 1 refuted), then a targeted re-verification
of the closest prior art. Wrote `literature/sources.md` (per-source notes, grouped by
area, with verification status) and `literature/REVIEW.md` (synthesis).

Key outcomes:
- **Foundations solid:** the fixed-reservoir/trained-readout core is textbook ESN/LSM
  (Jaeger, Maass, Lukoševičius & Jaeger); the spectral-radius/ESP regime is real but
  the edge-of-chaos optimum is *disputed* — vindicating the planned empirical sweep.
- **Sharp theoretical gap:** finite-precision transformer ⊆ TC⁰/FO(M) *per pass*
  (Merrill & Sabharwal; Hahn); cross-pass state feedback is the documented escape —
  but the known proofs need *arbitrary precision*, so whether finite-precision
  reservoir state lifts the bound is **open** (paper must pose, not assert).
- **Novelty axis:** every prior recurrence-augmented transformer (Transformer-XL,
  Compressive, Universal, Block-Recurrent, Memorizing, RMT, RWKV, RetNet, S4/Mamba,
  Titans) uses *trained* recurrence carrying state *within a sequence*. None uses a
  *fixed-random* reservoir with state across *independent* passes — the Reservoir
  Agent's empty cell. Block-Recurrent independently confirms the "model learns to
  ignore the recurrent state" failure mode the plan worries about.
- **Honest caveat logged:** the reservoir-×-transformer and always-on-agent areas
  were not verification-complete (some returned arXiv IDs didn't resolve and were
  discarded, not cited); a citation-checked follow-up is queued in `todo.md` before
  any hard novelty claim is published. Reflected the one-line summary into
  `docs/index.html` (pillar 2).

## 2026-05-29 — Bootstrap 5: long-horizon todo.md written

Wrote `todo.md` from the literature gap: hypotheses H1–H4 (feasibility/non-destruction,
healthy-dynamics regime, state-readability, cross-pass behaviour change); experiments
tiered near-term (reservoir core, model surgery, spectral-radius dynamics sweep —
this session) / mid-term (harness fork PoC, benchmark+training-data design, readout+LoRA)
/ long-term aspirational (N-seed selection, always-alive runtime, Hermes scale-up);
the scoped theory section; the open questions carried from the review (citation-checked
novelty follow-up; the finite-precision TC⁰ question; edge-of-chaos at attention scale);
and the report shape.

## 2026-05-29 — Bootstrap 6: live & public + report-site polish

Repo is public at <https://github.com/EmmaLeonhart/reservoiragent>; Pages builds via
GitHub Actions and the site is live (HTTP 200) at
<https://emmaleonhart.github.io/reservoiragent/>. (The work-loop cron had already
fixed an earlier failed Pages deploy by auto-enabling Pages in the workflow.)

Report-site polish from user feedback ("repo link doesn't work", "needs a better
preview", "use the diagrams I added"):
- Fixed the GitHub repo link (was `href="#"`).
- Added Open Graph + Twitter-card meta and a purpose-built 1200×630 social-preview
  card (`docs/og-preview.png`) composed from the user's architecture diagram + title.
- Embedded the user's two canonical diagrams as figures with captions:
  `docs/diagram-architecture.png` (per-pass architecture + ensemble training) as the
  hero, and `docs/diagram-runtime.png` in a new "always-alive runtime" section.
  (Discarded an earlier hand-authored schematic in favour of the user's diagrams.)
- Provenance for the user-supplied screenshots noted in `data_lake/README.md`.

## 2026-05-29 — Bootstrap 7: bootstrap complete; real implementation queue set

Replaced the bootstrap `## Active` section with the real implementation queue,
decomposed from `todo.md` §B near-term (feasibility + dynamics): (1) scaffold pkg+CI,
(2) reservoir core TDD, (3) dynamics metrics TDD, (4) synthetic spectral-radius sweep,
(5) model surgery / reservoir injection into GPT-2-small with H1 regression, (6) real-
attention sweep + FINDINGS write-up, (7) ambitious reach (2-pass PoC + tiny N-seed,
per the user's "just to see if we can" todo note). Mirrored all seven into the task
tool (#7–#13). The three crons are kept running through the re-fill (written atomically),
and `## Always last` keeps them alive. The work-loop and the one-shot 8h kickoff cron
(`0bacbec1`, ~2026-05-30 04:24 local) both draw from the top of this queue.

**Bootstrap is complete.** From here the autonomous loop executes the implementation
queue; the next code change is the package/CI scaffold (item 1).

## 2026-05-29 — Report diagrams converted to real SVGs (was raster PNG)

User feedback: the page figures should be SVG, and the recovered SVGs render
black-boxed (correctly diagnosed as a theming artifact, not missing content). Root
cause: the recovered diagrams use Claude's CSS classes (`node c-teal`, …) + CSS
custom properties that are undefined standalone — and cairosvg/PDF don't resolve
`var()` or class selectors even though browsers do. `data_lake/retheme_diagrams.py`
now **bakes literal fill/stroke** onto every box (per the `c-*` colour) and resolves
every `var(--…)`, in the report's light palette matching the user's screenshots; it
also normalises a few glyphs (subscripts, heavy ✕) the headless/PDF fonts lack.
Outputs `docs/diagram-architecture.svg` (forward-pass) and `docs/diagram-runtime.svg`
from the recovered `-01`/`-02` sources; the page now embeds the SVGs and the
social-preview card is rebuilt from the crisp SVG render. Old raster diagram PNGs
removed. (Resolves queue item 9.)

## 2026-05-29 — Report PDF build fixed

`docs/report.pdf` was never produced (workflow only built it if `FINDINGS.md`
existed — it didn't — and installed no PDF engine). Wrote an honest in-progress
`FINDINGS.md`; `pages.yml` now builds the PDF via weasyprint in an isolated venv
(full Unicode) with `report-print.html` styling the paper theme, and fails loudly
rather than silently skipping. Verified: live `report.pdf` is HTTP 200,
`application/pdf`, `%PDF-1.7`, ~29 KB; Pages build green.

## 2026-05-29 — New diagram: reservoir-augmented residual stream (queue item 8)

`data_lake/build_residual_reservoir_svg.py` generates
`docs/diagram-residual-reservoir.svg` — an SVG recreation of the residual-stream
picture (`…preview-03.webp`: token streams + FFN-query/attention/FFN-value neurons
predicting "Paris") **with the reservoir added**: a fixed-random amber reservoir
column that joins the attention layer as extra keys/values (read W_in, write W_out),
with a recurrence loop and state persisting across passes. Embedded on the report
page under a new "How the reservoir enters attention" section. Resolves queue item 8.

## 2026-05-29 — Implementation 1: package + tests + CI scaffold

First implementation step. Created the package skeleton: `pyproject.toml`
(`reservoir-agent`, src layout; core deps numpy+matplotlib kept light so CI is fast,
with torch+transformers behind a `models` extra for the later model-surgery work),
`src/reservoir/__init__.py`, `scripts/run.py` (entry point with `--version`),
`tests/test_smoke.py`, and `.github/workflows/ci.yml` (setup-python →
`pip install -e ".[dev]"` → `pytest` → smoke-run the entry point). Verified locally:
`pytest` 2 passed; `python scripts/run.py --version` → 0.0.1. (Used pyproject with a
`models` extra rather than a single requirements.txt with torch/transformers, so CI
stays fast and green; the heavy deps install only where needed.) Resolves queue item 1.

## 2026-05-29 — Implementation 2: echo-state reservoir core (TDD)

`src/reservoir/echo_state.py` — `EchoStateReservoir`: fixed sparse `W_r` rescaled to a
target spectral radius, fixed `W_in`, leaky update
`r(t)=(1−a)·r(t−1)+a·tanh(W_r·r(t−1)+W_in·x(t))`; `step`/`run`/`reset` +
`spectral_radius_actual`. Tests written first (`tests/test_echo_state.py`, 7):
spectral-radius scaling accurate to 1e-6; echo-state contraction (ρ<1, zero input →
forgets initial condition, decays to null state); state bounded/finite even at ρ=1.5;
shapes/dtype; seed reproducibility; leak-rate=0 freezes state; sparsity controls
density. Full suite 9 passed locally. Resolves queue item 1 (was item 2).

## 2026-05-29 — Implementation 3: reservoir dynamics metrics (TDD)

`src/reservoir/metrics.py` — `state_variance`, `saturation_fraction` (|r|>thr),
`participation_ratio` (effective dimensionality = (Σλ)²/Σλ² of the unit covariance),
and `trajectory_distinguishability` (mean per-step RMS distance between two
trajectories). Tests first (`tests/test_metrics.py`, 5) each against a known answer:
saturation 0.75 on a constructed array; variance 0 for constant + matches numpy;
PR≈1 for a rank-1 trajectory and PR≈K for K independent units; distinguishability
1.0 for zeros-vs-ones and 0 for identical. Full suite 14 passed locally.
Resolves queue item 1 (dynamics metrics).

## 2026-05-29 — Implementation 4: spectral-radius dynamics sweep (first result)

`src/reservoir/sweep.py` + a `sweep` subcommand in `scripts/run.py` drive a fixed
reservoir across ρ ∈ [0.1, 2.0] on synthetic input, logging metrics to
`results/sweep_synthetic.json` and a figure to `docs/sweep_synthetic.png`.
`tests/test_sweep.py` (5) assert the qualitative physics.

**First measured result (H2):** the echo state property breaks **sharply at ρ ≈ 1**.
Using an *autonomous* (zero-input) init-forgetting probe, two random initial states
converge identically (forgetting = 0.000) for ρ ≤ 0.9, then diverge abruptly
(0.05 → 0.38 → 0.95) for ρ ≥ 1.0 — the classic edge-of-chaos boundary, now measured
in this injection setup. Saturation and participation ratio rise smoothly with ρ.
Important nuance (kept in the write-up, not papered over): under unit-scale *input*
drive the reservoir forgets its init across all ρ (strong input enforces the ESP), so
the ρ ≈ 1 boundary is the regime that governs **unprompted, input-free passes** — where
the agent runs on reservoir state alone. The first probe (driven) masked this; switched
to the autonomous probe to expose the boundary correctly. Figure + result reflected in
the `docs/` Findings section and pillar 3. Full suite 19 passed locally.
Resolves queue item (spectral-radius sweep).

## 2026-05-29 — Implementation 5: model surgery / reservoir injection (H1 ✓)

`src/reservoir/inject.py` — `ReservoirInjectedLM`: loads a pretrained GPT-2, hooks a
mid-depth block so its hidden states drive the fixed reservoir (read via `W_in`) and
the reservoir state is written back into the residual stream via readout `W_out`
(`h' = h + W_out·r(t)`). Feasibility-scoped injection (residual-stream write; the
KV-append variant is the ambitious reach). `tests/test_inject.py` (3, torch+transformers
`importorskip` so the light CI job skips them, run locally on `sshleifer/tiny-gpt2`):

- **H1 non-destruction ✓** — with `W_out=0` the injected model's next-token logits are
  identical to vanilla GPT-2 (`allclose`, atol 1e-5): the architecture degrades
  gracefully to the base model.
- The injection is live — a nonzero `W_out` changes the logits.
- Reservoir state **persists across independent forward passes** (a genuine time axis):
  state after two passes ≠ after one.

Full suite 22 passed locally (3 inject tests skip in CI). Resolves queue item
(model surgery, H1).

## 2026-05-29 — Implementation 6: sweep on real GPT-2 activations + FINDINGS write-up

Added a stream-driven sweep (`run_sweep_stream`/`measure_point_stream` in
`src/reservoir/sweep.py`, z-scoring the input) and a real-activation extractor
(`extract_layer_stream` in `inject.py`), plus a `sweep-real` subcommand. Ran it on
real GPT-2 mid-layer activations (two 198-token streams, K=150) →
`results/sweep_real.json` + `docs/sweep_real.png`.

**Results (written into `FINDINGS.md` + `docs/` Findings):**
- The **ρ ≈ 1 echo-state boundary survives on real activations** — even sharper
  (autonomous forgetting 0.000 for ρ ≤ 0.9 → 0.10 at ρ = 1 → ~0.95 above).
- **Real activations over-drive the reservoir**: saturation ≈ 0.86 (vs < 0.15 on
  synthetic) and participation ratio ≈ 0.41·K (vs ~0.05·K) — so input scaling must be
  tuned down for transformer-scale injection (the plan's anticipated open question).
- FINDINGS Results now reports H1 (non-destruction ✓) and H2 (the dynamics regime);
  Limitations note the residual-stream (vs KV-append) injection, untrained readout, and
  untuned input scaling.

A real-data subtlety handled honestly: the autonomous forgetting transition is sharp
for large K but noisy in the ρ∈[1,1.6] band for small K (two inits can share a basin),
so the stream-sweep test pins the two ends (forgets at ρ=0.4, retains at ρ=2.0) rather
than over-specifying the transition width — no test was weakened to pass. Full suite 23
passed locally. Resolves queue item (real sweep + write-up).
