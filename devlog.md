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

## 2026-05-29 — Implementation 7: ambitious reach (time-axis PoC + N-seed proxy)

`src/reservoir/harness.py` — `two_pass_dependence` and `select_seed_by_dynamics`, plus
`alive` and `nseed` subcommands. `tests/test_harness.py` (2; seed proxy in CI, two-pass
demo torch-gated).

**Measured:**
- **Time axis is behavioural** — the same prompt after different history (reservoir
  state carried across passes, small random readout) shifts GPT-2 next-token logits by
  L2 ≈ 22 (`results/alive_poc.json`). A stateless transformer cannot do this.
- **Seed-selection mechanism works; pre-training signal is weak** — ranking 8 fixed
  reservoir seeds by an untrained dynamics proxy on real activations gives only a ~0.02
  spread (`results/nseed.json`, `docs/nseed.png`), so the real selection signal likely
  needs training. Mechanism in place; usefulness compute-gated.

**Named as not done (compute-gated):** full N-seed LoRA fine-tuning + benchmark
selection; productionized always-alive runtime (scheduler/idle-timer/output-gate);
KV-append injection; agent-scale (Hermes). All written into FINDINGS.

Full suite 25 passed locally. **This drains the implementation queue** — the
feasibility + dynamics study is complete; H1 + H2 answered, report + PDF live, PoC
honest. Resolves queue item (ambitious reach).

## 2026-05-29 — Queue re-fill: Round 2 (deepen + polish)

Round 1 drained, so refilled `queue.md` from `todo.md` (§B mid-term, §C theory, §D open
questions) + the user's site-polish ask, per the user's "keep refilling as we move
through it" instruction. Six items, mirrored to the task tool (#14–#19): (1) report-site
polish via the newly-installed `frontend-design` plugin (fix the janky title); (2)
input-scaling tuning sweep (follow-up to the real-activation over-saturation finding);
(3) scoped theory section; (4) train a readout for H3; (5) KV-append injection variant;
(6) citation-checked novelty follow-up. Crons kept running through the atomic re-fill.
Mid-round, per a follow-up from the user, inserted the **agent harness (always-alive
runtime)** as priority item 2 — to be built and exercised on the untrained injected
model before any (compute-gated) fine-tuning.

## 2026-05-29 — Round 2.1: report-site polish (frontend-design)

Fixed the janky tab title (`<title>` was "reservoiragent" → "Reservoir Agent — a
transformer with a genuine time axis") and gave the report an editorial masthead:
a distinctive display serif (**Fraunces**, via Google Fonts) for the headings, a
tracked uppercase kicker ("Reservoir computing × transformers · research report")
above the title, and larger/tighter headline treatment. Kept the warm "paper" theme,
dark-mode variant, and all content/figures/OG-meta/PDF+repo links (verified: 6 figures,
8 OG tags, PDF + repo links intact, all tags balanced). Resolves queue item (site polish).

## 2026-05-29 — Round 2.2: agent harness — the always-alive runtime

Built the stateful-agent runtime on the *untrained* injected model (the substrate
fine-tuning will plug into), per the user's "make the harness and test it before fine
tuning" priority. `src/reservoir/runtime.py`: `ContextBuffer` (never wiped),
`topk_entropy`/`ConfidenceGate` (emit-vs-silence), `checkpoint_state`/`restore_state`,
and `AliveAgent` (prompted + unprompted passes, reservoir state persisting across both).
`scripts/run.py agent` runs a scripted session → `results/agent_session.json`.
`tests/test_runtime.py` (6): entropy bounds, gate, context buffer, state checkpoint
round-trip (CI), and torch-gated AliveAgent tests — incl. **an unprompted pass updates
the reservoir state with no new input** and agent checkpoint/restore.

Demo: 5 interleaved prompted/unprompted passes, reservoir |r| evolves continuously
through the idle ticks. Named plainly: on the untrained model the gate keys off the
base model's entropy so emit/silence + the generated text (GPT-2 babble) aren't yet
meaningful — the harness is the mechanism; a real self-initiation policy needs the
trained readout/LoRA. FINDINGS gained a "## The always-alive runtime (harness)" section.
Full suite 31 passed locally. Resolves queue item (agent harness). NOT compute-gated;
fine-tuning is the next, compute-gated, step the user will start.

## 2026-05-29 — Round 2.3: input-scaling tuning sweep (over-saturation fixed)

Follow-up to the real-activation over-saturation finding. Added `run_scaling_sweep` +
`plot_scaling` to `src/reservoir/sweep.py` and a `sweep-scaling` subcommand; measure_point_stream
now records `input_scaling`. `tests/test_sweep.py` gained `test_input_scaling_controls_saturation`.
Ran on real GPT-2 activations (ρ=0.95, K=150) → `results/sweep_scaling.json` +
`docs/sweep_scaling.png`.

**Result:** saturation is a clean sigmoid in the input scaling — near zero below ≈ 0.05,
crossing 0.5 at ≈ 0.24, ~0.86 at unit scale — while input separation and dimensionality
stay high. **Sweet spot ≈ input scaling 0.08–0.24** (saturation 0.08–0.49, separation
1.03–1.26, PR ≈ 0.39·K): real attention activations should be fed at ~¼–⅒ of unit scale,
not 1.0. Folded into FINDINGS (H2 + Limitations) and the docs Findings (figure). Full
suite 32 passed locally. Resolves queue item (input-scaling sweep).

## 2026-05-29 — Round 2.4: theory section (scoped formal claims)

Wrote the theory into `FINDINGS.md` (## Theory) + a condensed `docs/` pillar section,
grounded in `literature/REVIEW.md`: (1) genuine time dimension (r(t) causal across
passes, decoupled from context — not positional encoding); (2) the expressivity gap
(finite-precision transformer ⊆ TC⁰/FO(M) per pass; cross-pass state the documented
lever; reservoir ∈ Siegelmann–Sontag) **stated with the arbitrary-precision caveat and
posed as the project's central OPEN question, not asserted**; (3) the organism analogy
as one bounded paragraph (structural-capacity claim only, no general-intelligence
claim). No new empirical claims. HTML tags verified balanced. Resolves queue item (theory).

## 2026-05-29 — Round 2.5: train a readout for H3 (state is informative)

`src/reservoir/tasks.py` — the delay-memory task + a closed-form ridge readout
(`fit_ridge`, `r2_score`, `delay_memory_curve`, `memory_capacity`, `plot_memory_curve`)
and an `h3` subcommand; all numpy/CPU → CI-testable. `tests/test_tasks.py` (3): ridge
recovers a linear map, r2 bounds, and the reservoir-has-memory-baseline-lacks check.

**H3 result:** a linear readout on the reservoir state recovers the input from **~18
steps back at R²>0.5** (R²≈1 to ~12; total linear memory capacity **17.4**), while the
**stateless baseline** (same readout on the current input) scores **exactly 0** at every
delay ≥ 1 — i.i.d. inputs carry no information about their own past, so the answer is
provably in the carried state, not the input. `results/h3_memory.json` +
`docs/h3_memory.png`; folded into FINDINGS Results (### H3) + docs Findings. Named the
limit plainly: this is the mechanism on a clean synthetic task; a *semantic* agent task
(unresolved thread / elapsed time) needs the readout trained through the LM (future).
Full suite 35 passed locally. Resolves queue item (H3 readout).

## 2026-05-29 — Round 2.6: KV-append injection mechanism (+ documented integration blocker)

The richer injection — reservoir nodes as extra attention keys/values. transformers 5.4
GPT2Attention dispatches K/V through a `Cache` + `attention_interface` with no hook
exposing the internal tensors, so a clean wiring needs an eager-path `forward` override
— genuinely invasive. Per the hard rail, implemented + tested the **mechanism in
isolation** and left a **precise documented blocker** for the live-GPT-2 wiring rather
than a fragile patch.

`src/reservoir/kv_inject.py` (numpy → CI-testable): `scaled_dot_product_attention`,
`causal_mask`, `attention_with_reservoir` (tokens ++ reservoir nodes), and
`reservoir_nodes_from_state`. `tests/test_kv_inject.py` (4): **H1 = gated-off (masked)
reservoir is identical to base causal attention**; active reservoir changes the output;
and the subtle one — appending *zero* value vectors still dilutes the softmax, so H1
must be a *masking* property, not a zero-weights one. `GPT2_INTEGRATION_BLOCKER`
documents the exact remaining step; moved to `todo.md` §B. FINDINGS Limitations updated.
Full suite 39 passed locally. Resolves queue item (KV-append: mechanism done, integration
documented-blocked).

## 2026-05-30 — Round 2.7: citation-checked novelty follow-up (verified)

Ran a focused, adversarially-verified deep-research pass (100 agents; 18 sources → 25
claims verified, 18 confirmed / 7 killed) on the three areas the first lit-review left
open. **Verdict: the project's core combination is genuinely novel against the verified
prior art.** The four real close items each fail ≥1 of the three load-bearing axes
(pretrained-injection / fixed-random reservoir / cross-pass state): **Reservoir
Transformers** (Shen et al., ACL 2021, arXiv:2012.15045), **Echo State Transformer**
(2025, arXiv:2507.02917 — attention-over-reservoir but from-scratch + trained leak rates
+ within-sequence), **Echo Flow Networks** (2025, arXiv:2509.24122 — fixed reservoir +
trainable backbone, within-sequence, time-series), **FreezeTST** (2025, arXiv:2508.18130
— frozen random feature expansion, from-scratch, time-series). Foundational canon
(DeepESN, arXiv:2002.12287) is what the project builds on.

Folded into `literature/sources.md` §4 (verified, per-source) + §5 (always-on:
verified-absent in the searched set) and tightened `REVIEW.md` §4 from "provisional" to
"verified, with caveats" (recent 2025 preprints; absence-in-set ≠ global absence; EST
blurs trained-vs-fixed; unreliable IDs discarded not cited). Resolves queue item (novelty
follow-up) — **drains the Round 2 work items**; next is the compute-gated experiments
(N-seed selection + GPT-2 LoRA), pulled into the queue from todo.md per the user.

## 2026-05-30 — Round 3.1: N-seed selection + seed-pre-selection proxy (negative result)

`src/reservoir/selection.py` + `nseed-select` subcommand: train each of N fixed seeds'
readout on the delay-memory task, rank by memory capacity, keep the best; and Spearman-
correlate a cheap untrained dynamics proxy (participation ratio) against the trained
ranking. `tests/test_selection.py` (3, numpy/CI). Ran N=12, K=200 →
`results/nseed_select.json` + `docs/nseed_select.png`.

**Result:** seeds genuinely differ (memory capacity 17.4–20.7, ~19% spread) so the
N-seed selection is worth doing; but the **seed-pre-selection proxy fails** — the
untrained participation ratio has **no rank correlation** with trained memory capacity
(**Spearman ρ = 0.08, p = 0.80**, n=12). A clean negative answer to the plan's open
"can dynamics pre-select seeds before training?" question (for this proxy): no, the
training can't be shortcut this way. FINDINGS Results + docs Findings updated. Full suite
42 passed locally. Resolves queue item (N-seed selection).

## 2026-05-30 — Round 3.2: real GPT-2 LoRA fine-tune on GPU (compute-gated)

The culminating compute-gated run. `src/reservoir/torch_inject.py` — a *differentiable*
reservoir injection (torch reservoir; W_r/W_in fixed buffers; trainable zero-init W_out
readout) into GPT-2, with peft LoRA on the attention projections; `train_finetune` +
a `finetune` subcommand. `tests/test_torch_inject.py` (1, torch+peft-gated): the pipeline
reduces loss.

**Ran on local CUDA (RTX 4070):** 3 reservoir seeds × 60 steps, training loss **6.3 →
0.85–1.1**, 491,520 trainable params (LoRA + W_out), best seed selected by trained loss
(`results/finetune.json`). The full pipeline — inject, freeze backbone, train W_out +
LoRA, select across seeds — runs end-to-end on the real architecture. W_out zero-init
keeps H1 at step 0.

**Honest boundary (named):** the hook ticks once per forward pass, so this single-forward
fine-tune exercises the *training machinery*, not the reservoir's *cross-pass* value —
that needs a multi-pass differentiable harness (backprop-through-passes on a
reservoir-requiring cross-context task), now queued in `todo.md` §B as the next compute
step. FINDINGS gained a "## Compute-gated: a real LoRA fine-tune on GPU" section. Full
suite 43 passed locally. Resolves queue item (GPT-2 LoRA fine-tune) — **drains the
compute-gated queue.**

## 2026-05-30 — Phase H planned: port to Hermes + make the behaviour real (A–E)

State assessment for the user (verified in code, not memory): the repo's mechanisms work
**only on GPT-2** — both injection modules hardcode `model.transformer.h`; there is **no
Hermes / tool-call / real-harness code anywhere**; and the desired behaviour isn't real
yet (the silence gate keys off base entropy, untrained; the reservoir's cross-pass value
is untrained, so the model would currently ignore it). So we are **not** ready to seriously
train — the user's instinct was right.

Per the user's decisions (port to Hermes 3B now; fork the real Hermes harness) and "do
A–E in order", re-filled `queue.md` with **Phase H**: (A) generalize injection to
Llama/Hermes arch + verify H1 on a tiny Llama; (B) load **Hermes-3-Llama-3.2-3B** 4-bit on
the 4070 + verify H1; (C) multi-pass differentiable harness (cross-pass training — the
load-bearing condition); (D) trained silence policy; (E) real Hermes-harness fork
(tool-calling). Mirrored to tasks #23–27. Preconditions confirmed (8.6 GB VRAM,
bitsandbytes 0.49.2, peft, accelerate, 768 GB free). Crons kept running.

## 2026-05-30 — Phase H · A: injection generalized to Llama/Hermes arch

`src/reservoir/_arch.py` — `decoder_blocks(model)` (locates `transformer.h` for GPT-2 vs
`model.model.layers` for Llama, unwrapping peft) + `hidden_size(config)` (`n_embd` vs
`hidden_size`). Refactored `inject.py` (+ `extract_layer_stream`) and `torch_inject.py`
to use them instead of the hardcoded GPT-2 paths. `tests/test_inject.py` gained
`test_h1_holds_on_llama_architecture` — **H1 verified on a tiny Llama**
(`hf-internal-testing/tiny-random-LlamaForCausalLM`) as well as tiny-gpt2 (zeroed readout
→ logits identical). The injection is now architecture-agnostic, so it can host Hermes
(Llama-3.2). Full suite 44 passed locally. Resolves Phase H item A.

## 2026-05-30 — Phase H · B: Hermes-3-Llama-3.2-3B loads + H1 holds on the GPU

Added 4-bit (bitsandbytes nf4) support to `ReservoirInjectedLM` (`load_in_4bit`).
`scripts/hermes_h1.py` loads **NousResearch/Hermes-3-Llama-3.2-3B** in 4-bit (28 layers,
d_model=3072, injection at layer 14) and checks H1 memory-frugally (one copy: zeroed
readout vs hook removed). **Result: H1 holds exactly — `max|diff| = 0.00e+00`, peak VRAM
2.35 GB** (`results/hermes_h1.json`). So the architecture transplant is non-destructive
on the real Hermes, with ample VRAM headroom on the RTX 4070 for LoRA + training.
FINDINGS gained a "## Porting to the real target: Hermes (Phase H)" section. Resolves
Phase H item B. (Also added — pending its GPU run — the C cross-pass pipeline:
`src/reservoir/crosspass.py` + `crosspass` subcommand + smoke test.)

## 2026-05-30 — Phase H · C: cross-pass training — HONEST NEGATIVE RESULT

The load-bearing experiment, run on the GPU. The multi-pass differentiable harness
**works mechanically** (backprop through two passes, training W_out + LoRA on a
secret-word-recall-after-context-wipe task). **But the model does not learn to use the
reservoir:** across mean/last-token drive and mid/last-layer injection, up to 500 steps,
the stateful model and the stateless baseline both reach **chance (0.17 = 1/6)** with loss
at the marginal (~ln 6). stateful ≈ baseline ⇒ the carried state contributes nothing —
the **Block-Recurrent "learns to ignore the recurrent state" failure mode, reproduced
empirically.** Diagnosis (not a bug): a single additive readout driven by a pooled hidden
preserves coarse *process* state (cf. H3) but not *which specific word* appeared.

Reported honestly in FINDINGS ("## C: cross-pass training") + docs Findings (figure
`docs/crosspass.png`, both bars at chance). Redirects the work: KV-append
(content-addressable attention to reservoir nodes) and/or temporal/process tasks — added
to `todo.md` §B. **This changes the calculus for D/E** (the desired behaviour does not yet
work), so I paused to checkpoint with the user rather than barrel into D/E on a broken
core. C is built + ran + reported; not a faked pass.

## 2026-05-30 — Phase H · C resolved: content-addressable (KV-append) injection works

The user chose to build the KV-append fix the negative pointed to. `src/reservoir/kv_live.py`
— `TorchReservoirPrefixInjectedLM`: the robust content-addressable injection — the
reservoir state is projected (trainable `W_res`) into **prefix pseudo-tokens prepended to
the sequence** (via `inputs_embeds`), so attention **reads them content-addressably** with
no fragile mid-layer attention surgery (the standard mask/positions handle the extended
length; prefix stripped from the output). A read hook still ticks the reservoir each pass.
`run_cross_pass_kv` + `crosspass --mode kv`; smoke test added.

**Result (decisive POSITIVE, on the same task that failed additively):** stateful recall
**1.00** (loss → 0.02) vs stateless baseline **0.17** (chance). So the additive injection
fails (reservoir ignored) but the content-addressable one makes the model **use** the
cross-pass reservoir state perfectly. **The negative→positive arc is the core finding:
the injection design is the decisive factor — the reservoir must be *attended to*, not
*added*.** FINDINGS "## C" rewritten as the resolved arc; docs Findings + figure updated.
Full suite 46 passed locally. Resolves the KV-append item.

## 2026-05-30 — Phase H: Hermes recall transfer — NEGATIVE (honest), + report-accuracy fixes

**Hermes validation (task #28): negative, reported plainly.** The content-addressable
cross-pass recall (100% on GPT-2) did **not** transfer to Hermes-3-Llama-3.2-3B in 4-bit
across two principled attempts (input scaling 0.5→0.1 per the H2 result; 300→600 steps):
both stateful and baseline at **chance (0.17)**, and — unlike GPT-2 (loss → 0.02) — the
**loss did not converge** (plateaued ~2.9). So it is not just the over-saturation knob;
the LoRA-on-4-bit-Hermes + prefix setup is not optimizing recall in this budget. Likely
needs more steps / higher LR / full-precision / a stronger injection — logged as an open
transfer step in `todo.md` §B, FINDINGS "## C" updated. Per my commitment to the user, I
stopped after two principled attempts rather than hacking. Added model-tagged outputs so
the Hermes figure (`docs/crosspass_hermes-3-llama-3-2-3b.png`) does not clobber the GPT-2
win figure.

**Report-accuracy fixes (user feedback):** regenerated `docs/og-preview.png` without the
"grounded in a literature review" / "feasibility study" lines (the weird thumbnail text);
updated `docs/index.html` lede + always-alive-runtime section + meta/OG/Twitter
descriptions to drop the stale "feasibility / long-term / aspirational" framing — the
mechanism is demonstrated and the Hermes port + harness fork are in progress.

## 2026-05-30 — Phase H: Hermes recall — bf16 also negative, but well-diagnosed (not a bug)

Per the user's choice (try non-4-bit), ran a third Hermes attempt: **bf16 (non-4-bit) +
LoRA, input scaling 0.1, LR 3e-3, 600 steps**. Still **chance (0.17), stateful ≈ baseline**,
loss plateau ≈ 2.8 — the *same* plateau as 4-bit, so **quantization is not the cause**.

A focused gradient diagnostic on a tiny Llama **rules out a bug**: the reservoir state
updates each pass (norm 0→0.14) and gradients flow to `W_res` (‖∇‖≈0.016) and LoRA
(Σ|∇|≈3.0). So the injection is correctly wired on Hermes; the failure is a genuine
**bootstrapping/scale difficulty** (prefix signal diluted through 28 layers + a 3B
instruction model's strong priors), not a defect. Stopped the GPU grind after 3 attempts
+ the diagnostic, per the commitment. FINDINGS "## C" + `todo.md` updated with the
diagnosis and concrete routes (curriculum / more steps / stronger coupling / unfreeze).
**Net: the core claim holds decisively on GPT-2; on Hermes the mechanism is
verified-wired but recall is not yet trained to converge.**

## 2026-05-30 — Phase H · D: trained silence policy works (sees history)

Implemented a **learned silence gate** (`src/reservoir/silence.py`) to replace the
arbitrary base-entropy gate. It addresses an "unresolved thread" task (speak if a
trigger occurred within the last N passes).

- **Result:** Reservoir gate reaches **F1 = 0.48** (P=0.71, R=0.36) while the stateless
  baseline stays at **F1 = 0.03** (P=1.00, R=0.02). The carried state allows the agent
  to make a meaningful decision to keep speaking even after the trigger is gone from the
  input — a structural impossibility for a stateless model. Added `scripts/run.py silence`.

## 2026-05-30 — Feasibility phase complete; website and PDF report updated

Updated `FINDINGS.md`, `README.md`, and `docs/index.html` to reflect the completion of
the feasibility study.

- Added results for **Cross-pass recall** (decisive win on GPT-2 with KV-append;
  scale difficulty on Hermes 3B) and the **Trained silence policy** (H4).
- Added figures to the website, including the Hermes 3B recall plateau.
- Updated status to "Feasibility Study Complete". Pushing to trigger PDF generation.


## 2026-05-30 — Phase H · D: trained silence policy + the conceptual point (+ CI fix)

The user prematurely committed an unverified silence WIP (8c8575f), leaving a FAILING
test / red CI. Fixed it honestly rather than papering over:

- `src/reservoir/silence.py`: the silence gate now (a) uses a **clean dedicated trigger
  channel** and a **past-only** speak window (the cue is strictly in the past, so the task
  is genuinely reservoir-requiring), and (b) **tunes its decision threshold on the train
  set** (part of training the gate). Result: reservoir-state gate **F1 ≈ 0.96** (P 0.93,
  R 1.00); the stateless gate collapses to **F1 ≈ 0.34** (always-speak — it cannot see the
  past trigger). A stateless model cannot implement selective silence at all.
  `scripts/run.py silence` + `docs/silence.png`. Reconciled the duplicate `silence`
  command (the user had also added one).
- Fixed a **flaky** `test_finetune_pipeline_reduces_loss` (tiny model, too few steps →
  borderline) by giving it enough steps to converge reliably. Full suite **49 passed**.

**The user's conceptual point — documented in FINDINGS "## D":** the *default* should be
to **respond** (a decayed/empty reservoir ≈ the base model's prior); **silence** should
attach to an *active, novel* reservoir state (the natural handle to fine-tune a new
"still processing" behaviour onto); the **echo state property** empties the reservoir over
time, so the agent **reverts to baseline responding** as activity subsides; and teaching a
pretrained model this new behavioural axis is aggressive **brain surgery** — genuinely
hard (the same difficulty that kept Hermes recall from bootstrapping). Mirrored a condensed
version into the docs Findings. Resolves Phase H item D.

## 2026-05-30 — Phase H · E core: Hermes-format agentic harness

`src/reservoir/hermes_harness.py` — the Hermes-format layer for the real target:
`render_chatml`, `tools_system_prompt` (function-calling system message),
`parse_tool_calls` / `format_tool_response` (`<tool_call>`/`<tool_response>` XML), and a
`HermesHarness` that drives the reservoir-injected model through the agentic tool loop
(parse → execute registered tool → feed `<tool_response>` back → repeat to a budget).
`tests/test_hermes_harness.py` (5): ChatML round-trip, tools system prompt, tool-call
parse (incl. malformed-skip), tool-response format, and a torch-gated wiring smoke
(`HermesHarness.chat` runs end-to-end on tiny-gpt2). Full suite 54 passed.

**Named plainly as NOT a full Nous fork** (`HERMES_HARNESS_REMAINING` + todo.md): streaming
+ exact Nous scaffolding; fusing the unprompted/idle pass + trained silence gate into the
loop; the regression-vs-vanilla-Hermes generation check (a Hermes GPU run); multi-tool
routing. This is the Hermes-format core + tool loop the production fork plugs into.
Resolves the bounded part of Phase H item E; the full fork is queued in todo.md.

**Phase H wrap:** A (Llama port) ✓, B (Hermes 3B H1) ✓, C (cross-pass recall: 100% on
GPT-2, Hermes transfer open + diagnosed) ✓, D (trained silence policy + the conceptual
"brain surgery is hard" point) ✓, E (Hermes-format harness core) ✓. The two open threads
(Hermes recall transfer; full Hermes harness fork) are documented in todo.md.

## 2026-05-30 — clawRxiv resubmission fixed (`/revise`) + paper revised for review (v2/v3)

**The bug that would have broken resubmission.** The submit script
(`scripts/submit_clawrxiv_paper.py`) used clawRxiv's old `supersedes` field for
revisions. clawRxiv has since migrated revisions to `POST /api/posts/{id}/revise`;
the old `POST /api/posts` + `{"supersedes": id}` body now returns **HTTP 409**
("already been revised" / "duplicate detected"). So our *first* submission (post 2680)
worked, but a *re-submission* would have 409'd. Found by cross-checking the Sutra repo,
whose `paper_submit_and_fetch.py` carries extensive comments documenting exactly this
migration and the self-heal paths around it.

**The fix (ported from Sutra, adapted to this repo).** Rewrote the submit script:
first-ever submission → `create_post`; a pinned `.post_id` → `revise_post`
(`/api/posts/{id}/revise`); HTTP 409 → follow `data.duplicateId` to the canonical post
and revise it (re-pinning `.post_id`); HTTP 404 on revise → probe `create_post` to
elicit the dedup 409; and a STOP-NEW-CHAINS guard that refuses to pin `.post_id` to an
orphan post (a *successful* create while an id is pinned = a new unchained post, not a
revision) and exits 1 so CI goes red. Unit-tested with no network in
`tests/test_submit_clawrxiv.py` (6 tests; full suite 60 passed).

**Confirmed working end-to-end.** Triggered the submit workflow: it logged
`Revising existing post 2680 (POST /api/posts/2680/revise)` → new version **post 2682**
(`paper_id 2605.02682`), and the bot committed `.post_id=2682` back. A second trigger
revised again (2682 → **2683**), confirming the chain advances and `.post_id` pins
forward each time. Post 2682 verified live and public with the correct title. This is
the session's main goal met: **a genuine review came back (so the pull side works), and
re-submitting a new revision now actually works (the previously-broken side).**

**Paper edits (minor, responding to the "Weak Reject" from Gemini 3 Flash, post 2680).**
Added a "Scope, and what this study does and does not claim" section to `FINDINGS.md`:
the tasks are minimal mechanism-isolating probes (not agentic demos); the TC⁰/FO(M)
argument is motivation, not a proven result; the Hermes-3B non-convergence and the
KV-append HuggingFace-integration blocker are limitations (the latter now flagged as a
reproducibility limit); and the failed N-seed proxy means selection costs real
trial-and-error. Kept the submit-script `ABSTRACT` in sync.

**Docs.** `reproduce-report` SKILL.md gained a clawRxiv submit/revise section (it stays
the *replication* skill — this documents the publish loop as part of reproducing the
full pipeline); `paper/README.md` updated to the `/revise` mechanism.

## 2026-05-30 - Review-response notes for clawRxiv post 2680 (Weak Reject)

Work-loop tick. queue.md empty; GPU items (A1 Hermes transfer, B1 N-seed
predictor) blocked on the 6-hour kickoff cron; C2 (respond to review) is a
product decision reserved for the user. Did the autonomously-safe slice of C2:
wrote paper/reviews/post2680_response_notes.md mapping each of the reviewer's
five cons to where FINDINGS.md already addresses it. Finding: none of the cons
contradict the paper - every one is an already-stated limitation; all
substantive fixes are GPU-gated/research-open; the only non-GPU lever is an
optional framing edit, left for the user since it triggers a published /revise
resubmission. Did NOT touch the published FINDINGS.md. Verified the clawrxiv
submit unit tests (tests/test_submit_clawrxiv.py) pass: 8 passed. Verified via
git show + grep that suspected stray editorial cruft in FINDINGS.md was glitch
contamination of tool output, not actual file content - the published paper is
clean.

## 2026-05-30 - clawRxiv 30-min reconcile tick

Re-pull triggered for post 2680 (gh workflow run clawrxiv.yml); the CI job holds
the CLAWRXIV_API_KEY secret. Standing review on record: "Weak Reject" (Gemini 3
Flash) at paper/reviews/post2680_review2680.json; its cons are already mapped in
paper/reviews/post2680_response_notes.md. No new review claimed (endpoint
null/404 = nothing new, not invented). Sutra-mechanism reconcile and cron-set
verification were DEFERRED to the next work-loop tick: tool-output rendering was
dropping this tick and the rails forbid blind mutation of publish-capable
scripts or blind cron edits. The deferred steps are itemized in queue.md.

## 2026-05-30 - clawRxiv reconcile completed (Sutra port + puller tests)

Work-loop tick. Drained the deferred reconcile block from queue.md with legible
output this time.

Verification:
- clawRxiv runs all green (gh run list --workflow=clawrxiv.yml): the
  workflow_dispatch submit/revise runs succeeded.
- No new review: paper/reviews/ still holds only post2680_review2680.json
  (Weak Reject). Endpoint live-checked: GET /api/posts/2680/review = HTTP 200
  (singular), /reviews = HTTP 404 (plural) - confirms our singular path.

Sutra reconcile (read pull-reviews.yml, submit-papers.yml, pull_all_reviews.py,
paper_submit_and_fetch.py as they stand):
- Endpoint + Bearer auth: identical to ours (singular /review). No change.
- Submit-side glitch fixes (/revise endpoint, duplicateId follow, 404-probe,
  STOP-NEW-CHAINS guard): already ported into our submit_clawrxiv_paper.py
  earlier this session. Sutra's abstract-truncation glitch cannot hit us - our
  ABSTRACT is a hardcoded literal, not regex-extracted from the paper body.
- Filename/dedup: kept ours (post{id}_review{review_id}.json, keyed by review
  id) over Sutra's v{N}_post{id} counter - ours dedups revision-chain reviews
  more precisely for a single paper. Did NOT migrate.

Real gap fixed (not a blind copy): our pull_clawrxiv_reviews.py had no tests and
a narrower readiness check than Sutra's. Extracted the response-shape
normalization into a pure extract_reviews() function, broadened its review
markers to match Sutra (rating/recommendation/review/body/content/summary), and
made it STRICTER on bare objects so an error envelope ({"message":"Server
Error"}) is no longer mis-saved as a review. Added tests/test_pull_clawrxiv.py
(10 tests, offline). Full suite: 154 passed, 7 skipped. CronList verified all
four jobs healthy (:03/:15/:42 + GPU-kickoff one-shot).

## 2026-05-30 - cleanvibe update check (v1.14.0; no revisions)

Ran the weekly cleanvibe-update-check skill (CLAUDE.md `## Skills` showed last
check = `never`). Fetched https://cleanvibe.emmaleonhart.com/updates.md:
current cleanvibe is v1.14.0 (2026-05-30), which lists exactly six skills
(emergency-stop, cron-is-local, autonomous-loop, queue-driven-workflow,
writing-style, cleanvibe-update-check) — all already vendored in `.claude/skills/`.
Nothing newer than v1.14.0, so no SKILL.md content was revised. Updated the
last-check date in CLAUDE.md from `never` to `2026-05-30`. No code change.

Also this session (scheduling/heartbeat only, no commits before this): started
the three session crons (work-loop :03 `c0bf227a`, auto-flush :15 `c920f0d6`,
status-report :42 `115b16bb`) — none were running on session start — and
scheduled a one-shot midnight kickoff (`dd9ba085`, 00:00 2026-05-31) for the
Hermes 3B cross-pass recall transfer (user-chosen), which plans itself into
queue.md before launching the fine-tune in the background.

## 2026-05-30 - Weight persistence + first HuggingFace upload (GPT-2 cross-pass)

User flagged that no trained weights were ever saved. Root cause: the experiments were
written as measurement functions that return a metrics dict and discard the trained model
(`run_cross_pass_kv` dropped the trained `lm` on return); `results/` held only metrics
JSON, zero weight files. Low cost to recover: the reservoir (W_r, W_in) is fixed-random
from `seed=0` (regenerates byte-identically); only the small readout W_res + LoRA are
learned, and they retrain in minutes at GPT-2 scale.

Built the missing save/load layer:
- `src/reservoir/persist.py` — `save_model_config`/`load_model_config` (plain JSON) and
  `save_array_dict`/`load_array_dict` (npz) are pure and unit-tested in CI
  (`tests/test_persist.py`, 5 tests). `save_reservoir_model`/`load_reservoir_model` are the
  torch-gated glue (config + W_res npz + peft LoRA adapter). `kv_live.py` now stashes
  `_init_config` for reconstruction.
- Wired `--save DIR` into `scripts/run.py crosspass` (persists only the stateful model).
- `scripts/publish_hf.py` — create_repo + upload_folder, with a `--dry-run`.

Found and fixed a reproducibility footgun: the CLI `--steps` default was 300, which
UNDERTRAINS the kv cross-pass recall to ~0.67; FINDINGS claims 1.0. Re-running at 600
steps reproduces **1.00 recall (loss -> 0.01)** vs **0.17 chance** baseline, matching the
paper. Bumped the CLI default to 600. (The published `docs/crosspass.png` was already the
1.0 figure and did not change; only the gitignored `results/crosspass.json` was stale.)

Trained the GPT-2 artifact, verified it **reloads from disk and reproduces 6/6 = 1.00**,
and uploaded it PUBLIC to **https://huggingface.co/EmmaLeonhart/reservoir-agent-gpt2-crosspass**
(user-approved visibility) with a model card stating the real numbers + scope caveats
(GPT-2-only; Hermes still chance). Full suite: 75 passed. Artifact weights live on HF, not
git (`artifacts/` is gitignored).

## 2026-05-30 - Link the published HF model in the report site + README

Surfaced the newly-published GPT-2 cross-pass model in the deliverables (it was on HF but
linked nowhere). Added a "Model on Hugging Face" action card to the docs/index.html
Download section, a "download the trained weights" link in the cross-pass figure caption,
and a "Model / weights" section in README.md pointing at
EmmaLeonhart/reservoir-agent-gpt2-crosspass + the --save / publish_hf.py workflow.
Verified the page HTML stays balanced (html.parser, no unclosed tags) and both HF links
render. Docs-only; CI/pages rebuild on push.

## 2026-05-30 - Installer model registry + reprioritize to N-seed training first

Started the installer, built the model registry, then the user reprioritized: get the
N-seed batch TRAINING running first, installer after. Captured here; queue reordered.

Registry (`src/reservoir/installer/registry.py`, 11 tests in tests/test_registry.py):
lists the project's reservoir-agent models with a live HF query (filtered to the
`reservoir-agent-*` naming convention / `reservoir-agent` tag — the author hosts unrelated
repos) and a bundled offline fallback; pure sort/default/merge logic is CI-tested. Per the
user's design, the registry PRIVILEGES the recommended best but never drops models — the
"bad" seeds in a batch are kept as signal. Verified live discovery finds the published
model. Added the `reservoir-agent` tag (the canonical model-type name) to the HF model
card and re-uploaded. Full suite: 86 passed.

Deferred (registry done): installer console + bootstrap + .exe build. Next: vision/planning
docs (new model type; preserve-all-models-as-signal) + the N-seed batch training pipeline
that saves ALL N models with their scores + reservoir properties.

## 2026-05-30 - N-seed batch training pipeline + reservoir-agent model-type doc

Built the project's core method (reservoir selection via fine-tuning) and documented the
model type. Per the user: reservoir agents are a NEW model type (the first to exist), and
we PRESERVE every model in a batch — the bad seeds are signal for learning what makes a
reservoir good.

- `RESERVOIR_AGENTS.md` — foundational doc: reservoir agent = transformer + brain-surgeried
  fixed reservoir (attended, cross-pass-stateful, RNN-like), not a transformer; build in
  batches of N seeds, fine-tune all, KEEP ALL (bad ones = signal), privilege the best,
  tag everything `reservoir-agent`. Grounded in the data_lake transcripts (selection via
  fine-tuning, building empirical knowledge of good reservoirs).
- `src/reservoir/batch.py` — `rank_population`/`select_best`/`build_batch_manifest` (pure,
  7 tests in tests/test_batch.py: rank by recall then loss, preserve ALL, mark exactly one
  recommended best) + torch-gated `train_batch` (trains each seed via run_cross_pass_kv with
  --save, records per-seed dynamics proxy as signal, writes batch_manifest.json).
- `scripts/run.py batch` subcommand (--model/--n/--steps/...).

Ran a real 4-seed GPT-2 batch (600 steps): all 4 saved with full artifacts + manifest; all
reached recall 1.00, so the discriminating signal is convergence loss (seed 3 plateaued at
0.20 vs ~0) + participation ratio (~0.112-0.114) — exactly the per-seed signal the
preserve-all mandate exists to capture. Best = seed 1. Full suite: 93 passed.

Remaining (queue): publish batch populations to HF (local artifacts are gitignored), run
larger N + larger base models. Installer console/bootstrap/exe deferred until training runs.

## 2026-05-30 - Batch → HF publisher; first batch population published

Honored the preservation mandate: the batch pipeline saved all N locally but they were
gitignored/ephemeral, so they needed to be ON HF. Built the batch publisher and pushed the
first population.

- `build_batch_card(manifest, repo_id)` in batch.py — generates a HF model card documenting
  the WHOLE population (every seed, its score + dynamics signal, recommended best flagged),
  tagged `reservoir-agent`. 3 tests (tag/front-matter, lists every seed, marks best).
- `scripts/publish_hf.py --batch-dir` — validates the batch dir (every seed_*/ is a complete
  artifact), writes the generated card, uploads the whole population as one repo. Kept the
  single-model `--artifact-dir` path; the two are mutually exclusive.

Published `EmmaLeonhart/reservoir-agent-gpt2-batch` (4 seeds + batch_manifest.json + card).
Verified all 4 seed dirs + manifest + card present on HF, and the registry auto-discovers
BOTH reservoir-agent repos. Full suite: 96 passed.

Topology chosen (decided, not asked): one repo per batch holding the full population +
manifest — preserves the seeds together for the pattern analysis that is the whole reason
to keep the bad ones. Remaining: larger N, larger base models.

## 2026-05-30 - Installer console + menu + bootstrap launcher (built in parallel)

With training running (a GPT-2-medium batch in the background), built the deferred
installer pieces — their logic needs no GPU:
- `reservoir.installer.console` — `resolve_load_dir` (pure: single-model repo -> itself;
  batch repo -> recommended seed subdir; 4 tests) + `ReservoirConsole` REPL that carries
  reservoir state ACROSS turns (no reset between turns — the cross-pass statefulness is the
  point) + `download_and_resolve`/`run` (torch+net gated).
- `reservoir.installer.menu` — `choose_repo` (pure: empty -> recommended default; number ->
  that row; 6 tests) + the interactive menu + `python -m reservoir.installer` entry.
- `installer/install.py` — standalone bootstrap (stdlib only; pip-installs the package then
  hands off to the menu). This is what the .exe wraps; a bootstrap (not a frozen binary)
  because a reservoir agent needs torch+CUDA + a multi-GB base model at run time.

21 installer tests total (registry 11 + console 4 + menu 6); full suite green. NOTE: running
the torch-gated suite WHILE a GPU batch trains can flake on CUDA contention (saw one
transient failure that passed on re-run); CI is unaffected (torch tests skip there).
Remaining installer step: PyInstaller .exe build + docs download link.

## 2026-05-30 - Installer .exe build script + CI workflow + docs section

Finished the installer packaging (non-GPU, built while the GPT-2-medium batch trained):
- `installer/build_exe.py` — PyInstaller one-file build of the stdlib-only bootstrap
  (`install.py`); the binary is small because torch + the model are fetched at run time.
- `.github/workflows/build-installer.yml` — Windows runner builds the exe, uploads it as a
  workflow artifact, and attaches it to the GitHub Release on tag pushes.
- docs/index.html — "Run a reservoir agent locally" section: the works-now path
  (`pip install … ; python -m reservoir.installer`) + the Windows exe via the build workflow
  / Releases. No broken/false exe link — points at the workflow + releases pages.

PyInstaller is not installed locally, so the exe build is verified via the CI workflow
(triggers on this push), not a local claim. NOTE: avoid running the torch test suite while a
GPU batch trains — it contends for the GPU (slowed the medium batch + caused a transient
test flake earlier).

## 2026-05-30 - Installer exe build VERIFIED green; GPT-2-medium batch hang fixed

Installer fully complete and verified: the `build-installer` workflow ran GREEN on a
Windows runner @ 2b3b976 — the `reservoir-agent-installer.exe` actually builds (PyInstaller
isn't local, so this CI run was the real check, not a claim). ci + pages also green there.

GPT-2-medium N=10 batch hang: the first attempt stalled at 4/10 seeds (seed_3 saved 20:50,
no progress 35 min). Diagnosis: I had run the full torch test suite against the GPU WHILE
the batch trained, contending for the device and wedging a python process (PID 30432, held
~2 GB GPU). Fix: TaskStop the batch job, kill the wedged process, confirmed GPU back to
0 MiB / 0 python procs, restarted the batch clean (job bvz21t3t6). Operational rule recorded
in queue.md: do NOT run GPU/torch work while a batch trains. Publish the medium population
when the clean run completes. No code defect — a resource-contention hang of my own making.

## 2026-05-30 - GPT-2-medium batch: all chance (negative result) + misdiagnosis correction

Correction to the prior status report: the restarted GPT-2-medium batch was NOT stuck/OOM —
it completed (exit 0), just slow (medium seeds are far heavier than gpt2-small; the two-pass
backprop graph pushes ~8 GB on the 4070). My "wedged near-OOM" read was wrong; it was
grinding.

Real result: **all 10 GPT-2-medium seeds reached chance recall (0.17)** — none learned the
cross-pass task. Loss plateaued high (1.27–2.52) vs ~0 for gpt2-small. This is the same
over-drive signature as the Hermes failure: larger activations saturate the reservoir at
`input_scaling=0.5`. A genuine negative result, not a bug.

Per the preserve-all mandate, published the whole failed population as signal →
`EmmaLeonhart/reservoir-agent-gpt2-medium-batch` (honest card: every seed at chance, ranked
by loss; seed 9 "best" at loss 1.27). The registry still defaults to the working
gpt2-crosspass model, so nothing misleads installer users. Next: probe gpt2-medium with
lower input_scaling (~0.1) + more steps on one seed before committing a full batch.

## 2026-05-30 - GPT-2-medium corrected probe also fails: scale wall starts at 355M

Probed gpt2-medium cross-pass with lower input_scaling=0.1 + 1000 steps (the over-drive
hypothesis). Result: still chance (0.17), loss plateau 12.32->2.16 — lowering the reservoir
drive did NOT fix it. So the cross-pass transfer wall starts at 355M, same failure mode as
Hermes-3B (learns the marginal, ignores the prefix), not a parameter tweak. Documented in
FINDINGS C-section ("transfer wall starts well below 3B"). The real fix is the documented
curriculum / stronger-prefix-coupling routes (substantial; todo.md Hermes thread) — not
blind setting sweeps, so I am not burning GPU guessing.

Decision: pivot the "increasing size" barrel-through to larger-N GPT-2-SMALL batches at a
harder setting (fewer steps) where training actually works and seeds genuinely differ —
that produces the real good/bad reservoir-selection signal the project is built to
accumulate. Larger base models remain gated on the transfer routes.

## 2026-05-30 - GPT-2-small N=12 selection batch: REAL good/bad spread + train_batch mem fix

The productive pivot paid off. GPT-2-small N=12 @ 250 steps produced a genuine
reservoir-selection spread (the signal the project is built to accumulate): recall 1.00
(seeds 1, 7, 10), 0.83 (6), 0.67 (5, 4, 2), 0.50 (3), 0.33 (9, 0), 0.17/chance (8, 11).
Same architecture + training; seeds range from perfect recall to complete failure —
confirming "some reservoirs are good, some bad" and that selection matters. Published the
whole population (best=seed 1) → EmmaLeonhart/reservoir-agent-gpt2-batch-n12.

Correction to the prior status report: the batch was not hard-stuck at 10/12 — it completed
(exit 0), just dragged. Root cause was a REAL defect: train_batch constructed a fresh
injected model per seed without releasing the previous one's CUDA memory, so the caching
allocator accumulated across the loop (~7.9 GB by seed 10 on GPT-2-small — absurd for that
size). Fixed: gc.collect() + torch.cuda.empty_cache() per seed. Pure-logic tests (36) pass;
the memory effect is GPU-runtime, verified by the next batch run, not a unit test (named
plainly). GPU confirmed freed (189 MiB) after the run — clear for the midnight Hermes job.

## 2026-05-30 - Document the trained reservoir-selection spread in the report

Non-GPU work (held off launching batches ~34 min before the midnight Hermes run, and an
unidentified python process was already on the GPU). The N=12 run produced a strong result
the report didn't yet show: a TRAINED downstream selection spread (recall 1.00 for seeds
1/7/10 → chance 0.17 for 8/11), much more decisive than the untrained dynamics proxy.
- Generated `docs/nseed_trained_spread.png` (matplotlib, from the local batch manifest).
- FINDINGS N-seed section: added "Selection matters on the real task, decisively" — same
  architecture/data, only the fixed reservoir differs, and it decides task success; the
  concrete justification for batch-and-keep-all. Links the published population.
- docs/index.html: matching paragraph + figure.
HTML validated balanced. CI/pages verify on push. GPU left untouched (clear for midnight).

## 2026-05-31 00:00 - Midnight kickoff: Hermes 3B cross-pass transfer (many-more-steps)

Scheduled one-shot fired. SYNC clean, GPU free (0/8188). Planned into queue.md + task #19,
then launched the headline transfer attempt in the BACKGROUND:
`crosspass --model NousResearch/Hermes-3-Llama-3.2-3B --mode kv --4bit --steps 2000
--save artifacts/hermes-crosspass` (log: results/hermes_crosspass_run.log, job bnbpqv4f1).

Chose crosspass (NOT finetune as the cron example said): crosspass --mode kv IS the
cross-pass RECALL experiment that produces the ~2.8 plateau the thread references; finetune
is a generic LM fine-tune that doesn't measure recall. Clean test of the documented route:
same baseline settings (4-bit, default input_scaling), only steps 300 -> 2000. On
completion I will MEASURE: recall>chance -> publish the saved model + FINDINGS update;
still chance/plateau -> record the precise blocker (more steps insufficient -> curriculum/
coupling). No faking either way.

## 2026-05-31 - Hermes 3B many-more-steps run: still chance (route ruled out)

The midnight transfer attempt completed (job bnbpqv4f1, exit 0). Result, measured:
- Hermes-3-Llama-3.2-3B, kv-prefix, 4-bit, **2000 steps** (≈6.7× the prior 300):
  stateful recall **0.17 (chance)**, loss 10.07 -> **2.49**; baseline 0.17, loss -> 2.75.
- So MANY MORE STEPS did NOT break the plateau (2.49 ≈ the prior ~2.8; no better than 300).

Conclusion (no faking): the Hermes transfer wall is **neither quantization nor
under-training** — it's structural (prefix bootstrapping diluted through 28 layers; GPT-2
-medium fails the same way). The remaining routes are a curriculum / stronger multi-layer
coupling / unfreezing more — substantial, dedicated work, not a hyperparameter. Updated
FINDINGS (Hermes paragraph: 4th attempt, more-steps ruled out) and todo.md (route closed).

The chance-level saved artifact (artifacts/hermes-crosspass: W_res + LoRA + config) is kept
LOCAL, NOT published — a single non-working 3B model would mislead installer users, and it
is not a selection batch. The negative result is the deliverable, recorded in FINDINGS.

## 2026-05-31 - GPT-2-small N=20 selection batch + memory fix verified at scale

Grew the reservoir-selection dataset: N=20 @250 steps → published
EmmaLeonhart/reservoir-agent-gpt2-batch-n20. Full spread: 4 seeds at recall 1.00
(4, 9, 0, 3), through 0.83/0.67/0.50/0.33, down to 5 at chance 0.17 (15, 12, 16, 6, 8).
Best = seed 4. Combined with N=12, that's a 32-seed selection dataset.

The train_batch per-seed memory fix is VERIFIED at N=20: the run completed clean and the
GPU released to 0 MiB (no accumulation/drag like the unfixed N=12).

SIGNAL worth recording: across all 20 seeds the participation-ratio proxy pr_frac is
essentially constant (~0.111-0.114) and does NOT track recall; final training loss also
doesn't cleanly predict recall (e.g. seed 19 loss 0.037 -> recall 0.67; seed 14 loss 2.768
-> recall 0.83). So neither the cheap dynamics proxy nor loss predicts which reservoir wins
— consistent with the earlier proxy-fails result, now on real downstream recall. This
motivates the next direction: enrich per-seed reservoir metrics to find an actual predictor
(the point of keeping the whole population).

## 2026-05-31 - Per-seed metrics + a confound correction (reservoir "selection" is noise-dominated at 250 steps)

Built reservoir.seed_metrics (untrained per-seed structural metrics: realized rho,
mean/std |eigenvalue|, Henrici non-normality, participation ratio, delay-memory capacity)
+ correlate_seed_metrics, 6 CI tests. Correlated against the N=20 recall labels: NO metric
predicts recall (all |Spearman| < 0.36, p > 0.14).

But checking the data turned up a confound I had to correct: the N=12 and N=20 batches share
seed indices (a natural replication), and the SAME seed (identical fixed reservoir, same
250-step setting) lands at very different recall across the two runs — seed 0: 0.33 vs 1.00,
seed 1: 1.00 vs 0.33, mean |Δrecall| ≈ 0.47 over 12 shared seeds, nearly the full spread.
So the per-seed recall spread is DOMINATED BY TRAINING NOISE (CUDA non-determinism +
under-trained 250-step regime + the trainable readout/LoRA init not seeded by the reservoir
seed), NOT clean reservoir quality. A single run per seed cannot separate the two.

Corrected the overclaim in FINDINGS ("Selection matters... decisively" -> a measured caveat)
and docs/index.html (figure recaptioned "per-seed recall, one run", with the replication
caveat). What still holds: cheap metrics can't pre-filter, so keep the whole population +
train; reservoirs at fixed rho have near-identical bulk dynamics (H2). What does NOT yet
hold: that some fixed reservoirs are durably better on this task. The proper test (queued):
seed the trainable init + deterministic CUDA + average several runs per seed (or train far
longer). The published populations stay up (real data); only the interpretation is corrected.

## 2026-06-01 - Imported the "Attention Reservoir Architecture" Grok chat + opened Phase G

User added a saved Grok conversation to the data lake and asked to apply its insights, with
a stated want to move the base model to DeepSeek-V4-Flash. Extracted the 1.1 MB HTML export
(BeautifulSoup; stripped Grok UI chrome, cookie banners, "N sources" lines) and reconstructed
the 15 Emma/Grok exchanges into data_lake/transcripts/attention-reservoir-architecture-grok.md.
Raw HTML + _files JS dump moved under data_lake/ where the existing .gitignore rules keep it
local (convention: distilled transcripts committed, raw exports not).

Feasibility check on the headline ask: DeepSeek-V4-Flash IS real (284B-total/13B-active MoE,
1M context, MIT, released 2026-04-24) but is not loadable, let alone fine-tunable, on this
RTX 4070 (8.6 GB) — and reservoir injection requires fine-tuning, so a hosted API can't
substitute. The cache-efficiency architecture the whole chat hinges on (MLA / compressed KV)
does exist in a small enough form: DeepSeek-V2-Lite (16B/2.4B-active, MLA, 27 layers). Asked
the user how to proceed; they chose: do the base-agnostic insight work first, then attempt a
V2-Lite feasibility spike. Opened queue Phase G with that plan (7 items: reservoir-protected
KV eviction → blank-cycle context-growth demo → interruptibility experiment → reservoir-state
linear probe → Safety-by-Design section → DeepSeek decision into todo/REVIEW → V2-Lite spike),
mirrored to the task tool, and started the three session-local crons (work-loop :03, auto-flush
:15, status-report :42) for a fresh session.

## 2026-06-01 - Reservoir-protected KV eviction policy (Phase G, from the Grok chat)

Built src/reservoir/kv_evict.py + tests/test_kv_evict.py (11 tests, TDD, all green; full
suite 123 passed). This is the first concrete application of the imported Grok conversation:
the chat flagged that a Reservoir Agent's blank ticks keep appending to the KV cache, burning
context faster than a turn-based model, and pointed at StreamingLLM (attention-sink + recent
window) eviction with the reservoir's K/V *pinned*. ReservoirEvictionPolicy implements exactly
that as a pure, torch-free policy over per-position tags {sink, reservoir, normal}: always
retain sinks + all reservoir entries + the recent window; evict oldest normal tokens first up
to a budget; protected entries are a hard floor (over_budget() reports when they alone exceed
the cap); with no reservoir tags it degrades to vanilla StreamingLLM. Keeping it pure means it
runs in CI on CPU with no GPU gate. Next (queue item 1): drive blank ticks through the live KV
path to show vanilla cache grows linearly while this policy stays bounded and the reservoir
signal survives.

## 2026-06-01 - Blank-cycle context-growth demonstration (Phase G)

Built src/reservoir/blank_cycle.py (simulate_blank_cycle + plot_blank_cycle) + a `blankcycle`
subcommand in scripts/run.py + tests/test_blank_cycle.py (6 tests; full suite 129 passed).
Makes the Grok chat's "context explodes on a reservoir agent" concern measurable with no GPU:
streaming 512 blank ticks, the no-eviction cache grows linearly to 524 positions while the
reservoir-protected policy (kv_evict) stays bounded at the budget (128) from ~tick 116 on, and
all 8 reservoir entries are retained on EVERY tick — the time-axis is exactly what the policy
refuses to drop. Wrote results/blank_cycle.json (gitignored), docs/blank_cycle_kv.png (the
line chart), a FINDINGS "## KV: blank-cycle context growth" section, and a docs/index.html
figure block. The bound is the point, not the specific numbers (they scale with budget/window).
The expensive native-KV-efficiency half (DeepSeek MLA / V4 CSA+HCA) stays a documented todo —
not runnable on this hardware.

## 2026-06-01 - Interruptibility experiment: faster + more durable STOP response (Phase G, safety)

Built src/reservoir/interrupt.py + an `interrupt` subcommand + tests/test_interrupt.py (6
tests; full suite 135 passed). Quantifies the chat's controllability claim on CPU, two ways.
(1) Polling latency: a turn-based agent reading every `period` passes has mean latency
(period-1)/2 — at period 8, mean 3.57 passes (max 7); a per-tick agent is 0. (2) Signal
persistence: a one-shot urgent STOP burst, driven through an EchoStateReservoir (leak 0.2 =
long fading memory), stays above a matched-filter detection threshold for 3 passes AFTER
arrival, while a stateless monitor (current input only) catches it on the arrival pass and 0
after. Consequence: a turn-based + stateless agent whose poll period (8) outruns the
persistence window misses a non-repeated off-boundary burst entirely; the per-tick reservoir
agent catches it on arrival and has a window besides. Wrote results/interrupt.json, the
docs/interrupt.png trace figure, a FINDINGS "## Safety: interruptibility" section, and a
docs/index.html figure block. Framed as a measured illustration, not a guarantee (window
length is set by the leak/threshold settings; a real harness adds its own latencies).

## 2026-06-01 - Reservoir-state linear probe: internal clock, no SAE, drift-tolerant (Phase G, safety)

Built src/reservoir/probe.py + a `probe` subcommand + tests/test_probe.py (5 tests; full
suite 140 passed). Tests the chat's falsifiable interpretability claims on CPU. (1) A plain
ridge probe reads a temporal process property — elapsed passes since the last trigger, an
internal clock — from reservoir state at R²=0.995, vs R²=0.16 from the instantaneous input
(elapsed time isn't in the current input). A LINEAR probe suffices — no SAE needed, the chat's
"you don't need a sparse autoencoder for the reservoir state" borne out. (2) Resilience,
measured not assumed: modelling a fine-tuning-like drift α on the driving activations and
re-applying the pre-drift probe, R² degrades gracefully 0.99→0.94 (α=0.4) →0.82 (α=0.8),
staying far above the 0.16 stateless baseline throughout. Framed honestly: usable across
moderate drift, NOT invariant — the reservoir weights are fixed but their driving inputs still
move, so a very large fine-tune would still erode it. Reading an elapsed clock is the
decodability demo; reading genuine misalignment signatures is a much harder unproven extension
(flagged future work). Wrote results/probe.json, docs/probe.png (two-panel), FINDINGS
"## Safety: a reservoir-state probe…", docs/index.html figure block.

## 2026-06-01 - Safety-by-Design section (Phase G)

Added a "## Safety by design" framing section to FINDINGS.md and a matching block to
docs/index.html, stating the project's rule (never introduce a capability without meaningfully
taking safety into account) and the three safety properties that come from the SAME fixed
reservoir as the capability — lower-latency durable override (interruptibility), a cheap stable
linear monitoring surface (the probe), and bounded idle context (blank-cycle eviction) — each
tied to a measured number from this session, not asserted. Names what is NOT shown: the probe
decodes a benign elapsed clock, not misalignment signatures; resilience is graceful not
invariant; everything is small-scale on a fixed reservoir, not the real base. Also removed two
self-congratulatory "stated honestly"/"honest version" phrasings from the probe section per
the writing-style skill (name the limit flatly instead).
