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

## 2026-06-01 - DeepSeek base-model decision recorded (Phase G)

Recorded the base-model direction from the Grok chat into the project's planning + literature.
todo.md gets a new "Base model — moving off Hermes to a KV-efficient base" subsection:
DeepSeek-V4-Flash as the aspirational target (real, 284B/13B-active MoE, 1M ctx, MIT,
2026-04-24 — but not loadable/fine-tunable on the 8.6 GB 4070, so cloud/big-GPU only) and
DeepSeek-V2-Lite as the realistic local MLA base to attempt injection on next (16B/2.4B-active,
MLA, 27 layers), with a decomposed spike and the reservoir×compression research question.
literature/REVIEW.md §1 gets a KV-cache paragraph and sources.md a new §6 with the verifiable
citations (StreamingLLM arXiv:2309.17453, H2O arXiv:2306.14048, MLA/DeepSeek-V2 arXiv:2405.04434)
plus the V4-Flash release entry flagged as reported-not-independently-verified for its CSA/HCA
detail. Closes the writing half of the user's DeepSeek ask; the GPU spike is the one remaining
Phase G item.

## 2026-06-01 - DeepSeek-V2-Lite feasibility analysis (Phase G; config-only, no weight download)

Resolved most of the V2-Lite spike's uncertainty cheaply, without the 31 GB weight download.
Environment: torch 2.10+cu128, transformers 5.4.0 (which supports `deepseek_v2` NATIVELY — no
trust_remote_code), bitsandbytes 0.49.2, peft, accelerate all present; RTX 4070 8.6 GB, 763 GB
free. Pulled the config (a few KB): 27 layers, hidden 2048, 16 heads, MLA kv_lora_rank=512
(qk_rope 64 / qk_nope 128 / v_head 128; queries uncompressed in Lite), MoE 64 routed experts
(6 active) + 2 shared, first_k_dense_replace=1. Mid-layer injection point = layer 13/27.

Go/no-go: arch support GO; the kv_live.py prefix-injection mechanism is architecture-agnostic
(inputs_embeds + causal path), so the port is bounded — a _arch.py deepseek_v2 branch, LoRA
targets set to the MLA projection names (q_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj), layer=13.
VRAM is the constraint: 16B at 4-bit ≈ 8 GB of weights vs 8.6 GB, so a pure-GPU load is at the
edge and device_map="auto" + CPU offload is required. The one thing the analysis can't settle
is whether QLoRA *training* fits in 8.6 GB with offloaded experts — only a real attempt does.
Did NOT run the ~9 GB 4-bit download + load blind in this session: heavy download + uncertain
training fit, left as a resource-gated local step (tightened queue item + recorded in todo.md).
Not a faked completion — the load was not run, and that is stated.

## 2026-06-01 - Importance-based (H2O) eviction added to the reservoir-protected policy (Phase G)

Work-loop tick: the only remaining Phase G item (the V2-Lite injection attempt) is
resource-gated, so promoted the next bounded, CPU-verifiable item — the H2O heavy-hitter
eviction the chat raised and sources.md §6 had flagged as planned. Extended
ReservoirEvictionPolicy.retained_indices/evicted_indices with an optional per-position
`scores` argument: when given, the budget headroom is filled with the highest-scoring
evictable normal tokens (heavy hitters) instead of the newest, ties broken toward recency,
while sink + reservoir + recent window stay pinned exactly as before. scores=None keeps the
StreamingLLM recency behaviour (regression test added). 5 new tests (suite 145 passed),
docstring + a one-line FINDINGS note. Position-based and importance-based eviction now sit
under one interface with the reservoir pinned either way.

## 2026-06-01 - README brought current with Phase G (work-loop housekeeping)

Work-loop tick: the only remaining Phase G queue item is the resource-gated V2-Lite attempt,
so did the standing README-currency obligation instead. README.md was stale (predated the
session) — added a "Safety & runtime (Phase G)" findings group (interruptibility latency 0 vs
3.57 + 3-pass persistence; linear probe R²≈0.99 vs 0.16, no SAE, graceful drift; bounded idle
context via reservoir-pinned eviction), an "## Experiments" section listing the runnable
run.py subcommands (verified all present via `run.py -h`), and the DeepSeek-V2-Lite base
direction + a pointer to the imported Grok transcript. No code change; suite unchanged at 145.

## 2026-06-01 - DeepSeek-V2-Lite dropped; learned cache-compression is frontier-scale-only

User decision: drop the V2-Lite base switch. Reason: V2-Lite has MLA (fixed low-rank KV
compression, ~8.9× smaller cache than MHA — verified from its config: 576 vs ~5120 dims/token)
but NOT the *learned, fine-tunable* compression that is the actual point — being able to
fine-tune the cache manager to lean on the reservoir for long-idle signal. Verified the
landscape: learned/trainable compression is DeepSeek Sparse Attention (DSA, first in V3.2,
671B) and the V4 CSA+HCA hybrid (V4-Flash, 284B); there is no runnable-size open model with it
— it's frontier-scale-only. So "the next biggest model that has it" is V4-Flash/V3.2, neither
locally runnable. The base-model question is now a fork for the user: (a) cloud/rented-GPU
fine-tune of V4-Flash/V3.2 (the only way to test learned-compression × reservoir), or (b) drop
the learned-compression angle locally and keep the reservoir-pinned eviction (kv_evict.py) as
the local cache story on GPT-2/Hermes. Removed the V2-Lite queue item + task; recorded in
todo.md (config-only analysis retained there in case a fixed-MLA base is ever wanted). Phase G
(the buildable Grok insights) is complete; the base switch is parked on the user's call.

## 2026-06-01 - Base-model question resolved: drop the learned-compression angle locally

User chose: drop the angle, stay local. Local hardware can't have learned/fine-tunable cache
compression (frontier-scale-only), so the local cache story is settled as the reservoir-pinned
eviction (kv_evict.py) — it bounds blank-tick context burn without needing a learned compressor.
Local work continues on GPT-2/Hermes. The learned-compression × reservoir hypothesis (does
fine-tuning a DSA/CSA-HCA compressor teach it to defer to the reservoir for idle signal?) is now
an explicit cloud/future experiment, deferred until there's a compute budget. Updated todo.md +
queue.md to mark Phase G (the buildable Grok insights) complete and the base switch resolved.

## 2026-06-02 - Phase I opened: fix the dead train_seed confound (controlled selection experiment)

User resumed local GPU work, choosing the N-seed controlled experiment (over the Hermes recall
transfer). Goal: settle whether reservoir "selection" is real or training-noise, after the
2026-05-31 finding that the per-seed recall spread at 250 steps was noise-dominated (same
reservoir seed → 0.33 vs 1.00 across runs). Root cause identified: kv_live.py declared a
`train_seed` parameter but never used it — the trainable W_res + LoRA init was unseeded, so two
runs of the SAME fixed reservoir started from different inits.

Fixed it: kv_live now seeds the torch RNG with train_seed before constructing W_res + the LoRA
adapter, making the trainable init a controlled variable (train_seed=None keeps the old unseeded
behaviour). tests/test_train_seed.py (4 torch-gated tests, local-green; skip in CI): same
train_seed → identical W_res weight/bias and identical LoRA init; different train_seed → differs;
train_seed recorded on the instance. Full suite 149 passed locally, no regression. Restarted the
three crons for the resumed work. Next: a deterministic-CUDA helper + thread train_seed through
run_cross_pass_kv, then the controlled runner + variance analysis, then the local run.

## 2026-06-02 - Deterministic-CUDA helper + train_seed plumbing (Phase I); run is now bit-reproducible

Closed the other half of the confound. Added src/reservoir/determinism.py `set_deterministic(seed)`
— seeds Python/NumPy/torch RNGs, sets CUBLAS_WORKSPACE_CONFIG, cudnn.deterministic=True /
benchmark=False, and torch.use_deterministic_algorithms(True, warn_only=True). Threaded a
`train_seed` (and `deterministic`) param through run_cross_pass_kv: `seed` fixes the reservoir
(W_r/W_in), `train_seed` fixes the trainable init (via the kv_live fix) AND the data order,
independently — so several runs of one reservoir vary only by train_seed. train_seed=None
reproduces prior behaviour. tests/test_determinism.py (3 torch-gated, local-green, CI-skip): the
key one trains tiny-gpt2 twice with the same (seed, train_seed) and asserts loss_start, loss_end
and recall_accuracy are BIT-identical; a different train_seed changes the trajectory; and
set_deterministic sets the cudnn flags. Full suite 152 passed. With init + data + kernels all
controlled, any recall spread that survives across DIFFERENT reservoir seeds is reservoir
quality, not run-to-run noise — which is exactly what the next item (controlled_selection +
variance analysis) measures.

## 2026-06-02 - Controlled-selection runner + ANOVA analysis (Phase I)

Built src/reservoir/controlled.py: controlled_selection (torch-gated; trains R runs per reservoir
seed, same train_seeds across seeds) + selection_signal (pure; one-way ANOVA over recall grouped
by reservoir seed → per-seed means, between- vs within-seed mean squares, F, p-value via
scipy.stats.f, and a verdict selection_is_real = p<0.05) + plot_controlled. Added a `controlled`
run.py subcommand. tests/test_controlled.py: 6 pure analysis tests (CI-run) — F matches a
hand-computed 24.0 on a known example, a strong between-seed signal reads real (p<0.05), pure
within-seed noise reads F=0 / not real, zero within-variance → F=inf → real, per-seed means
reported, <2 seeds raises — plus 1 torch-gated runner smoke (2 seeds × 2 runs returns the right
shape). Full suite 159 passed. This is the analysis that will judge the actual run; next item is
the local GPU run (N seeds × R runs at higher steps) + the write-up.

## 2026-06-02 - Controlled N-seed run: selection is noise, not signal, at 250 steps (Phase I complete)

Ran the controlled experiment (scripts/run.py controlled, 6 reservoir seeds × 4 runs × 250
steps, GPT-2, on the 4070). With the confounds removed — train_seed now seeds the trainable
init, and set_deterministic forces the math SDP kernel so runs are bit-identical for a fixed
(seed, train_seed) on CUDA (verified) — the per-seed mean recall ranges 0.33–0.75, but the
within-seed spread is as wide as the between-seed spread (seed 0 spans 0.33→1.00 across inits).
One-way ANOVA over recall by reservoir seed: F=1.30 (df 5,18), p=0.31 → selection_is_real=False.

So the verdict, now controlled rather than suspected: at 250 steps, reservoir "selection" is not
a real signal — which trainable init you get matters more than which fixed reservoir you drew.
This converts the 2026-05-31 suspected-artifact into a controlled negative result. It does not
rule out selection mattering at a far larger training budget (where init noise should shrink) —
that longer run is the named follow-up (queue + todo). Also strengthened determinism.py to force
the deterministic SDP kernel after the mem-efficient attention backward was found nondeterministic
on CUDA. Updated FINDINGS (the per-seed-spread section now reports the controlled result) and
docs/index.html (+docs/controlled.png). Full suite 159 passed. Phase I complete.

## 2026-06-04 — `run_agent.bat`: local recall-demo + stateful REPL launcher

Added a repo-local "does it actually work?" launcher (NOT the distributable — that's the
`reservoir-agent-installer.exe`). `run_agent.bat` resolves the local Python, then runs
`python -m reservoir.installer`, which auto-picks the most-recent recommended HF model
(`registry.default_model`), downloads it, prints a guided cross-pass recall demo, and drops
into the stateful REPL. New flags: `--menu` (chooser), `--demo-only`, `--no-demo`.

Pieces: factored `eval_recall`/`recall_accuracy` out of `run_cross_pass_kv` (crosspass.py,
DRY); `recall_demo_session` printer (console.py); fixed a real REPL bug — `step()` used
`model.generate`, which fires the read hook but never applies the trained reservoir→prefix
*write* path (only `forward_logits` does), so the old REPL accumulated state but never fed it
back. New `generate_stateful` greedy-decodes over `forward_logits`, making the REPL genuinely
stateful. `menu.main` defaults to auto-pick + demo + REPL; `console.run` grew `demo`/`repl`
switches and reads `n_keys` from the saved model meta.

Verified on the **downloaded** weights (`--no-hf --demo-only`): stateful recall **100%**,
baseline (state wiped between passes) **0%**, chance 17% — the headline cross-pass result
reproduces from the published artifact, not just from a fresh training run. REPL path
exercised end-to-end (pipes a turn, generates, exits clean). Full suite 169 passed. New
tests: eval_recall/recall_accuracy, recall_demo_session, generate_stateful (torch-gated),
and six menu flag-routing tests. Spec + plan under `docs/superpowers/`.

## 2026-06-05 — Real-time agent app + the stateful-task loss battery (first training)

Reframed the project around the user's actual goal: a **real-time, always-alive, independent
agent**, not a chatbot or the narrow colour-recall trick. Behaviour must be a *training
discovery*, never a system prompt.

Built the harness (UI-out): `src/reservoir/alive.py` (`AliveEngine` — continuous prompted/idle
tick loop, live `readout_scale` gain, no_grad fix so the live loop doesn't accumulate the
through-pass autograd graph and OOM); `app/server` (websockets, owns the engine on a bg thread);
`app/electron` (two-pane live UI: agent stream | you, telemetry strip with tick/entropy/state·cos,
reservoir-gain slider). `run_agent.bat` now launches the app; recall demo preserved as
`run_recall_demo.bat`. Runs Qwen2.5-1.5B + reservoir on the 4070, GPU bounded ~6.4GB, coherent
live text. Honest: this is the UNTRAINED substrate — proves the harness, not trained behaviour.

Designed + built the loss battery (the real objective): `episode.py` (Episode/Step + SILENCE,
`episode_loss` backprop-through-passes, `episode_eval`), `battery.py` (8 generators: recall,
accumulate, sequence, deferred, timed, interrupt, selfinit, silence), `train_battery.py`. Spec:
`docs/superpowers/specs/2026-06-05-stateful-loss-battery-design.md`.

First GPT-2 training runs (findings, not yet a win):
- 400 steps, flat 8-way mix, 24-word vocab: loss 11.6->1.6; silence=1.0, selfinit/timed~0.7
  (the silent portions), all content tasks (recall/accumulate/sequence/deferred)=0.0.
- 800 steps, content-upweighted, 12-word vocab: loss 10.8->3.6; recall flickered to 0.38 then
  collapsed; content tasks ~0; silence stuck at 1.0.
- Diagnostic, content-only (no silence), 500 steps: recall reaches only 0.38; others ~0.
Conclusion: the framework is correct and tasks *begin* to learn, but (a) each task needs hundreds
of dedicated steps (recall alone needed 400 -> composite needs thousands), and (b) silence-as-eos
competes with content in one distribution — the gate ("when to speak") should be a SEPARATE HEAD
from emission. Next: a separate gate head + a long (thousands-of-steps) run, then load the trained
model into the live app. New tests: test_alive, test_battery. Full suite green.

## 2026-06-05 (cont.) — Qwen battery runs: the content-vs-temporal split

Moved training to Qwen2.5-1.5B (GPT-2's 124M lacks capacity — confirmed). Through-passes
backprop fits easily (peak 3.3GB/8GB). Two runs:
- lr=1e-3 flat, 1500 steps: peaked at step 600 (recall 0.50, silence 1.0, timed 0.71,
  selfinit 0.67) then DEGRADED to 1500 (overshoot) — and saved the final/worst weights.
- Fixed the trainer (cosine LR to 0 + keep BEST-eval checkpoint, not last), re-ran lr=5e-4:
  best at step 1200 (mean 0.34) — silence 1.0, timed 0.71, selfinit 0.67, interrupt 0.25,
  deferred 0.12, but recall/accumulate/sequence = 0.

Consistent finding across all GPT-2 + Qwen runs: the reservoir+small-readout reliably learns
the TIMING/GATING family (when to be silent, count passes -> silence~1.0, timed~0.7,
selfinit~0.67) but NOT the CONTENT-MEMORY family (recall/accumulate/sequence ~0, deferred
~0.12), even on Qwen. Lowering lr didn't rescue content (recall actually peaked higher at
lr=1e-3). This is a capacity/architecture result, not "needs more steps". Next moves:
(1) separate gate head so silence-as-eos stops competing with content; (2) more content
capacity (bigger reservoir / more prefix tokens / larger readout); (3) bigger eval set to cut
the per-eval variance. The best Qwen checkpoint (good timing, weak content) is saved at
artifacts/qwen-battery.

## 2026-06-05 (cont.) — gate head + N-seed population: content unlocks (partially)

Implemented all four next-step levers: (1) separate gate head (speak/silent BCE from
reservoir state, so silence stops training as predict-eos and cannibalising content);
(2) content capacity (n_reservoir 512->1024, n_prefix 8->16); (3) bigger eval (eval_n 16);
(4) multiple reservoirs — train_battery_population trains N seeds, keeps ALL + a
batch_manifest recommending the best (the N-seed design, RESERVOIR_AGENTS.md). Gate head
saved/loaded by persist (backward-compatible). All existing tests still green.

N-seed Qwen run (3 seeds x 1200 steps, gate head, 1024-node reservoir, content-upweighted):
best seed 0 mean 0.41 (was 0.34 single-seed/no-gate). Content tasks moved OFF zero for the
first time: accumulate 0.38 (seed 0), recall 0.31 (seed 1), deferred 0.19 (seed 2). Gating
solid: silence 1.0, interrupt 0.56 (up from ~0.06), selfinit 0.62, timed 0.56. sequence
still 0 (hardest). Notable: seeds SPECIALISE — seed 0 best at accumulate, seed 1 best at
recall; no seed dominates -> the population design earns its keep. Still not a strong agent:
content weak/noisy. Saved population at artifacts/qwen-battery-pop (best=seed_0). Next: wire
the trained best seed into the live app (app/server -> load_reservoir_model) so the agent
shown is finally trained; push content further (more steps / bigger reservoir / per-task).

## 2026-06-05 (cont.) — large run results, the expansion finding, and the massive-run setup

8-hour large run (Qwen+1024 reservoir, 1200-word vocab) finished: 16 epochs, best at epoch 3
(mean 0.332, ~2h), then overtrained/destabilised (epoch 11 collapsed to 0.031). recall=0.00
EVERY epoch — content dead at 1200 words / 1024 nodes. Only temporal/gating learned (timed
~0.62, silence ~1.0, selfinit ~0.64), peaking early. All epochs streamed to
hf.co/EmmaLeonhart/reservoir-agent-qwen-battery-large.

KEY FINDING (user-driven): the reservoir is undersized AND collapsing. Qwen feeds it a
1536-dim layer (28 layers x 1536); we ran 512-1024 nodes = 0.3-0.7x the input — a reservoir
is supposed to EXPAND, not compress. Measured effective dimensionality (corrected; an earlier
~72 used a wrong toy input dim): plateaus ~150-186 regardless of nominal size (16x more nodes
barely moves it), 74% of cells saturated under input_scaling=0.5 (detune -> 13%). Low-dim
temporal state fits the ~180 usable dims (works); high-dim symbolic content doesn't (fails).
Positive framing: real temporal + low-level symbolic signal from a badly misconfigured
reservoir => fixable engineering, not a dead end.

Set up for the next run (the user's massive-reservoir vision): fixed _build_reservoir_weights
to use power iteration for the spectral radius (full eigvals is O(K^3) and stalled at 12k
nodes); small K keeps exact. Probed: n_res=8192 (5.3x input, 8x tonight's 1024) trains at
5.9GB/8GB; 12288 peaks 7.5GB (edge). train_large gained epoch-count mode (RESERVOIR_EPOCHS) +
RESERVOIR_INSCALE (detune). Crons scheduled (session-only): 09:55 initial paper writeup of the
8h run; 12:00 launch the massive run (8192 nodes, input_scaling 0.1, E+1=17 epochs, upload
each epoch to reservoir-agent-qwen-battery-massive-v3) and set up an hourly paper-update cron.
Full 25%/60k-node vision needs sparse W_r + a down-projection + bigger hardware (future work).

## 2026-06-05 (cont.) — massive (8192-node) run: stopped at 5 epochs, peaks at epoch 1

Ran the 8192-node reservoir (5.3x the 1536-dim input, input_scaling 0.1 detuned) on the
battery, epoch-count mode, uploading each epoch to hf.co/EmmaLeonhart/reservoir-agent-qwen-
battery-massive-v3. Trajectory: epoch 0 mean 0.239, epoch 1 0.349 (BEST, past the 1024-node
run's 0.332), epoch 2 0.302, epoch 3 0.286, epoch 4 0.008 (collapse). Stopped by decision
after epoch 4 — the result peaks within ~1 epoch and only degrades after, so one epoch is
enough. Content-memory tasks never recovered (recall 0 throughout; accumulate flickered to
<=0.12 then vanished); the 5.3x expansion lifts temporal/gating but not symbolic content. A
reservoir genuinely larger than its input needs sparse W_r + bigger hardware (future work).
FINDINGS + docs updated to the final result; hourly paper-tracking cron retired.

## 2026-06-06 — reproducibility audit: KV cross-pass recall reproduces from scratch

User asked to audit the reproducibility of the load-bearing result (the 100% cross-pass
recall via content-addressable KV-prefix injection — the one a reviewer is most likely to
rerun, since it goes through kv_live.py's bespoke attention path rather than stock HF). Ran
the documented recipe from a clean checkout: `python scripts/run.py crosspass --mode kv
--steps 600 --model gpt2`. Result matches FINDINGS exactly: **stateful recall = 1.00** (loss
12.51 -> 0.0001), **stateless baseline (reservoir wiped between passes) = 0.17** (chance =
1/6, loss 16.31 -> 2.61), on CUDA. So the central claim reproduces end-to-end through the
bespoke path. The reproducibility *limitation* is unchanged and still named in FINDINGS: the
winning variant runs through kv_live.py, not a stock `transformers` model (the HF GPT-2
KV-append integration is a documented blocker) — the audit confirms the bespoke path itself
is reliable, not that it's been ported to stock HF. Audit was non-mutating (backed up and
restored docs/crosspass.png; results/ is gitignored). No code change.

## 2026-06-06 — session handoff (abstract, repro audit, publish, CI fix)

Brief handoff of what this session did and what's still in flight.

**Done & pushed:**
- **`!runAgent.bat`** now opens the Electron app (forwards to `run_agent.bat`); removed the
  old duplicate `!run_agent.bat` (it only ran the installer console, which still lives in
  `run_recall_demo.bat`). Verified the Electron server's imports load (torch+CUDA, websockets,
  reservoir); did NOT validate the GUI window opening headlessly.
- **Reproducibility audit** of the load-bearing result: reran `run.py crosspass --mode kv
  --steps 600` from clean → stateful recall **1.00**, stateless baseline **0.17** (chance).
  Reproduces end-to-end through the bespoke `kv_live.py` path. The stock-HF integration
  blocker is unchanged and still named in FINDINGS limitations.
- **Abstract** added to `FINDINGS.md` as the PDF lead (injection-design result → reproduced
  1.00 vs 0.17 → dynamics regime → 355M/3B scaling wall → content/temporal split). Site lede
  (`docs/index.html`) corrected to the honest scaling-boundary framing.
- **Published**: fast-forwarded `main` (was 25 commits behind) → Pages rebuilt the site +
  `report.pdf`, deploy verified green. NB: publishing requires a push to `main` (the Pages
  workflow only triggers on main/master); the permission classifier gates direct main pushes,
  so this needed explicit user authorization.
- **CI deprecation fix**: `pages.yml` now sets `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` to
  silence the Node-20 deprecation warning without chasing action version tags.

**In flight (for you to pick up):**
- **`controlled --steps 1500`** GPU job running in background (task #1; queued in `queue.md`,
  writes to `results/controlled_1500.json` + `docs/controlled_1500.png`, NOT clobbering the
  published 250-step negative). Question: does reservoir-seed selection become a real signal
  at higher budget (where init noise shrinks)? When it finishes: read the F/p + per-seed
  means, add a FINDINGS paragraph + the new figure, append a devlog entry, delete the queue
  item, and republish `main` the same way. The result is real either way (signal or a
  strengthened negative).

**Crons:** three session-local crons are running (work-loop :03, auto-flush :15,
status-report :42); they die when this session ends.

## 2026-06-06 — longer-budget controlled run (1500 steps): negative strengthened

Ran the user-approved follow-up `scripts/run.py controlled --steps 1500 --n-seeds 6 --runs 4`
(6× the 250-step budget) to test whether reservoir-seed selection becomes a real signal once
run-to-run init noise shrinks. It does not. Per-seed mean recall 0.21–0.83 (wider than the
250-step 0.33–0.75), but within-seed spread is still as wide (seed 4: 1.00,1.00,0.17,0.17):
**F=1.43 (df 5,18), p=0.26 → selection_is_real=False** (best seed 5 @0.83, worst seed 2 @0.21).
So more budget strengthens, not overturns, the controlled negative — the trainable init still
dominates over the fixed reservoir. Verdict now holds across 250→1500 steps: select over runs,
not reservoir seeds. Wrote results/controlled_1500.json + docs/controlled_1500.png (separate
files — the published 250-step result is untouched). FINDINGS N-seed section updated.

## 2026-06-06 — paper: academic-register pass (addresses clawRxiv review con #5)

Work-loop fallback (queue drained, user said "work on the paper"). The clawRxiv reviews
(Gemini 3 Flash, "Weak Reject", posts 2680/2685) flag con #5: informal, project-log register
("cleanvibe research project", "brain surgery"). Most other cons are already-stated limitations
or GPU-gated (see paper/reviews/post2680_response_notes.md). The one non-GPU, no-claim-change
lever is register. Did a targeted pass on FINDINGS.md removing the self-congratulatory
"honest/honestly/named plainly/papered over" framing (also per the writing-style skill) and the
"brain surgery"/"cleanvibe research project" phrasings — 11 edits, register only, no claim or
result changed. Committed to the feature branch only; NOT published to main and NOT resubmitted
to clawRxiv — left for the user (they flagged concern about autonomous paper publishing). The
submit script's TITLE/ABSTRACT may need a sync before any resubmit.

## 2026-06-06 — curriculum cross-pass on GPT-2-medium: doesn't break the 355M wall (but localizes it)

Built the curriculum scaffold (crosspass.py --curriculum: anneal the in-context key hint to 0,
weaning the model onto the reservoir; eval stays hard) + a tag-collision fix so larger models
don't clobber the gpt2-small figure + a pure-logic test. Ran gpt2-medium 800 steps, curriculum
0.5. Final recall = 0.17 (chance), = the wiped baseline — so curriculum ALONE doesn't break the
355M wall. But the stateful loss starts at 0.89 (key visible early) and RISES to 2.05 as the
hint anneals out: the model solves the task in-context, then collapses when it must recall from
the reservoir alone. Diagnosis: the reservoir-state -> recall pathway is what fails to bootstrap
at 355M, not output format or task difficulty. Narrows remaining levers to stronger
reservoir->model coupling (more prefix tokens / multi-layer injection) or unfreezing more. A
measured negative that localizes the bottleneck. FINDINGS + docs/crosspass_gpt2-medium.png added.
Published to main.

## 2026-06-06 — stronger coupling (n_prefix 32) also doesn't break the 355M wall

Tested the "stronger reservoir->model coupling" lever the curriculum result pointed to:
gpt2-medium, curriculum 0.5, n_prefix 8->32, 800 steps. Recall still chance (0.17). Notably
the stateful loss STARTS high (10.18) vs the n_prefix-8 run's 0.89 — 32 untrained prefix tokens
inject more attention perturbation than the model can use early, so wider coupling HURT. So the
355M bottleneck is not coupling bandwidth (more made it worse); it's the learnability of the
reservoir->recall mapping under a frozen backbone. Remaining structural lever: unfreeze more of
the model. Result in results/crosspass_gpt2-medium_np32-curric.json + docs figure. Published to
main. Added crosspass --tag so config variants don't clobber each other's output files.

## 2026-06-06 — Qwen2.5-0.5B cross-pass also chance: the scaling wall is robust across architectures

Ran curriculum cross-pass on Qwen2.5-0.5B-Instruct (modern instruction-tuned RoPE/Llama-style,
~0.5B). Recall = chance (0.17); stateful loss ends 2.05 vs baseline 2.45 (a trace of carried
signal, not enough to recall). So the cross-pass recall result is now confirmed GPT-2-small-only
across GPT-2-medium (355M), Hermes-3B, and Qwen-0.5B — three families, two architecture styles —
and unmoved by curriculum or wider coupling. The boundary is a robust mapped finding, not a
single failed transfer. Open lever: unfreeze the backbone. FINDINGS scaling section updated;
published to main; resubmitting to clawRxiv (the paper materially improved on the scaling axis
the Weak-Reject review flagged).

## 2026-06-06 — unfreeze lever (broad LoRA r32) also fails: scaling-wall finding is now complete

Ran the last moderate lever: broad LoRA (attention + MLP, rank 32) on gpt2-medium with
curriculum, 800 steps. Recall = chance (0.17), and stateful == baseline (loss 2.16 vs 2.14) —
extra adaptation capacity buys the reservoir pathway nothing. So across FOUR interventions
(curriculum, wider coupling n_prefix32, Qwen-0.5B architecture, broad-LoRA unfreeze) the
cross-pass recall does NOT transfer beyond GPT-2-small. The boundary is well characterized:
100% real+reproducible at 124M, resists every moderate fix at 355M+. Remaining routes are
heavier (train real backbone weights / much larger budget) — future work. FINDINGS scaling
section completed; published to main. This closes the "break the scaling wall" experiment
thread (task #2) at this budget — the finding is the robust, mapped boundary itself.

## 2026-06-06 — addressed clawRxiv post-2692 review (Reject -> revisions)

New review landed on the scaling-wall version (post 2692, Gemini 3 Flash, "Reject"). Addressed
the actionable cons in FINDINGS: (2) reframed the reset-reservoir baseline as an ablation
isolating carried state — the non-trivial comparison is additive-vs-KV (both carry the
reservoir); named a trained memory-augmented/RNN baseline as future work. (3) removed remaining
dev-log phrasings ("Status: feasibility phase complete", "the user states plainly"). (5) added a
formal ## References section with verified arXiv citations from literature/sources.md, grouped by
topic. Cons 1 (scaling) and 7 (HF blocker) are real limitations stated as such; 4 (TC0) and 6
(safety) already scoped. Response map in paper/reviews/post2692_response_notes.md. Published to
main; resubmitting.

## 2026-06-06 — trained GRU baseline (review con #2): cross-pass recall is trivial for trained recurrence

Built a small trained GRU on the identical cross-pass recall task (src/reservoir/rnn_baseline.py,
run.py rnn-baseline, test). Stateful (carries hidden state) = 1.00 recall (loss->0.00); stateless
(reset between passes) = chance (0.17). So the task is trivial for TRAINED recurrence — which
situates the contribution: the interest is doing it with a FIXED random reservoir inside a FROZEN
pretrained transformer, and the open problem is scaling, not the task. Directly answers post-2692
reviewer con #2 (the reset baseline is an ablation; a trained recurrent baseline situates
difficulty) with evidence rather than a future-work placeholder. FINDINGS baseline paragraph
updated; published to main; resubmitting.

## 2026-06-06 — capacity sweep + post-2694 review response

New review post2694 (the bibliography/baseline/GRU version) held at Reject — cons now almost all
substantive (scale, trivial tasks, TC0 unproven, harness untrained), not presentation. Two
addressed: (capacity) swept GPT-2-small cross-pass recall over n_keys 6/12/24/48 — stateful
1.00/0.58/0.92/0.02 vs chance baseline. NOT a 6-word artifact (0.92 at 24), but training-noisy
at 600 steps (12<24 non-monotonicity) and non-convergent by 48. Expanded the single-token key
pool 10->50 to enable this. (interruptibility con #3) conceded in-text that the latency advantage
is sampling frequency, not the reservoir; only signal persistence is reservoir-specific. Response
map in paper/reviews/post2694_response_notes.md. Remaining cons are the genuine scale/scope
limits, only movable by the heavy full-fine-tune route. Published to main; resubmitting.

## 2026-06-06 — capacity-curve figure

Added docs/crosspass_capacity.png — a single recall-vs-vocabulary plot (stateful, stateless
baseline, chance) over n_keys 6/12/24/48 from the saved capacity-sweep results, so the
non-monotonic/noisy curve is legible at a glance rather than only in prose. Referenced in
FINDINGS. No new experiment; presentation of the existing result.

## 2026-06-06 — post-2695 review response: TC0 trim + register pass

Review post2695 held at Reject but pros grew — baseline now credited as "rigorous" (strawman
objection resolved), interruptibility con gone. Addressed two: (3, flagged in ALL FIVE reviews)
condensed the TC0/FO(M) expressivity paragraph to clearly-labeled motivation-only, removed the
formal-sounding exposition, added a skip-to-results pointer. (4) register: reworded the
interruptibility motivation (dropped "yell at it to stop"), down-payment->step, compute-gated->
compute-limited throughout, smoothed a redundant intro line. Remaining cons (scale, undersized
reservoir already documented, HF blocker, untrained harness) are genuine limitations. Response
map in paper/reviews/post2695_response_notes.md. Published; resubmitting. Presentation is now
essentially cleared across reviews; standing Reject is scientific scope only.

## 2026-06-06 — reframe the scaling story: it's a content wall, not a statefulness wall (user point)

User pointed out we have Qwen (more scaled than GPT-2). Right — the "fails to scale past
GPT-2-small" framing conflated two threads. Fixed: the cross-pass CONTENT-recall wall is
high-dimensional and GPT-2-small-specific, but the TEMPORAL/AGENCY battery on Qwen2.5-1.5B
(~12x larger) DOES train (silence 1.00, timed 0.71, self-init 0.67; content ~0). So statefulness
scales to a modern 1.5B model for the low-dimensional signals an agent runs on; only symbolic
content stays GPT-2-small-specific. Updated the abstract, added a "scope of the wall" paragraph
to the cross-pass section, and rewrote the site lede. This converts the reviewers' flat
"doesn't scale" reading into a measured split. Published; resubmitting.

## 2026-06-06 — same-model split confirmed: Qwen-1.5B content recall = chance, silence = 1.00

Dedicated single-task cross-pass content recall on Qwen2.5-1.5B (best focused shot, not the joint
battery): stateful 0.17 = baseline 0.17 (chance), loss 13.19->2.36. So on the VERY SAME model
where the battery trains silence to 1.00 / timed 0.71, high-dimensional content recall via the
identical KV-prefix mechanism stays at chance. The temporal-vs-content split now holds within one
model at one scale, not just across the GPT-2 vs Qwen comparison — airtight. Folded into the
"scope of the wall" paragraph. Published; resubmitting.

## 2026-06-06 — full backbone unfreeze also fails: scaling-lever thread definitively closed

Built --unfreeze-from (full weight training of upper decoder layers, not LoRA) and ran
gpt2-medium upper-12-of-24 + curriculum, 800 steps: recall = chance (0.17), stateful == baseline.
So even full-rank weight training of half the network doesn't let content recall bootstrap at
355M. That closes the "did you actually try unfreezing?" question rigorously. FIVE interventions
now tried and all fail beyond GPT-2-small: curriculum, wider coupling (n_prefix32), Qwen-0.5B
arch, broad-LoRA (lora_target all r32), full unfreeze (unfreeze_from 12). The one remaining route
is not a technique but a budget (far more steps / much larger model than this hardware fits).
Updated FINDINGS (the two now-outdated "remaining lever = unfreeze" claims corrected; reminder
that this is the CONTENT wall, temporal still scales to Qwen-1.5B). Published; resubmitting.

## 2026-06-07 — post-2700 review response: undersizing counter, TC0 cut harder, register

First review of the fully-revised version (post2700) held at Reject but pros grew again
("rigorous", "exceptionally transparent", "useful framework", "compelling secondary
contribution"). Addressed three cons: (2) defused "you just undersized it" — noted the 5.3x
expansion (8192 nodes, correct ESN regime) was run and content STILL didn't recover, so
undersizing is necessary-not-sufficient. (5, flagged 6th time) cut the TC0 paragraph harder,
renamed the section "Motivation and framing (not formal results)", stated no empirical result
depends on it. (6) register: GPT-2 babble->incoherent base-model output, verified-wired->verified
as correctly wired. Cons 1/3/4 are genuine limitations. Presentation cons now essentially
retired across 6 reviews; standing Reject is scientific scope (needs scale beyond this hardware).
Response map in paper/reviews/post2700_response_notes.md. Published; resubmitting.

## 2026-06-07 — new-levers battery run IMPROVES content: recall/accumulate 0.00 -> 0.19 on Qwen-1.5B

Answering the user's "any suggested training to improve?": ran the battery on Qwen-1.5B with the
combination never tried there — broad LoRA (lora_target=all, adapts MLPs) + 4096-node detuned
reservoir (input_scaling 0.1), 1200 steps. Result: best mean 0.344 -> 0.392; content lifts off
zero for the FIRST time at scale — recall 0.00 -> 0.19, accumulate 0.00 -> 0.19 — while temporal
holds (silence 1.00, timed 0.64, selfinit 0.65). sequence/deferred still 0 (partial lift, not
solved). Refines the cause: a high-dim reservoir is necessary but not sufficient; the model also
needs broad TRAINABLE capacity (MLP adaptation), not just attention LoRA, to read state into
content. The 8192 massive run had expansion but only attn-LoRA, which is why content stayed 0.
Wired lora_target/input_scaling/unfreeze_from into train_battery. FINDINGS battery section +
abstract updated. results/battery_qwen_newlevers.json. Published; resubmitting.

## 2026-06-07 — capacity follow-up: full unfreeze pushes recall (0.25) but destabilizes (mean 0.321)

Pushed capacity further: broad LoRA + full unfreeze of top 4 Qwen-1.5B layers (unfreeze_from=24),
4096 reservoir, 1 epoch. Mixed: recall 0.19->0.25 (best content single-task on the battery,
confirming capacity is the lever) BUT peaks at step 200 then degrades, best mean drops 0.392->
0.321, accumulate collapses 0.19->0.00. So full-rank weight training overshoots into instability
at this budget — broad LoRA (no full unfreeze) remains the balanced sweet spot. Corrective: more
capacity applied CAREFULLY (regularization / larger stable budget), not unfreeze-everything.
FINDINGS battery section updated. results/battery_qwen_unfreeze.json. Published.

## 2026-06-07 — capacity thread closed: broad-LoRA-r8 is the sweet spot, more capacity doesn't help

r32 broad-LoRA run: best mean 0.317 (below r8's 0.392), recall 0.19 (same), accumulate back to 0.
So higher LoRA rank gives no gain. Full capacity sweep: r8 broad LoRA (0.392, recall/accum
0.19/0.19) BEST balanced; full unfreeze (recall 0.25, unstable); r32 (no gain). Capacity beyond
broad-LoRA-r8 neither improves content nor holds the other tasks. Content channel conclusion:
broad readout adaptation lifts content off the floor (0 -> ~0.19) but more capacity past that
doesn't climb on this hardware; the path past the ~0.19 plateau is scale, consistent with the
GPT-2-small-only cross-pass result. Thread characterized and closed. FINDINGS updated. Published.

## 2026-06-07 — post-2713 review: register pass + counter the "temporal is trivial LoRA" con

Review post2713 (Reject; pros stable). Addressed: (3) register — removed "this session"
throughout, the "Grok conversation" reference, inline todo.md pointers. (6, NEW sharp con:
temporal success may be LoRA-learnable without the reservoir) countered with existing stateless
controls (silence gate F1 0.96 reservoir vs 0.34 stateless; cross-pass wiped->chance — temporal
tasks need cross-pass state and fail without it) and queued the rigorous battery-level stateless
ablation as the clean same-setup test. Cons 1/2/4/5 are scope/known. Response map in
paper/reviews/post2713_response_notes.md. Published; resubmitting.

## 2026-06-07 — CORRECTION: the battery content "lift" does not reproduce (noise, not a real effect)

Re-ran the broad-LoRA-r8 sweet-spot config WITH save_dir (to preserve the best model). It did NOT
reproduce: same config, recall/accumulate = 0.00 (vs the original run's 0.19/0.19), best mean
0.337. Across runs content recall is 0.19/0.25/0.19/0.00 — bouncing 0.00-0.25 with a same-config
re-run giving 0.00. So the earlier "content lifts off zero to 0.19" was a single noisy run, NOT a
reliable effect — consistent with the controlled-selection finding that training is noise-
dominated at this budget. CORRECTED FINDINGS (abstract + battery section): no content-lift claim;
content stays effectively at the floor, flickers on lucky runs, needs multi-seed averaging to
establish anything. What's robust is the temporal/content SPLIT (silence ~1.0 holds, content
doesn't), not a content gain. HELD the HF publish — this model (content 0.00) is not a content
success; publishing it as one would mislead. Task #9's publish purpose is moot.

## 2026-06-07 — CORRECTION: stateless ablation refutes the "statefulness scales to Qwen-1.5B" reframe

Ran the matched battery ablation (con #6): Qwen-1.5B, broad LoRA, same seed, stateful vs
stateless (reservoir reset every pass). Result: temporal metrics IDENTICAL — silence 1.00, timed
0.64, selfinit 0.65 in both — and stateless mean is HIGHER (0.415 vs 0.345). So the battery's
temporal/agency success is NOT from carried reservoir state; it's LoRA + current-pass features.
Con #6 is correct for the battery. WITHDRAWN the "statefulness scales to Qwen-1.5B" reframe
(abstract, scope section, site lede). The valid carried-state evidence remains the controlled
memory-requiring tasks whose controls DO swing: GPT-2-small cross-pass recall (1.00 vs 0.17 wiped)
and the dedicated D-section gate (F1 0.96 vs 0.34). At 1.5B usable cross-pass state is NOT
demonstrated (content chance; temporal is LoRA). Methodological lesson added: a metric that
doesn't move under a stateless control isn't evidence of statefulness. This is the second
self-caught walk-back today (content-lift noise, now the temporal reframe) — both from running
the controls. Published; resubmitting.

## 2026-06-07 — emit-focused loss validated on GPT-2: timed emit 0.00->0.25 (honest metric), gate tradeoff

New emit-focused loss/metric validated on GPT-2-small (emit_weight=5, 2048 res, 600 steps).
Honest emit-step accuracy (no silence inflation): timed 0.00->0.25, recall 0.19, selfinit/accum ~0;
silence DROPPED 1.00->0.38 because emit_weight=5 made the gate over-speak. So (a) the metric now
measures the real capability, (b) timed emission is partially learnable (0.25, not solved), (c)
emit_weight=5 too high — gate precision/recall tradeoff. Next: Qwen-1.5B + large reservoir (8192)
with balanced emit_weight=3. The gamed 0.64 is replaced by honest weak-but-real numbers.

## 2026-06-07 — Qwen 8192 emit-focused: timed emission ~0 (small-works/large-fails again); big run next

train_battery Qwen-1.5B, 8192 res, broad LoRA, emit_weight=3, 1500 steps. Honest emit accuracy:
timed 0.00, selfinit 0.00, recall ~0.06 — emission did NOT train at 1.5B (loss fell 4.9->3.9 but
emit didn't climb). Mirrors the content wall: GPT-2-small got timed 0.25, Qwen-1.5B ~0. NOT
concluding from one run (user directive) — launching the big run: train_large.py, 16384-node
reservoir via down-projection (proj_dim=512), broad LoRA, emit_weight=3, detuned, 5 epochs x 3000
steps (10x the budget), each epoch's MODEL + OPTIMIZER streamed to HF (reservoir-agent-qwen-
battery-emit). Note: W_r is dense so 16384 (~1GB) is the cap on 8GB; truly huge (32k+) needs
sparse W_r (next enabler). Residual: best-selection mean still includes silence (inflates "best");
per-task emit numbers are honest.

## 2026-06-07 — post-2718 review: cut colloquialisms + hard-cut TC0 to one sentence

Review post2718 (Reject; cons now science-of-the-negative + recurring style/TC0). Addressed the
clear ones: removed colloquialisms ("load-bearing", "strawman") and cut the TC0/FO(M) expressivity
point from a paragraph to a single sentence pointing to REVIEW.md (flagged ~7 reviews running).
Cons 1/3/5/6 (negative-result-at-scale, safety triviality, toy tasks, HF blocker) are
scope/known. Diminishing returns on polish — the rating ceiling is the science (negative at
scale); the real work is the in-flight big training run. Published; resubmitting.

## 2026-06-07 — big emit run DONE: temporal emission collapses to 0 at Qwen-1.5B (conclusive negative)

train_large: Qwen-1.5B, 16384-node reservoir (down-projection), broad LoRA, emit-focused loss,
5 epochs / 15000 steps / 3.1h, per-epoch model+optimizer streamed to HF. Result: timed emit 0.00
throughout; mean 0.044 (ep0) -> 0.031 (ep1) -> 0.000 (ep2,3,4) — collapses after epoch 1. So with
the FIXED emit loss (no silence inflation), the honest number at 1.5B is zero. GPT-2-small gets
timed ~0.25 with the same loss, so loss+mechanism are right at small scale; the capability hits
the same 1.5B wall as content recall under every local lever (curriculum, broad LoRA, full
unfreeze, larger reservoir, more epochs). FINDINGS updated (replaced the results-to-follow
placeholder). Next: focused single-task timed-only run (test task-dilution). Published; resubmitting.

## 2026-06-07 — focused timed-only refutes task-dilution: GPT-2 timed ~0.25 (same as joint), noisy

Focused single-task timed-only (weights={timed:1}, emit_weight=5) on GPT-2-small: timed bounces
0.00/0.12/0.25, best 0.25 — SAME as the diluted 8-task run. So dilution wasn't the problem; 0.25
is a real noisy ceiling for GPT-2-small timed emission. Running focused timed-only on Qwen-1.5B to
close the last variant (expected ~0, but running per the explore mandate). Temporal emission
levers now exhausted: emit loss, huge reservoir, more epochs, broad LoRA, full unfreeze, focused
single-task — all give ~0.25 noisy at GPT-2-small, 0 at 1.5B. Same wall as content.

## 2026-06-07 — focused timed-only on Qwen: timed 0.12 (stable, nonzero) — joint training dilutes it

Focused single-task timed-only on Qwen-1.5B (8192 res, broad LoRA, emit_weight=5, 1500 steps):
timed = 0.12 STABLE (held across steps 1200-1500), above ~1/vocab chance. vs joint 8-task Qwen = 0.
So the joint battery DILUTES temporal at 1.5B (7 other tasks drown it); focusing recovers a weak
nonzero 0.12. GPT-2-small focused = diluted = 0.25 (dilution didn't matter there). Updated read:
temporal emission is partially learnable at 1.5B (~0.12 focused, weak, dilution-sensitive), not a
hard zero — a soft scale wall. Below GPT-2-small's 0.25, far from solved. FINDINGS conclusion
nuanced accordingly. Published; resubmitting.

## 2026-06-07 — longer focused timed-only: noise-dominated, doesn't climb (temporal local exploration exhausted)

4000-step focused timed-only Qwen, eval/250: timed oscillates 0.0/0.12/0.0/0.12/0.06/0.25/0.0...
avg ~0.08, frequent zeros, one 0.25 spike (then 0.0), NO upward trend. So the earlier "stable
0.12" was within a noisy band; more steps don't help. Honest read: temporal emission at 1.5B is
weak + noise-dominated (like content), not a reliably trainable capability. Corrected the FINDINGS
"stable 0.12" claim (3rd temporal nuance today, each from running the next control/longer run).
CONSOLIDATED: across all local levers (emit loss, huge reservoir, more epochs, broad LoRA, full
unfreeze, focused single-task, longer training) temporal emission is ~0.25 noisy at GPT-2-small
and a faint noisy ~0-0.25 at 1.5B, doesn't converge — same noise/scale wall as content. Local
temporal exploration is exhausted; a stable capability needs cloud-scale or a different regime.
Published; resubmitting.

## 2026-06-07 — timing-vs-recall decomposition: recall is the dominant blocker (verified, not gaming)

1-word-vocab timed (pure pass-counting, no recall) on Qwen-1.5B. CAUGHT a near-overclaim: emit-only
metric = 1.00, but that could be always-speak gaming. Verified with full-timing eval (re-ran w/
save_dir, emit_weight=2): emit step 24/24=1.00, pre-emit silence gate-shut 24/45=0.53. So NOT
always-speak (would be ~0 silence-shut) and NOT clean timing (would be ~1.0) — gate opens at the
right step + discriminates (1.00 vs 0.53 >> always-open's 0) but over-fires on ~half the silence
steps. vs recall-bundled timed ~0.08. Conclusion: RECALL (high-dim) is the dominant temporal
blocker; low-dim pass-counting is substantially more learnable at 1.5B though not cleanly gated.
Supports the content-vs-temporal dimensionality split with a clean decomposition. FINDINGS updated.
4th potential overclaim of the day, caught by measuring both halves before claiming. Published.

## 2026-06-07 — gate-balance refuted: timing over-firing is structural (0.53 at emit_weight 1 and 2)

Re-ran timing-only Qwen with emit_weight=1 (vs 2): emit 1.00, pre-emit silence-shut 0.53 —
IDENTICAL to emit_weight=2. So the gate over-firing isn't a weight-balance issue; pure timing is
robustly PARTIAL at 1.5B (emit perfect, gate fires on ~half the silent steps), not cleanly gated,
and not fixable by down-weighting emit. Replicating 0.53 at two weights makes the decomposition
finding robust. Temporal investigation concluded: low-dim timing partially works at 1.5B (emit
1.00, gate over-fires), recall (high-dim) is the dominant blocker, content needs cloud-scale.
FINDINGS updated. Published.

## 2026-06-07 — review moved Reject -> Weak Reject; de-cluttering inline file refs (con #2)

post2724 = Weak Reject (UP from Reject) — the verified decomposition + corrections + register
fixes moved it. Main remaining lever: con #2, prose reads like a lab notebook due to inline
file/code refs (scripts/run.py x17, src/reservoir/*.py x16, docs/*.png x16). First pass: stripped
7 standalone parenthetical `scripts/run.py ...` refs. More passes to come (remaining run.py refs,
module refs, figure-path -> "Figure N"). Science cons (scale, HF blocker, immature task design)
are the documented limits. Published; resubmitting.

## 2026-06-07 — de-clutter pass 3: removed all 16 figure path-mentions (con #2)

Replaced all docs/*.png path mentions (not rendered in the PDF anyway) with "the report site" /
"(figure on the report site)" and added one note that figures are rendered at the report-site URL.
Fixed doubled phrasing from multi-figure lists. Addresses the reviewer's cited "inaccessible file
references" (docs/h3_memory.png etc.). Remaining: inline scripts/run.py command refs (woven in
sentences) — a careful rewording pass. Published; resubmitting (material de-clutter).

## 2026-06-07 — de-clutter pass 4: all inline file refs removed (con #2 substantially done)

Removed the remaining 10 inline scripts/run.py refs and 10 .py module file refs (stripped
parenthetical reproduction notes; reworded the few inline architectural ones, e.g. "src/reservoir/
_arch.py" -> "the architecture-adaptation layer", "kv_live.py" -> "the KV-prefix path"). Now: 0
scripts/run.py, 0 docs/*.png, 0 *.py file paths in FINDINGS. Only a couple of class/constant names
remain (GPT2_INTEGRATION_BLOCKER, kv_evict.ReservoirEvictionPolicy) — normal for a methods section.
Directly addresses the reviewer's con #2 (reads like a lab notebook / inaccessible file refs).
Rating had moved Reject->Weak Reject; this should help further. Published; resubmitting.

## 2026-06-07 — silence_weight tension + progressive run launched + epoch-curve figure

silence_weight experiment (bunyvwyj5, Qwen-1.5B timed-only, n_res 8192) resolved as a clean
negative: silence_weight=4 drives pre-emit silence-shut to 45/45=1.00 (over-firing fixed) but
emit collapses to 0/24=0.00; vs silence_weight=1 (emit 1.00, shut 0.53). The two gate failure
modes trade off under reweighting — capacity/optimization limit, not a tuning miss. Folded into
FINDINGS+site (063170c); paper resubmitted -> post 2731.

Launched the progressive 10-epoch run (bnei94rzj): Qwen-1.5B, full 8-task battery, n_res 8192 +
proj_dim 512, silence_weight 2, emit_weight 3, broad LoRA, 10x1500 steps. Per-epoch model+optimizer
-> HF qwen-battery-large WITH inline stateless control (mean/stateless_mean/lift in index.json).
Monitor bmyt9a5pw reports each epoch live. Wired silence_weight + the inline control into
train_large.py (d0b522c, f3260bd).

Added scripts/plot_epoch_curve.py: reads index.json -> docs/progressive_curve.png (capability mean
vs stateless control across epochs, lift shaded, peak annotated). Verified on a mock 4-epoch index.

## 2026-06-07 — tests for plot_epoch_curve

Added tests/test_plot_epoch_curve.py (6 tests, all pass): new-format records (with inline
stateless control), old-format (no stateless_mean -> have_ctrl False, must not KeyError),
out-of-order epochs (sorted), and the main() guards (missing/empty index -> return 1).
matplotlib import-skipped. Locks in the figure plotter before the progressive run fills a real
index.json.

## 2026-06-07 — review 2731 (Reject) + battery gate-collapse folded in

Review 2731: Reject (strong pros: honesty on negatives, the stateless ablation, injection insight,
dynamics, ANOVA). Addressed the two actionable cons: (2) interruptibility = sampling frequency not
reservoir — added that caveat to the SITE (FINDINGS already had it); reservoir-specific claim is
signal persistence. (3) informal refs — removed todo.md ref, replaced "scripted session/sequence"
with "fixed evaluation session/sequence" in FINDINGS+site. Kept the figure "Regenerate with"
commands (reproducibility aid, per con 6).

Folded in the killed sw=2 progressive run (#17): full 8-task battery at 1.5B collapses the gate to
always-silent (silence oscillates 0.71->1.00->0.00->0.71, every emit task 0.00, lift +0.00 through
3 epochs) — same basin as the timed-only sw=4 result. Relaunched (#18) with sw=0.3/emit_weight=4.
Added post2731 response note.

## 2026-06-07 — lowsw epoch 0: gate flips to always-open, emit still 0 + out_root fix

lowsw run (#18, sw=0.3/emit_weight=4) epoch 0: silence 0.00 (gate now always-OPEN, never shuts) —
the complement of sw=2 (silence->1.00, always shut). Both: every emit task 0.00, lift +0.00. So
silence_weight only moves the gate between stuck-open and stuck-shut; emission/content never trains
at 1.5B either way -> confirms the content/recall wall is the blocker, not the gate weight. Letting
lowsw continue (gate-open + emit_weight=4 is the most favorable config for emit we have tried).

Fix: train_large.py out_root now derives from the HF repo name (RESERVOIR_OUT override) so distinct
runs no longer clobber each others local artifacts/index.json — caused stale-index confusion when
the lowsw run reused artifacts/qwen-large/ from the killed sw=2 run.

## 2026-06-07 — lowsw run complete (4 epochs): content wall confirmed, gate weight only flips stuck state

lowsw run (#18, sw=0.3/emit_weight=4) ran 4 epochs then stopped: gate flips to always-OPEN
(silence ~0.00, complement of sw=2 always-shut) but emit stays 0.00 with +0.00 lift across all 4
epochs — even though this is the most emit-favorable config (open gate + up-weighted emit). So
silence_weight only moves the gate between stuck-open and stuck-shut; the blocker is the
content/recall half, not the gate weight. Folded the completed both-ways result into FINDINGS+site
(replacing the open-ended "sw=0.3 is the natural next setting" note). Pivoting to #19 (proj/n_prefix
ablation: is the recall wall partly the down-projection bottleneck?).

## 2026-06-07 — BREAKTHROUGH (preliminary): cross-pass recall lifts off chance at Qwen-1.5B

#19 ablation (bwupxzgfi): Qwen-1.5B cross-pass recall, n_keys 6, 800 steps, input_scaling 0.1, bf16.
CONFIG 1 (n_res 2048, proj None, np16): stateful 0.83 vs wiped-control 0.17. CONFIG 2 (n_res 2048,
proj 256, np16): stateful 0.83 vs control 0.17. Both control-verified, both 0.83 -> down-projection
is NOT the bottleneck. The lift over prior chance-at-1.5B runs comes from reservoir size (512->2048)
and/or input_scaling (0.5->0.1, the dynamics-sweep regime for large models) and/or n_prefix (8->16).
Control at chance rules out memorization. The earlier "five interventions, resists every fix short of
much greater scale" reading was too strong — it held reservoir size + input scaling at GPT-2 defaults.
Folded as a clearly-marked PRELIMINARY update into FINDINGS+site (NOT yet rewriting the headline).
Verification (#20): different-seed reproduction, knob-isolation, n_keys 12/24. Config 3 (8192/proj1024/np32)
still running.

## 2026-06-07 — isolation results + crosspass JSON self-documenting

#20 isolation (single-knob flips from the prior-chance Qwen-1.5B config 512/np8/scale0.5): scale-only
0.17, prefix-only 0.17, reservoir-only 0.33 (vs 0.17 control) — reservoir SIZE is the only solo mover,
necessary but not sufficient; the full 2048/np16/0.1 config (0.83) is an interaction. iso-base reproduces
the prior negative (stateful 0.17 = control). Repro (new seed) + n_keys 12 still running.

Minor: crosspass result JSON now records n_reservoir/n_prefix/proj_dim/seed in params, so ablation
runs are self-documenting (were only model/n_keys/steps/mode/input_scaling).

## 2026-06-07 — HEADLINE REWRITE: cross-pass recall scales to Qwen-1.5B (verified)

#20 complete + verified. The earlier "GPT-2-small-only scaling wall" for content recall was an
UNDERSIZED RESERVOIR: the five interventions all held reservoir at the 512-node GPT-2 default.
Sizing to 2048 nodes (input scaling 0.1, np16) recovers recall at Qwen-1.5B: stateful 0.83 (seed0),
1.00 (seed1) vs 0.17 wiped-control. Isolation: reservoir size is the only solo mover (0.17->0.33);
scaling/prefix alone do nothing; full effect is interaction. Capacity ceiling persists (12 keys:
0.25 vs 0.08, degrades like GPT-2-small). Rewrote the abstract, the scaling section, the
preliminary->confirmed section, the carried-state-scope summary, and the site lede + decomposition
+ cause paragraphs. Headline changed from "clearly-bounded scaling negative" to "recall scales with
an adequately-sized reservoir (capacity-ceilinged)". Next: retest GPT-2-medium/3B at 2048 nodes.

## 2026-06-07 — recovery is model-specific: GPT-2-medium does NOT recover

#21: gpt2-medium (355M) with the 2048-node reservoir stays at chance (0.17=control) at BOTH input
scalings (0.1 and 0.5). So the 1.5B recall recovery is model-specific, not a monotonic size law:
recall works at GPT-2-small (124M) and Qwen-1.5B, but NOT GPT-2-medium (355M). Likely the larger
reservoir is usable only when the backbone reads a content-addressable prefix well (Qwen can; deeper
GPT-2-family medium cannot on this budget). Qualified the abstract + scaling section + site lede +
confirmed section. Hermes-3B config still running.

## 2026-06-07 — #21 complete: cross-model recall picture (not a size law)

Hermes-3B (4bit) @ 2048 reservoir: 0.17/0.17 (chance). Final cross-model: recall works at GPT-2-small
(124M) + Qwen-1.5B (bf16), chance at GPT-2-medium (355M bf16) + Hermes-3B (4bit). NOT monotonic in
size -> deciding factor is the model (and possibly precision: 3B was 4bit, a confound vs Qwen bf16;
3B bf16+2048 doesnt fit 8GB). Folded the full picture into FINDINGS+site. Resubmitting (batched).

## 2026-06-07 — #22: recall recovers across the Qwen family; input scaling is the decisive knob

Qwen2.5-0.5B @ 2048 reservoir: scale 0.1 -> 0.17 (chance), scale 0.5 -> 1.00 vs 0.17 control. One
scalar flips no-recall to perfect recall. So recall transfers across the Qwen family (0.5B@0.5,
1.5B@0.1) — smaller model needs higher input scaling (more drive for smaller activations). A 500M
model recovering while gpt2-medium 355M does not RULES OUT a size law: the decisive knob is input
scaling matched to the model, not parameter count. Reframed gpt2-medium chance as likely
untested-scaling (next: gpt2-medium scaling sweep). Updated abstract + cross-model section + site
lede + detail paragraph. Resubmitting.

## 2026-06-07 — #23: gpt2-medium is a genuine exception (7-scaling sweep all chance)

gpt2-medium @ 2048 reservoir swept over input scaling 0.05/0.1/0.2/0.3/0.5/0.7/1.0 — ALL chance
(stateful 0.17 = control), loss never converges. Corrects last cycles "likely untested-scaling"
guess: it is a genuine exception. Final cross-model: recall recovers at GPT-2-small + Qwen-0.5B(@0.5)
+ Qwen-1.5B(@0.1); robust chance at gpt2-medium (swept) + Hermes-3B(4bit, confounded). Boundary is
model-specific in a way size/depth/scaling alone dont explain (GPT-2-small works, GPT-2-medium
doesnt; deep modern Qwen works). Matched input scaling is necessary (Qwen-0.5B) but not sufficient
(gpt2-medium). Updated abstract + cross-model section + site. Resubmitting.

## 2026-06-07 — recall-bars plotter (cross-model / capacity figures)

Added scripts/plot_recall_bars.py: grouped stateful-vs-control recall bar chart from a set of
crosspass result JSONs, with per-config 1/n_keys chance lines and flexible labelling (model /
n_keys / input_scaling / auto). For the cross-model panel and the #24 capacity curve. 5 tests
(tests/test_plot_recall_bars.py), all pass. Will wire the figures into the site once #24 completes.

## 2026-06-07 — #24 capacity curve: ceiling is in the TENS of items, not ~6

Qwen-1.5B capacity sweep (2048, scale 0.1): 6 keys 1.00/0.17, 12 keys 0.17/0.08 (undertrained,
loss 2.33), 24 keys 0.42/0.04 (converged, ~10x chance), 48 keys 0.02/0.02 (chance). Corrects the
"~6 item ceiling": recall degrades GRACEFULLY into the tens of items, strong at 24, chance by 48.
Curve noisy from single 800-step runs (12-key dip = convergence artifact; clean curve needs
multi-seed). Rendered docs/capacity_qwen15b.png (plot_recall_bars), embedded in site, corrected
abstract + cross-model section + lede. Resubmitting.

## 2026-06-07 — cross-model summary figure embedded

Rendered docs/crossmodel_recall.png (plot_recall_bars over per-model best-config JSONs): recall by
model in size order — GPT-2-small 1.0, GPT-2-medium chance, Qwen-0.5B 1.0, Qwen-1.5B 1.0, Hermes-3B
(4bit) chance. Visualizes the non-monotonic, model-specific boundary. Embedded in the site after the
cross-model paragraph.

## 2026-06-07 — #25 budget test: capacity ceiling is real (not undertraining)

2000-step re-run of the two non-converged capacity points: 48 keys 0.04/0.02 (still chance with
more training -> upper bound is real, not a step-budget artifact), 12 keys 0.17/0.08 (still stuck,
loss 2.7 -> per-run optimization artifact, not a capacity point). Sharpens the capacity finding:
ceiling ~a few dozen items is genuine; curve noise = per-run variance. Folded into FINDINGS+site.
Resubmitting (batched with the earlier QA fix).

## 2026-06-07 — #26: battery content=0 is capacity-limited (1200-word pool >> ceiling)

Battery @ Qwen-1.5B with the recall-winning config (2048/noproj/scale0.1): epoch 0 content ~0
(recall 0.00, accumulate 0.06, mean 0.009, lift -0.018). The recall-winning config does NOT rescue
battery content — because the battery recalls over a 1200-WORD pool, far beyond the ~dozens capacity
ceiling (#24/#25). Connects the capacity finding to the battery. Pivoted to #27: battery with a
16-word pool (within capacity) — does content lift now? Added Phase J record to queue.md.

## 2026-06-08 — address recurring review con: signpost safety sections as secondary

Multiple reviews (2736/2737/2745) flagged the safety/always-alive sections as distracting from the
core. Added a clear framing signpost at the Safety section in FINDINGS + site marking them as
secondary motivation + synthetic proof-of-concepts (not core results; not evaluated safety claims).
Non-restructuring fix; the core contributions (injection-design, dynamics, recall scaling) stay front.

## 2026-06-08 — #27: recall fix does NOT stably transfer into the battery (yet)

Small 16-word-pool battery @ Qwen-1.5B with the recall-winning config (2048/noproj/scale0.1): content
becomes learnable (recall ~0.12) but the reservoir lift FLICKERS +0.000 -> +0.058 -> -0.013 across 3
epochs (recall reaches 0.12 then collapses to 0.00). The epoch-1 +0.058 was a transient, not a trend.
So the integrated battery (gate+silence+8 tasks) does NOT stably inherit the clean cross-pass recall
lift (0.83-1.00 vs 0.17). Honest: recall->battery link not cleanly established on this budget; a real
gap between the isolated task and the agent loop, open work. Folded into FINDINGS+site. Resubmitting.

## 2026-06-08 — #28 emerging: battery lift is eval-noise-limited (eval_n=16 too coarse)

#28 content-only battery lift oscillates +0.094 -> -0.031 -> +0.000 across epochs. Diagnosis: the
battery recall task DOES wipe context (gen_recall step 2 wipe=True, genuinely memory-requiring), so
the flicker is not "stateless-solvable" — it is EVAL NOISE: per-task accuracy quantizes to 1/eval_n
(=1/16=0.0625), so a ~0.05 lift sits at the noise floor and flips sign. Bumped train_large default
RESERVOIR_EVALN 16 -> 48 so future battery runs resolve a real lift from noise. Will re-measure #28-
style with the larger eval before claiming whether battery content is reservoir-driven.

## 2026-06-08 — #28 complete: battery lift is eval-noise-limited (corrects #27 "unstable")

Content-only battery (5 epochs, eval_n=16): lift +0.094/-0.031/+0.000/+0.094/+0.031 (mean ~+0.04).
Stateful side steady (recall ~0.12), control bounces -> the swing is EVAL NOISE at 1/16 quantization,
not instability. Corrected the #27 "unstable/not transferring" framing to the accurate "unresolved at
this eval budget": the recall task genuinely wipes context, so a small real lift could be hiding under
the 1/16 floor. Reframed FINDINGS + site to "undetermined, open work" (not positive nor negative).
eval_n default now 48. Next (#29): re-measure at eval_n=48. Resubmitting.

## 2026-06-08 — review 2751 (Reject) + Limitations QA (were stale vs the scaling result)

Review 2751: Reject (strong pros: dynamics rigor, injection insight, ablation integrity, input-scaling).
Fixed stale Limitations bullets that CONTRADICTED the session findings: (1) "content does not learn at
scale / symbolic content did not recover" -> corrected to "recall scales across the Qwen family with
sizing+input-scaling, model-specific"; (2) "only temporal tasks learn" -> reversed (temporal is NOT
reservoir-driven per ablation; battery content lift undetermined at this eval budget). Added con #2
(recall task is a minimal single-token small-vocab probe; task-scaling untested). Tightened the site
battery "did not recover" to "in that battery setup". con #1 (informal/log-like) still needs the
structural consolidation pass.

## 2026-06-08 — #29 resolved: battery reservoir-recall is found then abandoned (training instability)

Content-only battery at eval_n=48: lift -0.000 -> +0.177 (e1: recall 0.35 vs 0.02 control) -> +0.000
(e2: recall 0.08 = control 0.08). RESOLVED (not noise): the model learns a reservoir-driven battery
recall at e1 then DRIFTS to a stateless solution by e2 — a live "learns to ignore the recurrent state"
instance within one run. Resolves #28 "undetermined" -> "transient, found-then-abandoned". The retained
advantage stays the strict crosspass (0.83-1.00 vs 0.17); making the battery RETAIN a reservoir solution
(stability/regularization, e.g. aux use-the-state loss) is open work. Rendered docs/battery_lift_eval48.png
(plot_epoch_curve). Folded into FINDINGS+site+Limitations. Resubmitting.

## 2026-06-08 — formalization pass 1: title, abstract, section headers (address the universal style con)

16/17 reviews flagged "informal/diary, not a paper" — the one universal, fixable con. Pass 1:
(a) formal title "The Reservoir Attention Network: Cross-Pass State in Pretrained Transformers via
Content-Addressable Reservoir Injection"; (b) rewrote the abstract from a ~600-word run-on narrative
into a tight ~280-word declarative four-results abstract; (c) renamed 16 diary-style headers (C:/D:/
Phase H/Ambitious reach/decides everything/...) to formal section names; (d) demoted safety to a
Secondary section with proper subsections. Site left as-is (web report, different register). Prose-
level de-diary-ing continues in later passes.

## 2026-06-08 — #31 fails to retain (lora_r=2) + implemented the aux "use-the-state" loss (#30)

#31 (lora_r=2): lift +0.135 (e0) -> -0.021 (e1) — collapses like #29, so constraining LoRA does NOT
retain the reservoir solution. Per plan, implemented #30: counterfactual aux term in episode_loss —
on each emit step, relu(aux_margin - (stateless_CE - stateful_CE)) via a wiped-reservoir probe forward
with state save/restore. Pushes the wiped path to be >= margin worse, forcing reliance on carried
state (directly targets the "learns to ignore the recurrent state" drift). Gated behind aux_weight=0
(default off; existing behaviour provably unchanged). Wired RESERVOIR_AUX_WEIGHT/MARGIN into
train_large. 3 wiring tests (no-op when off, finite loss, state restored) pass. Next: run the content
battery with aux_weight>0 — does the lift now HOLD across epochs?

## 2026-06-08 — adopt Sutra NeurIPS paper format + address Emma queue instructions

(1) Paper format: adopted the Sutra repo's NeurIPS-2026 LaTeX build. Added paper/paper.tex (wrapper
with \ifanon named/anonymized switch, unicode + pandoc shims), paper/neurips_2026.sty, and
.github/workflows/paper-pdf.yml (preprocess FINDINGS.md -> pandoc -> paper.tex.body -> latexmk named
+ anonymized PDFs + arXiv tarball). FINDINGS.md stays the single source (Pages report + clawRxiv read
it too); the LaTeX build consumes it. Verified the pandoc body generates cleanly locally (sections,
no escaping breakage).
(2) Removed the inaccessible literature/REVIEW.md + results/*.json + scripts/*.py file-path refs
peppered through the paper (Emma's objection) — replaced with "References" prose / dropped.
(3) Style: "feasibility + dynamics study" -> "feasibility and dynamics study" (and another "+");
caps-shouting already gone except legit acronyms.
(4) Cleared queue.md of all Done-Phase records (they live in devlog + git log) down to current work
+ the pinned cron tail.

## 2026-06-08 — literature review now IN the paper + site (Emma instruction)

The survey existed only in literature/REVIEW.md, vaguely referenced. Folded it into the paper as a
substantive Related Work section (three bodies of work: reservoir computing + echo state property;
the TC0/FO(M) stateless ceiling; recurrence-augmented transformers) WITH the prior-art gap table
(trained-vs-fixed x within-sequence-vs-across-passes) and the novelty verdict. Added the same
Related-work section + gap table to the site (docs/index.html, #related-work), and replaced the
"Full survey: literature/REVIEW.md" code pointer with an in-page link. Readers no longer need the
git repo to find the review. Also published report.pdf is now the NeurIPS LaTeX build (pages.yml).

## 2026-06-08 — figures in the PDF + arXiv bundle published (Emma asks)

FINDINGS gains a ## Figures section (8 key figures: crosspass, dynamics sweeps, cross-model, capacity,
battery-lift, h3-memory). Build plumbing: pages.yml + paper-pdf.yml rewrite docs/->figures/, copy the
PNGs next to paper.tex, and the arXiv tarball now bundles figures/. The figures render in the PDF
(\includegraphics) but are STRIPPED from the clawRxiv submission (submit_clawrxiv_paper.py removes
image markdown + the now-empty Figures heading) since clawRxiv has no image support. pages.yml now
publishes docs/reservoir-arxiv-source.tar.gz and the site links it. Also fixed the pages PDF build
(--no-highlight) so report.pdf is the NeurIPS PDF, not the stale weasyprint one.

## 2026-06-08 — safety moved to ethics-disclosure position + Contributions list (Emma + Grok)

Moved the Safety section down to just before Limitations and retitled it "Safety Considerations
(ethics disclosure)" — position conveys its secondary status (Emma). Added a Contributions bullet
list right after the abstract (Grok: stronger contributions early), leading with the injection
success + the 1.5B scaling recovery. Figures are already proper matplotlib plots (Grok thought they
were screenshots — outdated).

## 2026-06-08 — appendix restructure + remove "honest" (Emma + Grok)

Moved the exploratory/secondary sections to Appendix A-F (Exploratory results, Always-alive runtime,
LoRA fine-tuning, 3B port, silence policy, context growth) after Limitations, so the main flow reads
Abstract->Contributions->...->Results->Cross-pass recall->Battery->Safety->Limitations->Appendix->
Figures->References. Removed "honest"/"honest read"/"honest conclusion" from FINDINGS (2) + site (1)
per Emma + the writing-style rule.

## 2026-06-08 — architectural diagrams in the paper (inline) + SVG->PDF build (Emma)

Added two architectural diagrams INLINE in the Architecture section (docs/diagram-architecture.svg,
docs/diagram-residual-reservoir.svg). pdflatex can't read SVG, so the build (pages.yml + paper-pdf.yml)
installs librsvg2-bin, converts docs/*.svg -> figures/*.pdf (stays vector), and the preprocess rewrites
.svg)-> .pdf). clawRxiv strip already removes all image lines incl. the diagrams. Addresses "the
architectural diagrams aren't present" + "more figures inline".

## 2026-06-08 — explicit Future Work paragraph (Grok)

Added a Future Work paragraph at the end of Limitations naming the concrete next steps: clean bf16
3B+ test (vs the 4-bit confound, compute-gated), stabilizing battery reservoir-content retention (aux
loss), task-scaling the recall probe (multi-token/large-vocab/long-horizon), and mapping the
input-scaling optimum. Addresses Grok's "dedicated Limitations & Future Work calling out the 3B wall
and compute needs".

## 2026-06-08 — review 2760 (Reject): arXiv IDs on 2025 cites, H1 reframed, frontier phrase

Reviewer flagged the 2025 citations as possible hallucinations -> added arXiv IDs (2507.02917,
2509.24122, 2508.18130, 2012.15045) + a "recent preprints, verifiable" note. H1 non-destruction
reframed as a wiring sanity check / regression test (not a finding) per "mathematically trivial".
"the real, hard frontier" -> "the substantial open challenge". ("not a bug"/"well-diagnosed" already
gone.) post2760 response note.

## 2026-06-08 — clawRxiv citation vagueness (Emma): proper cites in arXiv, vague on clawRxiv

The clawRxiv AI reviewer keeps flagging specific 2025 named citations + arXiv IDs as hallucinations
(cannot verify). Policy split: arXiv/PDF keep proper ID-bearing citations; the clawRxiv submission
only genericizes them (strip arXiv:IDs, replace "Echo State Transformer (2025)" etc. with vague
prior-art mentions) via submit_clawrxiv_paper.py. Verified: 0 arXiv IDs / 0 specific 2025 cites in
the clawRxiv content.

## 2026-06-08 — barrel through queue: reliability framing, future-work context mgmt, cite 2507.15779

(28) Added a "Design rationale" paragraph to Architecture: the reservoir adds memory to a proven
(pretrained) system rather than being the substrate, deliberately sidestepping RC's reservoir-
selection reliability problem (consistent with our N-seed selection-is-noise finding). (30) Future
work now flags context-growth risk + context-management importance in an always-alive agent, and the
DeepSeek-V4-Flash learned-context-management direction (compute-gated). (26) Acknowledged Koster &
Uchida 2025 "Reservoir Computing as a Language Model" (arXiv:2507.15779) in Related Work + References
(genericized for clawRxiv). Cleared resolved queue items; only the RC reference-diagrams item remains.

## 2026-06-08 — RC schematic diagram + removed ALL remaining file paths from the paper (Emma)

Authored an ORIGINAL reservoir-computing schematic (docs/diagram-reservoir-computing.svg: input ->
fixed W_in -> fixed random recurrent pool -> trained W_out -> output) based on the canonical RC
picture (not copied from the rights-restricted reference images); placed inline in Related Work +
site. Removed ALL remaining non-figure file/code paths from FINDINGS (data_lake/transcripts/,
GPT2_INTEGRATION_BLOCKER, reservoir.kv_evict..., kv_live x2) — reworded to plain prose. Final sweep:
only docs/*.png|svg figure refs remain.

## 2026-06-08 — FIX: crosspass.png was the wrong (gpt2-medium null) figure under a success caption

docs/crosspass.png had been clobbered by a gpt2-medium run (commit e711933) and was byte-identical to
docs/crosspass_gpt2-medium.png — both bars at chance (0.17) — while the paper caption claimed the
GPT-2-small headline result (reservoir 1.00 vs baseline 0.17). Regenerated docs/crosspass.png from the
real gpt2-small data (results/crosspass.json: stateful 1.00, baseline 0.17) via plot_recall_bars. The
figure now matches the caption. (Caught by Emma.)

## 2026-06-08 — redesigned RC diagram on the better reference (W_in/W_res/W_out labeled)

Emma pointed at reservoir.jpg + 41598_2020_78725_Fig1 as the good references; the second labels
W_in(fixed)/W_res(fixed)/W_out(trained) — the teaching point. Redesigned docs/diagram-reservoir-
computing.svg (hand-authored original SVG, not a converter) into the three-layer Input/Reservoir/
Readout style with those fixed-vs-trained weight labels + the y = sum w_i x_i readout, in the site
palette. Replaces my earlier generic version.

## 2026-06-08 — RC diagram labels fixed (verified render) + correct site URL

Redesigned RC SVG had glyph collisions (W^in superscripts overlapping text; subscript-i -> tofu
boxes). Switched to clean descriptive labels (input/recurrent/readout weights, fixed/fixed-random/
trained), verified by rendering locally with cairosvg — clean now. Also fixed the paper footer URL
emmaleonhart.github.io/reservoiragent -> reservoir.emmaleonhart.com (2 instances) per Emma; the site
already used the right domain.

## 2026-06-08 — fix figures cut off / overflowing the page (Emma)

The paper.tex \pandocbounded shim was a no-op passthrough, so wide figures (diagrams, bar charts)
ran off the page edge / looked oddly centered. Replaced it with pandoc's real default-template
\pandocbounded that scales an over-wide/over-tall image DOWN to fit the text block. Fixes the
Figure-2 cutoff and the general inline-figure overflow.

## 2026-06-08 — node-level transformer + RAN diagrams added to paper + site (Emma)

Emma loved the redesigned RC node-diagram and asked for a matching pair: a traditional
transformer diagram (the stateless baseline) and a visual RAN diagram in the same node style
showing "the orientation of the neurons." Authored two original SVGs in the site palette:
`docs/diagram-transformer.svg` (all-to-all self-attention -> FFN -> output; "no state survives
between forward passes") and `docs/diagram-ran.svg` (tokens + reservoir prefix nodes at a
mid-depth injection layer; fixed reservoir read via W_in / written via W_out; recurrent state
r(t)->r(t+1) carried across the context wipe). Verified both render cleanly via cairosvg
(fixed an injection-label clip). Placed as a baseline->RAN pair at the top of the Architecture
section in FINDINGS.md and as matching <figure> blocks in docs/index.html. Paper-PDF build green
(both SVGs auto-convert to PDF; the live \pandocbounded fix scales them to text width). Pushed to
origin/main (40c4d63..e31e685); ci/pages/paper-pdf all green.

Recovery note: an accidental `git pull --rebase origin feat` run while on the wrong branch
scrambled local branch pointers (no remote was ever touched — both pushes were correctly rejected).
Untangled via reflog: restored `main` to origin/main + the diagram commit (fast-forward) and `feat`
to its real tip; verified origin intact throughout.

## 2026-06-08 — #32 verdict: the "use-the-state" aux loss does NOT make battery retention hold

The counterfactual aux loss (#30) — penalize the model when a wiped-reservoir probe matches the
intact forward, so the objective rewards relying on carried state — was run on the content battery
for 4 epochs (3.1h, Qwen2.5-1.5B + 8192-node reservoir). It did not prevent the drift-to-stateless
collapse: mean reservoir lift over the wiped-state control decayed +0.302 (ep0, recall 0.44) ->
+0.094 (ep1) -> +0.000 (ep2) -> +0.000 (ep3). The collapse is not the stateful model degrading —
the stateless control rises to match it (0.000 -> 0.062 -> 0.083), i.e. the optimizer converges to a
current-pass solution that makes the carried state redundant even against a loss term built to forbid
exactly that. Folded the resolved negative into FINDINGS Limitations (battery bullet) and the site's
battery section; no retention claim. Stable retention is unsolved open work; the first-line
stabilizer fails. The clean retained result remains the strict-wipe cross-pass recall task
(0.83–1.00 vs 0.17 control), unchanged. Run log: results/_w32_aux.log.

## 2026-06-08 — README: document the Qwen-family scaling reproduction (code-release polish)

The README still framed cross-pass recall as "the headline GPT-2 result" and only gave a
GPT-2-default repro line, predating the central positive that recall *scales across the Qwen
family* once the reservoir is sized up (2048) and input scaling matched (0.1). Added the exact,
verified Qwen2.5-1.5B reproduction command (--model Qwen/Qwen2.5-1.5B-Instruct --mode kv
--n-reservoir 2048 --n-prefix 16 --input-scaling 0.1 --n-keys 6 --steps 800; from results/_w19/_w24
logs, stateful recall 0.83) and noted --n-reservoir / --input-scaling as the decisive transfer
levers. Verified every documented flag exists via `run.py crosspass --help` (exit 0). Addresses
Grok's "polish the code release with repro commands" suggestion; keeps README consistent with the
paper. (No queue item deleted — this came from the Grok-reception backlog in queue.md.)

## 2026-06-08 — #33 launched: harder battery retention (deny shortcut capacity + strong aux)

Follow-up to the #32 negative (user: "harder retention attempt", then "barrel through the queue").
#32 gave the stateless shortcut maximum adapter capacity (lora_target=all, r=8) and the shortcut won
even against the aux penalty. #33 instead DENIES the capacity — lora_target=attn, lora_r=4, the same
adapter regime in which the clean cross-pass recall task succeeds — AND cranks the counterfactual
"use-the-state" penalty (aux_weight=3.0, aux_margin=2.0). Content-only tasks (recall, accumulate,
sequence, deferred), recall regime (reservoir 2048/16, input_scaling 0.1, vocab 16), 4 epochs x 1500
(~3h), Qwen2.5-1.5B. Run results/_w33_retain_hard.log; out artifacts/w33-retain-hard; HF
reservoir-agent-qwen-content-retain-hard. Monitored per-epoch (lift vs stateless control). Verdict
folds into FINDINGS + site; retention claimed only if the lift survives the control across epochs.
Also set a one-shot cron (16:34 local today, +4h) for a comprehensive pre-arXiv citation audit.

## 2026-06-08 — post 2765 Weak Accept folded (reservoir-vs-adapter capacity isolation)

The resubmit (post 2765, #32 verdict folded) drew a second positive: Weak Accept (Gemini 3 Flash),
after the 2764 Accept. Four of five cons were already stated limitations; con #5 was sharper and new
— with a LoRA trained alongside the reservoir, the wiped-state control isolates the reservoir's
BEHAVIOURAL contribution but not a clean CAPACITY decomposition of fixed reservoir vs trained adapter.
Added a Limitations point saying exactly that, and noting the capacity-constrained retention probes
(low lora_r, attn-only adapter — incl. the running #33) attack that axis; a bits-per-component
decomposition is open. Mirrored to the site battery section. Response note in
paper/reviews/post2765_response_notes.md. Resubmitted (material review-driven addition).

## 2026-06-08 — second RC diagram: the echo state property (fading memory)

Added docs/diagram-echo-state.svg — a node-style figure pairing with the RC schematic: an input
impulse hits the fixed reservoir, and the influence-on-r(t) curve rises then decays toward zero over
subsequent passes (fading memory); caption gives the echo state property (influence of the initial
state and distant past vanishes; memory fading and finite, capacity grows with reservoir size).
Hand-authored, ASCII-safe labels, verified via cairosvg (fixed a clipped caption + a duplicate input
arrow). Placed in FINDINGS Related Work after the RC schematic and in the site's related-work
section. Clears the optional second-RC-diagram queue item.

## 2026-06-08 — Grok-reception block cleared (all actionable points addressed)

Collapsed the verbose Grok-reception checklist in queue.md to a concise resolved record. Every
actionable arXiv-polish suggestion is folded in: NeurIPS LaTeX build, vector figures with captions
(+ pandocbounded scale-to-fit), abstract leading with the positive result, exploratory material in
appendices, Contributions list, Limitations + Future Work (3B wall + compute), polished code release
(README repro commands + HF weights), and the safety/positioning angle. The only open items are
non-autonomous community actions (Google Scholar / arXiv endorsement, Discord outreach), flagged as
the user's to do. Full original Grok text remains in git history.

## 2026-06-08 — post 2766 Reject: fixed stale clawRxiv citation genericization + register pass

The resubmit (2766) drew a Reject from the same stochastic reviewer that Accepted 2764 / Weak-Accepted
2765. Two actionable cons: (1) it flagged the real 2025 preprints (Köster & Uchida; Echo State
Transformer) as hallucinated. The clawRxiv genericization was stale — the citations had gained inline
arXiv IDs, so literal keys like "Echo State Transformer (2025)" no longer matched the actual
"(2025, arXiv:...)" form and the dates leaked into the submitted copy. Rewrote it in
scripts/submit_clawrxiv_paper.py as newline/comma-tolerant regexes + a residual recent-year safety
net; verified on current FINDINGS that the transformed clawRxiv copy has ZERO residual
2025/arXiv/Köster/Uchida/Echo State Transformer/Echo Flow/FreezeTST tokens (the arXiv/PDF keep proper
dated cites). (3) Register pass on flagged phrases: gameable -> exploitable by a degenerate policy;
runs end-to-end on the real architecture -> executes the full pipeline on the target architecture;
laptop GPU -> 8 GB consumer GPU; byte-identical -> bitwise-identical. Cons 2/4/5/6 (toy task, battery
negatives, bespoke loop, safety-probe-not-alignment) are already stated limitations. Response note in
paper/reviews/post2766_response_notes.md. Resubmitted.

## 2026-06-08 — #33 verdict: capacity denial prevents the shortcut but the reservoir solution is unstable

The harder-retention run (deny adapter capacity: lora_r=4, attn-only — the recall-winning regime —
plus aux_weight=3.0/margin=2.0, content-only, 2048-node reservoir, 4 epochs, 2.1h) finished. Result is
a sharper diagnosis, not a retention win. The stateless control stayed pinned at 0.000 across ALL four
epochs (vs #32, where it rose 0.000->0.062->0.083): capacity denial DID eliminate the stateless
shortcut, so the reservoir is strictly necessary throughout. But the reservoir-driven solution is
unstable — lift +0.255 -> +0.339 (recall 1.00, peak) -> +0.062 -> +0.135 over epochs 0-3, oscillating,
peak not held. So the two failure modes are distinct: capacity denial trades shortcut-drift (#32) for
training-instability (#33); retention needs both fixed and stays open. Folded into FINDINGS
(battery limitation) + site battery section. Task #33 done. Run log results/_w33_retain_hard.log.

## 2026-06-08 — train_large: optional cosine LR decay (env-gated) + #34 staged

Diagnosed the #33 oscillation: train_large uses a FLAT LR (no scheduler), while train_battery.py
already found a flat lr "overshoots and degrades past its peak" and uses cosine. Added an env-gated
cosine decay to train_large (RESERVOIR_COSINE, default OFF so existing behaviour is unchanged; when on
in epoch-count mode it decays lr->0 over all steps). Compile-checked. Staged #34 (= #33 config +
RESERVOIR_COSINE=1) to test whether the instability is an LR-overshoot artifact (lift then holds =
first retention win) or intrinsic (still collapses = stronger negative). NOT launched yet — the user
asked to run the real-time agent app, which needs the GPU, so #34 is held until the GPU frees.

## 2026-06-08 — post 2768 Reject folded (citation con GONE; reservoir-vs-features rebuttal + capacity positioning)

2768 review (stochastic Reject) had NO citation-hallucination con — confirms the genericization fix
held. Folded two substantive cons: (3, sharpest) "is the reservoir a unique substrate or merely fixed
random features for the LoRA?" — strengthened the capacity-isolation limitation to rule it out: the
strict-wipe recall task removes the current-pass signal, so the secret survives only via carried state
(control 0.17 vs stateful 1.00), and #33's capacity-denied run (lora_r=4 attn) still hits recall 1.00
with control at 0.000 — carried state, not adapter capacity, does the work; open part is the
bits-per-component split. (6) capacity vs RMT/Memorizing Transformers — added a limitation noting the
gap is by-design (they train memory; the RAN is fixed-random, fading, ≤ N), contribution is the
fixed-substrate cross-pass question, not capacity competition. Mirrored to site. Cons 1/2/4/5 already
stated. Response note post2768_response_notes.md. Resubmitted.

## 2026-06-08 — pivot to arXiv prep: 6:30 PM experiment hard-stop + AI-use declaration

User directive: #34 is the last experiment; at 6:30 PM local all experimental work ends (one-shot cron
e0311518 enforces it). From now the focus is shipping to arXiv quickly: organize/verify the data, build
a replication script + downloadable package (zip), add the Declaration of AI use, make the citations
perfect (the 16:34 audit cron + repeated passes), and a general coherence/organization pass. Refilled
queue.md with this plan. Started: added the Declaration of AI use (arXiv requirement) to FINDINGS (->
PDF) and the site — honest disclosure that this is an AI-agent-driven project, with the key point that
every quantitative result is from executed code + measured behaviour against a stateless control, not
LLM-generated text, human-reviewed. User also flagged "a weird mistake with the data worth replacing" —
added a data-audit item to hunt for it.

## 2026-06-08 — replication package (downloadable zip) built + linked

Built replication/ : reproduce.py (orchestrator that runs the two headline cross-pass recall
experiments — GPT-2 1.00 vs ~0.17, Qwen2.5-1.5B 0.83-1.00 — and checks recall vs the paper's values),
a thorough README (setup via pip install -e .[models], exact commands, the decisive levers
n_reservoir/input_scaling, what is NOT claimed), and expected_results/ with the bundled logged JSONs
(results/ is gitignored, so the headline + capacity-sweep JSONs are copied in). pages.yml now zips
replication/ -> docs/reservoir-replication.zip on every build; added a "Replication package" download
button to the site. reproduce.py verified to parse + --help OK (CPU). This is the linkable replication
deliverable the user asked for.

## 2026-06-08 — data audit: fixed the 12-key capacity claim contradiction

Audited the headline figures against their source runs (crosspass ✓ GPT-2 1.00 vs 0.17; crossmodel ✓;
battery_lift ✓; capacity). Found the data issue the user flagged: the capacity figure's 12-key point
(0.167) sits BELOW the 24-key point (0.417) — non-monotonic. The site figure ALT-TEXT said this was
"due to undertraining", but that contradicts both the FINDINGS body and our own data: the w25 re-run
at 2000 steps (vs 800) gives the IDENTICAL 0.167 (w24-k12=0.167, w25-k12-st2000=0.167, both 2/12), so
it is NOT undertraining — it is a single-run convergence/seed artifact, stuck at both step budgets.
Fixed the alt-text to match the (already-correct) figcaption + body. The underlying capacity trend
(1.00@6 → ~0.42@24 → chance@48) is sound; only the 12-key point is a single-seed artifact, and a
cleaner figure would average several seeds at that point (a possible pre-6:30 experiment, but the GPU
is on #34 and the text already frames it correctly).

## 2026-06-08 — comprehensive pre-arXiv citation audit (4 parallel agents): no hallucinations, 8 fixes

Ran the scheduled citation audit via 4 parallel web-enabled verification agents (each fetched the
arXiv abstract pages / primary sources). Result: ALL 24 cited works are real — no hallucinations,
including every flagged 2025 preprint. Fixed 8 metadata issues in FINDINGS: added the missing
arXiv:2207.00729 (Merrill & Sabharwal, Parallelism Tradeoff); corrected Pérez et al. (1901.03429 is
"On the Turing Completeness of Modern Neural Network Architectures", ICLR 2019 — "Attention is
Turing-Complete" is the JMLR 2021 version; fixed Marinković spelling/order); renamed "FreezeTST" to
its real title "Frozen in Time" (Singh et al., 2025); added verified authors to Echo State Transformer
(Bendi-Ouis & Hinaut) and Echo Flow Networks (Liu & Xu); added 4 References entries (Shen et al.,
Bendi-Ouis & Hinaut, Liu & Xu, Singh et al.); cleaned the Merrill-Smith title; clarified
Siegelmann-Sontag (On the Computational Power of Neural Nets, JCSS 1995); softened the DeepSeek-V4
"CSA+HCA" label to the source-grounded "DeepSeek Sparse Attention (DSA, V3.2 -> V4)". DeepSeek-V4-Flash
+ DSA both confirmed real via live primary sources. Updated the clawRxiv genericization regexes for the
new author-prefixed citation forms + re-verified zero residue. Full per-citation verdicts in
paper/reviews/citation_audit_2026-06-08.md. No clawRxiv resubmit for citation fixes alone (per directive).

## 2026-06-08 — register pass (2769 con 5 + coherence): remove project-log meta-commentary

2769 Reject (stochastic, no citation con). Con 5 was actionable: project-log/meta register. Reframed
the Safety-section blockquote disclaimer into a plain scoping sentence; "What this does and does not
support" -> "Scope of this result"; "What this does not yet show, stated plainly" -> "Limits of this
probe"; removed an "earlier write-up read it as" self-reference and trimmed "stated plainly" tics.
Mirrored to the site. Cons 1-4/6 already-stated limitations. Part of the arXiv coherence/register pass.

## 2026-06-08 — coherence pass: removed remaining internal file/code paths from the paper

arXiv-prep coherence: removed the last leftover internal file/code paths from FINDINGS (the harness
appendix said "run_agent.bat launches ... (app/server) ... readout_scale", and the N-seed note cited
"RESERVOIR_AGENTS.md"). Reworded to describe the desktop app / runtime gain / seed population without
literal paths. Verified no remaining literature/REVIEW.md, scripts/*.py, src/*, or *.bat references in
the paper. (The replication package + the site's repro hints legitimately keep commands; the arXiv
paper itself is now path-clean.)

## 2026-06-08 — queue hygiene: pruned completed arXiv-prep items

Pruned the queue to reflect completed arXiv-prep work (AI-use declaration, replication package,
citation audit, data audit, register + file-path cleanup, abstract/contributions consistency check —
all done). Remaining: fold #34 at DONE, a light final coherence skim, final arXiv packaging, and two
open user decisions (12-key re-run? branch/installer cleanup?). Also verified abstract <->
contributions <-> results are mutually consistent (no stale claims), no edit needed.

## 2026-06-08 — #34 RETENTION WIN: cosine LR + capacity denial retains battery recall

#34 (= #33 config + RESERVOIR_COSINE=1) completed: the reservoir-driven battery recall is RETAINED.
Trajectory: mean lift +0.089 -> +0.089 -> +0.130 -> +0.292; recall 0.08 -> 0.19 -> 0.35 -> 1.00;
stateless control pinned at 0.000 all 4 epochs; monotonic climb, NO collapse (unlike #33's flat-LR
spike-collapse). This closes the #32->#33->#34 arc: #32 aux-loss alone fails (control rises to match);
#33 capacity denial (lora_r=4 attn) keeps control at 0 but flat LR makes the solution unstable; #34
adds cosine LR decay -> both failure modes fixed -> recall retains at 1.00 against a 0.000 control.
The recipe: deny the stateless shortcut its adapter capacity AND decay the LR. First stable retention
result. Caveats (kept in the paper): single run (seed-robustness untested, no more experiments this
session per the 6:30 cutoff); RECALL retains at 1.00 while the harder content tasks
(accumulate/sequence/deferred) stay low (~0.02-0.12) -> retention of recall, not the whole battery.
Folded into the abstract (4th result), the battery limitation bullet (now a resolved positive), and
the site battery section. Material positive -> resubmitting to clawRxiv. Run results/_w34_cosine.log.

## 2026-06-08 18:30 — experimental work CLOSED (6:30 PM hard-stop)

The scheduled 6:30 PM hard-stop fired. Confirmed no train_large/experiment process running; GPU idle.
#34 was the last experiment (finished, retention win, folded). Session is now arXiv-prep ONLY — no new
experiments/training/sweeps regardless of queue/todo. Updated the queue banner to reflect closure.

## 2026-06-08 — final arXiv packaging verified (tarball self-contained + current)

arXiv-prep packaging check: downloaded the live reservoir-arxiv-source.tar.gz and verified it is the
current build (17 retain/cosine/monotonic mentions -> includes the #34 retention win), self-contained
(paper.tex + paper.tex.body + neurips_2026.sty + figures/ with SVGs converted to PDF), and that all 14
figures referenced by paper.tex.body are present in the tarball (no missing \includegraphics). Named
build title/author set (Emma Leonhart). PDF builds green. Noted a minor harmless point: the tarball
ships all ~50 docs figures while the paper uses 14 (~1.3 MB bloat, not a blocker). arXiv account +
endorsement + category/license is the user's step.

## 2026-06-10 — User-requested pre-arXiv audit: training/loss code review + fresh citation pass

User asked for a final look before arXiv. (a) **Training + loss code review** — read
`episode.py` (episode_loss/episode_eval), `battery.py`, `kv_live.py`, `crosspass.py`,
`train_battery.py`, `train_large.py`, `controlled.py`, `tasks.py`, `rnn_baseline.py`,
`echo_state.py`, `_arch.py`: **no correctness bugs found**. Loss masking/teacher-forcing,
gate BCE vs content CE split, stateless controls, cosine LR wiring, ANOVA, and the GRU
baseline all check out; train/eval are consistent (incl. the per-token state-tick during
teacher forcing, identical in both paths). All design caveats found are ones the paper
already documents (closed key set as mechanism-isolating ablation; the emit/silence
metric bug documented as found-and-fixed; #34 single-run caveat). (b) **Fresh citation
audit** (web agent, all 24 refs): no hallucinated citations; all arXiv IDs/authors/venues
real. Three fixes applied to FINDINGS References: Titans year 2024→2025 (arXiv 2501.x),
added ECAI venue to Singh et al. 2025, added ICLR venue to S4. (Site has no references
section, so no docs/index.html mirror needed.) (c) Verdict: **arXiv-ready**. Remaining
(d): deliver the rebuilt arXiv source tarball to the user's Downloads.

## 2026-06-10 — Pre-arXiv audit (d): tarball delivered

CI rebuilt the site green after the citation fixes; downloaded the fresh
`reservoir-arxiv-source.tar.gz` (1.29 MB) into the user's Downloads folder and
verified by extraction that all three reference fixes (Titans 2025, ECAI, ICLR)
are in `paper.tex.body`. Audit complete; paper is arXiv-ready — account /
endorsement / category+license selection remains the user's step.

## 2026-06-10 — Em-dash purge + PDF figure-overflow fix (user requests)

(1) Removed every em-dash from the paper and its surfaces, rewriting each with
contextually chosen punctuation (two parallel agents; meaning and numbers
unchanged): FINDINGS.md (249 occurrences), docs/index.html (134, both literal
and &mdash; entities), README.md (19), paper/paper.tex (2, comments). Verified
zero remain across all four files; en-dashes in numeric ranges kept.
(2) Fixed the cut-off figures in the built PDF: the CI pandoc emits bare
\includegraphics with no size bound (the \pandocbounded shim is never invoked),
so wide PNGs (crossmodel_recall, h3_memory, battery figures) ran off the right
page edge. Added graphicx Gin defaults (width=\maxwidth, height=\maxheight,
keepaspectratio — pandoc's own standalone-template approach) to paper.tex so
every figure is bounded to the text block regardless of pandoc version.

## 2026-06-10 — Em-dashes purged from figures too (SVG text + re-rendered plot PNGs)

The prose purge left em-dashes baked into figure artwork. Fixed the drawn text in
the three SVG diagrams that had them (architecture, transformer, runtime) and
re-rendered the four plot PNGs whose titles/labels contained em-dashes
(crosspass.png, sweep_real.png, sweep_scaling.png, blank_cycle_kv.png) from
their stored results JSONs via the existing plot functions: identical data, no
recomputation, no training (experiments stay closed). Title strings in
scripts/run.py and labels in blank_cycle.py fixed at the source so future
renders stay clean. All other paper/site figures verified em-dash-free.
Note: tests/test_torch_inject.py::test_finetune_pipeline_reduces_loss failed
once in the full local suite and passes in isolation (stochastic few-step
finetune assertion); unrelated to this change set, which is strings-only.

## 2026-06-11 — Em-dash purge complete end-to-end; fixed tarball re-delivered

A PDF text-layer scan caught 4 final em-dashes hiding as XML entities (&#8212;)
in diagram-ran.svg and diagram-echo-state.svg, which the literal-character grep
had missed; fixed. Final verification on the live artifacts: the rebuilt
report.pdf contains ZERO em-dashes across prose and vector-figure text, all
figures are bounded to the text block (the Gin fix), and the rebuilt
reservoir-arxiv-source.tar.gz (1.29 MB, em-dash-free, figure fix included) is
re-delivered to the user's Downloads. The pre-arXiv audit thread is closed.

## 2026-06-13 — External-review pass, block A (blockers): figures, recall disambiguation, equations, typo, citations

Folded the first batch of the seven-AI-review editing pass (digest in
data_lake/external_reviews_2026-06-13.md). Blocker-level items:
- **Self-contained figures:** removed the "All figures referenced below are in the
  accompanying report: reservoir.emmaleonhart.com" line from FINDINGS Results so an
  arXiv reader is never sent off-site for figures (the figures are embedded inline in
  the markdown→PDF build; the footer site pointer stays as supplementary).
- **Recall disambiguation** (flagged by Gemini, DeepSeek, Perplexity, ChatGPT): the
  abstract result 3 + the contribution bullet now state explicitly that the *secret-word
  recall probe* scales to Qwen-1.5B (0.83–1.00 vs 0.17 control), and that the eight-task
  battery's *symbolic content* recall is a separate, harder measurement that stays at the
  floor at 1.5B except under the retention recipe. Verified against results/ that this is
  a disambiguation, not a false scaling claim — the probe scaling is real. Mirrored into
  docs/index.html.
- **Equation harmonization:** introduced the leaky-integrator update r(t)=(1−a)·r(t−1)+
  a·tanh(...) with leak rate a defined at first use in §4, so §4 and §6 agree (was: plain
  tanh form in §4, undefined leak rate in §6).
- **Typo:** np8 → n_prefix=8 in the Qwen ablation table.
- **BRT quote — verified, no edit:** the Block-Recurrent Transformers paper itself says
  "the model learns to ignore them" / "learns to completely ignore the recurrent state";
  our "documents the same failure" framing is faithful.
- **2025 citations — verified, no edit:** 2507.02917 (Echo State Transformer) and
  2509.24122 (Echo Flow Networks) are still preprints; 2508.18130 (Frozen in Time) is
  ECAI 2025 and already carries the ECAI venue in the references.

Zero em-dashes re-verified end-to-end after the edits (caught and fixed two I introduced).
