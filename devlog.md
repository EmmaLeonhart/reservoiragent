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
