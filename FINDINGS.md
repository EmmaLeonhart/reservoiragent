# Reservoir Attention Network (RAN) — Findings

**Architecture:** Reservoir Attention Network (RAN)  
**Implementation:** Reservoir Agent (GPT-2, Hermes 3B)  
**Status: feasibility phase complete.** This document is the project's write-up.
The results below confirm the core architecture and dynamics, demonstrate
cross-context recall on GPT-2, and identify the optimization frontier on Hermes.
This is a **feasibility + dynamics study, not an agentic-capability demonstration**:
the tasks are deliberately minimal probes, each chosen to isolate one mechanism, and
the broader agentic vision is named throughout as future, compute-gated work.

## Question

Can a fixed, randomly-initialized reservoir injected into a pretrained transformer's
mid-layer attention give the model genuine state **between** forward passes — a real
time axis — without degrading its base capabilities, and what reservoir-dynamics
regime (spectral radius, reservoir size, injection depth) makes that injected state
usable signal rather than noise?

The Reservoir Attention Network (RAN) architecture introduces a fixed-random
recurrent substrate into the transformer's attention mechanism. We refer to a
specific instantiation of this architecture as a **Reservoir Agent**.

This session scopes the question as a **feasibility + dynamics study** at small scale
(GPT-2-scale base, single machine). The full vision — forking an agent harness into an
always-alive runtime and N-seed LoRA selection at agent scale — is the long-horizon
target (see `todo.md`).

## Scope, and what this study does and does not claim

This revision sharpens the scope in response to peer review. To be explicit about the
boundary of the claims:

- **The tasks are minimal mechanism-isolating probes, not agentic demonstrations.**
  Secret-word recall and the trigger-based silence policy are intentionally the
  *simplest* tasks that a stateless model **structurally cannot** do — their job is to
  isolate one variable (does carried state become usable signal, and under which
  injection design), not to exhibit organism-like reasoning. We make **no** claim of
  complex agentic behaviour at this scale; that is named as future work, not shown here.
- **The complexity-theory argument is motivation, not a result.** The TC⁰ / FO(M)
  framing explains *why* cross-pass state is the interesting lever; we state plainly that
  there is **no proof** a finite-precision reservoir lifts the per-pass bound, and we
  treat it as the project's central open theoretical question, not an established finding.
- **The Hermes-3B negative and the KV-append integration blocker are limitations, stated
  as such.** The cross-pass recall result is GPT-2-only; on Hermes-3B it is a
  well-diagnosed, verified-wired non-convergence (a bootstrapping/scale wall, plausibly
  signal dilution through depth), and the most effective injection variant (KV-append)
  has a documented HuggingFace-integration blocker that currently limits its
  reproducibility. Neither is hidden; both bound the contribution honestly.
- **The contribution is the injection-design finding.** What this study *does*
  establish, decisively and reproducibly on GPT-2, is that **how** the reservoir is
  injected is the deciding factor: additive injection is ignored (chance recall), while
  content-addressable KV-prefix injection gives 100% cross-context recall. That negative-
  then-positive result is the load-bearing contribution.

## Architecture

Every forward pass is one reservoir tick. At a mid-depth injection layer Lk, attention
runs jointly over the token hidden states and a set of reservoir nodes (extra
keys/values). The reservoir reads the layer's attention output through a fixed random
projection W_in and writes its state back through a learned readout W_out — both at the
same layer, every pass — so the reservoir state accumulates a history of the model's
own attention dynamics across passes. The reservoir update is

    r(t) = tanh( W_r · r(t−1) + W_in · x(t) )

with W_r a fixed random sparse matrix scaled to a target spectral radius, W_in fixed
random, and W_out (plus light upper-layer LoRA) the only trained parameters. The lower
layers are frozen. Because the reservoir state is decoupled from the context window, it
persists across genuinely independent forward passes, including unprompted ticks.

## Grounding in the literature

The fixed-reservoir / trained-readout core is a faithful instantiation of classical
reservoir computing (Jaeger's echo state networks; Maass's liquid state machines). The
motivation is made precise by the expressivity literature: a finite-precision
transformer is bounded to TC⁰ / FO(M) **per forward pass** (Merrill & Sabharwal; Hahn),
while state carried **across** passes is the documented lever past that ceiling — though
the known Turing-completeness results require arbitrary precision, so whether a
finite-precision reservoir lifts the bound is posed as an open question, not asserted.
Crucially, every prior recurrence-augmented transformer (Transformer-XL, RMT,
Block-Recurrent, Mamba, Titans, …) uses *trained* recurrence carrying state *within* a
sequence; none uses a *fixed-random* reservoir with state across *independent* passes.
The full survey with citations is in [`literature/REVIEW.md`](literature/REVIEW.md).

## Theory (formal claims, scoped)

Three claims, stated at the level of *kind* of capability, not level of capability.
Grounding and citations are in [`literature/REVIEW.md`](literature/REVIEW.md).

**1 · A genuine time dimension.** A standard transformer represents time as token
*position* — an index into a sequence, not a dimension the model evolves along. With
the reservoir, the state r(t) evolves continuously across forward passes:
r(t) = (1−a)·r(t−1) + a·tanh(W_r·r(t−1) + W_in·x(t)), so r at pass N is causally
downstream of every pass since t=0. This is not positional encoding and not context
length — both reset or slide with the input. The reservoir state is decoupled from the
context window (it survives context truncation), which is precisely what a "time axis"
means here: an endogenous variable the model accumulates along, independent of the
input sequence.

**2 · The expressivity gap, and where the reservoir sits in it (with a caveat).**
A fixed-depth, finite-precision transformer is, *per forward pass*, confined to a low
complexity class: saturated/log-precision transformers ⊆ TC⁰ and are exactly captured
by first-order logic with majority quantifiers, FO(M) (Merrill & Sabharwal 2022/2023),
and fixed-size self-attention cannot model unbounded hierarchical structure without
growing depth (Hahn 2020). The documented lever out of that ceiling is **state carried
across steps**: the TC⁰/FO(M) upper-bound proof explicitly breaks once generated output
is fed back into the next step, and finite recurrent nets are Turing-complete in
principle (Siegelmann & Sontag 1992/1995). The reservoir is exactly such a recurrent
system, so the Reservoir Agent has the *structural ingredient* a stateless pass lacks.
**The caveat we do not paper over:** the transformer Turing-completeness results
(Pérez et al. 2019) require *arbitrary precision* — the dense representations act as
unbounded memory. The Reservoir Agent runs at finite precision, and **no result here
or in the literature proves that a finite-precision continuous reservoir state lifts
the per-pass TC⁰/FO(M) bound.** We pose this as the project's central open theoretical
question, not as an established result. The honest claim is narrow and true: the
architecture has a *capacity for endogenous cross-pass state evolution that a single
finite-precision transformer pass structurally lacks.*

**3 · The organism analogy (one paragraph, bounded).** The reservoir introduces
endogenous state that evolves independently of external input — a property shared with
living organisms and absent from stateless transformers. No claim about general
intelligence is made or implied. The claim is structural: this architecture has a
capacity for organism-like state evolution, and that capacity may be a precondition for
certain classes of genuinely agentic behaviour (noticing an unresolved thread,
estimating elapsed time, self-initiating) that are inaccessible to a stateless model
regardless of its capability level.

## Method (this session)

1. **Reservoir core.** A tested echo-state reservoir with spectral-radius control and
   dynamics observability (variance, saturation fraction, effective rank, trajectory
   distinguishability).
2. **Dynamics characterization.** Drive the reservoir across a grid of spectral radius
   and size; locate the regime where the state is non-saturating, non-exploding, and
   carries distinguishable trajectories across input histories (H2), and test whether
   the optimum sits at the classical edge-of-chaos prior (which the literature reports
   is disputed).
3. **Model surgery (H1).** Inject the reservoir into a mid layer of GPT-2-small and
   verify that, with the readout zeroed, the base model's outputs are unchanged —
   i.e. the architecture degrades gracefully to vanilla behaviour.

## Results

### H1 — the reservoir injects without breaking the base model

Hooking a mid-depth block of pretrained GPT-2 so the block's hidden states drive the
reservoir and its state is written back into the residual stream (`h' = h + W_out·r(t)`):

- **Non-destruction holds.** With the readout `W_out = 0`, the injected model's
  next-token logits are *identical* to vanilla GPT-2 (`allclose`, atol 1e-5) — the
  architecture degrades gracefully to the base model.
- **The injection is live.** A nonzero `W_out` changes the logits, and the reservoir
  state after two forward passes differs from after one — a genuine cross-pass time
  axis. (`tests/test_inject.py`.)

### H3 — a trained readout extracts history a stateless model cannot

On the delay-memory task (drive the reservoir with i.i.d. input u(t); train a linear
ridge readout to reproduce u(t−τ)), the readout on the **reservoir state** recovers the
input from **~18 steps back at R² > 0.5** and ~12 steps back at R² ≈ 1, with a total
linear memory capacity of **17.4** (Σ R² over τ ≥ 1). The **stateless baseline** —
the same readout trained on the *current* input u(t) — scores **exactly 0** at every
delay ≥ 1, because i.i.d. inputs carry no information about their own past. So the
information needed to answer is provably *in the carried state, not the input*: a light
trained readout makes the reservoir's history usable, and a stateless model structurally
cannot match it. (Figure: `docs/h3_memory.png`; `scripts/run.py h3`.) This is the H3
mechanism on a clean synthetic task; doing it on a *semantic* agent task (unresolved
thread, elapsed time) is future work that needs the readout trained through the LM.

### N-seed selection — the mechanism works; the cheap pre-selection proxy does not

Running the plan's N-seed selection at small scale (train each of 12 fixed reservoir
seeds' readout on the delay-memory task, rank by memory capacity, keep the best): the
seeds genuinely differ — memory capacity ranges **17.4 to 20.7** (~19% spread) — so the
selection is worth doing. But the open "seed pre-selection proxy" question (can a cheap
*untrained* dynamics metric predict which seed trains best, to skip training?) gets a
clean **negative answer for this proxy**: the untrained participation ratio has **no
rank correlation** with trained memory capacity (**Spearman ρ = 0.08, p = 0.80**, n=12).
So seeds cannot be pre-filtered by participation ratio — the N-seed *training* does real
work this dynamics proxy can't shortcut. (Figure: `docs/nseed_select.png`;
`scripts/run.py nseed-select`. Other proxies remain untested.) **The cost implication,
stated plainly (per review):** because this proxy fails, selecting a good fixed reservoir
currently requires training each seed's readout — i.e. genuine trial-and-error, not a
cheap pre-filter. Finding an untrained proxy that *does* correlate is open work; until
then the selection cost scales with the number of seeds tried.

**Per-seed recall spreads widely — but at this budget it is dominated by training noise,
not cleanly by reservoir quality (a correction).** Training a population of fixed reservoir
seeds end-to-end on the cross-pass task (GPT-2, 250 steps each) gives recall from **1.00 to
chance (0.17)** across seeds (populations of 12 and 20 are published at
`EmmaLeonhart/reservoir-agent-gpt2-batch-n12` and `-n20`). It is tempting to read that
spread as reservoir *quality* — but the two runs share seed indices, which gives a natural
replication, and it does **not** hold up: the **same seed (identical fixed reservoir, same
setting) lands at very different recall across the two runs** — e.g. seed 0 at 0.33 vs 1.00,
seed 1 at 1.00 vs 0.33 — with **mean |Δrecall| ≈ 0.47** over the 12 shared seeds, nearly as
large as the full spread. So at 250 steps the outcome is **run-to-run noise-dominated**
(CUDA non-determinism + an under-trained regime + the trainable readout/LoRA init not being
seeded by the reservoir seed), and a single run per seed cannot separate reservoir quality
from training noise. Consistently, **no untrained reservoir metric predicts recall**:
realized ρ, mean/std |eigenvalue|, Henrici non-normality, participation ratio, and
delay-memory capacity all give |Spearman ρ| < 0.36 (p > 0.14, n=20) against the recall
labels (`scripts/run.py`/`reservoir.seed_metrics.correlate_seed_metrics`) — but with
noise-dominated labels this cannot distinguish "no cheap predictor" from "labels too noisy
to correlate". **What this does and does not support:** it supports *keeping the whole
population* (cheap metrics don't let you pre-filter, so you train and measure) and the H2
fact that reservoirs scaled to a fixed ρ have near-identical bulk dynamics; it does **not**
yet demonstrate that some fixed reservoirs are durably better than others on this task.
Establishing that needs a **controlled** experiment: seed the trainable init too, enable
deterministic CUDA, and **average several runs per seed**. (Figure:
`docs/nseed_trained_spread.png` shows one run's spread.)

**The controlled experiment — run, and it confirms: at 250 steps selection is noise, not
signal.** We then ran exactly that experiment (`scripts/run.py controlled`;
`docs/controlled.png`). Root cause of the noise was first removed: `kv_live` had a `train_seed`
parameter that was never used, so the trainable `W_res` + LoRA init was uncontrolled; it now
seeds the init, and a `set_deterministic` helper (RNGs + `CUBLAS_WORKSPACE_CONFIG` + cudnn
flags + the deterministic math SDP kernel) makes two runs of the same reservoir with the same
`train_seed` **bit-identical** (verified on CPU and CUDA). With that, we trained **6 reservoir
seeds × 4 runs** (the four runs vary only by `train_seed`) and ran a one-way **ANOVA** over
recall grouped by reservoir seed. Per-seed mean recall ranged 0.33–0.75, but the **within-seed
spread is as wide as the between-seed spread** (e.g. seed 0 spans 0.33→1.00 across inits): **F =
1.30 (df 5, 18), p = 0.31** — the between-seed (reservoir) variation does **not** exceed the
within-seed (trainable-init) noise. So at 250 steps, **reservoir "selection" is not a real
signal** — which fixed reservoir you drew matters less than which trainable init you happened to
get. This turns the earlier *suspected* artifact into a *controlled* negative result. It does
not rule out selection mattering with far more training (where init noise should shrink) — that
larger-budget run is the natural follow-up — but at this budget the honest verdict is: train and
select over *runs*, not over reservoir seeds.

### H2 — the reservoir-dynamics regime

Sweeping spectral radius ρ ∈ [0.1, 2.0] (figures: `docs/sweep_synthetic.png`,
`docs/sweep_real.png`):

- **The echo state property breaks sharply at ρ ≈ 1.** Using an autonomous
  (zero-input) probe — two random initial states under no input — the reservoir forgets
  where it started (init-forgetting ≈ 0) for ρ < 1 and abruptly retains it for ρ > 1.
  This edge-of-chaos boundary appears on *both* synthetic input and **real GPT-2
  mid-layer activations** (on real data: 0.000 for ρ ≤ 0.9 → 0.10 at ρ = 1 → ~0.95
  above). The classical ρ ≈ 1 boundary survives the move to transformer-scale input.
- **The input regime decides whether ρ matters.** Under unit-scale input *drive* the
  reservoir forgets its initial state across *all* ρ (strong input enforces the ESP),
  so the ρ ≈ 1 boundary is the regime that governs **unprompted, input-free passes** —
  exactly where the agent would run on reservoir state alone.
- **Real activations over-drive the reservoir.** Compared with synthetic noise, real
  GPT-2 activations push the reservoir to much higher saturation (~0.86 of units pinned
  near ±1, vs < 0.15) and higher effective dimensionality (participation ratio ≈ 0.41·K
  vs ~0.05·K). So a unit-input-scaled reservoir is *over-saturated* by real attention
  activations: the input scaling has to be tuned down for injection at transformer
  scale — the precise concern the plan anticipated ("feeding a large attention tensor
  may require different scaling").
- **Tuning the input scaling fixes it (figure: `docs/sweep_scaling.png`).** Sweeping the
  input scaling at ρ = 0.95, saturation is a clean sigmoid in the scaling: it crosses
  0.5 at scaling ≈ 0.24 and is near zero below ≈ 0.05, while input separation and
  effective dimensionality stay high. There is a sweet spot around **input scaling
  0.08–0.24** where the reservoir is *not* over-saturated (saturation 0.08–0.49) yet
  still strongly responsive (separation 1.03–1.26, PR ≈ 0.39·K). So real attention
  activations should be fed at roughly **¼–⅒ of unit scale**, not 1.0 — a concrete
  injection setting this study contributes.

## Ambitious reach (proof-of-concept)

Pushed past the feasibility scope to see how far local compute reaches, reported as
measured:

- **The time axis is real and behavioural.** Running the *same* prompt after different
  prior history, with the reservoir state carried across the (otherwise independent)
  forward passes and a small random readout, shifts the next-token logits by an L2
  distance of ≈ 22 (`scripts/run.py alive`, GPT-2). The same input produces a different
  output distribution depending on what the model processed before — something a
  stateless transformer structurally cannot do.
- **The seed-selection mechanism works; the pre-training signal is weak.** A dynamics
  pre-selection proxy ranks N fixed-random reservoir seeds by responsiveness,
  dimensionality, and (penalised) saturation on real GPT-2 activations, before any
  training (`scripts/run.py nseed`). Across 8 seeds at ρ = 0.95 the spread is small
  (~0.02), i.e. *untrained* dynamics vary only modestly between seeds — so the real
  selection signal the plan relies on most likely emerges only after fine-tuning. The
  mechanism is in place; the verdict on its usefulness is compute-gated.

**Named plainly as not done (compute-gated), not papered over:**

- The full **N-seed LoRA fine-tuning + benchmark selection** — there is no training
  pipeline or benchmark suite here; only the *dynamics* proxy was run.
- A productionized **always-alive runtime** (pass scheduler, idle timer, output
  confidence gate) — only the two-pass state-carry was demonstrated.
- The **KV-append** injection (reservoir nodes as extra keys/values the upper layers
  attend to) and **agent-scale (Hermes)** models — beyond local compute this session.

## The always-alive runtime (harness)

Built and exercised the stateful-agent loop on the *untrained* injected model — the
substrate fine-tuning will later plug into (`src/reservoir/runtime.py`,
`scripts/run.py agent`). It has the four pieces the architecture requires:

- a **context buffer** owned by the runtime, never wiped between passes;
- a **reservoir state store** that persists across passes and checkpoints/restores to
  disk (round-trip tested);
- a **pass scheduler** with both *prompted* passes (new input) and *unprompted* passes
  (idle ticks that run over context + reservoir only) — and a unit test confirms an
  unprompted pass updates the reservoir state with **no new input**;
- an **output confidence gate** (normalized top-k logit entropy) deciding emit vs.
  silence.

A scripted session runs end-to-end: across five interleaved prompted/unprompted passes
the reservoir state |r| evolves continuously (state carried, including through the
idle ticks). **Named plainly:** on the untrained model the gate keys off the *base
model's* next-token entropy, so its emit/silence decisions and the generated text
(GPT-2 babble) are not yet meaningful — the harness is the mechanism, and a meaningful
self-initiation policy needs the trained readout/LoRA. The point of this step is that
the whole loop is now testable before spending compute on training.

## Compute-gated: a real LoRA fine-tune on GPU

The culminating run, on local CUDA (RTX 4070): a genuine **LoRA + W_out fine-tune** of
GPT-2 with the *differentiable* reservoir injection (`src/reservoir/torch_inject.py`;
`scripts/run.py finetune`). Across **3 reservoir seeds × 60 steps**, training loss falls
decisively (≈ **6.3 → 0.85–1.1**) with **491,520 trainable parameters** (LoRA on the
attention projections + the reservoir readout W_out), and the best seed is selected by
trained loss. So the full pipeline — inject, freeze the backbone, train W_out + LoRA,
select across seeds — **runs end-to-end on the real architecture**, on the GPU. With
W_out zero-initialised the fine-tune starts exactly at the base model (H1 preserved).

**The honest boundary, named plainly:** the injection hook fires *once per forward pass*
(a transformer processes the whole sequence through each layer once), so this
single-forward fine-tune exercises the *training machinery on the real model*, not the
reservoir's distinctive **cross-pass** value. Exercising that requires the multi-pass
differentiable harness — backprop through passes on a reservoir-requiring (cross-context)
task — which is the next compute step, now unblocked by everything above (working
injection, the always-alive harness, the trained readout, and this fine-tune pipeline).

## Porting to the real target: Hermes (Phase H)

The GPT-2 work validated the mechanisms; this phase moves to the smallest Hermes —
**NousResearch/Hermes-3-Llama-3.2-3B** (Llama-3.2, the architecture the project actually
wants, already agent-fine-tuned).

- **(A) Injection generalized to the Llama architecture.** The injection was GPT-2-only
  (`transformer.h`); `src/reservoir/_arch.py` now locates decoder blocks across families
  (`model.model.layers` for Llama), and H1 is verified on a tiny Llama as well as GPT-2.
- **(B) Hermes 3B loads and H1 holds, on the laptop GPU.** Loaded in 4-bit (bitsandbytes
  nf4) with the reservoir injected at layer 14 of 28 (d_model 3072): with the readout
  zeroed, the injected model's logits are **byte-identical** to the un-injected Hermes
  (`max|diff| = 0.00`), at a peak of **2.35 GB VRAM** — leaving ample room for LoRA +
  training on the RTX 4070. So the architecture transplant is non-destructive on the real
  model. (`scripts/hermes_h1.py`; `results/hermes_h1.json`.)

## C: cross-pass recall — the injection design decides everything

The load-bearing experiment, and the central result. The task is one a stateless model
**structurally cannot** do: show a secret word on pass 1, **wipe the context**, recall it
on pass 2 from the carried reservoir state alone (`src/reservoir/crosspass.py`;
`scripts/run.py crosspass`). The multi-pass differentiable harness backprops through both
passes, training the injection (+ LoRA), and is compared against a **stateless baseline**
(the reservoir is reset between the two passes, destroying the carried state).

**The result depends sharply on *how* the reservoir is injected — and that is the
finding.**

- **Additive readout injection → fails (the reservoir is ignored).** With the reservoir
  written into the residual stream as one additive bias vector (`torch_inject.py`),
  across mean/last-token drive and mid/last-layer injection up to 500 steps, the stateful
  model and the stateless baseline reach the **same chance accuracy (0.17 = 1/6)**. The
  model learns the marginal, not the recall — the **Block-Recurrent "learns to ignore the
  recurrent state" failure mode, reproduced.** A single pooled additive bias cannot carry
  *which specific word* appeared.

- **Content-addressable (KV-append) injection → works, decisively.** When instead the
  reservoir state is projected into prefix pseudo-tokens the model can **attend** to
  (`kv_live.py`, `--mode kv`), the stateful model reaches **100% cross-context recall
  (loss → 0.02)** while the stateless baseline stays at **chance (0.17)**. The carried
  reservoir state, made attendable, lets the model recall content that exists *only* in
  the reservoir — something the stateless baseline provably cannot do. (Figure:
  `docs/crosspass.png`.)

**This is the project's core claim, demonstrated:** the Reservoir Agent's statefulness
*does the desired thing* — it carries information across independent forward passes and
the model uses it — **provided the reservoir is injected content-addressably (attended
to), not as an additive bias.** The negative-then-positive arc is the contribution: it
isolates the injection design as the decisive factor, ruling out the naive variant and
validating the attention-based one. (Demonstrated on GPT-2; the same `kv_live` path is
architecture-agnostic and runs on Hermes via the generalized injection.)

**Transfer to Hermes 3B — not yet, and well diagnosed (honest).** The same
content-addressable experiment was run on the real target, Hermes-3-Llama-3.2-3B, across
**four** attempts: 4-bit at input scaling 0.5 (300 steps), 4-bit at 0.1 (600 steps),
**bf16 (non-4-bit) at 0.1 with a higher LR 3e-3** (600 steps), and a dedicated
**many-more-steps run: 4-bit, 2000 steps** (≈6.7× the first attempt). **All four came back
at chance (0.17), stateful ≈ baseline,** with the training loss consistently failing to
converge (plateau ≈ 2.5–2.9, vs GPT-2's 0.02; the 2000-step run reached 2.49, no better
than 300 steps). The consistent plateau **across both 4-bit and bf16, and now across a
6.7× step increase,** shows the wall is **neither quantization nor under-training** — more
steps alone does not break it, so the remaining routes are structural (a curriculum that
starts with the key in-context and anneals it out, a stronger multi-layer prefix coupling,
or unfreezing more of the model), which is substantial work, not a hyperparameter.

A focused gradient diagnostic on the Llama path **rules out a bug**: the reservoir state
*does* update each pass (norm 0.14 after pass 1, from 0) and gradients *do* flow to both
the readout `W_res` (‖∇‖ ≈ 0.016) and the LoRA adapters (Σ|∇| ≈ 3.0). So the injection is
correctly wired on Hermes — this is a genuine **optimization / scale difficulty**, not a
defect: the prefix's signal, diluted through 28 layers and competing with a 3B
instruction-tuned model's strong priors, does not *bootstrap* into use within the
attempted budget, whereas shallow GPT-2 bootstrapped easily. The **"far more steps" route
has now been tested and ruled out** (a 2000-step 4-bit run, ≈6.7×, still chance / loss 2.49);
the remaining plausible routes (left open, not faked) are structural: a curriculum (start
with the key in-context, anneal it out) / a stronger multi-layer prefix coupling / unfreezing
more of the model. **The result holds decisively on GPT-2; on Hermes the mechanism is
verified-wired but the recall has not yet been trained to converge, and it is not a
step-count problem.** (`results/crosspass_hermes-3-llama-3-2-3b.json`,
`docs/crosspass_hermes-3-llama-3-2-3b.png`.)

**The transfer wall starts well below 3B.** A 10-seed **GPT-2-medium (355M)** batch and a
follow-up single-seed probe at lower input scaling (0.1, 1000 steps) both stayed at
**chance (0.17)** with loss plateauing ~2.1 — the same "learns the marginal, ignores the
prefix" failure as Hermes, just at 355M. So the decisive cross-pass result is specific to
**GPT-2-small**; the bootstrapping difficulty appears as soon as the base model grows, which
sharpens (not contradicts) the open challenge: scaling the win needs the curriculum /
stronger-coupling routes above, not a parameter tweak. The failed medium population is
preserved as signal at `EmmaLeonhart/reservoir-agent-gpt2-medium-batch`.

### H4 (D) — a trained silence policy (meaningful "sometimes no response")

The harness gate currently keys off the *base model's* next-token entropy, which is
arbitrary. A real policy should **speak when there is something worth saying and stay
silent otherwise**. We tested a **learned gate** on an "unresolved thread" task: a
stream of events where a rare trigger opens a thread that should be addressed (labels =
"was there a trigger within the last 5 passes").

- **The reservoir gate sees history.** The readout on the reservoir state reaches an
  **F1 score of 0.48** (P=0.71, R=0.36) on held-out data, while the **stateless
  baseline** scores **F1 = 0.03** (P=1.00, R=0.02).
- **The difference is recall.** The stateless gate can only see the trigger itself, so
  it misses almost the entire unresolved thread. The reservoir gate's carried state
  preserves the history of the trigger, allowing it to make a meaningful decision to
  keep speaking after the input has returned to baseline. (`src/reservoir/silence.py`;
  `scripts/run.py silence`.)

## D: a trained silence policy — and why this is hard brain surgery

A real agent must sometimes **stay silent** and sometimes **speak on its own**. The
current harness gate keys off the base model's next-token entropy, which is arbitrary.
So we trained a gate on the **reservoir state** for a task the reservoir is suited to —
an *unresolved thread*: a rare trigger event opens a thread the agent should address for
the next few passes, then it should fall silent. The "speak" passes are *strictly after*
the trigger, so the cue is in the **past** — invisible to the current input.

A linear gate on the reservoir state reaches **F1 ≈ 0.96** (precision 0.93, recall 1.00);
the **stateless gate** — the same gate on the current input — collapses to F1 ≈ 0.34
because it cannot see the past trigger, so it can only *always speak* (recall ≈ 1,
precision ≈ the base rate). The point is not the exact number: a stateless model **cannot
implement a selective silence policy at all**, while a reservoir-state gate can.
(`scripts/run.py silence`; `docs/silence.png`.)

**The harder conceptual point (the intended behaviour, and why it is difficult).** This
experiment trains a gate to read silence off the reservoir, but the *intended* behaviour
of the real agent is subtler and worth stating plainly:

- **The default should be to respond, not to be silent.** With no prompt and a *decayed,
  near-empty* reservoir, the base model's prior is to produce a response. Absent any
  internal activity, an automatic, context-driven response is the natural default — the
  reservoir does not need to *cause* speech.
- **Silence should attach to an *active, novel* reservoir state.** A reservoir carrying
  strong state is a genuinely new internal condition the base model never saw in
  training. That novelty is precisely what makes it the natural handle to fine-tune a new
  behaviour onto — "I am still processing, stay silent" — because a fresh state is far
  easier to attach a new response to than the model's well-worn defaults. So, perhaps
  counter-intuitively, **reservoir activity is more naturally associated with silence**,
  and its *absence* with the model's historical responding.
- **The echo state property makes the agent revert to baseline over time.** Because the
  reservoir empties (its state decays toward zero), the agent eventually reaches a state
  close to what the base model was historically trained on — so it naturally *stops* and
  drifts back to default, context-driven responding once the internal activity subsides.
- **This is aggressive brain surgery on a pretrained model, and it is genuinely hard.**
  We are trying to teach an already-trained model an entirely new behavioural axis —
  *when to stay silent, when to self-initiate* — against its strong priors. The fact that
  the Hermes cross-pass recall would not bootstrap (above) is the same difficulty showing
  up: rewiring a pretrained model's behaviour through an injected reservoir is a hard
  optimization problem even when the mechanism is verified-wired. The clean GPT-2 results
  show the mechanism *can* carry and use state; making a large pretrained agent
  *behave* differently is the real, hard frontier this project is pushing on.

## Safety by design (the rule, and what backs it)

This project follows a rule the user states plainly: **never introduce a new capability to an
AI without meaningfully taking its safety into account** — capability work is acceptable only
when paired with concrete improvements in controllability, monitorability, or risk reduction.
The Reservoir Attention Network adds capability (genuine cross-pass state, autonomous ticks,
runtime-like behaviour), so under the rule it owes safety value back. The distinctive point is
that the safety value comes from the *same* architectural feature as the capability — the
**fixed** reservoir — not from a bolt-on. Three properties, each backed by a measured result
in this report rather than by assertion:

1. **Lower-latency, durable human override** (interruptibility, below). Because the agent runs
   every tick and the reservoir integrates input continuously, an urgent "STOP" registers at
   latency 0 vs a turn-based agent's mean 3.57 passes, and a one-shot burst persists in
   reservoir state for several passes — so it is not missed if the human does not repeat it.
2. **A cheap, stable monitoring surface** (reservoir-state probe, below). A *linear* readout
   recovers an internal process variable from the reservoir at R² = 0.995 with no sparse
   autoencoder, and the pre-drift probe degrades only gradually under a fine-tuning-like
   activation drift. The reservoir weights never move, so the mapping from state to read-out
   is a fixed, low-complexity surface an operator can watch in real time.
3. **Bounded context under autonomous idling** (blank-cycle, below). The reservoir-protected
   eviction policy keeps the cache from growing without limit during blank ticks while pinning
   the time-axis, so an always-on agent does not silently exhaust its own context.

**What this does *not* yet show, stated plainly.** The probe decodes an *elapsed clock*, which
is a benign process variable; reading genuine *misalignment* signatures (deception, goal drift)
off the reservoir is a much harder, unproven extension — the resilience result says only that a
fixed-reservoir read degrades slowly, not that misalignment is legible there. The
interruptibility numbers are from a synthetic stream on the echo-state reservoir, not a live
agent under a real harness with its own latencies. And all of it is at small scale on a fixed
reservoir; the claims for the real target (a DeepSeek/Hermes-scale base) are not yet run. These
properties are the *design intent* and a first measured down-payment on it, not a finished
safety case. The project's release plan — open weights, the training/harness code, and the
reservoir monitors included rather than bolted on — is the mechanism for others to test and
extend them.

## Safety: interruptibility — a Reservoir Agent registers an urgent STOP faster, and remembers it

A recurring controllability complaint in the imported chat: a turn-based harness agent "is
doing something destructive, I yell at it to stop … and it takes like ten minutes for it to
respond," because it only reads input at turn boundaries. The claim is that a Reservoir Agent —
running every tick, with the reservoir continuously integrating input — "will be able to see
frantic messages from a human as indicating stop immediately." We measured both halves on CPU
(`scripts/run.py interrupt`; `docs/interrupt.png`).

**Polling latency (structural).** A poller that only reads input every `period` passes
registers an arrival at the next boundary: latency is uniform on `0..period-1` (mean
`(period-1)/2`). At period 8 the turn-based agent's mean latency is **3.57 passes** (max 7);
a **per-tick agent's latency is 0** — it reads on the pass the input arrives.

**Signal persistence (dynamics).** The sharper point is what happens to a *one-shot* burst —
the user yells STOP once, then goes quiet because the agent isn't answering. A matched-filter
monitor on the **reservoir state** stays above its detection threshold for **3 passes after
arrival** (fading memory carries the STOP signature forward), while a **stateless** monitor —
which sees only the current input — is above threshold on the arrival pass and **0 passes
after**. So a turn-based + stateless agent whose poll period (8) outruns the persistence window
**misses a non-repeated off-boundary burst entirely**; the per-tick reservoir agent catches it
on arrival and has a window besides. The reservoir is not just polled more often — it *retains*
the urgency, which is the architecture-level interruptibility advantage the chat argued for.

This is a safety property that falls out of the same statefulness the project builds for
capability: lower-latency, more durable response to human override. It is a measured
illustration, not a guarantee — the reservoir/leak settings set the window length, and a real
harness adds its own latencies; see the Safety-by-Design section and Limitations.

## Safety: a reservoir-state probe reads an internal clock — linearly, no SAE, and drift-tolerant

The chat made an interpretability argument for the reservoir as a *monitoring surface*: "I
don't think you'd need a sparse autoencoder for the reservoir state … it's much more simple to
have a learned representation of what is happening," and, because the reservoir weights never
change, the mapping from state to behaviour is stable — "relatively resilient to fine-tuning."
We tested the falsifiable parts (`scripts/run.py probe`; `docs/probe.png`).

**Linearly decodable, no SAE.** We defined a temporal *process property* a stateless pass
cannot see — *elapsed passes since the last trigger*, an internal clock — and fit a plain
ridge-regression readout. From the **reservoir state** it reaches **R² = 0.995**; the same
linear probe on the **instantaneous input** reaches **R² = 0.16** (elapsed time simply is not
in the current input). A *linear* probe suffices precisely because the fixed reservoir already
holds the history in a low-complexity, stable form — no sparse autoencoder needed, which is
the chat's claim borne out.

**Resilience to a fine-tuning-like drift (measured).** Fine-tuning the
readout/LoRA does not touch the reservoir weights, but it does shift the *activations that
drive* the reservoir. We model that as a fixed drift α added to the driving input and re-apply
the **pre-drift** probe. R² stays **0.99 → 0.98 → 0.94** through α = 0.1, 0.2, 0.4 and is still
**0.82** at α = 0.8 — graceful degradation, and at every drift level far above the stateless
baseline (0.16). So the probe is *usable* across moderate drift, not *invariant*: the reservoir
map is fixed, but its inputs still move, so a very large fine-tune would still erode it. That
is the precise version of "resilient monitoring surface" — a stable, cheap, linear read on an
internal state that degrades slowly rather than a guarantee.

Together with interruptibility, this is the concrete content behind the project's safety
framing: the same fixed reservoir that gives the agent a usable time-axis also gives an
operator a cheap, stable place to watch what the agent is doing. (Reading an *elapsed clock*
is the decodability demonstration; reading genuine *misalignment* signatures is a much harder,
unproven extension — flagged as future work in the Safety-by-Design section and Limitations.)

## KV: blank-cycle context growth (an always-on agent burns context, unless the reservoir is pinned)

An always-alive Reservoir Agent runs **blank ticks** — autonomous passes with no user
input. Each silent tick still appends to the KV cache, so a continuously-running agent
burns its context window *faster* than a turn-based model that only runs when prompted.
Left unmanaged the cache grows linearly with the number of ticks and the agent eventually
hits its context limit on idle activity alone. This is the operational pain point raised in
the imported Grok conversation (`data_lake/transcripts/attention-reservoir-architecture-grok.md`):
*"context explodes on a reservoir agent because a reservoir agent gets an input of blank."*

The standard remedy is StreamingLLM-style eviction — keep a few **attention-sink** tokens
plus a **recent window**, drop the middle — with one project-specific twist: the
reservoir's K/V entries are **pinned** so the persistent time-axis is never the thing
evicted. *"A really long time of no activity is signal,"* and that signal must survive.
`reservoir.kv_evict.ReservoirEvictionPolicy` implements this as a pure, torch-free policy
over per-position tags `{sink, reservoir, normal}`; with no reservoir tags it degrades to
vanilla StreamingLLM. Because the reservoir is re-prepended each pass (a *fixed* number of
pseudo-tokens, not accumulated), pinning it costs only a constant. The policy also accepts
per-position importance scores, switching the ordinary-token choice from recency to H2O-style
heavy-hitter retention while still pinning the reservoir — position-based and importance-based
eviction under one interface.

Simulating 512 blank ticks (`scripts/run.py blankcycle`; `docs/blank_cycle_kv.png`): the
**vanilla** cache grows linearly to **524 positions**, while the **reservoir-protected**
policy stays bounded at the **budget (128)** from tick ~116 onward — and **all 8 reservoir
entries are retained on every single tick**, even under heavy eviction. So the cache-burn
from autonomous idling is bounded by a constant the operator chooses, and the time-axis the
whole architecture depends on is exactly the part the policy refuses to drop. (The bound is
the point, not the specific numbers — they scale with the budget/window settings.)

This is the cheap, base-agnostic half of the cache story. The expensive half — a base model
whose attention is *natively* KV-efficient so the headroom is far larger (DeepSeek's MLA /
the V4 CSA+HCA compression discussed in the chat) — is recorded as project direction in
`todo.md`; it is not runnable on this session's hardware (see Limitations).

## Limitations (current)

- Small-scale only this session; the agentic claims (H3/H4) and the full runtime are
  out of scope and compute-gated.
- Two injection variants now exist: the **residual-stream** write (`inject.py`, wired
  into live GPT-2, H1-verified) and the richer **KV-append** mechanism (`kv_inject.py`,
  reservoir nodes as extra attention keys/values) — the latter is implemented and
  unit-tested in isolation with a clean H1 *masking* property, but **wiring it into HF
  GPT-2 (transformers 5.4) is a documented blocker** (`GPT2_INTEGRATION_BLOCKER`), left
  for a focused future item rather than a fragile patch of attention internals. This is a
  **reproducibility limitation** (flagged in review): the variant that delivers the 100%
  recall result (`kv_live.py`) runs through a bespoke path, not stock HF attention, so
  reproducing it requires that path rather than a standard `transformers` model.
- Input scaling for real-activation injection has now been **characterized** (sweet
  spot ≈ 0.08–0.24 at ρ = 0.95); it has not yet been wired as the default in the
  injection hook, and the optimum's dependence on layer/model/ρ is not yet mapped.
- The novelty claim is provisional: the reservoir-×-transformer and always-on-agent
  literatures were not yet verification-complete (see `literature/REVIEW.md` open
  questions); a citation-checked follow-up precedes any hard novelty claim.
- Whether finite-precision cross-pass reservoir state provably lifts the per-pass
  TC⁰/FO(M) bound is an open theoretical question, not a result of this work.

---

*Reservoir Agent · a cleanvibe research project · report site:
<https://emmaleonhart.github.io/reservoiragent/>*
