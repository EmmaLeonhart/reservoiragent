# Reservoir Agent — Findings

**Status: in progress.** This document is the project's write-up. It is kept current
as results land; the sections below state the question and method now, and report
results honestly as the experiments run. Until the experiments produce metrics, the
Results section says so plainly rather than implying findings that do not yet exist.

## Question

Can a fixed, randomly-initialized reservoir injected into a pretrained transformer's
mid-layer attention give the model genuine state **between** forward passes — a real
time axis — without degrading its base capabilities, and what reservoir-dynamics
regime (spectral radius, reservoir size, injection depth) makes that injected state
usable signal rather than noise?

This session scopes the question as a **feasibility + dynamics study** at small scale
(GPT-2-scale base, single machine). The full vision — forking an agent harness into an
always-alive runtime and N-seed LoRA selection at agent scale — is the long-horizon
target (see `todo.md`).

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
`scripts/run.py nseed-select`. Other proxies remain untested.)

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

## Limitations (current)

- Small-scale only this session; the agentic claims (H3/H4) and the full runtime are
  out of scope and compute-gated.
- Two injection variants now exist: the **residual-stream** write (`inject.py`, wired
  into live GPT-2, H1-verified) and the richer **KV-append** mechanism (`kv_inject.py`,
  reservoir nodes as extra attention keys/values) — the latter is implemented and
  unit-tested in isolation with a clean H1 *masking* property, but **wiring it into HF
  GPT-2 (transformers 5.4) is a documented blocker** (`GPT2_INTEGRATION_BLOCKER`), left
  for a focused future item rather than a fragile patch of attention internals.
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
