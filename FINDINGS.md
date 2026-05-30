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

*No experimental results yet.* The reservoir core, dynamics sweep, and model-surgery
regression are implemented and run in the steps that follow; their metrics
(`results/*.json`) and figures will be reported here as they land. This section will
not claim a result until one has been measured.

## Limitations (current)

- Small-scale only this session; the agentic claims (H3/H4) and the full runtime are
  out of scope and compute-gated.
- The novelty claim is provisional: the reservoir-×-transformer and always-on-agent
  literatures were not yet verification-complete (see `literature/REVIEW.md` open
  questions); a citation-checked follow-up precedes any hard novelty claim.
- Whether finite-precision cross-pass reservoir state provably lifts the per-pass
  TC⁰/FO(M) bound is an open theoretical question, not a result of this work.

---

*Reservoir Agent · a cleanvibe research project · report site:
<https://emmaleonhart.github.io/reservoiragent/>*
