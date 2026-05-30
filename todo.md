# todo.md — long-horizon research plan (Reservoir Agent)

**This is the project's horizon, not its next step.** Items here are *abstract
destinations*. When work begins on one, it is pulled out, decomposed into concrete
executable steps in `queue.md`, mirrored into the task tool, and executed. As
`queue.md` drains, refill it from here. See `CLAUDE.md` § "Queue and longer-horizon
work". Grounding for all of this is `literature/REVIEW.md`.

The research question (see `README.md`): can a fixed, randomly-initialized reservoir
injected into a pretrained transformer's mid-layer attention give the model genuine
state between forward passes — a real time axis — without degrading base
capabilities, and what reservoir-dynamics regime makes that state usable signal
rather than noise?

---

## A. Hypotheses to test

- **H1 (feasibility / non-destruction).** A reservoir sidecar can be injected at a
  mid-depth attention layer of a *pretrained* transformer such that, when the readout
  W_out is zero (or near-zero), the base model's outputs are *unchanged* — i.e. the
  architecture degrades gracefully to vanilla behaviour. *(This session.)*
- **H2 (healthy-dynamics regime exists).** There is an identifiable regime of
  (spectral radius ρ, reservoir size K, injection depth, input scaling) in which the
  reservoir state is non-saturating, non-exploding, has fading memory (echo state
  property), and produces *trajectories that are distinguishable across different
  input histories* — the precondition for the state to carry usable signal. The
  classical ρ≈0.9 edge-of-chaos prior may or may not be optimal here (the literature
  says it is disputed). *(This session.)*
- **H3 (the state is readable / informative).** A light readout (W_out + LoRA) can
  learn to extract, from the reservoir state, information about the model's *process
  history* (e.g. elapsed pass-count, an unresolved earlier thread) that a stateless
  pass cannot recover from the context alone. *(Later — needs the training/eval
  pipeline.)*
- **H4 (cross-pass state changes behaviour).** With state persisting across
  *independent* forward passes (including unprompted ticks), the model exhibits
  behaviour structurally unavailable to a stateless baseline on tasks designed to
  require temporal state. *(Later — the headline agentic claim.)*

## B. Experiments / things to build

### Near-term (feasibility + dynamics — this session's scope)
- **Reservoir core.** A clean, tested echo-state reservoir module
  (`r(t)=tanh(W_r·r(t-1)+W_in·x(t))`, fixed sparse W_r scaled to target ρ, fixed W_in)
  with spectral-radius control and observability (variance, saturation fraction,
  effective rank, trajectory distance). Pure, unit-tested, runs on CPU.
- **Model surgery.** Inject the reservoir as a sidecar reading mid-layer hidden
  states of a small pretrained transformer (GPT-2-scale via HF transformers) and
  writing r(t) back into that layer's key/value sequence through W_out. Verify the
  base model is byte-for-byte unchanged when W_out=0 (H1 regression suite).
- **Spectral-radius dynamics sweep.** Drive the reservoir with real attention-output
  streams across a grid of ρ (and K, depth); measure saturation, variance,
  echo-state/fading-memory behaviour, and trajectory distinguishability. Locate the
  healthy regime (H2). Metrics → `results/`, figures → `docs/`.

### Mid-term (the real architecture)
- **Transfer the cross-pass recall win to Hermes 3B (open).** Content-addressable
  KV-prefix recall is 100% on GPT-2 but did **not** transfer to Hermes-3-Llama-3.2-3B in
  4-bit (2 attempts; loss didn't converge, recall at chance). Try: more steps / higher
  LR / full-precision (non-4-bit) LoRA / a stronger injection; diagnose why the 4-bit
  Hermes optimization stalls.
- **Make the reservoir actually USED across passes (redirect from the C negative).**
  The cross-pass *content-recall* experiment showed the model ignores the reservoir
  (stateful ≈ stateless ≈ chance). Two directions the result points to: (1) **wire
  KV-append into the live model** so upper layers *attend* to reservoir nodes
  (content-addressable) instead of one additive bias (mechanism built in `kv_inject.py`;
  see `GPT2_INTEGRATION_BLOCKER` — now also needs the Llama path); (2) target **temporal/
  process** cross-pass tasks (elapsed-pass-count, unresolved-thread, state-change) that
  match what reservoirs encode well (per the H3 delay-memory result), rather than
  specific-content recall. These are the path to "statefulness that does something".
- **Multi-pass differentiable harness — exercise the reservoir's cross-pass value.**
  The single-forward LoRA fine-tune (`scripts/run.py finetune`) trains the pipeline but
  not the cross-pass time axis (the hook ticks once per forward). Build a differentiable
  multi-pass loop (backprop-through-passes) on a **reservoir-requiring cross-context
  task** — e.g. show a fact on pass 1, truncate context, recall it on pass N from the
  reservoir alone — and train W_out (+ LoRA) so the injected model beats a stateless
  baseline. This is the experiment that would empirically demonstrate the reservoir's
  distinctive value. Substantial; now unblocked by the working injection + harness +
  fine-tune pipeline.
- **Minimal harness fork — the proof-of-concept moment.** Two forward passes in
  sequence *without* reinitialising the reservoir; show r(t) at pass 2 depends on
  pass 1. The smallest demonstration of a genuine time axis.
- **Benchmark + training-data design (one problem, two directions).** Tasks a
  stateless model *structurally* cannot do (unresolved-thread detection, elapsed-time
  estimation, state-change detection, appropriate silence). Must defeat a strong
  long-context baseline. Training data must *require* the reservoir (Block-Recurrent
  shows models otherwise learn to ignore recurrent state).
- **Readout + light LoRA training (H3).** Train W_out (+ upper-layer LoRA) on
  reservoir-requiring data; show the state becomes informative. *(Mechanism shown:
  the delay-memory readout result — `scripts/run.py h3`. Training W_out through the LM
  on a semantic task remains.)*
- **Wire KV-append into live GPT-2 (`GPT2_INTEGRATION_BLOCKER`).** The mechanism is
  built + unit-tested (`src/reservoir/kv_inject.py`); the remaining step is an
  eager-path `GPT2Attention.forward` override (transformers 5.4) that appends the
  reservoir key/value rows and extends the attention mask. Bounded but version-sensitive.

### Long-term (aspirational — compute-gated, full vision)
- **N-seed selection pipeline.** 10–20 reservoir seeds × LoRA fine-tunes, keep the
  best by benchmark. Investigate a dynamics-based seed pre-selection proxy.
- **Always-alive runtime.** Fork the Hermes harness into a stateful process: pass
  scheduler (prompted + unprompted passes), persistent context buffer, reservoir
  state store (GPU-pinned, checkpointed), output confidence gate. Regression vs
  vanilla Hermes throughout.
- **Scale-up to an agentic model (Hermes).** The original target; requires real GPU
  budget. Tracked here as the destination, not a near-term step.
- Still do this as a part of the workflow run even though I might not have enough local compute. Just to see if we can do it.

## C. Theory to write (correctly scoped)
- The genuine time-dimension argument: r(t) is causally downstream of every pass.
- The expressivity framing: finite-precision transformer ⊆ TC⁰/FO(M) per pass;
  cross-pass state is the documented lever past that ceiling — **stated with the
  arbitrary-precision caveat**, posing (not asserting) whether a finite-precision
  reservoir lifts the bound. Reservoir ∈ Siegelmann–Sontag recurrent class supplies
  the structural ingredient.
- The organism analogy as a single, clearly-bounded discussion paragraph.

## D. Open questions carried from the literature review
- **Citation-checked novelty follow-up.** The reservoir-×-transformer and
  always-on-agent literatures (§4–§5 of `literature/sources.md`) were not
  verification-complete. Before publishing any hard novelty claim, run a dedicated,
  citation-verified review of those areas (and re-check the fast-moving 2024–2026
  recurrent-transformer landscape).
- **Theoretical open question (above): does finite-precision cross-pass reservoir
  state provably lift TC⁰/FO(M)?** Pose precisely; do not assert.
- **Does the edge-of-chaos prior survive the high-dimensional attention-input
  regime?** Empirical, answered by the dynamics sweep.

## E. The report (the deliverable shape)
- `FINDINGS.md` (question → method → results → limitations) kept current as results
  land; the themed `docs/` GitHub Pages site + built `report.pdf`. Capability first,
  mechanism second, formal claims third (scoped), organism framing last. Re-theme the
  recovered `data_lake/diagrams/*.svg` (or redraw) for the published site.
