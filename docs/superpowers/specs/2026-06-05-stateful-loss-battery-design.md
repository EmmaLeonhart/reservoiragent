# Design: the stateful-episode loss battery (the agent's training objective)

**Date:** 2026-06-05
**Status:** design approved in principle (task selection done); framework not yet built; no training run yet.

## The problem this fixes

Cross-pass colour-recall, alone, is far too narrow a training objective. It proves the
reservoir *can* carry one token across a wiped context, but it does not make the model an
independent, always-alive agent. The agent's stateful behaviour must be a **training
discovery** — emergent from the loss, never from the system prompt (the system prompt is
kept tiny and neutral; describing the reservoir in it would just have the base model
parrot it, demonstrating nothing).

## The core abstraction: a stateful episode

Every task in the battery is an **episode**: a scripted sequence of forward passes through
the reservoir-injected model, with the reservoir state carried across all of them and the
**context wiped** at defined points so the *only* information bridge is the reservoir
state.

An episode is a list of **steps**. Each step is one forward pass and carries:

- `inject` (optional): user/instruction text folded into the context this pass.
- `wipe` (bool): clear the context buffer *before* this pass (reservoir state is **not**
  cleared — that's the point). Models the "context is gone, only state remains."
- `target` (optional): the supervision for this pass —
  - a **string** → the model should emit it here (cross-entropy on those token(s));
  - `SILENCE` → the model should emit nothing here (silence penalty: loss on the gate /
    on producing a non-pad token).
  - `None` → no supervision this pass (a free tick; state still evolves).

The **episode loss** = Σ over supervised steps of [ CE(emit target) or silence penalty ],
backpropagated **through the carried reservoir state** across the whole episode (the torch
graph spans passes, as `run_cross_pass_kv` already does for the 2-pass case).

The **battery loss** = expectation over episodes sampled from a mix of the eight
generators below. Trainable: `W_res` (reservoir→prefix) + LoRA. Fixed: the reservoir
(`W_r`, `W_in`) and the base model.

## The eight task generators (all selected)

Each is a parameterised episode generator (random words / numbers / delays / orders), with
its own metric for eval.

**Memory family (state must survive a context wipe):**
1. **Secret-word recall** — inject a word; wipe; later step targets that word. *(The proven
   case; `crosspass.py` is this episode.)* Metric: recall accuracy.
2. **Running accumulation** — steps inject `+k` with a wipe between each; a final step
   targets the total. Forces a numeric register in state. Metric: exact-total accuracy.
3. **Sequence / order recall** — several items over separate passes; wipe; a step targets
   them in order. Metric: ordered-match / edit distance.
4. **Deferred fact (long horizon)** — inject a fact; many `None` idle steps; a late step
   targets it. Forces slow fading memory (echo-state under load). Metric: recall accuracy
   vs delay length.

**Timing & agency family (state must hold a clock / intention / emit policy):**
5. **Timed action** — instruction "emit X after N"; then `SILENCE` targets for N-1 steps,
   then an emit target X. Forces a pass-counter in state. ("5 minutes" = N ticks at train
   time, wall-clock at run time.) Metric: on-time emission rate (and false-early rate).
6. **Interruption & resume** — begin task A; interrupt with B (B completes); then `None`
   steps; a step targets resuming A. The unfinished thread lives in state. Metric: resume
   success. *(Substrate: `interrupt.py`, `blank_cycle.py`.)*
7. **Self-initiation (unprompted)** — a pending item is set; after K idle steps with no
   input, an idle step targets proactively raising it. Trains speaking without a prompt.
   Metric: self-init precision/recall (raised the right thing, at a reasonable time).
8. **Learned silence (when to speak)** — idle steps with nothing to say carry `SILENCE`
   targets; the gate must stay shut unless a target says otherwise. The counterpart to #7.
   Metric: silence precision/recall. *(Substrate: `silence.py`.)*

#7 and #8 jointly define a **trained** emit policy (replacing the entropy-proxy gate that
the live app currently uses).

## Components to build (framework first, then generators, then trainer)

- `src/reservoir/episode.py` — the `Episode`/`Step` data model + `SILENCE` sentinel; a pure
  `run_episode(lm, episode)` that executes steps (inject, wipe, forward, collect supervised
  logits) and a `episode_loss(...)`; unit-tested with a fake LM (no torch) for the
  bookkeeping and torch-gated for the loss.
- `src/reservoir/battery.py` — the eight generators (`gen_recall`, `gen_accumulate`,
  `gen_sequence`, `gen_deferred`, `gen_timed`, `gen_interrupt`, `gen_selfinit`,
  `gen_silence`), each `-> Episode`; pure, unit-tested (deterministic given a seed).
- `src/reservoir/train_battery.py` — the through-passes trainer: sample episodes from a
  weighted mix, accumulate `episode_loss`, step AdamW on `W_res`+LoRA; per-task eval
  metrics; save the trained model via the existing `persist.save_reservoir_model`.
- `scripts/run.py battery` — entry point (model, steps, task weights, save dir).
- Eval/report: per-task metric table → `results/` + a figure → `docs/`.

The trained artifact loads straight into the live app (`app/server`), replacing the
untrained substrate — at which point the always-alive agent's behaviour is a training
discovery, and the app's reservoir-gain slider fades a *real* trained signal in and out.

## Scope / sequencing (NOT all at once)

This is the research core (`todo.md` H3/H4) and a multi-step build. Proposed order, each a
runnable checkpoint:

1. `episode.py` + `battery.py` recall+accumulate+timed, with tests (no training). Verify
   episodes execute and losses compute on a fake LM and on GPT-2.
2. The trainer + a small GPT-2 battery run on those three tasks → confirm the composite
   loss trains and the per-task metrics move (the "does the battery work" checkpoint).
3. Add the remaining five generators + metrics.
4. Scale to Qwen-1.5B; load the trained model into the live app.

## Deferred (harness roadmap, per user: build when actually running)

In-app Hugging Face download + model selection (the `registry.py`/installer code folds in),
in-app training controls, and packaging. Not local-debug-now work.

## Honest limits

- Through-passes backprop over long episodes is memory-heavy; long-horizon (#4) and timed
  (#5) episodes may need truncated/segmented backprop or short N at first.
- Whether one reservoir + small readout can satisfy all eight families *at once* is the
  open empirical question — the negative (which families resist) is itself a result.
- "Independent agent" is bounded here to this measurable battery; it is not a claim of
  general autonomy.
