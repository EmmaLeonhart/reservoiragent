# Reservoir Agent — Research & Engineering Plan

## What this is

A new model class: a reservoir-augmented transformer agent with a genuine time dimension. The core idea is to take a pretrained agentic transformer (Hermes) and inject a fixed randomly-initialised reservoir into its attention mechanism at a chosen mid-depth layer. The reservoir accumulates state across every forward pass, giving the model persistent internal dynamics that evolve independently of external input. This is architecturally distinct from all existing transformer variants in one precise sense: **the model has state between forward passes**. It is not stateless. It has a time axis.

Two formal properties follow from this:

1. **Genuine time dimension.** r(t) at pass 1000 is causally downstream of every forward pass since t=0. This is not positional encoding or context length — it is actual temporal state evolution.
2. **Turing completeness.** The reservoir is a recurrent system satisfying Siegelmann & Sontag (1992/1995). Standard transformers under finite precision and finite context are not Turing complete (Hahn 2020; Pérez et al. 2021 with caveats). The composite architecture occupies a strictly higher computational class.

Neither claim is about capability level. They are claims about capability *kind*. The architecture has a capacity for endogenous state evolution that standard transformers structurally lack — a property shared with living organisms, though no claims about general intelligence are made or implied.

---

## Why this approach

Pure reservoir computing is expensive: you are betting entirely on a random initialisation being lucky, with no fallback. Training a large reservoir model from scratch costs on the order of training any large model — the stochasticity of the reservoir just adds variance on top of that cost.

This approach solves that problem by anchoring to a pretrained model that is already known to be good. The downside case of a bad reservoir seed is now "performs like vanilla Hermes" — not "produces garbage." This makes N-seed selection tractable: run N LoRA fine-tuning jobs with different reservoir initialisations, evaluate each on a benchmark suite, keep the best. The cost per experiment is a LoRA run, not a full training run.

The three-way inheritance:
- From transformers: pretrained world knowledge, cheap to fine-tune, strong agentic baseline
- From reservoir computing: rich nonlinear temporal dynamics, no vanishing gradient, long-range dependency
- Avoided: transformer statelessness, reservoir training cost

---

## Architecture

### Core design

A Hermes model with a single injection layer Lₖ at approximately 40–60% depth. At this layer, every forward pass does two things:

**Read:** the full attention output (hidden states `[seq_len × d_model]`) at Lₖ is projected via fixed random matrix W_in into the reservoir as input.

**Write:** the current reservoir state r(t) is projected via learned matrix W_out back into token-space and appended to the key/value sequence of Lₖ's attention, so upper layers can attend to it.

The reservoir update equation:

```
r(t) = tanh(W_r · r(t−1) + W_in · attn_output(t))
```

Where:
- W_r is fixed random sparse matrix, spectral radius tuned near 0.9 (edge-of-chaos prior; actual value determined empirically)
- W_in is fixed random projection matrix
- W_out is **learned** during fine-tuning — this is the primary thing fine-tuning teaches
- r(t) lives in GPU memory as a persistent tensor, mutated in-place each pass

### What is frozen vs trained

- Lower layers (L₁ … Lₖ₋₁): frozen Hermes weights
- W_r, W_in: fixed random, never trained
- W_out (reservoir readout projection): learned
- Upper layers (Lₖ₊₁ … Lₙ): LoRA fine-tuned
- Injection layer Lₖ attention: minimally modified to accept extended key/value sequence

### Key architectural property

The reservoir and context are **independent**. The context tracks content — what was said. The reservoir tracks process — what the model's attention has been doing. When context resets on a long session, the content history resets but the reservoir state persists. This gives the agent a form of memory of prior cognitive state even after context truncation, which is a feature rather than a bug.

---

## Runtime

This architecture requires a fundamentally different execution model from standard inference. Standard transformers are stateless request-response machines. This one is not.

### What changes

| Standard Hermes | Reservoir Agent |
|---|---|
| Stateless request handler | Stateful always-alive process |
| Model loaded per request | Model + reservoir pinned in GPU memory |
| Context rebuilt each call | Context buffer owned by runtime, appended to |
| No between-pass logic | Scheduler, idle timer, confidence gate |
| Synchronous tool loop | Asynchronous tool execution |

### Runtime components

**Pass scheduler:** decides when to run a forward pass. Two paths:
- *Prompted pass:* new input arrives, append to context, run pass
- *Unprompted pass:* idle timer fires with no new input, run pass over context + reservoir state only

**Persistent context buffer:** owned by the runtime, never rebuilt from scratch. The model process appends to it.

**Reservoir state store:** r(t) tensor in GPU memory, mutated in-place each pass, periodically checkpointed to disk.

**Output confidence gate:** after each pass, decide whether to emit output or stay silent. Starting proxy: entropy of top-k logits. Whether this needs to be a learned head is an open question. Silent passes update reservoir state and schedule the next pass — from the outside nothing happened, but internally the agent processed its current situation.

### Harness fork

The standard Hermes harness cannot be used. It needs to be forked and rewritten around the always-alive process model. The existing Hermes tool-call formatting and function-calling behaviour should be preserved where possible — regression testing against vanilla Hermes is an explicit early milestone.

---

## Training pipeline

### Approach

N-seed parallel LoRA fine-tuning:

1. Initialise N reservoir instances with different random seeds
2. Fine-tune each with LoRA on the same training data
3. Evaluate each on the benchmark suite
4. Keep the best-performing seed, discard the rest

N is probably 10–20. Each run is cheap relative to full training. The selection criterion is benchmark performance after a fixed number of training steps — not convergence, because convergence per seed is too expensive to wait for.

### Training data requirements

Standard instruction-following data is insufficient. The model needs training examples that *require* using the reservoir state to produce the correct output — otherwise it will learn to ignore the reservoir entirely since it isn't needed for standard next-token prediction.

Training data must include:
- Multi-pass tasks with deliberate temporal gaps
- Unresolved threads that require noticing something from N passes ago
- Examples of correct self-initiation (agent speaks on unprompted pass)
- Examples of correct silence (agent stays silent on unprompted pass)

The training data design and benchmark design are the same problem from two directions and should be developed together.

---

## Benchmarks

### Core requirement

Tasks where a stateless model **structurally cannot** succeed — not just tasks where it happens to fail. The distinction matters for the paper's claims. If a strong long-context baseline can fake temporal awareness by attending back over history, the benchmark is not testing what you think it is.

### Design principles

- Relevant signal must be subtle when it first appears — not flagged or emphasised — so context-scanning alone cannot find it
- The thing that matters must be a *change in world state* between when information appeared and now, not just distance in context
- Self-initiation tasks need an operationalised notion of appropriately timed output — not too much, not too little
- Tasks should have clear pass/fail criteria that don't require human judgment to score

### Candidate task types

**Unresolved thread detection:** agent is given two tasks in separate passes, completes one, second is never mentioned again. Correct behaviour: agent eventually notices and asks. Stateless model with no unprompted passes cannot do this structurally.

**Elapsed time estimation:** agent is asked to estimate how long it has been working on something, without being told. Reservoir state encodes a compressed history of pass count and attention patterns. Stateless model has no signal for this.

**State change detection:** a fact established early in a long session is later implicitly contradicted. Agent should notice the contradiction. Reservoir fingerprint of the earlier passes differs from the current one.

**Appropriate silence:** agent should not speak on every unprompted pass, only when it has something worth saying. Evaluate precision and recall of self-initiation decisions.

---

## Open questions

### Resolved or have clear answers
- **Reservoir/context independence:** independent by design, feature not bug
- **Multi-seed selection:** benchmark performance after fixed training steps
- **Session continuity / identity:** not a priority for initial implementation
- **Regression against vanilla Hermes:** explicit early milestone, build the suite first

### Active open questions

**Spectral radius at transformer scale.** Standard W_r tuning is calibrated for small scalar inputs. Feeding a large attention tensor may require different scaling. Treat as an explicit early experiment — probe reservoir dynamics (variance, saturation, trajectory distinguishability) across spectral radius values before committing to a training run.

**What fine-tuning teaches the lower layers.** If lower layers are frozen, W_in learns only from gradient signal flowing down from upper layers and the readout. Whether that signal is rich enough to shape W_in well is unknown. May be an argument for feeding attention output hiddens rather than raw attention weights — hiddens carry more structured semantic signal.

**Output confidence threshold design.** Top-token entropy as a starting proxy. May need to be a learned gate head with r(t) as input rather than just logits. Needs labels for "correctly silent" passes to train — which requires the training data pipeline to be in place first.

**Whether unprompted behaviour needs explicit training signal or emerges.** Probably the most interesting empirical question in the project. Run ablations: one model trained with explicit self-initiation examples, one without. See if the behaviour emerges from reservoir dynamics alone.

**Seed pre-selection proxy.** Can reservoir dynamics metrics — stability, expressivity, trajectory distinguishability across input regimes — predict final fine-tuned performance before training? If yes, pre-select seeds before any LoRA runs, dramatically reducing compute cost. Worth one experiment.

**Attention head interference.** Some heads at Lₖ may shift their attention distribution toward reservoir nodes in ways that degrade base capabilities. Subtle degradation in tool-call formatting or instruction following. Regression suite catches this.

---

## Formal claims for the paper

### Time dimension claim (abstract / introduction)
Transformers represent time as token position — an index, not a dimension the model evolves along. The reservoir agent has a genuine time axis: r(t) evolves continuously across forward passes, causally accumulating the history of the model's attention dynamics. This is not a metaphor.

### Turing completeness claim (theory section, ~half page)
1. Standard transformers under finite precision and finite context are not Turing complete — cite Hahn (2020), Pérez et al. (2021) with caveats on their assumptions.
2. The reservoir is a recurrent system satisfying Siegelmann & Sontag (1992/1995) — Turing complete in the same sense any RNN is, with the same caveats about unbounded runtime and precision.
3. The mechanism is the time dimension: Turing completeness requires unbounded computation over time, and this architecture has time where transformers do not.
4. Therefore the reservoir agent occupies a strictly higher computational class than a standard transformer.

**Important:** this section makes no claim about practical performance. Turing completeness is a theoretical ceiling. The empirical claims live in the benchmark results and are separate.

### Organism analogy (discussion section, optional, single paragraph)
The reservoir introduces endogenous state that evolves independently of external input — a property shared with living organisms and absent from standard transformers. No claims about general intelligence are made. The claim is structural: this architecture has a capacity for organism-like state evolution, and this capacity may be a precondition for certain classes of genuinely agentic behaviour that are inaccessible to stateless models regardless of their capability level.

---

## Build order

1. **Model surgery** — inject reservoir at Lₖ, verify basic Hermes behaviour survives, run reservoir with random W_out to confirm it produces output (garbage is fine at this stage). Build regression suite against vanilla Hermes.
2. **Reservoir infrastructure** — persistent state tensor in GPU memory, efficient update kernel, basic observability tooling (visualise r(t) evolution, check for saturation/explosion).
3. **Spectral radius experiment** — probe dynamics across W_r initialisations before any training. Establish what a "healthy" reservoir looks like empirically.
4. **Minimal harness fork** — two passes in sequence without reinitialising the reservoir. This is the proof-of-concept moment.
5. **Benchmark and training data design** — these are the same problem, develop together. Need at least 5 structurally clean eval tasks before running any fine-tuning.
6. **N-seed fine-tuning pipeline** — LoRA runs, parallel if hardware allows, evaluated on benchmark suite.
7. **Output confidence gate** — implement entropy proxy first, evaluate whether a learned head is necessary.
8. **Paper** — capability first (what it can do that nothing else can), mechanism second (the reservoir), formal claims third (time dimension, Turing completeness), organism framing last (discussion only).

---

## Repository

`reservoir-agent`

A fork of the Hermes harness rewritten around an always-alive process model. The Hermes tool-call interface is preserved. The execution model is not.
