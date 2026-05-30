# Literature Review — Reservoir Agent

**Question.** Can a *fixed, randomly-initialized* reservoir injected into a
*pretrained* transformer's mid-layer attention give the model genuine state
*between* forward passes — a real time axis — without degrading base capabilities,
and what reservoir-dynamics regime (spectral radius, size, injection depth) makes
that injected state usable signal rather than noise?

This review grounds the project in three bodies of work: (1) reservoir computing,
(2) the expressivity gap between stateless transformers and stateful recurrent
systems, and (3) the family of recurrence-augmented transformers that is the
closest prior art. Per-source notes and citations are in [`sources.md`](./sources.md).
How it was built — and its limits — is in *Method & confidence* at the end.

---

## 1. What is already known

**Reservoir computing is a mature, faithful home for the core mechanism.** The
Reservoir Agent's defining move — fix the recurrent weights (W_r, W_in) at random
and train *only* a readout (W_out) — is exactly the echo-state-network (Jaeger
2001) / liquid-state-machine (Maass 2002) paradigm, surveyed and codified by
Lukoševičius & Jaeger (2009). This paradigm is well-established and has historically
*outperformed* fully-trained RNNs on many temporal tasks. So the project is not
inventing a learning rule; it is *relocating* a proven one into a new substrate.

**The dynamics regime the project wants to characterize is real but genuinely
unsettled.** Usable reservoir behaviour is governed by the **echo state property**
(ESP): the influence of past state and input must fade rather than persist or
amplify. Scaling the recurrent matrix so its spectral radius ρ(W) < 1 *almost
always* secures the ESP, but — contrary to a widespread textbook shorthand — that
condition is **neither exactly necessary nor exactly sufficient** (Lukoševičius &
Jaeger 2009 explicitly correct this; the conservative sufficient condition is on the
largest singular value). The "operate at the edge of chaos for maximal computational
power and memory" heuristic is **disputed**: counterexamples show short-term memory
capacity can peak *before* or *away from* the edge (arXiv:2012.01409, 1912.11213).
**Implication for us:** the classical ρ≈0.9 recipe is a *prior, not an answer* — the
spectral-radius sweep is a legitimate open empirical question, which is precisely
what this project's dynamics experiment measures.

**Reservoir memory is fading transient memory, capacity-bounded by size.** It is
*not* attractor-based working memory; long spans require large reservoirs and/or
leaky integrators with long time constants, and linear memory capacity is bounded by
reservoir size (MC ≤ N) (Jaeger 2012). **Implication:** reservoir size K and leak
rate are the knobs for *how much* cross-pass state can be carried, and the plan's
"memory of prior cognitive state after context truncation" should be framed as
*slowly-decaying* state, not permanent memory.

**The stateless-transformer limitation is sharp and provable.** A fixed-depth,
finite-precision transformer is, *per forward pass*, confined to a low complexity
class: saturated/float transformers ⊆ **TC⁰** (Merrill, Sabharwal & Smith 2022),
and log-precision transformers are exactly captured by first-order logic with
majority quantifiers, **FO(M)** — the tightest known bound, provably unable to
compute e.g. boolean matrix permanents (Merrill & Sabharwal 2023). Fixed-size
self-attention also cannot model hierarchical/recursive structure as length grows
without growing depth/heads (Hahn 2020). This is the limitation the project's time
axis is meant to address — and it can now be stated rigorously.

**Cross-pass state is the documented lever out of that ceiling — but with a
precision caveat.** The TC⁰/FO(M) upper-bound proof *explicitly breaks* once the
model feeds generated output back into its input at the next step (Merrill &
Sabharwal 2023, fn. 5), and under such feedback with *arbitrary precision* the
Transformer is Turing-complete from internal dense representations alone, no external
memory (Pérez et al. 2019). Finite recurrent sigmoidal nets are Turing-complete in
principle (Siegelmann & Sontag 1991/1995). **But** the transformer
Turing-completeness results rely on *unbounded precision* (dense reps acting as
unbounded memory); at finite precision they fail. So the structural ingredient for
escaping statelessness — recurrent state across passes — is exactly what the
reservoir adds, while a *proof* that a finite-precision continuous reservoir state
lifts the bound does **not** exist and must not be asserted.

## 2. The closest prior art, and the gap

A decade of work adds recurrence/state/memory to transformers (Transformer-XL,
Compressive Transformer, Universal Transformer, Block-Recurrent Transformer,
Memorizing Transformers, RMT, RWKV, RetNet, S4/Mamba, Titans — see [`sources.md` §3](./sources.md)).
Classified on the two axes that matter here:

| System | Recurrence: trained or fixed-random? | State: within-sequence or across *independent* passes? |
|---|---|---|
| Transformer-XL | trained | within sequence (cached segment, stop-grad) |
| Compressive Transformer | trained | within sequence (compressed cache) |
| Universal Transformer | trained | **intra-pass** depth recurrence (not temporal) |
| Block-Recurrent Transformer | trained (LSTM gates) | within sequence |
| Memorizing Transformers | trained retrieval | within document (stored kNN, non-dynamical) |
| Recurrent Memory Transformer | trained (memory tokens) | across segments of one sequence |
| RWKV / RetNet | trained | within sequence (RNN-form state) |
| S4 / Mamba | trained (learned SSM) | within sequence |
| Titans | trained (test-time weight updates) | within a stream |
| **Reservoir Agent** | **fixed-random** | **across genuinely independent forward passes** |

**The gap, stated sharply.** Every prior system uses **trained** recurrence and
carries state **within a (possibly very long) sequence or segment chain**. The
Reservoir Agent occupies the empty cell: a **fixed-random** reservoir whose state
persists **across independent forward passes**, including *unprompted* ticks with no
new input — and it is injected into a **pretrained, frozen** backbone, trained only
through a readout + light LoRA. Two corroborating specifics from the prior art:
- Block-Recurrent Transformer documents that recurrent transformers have a real
  failure mode where "the model learns to completely ignore the recurrent state."
  This independently confirms the plan's central training-data concern: tasks must
  *require* the reservoir or it will be ignored.
- Universal Transformer's "recurrence" is in depth within one pass — a useful
  reminder to keep the project's claim precise: this is *temporal* recurrence across
  passes, a different axis.

## 3. What this project adds

1. **A substrate transplant.** It moves the fixed-reservoir/trained-readout paradigm
   from scalar/low-dimensional ESN inputs into the *high-dimensional internal
   activations of a pretrained transformer*, with **attention as the readout
   surface** (reservoir nodes as extra keys/values). Whether the classical ESP /
   spectral-radius recipes transfer to this regime is genuinely unknown — the
   project's first empirical contribution.
2. **State across independent passes, not within a sequence.** Unlike all of §2, the
   carried state is decoupled from the context window and survives context
   truncation — the property that motivates the "agent that keeps processing between
   prompts" runtime.
3. **A theory section that is correctly scoped.** It can state the per-pass TC⁰/FO(M)
   ceiling and the cross-pass escape *precisely*, including the arbitrary-precision
   caveat, rather than overclaiming a Turing-completeness proof for this mechanism.
4. **A feasibility + dynamics result (this session's deliverable):** can the
   injection be done on a small pretrained model without breaking it (regression),
   and what (spectral radius, size, depth) regime yields distinguishable,
   non-saturating, non-exploding reservoir trajectories.

## 4. Open questions / caveats (carried into `todo.md`)

- **Novelty is provisional.** "No one has done exactly this" is established only
  *against the verified set* (reservoir foundations + expressivity). The
  reservoir-×-transformer and always-on-agent literatures (§4–§5 of `sources.md`)
  were **not** verification-complete; a dedicated, citation-checked follow-up is
  required before publishing a hard novelty claim. (Some search-returned arXiv IDs in
  those areas did not resolve and were discarded rather than cited.)
- **Does finite-precision cross-pass reservoir state provably lift the TC⁰/FO(M)
  bound?** The known escape routes need arbitrary precision and token-level feedback,
  not continuous hidden-state feedback. This is an **open theoretical question**, not
  an established result — the paper should pose it, not assert it.
- **Does the edge-of-chaos prior survive the transformer-scale, high-dimensional
  input regime?** The optimum is task/architecture dependent even for classical ESNs;
  reading from and writing into attention activations may move it. Empirical.
- **Will fine-tuning teach the model to use the reservoir, or ignore it?** Prior art
  (Block-Recurrent) shows "ignore the state" is a real local optimum; the training
  data must structurally require the carried state.

---

## Method & confidence

The §1–§2 findings were produced by a multi-agent deep-research pass (5 search
angles → 21 sources fetched → 96 candidate claims → 25 adversarially verified with
3-vote refutation; 24 confirmed, 1 refuted). All rest on primary/canonical sources
(Jaeger, Maass, Lukoševičius, Hahn, Merrill & Sabharwal, Pérez, Siegelmann &
Sontag, Weiss). One refuted claim (a PARITY-specific reading of Hahn 2020) was
excluded; one (Pérez Turing-completeness) carries an explicit arbitrary-precision
caveat. The §3 prior-art table is the *decisive novelty axis*; the deep-research pass
did not finish verifying it, so the four pivotal entries (Transformer-XL, RMT,
Block-Recurrent, Titans) were re-verified by targeted search here and the rest cited
from canonical record. §4–§5 are openly incomplete and tracked as open questions
above. The prior-art landscape (esp. 2024–2026) is fast-moving and must be
re-checked before publication.
