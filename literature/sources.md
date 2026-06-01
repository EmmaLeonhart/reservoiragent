# Sources — Reservoir Attention Network (RAN) literature review

One entry per source: citation · key claim relevant to us · method (one line) ·
how it relates to / differs from the **Reservoir Attention Network (RAN)** 
(fixed-random reservoir injected into a *pretrained* transformer's mid-layer 
attention, with state persisting across *independent* forward passes via a 
learned KV readout). We refer to a specific instantiation of this architecture 
as a **Reservoir Agent**.

**Verification status.** Entries in §1 (reservoir foundations) and §2
(expressivity) were produced and adversarially fact-checked by a multi-agent
deep-research pass (21 sources → 96 claims → 25 verified, 24 confirmed / 1
refuted). Entries in §3 (recurrent-transformer prior art) are the *decisive
novelty axis*; the deep-research pass fetched but did not finish verifying them
(budget-dropped), so the four pivotal ones (Transformer-XL, RMT, Block-Recurrent,
Titans) were re-verified here by targeted search, and the remaining canonical
ones are cited from established record. §4–§5 remain partially open (see
`REVIEW.md` → Open questions).

---

## §1 — Reservoir computing & echo-state foundations  *(verified)*

### Jaeger 2001 — "The 'echo state' approach to analysing and training recurrent neural networks"
GMD Report 148, German National Research Center for Information Technology.
<https://www.ai.rug.nl/minds/uploads/EchoStatesTechRep.pdf>
- **Key claim:** Training "modifies only the weights to output units… this amounts
  to a linear regression task." The recurrent reservoir is fixed; only the readout
  is learned. Gives the spectral-radius propositions: largest singular value < 1 is
  *sufficient* for the echo state property (ESP); |λ_max| > 1 (spectral radius > 1)
  → "asymptotically unstable null state… no echo states" for input sets containing 0.
- **Method:** Random fixed recurrent net + linear readout trained by regression.
- **Relation:** This *is* the paradigm the **Reservoir Attention Network (RAN)** 
  instantiates (W_r, W_in fixed; W_out learned). The spectral-radius regime is 
  exactly what this project proposes to characterize — but at 
  transformer-attention scale, not scalar inputs.

### Maass, Natschläger & Markram 2002 — "Real-time computing without stable states: a new framework for neural computation based on perturbations" (Liquid State Machines)
*Neural Computation* 14(11):2531–2560. <https://pubmed.ncbi.nlm.nih.gov/12433288/>
- **Key claim:** The transient (non-attractor) dynamics of a "sufficiently large
  and heterogeneous neural circuit may serve as universal analog fading memory,"
  with only a task-specific readout trained. "Universal" here is *bounded*
  (real-time computing on time-varying inputs with fading memory), **not**
  Turing-universal.
- **Method:** Fixed recurrent "liquid" + trained readout; computation on perturbed
  transient states.
- **Relation:** The other root of reservoir computing; supplies the "compute on
  transient state, train only the readout" justification. The bounded sense of
  "universal" is a caveat the project's organism/Turing framing must respect.

### Lukoševičius & Jaeger 2009 — "Reservoir computing approaches to recurrent neural network training"
*Computer Science Review* 3(3):127–149.
<https://www.sciencedirect.com/science/article/abs/pii/S1574013709000173>
- **Key claim:** Defines reservoir computing (RNN generated randomly, only readout
  trained; "outperformed classical fully trained RNNs in many tasks"). Explicitly
  *corrects* the common misconception that ρ(W) < 1 is necessary-and-sufficient for
  the ESP ("This is wrong"). Notes the edge-of-chaos / high-computational-power
  claim is "**not undisputed**" and "does not universally imply that such reservoirs
  are optimal."
- **Method:** Survey + practical guide.
- **Relation:** The methodological backbone for the dynamics experiment, *and* the
  warning that classical ESN recipes are a prior, not an answer — the optimum is
  task/architecture dependent, which is why the project measures it empirically.

### Jaeger 2012 (reprint of 2002 NIPS work) — short-term memory in echo state networks
<https://www.ai.rug.nl/minds/uploads/2478_Jaeger12.pdf>
- **Key claim:** Reservoir memory is "based on transient (albeit slow) dynamics and
  decay[s] with time" — **not** attractor-based working memory. Long memory spans
  come from large reservoirs and/or leaky-integrator neurons with long time
  constants. Memory capacity is bounded by reservoir size (MC ≤ N).
- **Method:** Linear short-term memory capacity analysis.
- **Relation:** Tells us *what kind* of state the reservoir can carry across passes
  (decaying transient, capacity ∝ size) and that size K and leak rate are the knobs.
  Tempers the "memory of prior cognitive state after context truncation" claim: it
  is fading, not persistent.

### Disputes on edge-of-chaos optimality *(corroborating, verifier-cited)*
- "Do Reservoir Computers Work Best at the Edge of Chaos?" <https://arxiv.org/abs/2012.01409>
- "Optimal short-term memory before the edge of chaos" <https://arxiv.org/abs/1912.11213>
- **Relation:** Concrete counterexamples where capacity peaks away from / before the
  edge — direct support for treating the spectral-radius sweep as an open empirical
  question rather than assuming ρ≈0.9 is best.

---

## §2 — Expressivity: transformer limits vs. recurrent Turing-completeness  *(verified)*

### Hahn 2020 — "Theoretical Limitations of Self-Attention in Neural Sequence Models"
*TACL*. <https://arxiv.org/abs/1906.06755>
- **Key claim:** Fixed-size self-attention "cannot model hierarchical structure…
  unless the number of layers or heads increases with input length." (Note: the
  PARITY-specific phrasing was *refuted* in our verification, vote 1–2; the
  hierarchical-recursion limitation survived, 3–0.)
- **Relation:** First-order statement that a single stateless pass is structurally
  bounded — the limitation the project's cross-pass state aims to address.

### Merrill & Sabharwal 2023 — "The Parallelism Tradeoff: Limitations of Log-Precision Transformers"
NeurIPS 2023. <https://papers.neurips.cc/paper_files/paper/2023/file/a48e5877c7bf86a513950ab23b360498-Paper-Conference.pdf>
(with Merrill, Sabharwal & Smith 2022, *saturated transformers ⊆ TC⁰* — <https://arxiv.org/abs/2106.16213>)
- **Key claim:** Any log-precision transformer classifier is expressible in
  first-order logic with majority quantifiers, **FO(M)** — the tightest known upper
  bound; provably unable to compute boolean matrix permanents. Saturated/float
  transformers ⊆ TC⁰. **Crucially (footnote 5):** the proof "no longer goes through
  if the decoder can generate tokens that get added to the input at the next step" —
  i.e. genuine cross-pass state feedback escapes the bound.
- **Relation:** The sharpest form of the gap. Per *forward pass*, a finite-precision
  transformer is in TC⁰/FO(M); carrying explicit state across passes is *exactly*
  the documented lever out of that ceiling.

### Pérez, Barceló & Marinkovic 2019/2021 — "Attention is Turing-Complete"
ICLR 2019 / JMLR 2021. <https://arxiv.org/abs/1901.03429>
- **Key claim:** The Transformer (and Neural GPU) are Turing-complete "exclusively
  based on their capacity to compute and access internal dense representations" —
  no external memory needed. **Caveat (our verification, 2–1):** holds *only* under
  arbitrary/unbounded precision (dense reps act as unbounded memory) and
  autoregressive output-feedback; under finite precision it fails.
- **Relation:** Shows recurrence/feedback + unbounded precision ⇒ universality. The
  Reservoir Agent operates at *finite* precision, so whether a continuous reservoir
  state across passes lifts the bound is **open** — the project must not overclaim.

### Siegelmann & Sontag 1991/1995 — Turing-completeness of finite recurrent nets
Applied Math Letters 1991; JCSS 1995. <http://www.sontaglab.org/PUBDIR/Author/SIEGELMANN-HT.html>
- **Key claim:** A finite net of sigmoidal neurons (<100,000; later 886 processors)
  simulates a universal Turing machine in real time with rational weights, no
  high-order connections. Rational weights → Turing-equivalent; real weights →
  super-Turing.
- **Relation:** The reservoir is a recurrent system of this class, so the *structural
  ingredient* for universal computation is present — the basis for the paper's
  Turing-completeness section, with the precision caveat above.

### Weiss, Goldberg & Yahav 2021 — "Thinking Like Transformers" (RASP)
ICML 2021. <https://arxiv.org/abs/2106.06981>
- **Key claim:** Maps attention + FFN to programming primitives (RASP), a
  computational model for transformer-encoders. *(Medium confidence; relevance to
  KV-injection is indirect; RASP-completeness has known limits.)*
- **Relation:** A lens for reasoning about how an injected reservoir readout in the
  KV sequence interacts with the transformer's native computation.

---

## §3 — Recurrence / state / memory added to transformers  *(closest prior art — the gap)*

For each: **recurrence trained or fixed-random?** and **state within-sequence or
across independent passes?** — the two axes on which the **Reservoir Attention 
Network (RAN)** differs.

### Dai et al. 2019 — Transformer-XL  *(re-verified)*
ACL 2019. <https://arxiv.org/abs/1901.02860>
- Segment-level recurrence: cache previous segment's hidden states (stop-gradient)
  as extended context. **Trained** recurrence; state carried **within one long
  sequence** (sliding window), not across independent passes.

### Rae et al. 2020 — Compressive Transformer
ICLR 2020. <https://arxiv.org/abs/1911.05507>
- Adds a compressed long-term memory to Transformer-XL's cache. **Trained**;
  **within-sequence**.

### Dehghani et al. 2019 — Universal Transformer
ICLR 2019. <https://arxiv.org/abs/1807.03819>
- Recurrence is in **depth** (weight-tied layers applied repeatedly within a single
  forward pass) with dynamic halting — **not** temporal state across passes.
  **Trained**; intra-pass.

### Hutchins et al. 2022 — Block-Recurrent Transformer  *(re-verified)*
NeurIPS 2022. <https://arxiv.org/abs/2203.07852>
- A transformer layer applied recurrently over blocks, with **LSTM-style trained
  gates**. State carried **within a long sequence**. **Directly relevant warning:**
  "Recurrence has a failure mode where the model learns to completely ignore the
  recurrent state… a local optimum" — exactly the risk the plan flags for the
  reservoir (training data must *require* the state).

### Wu et al. 2022 — Memorizing Transformers
ICLR 2022. <https://arxiv.org/abs/2203.08913>
- Non-differentiable kNN retrieval over a frozen cache of past (key,value) pairs.
  Memory is **stored, not dynamical**; retrieval **trained**, memory **within a
  document/stream**. No recurrent state evolution.

### Bulatov, Kuratov & Burtsev 2022 — Recurrent Memory Transformer (RMT)  *(re-verified)*
NeurIPS 2022. <https://arxiv.org/abs/2207.06881>
- Special read/write **memory tokens** pass state between segments of one long
  sequence. **Trained**; **across segments of a single sequence** (recurrence within
  a chained sequence, not across independent agent forward passes).

### Peng et al. 2023 — RWKV
<https://arxiv.org/abs/2305.13048>  ·  Sun et al. 2023 — RetNet <https://arxiv.org/abs/2307.08621>
- Linear-attention / retention formulations with an **RNN-form recurrent state**
  during inference. **Trained**; state evolves **within a sequence**.

### Gu, Goel & Ré 2022 — S4  <https://arxiv.org/abs/2111.00396>  ·  Gu & Dao 2023 — Mamba <https://arxiv.org/abs/2312.00752>
- Structured state-space models: a **trained**, (selectively) linear recurrent state
  threaded **within a sequence**. The recurrence matrices are learned, not fixed-random.

### Behrouz, Zhong & Mirrokni 2024 — Titans  *(re-verified)*
<https://arxiv.org/abs/2501.00663>
- A neural long-term memory module that updates **its own weights at test time**
  (gradient-based "surprise" memory) during the forward pass. **Trained** (and
  test-time-learned); operates **within a stream**. State is a learned memory, not a
  fixed-random reservoir.

**Pattern across §3:** every system uses **trained** recurrence, and every one
carries state **within a (possibly very long) sequence or segment chain**. None
uses a **fixed-random** reservoir, and none carries state across **genuinely
independent** forward passes (the agent's prompted *and unprompted* ticks). That
empty cell is where the Reservoir Agent sits.

---

## §4 — Reservoir × transformer / reservoir-in-pretrained-net  *(verified — the novelty axis)*

A dedicated, citation-verified second pass (18 sources → 25 claims adversarially
verified, 18 confirmed). The four close items below each merge a reservoir with a
transformer, but **every one fails on at least one of the three load-bearing axes**:
(a) injection into a *pretrained/frozen* backbone, (b) the recurrent part kept
*fixed-random* while only a readout is trained, (c) state persisting *across
independent forward passes*.

### Shen, Baevski, Morcos, Keutzer, Auli & Kiela 2021 — "Reservoir Transformers"
ACL-IJCNLP 2021. <https://arxiv.org/abs/2012.15045> · <https://aclanthology.org/2021.acl-long.331/>
- **Key claim:** "transformers obtain impressive performance even when some of the
  layers are randomly initialized and never updated" — non-linear *reservoir* layers
  interspersed with regular transformer layers, with "subsequent transformer layers
  acting as readout functions"; improves wall-clock time-to-convergence on MT/MLM.
- **Differs on all three axes:** trained end-to-end **from scratch** (not injected into
  a pretrained backbone); reservoir is a **frozen feed-forward layer in the stack**, not
  an ESN injected into attention as keys/values; **no cross-pass state** (one sequence
  per pass). The closest-named prior art on "fixed-random layers inside a transformer."

### Bendi-Ouis & Hinaut 2025 — "Echo State Transformer" (EST)
Inria/Mnemosyne. <https://arxiv.org/abs/2507.02917>
- **Key claim:** several parallel random recurrent reservoirs as a fixed-size working
  memory, with **attention applied over the reservoir units instead of input tokens**
  (attention-as-readout, linear complexity). The most direct reservoir+attention hybrid.
- **Differs:** trained **from scratch**; reservoirs **not purely fixed** ("classical
  reservoir hyperparameters controlling the dynamics are now trained" — adaptive leak
  rate); state persists **within a sequence** per timestep, no cross-pass/idle axis.
  (The item nearest to blurring the trained-vs-fixed line.)

### Liu & Xu 2025 — "Echo Flow Networks" (EFN / EchoFormer)
Stevens Institute. <https://arxiv.org/html/2509.24122>
- **Key claim:** a **fixed** randomly-initialized streaming reservoir (X-ESN, updated
  without backprop) fused with a **trainable** transformer forecaster (PatchTST) via a
  **trained cross-attention** readout.
- **Differs:** dual-stream, fused at the **input** level (not an ESN in a frozen
  transformer's internal attention); **backbone is trainable**, not frozen-pretrained;
  reservoir state is **within-sequence** (rolling windows, reinitialised); time-series
  forecasting, not LLM agents.

### Singh, Sharma, Dey & Raman 2025 — "FreezeTST (Frozen in Time)"
ECAI 2025. <https://arxiv.org/abs/2508.18130>
- **Key claim:** frozen random-feature ("reservoir") blocks interleaved with trainable
  transformer layers that "query this memory through self-attention" — nonlinear memory
  at zero optimisation cost.
- **Differs:** "reservoir" = fixed random **feature expansion recomputed each pass**
  (not even a recurrent ESN state), trained **from scratch**, time-series, **no cross-pass
  persistence**.

### Gallicchio & Scardapane 2020 — "Deep Randomized Neural Networks" (+ DeepESN)
Springer 2020 / <https://arxiv.org/abs/2002.12287>; DeepESN survey <https://arxiv.org/abs/1712.04323>
- **Key claim:** the recurrent layer "can be left untrained after initialization,
  provided the Echo State Property is in place… the only trainable part is the output
  readout," trained by ridge regression; extends to *deep* stacks of fixed reservoirs.
- **Relation:** foundational canon the project **builds on** (fixed reservoir + trained
  readout, incl. deep stacks). Recurrence is within-sequence; none of it involves a
  pretrained transformer, attention readout, or cross-pass state. (This survey itself
  cites "Reservoir Transformers" as the recognised "fixed layers in a transformer" work.)

**Verdict (verified, high confidence):** no source in the verified set does all three
of (a)+(b)+(c); the project's specific topology is **not pre-empted**. *Caveats:* the
close items are very recent 2025 preprints (fast-moving field); verified absence within
the searched set is not proof of global absence; the frozen-backbone-adapter cluster was
the weakest-covered corner (negative claims there were split/refuted). Several
search-returned IDs were unreliable / didn't resolve and were **discarded, not cited**.

## §5 — Endogenous / always-on / between-request computation  *(verified-absent in the searched set)*

The dedicated pass specifically searched "persistent / always-on stateful LLM agents
across independent calls." **No verified source** persists endogenous LLM state across
*independent* forward passes (a genuine cross-pass / between-request time axis, including
computation with **no input**) in the sense the Reservoir Agent's runtime intends — as
distinct from a KV cache, a context window, or external memory/RAG. This is reported as
*verified-absent within the searched set* (not proof of global absence); the
cross-pass / always-on setting is the project's least-pre-empted axis. (Open question
carried forward in `REVIEW.md`: whether recurrent-memory-transformer / SSM-agent /
test-time-training literatures, which these reservoir-focused searches would not surface,
contain a cross-pass-state system.)

---

## §6 — KV-cache management & compressed attention  *(motivates the base-model direction)*

Added 2026-06-01 from the imported Grok conversation
(`data_lake/transcripts/attention-reservoir-architecture-grok.md`). Relevance: a Reservoir
Agent injects persistent K/V every pass and runs blank ticks, so it burns context faster than
a turn-based model. Two families bear on this — *eviction* policies (what to keep in a fixed
cache) and *architecturally* compressed attention (a smaller cache to begin with). The first
we implemented over (`src/reservoir/kv_evict.py`, pinning the reservoir); the second motivates
moving the base off Hermes.

### Xiao et al. 2023 — "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM)
arXiv:2309.17453. <https://arxiv.org/abs/2309.17453>
- **Key claim:** keeping a few initial "attention-sink" tokens plus a rolling recent window
  lets a pretrained LLM stream over effectively unbounded input without fine-tuning or a
  catastrophic perplexity blow-up; the sink tokens absorb otherwise-misplaced attention mass.
- **Relation to RAN:** this is the exact template `kv_evict.py` follows — sink + recent window
  — with the project-specific addition that the reservoir's K/V entries are *pinned* so the
  time-axis is never evicted. With no reservoir tags our policy *is* StreamingLLM.

### Zhang et al. 2023 — "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"
arXiv:2306.14048. <https://arxiv.org/abs/2306.14048>
- **Key claim:** a small set of "heavy-hitter" tokens (high accumulated attention) dominates
  quality; scoring tokens by attention and evicting the rest keeps a small KV cache with little
  loss. An importance-based alternative to position-based (sink/recent) eviction.
- **Relation to RAN:** an importance signal the reservoir-protected policy could later use to
  rank *normal* tokens for eviction; orthogonal to (and combinable with) pinning the reservoir.

### DeepSeek-AI 2024 — "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (MLA)
arXiv:2405.04434. <https://arxiv.org/abs/2405.04434>
- **Key claim:** Multi-head Latent Attention (MLA) compresses keys/values into a low-rank
  latent vector, shrinking the KV cache by ~an order of magnitude vs MHA while preserving
  quality; DeepSeek-V2-Lite (16B total / 2.4B active, 27 layers) is the small open MLA model.
- **Relation to RAN:** the *architectural* cache-efficiency the chat argues gives the
  persistent reservoir headroom. DeepSeek-V2-Lite is the realistic local base to attempt
  reservoir injection on (feasibility spike queued); MLA is the mechanism we want under us.

### DeepSeek-AI — DeepSeek-V4-Flash model release (hybrid compressed attention)
Hugging Face model card <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash> (MIT, released
2026-04-24); 284B-total / 13B-active MoE, 1M context.
- **Key claim (as reported by the release / the imported chat, not independently verified
  here):** a hybrid attention stack interleaving Compressed Sparse Attention (CSA, moderate
  compression with learned top-k selection) and Heavily Compressed Attention (HCA, aggressive
  grouping of long history) to support 1M-token context at a fraction of prior KV-cache cost.
- **Relation to RAN:** the aspirational base — its learned compression is what the chat
  hypothesises could be fine-tuned to route long-idle "nothing happened" signal through the
  reservoir. **Not runnable locally** (284B won't fit on 8.6 GB even at 4-bit; injection needs
  fine-tuning, so a hosted API can't substitute). Tracked as a cloud/big-GPU destination. The
  specific CSA/HCA mechanism and compression ratios are from the release/chat and were not
  fact-checked against a peer-reviewed source.
