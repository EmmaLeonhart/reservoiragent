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
- **Transfer the cross-pass recall win to Hermes 3B (open; well-diagnosed).** 100% on
  GPT-2 but chance on Hermes across **4** attempts (4-bit ×2 + bf16+higher-LR + a dedicated
  **2000-step** 4-bit run); loss plateaus ~2.5–2.8 regardless of quantization. A gradient
  diagnostic confirms it is **NOT a bug** (state updates; ∇ flows to W_res + LoRA) — it is a
  bootstrapping/scale difficulty (the prefix signal is diluted through 28 layers vs GPT-2's
  shallow stack); GPT-2-medium (355M) fails the same way, so the wall starts well below 3B.
  **"Many more steps" is now RULED OUT** (2000 steps ≈6.7× still chance/2.49). Remaining
  routes are structural: a **curriculum** (key in-context first, anneal out), a stronger
  prefix coupling (inject the prefix at multiple layers, or larger n_prefix), or unfreezing
  more of the model. Substantial; needs real compute + a dedicated effort.
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

### Base model — moving off Hermes to a KV-efficient base (the Grok-chat direction)
Grounding: `data_lake/transcripts/attention-reservoir-architecture-grok.md`. The chat's case
is that an always-on Reservoir Agent burns context on blank ticks, so the base should be one
whose attention is *natively* KV-efficient (compressed cache) to give the persistent reservoir
headroom, and whose learned compression can be fine-tuned to lean on the reservoir for
long-idle signal. The KV literature is now in `literature/REVIEW.md` §1 (StreamingLLM, H2O,
MLA, the V4 CSA/HCA hybrid).
- **DeepSeek-V4-Flash — aspirational target, not local.** Real (284B-total / 13B-active MoE,
  1M context, hybrid CSA+HCA attention, MIT, released 2026-04-24). Reservoir injection needs
  fine-tuning, and 284B params won't load on the 8.6 GB RTX 4070 even at 4-bit (~140 GB), and
  a hosted API can't expose mid-layer attention for injection. So this is a **cloud / big-GPU**
  destination, tracked as direction, not a near-term step.
- **DeepSeek-V2-Lite — DROPPED (user, 2026-06-01).** Reason: V2-Lite has MLA (fixed low-rank
  KV compression) but **not the *learned, fine-tunable* compression** the user actually wants —
  the ability to fine-tune the cache manager so it learns to defer to the reservoir for
  long-idle signal. That learned compression is **DeepSeek Sparse Attention (DSA, V3.2)** /
  **CSA+HCA (V4)**, which only exists at frontier scale: V3.2 = 671B, V4-Flash = 284B — **no
  runnable-size open model has it**. **RESOLVED (user, 2026-06-01): drop the learned-compression
  angle for local work.** Local hardware can't have learned compression, so the local cache
  story is the **reservoir-pinned eviction** (`src/reservoir/kv_evict.py`, StreamingLLM + H2O
  with the reservoir pinned) — it bounds blank-tick context burn without needing a learned
  compressor. Stay on GPT-2/Hermes locally. The **learned-compression × reservoir** hypothesis
  (does fine-tuning a DSA/CSA-HCA compressor teach it to defer to the reservoir for long-idle
  signal?) becomes an explicit **cloud/future experiment** — only runnable by renting GPUs for
  V4-Flash/V3.2, deferred until there's a compute budget and a reason to commit.
  _(Config-only V2-Lite analysis retained below for if a small fixed-MLA base is ever wanted.)_
- **DeepSeek-V2-Lite — config-only analysis (retained, not the plan).** 16B total / 2.4B
  active, MLA, MIT. **Feasibility analysis done 2026-06-01 (config-only, no weight download):**
  transformers 5.4.0 supports `deepseek_v2` **natively** (no `trust_remote_code`); config
  confirms 27 layers, hidden 2048, 16 heads, MLA `kv_lora_rank=512` / `qk_rope_head_dim=64` /
  `qk_nope_head_dim=128` / `v_head_dim=128` (queries uncompressed in Lite: `q_lora_rank=None`),
  MoE = 64 routed experts (6 active) + 2 shared, `first_k_dense_replace=1` (layer 0 dense, 1–26
  MoE). **Mid-layer injection point = layer 13 of 27.** VRAM: 16B at 4-bit ≈ 8 GB of weights
  vs the 4070's 8.6 GB → a pure-GPU load is at/over the edge; **`device_map="auto"` + CPU
  offload is required** (763 GB disk, ample RAM). The `kv_live.py` prefix-injection mechanism
  is architecture-agnostic (works through `inputs_embeds` + the causal path, no MLA surgery),
  so the port is bounded: (a) a `_arch.py` `deepseek_v2` branch (`decoder_blocks =
  model.model.layers`, `hidden_size` from config); (b) LoRA `target_modules` set to the MLA
  projection names (`q_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` / `o_proj`) instead of GPT-2
  `c_attn`; (c) `layer=13` for the reservoir read hook. **Remaining (resource-gated):** the
  actual ~9 GB 4-bit download + load + a QLoRA-fit test — the one thing the analysis can't
  settle is whether QLoRA training fits in 8.6 GB with offloaded experts (likely tight; only
  a real attempt resolves it). Queued as the next local step.
- **Reservoir-friendly compression interaction (research question).** If the base has learned
  KV compression (CSA/HCA), does fine-tuning teach it to route long-idle "nothing happened"
  signal through the reservoir and compress raw blank tokens harder? The chat's hypothesis;
  only testable once a compressed-attention base actually runs. Pair with the reservoir-pinned
  eviction policy already built (`src/reservoir/kv_evict.py`).
- **Study, don't fork, the Hermes/Nous agent training.** Adapt their trajectory/data-generation
  ideas to the simpler local harness rather than forking the full stack (the user's call); the
  reward/scaffolding don't transfer cleanly to a reservoir agent.

### Long-term (aspirational — compute-gated, full vision)
- **N-seed selection pipeline.** 10–20 reservoir seeds × LoRA fine-tunes, keep the
  best by benchmark. Investigate a dynamics-based seed pre-selection proxy.
  - **Controlled result (2026-06-02): at 250 steps, selection is NOT significant** (ANOVA
    F=1.30, df 5,18, p=0.31, over 6 seeds × 4 controlled runs; within-seed init spread ≈
    between-seed). The trainable init matters more than the reservoir at this budget. See
    `FINDINGS.md` + `docs/controlled.png`; runner `scripts/run.py controlled`.
  - **Open follow-up:** does selection become real at a far larger training budget (e.g.
    `controlled --steps 1500`, 6×4 ≈ 1 hr local) where init noise shrinks? Either finds a real
    reservoir signal at higher budget or strengthens the negative. Bounded; awaiting interest.
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
- **Citation-checked novelty follow-up — DONE (2026-05-30).** Searched-prior-art
  sweep recorded in `literature/novelty_recheck.md`; claim positioned against
  Titans (test-time memory, arXiv 2501.00663) and the 2025 reservoir-transformer
  line. Standing caveat: re-run the sweep before any hard novelty claim in a
  *submitted paper*.
- **Theoretical open question (above): does finite-precision cross-pass reservoir
  state provably lift TC⁰/FO(M)?** Pose precisely; do not assert.
- **Does the edge-of-chaos prior survive the high-dimensional attention-input
  regime?** Empirical, answered by the dynamics sweep.

## E. The report (the deliverable shape)
- `FINDINGS.md` (question → method → results → limitations) kept current as results
  land; the themed `docs/` GitHub Pages site + built `report.pdf`. Capability first,
  mechanism second, formal claims third (scoped), organism framing last. Re-theme the
  recovered `data_lake/diagrams/*.svg` (or redraw) for the published site.

### Phase H remaining (after the A–E core)
- **Full Nous Hermes harness fork.** The Hermes-format core is built
  (`src/reservoir/hermes_harness.py`); the full fork needs: streaming + the exact Nous
  system-prompt scaffolding/stop strings; fusing the unprompted/idle pass + the trained
  silence gate into the loop (the always-alive behaviour, using `runtime.py` + `silence.py`);
  a regression suite vs vanilla Hermes (tool-call formatting unchanged with the reservoir
  injected — H1 at generation level, a Hermes GPU run); multi-tool routing + error recovery.
  See `HERMES_HARNESS_REMAINING`.
