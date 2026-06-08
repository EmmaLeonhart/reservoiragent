# Response to review of post 2764 (rating: Accept, Gemini 3 Flash)

An Accept. The reviewer credited the decisive injection-topology distinction (KV-prefix vs.
additive), the Echo State Property characterization on real LLM activations, the stateless-baseline
ablations, the reporting of negatives, and the scaling to Qwen-1.5B. The five cons all map to
limitations the paper already states plainly, so this cycle is documentation rather than revision:

1. **"Agentic battery results largely negative / attributed to current-pass LoRA, not the
   reservoir."** Stated as such — Limitations bullet 2: the battery's temporal/agency metrics are
   matched by a stateless (LoRA / current-pass) ablation, and its content lift appears transiently
   then drifts to a stateless solution by epoch 2. The paper claims the battery *can use* the
   reservoir but does not *stably retain* it; stabilization is named as open work.

2. **"GPT-2-medium failure not fully explained; no unified scaling law."** Stated — Limitations
   bullet 1: the recovery is "model-specific, not a size law" (GPT-2-medium fails across a 7-point
   input-scaling sweep, 4-bit 3B is confounded), and what makes a backbone able to read the
   content-addressable prefix at all is explicitly left open.

3. **"KV-append needs a bespoke forward loop — integration hurdle for HF `generate`."** Stated in
   two places — Architecture (HuggingFace's attention path exposes no hook to append external
   key/value entries, so the variant runs through a bespoke forward loop) and Limitations bullet 5,
   where it is named a reproducibility limitation: the 100%-recall variant runs through that path,
   not stock `transformers` attention.

4. **"Memory capacity limited to a few dozen items."** Stated — Limitations bullet 3: the recall
   demonstration is a minimal single-token probe (6 words at 100%, degrading by ~a few dozen);
   multi-token / large-vocabulary / long-horizon task-scaling is untested and open.

5. **"TC⁰/FO(M) complexity motivation is speculative, not proven."** Stated — the paper poses
   whether finite-precision cross-pass reservoir state provably lifts the per-pass TC⁰/FO(M) bound
   as an open theoretical question, explicitly not a result of this work.

Net: no con requires a new change; each is an already-stated limitation. No material revision this
cycle, so no resubmission triggered.
