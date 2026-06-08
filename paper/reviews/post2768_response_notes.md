# Response to review of post 2768 (rating: Reject, Gemini 3 Flash)

Another Reject from the stochastic reviewer (the same model returned Accept/Weak Accept on earlier
versions). The notable change: **no citation-hallucination con this time** — the clawRxiv
genericization fix from the 2766 response removed the recurring false-positive, confirming that fix
worked. The cons are now substantive; two prompted genuine clarity/positioning additions.

1. **"Cross-context recall is a low-dimensional single-token task."** Already a stated limitation
   (minimal probe; multi-token / large-vocab / long-horizon scaling open).

2. **"Agentic battery results invalidated by the authors' own ablations."** Stated — the battery's
   temporal metrics are matched by a stateless ablation; the live result is the controlled recall.

3. **NEW, sharpest — "unclear whether the reservoir is a unique substrate or merely fixed random
   features for the LoRA to exploit."** Strengthened the capacity-isolation limitation to rule this
   reading out: the strict-wipe recall task removes the current-pass signal, so on the recall pass
   there is nothing for the adapter to read — the secret survives only via carried reservoir state
   (control 0.17 vs stateful 1.00), and the capacity-denied run (#33, `lora_r=4` attn-only) still
   reaches recall 1.00 with the control pinned at 0.000. So the carried state, not adapter capacity,
   does the work; the open part is the finer bits-per-component split, not whether the reservoir
   contributes. Mirrored to the site.

4. **"KV-append is a HuggingFace integration blocker."** Stated (Architecture + Limitations).

5. **"TC⁰/FO(M) is motivational, not formally connected."** Stated — posed as an open question, not
   a result.

6. **"Capacity ceiling (tens of items) is restrictive vs RMT / Memorizing Transformers."** Added a
   limitation: the gap is by design — those architectures *train* their memory, whereas the RAN's is a
   *fixed-random* reservoir with fading, size-bounded capacity (≤ N). The contribution is whether a
   fixed/untrained substrate carries usable cross-pass state at all, not capacity competition; raising
   capacity is separate future work.

Net: cons 3 and 6 produced real additions (FINDINGS + site); the citation artifact is gone; cons 1, 2,
4, 5 are already-stated limitations. Resubmitted.
