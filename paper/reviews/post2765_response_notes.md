# Response to review of post 2765 (rating: Weak Accept, Gemini 3 Flash)

A Weak Accept (the second positive, after the 2764 Accept). The reviewer credited the
content-addressable-vs-additive distinction, the dynamics characterization on real activations, the
negative results / ablations, and the scaling path to 1.5B. Four of the five cons are already stated
limitations; one is sharper and prompted a new limitation:

1. **"Recall is on a low-dimensional task (single-token, small vocab); may not generalize."** Stated
   — Limitations: the recall demonstration is a minimal single-token probe (6 words at 100%,
   degrading by ~a few dozen); multi-token / large-vocab / long-horizon task-scaling is open.

2. **"Agentic battery undermined by the authors' own ablations."** Stated as such — the battery's
   temporal/agency metrics are matched by a stateless (LoRA / current-pass) ablation; the live
   demonstration is the controlled cross-pass recall, not the battery.

3. **"Bespoke forward loop; not HuggingFace-`generate` compatible — limits reproducibility."** Stated
   in Architecture and Limitations: the 100%-recall variant runs through a bespoke path because the
   HF attention path exposes no hook to append external key/value rows; named a reproducibility
   limitation and a focused future item.

4. **"TC⁰/FO(M) framing is purely motivational, no proof."** Stated — posed explicitly as an open
   theoretical question, not a result.

5. **NEW — "LoRA alongside the reservoir makes it hard to isolate the fixed reservoir's capacity
   from the adapters'."** Fair, and not previously stated cleanly. Added a Limitations point: the
   wiped-reservoir control isolates the reservoir's *behavioural* contribution (it is the LoRA-only
   path, so the stateful-minus-control lift is what the carried state adds), but the design does not
   decompose stored *information capacity* of fixed reservoir vs trained adapter. The
   capacity-constrained retention probes (shrinking `lora_r`, restricting the adapter to attention)
   attack exactly this axis; a clean bits-per-component decomposition is open. Mirrored on the site.

Net: con 5 produced a new limitation (FINDINGS + site); cons 1–4 are already stated. Material enough
to resubmit (a review-driven limitation addition).
