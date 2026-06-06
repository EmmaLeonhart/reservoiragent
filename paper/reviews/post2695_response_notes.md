# Response notes — clawRxiv post 2695 review (Gemini 3 Flash, "Reject", 2026-06-06)

Review of the capacity-sweep + baseline-reframing version. Rating held at Reject, but the
**pros grew** — it now credits the stateless baseline as "rigorous" (the prior "strawman"
objection is resolved) and the interruptibility con dropped. Remaining cons map:

| # | Con | Class | Response |
|---|---|---|---|
| 1 | Doesn't generalize past GPT-2-small | Scientific limitation | Real, thoroughly mapped (four levers + capacity sweep). The scale fix is the heavy backbone-training route (future work). |
| 2 | Reservoir undersized (512–1024 vs 1536 input) → dimensionality collapse → content-task failure | Already documented | This is the paper's own diagnosis (the reservoir-expansion section). The 8192-node run was tried and content still didn't recover, so undersizing is part of it but not the whole fix — already in FINDINGS. |
| 3 | TC0/FO(M) speculative, no proof | **Addressed (trim)** | Flagged in all five reviews. Condensed the expressivity paragraph to clearly-labeled "motivation only, explicitly not a result", removed the formal-sounding exposition, added a "skip to results" pointer. |
| 4 | Informal phrasings ("yell at it to stop", "down-payment", "compute-gated") | **Addressed (register)** | Reworded the interruptibility motivation (no "yell"), "down-payment"→"step toward", "compute-gated"→"compute-limited" throughout; smoothed a redundant intro line. |
| 5 | KV-append HuggingFace blocker | GPU/engineering | Real, already named. |
| 6 | Always-alive runtime untrained / vision unproven | Acknowledged limitation | Already stated the harness runs the untrained substrate; a trained-policy eval is compute-limited future work. |

Net: cons 3 and 4 addressed; 1/5/6 are genuine limitations stated as such; 2 already in the
paper. The presentation objections are now essentially cleared across reviews — the standing
"Reject" is driven by scientific scope (scale + trivial tasks), which only the heavy
full-fine-tune route moves.
