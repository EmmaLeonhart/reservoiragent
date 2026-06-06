# Response notes — clawRxiv post 2692 review (Gemini 3 Flash, "Reject", 2026-06-06)

The rating dropped from Weak Reject (2680/2685) to Reject, largely because the now
fully-documented scaling failure reads as "doesn't scale." Mapping each con to the response:

| # | Con | Class | Response |
|---|---|---|---|
| 1 | Cross-pass recall fails to scale past GPT-2-124M | Scientific limitation | Genuine and now thoroughly documented (four levers tried: curriculum, wider coupling, Qwen-0.5B, broad-LoRA unfreeze — all chance). Stated as the study's boundary; the fix (real backbone training / larger budget) is named future work, not GPU-available here. |
| 2 | Stateless baseline is a "strawman" | **Addressed (framing)** | Added a paragraph: the reset-reservoir baseline is an *ablation* isolating carried state, not a competitor; the non-trivial comparison is additive-vs-KV (both carry the reservoir, only injection differs). A trained memory-augmented transformer / RNN baseline named as future work. |
| 3 | Informal dev-log writing style | **Addressed (revision)** | Removed "Status: feasibility phase complete", "the user states plainly" (prior pass removed "honest/named plainly/brain surgery/cleanvibe research project"). |
| 4 | TC0/FO(M) purely motivational, no proof | Already scoped | The section already states it is motivation, not a result, with the arbitrary-precision caveat. No new claim. |
| 5 | No formal bibliography (refs point to local files) | **Addressed (revision)** | Added a formal `## References` section with the verified citations (arXiv ids from literature/sources.md) grouped by topic. |
| 6 | Safety section speculative / synthetic | Already caveated | The section already states the probe decodes a benign clock, not misalignment, and the numbers are synthetic-stream illustrations, not a finished safety case. |
| 7 | KV-append HuggingFace integration blocker | GPU/engineering | Real reproducibility limitation, already named; the fix is HF KV-cache plumbing, not a paper edit. |

Net change this revision: 2, 3, 5 addressed in FINDINGS; 1, 7 are real limitations stated as
such; 4, 6 already scoped. Resubmitted to supersede.
