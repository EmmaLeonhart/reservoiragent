# Response notes — clawRxiv post 2694 review (Gemini 3 Flash, "Reject", 2026-06-06)

This is the review of the version *with* the formal bibliography, baseline reframing, and the
GRU baseline. Rating held at Reject — and the cons are now almost entirely the **substantive
scientific limitations**, not the presentation gaps (which the prior revision fixed). Map:

| # | Con | Class | Response |
|---|---|---|---|
| 1 | 100% recall limited to GPT-2-small; fails to scale → generalizability | Scientific limitation | Real and now thoroughly mapped (four levers tried, all chance). Added a capacity sweep showing the win is *not* a 6-word artifact (0.92 at 24 keys), while being clear it is budget-limited at higher vocab. The scale fix is the heavy backbone-training route (future work). |
| 2 | Tasks too simple for the "agentic" framing | Framing | The study is already scoped as feasibility+dynamics with minimal mechanism-isolating probes, not an agentic demo; the "Agent" terminology names the architecture instantiation, not a demonstrated capability. No new claim. |
| 3 | Interruptibility advantage is trivial (sampling frequency, not the reservoir) | **Addressed (concession)** | Granted in-text: the latency half *is* per-tick sampling, available to any per-tick agent; only the signal-*persistence* half (a one-shot STOP surviving several passes in reservoir state) is reservoir-specific. |
| 4 | KV-append HuggingFace blocker hinders reproducibility | GPU/engineering | Real, already named; the fix is HF KV-cache plumbing, not a paper edit. |
| 5 | TC0/FO(M) speculative, no proof | Already scoped | Flagged in all four reviews; already stated as motivation-not-result with the arbitrary-precision caveat. Candidate for further trimming/relegation if the rating depends on it. |
| 6 | Always-alive harness lacks evaluation on non-trivial tasks (a "wrapper") | Acknowledged limitation | Already stated that the harness runs the *untrained* substrate and demonstrates the loop, not a trained policy. A trained-policy evaluation is compute-gated future work. |

Net this revision: capacity sweep added (con 1 robustness); interruptibility concession (con 3).
Cons 1/4/6 are genuine limitations stated as such; 2/5 already scoped. The recurring pattern
across four reviews: the presentation is now adequate, and the remaining objection is the
scientific scope (scale + task difficulty), which only the heavy full-fine-tune route changes.
