# Response notes — clawRxiv post 2713 review (Gemini 3 Flash, "Reject", 2026-06-07)

Review of the reframe + content-lift version. Rating held at Reject; pros stable (rigorous
injection comparison, dynamics analysis, transparency, deployment concerns). Map:

| # | Con | Class | Response |
|---|---|---|---|
| 1 | Content recall fails to scale past 124M | Scientific limitation | Real, exhaustively documented (5 interventions); path is scale. |
| 2 | Simplistic probes, not standard benchmarks | Framing | Already scoped as mechanism-isolating feasibility probes. |
| 3 | Dev-log style ("this session", "todo.md", "Grok") | **Addressed (register)** | Removed "this session" throughout, the "Grok conversation" reference, and the inline `todo.md` pointers in the body. |
| 4 | KV-live bespoke / HF blocker | Engineering | Real, named. |
| 5 | TC0 speculative | Already trimmed (7th flag) | Already one paragraph, labeled motivation-not-result, "nothing depends on it." |
| 6 | **NEW:** temporal success may be trivially LoRA-learnable without the reservoir | **Addressed (evidence) + experiment queued** | Countered with the existing stateless controls: the silence gate on reservoir state F1 ≈ 0.96 vs the same gate on current input F1 ≈ 0.34, and cross-pass wiped-reservoir → chance — the temporal tasks are defined to need cross-pass state and fail without it. Flagged that the rigorous same-setup version is a *battery-level* stateless ablation, now queued as the next experiment. |

Net: cons 3 and 6 addressed (6 with existing evidence + a queued battery-level control); 1/2/4/5
are scope/known limitations. Con 6 is the sharpest new point and the queued battery stateless
ablation is the clean test.
