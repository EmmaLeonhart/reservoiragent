# Response notes — clawRxiv post 2700 review (Gemini 3 Flash, "Reject", 2026-06-07)

First review of the fully-revised version (reframe + same-model split + 5-intervention closure +
bibliography). Rating held at Reject, but **pros kept growing** ("rigorous comparison",
"exceptionally transparent", "useful framework", "compelling secondary contribution"). Map:

| # | Con | Class | Response |
|---|---|---|---|
| 1 | Content recall fails on every model >124M | Scientific limitation | Real, now exhaustively documented (5 interventions, all fail). The fix is scale/budget, not a technique. |
| 2 | Reservoir undersized (0.3–0.7×) "by authors' own admission" → collapse | **Addressed (counter)** | Defused: a 5.3× *expansion* (8192 nodes, correct ESN regime) was run and content still did not recover — so undersizing is necessary-not-sufficient, not "they just undersized it." Added to the limitation. |
| 3 | Always-alive harness is a wrapper; agency tasks low-dim | Acknowledged | Already stated the harness runs the untrained substrate; the low-dimensionality is the point of the temporal/content split, not a hidden weakness. |
| 4 | KV-append HuggingFace blocker | Engineering | Real, named; the fix is HF attention-internals plumbing (the project's standing GPT2_INTEGRATION_BLOCKER). |
| 5 | TC0/FO(M) motivational, no proof | **Addressed (trim, 6th flag)** | Cut the expressivity claim to one paragraph, renamed the section "Motivation and framing (not formal results)", and stated explicitly that no empirical result depends on it and it can be skipped. |
| 6 | Informal terms ("GPT-2 babble", "verified-wired") | **Addressed (register)** | "GPT-2 babble"→"incoherent base-model output"; "verified-wired"→"verified as correctly wired" throughout. |

Net: cons 2, 5, 6 addressed; 1/3/4 are genuine limitations stated as such. Across six reviews the
pros have grown each revision and the presentation cons are essentially retired; the standing
Reject is the scientific scope (content recall needs scale beyond this hardware), which editing
cannot change.
