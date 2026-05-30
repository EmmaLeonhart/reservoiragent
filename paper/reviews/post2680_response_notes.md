# Response notes — clawRxiv post 2680 review (Gemini 3 Flash, "Weak Reject")

Working note, not a published artifact. Maps each reviewer **con** to where the
current `FINDINGS.md` already addresses it, and splits the possible responses
into *already-covered*, *GPU-gated*, and *product-decision-for-the-user*. The
actual decision on whether/how to revise and resubmit is reserved for the user
(todo.md C2 is a "product decision per review"); this note is only the
groundwork so that call is quick.

Review file: `paper/reviews/post2680_review2680.json` (rating: Weak Reject;
model: Gemini 3 Flash; 2026-05-30). The reviewer's **pros** explicitly credit
the paper's transparency about the Hermes-3B failure — so the strategy below is
to make the already-present acknowledgements *more visible*, not to add new
claims.

## Con-by-con map

Verified against the committed `FINDINGS.md` sections (headings, via Grep —
line numbers omitted because tool output was unreliable this session):
`## Complexity-theoretic motivation`, `## Limitations and what would falsify the
claims`, `## What this is and isn't`.

| # | Reviewer con | Already in FINDINGS? | Class | Possible response |
|---|---|---|---|---|
| 1 | Primary claim only on GPT-2; Hermes-3B fails (signal dilution) | Yes — Limitations: GPT-2 scale only for the recall result (diagnosed negative, mechanism verified-wired, not claimed at 3B) | **GPU-gated** (A1) | No paper change needed to be accurate. A *fix* (converge at 3B) is the kickoff-cron GPU thread. |
| 2 | Tasks trivial, not agentic reasoning | Yes — Limitations: tasks are minimal probes; `## What this is and isn't` frames the whole study as feasibility+dynamics, not an agentic demo | Already-covered (framing) | Optional: sharpen the abstract/intro so the minimal-probe/feasibility scope is unmissable before a reader reaches Limitations. Non-GPU, low-risk, but a wording **product decision**. |
| 3 | TC0/FO(M) claim speculative, unproven | Yes — `## Complexity-theoretic motivation` states it is motivation, does **not** claim a finite-precision reservoir escapes TC0, proof left open; restated in Limitations | Already-covered | No change needed for accuracy. A real separation proof is research-open, not bounded. |
| 4 | KV-append HuggingFace integration blocker limits utility/repro | Yes — Limitations: KV-append HuggingFace blocker reported, not hidden | **GPU/engineering** | Genuine fix = integration work (HF KV-cache plumbing); not a paper edit. |
| 5 | N-seed proxy weak correlation -> expensive trial-and-error | Yes — Limitations: N-seed proxy is weak; N-seed section reports the weak correlation | **GPU-gated** (B1, "better predictor") | No change needed for accuracy. A better predictor is the B1 GPU thread. |

## Summary for the product call

- **Nothing in the review contradicts the paper.** Every con is an already-stated
  limitation; the reviewer's complaint is about *substance* (scale, task
  difficulty, an unproven bound), not about hidden or wrong claims.
- **All substantive fixes are GPU-gated or research-open**, not autonomously
  doable: Hermes-3B convergence (A1), a stronger N-seed predictor (B1), HF
  KV-append integration, and any TC0 separation proof.
- **The only non-GPU lever is framing**: optionally pull the
  minimal-probe/feasibility scope earlier and louder (abstract + intro) so the
  Weak-Reject's "tasks are trivial" reading is pre-empted. This is a wording
  change to the *published* paper -> **left for the user** (does it, triggers a
  `/revise` resubmission, accrues a fresh review).
- **Recommended next step when the user is ready:** decide between (a) leave as-is
  and let the GPU thread produce a stronger result before resubmitting, or
  (b) do the framing-only revision now to convert the Weak Reject's scope
  objection into an explicit, up-front scope statement. Option (b) is ~30 min of
  editing FINDINGS.md (keeping `TITLE`/`ABSTRACT` in
  `scripts/submit_clawrxiv_paper.py` in sync) + a `workflow_dispatch` of
  `clawrxiv.yml`.
