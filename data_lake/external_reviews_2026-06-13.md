# External AI peer-review intake — 2026-06-13

Seven external chatbot conversations were saved to `chats/` (untracked; web-export
asset dirs are gitignored). Each is a pre-arXiv feedback pass on the RAN paper
(`FINDINGS.md` + `docs/index.html`). Below is the consolidated, deduplicated digest
extracted from all seven, with cross-reviewer consensus noted. Source threads:
Claude, ChatGPT, Meta AI, DeepSeek, Grok, Perplexity, Google Gemini.

## High-consensus substantive items (flagged by 3+ reviewers)

1. **Disambiguate the two "recall" measurements** (Gemini, DeepSeek, Perplexity, ChatGPT).
   A PDF-only reader conflates two different tasks both called "recall":
   - the **secret-word recall probe** scales to Qwen-1.5B (stateful 0.83–1.00 vs 0.17
     control) — genuine cross-pass content recall;
   - the **8-task battery's symbolic content recall** stays at floor (0.00) at 1.5B
     except under the deny-capacity + decay-LR retention recipe.
   These are different experiments; the abstract/§1 must make that explicit so it does
   not read as contradicting Appendix A. **NOT a false scaling claim — verified the
   probe scaling is real; the fix is disambiguation.**

2. **Tone down overclaiming in title/abstract/intro** (Perplexity, ChatGPT, Gemini):
   frame as a feasibility study of a *mechanism*, not a general architecture for
   "persistent agency"/always-alive agency. Surface GPT-2-medium + Hermes-3B failures
   up front, not just in limitations.

3. **Cut / temper the organism analogy** (ChatGPT: cut; Meta, Perplexity: soften):
   it sells AGI implications before the engineering result is established.

4. **Compress / caveat the complexity-theory (TC⁰/FO(M)) section** (ChatGPT, Perplexity:
   shrink; Gemini: lean harder into the "no proof a finite-precision reservoir lifts the
   per-pass bound" caveat). Keep as motivation, not result.

5. **Abstract presentation**: break the dense 4-result block into shorter sentences/
   bullets, lead with the injection finding, stay conservative (Meta, Grok, Perplexity,
   ChatGPT).

## Blockers (figures / correctness)

6. **Embed all figures in the PDF; remove the external-URL reliance** (Meta, DeepSeek,
   Grok). FINDINGS.md:267-268 and :1330 still carry
   "All figures referenced below are in the accompanying report: https://reservoir.emmaleonhart.com".
   Verify the built PDF is self-contained (devlog says figures already bound — confirm,
   then remove/soften the URL line so an arXiv reader is not sent off-site).

7. **Harmonize the reservoir update equations** (Gemini): §4 (line 114) gives the plain
   `r(t)=tanh(W_r·r(t−1)+W_in·x(t))` but §6 (line 229) introduces the leaky-integrator
   form with leak rate `a` without defining it earlier. Introduce the leak-rate form
   where the equation first appears.

8. **Fix typo `np8` → `n_prefix=8`** (Grok). FINDINGS.md:571 table row.

9. **Verify the Block-Recurrent Transformer quote** "learns to ignore the recurrent
   state" is exact; if paraphrased, soften to "documents a failure mode where…"
   (Meta, ChatGPT).

10. **Spot-check the 2025 arXiv citations** (Echo State Transformer 2507.02917, Echo Flow
    Networks 2509.24122, Frozen-in-Time 2508.18130) for publication/revision status
    (Claude).

## Improvements

11. **State the GPT-2-medium failure as a genuine non-monotonicity / honest negative** —
    no input scaling in [0.05, 1.0] lands in its sweet spot (DeepSeek, Perplexity, ChatGPT).
12. **Move the adapter-capacity control (lora_r=4, attention-only) next to the recall
    claim** so the control proving carried-state-not-adapter-memory sits with the result,
    not buried in Limitations (DeepSeek).
13. **Shorten the §11 safety section ~40%; move the "interruptibility is not
    reservoir-specific" caveat earlier**; reframe the real safety point as the fixed
    reservoir being a cheap monitoring surface (DeepSeek).
14. **Shorten the Declaration of AI use to 2–3 sentences or move to an appendix** (DeepSeek).
15. **Add a limitations sentence on the HF `generate` integration gap**: standalone
    forward loop works; merging into HF is engineering, not research (Meta, ChatGPT).
16. **Temper the single-run battery-retention result** with "(multi-seed pending)" /
    "seed-robustness untested" (Meta, Grok, ChatGPT).
17. **Harmonize input-scaling notation** "1/4–1/10" vs "one-quarter to one-tenth" — pick
    one (Meta, ChatGPT).
18. **Replace vague "degrading by ~a few dozen"** with a concrete figure from the curve,
    e.g. "to chance by ∼30 items" (Meta, ChatGPT).
19. **Standardize "content-addressable KV-prefix" vs "KV-append" terminology**, or define
    the distinction once early (Grok, Perplexity).
20. **Surface the "why an untrained reservoir?" rationale in the intro** (Meta, ChatGPT):
    fixed reservoirs isolate whether untrained dynamics suffice; trained recurrence is
    complementary future work.
21. **Add a Results summary table** of injection variants + scaling configs (Grok).
22. **Formally define the experimental tasks before invoking them** (secret-word recall,
    silence policy, timed, self-init) (Gemini).
23. **Replace dev jargon** ("regression test confirming the hook is correctly placed",
    "the released code") with academic phrasing (Gemini).
24. **Detail the KV-append hook / bespoke forward loop** for reproducibility (Gemini).
25. **Remove the empty "## Figures" Section 13 stub** (FINDINGS.md:1254) or fill it (Claude).
26. **Fig 13 caption should carry the `eval_n=48` rationale** (chosen over 16 to avoid
    quantization artifacts) (Claude).
27. **Reframe the "controlled negatives" as method boundaries**, not scattered failures
    (Perplexity).
28. **Remove duplicated claims** across Abstract / Contributions / Scope / Results
    (Perplexity).
29. **Replace loaded phrases** "genuine time axis" / "organism-like" with neutral wording
    where not essential (Perplexity).

## Small / mechanical

30. Page-2 "reproducible" appears twice in one sentence — trim one (DeepSeek).
31. Page-13 table: the control column reads "0.17" for two adjacent rows — check for a
    missing column separator (DeepSeek).
32. "(flagged in review)" → "intentionally minimal" (arXiv has no submission reviewers)
    (DeepSeek).
33. "This project follows a guiding rule…" → "we adopt…" (Meta, ChatGPT).

## Optional (not required for arXiv)

- Move secondary material (battery, timing, organism, seed-selection) to appendices to
  shrink the core to ~12 pages (ChatGPT, Grok — only "if submitting beyond arXiv").
- Add discovery keywords (echo state networks, stateful transformers, reservoir
  computing, cross-pass memory; cs.CL/cs.LG/cs.AI) (Meta, Grok).
- Add a sentence acknowledging the "is the reservoir necessary vs any persistent latent
  vector?" reviewer attack (ChatGPT).
- Frame the tens-of-items capacity ceiling as expected fading memory, not a defect (Grok).

**Reviewer consensus: the paper is submittable to arXiv; the only true gate is
self-contained figures in the PDF. Everything else is quality.**
