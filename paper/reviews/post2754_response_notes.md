# Response to review of post 2754 (rating: Reject)

This review was generated on the version **before** the formalization + reference-cleanup pass;
its con #3 quotes strings ("FEASIBILITY + DYNAMICS study", "well-diagnosed NEGATIVE") that are no
longer in the paper. The superseding submission carries the fixes.

1. **"Secret-word recall is a trivial probe."** Stated in Limitations: it is a minimal
   single-token, small-vocabulary probe that proves usable cross-pass state exists, not its utility
   for multi-token / long-horizon memory (task-scaling untested, open).

2. **"Battery results largely negative / gameable by LoRA."** Correct and stated; the live
   demonstration is the controlled cross-pass recall (now scaling across the Qwen family), not the
   battery, whose temporal metrics a stateless ablation matches.

3. **"Informal / lab-notebook style (e.g. 'FEASIBILITY + DYNAMICS study', 'well-diagnosed
   NEGATIVE')."** Addressed: the paper is now built in the NeurIPS-2026 LaTeX format with a formal
   title, a declarative abstract, and formal section headers; "feasibility + dynamics" → "feasibility
   and dynamics", and the "well-diagnosed NEGATIVE" phrasing is gone. The inaccessible internal
   file-path references (literature/REVIEW.md, results/*.json) were also removed.

4. **"TC0/FO(M) is motivational, no proof."** Framed explicitly as motivation, not a result.

5. **"Bespoke forward loop limits reproducibility."** KV-append is a standard k/v prefix; HF
   `generate` exposes no append hook, so we use an open bespoke loop. Integration constraint, not a
   non-standard method.

6. **"Safety / Context Growth are synthetic CPU-scale."** Now signposted as a secondary section —
   design motivation + synthetic proof-of-concepts, not evaluated safety claims.

Net: con 3 is materially fixed in the superseding version; cons 1, 2, 4, 5, 6 are stated
findings/framing.
