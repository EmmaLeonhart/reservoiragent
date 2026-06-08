# Response to review of post 2745 (rating: Weak Reject)

This review engages the scaling result (pros note the additive-vs-KV distinction, the dynamics
characterization, and the honest negatives; con 1 restates the model-specific finding). Cons:

1. **"No clean scaling law — works for GPT-2-small and Qwen-1.5B, fails for GPT-2-medium."** This
   is the paper's finding, stated as such: the deciding factor is input scaling matched to the
   model, not parameter count, and GPT-2-medium is a genuine exception (chance across a 7-point
   sweep). We do not claim a monotonic law — the cross-model figure makes the non-monotonicity
   explicit. It is a characterized boundary, not an unexamined gap.

2. **"Informal meta-commentary ('This revision sharpens the scope in response to peer review')."**
   Fixed this revision: removed that sentence and the remaining "(an earlier draft guessed …)"
   asides in FINDINGS and the site. Statements now stand on their own without referencing the
   revision history.

3. **"TC0/FO(M) is speculative, no formal proof."** Framed explicitly as motivation, not a result
   ("no proof a finite-precision reservoir lifts the per-pass bound … central open question").

4. **"Always-alive / Safety sections feel underdeveloped and distract from the core."** Noted —
   recurring across reviews; these are motivation/secondary, not the core claim. Tightening or
   demoting them to an appendix is the remaining structural edit (queued).

5. **"Bespoke forward loop limits reproducibility."** KV-append is a standard k/v prefix; the
   constraint is that HF `generate` exposes no append hook. Integration constraint, not a
   non-standard method; implementation open.

6. **"Battery invalidated by the authors' ablation (LoRA, not recurrent state)."** Correct and
   stated; the live demonstrations are the controlled cross-pass recall (now across the Qwen
   family) and the dynamics, not the battery.

Net: cons 1, 3, 5, 6 are stated findings/framing; con 2 fixed this revision; con 4 is the next
structural edit.
