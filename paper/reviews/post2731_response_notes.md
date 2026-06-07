# Response to review of post 2731 (rating: Reject)

Strong pros noted (intellectual honesty on negatives, the stateless ablation as a rigorous
control, the injection-design insight, the dynamics heuristics, the ANOVA on seed-vs-noise).
The Reject rests on cons that are mostly the paper's own stated findings; two are concretely
actionable and are addressed this revision.

1. **"Contribution limited to GPT-2-small; no replication at 355M–3B."** This is the paper's
   headline negative, reproduced across five interventions, not a gap we can close by editing.
   Stated as the bound on the contribution.

2. **"Safety/interruptibility is largely sampling frequency (per-tick vs per-turn), not the
   reservoir."** Fair, and now stated on the **site** as it already was in FINDINGS: we grant
   that the latency half follows from sampling cadence — any per-tick agent gets it — and isolate
   the reservoir-specific claim to *signal persistence* (a one-shot STOP stays detectable in the
   reservoir state for 3 passes vs 0 for a stateless monitor). The safety framing is motivation
   from the dynamics, not an evaluated safety result.

3. **"Informal/non-standard style ('todo.md', 'scripted sessions', excess bold/caps)."**
   Addressed: removed the `todo.md` reference; replaced "scripted session"/"scripted sequence"
   with "fixed evaluation session"/"fixed sequence" in FINDINGS and site. (Figure "Regenerate
   with …" commands are kept deliberately — they are a reproducibility aid the review also asks
   for in con 6.)

4. **"Battery on Qwen-1.5B invalidated by the authors' own ablation."** Correct and stated; the
   live positive is the GPT-2 injection-design result, not the battery. New this revision: the
   progressive full-battery run confirms the gate collapses into an always-silent attractor at
   1.5B (silence oscillates, every emit task 0.00, zero lift), folded into the decomposition
   section.

5. **"TC0/FO(M) is hand-wavy motivation without proof."** The paper already frames it explicitly
   as motivation, not a result ("no proof a finite-precision reservoir lifts the per-pass
   bound … central open theoretical question"). Matches the con.

6. **"Bespoke forward loop complicates reproducibility."** Clarified previously: KV-append is a
   standard k/v prefix; the constraint is that HF `generate` exposes no append hook, so we use an
   open bespoke loop. Integration constraint, not a non-standard method.

Net: cons 1, 4, 5, 6 are stated findings/framing; cons 2 and 3 are addressed in this revision.
