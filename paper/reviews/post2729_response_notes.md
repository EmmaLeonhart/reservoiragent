# Response to review of post 2729 (rating: Weak Reject)

Reviewer cons and how the paper now stands on each.

1. **"Fails to scale; cross-pass recall works only on GPT-2-small, fails at 355M+/Qwen/Hermes."**
   Accurate, and it is the paper's own headline result — a clearly-bounded scaling
   negative, reproduced across five interventions at 355M+. We do not claim scale; we
   localize the wall. Not a defect to fix, a finding to report.

2. **"Agentic demos largely debunked by the authors' own ablations (stateless+LoRA matches)."**
   Correct, and stated as such: the battery's temporal scores are not reservoir-driven
   (the stateless ablation matches), so the temporal claim was retracted. The remaining
   live positive is the GPT-2 injection-design result, not agency. The current
   silence_weight / progressive runs are aimed exactly at this gap — whether *any* of the
   temporal behaviour can be made reservoir-attributable; reported with the inline
   stateless control either way.

3. **"Informal/fragmented register ('well-diagnosed NEGATIVE', 'not a bug', 'operational
   pain point')."** Addressed this cycle: replaced "well-diagnosed" with "well-characterized"
   in FINDINGS and both site instances; rewrote the Hermes/KV limitation bullet in standard
   prose; "operational pain point" was already removed (af2bffa). Continuing to professionalize
   register as cycles allow.

4. **"Reproducibility limited by the KV-append integration blocker (bespoke implementation)."**
   Clarified in FINDINGS: KV-append is a *standard* key/value prefix; the constraint is that
   HuggingFace `generate` exposes no hook to append external KV entries, so we use a bespoke
   forward loop. It is an integration constraint, not a non-standard method, and the
   implementation is open. Reframed accordingly.

5. **"Safety claims rest on synthetic setups."** Agreed; the interruptibility/monitoring
   framing is motivation from the architecture, not a measured safety result. We do not
   present it as evaluated safety. (Tightening this framing remains on the list.)

6. **"TC0/FO(M) motivation never formally addressed/tested."** The paper already states this
   explicitly (Summary: "the complexity-theory argument is motivation, not a result… no
   proof a finite-precision reservoir lifts the per-pass bound… central open theoretical
   question"). It is framed as motivation, not a tested claim — which matches the con.

Net: cons 1, 2, 6 are the paper's stated findings/framing rather than fixable defects;
cons 3 and 4 are addressed in this revision; con 5 is acknowledged and queued.
