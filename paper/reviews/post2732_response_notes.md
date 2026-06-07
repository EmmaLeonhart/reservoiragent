# Response to review of post 2732 (rating: Weak Reject)

Rating recovered from Reject (2731) to Weak Reject after the previous revision. Cons:

1. **"Cross-pass state demonstrated only on GPT-2-small; fails at 355M–3B."** The paper's stated
   headline negative, reproduced across interventions. Not a fixable gap.

2. **"Battery largely invalidated by the authors' own ablation."** Correct and stated; the live
   positive is the GPT-2 injection-design result, not the battery. The new progressive-run
   gate-collapse data reinforces this rather than contradicting it.

3. **"Fragmented/informal; reads like a chronological log ('The larger-budget run — done')."**
   Addressed the quoted example: the N-seed larger-budget header is rewritten from the log-style
   "The larger-budget run — done, and the negative holds" to "At a larger budget the negative
   holds: at 1500 steps, selection is still not real." Continuing to convert log-style mini-headers
   to synthesized claims across cycles; the Abstract and Summary already read as a single argument.

4. **"'Always-alive runtime' / 'agentic' claims are infrastructure-level, not demonstrated
   agentic reasoning."** Fair — those are framed as the long-horizon target, out of scope for this
   feasibility+dynamics study (now stated as such where the runtime is mentioned). The paper does
   not claim demonstrated agentic reasoning; the live results are the controlled memory tasks.

5. **"TC0/FO(M) never formally linked to findings."** Framed explicitly as motivation, not a
   result, with the per-pass-bound question called the central open theoretical problem. Matches
   the con; a formal link is beyond this study.

Net: cons 1, 2, 5 are stated findings/framing; con 3 is being addressed progressively (the quoted
instance fixed this cycle); con 4 is a framing tightening already in place.
