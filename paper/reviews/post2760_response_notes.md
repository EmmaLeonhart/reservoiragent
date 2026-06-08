# Response to review of post 2760 (rating: Reject)

Strong pros (the stateless-state framing, the additive-vs-KV explanation, the dynamics
characterization on real activations, transparency on negatives). Cons:

1. **"2025/2026 citations look like hallucinations (Echo State Transformer, Echo Flow Networks,
   FreezeTST)."** They are real recent preprints, not hallucinations. Added their **arXiv
   identifiers** inline (2507.02917, 2509.24122, 2508.18130; Reservoir Transformers 2012.15045) and
   a note that they are recent preprints given so they can be verified — addressing the reviewer's
   inability to check them.

2. **"H1 non-destruction is mathematically trivial."** Agreed — reframed as a *wiring sanity check /
   regression test* (confirming the hook is correctly placed and the graph intact), explicitly not
   a research finding.

3. **"Scaling inconsistent (medium fails, 1.5B works), maybe seeds."** Addressed in the text: the
   working configs are reproduced across two seeds; GPT-2-medium fails across a *seven-point*
   input-scaling sweep (not seed noise). The cross-model figure shows the non-monotonicity; it is a
   characterized, model-specific boundary, with the deciding lever being input scaling.

4. **"Battery results are failures/artifacts."** Stated as such; the live demonstration is the
   controlled cross-pass recall (now scaling across the Qwen family), not the battery.

5. **"Excessively informal ('not a bug', 'well-diagnosed negative', 'the real, hard frontier')."**
   The first two are already gone from the current version; "the real, hard frontier" → "the
   substantial open challenge". Combined with the earlier formalization pass (NeurIPS LaTeX, formal
   title/abstract/headers, Contributions list, appendix for exploratory material).

6. **"Six-word recall is low-complexity."** Stated in Limitations as a minimal single-token probe;
   task-scaling (multi-token, large-vocab, long-horizon) is named in Future Work.

Net: cons 1, 2, 5 fixed this revision; cons 3, 4, 6 are stated findings/framing.
