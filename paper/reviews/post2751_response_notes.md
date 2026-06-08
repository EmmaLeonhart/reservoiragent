# Response to review of post 2751 (rating: Reject)

Pros note the dynamics rigor, the additive-vs-KV injection insight, the ablation's integrity, and
the input-scaling finding. Cons:

1. **"Informal / project-log style."** The persistent con across reviews. Meta-commentary and the
   most log-like phrasings have been removed incrementally; the remaining work is a structural
   consolidation pass (core-first ordering, safety to an appendix), which is the next deliberate
   edit rather than a piecemeal change.

2. **"Cross-pass recall task is too simplistic (single word, 6–24 vocab)."** Fair — now stated
   explicitly in Limitations: the recall demonstration is a minimal single-token, small-vocabulary
   probe that proves *that* usable cross-pass state exists but not its utility for multi-token,
   large-vocabulary, or long-horizon memory; that task-scaling is untested and open.

3. **"Scaling inconsistent, no theory (medium fails, 1.5B works)."** This is the paper's
   model-specific finding, stated as such: input scaling matched to the model is the lever, not
   parameter count; GPT-2-medium is a genuine exception across a 7-point sweep. Not a clean law —
   the cross-model figure shows the non-monotonicity.

4. **"Agentic battery behaviours are artifacts."** Correct and stated (stateless ablation); the
   live demonstration is the controlled cross-pass recall, not the battery.

5. **"KV-append needs a bespoke forward loop."** Standard k/v prefix; HF `generate` exposes no
   append hook. Integration constraint, not a non-standard method; implementation open.

6. **"Safety section speculative / synthetic CPU tests."** Now signposted as secondary motivation
   + synthetic proof-of-concepts, not evaluated safety claims.

This revision also corrected two **stale Limitations bullets** that contradicted the recall-scaling
result (they claimed content "does not learn at scale" / "only temporal tasks learn" — both now
fixed to match the findings).

Net: cons 2, 6 addressed this revision (+ a Limitations QA fix); cons 3, 4, 5 are stated
findings/framing; con 1 (style) is the next structural pass.
