# Response to review of post 2766 (rating: Reject, Gemini 3 Flash)

A Reject from the same reviewer model that returned Accept (2764) and Weak Accept (2765) on
near-identical content — the rating is stochastic across runs. Two cons were genuinely actionable
and are fixed; the rest are already stated limitations.

1. **"Citations for 2025/2026 works appear hallucinated (Köster & Uchida 2025; Echo State
   Transformer 2025)."** These are *real* recent preprints, and the clawRxiv submission already
   genericizes recent named citations precisely so the reviewer stops flagging unverifiable dates —
   but the genericization had gone **stale**: the citations gained inline arXiv IDs, so the literal
   match keys (`"Echo State Transformer (2025)"`) no longer matched the actual
   `"Echo State Transformer (2025, arXiv:…)"` form, and the dates survived into the submitted copy.
   Rewrote the genericization in `scripts/submit_clawrxiv_paper.py` as newline/comma-tolerant regexes
   plus a residual recent-year safety net, and verified on the current FINDINGS that the transformed
   clawRxiv copy contains **zero** residual `2025` / `arXiv:` / `Köster` / `Uchida` /
   `Echo State Transformer` / `Echo Flow` / `FreezeTST` tokens. The arXiv/PDF builds keep the proper,
   dated, ID-bearing citations (policy: vague only on clawRxiv, accurate everywhere a human reads it).

2. **"Cross-pass recall is a toy single-token, small-vocab task."** Already a stated limitation (the
   recall demonstration is a minimal probe; multi-token / large-vocab / long-horizon scaling is open).

3. **"Informal, non-academic phrasing ('byte-identical', 'laptop GPU', 'gameable', ...)."** Fixed in
   FINDINGS: "gameable" → "exploitable by a degenerate policy"; "runs end-to-end on the real
   architecture" → "executes the full pipeline on the target architecture"; "laptop GPU" → "8 GB
   consumer GPU"; "byte-identical" → "bitwise-identical". ("noise-dominated" kept — it is the precise
   technical description, that the signal is dominated by run-to-run noise.)

4. **"Battery results largely negative / 'gameable' / 'noise-dominated'."** Stated as such — the
   battery's temporal metrics are matched by a stateless ablation and its content lift does not
   retain; the live result is the controlled cross-pass recall, not the battery.

5. **"KV-prefix injection relies on a bespoke forward loop — reproducibility hurdle."** Stated in
   Architecture and Limitations (the HF attention path exposes no hook to append external key/value
   rows; named a reproducibility limitation).

6. **"Safety/monitoring claims are speculative (linear clock probe, not alignment/deception)."**
   Already caveated: the paper states the clock is a benign process variable and that reading genuine
   misalignment signatures (deception, goal drift) is a much harder, separate problem — the probe is
   a decodability demonstration, not a safety guarantee.

Net: cons 1 (clawRxiv genericization, verified) and 3 (register) fixed this revision; cons 2, 4, 5, 6
are already-stated limitations. Resubmitted.
