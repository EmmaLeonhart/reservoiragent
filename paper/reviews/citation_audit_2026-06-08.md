# Citation audit — 2026-06-08 (pre-arXiv)

Adversarial verification of every citation in FINDINGS.md, run by four parallel web-enabled agents
(each fetched the arXiv abstract page / searched primary sources). **Headline: no hallucinations —
every cited work is real.** 19 arXiv IDs + 5 non-arXiv refs checked; the high-risk 2025 preprints all
resolve to real papers. Several citations had wrong/missing metadata; all fixed in this pass.

## Verdicts

### Reservoir computing
- Jaeger (2001), GMD Report 148 — **OK**.
- Maass, Natschläger & Markram (2002), Neural Computation 14(11):2531–2560 — **OK** (full subtitle is "A new framework … based on perturbations").
- Lukoševičius & Jaeger (2009), Computer Science Review 3(3):127–149 — **OK**.
- Köster & Uchida (2025), arXiv:2507.15779, *Reservoir Computing as a Language Model* — **OK** (real; the clawRxiv reviewer's "hallucination" flag was wrong).
- Echo State Transformer, arXiv:2507.02917 — **FIXED**: real, by **Bendi-Ouis & Hinaut (2025)**, *Echo State Transformer: Attention Over Finite Memories*. Added authors inline + a References entry.
- Echo Flow Networks, arXiv:2509.24122 — **FIXED**: real, by **Liu & Xu (2025)**. Added authors + References entry.
- "FreezeTST", arXiv:2508.18130 — **FIXED**: "FreezeTST" is **not** the real title. Actual: **Singh et al. (2025)**, *Frozen in Time: Parameter-Efficient Time Series Transformers via Reservoir-Induced Feature Expansion and Fixed Random Dynamics*. Renamed inline to "Frozen-in-Time (Singh et al., 2025)" + References entry.
- Reservoir Transformers, arXiv:2012.15045 — **OK** (Sheng Shen et al.; arXiv 2020 / ACL 2021 — kept ACL year). Added a full References entry.

### Transformer expressivity
- Hahn (2020), TACL, arXiv:1906.06755 — **OK**; claim (no unbounded hierarchy without scaling layers/heads with input length) supported.
- Merrill, Sabharwal & Smith (2022), arXiv:2106.16213 — **OK**; claim (saturated/float transformers ⊆ TC⁰) supported. **FIXED** the title string (dropped the editorial "(⊆ TC⁰)"; added TACL).
- Merrill & Sabharwal (2023), *The Parallelism Tradeoff* — **FIXED**: added **arXiv:2207.00729**, TACL 2023 (ID was missing); FO(M) claim supported.
- Pérez et al. (1901.03429) — **FIXED**: arXiv:1901.03429 is *On the Turing Completeness of Modern Neural Network Architectures* (ICLR 2019); *Attention is Turing-Complete* is the JMLR 2021 version. Corrected title, author order, and **Marinković** spelling; noted both versions.
- Siegelmann & Sontag — **FIXED**: real title *On the Computational Power of Neural Nets*, JCSS 50(1):132–150, 1995 (COLT 1992 precursor); claim (finite RNNs Turing-complete at arbitrary precision) supported.
- Weiss, Goldberg & Yahav (2021), ICML, arXiv:2106.06981 (RASP) — **OK**.

### Recurrence-augmented transformers
- Transformer-XL (Dai et al., 2019, arXiv:1901.02860) — **OK**.
- Memorizing Transformers (Wu et al., 2022, arXiv:2203.08913) — **OK**.
- Block-Recurrent Transformers (Hutchins et al., 2022, arXiv:2203.07852) — **OK**.
- Recurrent Memory Transformer (Bulatov, Kuratov & Burtsev, 2022, arXiv:2207.06881) — **OK**.
- S4 (Gu, Goel & Ré, arXiv:2111.00396) — **OK** (arXiv 2021 / ICLR 2022; kept 2022).
- Mamba (Gu & Dao, 2023, arXiv:2312.00752) — **OK**.
- Titans (Behrouz, Zhong & Mirrokni, arXiv:2501.00663) — **OK** (v1 submitted Dec 31 2024; "2024" is accurate, "2025" by ID-month convention — left as 2024).

### KV-cache / efficient attention
- StreamingLLM (Xiao et al., 2023, arXiv:2309.17453) — **OK**.
- H2O (Zhang et al., 2023, arXiv:2306.14048) — **OK**.
- DeepSeek-V2 (DeepSeek-AI, 2024, arXiv:2405.04434, MLA) — **OK**.

### Prose facts (no formal cite)
- DeepSeek Sparse Attention (DSA), introduced in V3.2 — **CONFIRMED** real (arXiv:2512.02556 exists).
- DeepSeek-V4-Flash — **CONFIRMED** real (HF repo `deepseek-ai/DeepSeek-V4-Flash` + official announcement, Apr 2026). **FIXED**: the specific "CSA+HCA" attention label was not pinned to a primary DeepSeek source (official wording: "token-wise compression + DSA"), so the prose was softened to "learned sparse attention (DeepSeek Sparse Attention, introduced in V3.2 and carried into the V4 line)".

## Summary

- **Real / no hallucinations:** 24/24 works.
- **OK as-was:** ~14.
- **FIXED this pass:** 8 (added arXiv:2207.00729; corrected Pérez title/order/spelling; renamed "FreezeTST" → "Frozen in Time" (Singh et al.); added authors to Echo State Transformer / Echo Flow Networks; added 4 References entries with verified authors; cleaned the Merrill–Smith title; clarified Siegelmann–Sontag; softened the DeepSeek-V4 attention label).
- **UNVERIFIABLE / removed:** 0.

The clawRxiv genericization regexes were updated to match the new author-prefixed citation forms and re-verified (the transformed clawRxiv copy has zero residual 2025/arXiv/named-work tokens). Per the audit directive, no clawRxiv resubmission was triggered for citation fixes alone.
