# reservoiragent — Devlog

**This file is where "done" lives.** `queue.md` is delete-only: when a queue
item is finished, the item is **deleted from `queue.md`** and a dated entry
is **appended here**, in the same commit as the work, then pushed. Never
tick a box in place — a checked box left in `queue.md` is the failure mode
this file exists to prevent.

Also record releases (tag + a one-line note), notable milestones, and
anything else worth a chronological trail. Newest entries at the bottom.

This is the **same convention as the cleanvibe repo's own `devlog.md`** —
every cleanvibe-scaffolded project gets one for the same reason.

See `CLAUDE.md` § "Workflow Rules" and `queue.md`'s preamble.

---

## 2026-05-29 — Project scaffolded

Scaffolded with `cleanvibe new` (cleanvibe v1.13.0). Future entries
land here as queue items get deleted.

## 2026-05-29 — Bootstrap 1: autonomous loop started

Started the three session-local crons (`durable:false`): work-loop `3 * * * *`
(de9f5f85), auto-flush `15 * * * *` (42d311a4), status-report `42 * * * *`
(ab624872). Mirrored the bootstrap items into the task tool.

## 2026-05-29 — Bootstrap 2: data_lake triage + originating-context preservation

The project's originating context was three Claude chat HTML exports + a
user-authored `reservoir_agent_plan.md`. Preserved the durable, publishable parts
and protected privacy:

- `data_lake/extract_chat_context.py` distils each export into clean Markdown
  **transcripts** (`data_lake/transcripts/`, 24/6/32 turns) and recovers the
  **architecture diagrams** as SVG (`data_lake/diagrams/`: per-pass injection,
  always-alive runtime, joint attention layer + a transformer baseline) plus
  WEBP raster previews.
- The raw `*.html` exports + `*_files/` chrome are **gitignored** (kept local):
  they embed a private "Recents" sidebar of unrelated conversations and ~10 MB
  of web chrome each. Transcripts verified to contain zero sidebar leakage.
- Recovered SVGs use Claude's dark-theme CSS vars, so they render black-boxed
  standalone (labels/structure intact); they'll be re-themed for the `docs/`
  report. `data_lake/README.md` documents provenance.

## 2026-05-29 — Bootstrap 3: research question defined

After reading the plan + transcripts, pinned the question with the user (who chose
a **feasibility + dynamics study** scope for this session via a scoping checkpoint):

> Can a fixed, randomly-initialized reservoir injected into a pretrained transformer's
> mid-layer attention give the model genuine state between forward passes — a real time
> axis — without degrading its base capabilities, and what reservoir-dynamics regime
> (spectral radius, size, injection depth) makes that injected state usable signal
> rather than noise?

Scope this session: GPT-2-scale base on a single CUDA machine — inject the reservoir,
regression-test base behavior, characterize dynamics across spectral radius, write up
theory (time dimension; Turing-completeness via recurrence). Full Hermes-fork +
always-alive runtime + N-seed LoRA selection is the long-horizon `todo.md` target.
Written into `README.md`, `CLAUDE.md` (Project Description + Research question), and
`docs/index.html` (lede, question block, pillar 1).

## 2026-05-29 — Bootstrap 4: literature review (agentic RAG)

Ran a multi-agent deep-research pass (103 agents; 5 angles → 21 sources → 96 claims →
25 adversarially verified, 24 confirmed / 1 refuted), then a targeted re-verification
of the closest prior art. Wrote `literature/sources.md` (per-source notes, grouped by
area, with verification status) and `literature/REVIEW.md` (synthesis).

Key outcomes:
- **Foundations solid:** the fixed-reservoir/trained-readout core is textbook ESN/LSM
  (Jaeger, Maass, Lukoševičius & Jaeger); the spectral-radius/ESP regime is real but
  the edge-of-chaos optimum is *disputed* — vindicating the planned empirical sweep.
- **Sharp theoretical gap:** finite-precision transformer ⊆ TC⁰/FO(M) *per pass*
  (Merrill & Sabharwal; Hahn); cross-pass state feedback is the documented escape —
  but the known proofs need *arbitrary precision*, so whether finite-precision
  reservoir state lifts the bound is **open** (paper must pose, not assert).
- **Novelty axis:** every prior recurrence-augmented transformer (Transformer-XL,
  Compressive, Universal, Block-Recurrent, Memorizing, RMT, RWKV, RetNet, S4/Mamba,
  Titans) uses *trained* recurrence carrying state *within a sequence*. None uses a
  *fixed-random* reservoir with state across *independent* passes — the Reservoir
  Agent's empty cell. Block-Recurrent independently confirms the "model learns to
  ignore the recurrent state" failure mode the plan worries about.
- **Honest caveat logged:** the reservoir-×-transformer and always-on-agent areas
  were not verification-complete (some returned arXiv IDs didn't resolve and were
  discarded, not cited); a citation-checked follow-up is queued in `todo.md` before
  any hard novelty claim is published. Reflected the one-line summary into
  `docs/index.html` (pillar 2).

## 2026-05-29 — Bootstrap 5: long-horizon todo.md written

Wrote `todo.md` from the literature gap: hypotheses H1–H4 (feasibility/non-destruction,
healthy-dynamics regime, state-readability, cross-pass behaviour change); experiments
tiered near-term (reservoir core, model surgery, spectral-radius dynamics sweep —
this session) / mid-term (harness fork PoC, benchmark+training-data design, readout+LoRA)
/ long-term aspirational (N-seed selection, always-alive runtime, Hermes scale-up);
the scoped theory section; the open questions carried from the review (citation-checked
novelty follow-up; the finite-precision TC⁰ question; edge-of-chaos at attention scale);
and the report shape.
