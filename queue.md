# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Three-cron playbook.** Research runs under three local `CronCreate` jobs — **work-loop at :03** (drains this queue, refills from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). The **last two items are always pinned at the tail** (see `## Always last`). (See `CLAUDE.md` § "Autonomous productivity loop".)

---

## ⛔ EXPERIMENTAL WORK CLOSED — 6:30 PM hard-stop fired (2026-06-08 18:30)

**Experiments are CLOSED for the session.** `#34` was the last experiment (finished, retention win,
folded). The project is now in **arXiv-preparation mode ONLY**. NO new experiments / training / sweeps /
model launches — regardless of what any queue or todo item says. If a tick is tempted to start one,
refuse and redirect to arXiv prep. The goal: ship to arXiv quickly — verify data, replication package,
AI-use declaration (all done), keep citations + coherence perfect, finalize packaging.

## Current work — finish the last experiment, then arXiv prep

- **#34 — DONE: retention WIN (folded).** Cosine LR + capacity denial retains reservoir-driven recall:
  lift +0.089 → +0.089 → +0.130 → +0.292, **recall 0.08 → 0.19 → 0.35 → 1.00**, control pinned at 0.000,
  monotonic, no collapse. First stable retention result; the recipe is deny-the-shortcut-capacity +
  decay-the-LR. Folded into abstract + battery limitation + site (with single-run / recall-only caveats).
  Last experiment of the session — nothing after it.

## arXiv preparation — remaining

(Done and removed: AI-use declaration; downloadable replication package; comprehensive citation audit —
4 agents, no hallucinations, 8 fixes; data audit — found + fixed the 12-key "undertraining" error;
register pass + internal file-path cleanup; abstract↔contributions↔results consistency verified.)

- **Final arXiv packaging — VERIFIED (2026-06-08).** The live arXiv source tarball is current (includes
  the #34 retention win), self-contained (paper.tex + paper.tex.body + neurips_2026.sty + figures/), and
  all 14 figures referenced by the paper resolve; title/author metadata set; PDF builds green. Minor
  optional cleanup: the tarball ships all `docs/` figures (~50), but the paper uses 14 — harmless bloat
  (~1.3 MB), not a blocker. **arXiv account + endorsement + category/license selection is the user's step.**
- **Open user decisions (do NOT act autonomously):** (1) the 12-key capacity point — leave as a documented
  single-seed artifact, or re-run multi-seed for a clean figure (needs GPU, before 6:30); (2) delete the
  stale `feat/realtime-agent-app` branch + the idle `reservoir.installer` process.

## Grok's reception — addressed (record)

Grok's arXiv-polish suggestions are folded in; full text in git history (this commit's parent of
queue.md). Disposition: clean LaTeX pass → NeurIPS-2026 build (`paper/paper.tex` + `neurips_2026.sty`);
vector figures + captions, no screenshots → SVG diagrams + plot PNGs, `\pandocbounded` scale-to-fit;
abstract leads with the injection/scaling result while keeping caveats → done; exploratory/negative
material → appendices; stronger Contributions list early → done; dedicated Limitations + Future Work
calling out the 3B wall + compute → done; code release (README repro commands, weights on HF) → done;
safety/positioning angle → Safety section. **Not autonomous (the user's to do):** Google Scholar /
arXiv endorsement and Discord/community outreach.

## Current ongoing work

- **Battery reservoir-retention — resolved negative (folded).** The clean cross-pass recall task
  scales to the Qwen family (signal); the integrated battery learns a reservoir-driven recall then
  drifts to a stateless shortcut (#29), and neither low `lora_r` (#31) nor noise alone explains it.
  The counterfactual "use-the-state" aux loss (#30) was run on the content battery for 4 epochs
  (#32): it did **not** make the lift hold — mean lift decayed +0.302 → +0.094 → +0.000 → +0.000 as
  the stateless control rose to match. Verdict folded into FINDINGS Limitations + site (no retention
  claim). Stable retention remains open; the first-line stabilizer fails. Any further attempt is new
  todo.md work (e.g. a harder state-dependence regularizer, or a task design where the current pass
  *cannot* shortcut), not an open queue item.

- **Paper formalization (ongoing).** The paper now builds in the **NeurIPS-2026 LaTeX format**
  adopted from the Sutra repo (`paper/paper.tex` + `neurips_2026.sty` + `paper-pdf.yml`, source =
  `FINDINGS.md`). Title/abstract/section headers formalized; inaccessible internal file-path refs
  removed. Remaining: continue replacing any lab-notebook prose / bold-shouting with formal
  register as it surfaces in reviews.

- **Continuous paper/site upkeep.** Every new result updates FINDINGS **and** `docs/index.html`
  together; resubmit to clawRxiv when material; fold reviews as they arrive.

---

## Paper — two internal contradictions found in a 2026-06-21 review (need Emma's run data)

These are real and need the author's call against the actual experiment output; do
NOT guess the numbers (hard rule: real numbers only).

1. **Gate-D (unresolved-thread) F1 is reported two different ways.** `FINDINGS.md`
   line ~752 (summary) and lines ~1199-1201 (Appendix E) say reservoir **F1 ≈ 0.96
   (P=0.93, R=1.00)** vs stateless **≈ 0.34** (stateless = "always speak", recall≈1).
   But the main result section lines ~773-775 say reservoir **F1 = 0.48 (P=0.71,
   R=0.36)** vs stateless **F1 = 0.03 (P=1.00, R=0.02)** (stateless = misses the
   thread, recall≈0). Same experiment, but they disagree on both the reservoir number
   AND the stateless behavior. Resolve to the real run output and make all three
   locations agree. Results are gitignored, so the canonical numbers aren't in the
   repo — re-run or read the run log.
2. **KV-append vs KV-prefix terminology contradicts itself.** Line ~450 says "we use
   KV-append and KV-prefix interchangeably for the same injection," but lines ~104,
   ~1054, and ~1132 (Appendix A "not done") treat **KV-append** as a *distinct,
   not-done* richer variant (true KV-cache append, which HF `generate` won't expose a
   hook for) vs the **KV-prefix** path that was actually run. Pick one usage: either
   they're the same (then drop the "KV-append not done" bullet at line ~1132), or
   they're distinct (then fix line ~450 and use the names consistently throughout).

## Always last — restart the three crons and summarize

**These two items stay pinned to the tail of the queue at all times.** They are the closing half of the three-cron lifecycle in `CLAUDE.md` § "Autonomous productivity loop":

A. **Ensure the three crons are running** — work-loop (`3 * * * *`), auto-flush (`15 * * * *`), status-report (`42 * * * *`); start/restart as needed.
B. **Run the status-report action once more, independently** — an end-of-session summary.

---

## Pointers

- Long-horizon backlog (abstract goals, source of future queue items): `todo.md`.
- The literature review (the project's evidentiary base): `literature/REVIEW.md`.
- Completed work (chronological, with milestones): `devlog.md`.
- Narrative history: `git log`.
