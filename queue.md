# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Three-cron playbook.** Research runs under three local `CronCreate` jobs — **work-loop at :03** (drains this queue, refills from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). The **last two items are always pinned at the tail** (see `## Always last`). (See `CLAUDE.md` § "Autonomous productivity loop".)

---

## ⛔ 6:30 PM = END OF EXPERIMENTAL WORK (user directive, 2026-06-08)

**After 18:30 local today, NO new experiments / training / sweeps — period** (enforced by one-shot cron
`e0311518`). `#34` is the last experiment; let it finish, then the project is in **arXiv-preparation
mode only**. The goal from here is to ship to arXiv *quickly*: organize the data, build the replication
package, add the AI-use declaration, and get the citations + coherence perfect. Do NOT start a `#35`.

## Current work — finish the last experiment, then arXiv prep

- **#34 stabilization probe — cosine LR decay (in flight, finishes well before 6:30).** #33 showed the
  reservoir solution *oscillates* (lift peaks epoch 1 at recall 1.00, then collapses/rebounds), control
  pinned at 0. `train_large` used a flat LR; `train_battery.py` found a flat LR "overshoots and degrades
  past its peak". #34 = #33 config + `RESERVOIR_COSINE=1` (cosine decay 5e-4→0 over 6000 steps). Run:
  `results/_w34_cosine.log`. Fold the verdict into FINDINGS + site; claim retention only if the lift
  survives the control across epochs. **This is the last experiment — nothing after it.**

## arXiv preparation (the plan — start now, in parallel with #34 finishing)

- **Declaration of AI use (arXiv requirement).** Add an honest "Declaration of AI use" section to
  FINDINGS (→ PDF + site): this is an AI-agent-driven research project; the agent implemented the code,
  ran/analyzed the experiments, made the figures, and drafted the manuscript under human direction; all
  results are from executed code + measured outputs with stateless controls, human-reviewed. **Do first
  — required and quick.**
- **Replication script + downloadable replication package (zip).** Build a clear `replication/` with: a
  single script that reproduces the headline results (GPT-2 cross-pass recall 1.00 vs 0.17; Qwen-1.5B
  scaling 0.83–1.00), a README with exact commands + environment (the verified flags), pinned deps, and
  the key result JSONs. Package as a downloadable **zip** linkable from the site + paper. CI builds/ships
  it (like the arXiv source tarball). Verify the script actually runs (at least the CPU-only parts).
- **Citations — make them perfect (do repeatedly).** The 16:34 cron `c47ff97f` runs the comprehensive
  audit; beyond it, keep re-verifying every arXiv ID / author / year / claim-attribution against reality
  until flawless. Wrong citations on arXiv are permanent. This is high priority and worth multiple passes.
- **General coherence + organization pass.** Holistic read-through of FINDINGS: flow, no dangling/internal
  refs, intro↔results↔abstract consistency, remove any residual lab-notebook register, ensure the section
  order reads as a proper arXiv paper. Tighten; cut repetition.
- **Organize + verify the data.** Go through `results/` and `docs/` figures: confirm every figure/claim is
  backed by the correct run's data (watch for a figure regenerated from the wrong run — this kind of
  mismatch happened before with `crosspass.png`). **The user flagged "a weird mistake that might indicate
  something with the data is worth replacing" — actively hunt for it during this pass and fix/replace.**
- **Final arXiv packaging.** Confirm metadata (title/author/abstract/category/license), the NeurIPS PDF,
  and the arXiv source tarball are consistent and complete. (arXiv account + endorsement is the user's
  step — not autonomous.)

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
