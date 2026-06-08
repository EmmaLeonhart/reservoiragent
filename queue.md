# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Three-cron playbook.** Research runs under three local `CronCreate` jobs — **work-loop at :03** (drains this queue, refills from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). The **last two items are always pinned at the tail** (see `## Always last`). (See `CLAUDE.md` § "Autonomous productivity loop".)

---

## Current work

- **Battery reservoir-retention (in flight).** The clean cross-pass recall task scales to the Qwen
  family (signal); the integrated battery learns a reservoir-driven recall then drifts to a
  stateless shortcut (#29), and neither low `lora_r` (#31) nor noise alone explains it. The
  counterfactual "use-the-state" aux loss is implemented + tested (#30) and running on the content
  battery (#32) to test whether it makes the lift *hold* across epochs. Fold the resolved verdict
  into FINDINGS + site; only claim retention if it survives the control across epochs.

- **Paper formalization (ongoing).** The paper now builds in the **NeurIPS-2026 LaTeX format**
  adopted from the Sutra repo (`paper/paper.tex` + `neurips_2026.sty` + `paper-pdf.yml`, source =
  `FINDINGS.md`). Title/abstract/section headers formalized; inaccessible internal file-path refs
  removed. Remaining: continue replacing any lab-notebook prose / bold-shouting with formal
  register as it surfaces in reviews.

- **Continuous paper/site upkeep.** Every new result updates FINDINGS **and** `docs/index.html`
  together; resubmit to clawRxiv when material; fold reviews as they arrive.

- Also, I'm not sure if you have this thing properly. I don't know if you have this particular thing properly running. I believe the Sutra repository has something in it that generates an easy arXiv upload package. I think it's something like /arxiv.zip Although honestly I'm a bit confused about the Sutra repository too, I want us to generate something like this so that we create a decent arXiv uploading package thing. 

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
