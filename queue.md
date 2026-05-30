# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Why this file exists:** when a planning step produces a plan, that plan is written here BEFORE execution starts, so an interrupted session can pick up from the queue rather than from chat context that may be gone.

See `CLAUDE.md` § "Workflow Rules" and § "Research workflow" for how this file, planning mode, and the task tool stay in sync.

**Three-cron playbook.** Research IS extensive work, so it runs under three local `CronCreate` jobs — **work-loop at :03** (the engine that drains `queue.md` and refills it from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). On a fresh session they are **started** as the opening step (bootstrap step 1 below); on a mid-session **large-scale re-fill** of this queue the FIRST item worked is instead to **kill** the already-running crons. Either way the **last two items are always pinned at the tail** (see `## Always last`). Entering planning mode also disables the crons; their restart lives at the end of the queue. (See `CLAUDE.md` § "Autonomous productivity loop — the three-cron playbook".)

---

## Active — First-session bootstrap (research)

Work these top to bottom. **Delete each item from this file in the same commit that completes it, and append a dated entry to `devlog.md`.** Push after every step. When this whole section is gone, the project has finished bootstrap and the queue is ready to be repopulated with the real research/experiment work (see the final item).

1. **Create `todo.md` — the long-horizon research plan.** Informed by the gap the literature review surfaced, write `todo.md` as the project's long-term horizon: the hypotheses to test, experiments to run / things to build, and the eventual shape of the report. Items here are *abstract destinations*, decomposed into concrete steps in `queue.md` later. Use the format in `CLAUDE.md` § "Queue and longer-horizon work". Commit `todo.md` on its own.

2. **Go live: create a PUBLIC GitHub repo and push.** Public is required for free GitHub Pages. `gh repo create --public --source=. --push`. Confirm CI (`.github/workflows/`) is wired and set **Settings → Pages → Source: GitHub Actions** so `docs/` (the report site) and the built PDF deploy. From here every commit pushes and Pages/CI build as you go.

3. **Replace this bootstrap queue with the real research queue.** Pull the first item(s) from `todo.md` and decompose them into a concrete, ordered list of experiment / implementation tasks under a new `## Active` section (deleting this bootstrap section as part of the same edit). Mirror into the task tool. **Keep the `## Always last` section pinned at the very bottom.** The real queue's FIRST work item should **start the three crons** — unless this is a mid-session large-scale re-fill while they are already running, in which case the first item is instead to **kill them** (the pinned tail restarts them). Commit the new queue.

4. **Work the queue until the stop condition.** Pull the top item, do it, **delete it from `queue.md` AND append a dated entry to `devlog.md`** in the same commit, push, let CI run. Build under `src/`, run via `scripts/run.py`, capture metrics to `results/`, and keep `FINDINGS.md` + the themed `docs/` report current as results land. When `queue.md` empties, refill from `todo.md`. **Stop** when: the research question has a defensible answer (or a clearly reported partial result), `FINDINGS.md` and the published `docs/` report reflect it, `queue.md` is empty, and the repo is online with green CI/Pages. At that point, hand back to the user.

---

## Always last — restart the three crons and summarize

**These two items stay pinned to the tail of the queue at all times** — below every bootstrap step and below every real work item. They are the closing half of the three-cron lifecycle in `CLAUDE.md` § "Autonomous productivity loop":

A. **Ensure the three crons are running** — start them if this session never did, restart them if a planning burst / queue re-fill killed them: work-loop (`3 * * * *`), auto-flush (`15 * * * *`), status-report (`42 * * * *`).
B. **Run the status-report action once more, independently** — an end-of-session summary of everything that happened this session.

---

## Pointers

- Long-horizon backlog (abstract goals, source of future queue items): `todo.md`.
- The literature review (the project's evidentiary base): `literature/REVIEW.md`.
- Completed work (chronological, with milestones): `devlog.md`.
- Narrative history: `git log`.
