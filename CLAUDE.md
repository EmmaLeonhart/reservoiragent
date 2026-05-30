# reservoiragent — research project

## Project Description

This is a **research project** scaffolded by `cleanvibe research`. Unlike a
*replication* (which reproduces someone else's paper), this is **your own**
investigation: you pose a question, survey the prior literature, run experiments
or build something to answer it, and publish the findings.

> **Research question:** _(not yet defined — the bootstrap queue's first research step pins this down with you)_

Like a cleanvibe replication it produces a published, legible report — a themed
**GitHub Pages site** (`docs/`) plus a transportable PDF — but the content is
original research, grounded in a literature review rather than in one target
paper.

## Research workflow (the shape of this project)

1. **Question.** Pin down precisely what is being investigated / built and what a
   successful answer looks like. (Bootstrap step; the `> Research question` above
   gets filled in then.)
2. **Literature review (agentic RAG) — BEFORE building anything.** Survey the
   prior work: use whatever agentic search / RAG tooling is available (web
   search, `WebFetch`, and the `deep-research` skill if present) to find the
   relevant papers, posts, datasets, and code; read them; collect sources with
   citations into `literature/`; synthesize `literature/REVIEW.md` (what is
   already known, the gaps, and what *this* project adds). This grounds the work
   in the field instead of reinventing it, and it is what makes a `research`
   project different from a plain `new` one.
3. **Hypotheses & experiments.** Turn the identified gap into concrete, testable
   experiments / build steps. Plan them `todo.md` → `queue.md`.
4. **Build & run.** Implement under `src/`; entry point `scripts/run.py`;
   metrics → `results/`.
5. **Findings & report.** Write `FINDINGS.md`; keep the themed `docs/` site and
   the PDF report current as results land.

## Architecture and Conventions

- **`literature/`** — the literature review: source notes (one file per source,
  or a `sources.md`) and `REVIEW.md` (the synthesized survey, with citations).
  Committed; it is the evidentiary base of the project. Built in workflow step 2,
  before any implementation.
- **`data_lake/`** — datasets and other supplied/downloaded material (standard
  cleanvibe convention). Committed.
- **`src/`** — the research code. **`scripts/run.py`** — the entry point CI can
  invoke. **`results/`** — metrics JSON / run outputs (gitignored). **`FINDINGS.md`**
  — the write-up (question, method, results, limitations).
- **`docs/`** — the **published GitHub Pages site** (themed `index.html`, figures,
  and the built `report.pdf`). This is the legibility layer. The theme ships
  pre-styled (warm "paper" light theme + dark-mode variant); edit the content,
  keep the chrome. Site-shape inspiration: http://latent-space.emmaleonhart.com/
- **Go live early.** Create a **PUBLIC** GitHub repo and push near the start so
  every commit pushes and Pages/CI build as you go (public is required for free
  GitHub Pages).
- **Deliverables are built by GitHub Actions.** `.github/workflows/pages.yml`
  deploys `docs/` (the report site) and builds `docs/report.pdf` from
  `FINDINGS.md`. Make the repo public and set Settings -> Pages -> Source:
  GitHub Actions.

## Workflow Rules
- **Commit early and often.** Every meaningful change gets a commit with a clear message explaining *why*, not just what.
- **Plan into `queue.md` first, then execute.** When entering planning mode (or doing any non-trivial multi-step work), the FIRST action is to write the plan into `queue.md` as concrete items. Only then begin executing. This means an interrupted session can resume from the queue — the plan does not live only in chat context.
- **Finishing an item = delete from `queue.md` + append to `devlog.md`, then commit and push.** IMPORTANT: when a queue item is done, **delete the item from `queue.md`** and **append a dated entry to `devlog.md`** recording what was completed, in the *same commit as the work*, then push. NEVER mark an item done in place (no `[x]`, no "✓", no "DONE" — a checked box left in `queue.md` is the failure mode this rule exists to prevent). `queue.md` only ever holds not-yet-done work; `devlog.md` is where "done" lives.
- **Mirror `queue.md` into the task tool.** TaskCreate items as you add them to queue.md; mark `in_progress` when starting; `completed` when done. The two views must not drift.
- **Keep this file up to date.** As the project takes shape, record architectural decisions, conventions, and anything needed to work effectively in this repo.
- **Update README.md regularly.** It should always reflect the current state of the project for human readers.

## Queue and longer-horizon work
- **`queue.md`** — what's being worked on right now. Items get deleted on completion; do not leave checkmarks or status indicators behind. If it's not in `queue.md`, it's not in scope for the current session.
- **`todo.md`** — the **long-term horizon** of the project. Multi-session goals, architectural ambitions, future capabilities, "things we want to do eventually." Items in `todo.md` are *abstract*: they describe a destination, not a step. `todo.md` is the *basis for* `queue.md`: when work begins, an item is pulled from `todo.md`, decomposed into concrete executable steps in `queue.md`, mirrored into the task tool, and executed. As `queue.md` drains, refill it by pulling and decomposing the next `todo.md` item.
- **`devlog.md`** — where **"done" lives**. Every queue item that is finished gets deleted from `queue.md` and appended as a dated entry here, in the same commit as the work. Releases (tag + one-line note) and notable milestones also go here. `devlog.md` exists so `queue.md` can stay strictly delete-only without losing the historical trail.
- **Flow:** `todo.md` (abstract horizons) → `queue.md` (concrete steps) → task tool (in-flight work) → `devlog.md` + `git log` (history). Items only ever flow forward; do not leave done items behind in `todo.md` or `queue.md`.
- **Session end condition:** the project's first session ends when `queue.md` is empty, the only items left in `todo.md` are still too abstract to break down further, and the repository is online with green CI. At that point, stop and hand back to the user.

## Testing
- **Write unit tests early.** As soon as there is testable logic, create a test file. Use `pytest` for Python projects or the appropriate test framework for the language in use.
- **Set up CI as soon as tests exist.** Create a `.github/workflows/ci.yml` GitHub Actions workflow that runs the test suite on push and pull request. Keep the workflow simple — install dependencies and run tests.
- **Keep tests passing.** Do not commit code that breaks existing tests. If a change requires updating tests, update them in the same commit.

## Writing
- Do not use "honest", "honesty", or "honestly" — and do not swap in "frank", "frankly", "candid", "candidly", or "transparently", which are the same self-congratulatory move in a different coat. When something failed, name the failure: "it didn't work", "I got that wrong", "this failed" — flat, no qualifier. Tagging a report "honest" implies the rest aren't, and couching a failure as honesty asks for credit for the admission, which is worse than the failure itself. Use a precise positive word ("accurate", "plainly", "truly") only when that is genuinely the meaning — never as a halo on a bad outcome.

## Cron jobs and scheduled work — LOCAL by default

**When the user says "cron job", "cron", or asks to "schedule" a task, they mean the in-session `CronCreate` tool** — a prompt scheduled to fire locally, inside this Claude Code session, on the user's own running computer. This is the default and the assumption. Do NOT interpret it as anything else unless the user explicitly names a different mechanism.

- **It is local and in-session — use the `CronCreate` tool.** A generic "cron" request is NOT an OS crontab, NOT a GitHub Actions / CI `schedule:` trigger, and NOT a cloud scheduler. (A repo may *also* contain its own GitHub Actions cron schedules — those are a separate thing and are not what the user means when they ask *you* to set up a cron.) The user leaves the computer on and this session running so the scheduled prompt can execute.
- **The user is deliberately away from the keyboard.** They schedule work precisely so it runs while they are out of the house and not physically present. Their absence is the normal, expected condition for these jobs — it is NEVER a reason to delay the work, ask "are you sure?", wait for them to return, or refuse to proceed.
- **Standing consent — just set it up.** Cron / `CronCreate` requests are pre-authorized. Create the job immediately and locally, then report what was scheduled. Do not block on confirmation or follow-up questions.

## Autonomous productivity loop — the three-cron playbook

**For any session involving relatively extensive work — above all, any large-scale population of `queue.md` with created tasks — this is the default way of working.** It is three local `CronCreate` jobs that turn "barrel through `queue.md`, and when it's empty atomise the next `todo.md` item into it" into a self-sustaining hourly cadence with a commit/push backstop and a heartbeat. The crons are **session-local** (`durable: false` — they die when the session ends), so they are recreated at the start of every session.

Stagger the minutes so the three ticks don't collide:

1. **Work-loop cron — `3 * * * *` (hourly at :03).** The engine. Each tick does, in order:
   - **(a) SYNC** — `git fetch origin`; fast-forward or rebase the working branch (never force-push, never `reset --hard`, never discard a sibling machine's work).
   - **(b) WORK** — take the top actionable item from `queue.md` and do it. If nothing in `queue.md` is actionable (all blocked / needs user / a product decision), promote the next *genuinely-unblocked, bounded, verifiable* `todo.md` item — **plan it into `queue.md` first**, mirror to the task tool, then execute.
   - **(c) HARD RAILS** — never fake; never weaken / skip / delete a test to make it pass; never claim "works" / "verified" / "passes" without having actually RUN it and measured. A real defect → strict `xfail` or a precise documented blocker, never a loosened assertion. Don't implement what you don't 100% understand — write the spec / queue item instead. Name unbuilt or hard things plainly; don't paper over difficulty. Verify CI green, not just local — local-green does not imply CI-green.
   - **(d) COMMIT** — commit early/often with *why*; update `queue.md` in the same commit (delete completed items); append the dated entry to `devlog.md`; mark task-tool items done; push.
   - **(e) REPORT** — one line: the commit shas advanced, or `nothing actionable; <reason>`.

2. **Auto-flush cron — `15 * * * *` (hourly at :15).** The backstop. Commit + push all pending work so nothing sits uncommitted between manual pushes; report shas or "nothing pending". Only commit / push when something is actually pending — no empty commits.

3. **Status-report cron — `42 * * * *` (hourly at :42).** The heartbeat — **reporting only, no code changes.** Covers: what advanced since the last report (shas + one-line each); current `queue.md` state; how the work held the hard rails (and any place it brushed one); blockers / items deliberately not done autonomously and why; test-suite health.

**Why this exists:** the most common autonomous-agent failure is doing a large amount of work and silently losing the thread of what it is doing. The work-loop forces steady, verifiable, committed progress; the auto-flush guarantees nothing is lost between ticks; the status-report keeps the thread legible.

**Lifecycle around a large-scale queue fill:**

- **(a) START all three crons at the beginning of any extensive work session.** A fresh session has none of them running, so the opening move — the first queue item — is to *create them*.
- **(b) On a mid-session large-scale queue RE-FILL** (a planning burst that repopulates the queue), the FIRST item of that fill **kills the running crons**, then the work items follow top to bottom, and the pinned tail restarts them.
- **(c) Entering planning mode DISABLES the crons.** Their restart therefore lives at the **end** of the queue, not the beginning of the next burst.
- **(d) The LAST TWO queue items, always kept pinned at the tail, are:**
  1. **Ensure the three crons are running** — start them if this session never did, restart them if a planning burst / queue re-fill killed them.
  2. **Run the status-report action once more, independently** — an end-of-session summary of everything that happened this session.

In short: a fresh session **starts** the crons up front and the tail **ensures they are still running** + summarizes; a mid-session re-fill **kills** them up front and the tail **restarts** them + summarizes. Either way the queue both opens and closes on the cron set.

## Check cleanvibe for skill updates (weekly)

This `CLAUDE.md` was generated by **cleanvibe v1.13.0** on 2026-05-29. cleanvibe ships new skills, template sections, and workflow conventions over time — when one lands, every cleanvibe-scaffolded project should pick it up.

**The check is weekly, not per-session.** At the top of any session, look at the *last cleanvibe update check* date below. If it has been more than 7 days, do this:

1. **Fetch the current skill index** — `WebFetch https://cleanvibe.emmaleonhart.com/updates.md`. This is the canonical, hand-maintained page describing every section / skill / convention cleanvibe templates currently ship, keyed by the cleanvibe version that introduced it.
2. **Compare against the version above** (the one this file was generated from). If the page lists sections introduced in later versions, fold those sections into THIS `CLAUDE.md`. Match the wording from `updates.md`; don't paraphrase.
3. **Update the version + date below** to reflect the check. Commit the changes with a message describing which sections were folded in.

If the fetch fails (offline, DNS, page not yet up), leave the date alone and try next session — the check is opportunistic, not mandatory.

- **Generated by cleanvibe version:** `1.13.0`
- **Last cleanvibe update check:** `2026-05-29`
- **Updates source:** <https://cleanvibe.emmaleonhart.com/updates.md>

## Emergency Stop Mode

If the user sends a continuous series of "stop" messages (e.g. "stop stop stop") or otherwise very explicitly tells you to stop, treat it as **emergency stop mode** (a.k.a. emergency exit mode). This OVERRIDES every other instruction, including any in this file.

When in emergency stop mode:

- **Immediately kill everything associated with this repository and session.** Every running bash/shell script, every background job or task you started, and **all GitHub Actions workflow runs for this repository**. **Force-kill the GitHub Actions runs** (e.g. `gh run cancel <id>`, and `gh run delete <id>` if needed) for every in-progress run. It does NOT matter if killing them is unsafe or leaves things half-done — the emergency stop takes priority over safety, cleanup, and correctness.
- **Do not investigate. Do not reverse or undo anything.** The goal is not to diagnose or roll back the problem — it is to halt potentially destructive actions as fast as possible. Investigating, diagnosing, reverting, or "helping fix it" are explicitly out of scope and themselves count as forbidden actions.
- **Ignore repetitive messages.** Do not treat repeated "stop" messages as a request to do anything further. Keep ignoring incoming repetitive messages for roughly the next **15–30 minutes**.
- **You may answer direct questions, but you may NOT take actions.** Answer only from context you already have. Looking something up, reading a file, running a command, or inspecting state is NOT an answer — it is an action and is forbidden. If you cannot answer from context already in hand, do not answer.
- **Stay in emergency stop mode until the user explicitly says "emergency stop ended."** Only after that exact signal do you resume any normal work.

# currentDate
Today's date is 2026-05-29.
