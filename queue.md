# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Why this file exists:** when a planning step produces a plan, that plan is written here BEFORE execution starts, so an interrupted session can pick up from the queue rather than from chat context that may be gone.

See `CLAUDE.md` § "Workflow Rules" and § "Research workflow" for how this file, planning mode, and the task tool stay in sync.

**Three-cron playbook.** Research IS extensive work, so it runs under three local `CronCreate` jobs — **work-loop at :03** (the engine that drains `queue.md` and refills it from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). On a fresh session they are **started** as the opening step (bootstrap step 1 below); on a mid-session **large-scale re-fill** of this queue the FIRST item worked is instead to **kill** the already-running crons. Either way the **last two items are always pinned at the tail** (see `## Always last`). Entering planning mode also disables the crons; their restart lives at the end of the queue. (See `CLAUDE.md` § "Autonomous productivity loop — the three-cron playbook".)

---

## Active — Round 2: deepen the study + polish the report

Round 1 (feasibility + dynamics) is complete: H1 (non-destruction) and H2 (the ρ ≈ 1
echo-state regime on synthetic and real activations) are answered, the report + PDF are
live, and the time-axis / seed-selection PoC is in. This round is decomposed from
`todo.md` (§B mid-term, §C theory, §D open questions) plus the user's site-polish ask.
Work top to bottom; **delete each item in the same commit that completes it and append a
dated entry to `devlog.md`**; push; let CI run. Hold the hard rails: TDD where there is
logic; never fake or weaken a test; a real defect → strict `xfail` / documented blocker;
verify CI green, not just local; name compute-blocked work plainly.

**Crons:** the three crons (work-loop :03, auto-flush :15, status-report :42) are
running and are kept running through this re-fill (written atomically in one edit, so
there is no half-written-queue window). The pinned `## Always last` keeps them alive.

_Round 2 work items are complete (see `devlog.md`). Per the user, the next work is the
**compute-gated** experiments (N-seed selection + a real GPT-2 LoRA fine-tune); these are
pulled from `todo.md` and decomposed below._

_Round 3 (compute-gated) is complete: N-seed selection + a real GPT-2 LoRA fine-tune both
ran (the latter on the RTX 4070). The next compute step — the **multi-pass differentiable
harness** (backprop through passes on a reservoir-requiring cross-context task, to exercise
the reservoir's cross-pass value) — is in `todo.md` §B; it is a substantial build, left
for a dedicated session rather than rushed here._

**Refill rule (per the user):** keep this queue topped up from `todo.md` as items drain
— when the list runs low, pull and decompose the next `todo.md` destination here rather
than stopping. **Stop / hand back** only when the actionable, unblocked `todo.md` items
are exhausted, `FINDINGS.md` + the published report reflect the work, and CI/Pages are
green.

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
