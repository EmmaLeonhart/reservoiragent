# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Why this file exists:** when a planning step produces a plan, that plan is written here BEFORE execution starts, so an interrupted session can pick up from the queue rather than from chat context that may be gone.

See `CLAUDE.md` § "Workflow Rules" and § "Research workflow" for how this file, planning mode, and the task tool stay in sync.

**Three-cron playbook.** Research IS extensive work, so it runs under three local `CronCreate` jobs — **work-loop at :03** (the engine that drains `queue.md` and refills it from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). On a fresh session they are **started** as the opening step (bootstrap step 1 below); on a mid-session **large-scale re-fill** of this queue the FIRST item worked is instead to **kill** the already-running crons. Either way the **last two items are always pinned at the tail** (see `## Always last`). Entering planning mode also disables the crons; their restart lives at the end of the queue. (See `CLAUDE.md` § "Autonomous productivity loop — the three-cron playbook".)

---

## Active — Implementation: feasibility + dynamics study

Bootstrap is complete (see `devlog.md`). This is the real research queue, decomposed
from `todo.md` §B near-term (feasibility + dynamics). Work top to bottom; **delete each
item in the same commit that completes it and append a dated entry to `devlog.md`**;
push; let CI run. Build under `src/`, entry point `scripts/run.py`, metrics → `results/`,
figures → `docs/`. Hold the hard rails: TDD where there is logic; never fake or weaken a
test; a real defect → strict `xfail` or a documented blocker; verify CI green, not just
local; name compute-blocked work plainly.

**Crons:** the three crons (work-loop :03, auto-flush :15, status-report :42) are
already running and are kept running through this re-fill (it is written atomically in
one edit, so there is no half-written-queue window to protect against). The pinned
`## Always last` keeps them alive. **The work-loop and the one-shot 8h kickoff cron
(`0bacbec1`, ~2026-05-30 04:24 local) both draw from the top of this queue** — the 8h
cron is a guaranteed escalation/kickoff point, not the only start.

1. **Reservoir observability / dynamics metrics (TDD).** `src/reservoir/metrics.py`:
   state variance, saturation fraction (|r|>0.99), effective rank (participation ratio),
   and trajectory distinguishability between two input histories. Tests on synthetic
   signals with known answers. Commit.

2. **Spectral-radius dynamics sweep on synthetic input.** `scripts/run.py` drives the
   reservoir across a grid of ρ (and reservoir size K) with synthetic input streams;
   logs metrics to `results/sweep_synthetic.json`; renders figures to `docs/`. Identify
   the healthy regime (H2) — non-saturating, non-exploding, distinguishable trajectories;
   check whether the optimum sits at the classical edge-of-chaos prior. Commit.

3. **Model surgery: inject the reservoir into a small pretrained transformer (H1).**
   Hook a mid-depth layer of GPT-2-small (HF `transformers`): read its hidden states into
   the reservoir via fixed `W_in`, write `r(t)` back into that layer's key/value sequence
   via readout `W_out`. **Regression test (H1):** with `W_out=0` the base model's logits
   are unchanged vs vanilla GPT-2 (graceful degradation). Commit.

4. **Dynamics sweep on REAL attention streams + write up.** Drive the sweep (item 4) with
   real GPT-2 mid-layer activations; capture `results/sweep_real.json` + figures. Write
   `FINDINGS.md` (question → method → results → limitations) and fill the `docs/` Findings
   section + pillar 3 with the headline dynamics result. Commit; verify Pages updates.

5. **Ambitious reach (per `todo.md` §B note — "just to see if we can").** Attempt the
   harder build beyond local-compute comfort and **report honestly how far it gets**:
   the minimal harness fork (two forward passes without reinitialising the reservoir =
   the genuine-time-axis proof-of-concept), and, if it runs at all, a tiny N-seed (3–5)
   reservoir comparison. Anything compute-blocked → a precise documented blocker / strict
   `xfail`, never a faked or weakened result.

**Stop condition (then hand back):** the research question has a defensible answer or a
clearly-reported partial result; `FINDINGS.md` and the published `docs/` report reflect
it; `queue.md` is empty (refill from `todo.md` if more is in scope); repo online with
green CI/Pages.

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
