# reservoiragent — Work Queue (research)

**This file is a queue of *concrete, executable steps*, not a state snapshot.** It lists what is being worked on right now. Finished work lives in `devlog.md` (a dated entry) and `git log`; longer-horizon, *abstract* work lives in `todo.md` and gets decomposed into items here when it's ready to execute. **When an item is done, delete it from this file AND append a dated entry to `devlog.md` in the same commit, then push.** Do not add checkmarks, "done" markers, or status indicators in place. If an item is still here, it is not done.

**This is a `cleanvibe research` project** — your own investigation, not a replication. Its distinctive move is an up-front **literature review** (agentic RAG) before any building, and a published, themed GitHub Pages **report** under `docs/`.

**Why this file exists:** when a planning step produces a plan, that plan is written here BEFORE execution starts, so an interrupted session can pick up from the queue rather than from chat context that may be gone.

See `CLAUDE.md` § "Workflow Rules" and § "Research workflow" for how this file, planning mode, and the task tool stay in sync.

**Three-cron playbook.** Research IS extensive work, so it runs under three local `CronCreate` jobs — **work-loop at :03** (the engine that drains `queue.md` and refills it from `todo.md`), **auto-flush at :15** (commit/push backstop), and **status-report at :42** (heartbeat). On a fresh session they are **started** as the opening step (bootstrap step 1 below); on a mid-session **large-scale re-fill** of this queue the FIRST item worked is instead to **kill** the already-running crons. Either way the **last two items are always pinned at the tail** (see `## Always last`). Entering planning mode also disables the crons; their restart lives at the end of the queue. (See `CLAUDE.md` § "Autonomous productivity loop — the three-cron playbook".)

---

## Active — Phase H: port to Hermes + make the behaviour real (A–E)

Rounds 1–3 validated the *mechanisms on GPT-2*. This phase moves to the real target —
the smallest Hermes (**NousResearch/Hermes-3-Llama-3.2-3B**, Llama-3.2 arch) — and builds
the conditions for the desired behaviour (statefulness that *does* something; meaningful
silence; responding without a system prompt over time). The user asked for A–E **in
order**. Preconditions confirmed: RTX 4070 (8.6 GB VRAM), bitsandbytes 0.49.2 (4-bit
works on Windows), peft, accelerate, 768 GB free. Work top to bottom; **delete each item
in the same commit that completes it and append a dated entry to `devlog.md`**; push; let
CI run. Hard rails: TDD where there is logic; never fake/weaken a test; a real defect →
strict `xfail` / documented blocker; verify CI green, not just local; name hard/unbuilt
things plainly; the model-download/GPU steps are local-only (torch-gated tests skip in CI).

**Crons:** the three (work-loop :03, auto-flush :15, status-report :42) are running and
kept running through this re-fill (atomic edit). The pinned `## Always last` keeps them alive.

_**(C) resolved — and it's the core claim, demonstrated.** The additive injection failed
(reservoir ignored, chance recall), but the user-chosen **content-addressable KV-append
fix** (reservoir → attendable prefix tokens, `kv_live.py`) gives **100% cross-context
recall** vs **chance (0.17)** for the stateless baseline. The Reservoir Agent's
statefulness *does* the desired thing when the reservoir is **attended to, not added**.
See `FINDINGS.md` "## C: cross-pass recall" + `docs/crosspass.png`._

1. **(E) Fork the real Hermes harness (tool-calling + agentic loop).** Build a fork of the
   actual Nous Hermes harness — Hermes tool-call/ChatML formatting and the agentic loop —
   wrapping the always-alive runtime (prompted + unprompted passes, the trained gate),
   preserving Hermes' tool-call behaviour (regression vs vanilla Hermes is an explicit
   check). This is the runtime the serious training/eval runs in. Name plainly whatever
   part of the real harness is too large for this session and document it. Commit.

**Reality note (kept honest):** A+B are bounded engineering. **C is research-substantial**
(it's the condition that makes the idea work), **D depends on training-data design**, and
**E is a real systems build**. They are sequenced A→E per the user; each lands as real,
tested, committed work or a precise documented blocker — never a faked "it works".

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
