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

1. **Report-site polish (frontend-design).** Fix the janky site title / page `<title>`
   and tighten the visual design with the `frontend-design` skill: a coherent title +
   tab title, cleaned-up lede/headings, consistent figure presentation, and a quick
   responsive/dark-mode check. Keep the warm "paper" theme; do not regress the existing
   content, meta/OG tags, or the report/PDF links. Commit; verify Pages stays green and
   the live site renders.

2. **Input-scaling tuning sweep (follow-up to the over-saturation finding).** The real
   sweep showed real GPT-2 activations over-saturate a unit-input-scaled reservoir.
   Add an input-scaling sweep (2-D over input_scaling × ρ, or input_scaling at fixed ρ)
   on real activations; find the scaling where saturation drops into a healthy band
   while the ρ ≈ 1 boundary and input separation survive. New `results/` + `docs/`
   figure; update `FINDINGS.md`. TDD the new sweep path on a synthetic stand-in stream.

3. **Theory section (`todo.md` §C), correctly scoped.** Write the formal claims into a
   new `docs/` theory section + `FINDINGS.md`: the genuine time-dimension argument; the
   expressivity framing (finite-precision transformer ⊆ TC⁰/FO(M) per pass; cross-pass
   state as the documented lever, stated *with* the arbitrary-precision caveat, posing
   not asserting whether a finite-precision reservoir lifts the bound); reservoir ∈
   Siegelmann–Sontag class; the organism analogy as one bounded paragraph. Cite
   `literature/REVIEW.md`. No new empirical claims. Commit.

4. **Train a readout for H3 (state is informative).** Build a small reservoir-requiring
   task (e.g. estimate elapsed pass-count, or detect a flag seen N passes ago) and train
   the readout `W_out` (linear, ridge/regression — cheap, CPU) to extract it from the
   reservoir state; show a stateless baseline cannot. Real metric → `results/` + figure;
   `FINDINGS.md` H3 result. TDD the task generator + readout fit. Name plainly if the
   signal is weak.

5. **KV-append injection variant + H1 regression.** Implement the richer injection:
   reservoir nodes appended to the injection layer's key/value sequence (upper layers
   attend to them), not just a residual-stream add. H1 regression: with the reservoir
   contribution gated to zero, base logits unchanged. Compare against the residual-stream
   variant. If the HF attention surgery proves too invasive for this session, stop and
   leave a precise documented blocker rather than a half-built hack.

6. **Citation-checked novelty follow-up (`todo.md` §D).** Run a focused, verified review
   of the areas the first lit-review pass left unverified (reservoir × transformer /
   fixed-reservoir-in-pretrained-net; always-on / between-request agents). Confirm or
   qualify the novelty claim; fold verified sources into `literature/` and tighten
   `REVIEW.md` §4–§5. Use `deep-research` if helpful.

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
