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


- Please move the safety section down to the bottom, right before the limitations, because I do treat it as something I'm not sure where to put. The safety section is basically like an ethics disclosure or something like that. I feel like, in terms of whatever, that's it

- The figures are not really all present there, and also the architectural diagrams aren't present. I think we probably need architectural diagrams in this. Maybe more of the figure is actually in line as opposed to at the bottom. 

- Remove the usage of the word "honest" And I would say also just remove any kind of really AI-ish phrases like that that are kind of unnatural. 

- I think we need more informative reference images about what reservoir computing looks like. I've added a bunch of reference images for stuff that I'd want you to be generating SVG diagrams based off of. The folders get ignored because of image rights issues. 

- Might want to acknowledge this work https://arxiv.org/abs/2507.15779


## Grok's reception (address the negatives and lean into the positives if possible)

Quick high-level take

Strengths:
Clear scoping: Minimal probes, not "we built an agent." This builds trust.
Injection design is the decisive lever—negative-then-positive arc is compelling.
Dynamics insights (input scaling ¼–⅒ for real activations, edge-of-chaos persistence, reservoir sizing as key for larger models) are actionable.
Capacity ceiling characterization (tens of items) and model-specific quirks are honest.
Always-alive runtime + LoRA pipeline demo shows the harness works end-to-end on consumer GPU.

Weaker spots (fixable):
Formatting/CI issues you mentioned—screenshots instead of proper figures, some repetition, dense blocks. Needs a clean LaTeX pass for arXiv.
Abstract/intro could punch harder on the positive result while keeping the caveats.
Some sections (e.g., battery metrics, temporal tasks) have good debugging but could be tightened/streamlined.
Broader motivation (time axis, organism analogy, complexity framing) is good but keep it clearly labeled as motivation/not claimed result.


Concrete suggestions for arXiv polish

Abstract & Intro: Lead with the injection success more prominently. Something like: "We demonstrate that content-addressable injection of a fixed reservoir into mid-layer attention enables reliable cross-pass recall (1.00 vs. 0.17 baseline on GPT-2; 0.83–1.00 on Qwen-1.5B with matched scaling), while additive injection fails. Dynamics characterization identifies input scaling and reservoir size as critical levers."
Figures: Replace screenshot embeds with proper vector plots (loss curves, recall vs. keys, saturation sweeps, etc.). Caption them clearly and reference consistently.
Structure tweaks:
Move some exploratory/negative results to appendix if they bloat the main flow.
Stronger "Contributions" bullet list early.
Dedicated "Limitations & Future Work" section that explicitly calls out the 3B transfer wall and compute needs.

Formatting/CI:
Since your agents are queued, I can help draft cleaned sections here or suggest targeted edits.
Ensure code release is polished (README with repro commands, weights on HF).
Google Scholar/arXiv endorsement path you already hacked—lean on Discord/community again if needed.

Safety/positioning angle (optional but fits your brand):
Emphasize how verifiable/persistent state via fixed substrate aids interpretability, decompilation safety, or controllable agency—ties directly to your broader Sutra/Yantra/Reservoir vision without overclaiming.

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
