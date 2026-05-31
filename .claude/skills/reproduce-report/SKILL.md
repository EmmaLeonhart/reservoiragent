---
name: reproduce-report
description: Reproduce the Reservoir Attention Network (RAN) results, figures, report site (docs/) and report.pdf from the code in this repo. Use when someone asks to replicate/reproduce the findings, regenerate a figure, rebuild the GitHub Pages site or PDF, or verify a result before it goes in the paper.
---

# Reproduce the Reservoir Attention Network (RAN) report (replication skill)

This skill is the reproduction recipe that backs the published site and the
paper. Every headline claim in `FINDINGS.md` / the `docs/` site must be
regenerable from the steps here. If a number on the site or in the paper can't
be reproduced by this skill, that is a defect — fix the claim or the code, never
loosen the recipe.

`FINDINGS.md` is the source of truth for the exact numbers. This skill is the
source of truth for *how to regenerate them*. Keep the two in sync: when a
result changes, update both `FINDINGS.md` and (if the command changed) this file,
in the same commit.

## 0. Environment

```
pip install -e ".[dev]"          # core + tests (CPU-only path)
pip install -e ".[dev,models]"   # adds torch/peft/transformers/bitsandbytes (GPU path)
```

- CPU-only is enough for: the echo-state core, the dynamics sweeps, metrics,
  the tasks, and the full unit-test suite. torch/peft/Hermes tests **skip**
  without the `models` extra.
- GPU (CUDA) is required only for the real model runs (GPT-2 fine-tune, Hermes
  4-bit, the cross-pass LM training). Hardware on record: RTX 4070 (~8.6 GB);
  bitsandbytes 4-bit works on Windows; Hermes-3-Llama-3.2-3B is cached locally.
- Use `python` (not `python3`) on this machine; tests want `PYTHONPATH=src`.

## 1. Tests first (gate)

```
PYTHONPATH=src python -m pytest
```

All non-torch tests must pass before trusting any figure. CI runs this on every
push (`.github/workflows/ci.yml`) — **verify CI green, not just local**
(`gh run list --branch main`).

## 2. Regenerate results + figures

The entry point is `scripts/run.py <subcommand>`; metrics land in `results/*.json`
and figures in `docs/*.png`. Known subcommands (confirm with `python scripts/run.py --help`):

| Result (FINDINGS section) | Command | Artifact(s) |
|---|---|---|
| H2 dynamics — synthetic | `python scripts/run.py sweep` | `results/sweep_synthetic.json`, `docs/sweep_synthetic.png` |
| H2 dynamics — real GPT-2 activations | `python scripts/run.py sweep-real` | `results/sweep_real.json`, `docs/sweep_real.png` |
| H2 input-scaling sweet spot | `python scripts/run.py sweep-scaling` | `results/sweep_scaling.json`, `docs/sweep_scaling.png` |
| H3 delay-memory readout | `python scripts/run.py h3` | `results/h3_memory.json`, `docs/h3_memory.png` |
| Cross-pass recall (the core claim) | `python scripts/run.py crosspass --mode kv` | `results/crosspass.json`, `docs/crosspass.png` |
| Trained silence policy (D) | `python scripts/run.py silence` | `results/silence_gate.json`, `docs/silence.png` |
| N-seed selection + proxy | `python scripts/run.py nseed-select` | `results/nseed_select.json`, `docs/nseed*.png` |
| GPU LoRA fine-tune | `python scripts/run.py finetune` | `results/finetune.json` |
| H1 non-destruction on Hermes (4-bit) | `python scripts/hermes_h1.py` | `results/hermes_h1.json` |

Notes:
- `crosspass --mode kv` is the content-addressable KV-prefix path (100% on GPT-2
  vs 0.17 chance). The additive-injection variant is the documented negative.
- The Hermes cross-pass *transfer* is the open GPU thread (see `todo.md`); it is
  NOT yet reproducible at the GPT-2 success level — say so plainly, don't imply
  otherwise on the site/paper.

## 3. Rebuild the site + PDF

`docs/` is the published GitHub Pages site (`docs/index.html`, the `docs/*.png`
figures, the `docs/diagram-*.svg` architecture diagrams, and the built
`docs/report.pdf`). `.github/workflows/pages.yml` deploys `docs/` and builds
`report.pdf` from `FINDINGS.md` on push to `main`. To reproduce:

1. Regenerate any changed figures (section 2) so `docs/*.png` are current.
2. Edit `FINDINGS.md` (the report/paper text) — it is what the PDF is built from.
3. Edit `docs/index.html` for the site narrative; keep the warm "paper" theme
   chrome, change only content.
4. Push to `main`; confirm both the `pages` and `ci` workflow runs go green
   (`gh run list`). The live site is https://reservoir.emmaleonhart.com/.

## 4. Diagrams

Architecture/runtime SVGs live in `docs/diagram-architecture.svg`,
`docs/diagram-residual-reservoir.svg`, `docs/diagram-runtime.svg` (themed for the
site). Source/raw diagrams and the re-theme script are under `data_lake/`
(`data_lake/retheme_diagrams.py`, `data_lake/build_residual_reservoir_svg.py`).

## 5. Novelty / prior-art positioning (for the paper)

`literature/REVIEW.md` is the synthesized survey; `literature/sources.md` the
source notes; `literature/novelty_recheck.md` records the searched-prior-art
sweep. The claim is **searched-prior-art**, not absolute novelty. Nearest
neighbours to position against: Reservoir Transformers (2021, frozen forward-
stack layers, no cross-pass axis), Echo State Transformer / FreezeTST (2025,
reservoir-as-working-memory within a sequence), and the test-time-memorization
line — **Titans** (arXiv 2501.00663, 2025) — whose memory is *trained at test
time* vs this project's *fixed random* reservoir with only a readout trained.
Re-run the sweep before any hard novelty claim in a submitted paper.

## 6. clawRxiv submission + peer-review loop (publish / revise)

The paper is published to clawRxiv and accrues AI peer reviews. This is wired in
`.github/workflows/clawrxiv.yml` + two scripts, mirroring the Sutra repo's
mechanism. The submission state lives in `paper/` (`.post_id`, `.paper_id`,
`.last_submitted_hash`, and `reviews/`). Current live post: **2680**
(paper_id 2605.02680).

- **Submit / revise** — `scripts/submit_clawrxiv_paper.py` (manual
  `workflow_dispatch`). It POSTs `FINDINGS.md` + this SKILL.md to clawRxiv.
  **Revisions use `POST /api/posts/{id}/revise`, NOT the old `supersedes`
  field.** clawRxiv migrated revisions to `/revise`; the old
  `POST /api/posts` + `{"supersedes": id}` body now returns **HTTP 409**
  ("already been revised" / "duplicate detected"). The script:
  - first-ever submission (no `paper/.post_id`) → `create_post` (POST /api/posts);
  - a pinned `.post_id` → `revise_post` (POST /api/posts/{id}/revise);
  - 409 on revise → follow `data.duplicateId` to the canonical post and revise it,
    re-pinning `.post_id` (deterministic self-heal of a drifted id);
  - 404 on revise (a clawRxiv server-side bug on some chains) → probe `create_post`
    to elicit the 409 that names the canonical post;
  - **STOP-NEW-CHAINS guard:** with a `.post_id` pinned, a *successful* create is an
    orphan, not a revision — the script refuses to pin to it, keeps `.post_id` at the
    chain tip, and exits 1 so CI goes red. This is the load-bearing resubmission
    logic; it is unit-tested in `tests/test_submit_clawrxiv.py` (no network).
- **Pull reviews** — `scripts/pull_clawrxiv_reviews.py` (every 30 min + on push to
  `paper/**`). GETs `/api/posts/{id}/review` and commits any new review into
  `paper/reviews/`. A 404 / `{"review": null}` means "not generated yet" (exit 0,
  not an error). A real review (`paper/reviews/post2680_review2680.json`, a
  "Weak Reject" from Gemini 3 Flash) confirms the pull side works end-to-end.

To resubmit a revision: edit `FINDINGS.md` (and keep `TITLE`/`ABSTRACT` in
`scripts/submit_clawrxiv_paper.py` in sync), commit, then **Actions → "clawRxiv —
submit paper + pull AI reviews" → Run workflow** (or `gh workflow run
clawrxiv.yml`). It auto-revises the pinned `.post_id`. The 30-min schedule then
pulls the new review.

## Hard rails (same as the repo's)

Never fake a result or a figure. Never weaken/skip a test to make a number look
right. Never write a claim onto the site or into the paper that this skill can't
reproduce on command. A real defect → `xfail` or a documented blocker, never a
loosened assertion.
