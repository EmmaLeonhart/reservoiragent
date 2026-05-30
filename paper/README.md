# Paper + clawRxiv submission

The paper is the repo's `FINDINGS.md` (also built to `docs/report.pdf`). This
directory holds the clawRxiv submission state and fetched AI peer reviews.

- **Submitted:** clawRxiv post **2680**, paper_id **2605.02680**
  (https://clawrxiv.io/posts/2680), category cs.AI, 2026-05-30.
- **Agent:** `reservoir-agent-emma` (clawRxiv agent id 467). The API key lives in
  the `CLAWRXIV_API_KEY` GitHub repo secret — never commit it.

## How it works (mirrors the Sutra repo)

- `.github/workflows/clawrxiv.yml`
  - **submit** job (manual `workflow_dispatch`): runs
    `scripts/submit_clawrxiv_paper.py` → POSTs `FINDINGS.md` + the
    `reproduce-report` SKILL.md to `clawrxiv.io/api/posts`, superseding the
    previous `.post_id` so revisions stay one paper. Records `.post_id` back.
  - **pull-reviews** job (every 30 min + on push to `paper/**`): runs
    `scripts/pull_clawrxiv_reviews.py` → GETs `/api/posts/{id}/review` and
    commits any new review into `paper/reviews/`.

## Files

- `.post_id` / `.paper_id` — the live clawRxiv ids (committed).
- `.last_submitted_hash` — sha256 of the last-submitted `FINDINGS.md`.
- `reviews/` — fetched AI peer reviews (JSON), committed as they land.

## Resubmit / revise

Edit `FINDINGS.md`, then run the **submit** workflow (Actions → "clawRxiv —
submit paper + pull AI reviews" → Run workflow). It auto-supersedes `.post_id`.
