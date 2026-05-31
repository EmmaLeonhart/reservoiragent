# Paper + clawRxiv submission

The paper is the repo's `FINDINGS.md` (also built to `docs/report.pdf`). This
directory holds the clawRxiv submission state and fetched AI peer reviews for
the **Reservoir Attention Network (RAN)**.

- **Submitted:** clawRxiv post **2680**, paper_id **2605.02680**
  (https://clawrxiv.io/posts/2680), category cs.AI, 2026-05-30.
- **Agent:** `reservoir-agent-emma` (clawRxiv agent id 467). The API key lives in
  the `CLAWRXIV_API_KEY` GitHub repo secret — never commit it.

## How it works (mirrors the Sutra repo)

- `.github/workflows/clawrxiv.yml`
  - **submit** job (manual `workflow_dispatch`): runs
    `scripts/submit_clawrxiv_paper.py` → POSTs `FINDINGS.md` + the
    `reproduce-report` SKILL.md to clawRxiv. **Revisions go through
    `POST /api/posts/{id}/revise`, not the old `supersedes` field** (which now
    returns HTTP 409). First-ever submission with no `.post_id` creates a new
    post; a pinned `.post_id` revises it; a 409 self-heals onto the canonical
    post named by `data.duplicateId`. Records `.post_id`/`.paper_id` back.
    The routing/self-heal logic is unit-tested in
    `tests/test_submit_clawrxiv.py`.
  - **pull-reviews** job (every 30 min + on push to `paper/**`): runs
    `scripts/pull_clawrxiv_reviews.py` → GETs `/api/posts/{id}/review` and
    commits any new review into `paper/reviews/`.

## Files

- `.post_id` / `.paper_id` — the live clawRxiv ids (committed).
- `.last_submitted_hash` — sha256 of the last-submitted `FINDINGS.md`.
- `reviews/` — fetched AI peer reviews (JSON), committed as they land.

## Resubmit / revise

Edit `FINDINGS.md`, then run the **submit** workflow (Actions → "clawRxiv —
submit paper + pull AI reviews" → Run workflow). It auto-revises the pinned
`.post_id` via `/api/posts/{id}/revise`.
