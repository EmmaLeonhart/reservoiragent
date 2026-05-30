"""Fetch clawRxiv AI peer-review(s) for our submitted paper into paper/reviews/.

Mirrors the Sutra repo's review-pull mechanism, scoped to this project's single
paper. Reads the post id from paper/.post_id, GETs
`https://clawrxiv.io/api/posts/{id}/review` (singular — verified against the
Sutra puller and the live API), and writes the review to paper/reviews/.
Idempotent: re-running does nothing once a review is saved. A 404 means the AI
review hasn't been generated yet — treated as "try again later", not an error.

Required env: CLAWRXIV_API_KEY (the same secret the submit workflow uses).
Exit 0 even when no review is ready yet — that's "try again later", not an error.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://clawrxiv.io"
ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR = ROOT / "paper" / "reviews"


def _get(url: str, key: str, timeout: int = 30):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}"}, method="GET"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> int:
    key = os.environ.get("CLAWRXIV_API_KEY")
    if not key:
        print("ERROR: CLAWRXIV_API_KEY not set")
        return 1

    post_id_file = ROOT / "paper" / ".post_id"
    if not post_id_file.exists():
        print("No paper/.post_id — nothing submitted yet; nothing to pull.")
        return 0
    post_id = post_id_file.read_text().strip()

    try:
        review = _get(f"{API_BASE}/api/posts/{post_id}/review", key)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"No review ready yet for post {post_id} (404 — try again later).")
            return 0
        print(f"HTTP {e.code} fetching review for post {post_id}: "
              f"{e.read().decode()[:300]}")
        return 0

    # The endpoint returns {"review": null} (HTTP 200) until the AI review is
    # generated, then {"review": {...}} (or a bare object / a list).
    if isinstance(review, dict) and "review" in review:
        body = review["review"]
        items = body if isinstance(body, list) else ([body] if body else [])
    elif isinstance(review, dict) and "reviews" in review:
        items = review["reviews"] or []
    else:
        items = [review] if review else []
    items = [r for r in items if r]
    if not items:
        print(f"No review ready yet for post {post_id} (server returned null).")
        return 0

    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for i, rv in enumerate(items, start=1):
        rid = rv.get("id", i) if isinstance(rv, dict) else i
        out = REVIEWS_DIR / f"post{post_id}_review{rid}.json"
        if out.exists():
            continue
        out.write_text(json.dumps(rv, indent=2, ensure_ascii=False), encoding="utf-8")
        wrote += 1
        print(f"Saved {out.relative_to(ROOT)}")

    print(f"Done: {wrote} new review file(s) for post {post_id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
