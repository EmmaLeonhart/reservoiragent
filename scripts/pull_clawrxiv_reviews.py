"""Fetch clawRxiv AI peer-review(s) for our submitted paper into paper/reviews/.

Mirrors the Sutra repo's review-pull mechanism, scoped to this project's single
paper. Reads the post id from paper/.post_id, GETs
`https://clawrxiv.io/api/posts/{id}/review` (singular — verified against the
Sutra puller and the live API: singular returns 200, plural /reviews 404s), and
writes the review to paper/reviews/. Idempotent: re-running does nothing once a
review is saved. A 404 means the AI review hasn't been generated yet — treated
as "try again later", not an error.

Filename scheme: `post{post_id}_review{review_id}.json`, keyed by the review's
own id. This dedups multiple reviews per post (e.g. a fresh review after a
/revise) by review id rather than by a running version counter — strictly more
precise than Sutra's `v{N}_post{id}_review` scheme for our single-paper case.

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
from typing import Any

API_BASE = "https://clawrxiv.io"
ROOT = Path(__file__).resolve().parent.parent
REVIEWS_DIR = ROOT / "paper" / "reviews"

# Keys that mark a payload (or an item) as an ACTUAL review rather than an empty
# "not generated yet" envelope. Mirrors the Sutra puller's readiness check
# (paper_submit_and_fetch.fetch_review), which accepts review/body/content/
# rating — so a bare-object response with the review fields at top level (no
# {"review": ...} wrapper) isn't silently missed.
REVIEW_MARKERS = ("rating", "recommendation", "review", "body", "content", "summary")


def _looks_like_review(obj: Any) -> bool:
    """True if `obj` is a dict carrying at least one review content field."""
    return isinstance(obj, dict) and any(obj.get(k) for k in REVIEW_MARKERS)


def extract_reviews(payload: Any) -> list[dict[str, Any]]:
    """Normalize the /review endpoint's several response shapes into a list of
    review dicts. Returns [] when nothing is ready (never invents a review).

    Handled shapes:
      - {"review": null}                  -> []      (not generated yet)
      - {"review": {...}}                 -> [inner]
      - {"review": [..]}                  -> inner list
      - {"reviews": [..]}                 -> that list
      - a bare review object {rating:..}  -> [it]
      - a list of review objects          -> that list
      - anything else / falsy            -> []
    """
    if isinstance(payload, list):
        return [r for r in payload if r]
    if isinstance(payload, dict):
        if "review" in payload:
            body = payload["review"]
            items = body if isinstance(body, list) else ([body] if body else [])
            return [r for r in items if r]
        if "reviews" in payload:
            return [r for r in (payload["reviews"] or []) if r]
        # No envelope key — treat the dict itself as a review only if it has
        # review content (so a {"message": "Server Error"} envelope is ignored).
        return [payload] if _looks_like_review(payload) else []
    return []


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

    items = extract_reviews(review)
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
