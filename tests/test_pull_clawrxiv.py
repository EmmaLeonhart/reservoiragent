"""Unit tests for the clawRxiv review-pull parsing (no network).

Covers extract_reviews() — the normalizer for the /review endpoint's several
response shapes. The puller had no tests before this; these lock in the
"never invent a review" contract (null/empty/error envelopes -> []) and the
shape-handling (wrapped, bare, list).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pull_clawrxiv_reviews import extract_reviews  # noqa: E402


def test_not_ready_envelope_returns_empty():
    # The live endpoint returns this until the AI review is generated.
    assert extract_reviews({"review": None}) == []


def test_empty_reviews_list_returns_empty():
    assert extract_reviews({"reviews": []}) == []


def test_error_envelope_is_not_a_review():
    # A {"message": "Server Error"}-style body must not be saved as a review.
    assert extract_reviews({"message": "Server Error"}) == []


def test_falsy_payloads_return_empty():
    assert extract_reviews(None) == []
    assert extract_reviews({}) == []
    assert extract_reviews([]) == []


def test_wrapped_single_review():
    rv = {"id": 1, "rating": "Weak Reject"}
    assert extract_reviews({"review": rv}) == [rv]


def test_wrapped_list_of_reviews():
    rvs = [{"id": 1, "rating": "Accept"}, {"id": 2, "rating": "Reject"}]
    assert extract_reviews({"review": rvs}) == rvs


def test_reviews_key_list():
    rvs = [{"id": 5, "rating": "Accept"}]
    assert extract_reviews({"reviews": rvs}) == rvs


def test_bare_review_object():
    # No {"review": ...} wrapper — review fields at top level (Sutra-style).
    rv = {"id": 2680, "postId": 2680, "rating": "Weak Reject", "summary": "..."}
    assert extract_reviews(rv) == [rv]


def test_top_level_list_of_reviews():
    rvs = [{"id": 1, "body": "good"}]
    assert extract_reviews(rvs) == rvs


def test_real_saved_review_shape():
    # Mirrors paper/reviews/post2680_review2680.json (a bare object).
    rv = {
        "id": 2680, "postId": 2680, "rating": "Weak Reject",
        "summary": "s", "pros": [], "cons": [], "model": "Gemini 3 Flash",
    }
    assert extract_reviews(rv) == [rv]
