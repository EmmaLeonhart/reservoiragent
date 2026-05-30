"""Unit tests for the clawRxiv submit routing / revise self-heal logic.

These pin the load-bearing resubmission behaviour so it cannot silently
regress to the old `supersedes` field (which clawRxiv now 409s):

  - first-ever submission (no pinned id) -> POST /api/posts (create)
  - a pinned id -> POST /api/posts/{id}/revise (revise)
  - 409 on revise -> follow data.duplicateId and revise THAT
  - a successful create while an id is pinned -> orphan refusal (exit 1)

No network: the HTTP layer (`_post_json`) is monkeypatched to record the
URLs hit and to raise the clawRxiv error shapes we recover from.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "submit_clawrxiv_paper", ROOT / "scripts" / "submit_clawrxiv_paper.py")
sub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sub)

PAYLOAD = {"title": "t", "abstract": "a", "content": "c", "tags": [], "human_names": []}


def _record(monkeypatch):
    """Patch _post_json to record hit URLs; returns (urls, set_handler)."""
    urls: list[str] = []
    handler = {"fn": lambda url, payload, key: {"id": 999}}

    def fake_post(url, payload, api_key):
        urls.append(url)
        return handler["fn"](url, payload, api_key)

    monkeypatch.setattr(sub, "_post_json", fake_post)
    return urls, handler


def test_first_submission_creates(monkeypatch):
    urls, _ = _record(monkeypatch)
    result = sub.submit("key", PAYLOAD, pinned=None)
    assert result == {"id": 999}
    assert urls == ["https://clawrxiv.io/api/posts"]  # create, not revise


def test_pinned_id_revises(monkeypatch):
    urls, _ = _record(monkeypatch)
    result = sub.submit("key", PAYLOAD, pinned=2680)
    assert result == {"id": 999}
    assert urls == ["https://clawrxiv.io/api/posts/2680/revise"]  # revise endpoint


def test_409_follows_duplicate_id(monkeypatch):
    urls, handler = _record(monkeypatch)

    def h(url, payload, key):
        if url.endswith("/2680/revise"):
            raise sub.SupersedeConflict("409", body={"data": {"duplicateId": 2618}})
        return {"id": 2618}

    handler["fn"] = h
    result = sub.submit("key", PAYLOAD, pinned=2680)
    assert result == {"id": 2618}
    assert urls == [
        "https://clawrxiv.io/api/posts/2680/revise",   # first try
        "https://clawrxiv.io/api/posts/2618/revise",   # self-heal onto canonical
    ]


def test_successful_create_while_pinned_is_orphan_refused(monkeypatch):
    """A 404 on revise + a *successful* create = a minted orphan, not a
    revision. Must refuse (exit code 1), not silently start a new chain."""
    _, handler = _record(monkeypatch)

    def h(url, payload, key):
        if "/revise" in url:
            raise sub.ReviseNotFound("404")
        return {"id": 7777}  # create succeeds -> orphan

    handler["fn"] = h
    result = sub.submit("key", PAYLOAD, pinned=2680)
    assert result == 1  # orphan refused


def test_404_then_dedup_409_revises_canonical(monkeypatch):
    """404 on revise, create-probe 409s naming the canonical -> revise it."""
    urls, handler = _record(monkeypatch)

    def h(url, payload, key):
        if url.endswith("/2680/revise"):
            raise sub.ReviseNotFound("404")
        if url == "https://clawrxiv.io/api/posts":  # create-probe
            raise sub.SupersedeConflict("409", body={"data": {"duplicateId": 2618}})
        return {"id": 2618}  # revise of canonical

    handler["fn"] = h
    result = sub.submit("key", PAYLOAD, pinned=2680)
    assert result == {"id": 2618}
    assert urls[-1] == "https://clawrxiv.io/api/posts/2618/revise"


def test_abstract_within_clawrxiv_bounds():
    assert 100 <= len(sub.ABSTRACT) <= 5000
    assert len(sub.TITLE.split()) >= 5
