"""Tests for the installer model registry (pure logic — sort, default, merge, fallback).

The live HF query is network-gated and not tested here; these cover the deterministic
selection logic the installer menu depends on.
"""
import pytest

from reservoir.installer.registry import (
    sort_by_recency, default_model, merge_models, BUNDLED_MODELS, list_models,
)


def _m(repo_id, last_modified):
    return {"repo_id": repo_id, "last_modified": last_modified}


def test_sort_by_recency_newest_first():
    models = [_m("a", "2026-01-01T00:00:00"), _m("c", "2026-05-30T00:00:00"),
              _m("b", "2026-03-15T00:00:00")]
    assert [m["repo_id"] for m in sort_by_recency(models)] == ["c", "b", "a"]


def test_sort_handles_missing_timestamps_last():
    models = [_m("a", None), _m("b", "2026-05-30T00:00:00")]
    # a model with no timestamp must not crash and sorts after dated ones
    assert sort_by_recency(models)[0]["repo_id"] == "b"


def test_default_model_is_most_recent_when_none_recommended():
    models = [_m("old", "2026-01-01T00:00:00"), _m("new", "2026-05-30T00:00:00")]
    assert default_model(models)["repo_id"] == "new"


def test_default_model_privileges_recommended_over_recency():
    # the privileged "best" of a batch is the default even if a worse model is newer
    best = {"repo_id": "best", "last_modified": "2026-01-01T00:00:00", "recommended": True}
    newer_worse = {"repo_id": "newer", "last_modified": "2026-05-30T00:00:00",
                   "recommended": False}
    assert default_model([best, newer_worse])["repo_id"] == "best"


def test_default_model_most_recent_recommended_when_several():
    a = {"repo_id": "a", "last_modified": "2026-01-01T00:00:00", "recommended": True}
    b = {"repo_id": "b", "last_modified": "2026-05-30T00:00:00", "recommended": True}
    assert default_model([a, b])["repo_id"] == "b"


def test_all_models_retained_bad_ones_are_signal():
    # the registry never drops models — the bad seeds are kept as signal
    batch = [_m("seed0", "2026-05-30T00:00:00"), _m("seed1", "2026-05-30T00:00:00"),
             _m("seed2", "2026-05-30T00:00:00")]
    assert len(sort_by_recency(batch)) == 3


def test_default_model_empty_is_none():
    assert default_model([]) is None


def test_merge_dedupes_by_repo_id_preferring_hf():
    bundled = [_m("EmmaLeonhart/x", None)]
    hf = [_m("EmmaLeonhart/x", "2026-05-30T00:00:00")]
    merged = merge_models(hf, bundled)
    assert len(merged) == 1
    assert merged[0]["last_modified"] == "2026-05-30T00:00:00"   # hf metadata wins


def test_merge_keeps_distinct_repos():
    merged = merge_models([_m("a", "2026-05-30T00:00:00")], [_m("b", None)])
    assert {m["repo_id"] for m in merged} == {"a", "b"}


def test_bundled_includes_the_published_model():
    ids = {m["repo_id"] for m in BUNDLED_MODELS}
    assert "EmmaLeonhart/reservoir-agent-gpt2-crosspass" in ids


def test_list_models_offline_falls_back_to_bundled():
    # use_hf=False must return the bundled list (sorted), never raise
    models = list_models(use_hf=False)
    assert models and all("repo_id" in m for m in models)
