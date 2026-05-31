"""Tests for the N-seed batch logic (pure ranking + manifest).

The training itself is torch-gated; these cover the deterministic selection + preservation
logic: rank the population, privilege the best, and KEEP every model (bad seeds are signal).
"""
import pytest

from reservoir.batch import rank_population, select_best, build_batch_manifest


def _rec(seed, recall, loss_end):
    return {"seed": seed, "recall_accuracy": recall, "loss_end": loss_end}


def test_rank_by_recall_desc():
    recs = [_rec(0, 0.5, 0.1), _rec(1, 1.0, 0.2), _rec(2, 0.17, 0.3)]
    assert [r["seed"] for r in rank_population(recs)] == [1, 0, 2]


def test_rank_tiebreaks_on_lower_loss():
    recs = [_rec(0, 1.0, 0.30), _rec(1, 1.0, 0.05)]
    assert [r["seed"] for r in rank_population(recs)] == [1, 0]


def test_select_best_is_top_ranked():
    recs = [_rec(0, 0.5, 0.1), _rec(1, 1.0, 0.05)]
    assert select_best(recs)["seed"] == 1


def test_manifest_preserves_all_models():
    recs = [_rec(0, 0.17, 0.4), _rec(1, 1.0, 0.05), _rec(2, 0.33, 0.2)]
    man = build_batch_manifest(recs, model_name="gpt2")
    # every seed retained — the bad ones are signal, never dropped
    assert {m["seed"] for m in man["population"]} == {0, 1, 2}
    assert man["n"] == 3


def test_manifest_marks_exactly_one_recommended_best():
    recs = [_rec(0, 0.17, 0.4), _rec(1, 1.0, 0.05), _rec(2, 0.33, 0.2)]
    man = build_batch_manifest(recs, model_name="gpt2")
    recommended = [m for m in man["population"] if m.get("recommended")]
    assert len(recommended) == 1
    assert recommended[0]["seed"] == 1
    assert man["best"]["seed"] == 1


def test_manifest_population_is_ranked_with_ranks_assigned():
    recs = [_rec(0, 0.5, 0.1), _rec(1, 1.0, 0.05), _rec(2, 0.17, 0.3)]
    man = build_batch_manifest(recs, model_name="gpt2")
    assert [m["seed"] for m in man["population"]] == [1, 0, 2]
    assert [m["rank"] for m in man["population"]] == [0, 1, 2]


def test_manifest_empty_raises():
    with pytest.raises(ValueError):
        build_batch_manifest([], model_name="gpt2")
