"""Tests for the controlled-selection analysis (pure, CI) + a torch-gated runner smoke.

The analysis decides whether reservoir quality is a real signal (between-seed variation) above
run-to-run noise (within-seed variation) via a one-way ANOVA. Thresholds/values are hand-checked.
"""
import pytest

from reservoir.controlled import selection_signal


def test_f_ratio_matches_hand_computation():
    # Group 2,4,6 (mean 4) vs 10,12,14 (mean 12); grand 8.
    # SS_between = 3*16 + 3*16 = 96 (df 1); SS_within = 8 + 8 = 16 (df 4); F = 96/4 = 24.
    records = ([{"reservoir_seed": 0, "recall_accuracy": v} for v in (2, 4, 6)] +
               [{"reservoir_seed": 1, "recall_accuracy": v} for v in (10, 12, 14)])
    s = selection_signal(records)
    assert s["df_between"] == 1
    assert s["df_within"] == 4
    assert s["f_ratio"] == pytest.approx(24.0)
    assert s["between_seed_ms"] == pytest.approx(96.0)
    assert s["within_seed_ms"] == pytest.approx(4.0)


def test_strong_between_seed_signal_is_real():
    records = ([{"reservoir_seed": 0, "recall_accuracy": v} for v in (0.9, 1.0)] +
               [{"reservoir_seed": 1, "recall_accuracy": v} for v in (0.1, 0.0)] +
               [{"reservoir_seed": 2, "recall_accuracy": v} for v in (0.5, 0.6)])
    s = selection_signal(records)
    assert s["selection_is_real"] is True
    if s["p_value"] is not None:
        assert s["p_value"] < 0.05
    assert s["best_seed"] == 0
    assert s["worst_seed"] == 1


def test_pure_within_noise_is_not_real():
    # every seed has the same mean (0.5); all variation is within-seed -> F = 0, not significant.
    records = ([{"reservoir_seed": 0, "recall_accuracy": v} for v in (0.2, 0.8)] +
               [{"reservoir_seed": 1, "recall_accuracy": v} for v in (0.8, 0.2)] +
               [{"reservoir_seed": 2, "recall_accuracy": v} for v in (0.4, 0.6)])
    s = selection_signal(records)
    assert s["f_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert s["selection_is_real"] is False


def test_zero_within_variance_with_between_difference_is_real():
    # each seed perfectly reproducible but seeds differ -> within MS 0, F infinite -> real.
    records = ([{"reservoir_seed": 0, "recall_accuracy": 1.0} for _ in range(3)] +
               [{"reservoir_seed": 1, "recall_accuracy": 0.0} for _ in range(3)])
    s = selection_signal(records)
    assert s["within_seed_ms"] == pytest.approx(0.0)
    assert s["f_ratio"] == float("inf")
    assert s["selection_is_real"] is True


def test_per_seed_means_reported():
    records = ([{"reservoir_seed": 7, "recall_accuracy": v} for v in (0.0, 1.0)] +
               [{"reservoir_seed": 9, "recall_accuracy": v} for v in (0.5, 0.5)])
    s = selection_signal(records)
    assert s["per_seed_mean"]["7"] == pytest.approx(0.5)
    assert s["per_seed_mean"]["9"] == pytest.approx(0.5)
    assert s["per_seed_std"]["9"] == pytest.approx(0.0)


def test_needs_two_seeds():
    with pytest.raises(ValueError):
        selection_signal([{"reservoir_seed": 0, "recall_accuracy": 1.0}])


def test_controlled_selection_runner_smoke():
    # torch-gated: verify the runner trains and returns the right record shape (not the science).
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.controlled import controlled_selection

    recs = controlled_selection([0, 1], runs_per_seed=2, model_name="sshleifer/tiny-gpt2",
                                steps=3, n_keys=4, device="cpu", deterministic=True)
    assert len(recs) == 4                       # 2 seeds x 2 runs
    assert {r["reservoir_seed"] for r in recs} == {0, 1}
    assert all(0.0 <= r["recall_accuracy"] <= 1.0 for r in recs)
    assert all({"reservoir_seed", "run", "train_seed", "recall_accuracy"} <= r.keys()
               for r in recs)
