"""Tests for the ambitious-reach pieces.

The seed-selection proxy is pure numpy (runs in CI). The two-pass time-axis demo
needs torch + transformers and runs locally (skips in the light CI job).
"""
import numpy as np
import pytest

from reservoir.harness import select_seed_by_dynamics


def test_seed_selection_ranks_and_is_deterministic():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 16))
    b = rng.standard_normal((300, 16))
    ranked = select_seed_by_dynamics(a, b, seeds=[1, 2, 3, 4, 5], K=80, washout=60)
    assert len(ranked) == 5
    # sorted best-first by score
    assert all(ranked[i]["score"] >= ranked[i + 1]["score"] for i in range(len(ranked) - 1))
    # different seeds give genuinely different dynamics (so selection is meaningful)
    assert len({round(r["score"], 6) for r in ranked}) > 1
    # deterministic
    again = select_seed_by_dynamics(a, b, seeds=[1, 2, 3, 4, 5], K=80, washout=60)
    assert [r["seed"] for r in ranked] == [r["seed"] for r in again]


def test_two_pass_dependence_shows_time_axis():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from reservoir.harness import two_pass_dependence

    out = two_pass_dependence("sshleifer/tiny-gpt2", "A long story about the sea.",
                              "The capital of France is", seed=0, readout_scale=0.1,
                              n_reservoir=64, device="cpu")
    # the same prompt yields a different next-token distribution depending on history
    assert out["logit_l2_diff"] > 0.0
