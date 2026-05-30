"""Smoke test for the cross-pass recall pipeline (torch+peft; local, skips in CI).

Verifies the multi-pass differentiable loop runs end-to-end and returns a valid result.
The scientific claim (stateful recall beats the stateless baseline) is measured by the
real GPU run in `scripts/run.py crosspass`, not this tiny smoke test.
"""
import pytest


def test_cross_pass_pipeline_runs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass

    r = run_cross_pass("sshleifer/tiny-gpt2", n_keys=4, steps=6, lr=1e-2,
                       device="cpu", stateful=True)
    assert 0.0 <= r["recall_accuracy"] <= 1.0
    assert r["n_keys"] == 4
    assert "loss_end" in r


def test_cross_pass_kv_pipeline_runs():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass_kv

    # the content-addressable (KV-prefix) path must run end-to-end too
    r = run_cross_pass_kv("sshleifer/tiny-gpt2", n_keys=4, steps=4, lr=1e-2,
                          device="cpu", stateful=True, n_prefix=4)
    assert r["mode"] == "kv-prefix"
    assert 0.0 <= r["recall_accuracy"] <= 1.0
