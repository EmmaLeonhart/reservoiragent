"""Reproducibility: same (reservoir seed, train_seed) -> same run (torch+peft; local, CI-skip).

This is the end-to-end proof that the Phase-I confound is closed: once the trainable init and
data order are seeded by train_seed (and the reservoir by seed), two runs of the same fixed
reservoir are identical, so any spread that survives across *different* reservoir seeds is
reservoir quality, not run-to-run noise.
"""
import pytest


def test_set_deterministic_sets_backend_flags():
    pytest.importorskip("torch")
    import torch
    from reservoir.determinism import set_deterministic

    set_deterministic(7)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_run_cross_pass_kv_is_bit_reproducible_with_train_seed():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass_kv

    kw = dict(n_keys=4, steps=6, lr=1e-2, seed=0, device="cpu", deterministic=True)
    a = run_cross_pass_kv("sshleifer/tiny-gpt2", train_seed=1, **kw)
    b = run_cross_pass_kv("sshleifer/tiny-gpt2", train_seed=1, **kw)
    # identical reservoir AND identical train_seed -> the run is reproducible to the bit.
    assert a["loss_end"] == b["loss_end"]
    assert a["recall_accuracy"] == b["recall_accuracy"]
    assert a["loss_start"] == b["loss_start"]


def test_different_train_seed_changes_the_run():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.crosspass import run_cross_pass_kv

    kw = dict(n_keys=4, steps=6, lr=1e-2, seed=0, device="cpu")
    a = run_cross_pass_kv("sshleifer/tiny-gpt2", train_seed=1, **kw)
    c = run_cross_pass_kv("sshleifer/tiny-gpt2", train_seed=2, **kw)
    # same reservoir, different trainable init + data order -> a different trajectory
    assert a["loss_end"] != c["loss_end"] or a["recall_accuracy"] != c["recall_accuracy"]
