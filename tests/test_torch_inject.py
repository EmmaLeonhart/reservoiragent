"""Test the differentiable injection + LoRA fine-tune (torch+peft; local, skips in CI)."""
import pytest


def test_finetune_pipeline_reduces_loss():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from reservoir.torch_inject import train_finetune

    # tiny model, CPU — just prove the pipeline trains (loss decreases). Enough steps
    # that the decrease is reliable rather than borderline on a tiny random model.
    recs = train_finetune("sshleifer/tiny-gpt2", seeds=(0,), steps=30, lr=2e-3,
                          device="cpu")
    assert len(recs) == 1
    r = recs[0]
    assert r["n_trainable_params"] > 0
    assert r["loss_end"] < r["loss_start"]      # the fine-tune actually learns
