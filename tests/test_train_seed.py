"""train_seed must make the trainable init deterministic (torch+peft; local, skips in CI).

The N-seed controlled experiment needs the trainable W_res + LoRA init to be a *controlled*
variable, not uncontrolled noise. kv_live accepted a `train_seed` parameter but never used it,
so every run of the same reservoir started from a different init — a confound that dominated
the per-seed recall spread (devlog 2026-05-31). These tests pin the fix.
"""
import pytest


def _build(train_seed):
    from reservoir.kv_live import TorchReservoirPrefixInjectedLM
    return TorchReservoirPrefixInjectedLM(
        "sshleifer/tiny-gpt2", seed=0, train_seed=train_seed,
        device="cpu", n_reservoir=32, n_prefix=4)


def test_same_train_seed_gives_identical_w_res_init():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch
    a, b = _build(123), _build(123)
    assert torch.equal(a.W_res.weight, b.W_res.weight)
    assert torch.equal(a.W_res.bias, b.W_res.bias)


def test_different_train_seed_changes_w_res_init():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch
    a, c = _build(123), _build(999)
    assert not torch.equal(a.W_res.weight, c.W_res.weight)


def test_same_train_seed_gives_identical_lora_init():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import torch
    a, b = _build(5), _build(5)

    def lora_weights(lm):
        return [p.detach().clone() for n, p in lm.model.named_parameters()
                if "lora" in n.lower()]

    wa, wb = lora_weights(a), lora_weights(b)
    assert len(wa) > 0                                   # there are LoRA params to check
    assert all(torch.equal(x, y) for x, y in zip(wa, wb))


def test_train_seed_is_recorded():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    lm = _build(42)
    assert lm.train_seed == 42
