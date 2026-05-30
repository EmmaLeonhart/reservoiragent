"""Tests for the KV-append injection mechanism (numpy, runs in CI).

The key property is H1 at the mechanism level: with the reservoir nodes gated off
(masked), attention over [tokens ++ reservoir] is identical to ordinary causal
attention over the tokens alone; with them active, the output changes.
"""
import numpy as np

from reservoir.kv_inject import (
    scaled_dot_product_attention,
    causal_mask,
    attention_with_reservoir,
    reservoir_nodes_from_state,
)


def _qkv(h=2, t=5, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h, t, d)) for _ in range(3))


def test_h1_gated_off_equals_base_attention():
    Q, K, V = _qkv()
    rng = np.random.default_rng(1)
    res_K = rng.standard_normal((2, 3, 4))   # 3 reservoir nodes, nonzero
    res_V = rng.standard_normal((2, 3, 4))
    base = scaled_dot_product_attention(Q, K, V, causal_mask(Q.shape[1]))
    gated_off = attention_with_reservoir(Q, K, V, res_K, res_V, active=False)
    assert np.allclose(base, gated_off, atol=1e-12)      # masked reservoir -> identity


def test_active_reservoir_changes_output():
    Q, K, V = _qkv()
    rng = np.random.default_rng(2)
    res_K = rng.standard_normal((2, 3, 4))
    res_V = rng.standard_normal((2, 3, 4))
    base = scaled_dot_product_attention(Q, K, V, causal_mask(Q.shape[1]))
    active = attention_with_reservoir(Q, K, V, res_K, res_V, active=True)
    assert not np.allclose(base, active, atol=1e-4)


def test_zero_value_reservoir_still_dilutes_unless_masked():
    # appending ZERO value vectors is NOT identity (the zero key still scores 0 and
    # takes softmax mass) — which is why H1 must be a masking property.
    Q, K, V = _qkv()
    res_K = np.zeros((2, 3, 4))
    res_V = np.zeros((2, 3, 4))
    base = scaled_dot_product_attention(Q, K, V, causal_mask(Q.shape[1]))
    active_zero = attention_with_reservoir(Q, K, V, res_K, res_V, active=True)
    assert not np.allclose(base, active_zero)            # diluted, not identical


def test_reservoir_nodes_from_state_shapes():
    r = np.random.default_rng(0).standard_normal(32)
    n_head, head_dim, n_nodes = 4, 8, 3
    W_k = np.random.default_rng(1).standard_normal((n_nodes * n_head * head_dim, 32))
    W_v = np.zeros((n_nodes * n_head * head_dim, 32))
    rk, rv = reservoir_nodes_from_state(r, W_k, W_v, n_head, head_dim)
    assert rk.shape == (n_head, n_nodes, head_dim)
    assert rv.shape == (n_head, n_nodes, head_dim)
    assert np.allclose(rv, 0.0)
