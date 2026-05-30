"""KV-append injection — reservoir nodes as extra keys/values in attention.

The richer injection from the architecture: instead of adding the reservoir state to
the residual stream, the reservoir nodes join the injection layer's **key/value
sequence**, so every token's attention can read from (and, via the readout, write to)
the reservoir. This module implements and tests the *mechanism* in isolation — a
minimal multi-head attention over ``[tokens ++ reservoir nodes]`` — with the clean H1
property: when the reservoir nodes are gated off (masked out of the softmax) the output
is **identical** to ordinary causal attention; when active, the upper layers attend to
them.

H1 here is a masking property, not a zero-weights one: appending *zero* key/value
vectors would still dilute the softmax (a zero key scores 0, contributing exp(0) to the
normaliser), so "off" must mean *masked* (−∞ score), which is exactly identity to base.

Implemented in numpy so the mechanism is unit-tested in CI without torch. Wiring it
into a live HF GPT-2 is left as a documented blocker (see ``GPT2_INTEGRATION_BLOCKER``)
rather than a fragile patch of transformers' internals.
"""
from __future__ import annotations

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None) -> np.ndarray:
    """Multi-head SDPA. Q (h, tq, d), K/V (h, tk, d); mask (tq, tk) additive (−∞ to
    forbid). Returns (h, tq, d)."""
    Q, K, V = np.asarray(Q, float), np.asarray(K, float), np.asarray(V, float)
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)      # (h, tq, tk)
    if mask is not None:
        scores = scores + mask
    return _softmax(scores, axis=-1) @ V                  # (h, tq, d)


def causal_mask(t: int) -> np.ndarray:
    """(t, t) additive causal mask: 0 on/below the diagonal, −∞ above."""
    m = np.zeros((t, t))
    m[np.triu_indices(t, k=1)] = -np.inf
    return m


def attention_with_reservoir(Q, K, V, res_K, res_V, *, active: bool,
                             causal: bool = True) -> np.ndarray:
    """Attention over ``[tokens ++ reservoir nodes]``.

    Q, K, V: (h, t, d) token query/key/value. res_K, res_V: (h, n, d) reservoir nodes
    (attended by every query — they are not causal). ``active`` gates the reservoir:
    when False the reservoir keys are masked (−∞), so the result equals ordinary
    (causal) attention over the tokens alone — the H1 non-destruction property.
    """
    Q = np.asarray(Q, float)
    h, t, d = Q.shape
    n = np.asarray(res_K, float).shape[1]
    Kx = np.concatenate([np.asarray(K, float), np.asarray(res_K, float)], axis=1)
    Vx = np.concatenate([np.asarray(V, float), np.asarray(res_V, float)], axis=1)
    tok_mask = causal_mask(t) if causal else np.zeros((t, t))
    res_col = np.zeros((t, n)) if active else np.full((t, n), -np.inf)
    mask = np.concatenate([tok_mask, res_col], axis=1)    # (t, t+n)
    return scaled_dot_product_attention(Q, Kx, Vx, mask)


def reservoir_nodes_from_state(r, W_k, W_v, n_head: int, head_dim: int):
    """Project a reservoir state r (K,) into n_nodes key/value vectors shaped for
    multi-head attention: returns (res_K, res_V) each (n_head, n_nodes, head_dim).

    W_k, W_v: (n_nodes * n_head * head_dim, K) readout matrices. With W_v = 0 the
    reservoir values are zero — but note H1 still requires *masking* (see module
    docstring), not zero values."""
    r = np.asarray(r, float)
    n_total = W_k.shape[0]
    n_nodes = n_total // (n_head * head_dim)
    rk = (np.asarray(W_k, float) @ r).reshape(n_nodes, n_head, head_dim).transpose(1, 0, 2)
    rv = (np.asarray(W_v, float) @ r).reshape(n_nodes, n_head, head_dim).transpose(1, 0, 2)
    return rk, rv


GPT2_INTEGRATION_BLOCKER = """\
Wiring KV-append into a live HF GPT-2 (transformers 5.4) is deliberately NOT done this
session, to avoid a fragile patch. The mechanism above is implemented and tested in
isolation; the remaining integration step is precisely scoped:

- GPT2Attention.forward (transformers 5.4) computes query/key/value_states and then
  dispatches to `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)`,
  with a `Cache` object handling past_key_values. There is no hook that exposes the
  internal key/value tensors, so injection requires an eager-path `forward` override
  on the target block that (a) forces `config._attn_implementation = "eager"`,
  (b) appends the reservoir-derived key/value rows to key_states/value_states along the
  sequence dim, and (c) extends the attention mask with a reservoir column (0 when
  active, -inf when gated) — i.e. exactly `attention_with_reservoir` above, but against
  GPT-2's Cache/dispatch machinery.
- This is bounded but invasive and version-sensitive; it is queued as its own item
  rather than hacked in here. The residual-stream injection (src/reservoir/inject.py)
  remains the working, H1-verified injection used by the rest of the study.
"""
