"""A trained recurrent (GRU) baseline on the cross-pass recall task.

Requested by a clawRxiv reviewer (post 2692, con #2): the reset-reservoir baseline in the
cross-pass experiment is an *ablation* (it isolates carried state), not a competitive memory
model. A trained recurrent net is the stronger baseline that situates how hard the task itself
is. This module trains a small GRU on the *identical* task structure:

- **Pass 1** encodes a secret: tokens ``the secret word is <KEY>`` — the GRU reads them and
  carries a hidden state.
- The context is then **wiped** — pass 2 does not contain the key.
- **Pass 2** queries: ``the secret word was`` → the readout must produce ``<KEY>`` from the
  carried hidden state alone.

The ``stateful`` model carries the pass-1 hidden state into pass 2; the ``stateless`` control
resets it (the same ablation as the reservoir experiment). This is intentionally a tiny,
self-contained task (no pretrained LM, a small structured vocab) so the *baseline's* difficulty
is what is measured, not a language model's.

The expected outcome — a trained GRU solves it easily — is the point: it shows the cross-pass
recall task is trivial for *trained* recurrence, which is exactly why the contribution of the
reservoir work is doing it with a **fixed, random** reservoir inside a **frozen** pretrained
transformer (and why the open problem is scaling that, not the task itself).
"""
from __future__ import annotations


# vocab: 0=PAD, structural tokens, then n_keys secret-word ids starting at KEY0
PAD = 0
THE, SECRET, WORD, IS, WAS = 1, 2, 3, 4, 5
KEY0 = 6


def _sequences(key_id):
    """(pass-1 tokens with the key, pass-2 query tokens) for a given key id."""
    p1 = [THE, SECRET, WORD, IS, key_id]
    p2 = [THE, SECRET, WORD, WAS]
    return p1, p2


def run_rnn_baseline(*, n_keys: int = 6, steps: int = 400, lr: float = 1e-2, seed: int = 0,
                     hidden_size: int = 64, stateful: bool = True,
                     device: str | None = None) -> dict:
    """Train a GRU on cross-pass recall; return recall accuracy + losses (mirrors crosspass)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab = KEY0 + n_keys
    key_ids = [KEY0 + i for i in range(n_keys)]

    emb = nn.Embedding(vocab, 32, padding_idx=PAD).to(dev)
    gru = nn.GRU(32, hidden_size, batch_first=True).to(dev)
    readout = nn.Linear(hidden_size, vocab).to(dev)
    params = list(emb.parameters()) + list(gru.parameters()) + list(readout.parameters())
    opt = torch.optim.AdamW(params, lr=lr)

    def forward(key_id):
        p1, p2 = _sequences(key_id)
        t1 = torch.tensor([p1], device=dev)
        t2 = torch.tensor([p2], device=dev)
        _, h = gru(emb(t1))                       # pass 1: read the key, carry hidden state h
        h2_init = h if stateful else torch.zeros_like(h)   # stateless control wipes it
        out, _ = gru(emb(t2), h2_init)            # pass 2: query (no key in the tokens)
        return readout(out[:, -1])                # logits at the recall point

    rng = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        kid = key_ids[int(torch.randint(n_keys, (1,), generator=rng))]
        logits = forward(kid)
        loss = nn.functional.cross_entropy(logits, torch.tensor([kid], device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        correct = sum(int(forward(kid).argmax().item()) == kid for kid in key_ids)

    return {"model": f"gru-{hidden_size}", "stateful": stateful, "n_keys": n_keys,
            "steps": steps, "loss_start": losses[0], "loss_end": losses[-1],
            "recall_accuracy": correct / n_keys, "device": dev, "mode": "rnn-baseline"}
