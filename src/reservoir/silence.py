"""D: a trained silence policy — making "sometimes no response" meaningful.

The harness gate currently fires off the base model's next-token entropy, which is
arbitrary. A real policy should *speak when there is something worth saying and stay
silent otherwise*. The reservoir is suited to exactly this kind of **process/temporal**
judgement (cf. the H3 delay-memory result), so we train a gate on the reservoir state.

Task ("unresolved thread"): a stream of events; a rare **trigger** event opens a thread
that should be addressed. The agent **should speak** for the next few passes after a
trigger (the thread is unresolved) and **stay silent** otherwise. The label is therefore
"was there a trigger within the last `speak_window` passes" — a function of *history*,
not the current input.

We train a linear gate (ridge classifier) on the reservoir state and compare to a
**stateless** gate (the same readout on the current input only), which cannot see the
window and so can only fire on the trigger pass itself. All numpy / CPU.
"""
from __future__ import annotations

import numpy as np

from .echo_state import EchoStateReservoir
from .tasks import fit_ridge


def make_silence_task(T: int, *, n_input: int = 4, trigger_prob: float = 0.06,
                      speak_window: int = 5, seed: int = 0):
    """Return (inputs (T, n_input), speak_labels (T,) bool, triggers (T,) bool)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((T, n_input))
    triggers = rng.random(T) < trigger_prob
    u[triggers, 0] += 3.0                         # the trigger is a spike on dim 0
    labels = np.zeros(T, dtype=bool)
    last = -10 ** 9
    for t in range(T):
        if triggers[t]:
            last = t
        labels[t] = (t - last) < speak_window     # unresolved thread open
    return u, labels, triggers


def precision_recall_f1(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, bool)
    y_pred = np.asarray(y_pred, bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def evaluate_silence_gate(K: int = 300, *, T: int = 4000, washout: int = 100,
                          rho: float = 0.9, input_scaling: float = 0.5, seed: int = 0,
                          **task_kwargs) -> dict:
    """Train a reservoir-state gate and a stateless gate; return precision/recall/F1 for
    each on a held-out half of the stream."""
    u, labels, _ = make_silence_task(T, seed=seed, **task_kwargs)
    res = EchoStateReservoir(K, u.shape[1], spectral_radius=rho,
                             input_scaling=input_scaling, seed=seed)
    R = res.run(u)

    X = R[washout:]
    Xb = u[washout:]                              # stateless: current input only
    y = labels[washout:].astype(float)[:, None]
    n = X.shape[0]
    h = n // 2
    y_test = y[h:, 0].astype(bool)

    Wr = fit_ridge(X[:h], y[:h])
    pred_r = (X[h:] @ Wr)[:, 0] > 0.5
    Wb = fit_ridge(Xb[:h], y[:h])
    pred_b = (Xb[h:] @ Wb)[:, 0] > 0.5

    base_rate = float(y_test.mean())
    return {"reservoir_gate": precision_recall_f1(y_test, pred_r),
            "stateless_gate": precision_recall_f1(y_test, pred_b),
            "speak_base_rate": base_rate, "K": K, "T": T}
