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
                      speak_window: int = 5, trigger_mag: float = 6.0, seed: int = 0):
    """Return (inputs (T, n_input), speak_labels (T,) bool, triggers (T,) bool).

    Dim 0 is a dedicated trigger channel (0 except a spike on trigger passes); the other
    dims are background noise. So the trigger is a clean event the reservoir can encode.
    """
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((T, n_input)) * 0.5
    u[:, 0] = 0.0                                  # dedicated trigger channel
    triggers = rng.random(T) < trigger_prob
    u[triggers, 0] = trigger_mag                   # the trigger event (a clean spike)
    labels = np.zeros(T, dtype=bool)
    last = -10 ** 9
    for t in range(T):
        # speak in the passes STRICTLY AFTER a trigger (an unresolved thread the agent
        # should now address). The trigger step itself is not "speak", so at every speak
        # step the trigger is in the *past* — invisible to the current input, and only
        # recoverable from the carried reservoir state.
        labels[t] = 0 < (t - last) <= speak_window
        if triggers[t]:
            last = t
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
    y_train = y[:h, 0].astype(bool)
    y_test = y[h:, 0].astype(bool)

    def gate(features):
        W = fit_ridge(features[:h], y[:h])
        s_train = (features[:h] @ W)[:, 0]
        s_test = (features[h:] @ W)[:, 0]
        # pick the decision threshold on TRAIN to maximize F1 (part of training the gate)
        thr, best = 0.5, -1.0
        for t in np.quantile(s_train, np.linspace(0.02, 0.98, 49)):
            f1 = precision_recall_f1(y_train, s_train > t)["f1"]
            if f1 > best:
                best, thr = f1, t
        return precision_recall_f1(y_test, s_test > thr)

    return {"reservoir_gate": gate(X), "stateless_gate": gate(Xb),
            "speak_base_rate": float(y_test.mean()), "K": K, "T": T}
