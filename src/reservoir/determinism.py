"""Make a training run reproducible — the other half of closing the selection-noise confound.

Seeding the trainable init (``kv_live`` ``train_seed``) controls *where* a run starts; this
controls *how* it evolves, by pinning the RNGs and asking the backends for deterministic
kernels. Together they make two runs of the same fixed reservoir, with the same ``train_seed``,
land at the same recall — so any *remaining* spread across reservoir seeds is reservoir quality,
not run-to-run noise (the question Phase I tests).

CUDA determinism in particular needs ``CUBLAS_WORKSPACE_CONFIG`` set *before* the first CUDA
matmul, so call this early. ``warn_only=True`` keeps an op that lacks a deterministic kernel
from hard-failing the run (it warns instead) — we prefer a completed, mostly-deterministic run
to a crash.
"""
from __future__ import annotations

import os
import random


def set_deterministic(seed: int) -> None:
    """Seed Python/NumPy/torch RNGs and request deterministic backend kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        # older torch without warn_only; fall back to the strict form, still non-fatal
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
