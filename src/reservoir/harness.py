"""Minimal always-alive harness pieces (the ambitious reach).

Two things the full vision needs, demonstrated at minimal scale:

1. ``two_pass_dependence`` — the proof-of-concept that the agent has a time axis:
   run the SAME prompt after different histories, with the reservoir state carried
   across the (otherwise independent) forward passes. The output depends on what came
   before — something a stateless transformer cannot do.

2. ``select_seed_by_dynamics`` — a pre-selection proxy: rank N fixed-random reservoir
   seeds by their dynamics quality on a probe stream *before* any training, the cheap
   stand-in for the full N-seed LoRA selection (which is compute-gated).

What is NOT here, and is named plainly: the full N-seed LoRA fine-tuning + benchmark
selection, and a productionized always-alive runtime (scheduler, idle timer, output
gate). Those need a training pipeline and compute beyond this feasibility session.
"""
from __future__ import annotations

import numpy as np

from .sweep import measure_point_stream


def two_pass_dependence(model_name: str, prompt_before: str, prompt: str, *,
                        seed: int = 0, readout_scale: float = 0.05,
                        n_reservoir: int = 256, device: str | None = None) -> dict:
    """Does the output on ``prompt`` depend on whether ``prompt_before`` preceded it?

    Runs ``prompt`` twice with a nonzero reservoir readout: once with the reservoir
    state carried over from a pass on ``prompt_before`` (history), once from a fresh
    (reset) reservoir. Returns the L2 distance between the two next-token logit
    vectors — nonzero means the carried state changed the behaviour (a time axis).
    """
    from .inject import ReservoirInjectedLM

    lm = ReservoirInjectedLM(model_name, n_reservoir=n_reservoir, seed=seed,
                             device=device)
    rng = np.random.default_rng(seed)
    lm.set_readout(rng.standard_normal((lm.d_model, lm.n_reservoir)) * readout_scale)

    # with history: tick the reservoir on prompt_before, then read prompt
    lm.reset_reservoir()
    lm.logits(prompt_before)
    with_history = lm.logits(prompt).detach().to("cpu").numpy()

    # without history: fresh reservoir, same prompt
    lm.reset_reservoir()
    fresh = lm.logits(prompt).detach().to("cpu").numpy()

    return {
        "logit_l2_diff": float(np.linalg.norm(with_history - fresh)),
        "argmax_changed": bool(np.argmax(with_history) != np.argmax(fresh)),
    }


def select_seed_by_dynamics(stream_a, stream_b, seeds, *, rho: float = 0.95,
                            K: int = 200, washout: int = 20,
                            input_scaling: float = 0.3) -> list[dict]:
    """Rank reservoir seeds by a pre-training dynamics-quality proxy.

    For each seed, drive the reservoir at a fixed ρ with the probe streams and score
    it by input separation while penalising saturation — a cheap proxy for "this
    random reservoir carves up the input usefully", to be used before committing to a
    LoRA run. Returns records sorted best-first.
    """
    records = []
    for s in seeds:
        m = measure_point_stream(rho, stream_a, stream_b, K, washout=washout,
                                 seed=int(s), input_scaling=input_scaling)
        # reward responsiveness + dimensionality, penalise over-saturation
        score = m["input_separation"] + m["participation_ratio_frac"] - m["saturation"]
        records.append({"seed": int(s), "score": float(score),
                        "input_separation": m["input_separation"],
                        "participation_ratio_frac": m["participation_ratio_frac"],
                        "saturation": m["saturation"]})
    return sorted(records, key=lambda r: r["score"], reverse=True)
