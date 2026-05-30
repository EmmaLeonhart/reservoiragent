"""Tests for the spectral-radius sweep.

Kept small/fast, but they assert the *qualitative physics* the study relies on:
below ρ≈1 the reservoir forgets its initial condition (echo state property), and it
saturates more as ρ grows. If these break, the sweep's conclusions are meaningless.
"""
from reservoir.sweep import run_sweep, measure_point, healthy_regime


def test_record_shape_and_keys():
    recs = run_sweep([0.5, 1.0], K=40, T=200, washout=40)
    assert len(recs) == 2
    needed = {"rho", "K", "variance", "saturation", "participation_ratio",
              "init_forgetting", "input_separation"}
    assert needed <= set(recs[0])


def test_echo_state_property_holds_below_one_and_breaks_above():
    low = measure_point(0.4, K=120, T=400, washout=100)
    high = measure_point(1.5, K=120, T=400, washout=100)
    # autonomous probe: a sub-unit reservoir forgets its initial state entirely
    # (decays to the null state) ...
    assert low["init_forgetting"] < 0.01
    # ... while a super-unit reservoir retains a clear dependence on where it started
    # (the echo state property is broken above ρ ≈ 1).
    assert high["init_forgetting"] > 0.1


def test_saturation_increases_with_spectral_radius():
    recs = run_sweep([0.3, 2.5], K=80, T=400, washout=80)
    assert recs[1]["saturation"] > recs[0]["saturation"]


def test_reservoir_responds_to_input():
    # a healthy mid-range reservoir separates different input histories
    r = measure_point(0.9, K=80, T=400, washout=80)
    assert r["input_separation"] > 0.05


def test_healthy_regime_filters():
    recs = run_sweep([0.3, 0.9, 1.8], K=60, T=300, washout=60)
    healthy = healthy_regime(recs)
    assert all(h["init_forgetting"] <= 0.2 for h in healthy)
    assert any(r["rho"] < 1.0 for r in healthy)  # the sub-unit ones qualify


def test_stream_sweep_runs_and_preserves_esp_boundary():
    # exercise the real-activation code path with a synthetic stand-in stream (no
    # torch needed): the autonomous ESP boundary at rho~1 must still hold.
    import numpy as np
    from reservoir.sweep import run_sweep_stream

    rng = np.random.default_rng(0)
    stream_a = rng.standard_normal((300, 16))
    stream_b = rng.standard_normal((300, 16))
    # ESP holds well below 1 and is clearly broken well above it. (The exact
    # sharpness near rho=1 is K-dependent and noisy for small K, so the test pins
    # the two ends, not the transition width.)
    recs = run_sweep_stream([0.4, 2.0], stream_a, stream_b, K=80, washout=80)
    assert len(recs) == 2
    assert recs[0]["init_forgetting"] < 0.01      # rho=0.4 forgets its init
    assert recs[1]["init_forgetting"] > 0.1       # rho=2.0 retains it


def test_input_scaling_controls_saturation():
    import numpy as np
    from reservoir.sweep import run_scaling_sweep

    rng = np.random.default_rng(0)
    a = rng.standard_normal((300, 16)) * 8.0   # large-magnitude (real-activation-like)
    b = rng.standard_normal((300, 16)) * 8.0
    recs = run_scaling_sweep([0.02, 1.0], a, b, K=80, rho=0.9, washout=60)
    # smaller input scaling -> less drive -> less saturation
    assert recs[0]["saturation"] < recs[1]["saturation"]
    assert recs[0]["input_scaling"] == 0.02
