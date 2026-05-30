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
