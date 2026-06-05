"""Tests for the stateful-task battery generators (pure; deterministic given a seed).

The episode *execution* (episode_loss / episode_eval) is torch-gated and exercised by the
training run, not here.
"""
import numpy as np

from reservoir.battery import (GENERATORS, DEFAULT_WEIGHTS, sample_episode, make_eval_set,
                               gen_recall, gen_timed, gen_silence)
from reservoir.episode import Episode, SILENCE


def test_all_generators_produce_labeled_episodes_with_supervision():
    rng = np.random.default_rng(0)
    for name, fn in GENERATORS.items():
        ep = fn(rng)
        assert isinstance(ep, Episode)
        assert ep.task == name
        assert any(s.target is not None for s in ep.steps), f"{name} has no supervised step"


def test_recall_target_matches_the_injected_word():
    ep = gen_recall(np.random.default_rng(3))
    word = ep.steps[0].inject.split("is ")[1].rstrip(".")
    assert ep.steps[1].wipe is True
    assert ep.steps[1].target == f" {word}"


def test_timed_is_silent_then_emits():
    ep = gen_timed(np.random.default_rng(1))
    silents = [s for s in ep.steps if s.target is SILENCE]
    emits = [s for s in ep.steps if isinstance(s.target, str)]
    assert len(silents) >= 1
    assert len(emits) == 1            # exactly one emission, at the end
    assert ep.steps[-1].target is emits[0].target


def test_silence_task_is_all_silence_targets():
    ep = gen_silence(np.random.default_rng(2))
    assert all(s.target in (None, SILENCE) for s in ep.steps)
    assert any(s.target is SILENCE for s in ep.steps)


def test_sample_episode_is_deterministic_for_a_seed():
    a = sample_episode(np.random.default_rng(7))
    b = sample_episode(np.random.default_rng(7))
    assert a.task == b.task
    assert [s.target for s in a.steps] == [s.target for s in b.steps]


def test_make_eval_set_has_n_per_task_for_each_task():
    eps = make_eval_set(np.random.default_rng(0), n_per_task=3)
    counts = {}
    for ep in eps:
        counts[ep.task] = counts.get(ep.task, 0) + 1
    assert counts == {k: 3 for k in DEFAULT_WEIGHTS}
