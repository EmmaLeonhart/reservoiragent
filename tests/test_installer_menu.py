"""Tests for the installer menu's pure choice logic."""
import pytest

from reservoir.installer.menu import choose_repo


MODELS = [
    {"repo_id": "EmmaLeonhart/reservoir-agent-gpt2-crosspass", "recommended": True,
     "last_modified": "2026-05-30T00:00:00"},
    {"repo_id": "EmmaLeonhart/reservoir-agent-gpt2-batch", "last_modified": "2026-05-29T00:00:00"},
]


def test_empty_input_picks_default_recommended():
    assert choose_repo(MODELS, "") == "EmmaLeonhart/reservoir-agent-gpt2-crosspass"


def test_whitespace_input_picks_default():
    assert choose_repo(MODELS, "   ") == "EmmaLeonhart/reservoir-agent-gpt2-crosspass"


def test_numeric_choice_selects_that_row():
    assert choose_repo(MODELS, "2") == "EmmaLeonhart/reservoir-agent-gpt2-batch"


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        choose_repo(MODELS, "9")


def test_non_numeric_raises():
    with pytest.raises(ValueError):
        choose_repo(MODELS, "abc")


def test_empty_registry_raises():
    with pytest.raises(ValueError):
        choose_repo([], "")
