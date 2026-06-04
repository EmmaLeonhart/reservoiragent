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


import reservoir.installer.console as console
from reservoir.installer.menu import main

DEFAULT_REPO = "EmmaLeonhart/reservoir-agent-gpt2-crosspass"


def _spy(monkeypatch):
    calls = {}
    def fake_run(repo_id, **kw):
        calls["repo_id"] = repo_id
        calls.update(kw)
    monkeypatch.setattr(console, "run", fake_run)
    return calls


def test_default_flow_auto_picks_and_runs_demo_and_repl(monkeypatch):
    calls = _spy(monkeypatch)
    assert main(["--no-hf"]) == 0
    assert calls["repo_id"] == DEFAULT_REPO
    assert calls["demo"] is True and calls["repl"] is True


def test_demo_only_disables_repl(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--demo-only"])
    assert calls["demo"] is True and calls["repl"] is False


def test_no_demo_disables_demo(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--no-demo"])
    assert calls["demo"] is False and calls["repl"] is True


def test_menu_flag_uses_chooser(monkeypatch):
    calls = _spy(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")   # empty -> default
    main(["--no-hf", "--menu"])
    assert calls["repo_id"] == DEFAULT_REPO


def test_repo_id_skips_selection(monkeypatch):
    calls = _spy(monkeypatch)
    main(["--no-hf", "--repo-id", "Foo/bar"])
    assert calls["repo_id"] == "Foo/bar"


def test_list_prints_and_does_not_run(monkeypatch):
    calls = _spy(monkeypatch)
    assert main(["--no-hf", "--list"]) == 0
    assert "repo_id" not in calls
