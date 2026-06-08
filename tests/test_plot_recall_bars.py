"""Unit tests for the recall-bars plotter (scripts/plot_recall_bars.py).

Covers load() parsing of the crosspass JSON shape, the label-key variants, the main() guards,
and that a figure renders. matplotlib is import-skipped so the suite runs where it is absent.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_recall_bars import _label, load, main, plot_recall_bars  # noqa: E402

PARAMS = {"model": "Qwen/Qwen2.5-1.5B-Instruct", "n_keys": 6, "input_scaling": 0.1}


def _write(tmp_path, name, stateful, control, params=PARAMS):
    d = {"params": params, "results": {"stateful": {"recall_accuracy": stateful},
                                       "baseline": {"recall_accuracy": control}}}
    p = tmp_path / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


def test_load_parses_recall(tmp_path):
    f = _write(tmp_path, "a.json", 0.83, 0.17)
    recs = load([f])
    assert len(recs) == 1
    params, s, c = recs[0]
    assert s == pytest.approx(0.83) and c == pytest.approx(0.17)
    assert params["n_keys"] == 6


def test_label_variants():
    assert "Qwen-1.5B" in _label(PARAMS, "model")
    assert _label(PARAMS, "n_keys") == "6 keys"
    assert _label(PARAMS, "input_scaling") == "scale 0.1"
    assert "k=6" in _label(PARAMS, "auto")  # auto/unknown falls through to the composite label


def test_plot_renders(tmp_path):
    pytest.importorskip("matplotlib")
    recs = [(PARAMS, 0.83, 0.17), ({**PARAMS, "n_keys": 12}, 0.25, 0.08)]
    out = tmp_path / "bars.png"
    plot_recall_bars(recs, str(out), "test", "n_keys")
    assert out.exists() and out.stat().st_size > 0


def test_main_guard_on_missing_inputs(tmp_path):
    saved = sys.argv
    try:
        sys.argv = ["plot_recall_bars.py", str(tmp_path / "o.png"), "T", "model", str(tmp_path / "nope.json")]
        assert main() == 1  # all inputs missing -> guard returns 1
    finally:
        sys.argv = saved


def test_main_succeeds(tmp_path):
    pytest.importorskip("matplotlib")
    f = _write(tmp_path, "a.json", 1.0, 0.17)
    out = tmp_path / "o.png"
    saved = sys.argv
    try:
        sys.argv = ["plot_recall_bars.py", str(out), "T", "model", f]
        assert main() == 0
        assert out.exists()
    finally:
        sys.argv = saved
