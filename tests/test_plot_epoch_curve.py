"""Unit tests for the per-epoch training-curve plotter (scripts/plot_epoch_curve.py).

Covers the three index.json shapes the plotter must survive: new-format records (with the
inline stateless control), old-format records (no ``stateless_mean`` — the pre-control runs),
and the empty/missing cases the ``main()`` guard must reject without throwing. matplotlib is
import-skipped so the suite still runs where it is absent.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_epoch_curve import main, plot_epoch_curve  # noqa: E402

NEW = [
    {"epoch": 0, "mean": 0.10, "stateless_mean": 0.09, "lift": 0.01, "metrics": {}},
    {"epoch": 1, "mean": 0.22, "stateless_mean": 0.12, "lift": 0.10, "metrics": {}},
    {"epoch": 2, "mean": 0.31, "stateless_mean": 0.13, "lift": 0.18, "metrics": {}},
]
OLD = [  # pre-control format: no stateless_mean
    {"epoch": 0, "mean": 0.044, "metrics": {}},
    {"epoch": 1, "mean": 0.031, "metrics": {}},
]


def test_plots_new_format_with_control(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "curve.png"
    plot_epoch_curve(NEW, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plots_old_format_without_control(tmp_path):
    # have_ctrl must be False here and the plot must still render (no KeyError on stateless_mean).
    pytest.importorskip("matplotlib")
    out = tmp_path / "curve_old.png"
    plot_epoch_curve(OLD, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_records_are_sorted_by_epoch(tmp_path):
    # Out-of-order input must not crash and must still produce a figure.
    pytest.importorskip("matplotlib")
    shuffled = [NEW[2], NEW[0], NEW[1]]
    out = tmp_path / "curve_shuf.png"
    plot_epoch_curve(shuffled, str(out))
    assert out.exists()


def test_main_returns_1_on_missing_index(tmp_path):
    # The guard must reject a missing index without throwing.
    missing = tmp_path / "nope.json"
    assert main_argv(str(missing), str(tmp_path / "o.png")) == 1


def test_main_returns_1_on_empty_index(tmp_path):
    empty = tmp_path / "index.json"
    empty.write_text("[]", encoding="utf-8")
    assert main_argv(str(empty), str(tmp_path / "o.png")) == 1


def test_main_succeeds_on_real_index(tmp_path):
    pytest.importorskip("matplotlib")
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps(NEW), encoding="utf-8")
    out = tmp_path / "o.png"
    assert main_argv(str(idx), str(out)) == 0
    assert out.exists()


def main_argv(src, out):
    """Invoke main() with argv set, restoring it after (main reads sys.argv)."""
    saved = sys.argv
    try:
        sys.argv = ["plot_epoch_curve.py", src, out]
        return main()
    finally:
        sys.argv = saved
