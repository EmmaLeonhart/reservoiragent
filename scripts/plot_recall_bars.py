"""Grouped stateful-vs-control recall bar chart from a set of crosspass result JSONs.

Each input is a ``results/crosspass_*.json`` (written by ``run.py crosspass``): it has
``params`` (model, n_keys, input_scaling, and — for newer runs — n_reservoir/n_prefix) and
``results.{stateful,baseline}.recall_accuracy``. This draws one labelled group per JSON with the
stateful (carried-state) bar beside the wiped-reservoir control bar, and a dashed 1/n_keys chance
line, so a cross-model panel or a capacity sweep reads at a glance: where the carried state lifts
recall above its control, and where it sits at chance.

Run:  python scripts/plot_recall_bars.py OUT.png "Title" LABELKEY a.json b.json ...
  LABELKEY is the param to label each group by — "model", "n_keys", "input_scaling", or
  "auto" (model @ scaling, k=n_keys). Unknown -> auto.
"""
from __future__ import annotations

import json
import os
import sys


def _short_model(m: str) -> str:
    s = (m or "?").split("/")[-1]
    return s.replace("-Instruct", "").replace("Qwen2.5-", "Qwen-")


def _label(params: dict, key: str) -> str:
    if key == "model":
        return _short_model(params.get("model"))
    if key == "n_keys":
        return f"{params.get('n_keys', '?')} keys"
    if key == "input_scaling":
        return f"scale {params.get('input_scaling', '?')}"
    return f"{_short_model(params.get('model'))}\n@{params.get('input_scaling','?')}, k={params.get('n_keys','?')}"


def plot_recall_bars(records: list, out: str, title: str, label_key: str = "auto") -> None:
    """``records`` is a list of ``(params, stateful_acc, control_acc)`` tuples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [_label(p, label_key) for p, _, _ in records]
    stateful = [s for _, s, _ in records]
    control = [c for _, _, c in records]
    n = len(records)
    xs = list(range(n))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n), 4.5))
    ax.bar([x - w / 2 for x in xs], stateful, w, label="reservoir (carried state)", color="#1f6f8b", zorder=3)
    ax.bar([x + w / 2 for x in xs], control, w, label="stateless control (wiped)", color="#c0392b", zorder=3)
    # per-config chance line (1/n_keys) as small ticks
    for x, (p, _, _) in zip(xs, records):
        nk = p.get("n_keys") or 0
        if nk:
            ax.plot([x - 0.45, x + 0.45], [1.0 / nk, 1.0 / nk], color="#888", ls="--", lw=1, zorder=2)
    ax.plot([], [], color="#888", ls="--", lw=1, label="chance (1/n_keys)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("cross-pass recall accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load(paths: list) -> list:
    recs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        p = d.get("params", {})
        r = d.get("results", {})
        s = float(r.get("stateful", {}).get("recall_accuracy", 0.0))
        c = float(r.get("baseline", {}).get("recall_accuracy", 0.0))
        recs.append((p, s, c))
    return recs


def main() -> int:
    if len(sys.argv) < 5:
        print("usage: plot_recall_bars.py OUT.png TITLE LABELKEY a.json [b.json ...]")
        return 1
    out, title, label_key, paths = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("no input JSONs exist")
        return 1
    recs = load(paths)
    plot_recall_bars(recs, out, title, label_key)
    print(f"plotted {len(recs)} configs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
