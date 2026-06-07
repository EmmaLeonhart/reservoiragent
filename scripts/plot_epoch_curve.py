"""Plot the per-epoch training curve from a ``train_large.py`` run.

Reads ``artifacts/qwen-large/index.json`` (the per-epoch records written by
``train_large.py``'s ``checkpoint()`` — each has ``epoch``, ``mean``, and, when the inline
control is present, ``stateless_mean`` / ``lift``) and draws the capability mean vs the
stateless control across epochs, so "does more training help, and is any of it
reservoir-attributable?" is legible at a glance. Saves to ``docs/progressive_curve.png``.

This is the figure for the progressive run: we run a generous epoch budget precisely to see
the *shape* of the curve (when it peaks, whether it plateaus or overtrains, whether the lift
above the stateless floor ever goes positive) rather than guessing the optimum.

Run:  python scripts/plot_epoch_curve.py [index.json] [out.png]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_epoch_curve(records: list, path: str, *,
                     title: str = "Progressive battery training: capability vs stateless control") -> None:
    """Line plot of per-epoch capability ``mean`` and (if present) ``stateless_mean`` against
    epoch, with the lift shaded between them. ``records`` is the list from ``index.json``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: r.get("epoch", 0))
    xs = [r.get("epoch", i) for i, r in enumerate(recs)]
    means = [float(r.get("mean", 0.0)) for r in recs]
    have_ctrl = all("stateless_mean" in r for r in recs) and len(recs) > 0
    ctrl = [float(r.get("stateless_mean", 0.0)) for r in recs] if have_ctrl else None

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(xs, means, color="#1f6f8b", lw=2.4, marker="o", ms=5, zorder=4,
            label="reservoir (carried state)")
    if have_ctrl:
        ax.plot(xs, ctrl, color="#c0392b", lw=2.0, marker="s", ms=4, ls="--", zorder=3,
                label="stateless control (reservoir reset each pass)")
        ax.fill_between(xs, ctrl, means, where=[m >= c for m, c in zip(means, ctrl)],
                        color="#1f6f8b", alpha=0.12, zorder=1)
    best_i = max(range(len(means)), key=lambda i: means[i]) if means else 0
    if means:
        ax.annotate(f"peak {means[best_i]:.3f} @ epoch {xs[best_i]}",
                    (xs[best_i], means[best_i]), textcoords="offset points",
                    xytext=(0, 8), fontsize=8, color="#1f6f8b", ha="center")
    ax.set_xlabel("epoch")
    ax.set_ylabel("capability mean (emit-requiring tasks)")
    ax.set_ylim(-0.05, 1.05)
    if xs:
        ax.set_xticks(xs)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "artifacts", "qwen-large", "index.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "docs", "progressive_curve.png")
    if not os.path.exists(src):
        print(f"[plot_epoch_curve] no index at {src} (run not far enough along yet)")
        return 1
    with open(src, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        print(f"[plot_epoch_curve] {src} has no epochs yet")
        return 1
    plot_epoch_curve(records, out)
    print(f"[plot_epoch_curve] {len(records)} epochs -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
