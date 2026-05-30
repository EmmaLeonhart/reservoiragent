"""Entry point for Reservoir Agent experiments.

Subcommands:
    python scripts/run.py --version          # print the package version
    python scripts/run.py sweep              # spectral-radius dynamics sweep
        [--K 200] [--rho-min 0.1 --rho-max 2.0 --n 20]
        [--out results/sweep_synthetic.json] [--fig docs/sweep_synthetic.png]

Metrics are written to results/ (gitignored) and a figure to docs/ (published).
"""
import argparse
import json
import sys
from pathlib import Path

# Make stdout UTF-8 so unicode (ρ, ∈, …) prints on Windows consoles (cp1252) too.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import reservoir  # noqa: E402


def cmd_sweep(args) -> int:
    import numpy as np
    from reservoir.sweep import run_sweep, plot_sweep, healthy_regime

    rhos = list(np.round(np.linspace(args.rho_min, args.rho_max, args.n), 4))
    print(f"Sweeping K={args.K} over ρ in [{args.rho_min}, {args.rho_max}] "
          f"({args.n} points)…")
    records = run_sweep(rhos, K=args.K, T=args.T, washout=args.washout, seed=args.seed)

    out = (ROOT / args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": {"K": args.K, "T": args.T, "washout": args.washout,
                          "seed": args.seed, "rhos": rhos},
               "records": records}
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.relative_to(ROOT)}")

    fig = ROOT / args.fig
    fig.parent.mkdir(parents=True, exist_ok=True)
    plot_sweep(records, str(fig))
    print(f"wrote {fig.relative_to(ROOT)}")

    healthy = healthy_regime(records)
    if healthy:
        lo, hi = min(h["rho"] for h in healthy), max(h["rho"] for h in healthy)
        best = max(healthy, key=lambda r: r["input_separation"])
        print(f"healthy regime (ESP holds, not over-saturated): ρ ∈ [{lo}, {hi}]")
        print(f"most responsive healthy ρ = {best['rho']} "
              f"(input_separation={best['input_separation']:.3f}, "
              f"PR/K={best['participation_ratio_frac']:.2f})")
    else:
        print("no ρ in the grid met the healthy-regime criteria")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reservoir Agent experiment runner")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    sub = parser.add_subparsers(dest="command")

    s = sub.add_parser("sweep", help="spectral-radius dynamics sweep")
    s.add_argument("--K", type=int, default=200)
    s.add_argument("--rho-min", type=float, default=0.1)
    s.add_argument("--rho-max", type=float, default=2.0)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--T", type=int, default=600)
    s.add_argument("--washout", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/sweep_synthetic.json")
    s.add_argument("--fig", default="docs/sweep_synthetic.png")
    s.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    if args.version:
        print(reservoir.__version__)
        return 0
    if getattr(args, "func", None) is not None:
        return args.func(args)

    print(f"reservoir-agent {reservoir.__version__} — package imports OK.")
    print("Try:  python scripts/run.py sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
