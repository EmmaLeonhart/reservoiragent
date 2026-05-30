"""Entry point for Reservoir Agent experiments.

CI can invoke this; as experiments land (the spectral-radius dynamics sweep, the
model-surgery regression) they become subcommands here, writing metrics to
`results/` and figures to `docs/`. For now it reports the package is importable so
the scaffold + CI have something real to run.

Usage:
    python scripts/run.py            # smoke: confirm the package imports
    python scripts/run.py --version  # print the package version
"""
import argparse
import sys
from pathlib import Path

# Allow running from a checkout without an editable install.
SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import reservoir  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reservoir Agent experiment runner")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        print(reservoir.__version__)
        return 0

    print(f"reservoir-agent {reservoir.__version__} — package imports OK.")
    print("Experiments (dynamics sweep, model surgery) will be added as subcommands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
