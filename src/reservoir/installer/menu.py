"""The installer menu — list reservoir-agent models, let the user pick, launch the console.

``choose_repo`` (pure, unit-tested) maps the displayed list + a raw input string to a repo
id: empty input takes the recommended default; a number selects that row. The interactive
display and the console launch are thin wrappers around it.
"""
from __future__ import annotations

from .registry import list_models, default_model, sort_by_recency


def choose_repo(models: list[dict], raw: str) -> str:
    """Resolve a menu choice to a repo id.

    Empty/whitespace -> the recommended default (most-recent recommended, else most-recent).
    A 1-based number -> that row of the displayed (recency-sorted) list. Anything else, or
    out of range, raises ``ValueError``.
    """
    if not models:
        raise ValueError("no models available to choose from")
    ordered = sort_by_recency(models)
    raw = (raw or "").strip()
    if not raw:
        return default_model(models)["repo_id"]
    if not raw.isdigit():
        raise ValueError(f"not a valid choice: {raw!r}")
    i = int(raw)
    if not (1 <= i <= len(ordered)):
        raise ValueError(f"choice {i} out of range 1..{len(ordered)}")
    return ordered[i - 1]["repo_id"]


def _format_menu(models: list[dict]) -> str:
    ordered = sort_by_recency(models)
    default = default_model(models)
    lines = ["Available reservoir-agent models:"]
    for i, m in enumerate(ordered, 1):
        star = " (recommended)" if m.get("recommended") else ""
        is_def = " [default]" if default and m["repo_id"] == default["repo_id"] else ""
        lines.append(f"  {i}. {m['repo_id']}{star}{is_def}")
    lines.append("Enter a number, or press Enter for the default.")
    return "\n".join(lines)


def main(argv=None):
    import argparse
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")   # the demo prints ✓/✗ marks
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Install and run a reservoir-agent model.")
    ap.add_argument("--repo-id", default=None, help="skip selection; load this repo directly")
    ap.add_argument("--no-hf", action="store_true", help="use only the bundled model list")
    ap.add_argument("--list", action="store_true", help="list models and exit")
    ap.add_argument("--menu", action="store_true",
                    help="show the model chooser instead of auto-picking the default")
    ap.add_argument("--demo-only", action="store_true",
                    help="run the recall demo and exit (no REPL)")
    ap.add_argument("--no-demo", action="store_true",
                    help="skip the recall demo, go straight to the REPL")
    args = ap.parse_args(argv)

    from . import console

    models = list_models(use_hf=not args.no_hf)
    if args.list:
        print(_format_menu(models))
        return 0

    repo_id = args.repo_id
    if not repo_id:
        if args.menu:
            print(_format_menu(models))
            repo_id = choose_repo(models, input("model> "))
        else:
            d = default_model(models)
            if d is None:
                print("no reservoir-agent models available to run", file=sys.stderr)
                return 1
            repo_id = d["repo_id"]

    print(f"loading {repo_id} …")
    console.run(repo_id, demo=not args.no_demo, repl=not args.demo_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
