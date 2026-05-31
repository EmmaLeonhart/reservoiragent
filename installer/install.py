"""Reservoir Agent installer — standalone bootstrap launcher.

This is the entry point packaged into the downloadable ``reservoir-agent-installer.exe``
(see ``build_exe.py``). It must run with only the standard library — it does NOT import the
``reservoir`` package, because its first job is to *install* it.

What it does, in order:
1. Ensure the reservoir-agent package + its model deps are importable (``pip install`` from
   GitHub into the current interpreter / a venv if they are missing).
2. Hand off to ``python -m reservoir.installer``, which lists the project's reservoir-agent
   models (default = the recommended best), downloads the chosen one from the Hugging Face
   Hub, and drops the user into the stateful reservoir-agent console.

Why a bootstrap rather than a single frozen binary: a reservoir agent needs torch (+ CUDA)
and a multi-GB base model downloaded at run time, so freezing everything into one .exe is
impractical and fragile. The .exe is a thin launcher that sets the environment up and runs
the real package — that actually works.
"""
from __future__ import annotations

import subprocess
import sys

REPO = "git+https://github.com/EmmaLeonhart/reservoiragent.git"
# the package extras needed to load + run a reservoir agent (torch, transformers, peft, hub)
INSTALL_SPEC = f"reservoir-agent[models] @ {REPO}"


def _have_package() -> bool:
    try:
        import reservoir.installer.menu  # noqa: F401
        import torch  # noqa: F401
        import peft  # noqa: F401
        import huggingface_hub  # noqa: F401
        return True
    except Exception:
        return False


def ensure_installed() -> None:
    if _have_package():
        return
    print("Installing the reservoir-agent package and dependencies (first run)…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", INSTALL_SPEC])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ensure_installed()
    # hand off to the in-package menu (import only after install)
    from reservoir.installer.menu import main as menu_main
    return menu_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
