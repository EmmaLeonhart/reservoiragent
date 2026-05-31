"""Build the downloadable Windows installer (``reservoir-agent-installer.exe``).

Freezes the **stdlib-only bootstrap** (``install.py``) into a one-file executable with
PyInstaller. The bootstrap imports nothing heavy, so the binary is small and builds fast —
torch, the package, and the chosen model are fetched at *run* time, not bundled here. That
is the whole point of the bootstrap design (see ``install.py``).

    pip install pyinstaller
    python installer/build_exe.py            # -> dist/reservoir-agent-installer.exe

This is invoked on a Windows runner by ``.github/workflows/build-installer.yml``.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.join(HERE, "install.py")
NAME = "reservoir-agent-installer"


def main() -> int:
    cmd = [sys.executable, "-m", "PyInstaller", "--onefile", "--console",
           "--name", NAME, ENTRY]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
