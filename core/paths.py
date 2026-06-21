"""
Writable application paths.

When K2 runs from source (development) it keeps writing inside the repo, so
nothing about the dev workflow changes. When it runs as a frozen/installed app
(PyInstaller bundle under C:\\Program Files\\…, which is read-only for a normal
user) all generated files — the crash log, the motor-curve cache, CFD/FEM run
output — must go to a per-user writable location instead, or the app crashes
with PermissionError at launch.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _base() -> Path:
    """Base directory for app-generated, writable files."""
    if getattr(sys, "frozen", False):
        # Installed app — the bundle dir is read-only. Use a per-user folder.
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "K2Aerospace"
        if sys.platform == "win32":
            root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
            return Path(root) / "K2Aerospace"
        # Linux / other POSIX — XDG base dir.
        root = os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share")
        return Path(root) / "K2Aerospace"
    # Source / dev run — keep everything inside the repo (unchanged behavior).
    return Path(__file__).resolve().parents[1]


def bin_dir() -> Path:
    """Directory holding the bundled native solver binaries (ccx, SU2_CFD).

    Prefers a platform/arch-specific subdir when present
    (``bin/mac-arm64``, ``bin/mac-x86_64``), else falls back to the ``bin/``
    root. On Windows the subdir never exists, so this returns the same
    ``bin/`` root as before — Windows lookups are unchanged.
    """
    root = Path(__file__).resolve().parents[1] / "bin"
    if sys.platform == "darwin":
        sub = root / f"mac-{platform.machine()}"   # mac-arm64 / mac-x86_64
        if sub.is_dir():
            return sub
    return root


def user_data_dir(sub: str = "") -> Path:
    """Return a writable directory (created if needed). *sub* is an optional
    sub-path like ``"cfd_run"`` or ``"data/thrust_curves"``."""
    p = _base() / sub if sub else _base()
    p.mkdir(parents=True, exist_ok=True)
    return p
