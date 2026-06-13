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
import sys
from pathlib import Path


def _base() -> Path:
    """Base directory for app-generated, writable files."""
    if getattr(sys, "frozen", False):
        # Installed app — the bundle dir is read-only. Use a per-user folder.
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(root) / "K2Aerospace"
    # Source / dev run — keep everything inside the repo (unchanged behavior).
    return Path(__file__).resolve().parents[1]


def user_data_dir(sub: str = "") -> Path:
    """Return a writable directory (created if needed). *sub* is an optional
    sub-path like ``"cfd_run"`` or ``"data/thrust_curves"``."""
    p = _base() / sub if sub else _base()
    p.mkdir(parents=True, exist_ok=True)
    return p
