# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for K2 Aerospace.

Build:   pyinstaller K2.spec --noconfirm
Output:  dist/K2/K2.exe   (one-dir bundle — required: QtWebEngine + VTK do not
         work reliably in --onefile mode)

Bundles the project data/asset trees and the native solver binaries so the
frozen app finds them via the same  Path(__file__).parents[...] / "<dir>"
lookups it uses from source (one-dir keeps that relative layout under
_internal/).
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# ── Project data + native binaries (target path mirrors the source tree) ──────
datas = [
    ("data", "data"),                    # motor catalog, presets, cached curves
    ("visualization", "visualization"),  # cinematic three.js vendor + html/js
    ("bin", "bin"),                      # SU2_CFD.exe, ccx.exe (CalculiX), glut64.dll
]
binaries = []
hiddenimports = []

# ── Packages with dynamic imports / data files PyInstaller can't trace ────────
for pkg in ("pyvista", "pyvistaqt", "vtkmodules", "qtawesome", "sklearn", "reportlab"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("matplotlib")
hiddenimports += collect_submodules("scipy")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Dev-only trees — never imported by the app, keep them out of the bundle.
    # PyQt5/PySide* excluded: this machine has PyQt5 installed alongside PyQt6,
    # and PyInstaller aborts if it sees two Qt bindings. The app uses PyQt6 only.
    excludes=["tests", "validation", "scratch", "pytest", "_pytest",
              "PyQt5", "PySide2", "PySide6", "shiboken2", "shiboken6"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="K2",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # windowed app — no console window
    disable_windowed_traceback=False,
    icon="bin/k2.ico" if __import__("os").path.exists("bin/k2.ico") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="K2",
)
