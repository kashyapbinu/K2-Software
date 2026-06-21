# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for K2 Aerospace — macOS .app bundle.

Build (on a Mac):   pyinstaller K2-mac.spec --noconfirm
Output:             dist/K2.app   (one-dir .app — QtWebEngine + VTK need it)

This is the macOS counterpart to K2.spec (Windows). It is NEVER used on
Windows — the Windows build keeps using K2.spec unchanged. Differences:
  * BUNDLE() wraps the COLLECT into a clickable .app
  * icon is .icns (run packaging/mac/make_icns.sh first)
  * bundle_identifier set for codesign / notarization
  * Info.plist marks the app non-document, dark-mode aware
"""

import platform
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# ── Project data + native binaries (target path mirrors the source tree) ──────
# NOTE: bundles the whole bin/ tree. The Mac app only needs bin/mac-<arch>/;
# the Windows *.exe are inert on macOS (PE, not Mach-O) but bloat the bundle —
# prune them in packaging/mac/build_mac.sh before signing if size matters.
datas = [
    ("data", "data"),
    ("visualization", "visualization"),
    ("bin", "bin"),
]
binaries = []
hiddenimports = []

for pkg in ("pyvista", "pyvistaqt", "vtkmodules", "qtawesome", "sklearn", "reportlab", "gmsh"):
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
    console=False,
    disable_windowed_traceback=False,
    # Match the .app slice to the build host (arm64 on Apple Silicon CI/dev).
    target_arch=platform.machine(),
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

import os
app = BUNDLE(
    coll,
    name="K2.app",
    icon="bin/k2.icns" if os.path.exists("bin/k2.icns") else None,
    bundle_identifier="com.k2aerospace.k2",
    info_plist={
        "CFBundleName": "K2",
        "CFBundleDisplayName": "K2 Aerospace",
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "NSHighResolutionCapable": True,
        "NSRequiresAquaSystemAppearance": False,   # allow dark mode
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.education",
    },
)
