#!/usr/bin/env bash
# Build the K2 macOS .app (run on a Mac).
#   1. fresh venv + deps
#   2. icns
#   3. pyinstaller -> dist/K2.app
#   4. (optional) prune Windows binaries from the bundle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ARCH="$(uname -m)"   # arm64 | x86_64
echo "==> building K2.app for $ARCH"

# Native solver binaries must already be staged at bin/mac-$ARCH/
#   bin/mac-$ARCH/SU2_CFD   bin/mac-$ARCH/ccx
if [ ! -d "bin/mac-$ARCH" ]; then
    echo "WARNING: bin/mac-$ARCH/ missing — app will build but CFD/FEM solvers won't run."
    echo "         place mac-native SU2_CFD and ccx there first."
fi

python3 -m venv .venv-mac
# shellcheck disable=SC1091
source .venv-mac/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller

bash packaging/mac/make_icns.sh || echo "icns skipped (no source png?)"

rm -rf build dist
pyinstaller K2-mac.spec --noconfirm

# Prune the inert Windows binaries from the .app to slim it + ease notarization.
find "dist/K2.app" -type d -name win -prune -exec rm -rf {} + 2>/dev/null || true
find "dist/K2.app" -name '*.exe' -delete 2>/dev/null || true
find "dist/K2.app" -name '*.dll' -delete 2>/dev/null || true

echo "==> dist/K2.app ready"
