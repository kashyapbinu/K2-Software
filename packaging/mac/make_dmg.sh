#!/usr/bin/env bash
# Package dist/K2.app into a distributable .dmg (run on a Mac, after build_mac.sh).
# Uses create-dmg if available (brew install create-dmg), else falls back to
# a plain hdiutil image.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
APP="dist/K2.app"
DMG="dist/K2.dmg"
[ -d "$APP" ] || { echo "ERROR: $APP not found — run build_mac.sh first"; exit 1; }
rm -f "$DMG"

if command -v create-dmg >/dev/null 2>&1; then
    create-dmg \
        --volname "K2 Aerospace" \
        --window-size 540 380 \
        --icon-size 110 \
        --icon "K2.app" 140 190 \
        --app-drop-link 400 190 \
        "$DMG" "$APP"
else
    echo "create-dmg not found — building plain dmg with hdiutil"
    hdiutil create -volname "K2 Aerospace" -srcfolder "$APP" -ov -format UDZO "$DMG"
fi

echo "==> $DMG ready"
