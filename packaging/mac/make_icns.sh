#!/usr/bin/env bash
# Generate bin/k2.icns from a square PNG (macOS only — needs sips + iconutil).
# Run once before the .app build. Source PNG should be >=1024x1024 ideally;
# bin/k2_icon.png is used by default.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="${1:-$ROOT/bin/k2_icon.png}"
OUT="$ROOT/bin/k2.icns"
SET="$(mktemp -d)/k2.iconset"
mkdir -p "$SET"

for s in 16 32 64 128 256 512; do
    sips -z $s $s        "$SRC" --out "$SET/icon_${s}x${s}.png"      >/dev/null
    sips -z $((s*2)) $((s*2)) "$SRC" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null
done

iconutil -c icns "$SET" -o "$OUT"
echo "wrote $OUT"
