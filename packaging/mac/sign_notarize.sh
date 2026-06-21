#!/usr/bin/env bash
# Codesign + notarize + staple the .app and .dmg (run on a Mac).
#
# GUARDED: if the signing env vars are unset, this is a NO-OP that just prints
# how to enroll. So it is safe to call today (no Apple Developer account yet) —
# the unsigned .app still runs via right-click -> Open. Once you enroll, export
# the vars below and re-run; nothing else changes.
#
# Required env vars (set when you have an Apple Developer Program account):
#   DEV_ID_APP   "Developer ID Application: Your Name (TEAMID)"   # signing identity
#   AC_PROFILE   keychain profile name created via:
#                  xcrun notarytool store-credentials AC_PROFILE \
#                    --apple-id you@example.com --team-id TEAMID --password <app-specific-pw>
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
APP="dist/K2.app"
DMG="dist/K2.dmg"
ENTITLEMENTS="packaging/mac/entitlements.plist"

if [ -z "${DEV_ID_APP:-}" ] || [ -z "${AC_PROFILE:-}" ]; then
    cat <<'EOF'
[sign_notarize] SKIPPED — no signing identity configured.

The .app/.dmg are UNSIGNED. Users must right-click -> Open the first time
(Gatekeeper will warn). For a clean public release:

  1. Enroll: https://developer.apple.com/programs/  ($99/yr)
  2. Create a "Developer ID Application" certificate in Xcode/Keychain.
  3. Store notarization creds once:
       xcrun notarytool store-credentials AC_PROFILE \
         --apple-id you@example.com --team-id TEAMID --password <app-specific-pw>
  4. export DEV_ID_APP="Developer ID Application: Your Name (TEAMID)"
     export AC_PROFILE="AC_PROFILE"
  5. Re-run this script.
EOF
    exit 0
fi

echo "==> codesigning (hardened runtime, deep)"
codesign --force --deep --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" \
    --sign "$DEV_ID_APP" "$APP"
codesign --verify --strict --verbose=2 "$APP"

echo "==> notarizing .app (via dmg)"
bash packaging/mac/make_dmg.sh
codesign --force --timestamp --sign "$DEV_ID_APP" "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile "$AC_PROFILE" --wait
xcrun stapler staple "$DMG"
xcrun stapler staple "$APP"

echo "==> signed + notarized: $DMG"
