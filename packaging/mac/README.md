# macOS packaging

Builds a public-distributable `K2.dmg`. **Windows is untouched** — these files
are macOS-only and the Windows build keeps using `K2.spec` / `installer.iss`.

## Prerequisites (on a Mac)
- Python 3.11
- Mac-native solver binaries staged at `bin/mac-<arch>/` (`arch` = `arm64` or `x86_64`):
  - `bin/mac-<arch>/SU2_CFD`  — from https://su2code.github.io/download.html (or build)
  - `bin/mac-<arch>/ccx`      — `brew install calculix-ccx` then copy, or build
  - These are picked up automatically by `core.paths.bin_dir()`.
- For signing (public release): Apple Developer Program account ($99/yr).

## Local build
```bash
bash packaging/mac/build_mac.sh     # -> dist/K2.app
bash packaging/mac/make_dmg.sh      # -> dist/K2.dmg
bash packaging/mac/sign_notarize.sh # no-op until Apple cert configured
```

## CI build
`.github/workflows/build-mac.yml` — manual (`workflow_dispatch`) or on `v*` tag.
Runs on `macos-14` (arm64). Uploads `K2.dmg` as an artifact. TODO: wire solver
download in the "Stage mac solver binaries" step.

## Signing status: NOT YET configured
The `.dmg` is currently **unsigned**. Users must right-click → Open the first
time (Gatekeeper warns). To make it clean public-ready:
1. Enroll: https://developer.apple.com/programs/
2. Create a "Developer ID Application" certificate.
3. `xcrun notarytool store-credentials AC_PROFILE --apple-id <you> --team-id <TEAMID> --password <app-specific-pw>`
4. Local: `export DEV_ID_APP="Developer ID Application: ... (TEAMID)"; export AC_PROFILE=AC_PROFILE`
   CI: add repo secrets `DEV_ID_APP`, `AC_PROFILE` (+ cert import step).
5. Re-run `sign_notarize.sh` / the workflow. No code changes needed.

## Known Mac TODO in app code
- `ui/workspaces/cfd_workspace.py:1636` — disk-usage check uses a Windows
  drive string (`"C:\\"`); add a darwin branch before shipping.
