"""
K2 AeroSim — version single-source-of-truth.

Bump ``__version__`` here on every release, tag the repo ``v<version>``
(e.g. ``v0.1.2``), and publish a GitHub Release with the built installer
attached as an asset. The in-app updater (``core.updater``) compares this
value against the latest GitHub Release tag.

Keep these in sync with the same number:
  - this file
  - installer.iss            (#define MyAppVersion)
  - the git tag              (v<version>)
"""

__version__ = "0.1.1"

# GitHub repo that hosts the public Releases the updater pulls from.
# Form: "owner/name".
GITHUB_REPO = "kashyapbinu/K2-Software"
