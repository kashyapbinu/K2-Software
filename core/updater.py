"""
K2 AeroSim — in-app auto-updater (installer-swap strategy).

Flow
----
1. Query the GitHub Releases API for the latest published release of
   ``core.version.GITHUB_REPO``.
2. Compare its tag (``v0.1.2`` -> ``0.1.2``) against the running
   ``core.version.__version__``.
3. If newer, pick the platform installer asset (Windows ``.exe`` /
   macOS ``.dmg``), stream-download it to a temp file with progress.
4. Launch the installer and quit the app so the running files aren't
   locked while they're overwritten:
     - Windows: run the Inno Setup installer ``/SILENT`` then exit.
     - macOS:   ``open`` the ``.dmg`` (user drags to /Applications) then exit.

No third-party deps — uses urllib so it works in the frozen bundle without
adding to requirements.

Threading: network + download run on ``UpdateWorker`` (a ``QThread``); it
emits signals the UI binds to. Never touch widgets from the worker.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.version import __version__, GITHUB_REPO

logger = logging.getLogger("K2.Updater")

_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_TIMEOUT = 15  # seconds for the metadata request
_USER_AGENT = f"K2-AeroSim/{__version__}"


# ── Version compare (no hard dep on `packaging`) ──────────────────────────────
def _parse(v: str) -> tuple:
    """Loose semver tuple. 'v0.1.2' -> (0, 1, 2). Non-numeric parts -> 0.

    Strips a leading 'v' and anything after the first '-'/'+' (pre-release /
    build metadata) so '1.2.0-rc1' compares as (1, 2, 0).
    """
    v = v.strip().lstrip("vV")
    v = v.split("-")[0].split("+")[0]
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out) or (0,)


def is_newer(remote: str, local: str = __version__) -> bool:
    """True if *remote* version string is strictly newer than *local*."""
    a, b = _parse(remote), _parse(local)
    n = max(len(a), len(b))
    a += (0,) * (n - len(a))
    b += (0,) * (n - len(b))
    return a > b


def _asset_suffix() -> Optional[str]:
    """Installer file suffix expected on the current platform."""
    if sys.platform == "win32":
        return ".exe"
    if sys.platform == "darwin":
        return ".dmg"
    return None  # Linux: no installer asset convention yet


@dataclass
class UpdateInfo:
    version: str          # e.g. "0.1.2"
    tag: str              # e.g. "v0.1.2"
    notes: str            # release body (markdown)
    asset_name: str       # installer filename
    asset_url: str        # browser_download_url
    asset_size: int       # bytes


def _http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": _USER_AGENT,
        "Accept": "application/vnd.github+json",
    })
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_latest() -> Optional[UpdateInfo]:
    """Return UpdateInfo for the latest release, or None if none/older/no asset.

    Raises urllib.error.URLError on network failure (caller handles).
    """
    data = _http_get_json(_API_URL)
    tag = data.get("tag_name") or ""
    if not tag:
        logger.info("latest release has no tag_name")
        return None
    version = tag.lstrip("vV")
    if not is_newer(version):
        logger.info("up to date (local %s >= remote %s)", __version__, version)
        return None

    suffix = _asset_suffix()
    if suffix is None:
        logger.info("no installer convention for platform %s", sys.platform)
        return None

    asset = None
    for a in data.get("assets", []):
        name = a.get("name", "")
        if name.lower().endswith(suffix):
            asset = a
            break
    if asset is None:
        logger.warning("release %s has no %s asset", tag, suffix)
        return None

    return UpdateInfo(
        version=version,
        tag=tag,
        notes=data.get("body") or "",
        asset_name=asset.get("name", f"K2-Setup{suffix}"),
        asset_url=asset["browser_download_url"],
        asset_size=int(asset.get("size", 0)),
    )


def _download(url: str, dest: str, progress_cb=None, is_cancelled=None) -> None:
    """Stream *url* to *dest*. progress_cb(pct:int) called 0..100.

    Raises on failure. Removes the partial file if cancelled.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1024 * 256
        with open(dest, "wb") as f:
            while True:
                if is_cancelled and is_cancelled():
                    f.close()
                    try:
                        os.remove(dest)
                    except OSError:
                        pass
                    raise RuntimeError("cancelled")
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                read += len(buf)
                if progress_cb and total:
                    progress_cb(min(100, int(read * 100 / total)))
    if progress_cb:
        progress_cb(100)


def launch_installer_and_quit(installer_path: str) -> None:
    """Start the downloaded installer detached, so it can replace the running
    app after this process exits. Caller should quit the app right after.
    """
    if sys.platform == "win32":
        # Inno Setup: /SILENT shows only a progress bar, no wizard pages.
        # DETACHED so it survives our exit; it will relaunch K2 via the
        # installer's [Run] postinstall entry.
        DETACHED = 0x00000008  # CREATE_NEW_PROCESS_GROUP-friendly detach
        subprocess.Popen(
            [installer_path, "/SILENT", "/NOCANCEL"],
            creationflags=DETACHED | subprocess.CREATE_NO_WINDOW,
            close_fds=True,
        )
    elif sys.platform == "darwin":
        # Mount the dmg in Finder; user drags the app to /Applications.
        subprocess.Popen(["open", installer_path])
    else:
        raise RuntimeError(f"no installer launch path for {sys.platform}")


class UpdateWorker(QThread):
    """Background check + download. Bind signals from the UI thread."""

    # check phase
    update_available = pyqtSignal(object)   # UpdateInfo
    no_update = pyqtSignal()
    error = pyqtSignal(str)
    # download phase
    progress = pyqtSignal(int)              # 0..100
    downloaded = pyqtSignal(str)            # local installer path

    MODE_CHECK = "check"
    MODE_DOWNLOAD = "download"

    def __init__(self, mode: str, info: Optional[UpdateInfo] = None, parent=None):
        super().__init__(parent)
        self._mode = mode
        self._info = info
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self._mode == self.MODE_CHECK:
                info = fetch_latest()
                if info is None:
                    self.no_update.emit()
                else:
                    self.update_available.emit(info)
            elif self._mode == self.MODE_DOWNLOAD:
                if self._info is None:
                    self.error.emit("no update selected")
                    return
                dest = os.path.join(
                    tempfile.gettempdir(), self._info.asset_name
                )
                _download(
                    self._info.asset_url, dest,
                    progress_cb=self.progress.emit,
                    is_cancelled=lambda: self._cancel,
                )
                self.downloaded.emit(dest)
        except urllib.error.URLError as e:
            self.error.emit(f"network error: {getattr(e, 'reason', e)}")
        except Exception as e:  # noqa: BLE001 — surface anything to the UI
            if str(e) == "cancelled":
                return
            logger.exception("update worker failed")
            self.error.emit(str(e))
