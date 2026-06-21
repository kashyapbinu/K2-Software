"""
K2 AeroSim — Software Update dialog.

Drives ``core.updater``: checks GitHub Releases, shows the available version
and notes, downloads the installer with a progress bar, then launches it and
quits the app so the new version can overwrite the running one.

Usage:
    from ui.dialogs.update_dialog import check_for_updates
    check_for_updates(parent, silent=False)   # menu/toolbar action
    check_for_updates(parent, silent=True)    # startup: only speak if update
"""

import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar,
    QTextEdit, QMessageBox, QApplication,
)
from PyQt6.QtCore import Qt

from core.version import __version__
from core import updater
from core.updater import UpdateWorker, UpdateInfo

logger = logging.getLogger("K2.UpdateDialog")


class UpdateDialog(QDialog):
    """Modal dialog shown when an update is available."""

    def __init__(self, info: UpdateInfo, parent=None):
        super().__init__(parent)
        self.info = info
        self._worker = None
        self._installer_path = None

        self.setWindowTitle("Software Update")
        self.setMinimumWidth(520)

        lay = QVBoxLayout(self)

        head = QLabel(
            f"<b>K2 AeroSim {info.version} is available.</b><br>"
            f"You have {__version__}."
        )
        head.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(head)

        size_mb = info.asset_size / 1_048_576 if info.asset_size else 0
        sub = QLabel(
            f"Download: {info.asset_name}"
            + (f"  ({size_mb:.0f} MB)" if size_mb else "")
        )
        sub.setStyleSheet("color: #8b949e;")
        lay.addWidget(sub)

        if info.notes.strip():
            notes = QTextEdit()
            notes.setReadOnly(True)
            notes.setPlainText(info.notes.strip())
            notes.setMaximumHeight(200)
            lay.addWidget(QLabel("Release notes:"))
            lay.addWidget(notes)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.status = QLabel("")
        self.status.setStyleSheet("color: #8b949e;")
        lay.addWidget(self.status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_later = QPushButton("Later")
        self.btn_later.clicked.connect(self.reject)
        self.btn_update = QPushButton("Download && Install")
        self.btn_update.setDefault(True)
        self.btn_update.clicked.connect(self._start_download)
        btns.addWidget(self.btn_later)
        btns.addWidget(self.btn_update)
        lay.addLayout(btns)

    # ── download phase ────────────────────────────────────────────────────────
    def _start_download(self):
        self.btn_update.setEnabled(False)
        self.btn_later.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText("Downloading…")

        self._worker = UpdateWorker(UpdateWorker.MODE_DOWNLOAD, info=self.info, parent=self)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.downloaded.connect(self._on_downloaded)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_downloaded(self, path: str):
        self._installer_path = path
        self.status.setText("Download complete. Launching installer…")
        # Confirm the relaunch-and-quit step explicitly.
        resp = QMessageBox.information(
            self,
            "Install update",
            "The installer will now run and K2 AeroSim will close.\n"
            "Save any work before continuing.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if resp != QMessageBox.StandardButton.Ok:
            self.status.setText("Update downloaded — install cancelled.")
            self.btn_later.setEnabled(True)
            return
        try:
            updater.launch_installer_and_quit(path)
        except Exception as e:  # noqa: BLE001
            self._on_error(f"could not launch installer: {e}")
            return
        QApplication.quit()

    def _on_error(self, msg: str):
        self.progress.setVisible(False)
        self.status.setText("")
        self.btn_update.setEnabled(True)
        self.btn_later.setEnabled(True)
        QMessageBox.warning(self, "Update failed", msg)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
        super().closeEvent(event)


def check_for_updates(parent=None, silent: bool = False):
    """Kick off a background check. On result, show the update dialog
    (or, when not *silent*, an up-to-date / error message box).

    *silent* (startup auto-check): stay quiet unless an update exists.

    Returns the worker so the caller can keep a reference (else it's GC'd
    mid-flight). The worker parents to *parent* so it lives as long as the
    window.
    """
    worker = UpdateWorker(UpdateWorker.MODE_CHECK, parent=parent)

    def _on_available(info: UpdateInfo):
        UpdateDialog(info, parent).exec()

    def _on_none():
        if not silent:
            QMessageBox.information(
                parent, "Software Update",
                f"You're up to date.\nK2 AeroSim {__version__} is the latest version.",
            )

    def _on_error(msg: str):
        if not silent:
            QMessageBox.warning(
                parent, "Software Update",
                f"Could not check for updates.\n{msg}",
            )
        else:
            logger.info("silent update check failed: %s", msg)

    worker.update_available.connect(_on_available)
    worker.no_update.connect(_on_none)
    worker.error.connect(_on_error)
    worker.start()
    return worker
