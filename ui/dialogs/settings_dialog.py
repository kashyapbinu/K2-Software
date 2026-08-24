"""
Application settings.

Sectioned preferences dialog. Appearance changes apply as soon as they are
picked so the choice can be judged in place; everything else is written on OK.
"""

import logging
import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLabel,
    QComboBox, QCheckBox, QPushButton, QLineEdit, QFileDialog, QWidget,
    QDialogButtonBox, QListWidget, QStackedWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import QApplication

from ui import settings, theme
from ui.icons import icon

from ui import theme

logger = logging.getLogger("K2.SettingsDialog")


class SettingsDialog(QDialog):
    """Preferences, grouped into pages down the left edge."""

    theme_changed = pyqtSignal(str)   # 'dark' | 'light'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumSize(660, 460)
        self._initial_theme = settings.theme_mode()

        self._nav = QListWidget()
        self._nav.setFixedWidth(168)
        self._nav.setIconSize(QSize(15, 15))
        self._pages = QStackedWidget()

        self._add_page("Appearance", "settings", self._appearance_page())
        self._add_page("Console", "console" if _has_icon("console") else "settings",
                       self._console_page())
        self._add_page("Projects", "open", self._projects_page())
        self._add_page("Updates", "update", self._updates_page())
        self._nav.setCurrentRow(0)
        self._nav.currentRowChanged.connect(self._pages.setCurrentIndex)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._nav)
        body.addWidget(self._pages, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self._on_reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            self._restore_defaults
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setProperty("primary", True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 12)
        outer.setSpacing(12)
        outer.addLayout(body, 1)
        outer.addWidget(buttons)

    # -- page scaffolding -------------------------------------------------
    def _add_page(self, title: str, icon_key: str, widget: QWidget):
        item = QListWidgetItem(icon(icon_key), title)
        self._nav.addItem(item)
        self._pages.addWidget(widget)

    @staticmethod
    def _page(*groups) -> QWidget:
        w = QWidget()
        lo = QVBoxLayout(w)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(10)
        for g in groups:
            lo.addWidget(g)
        lo.addStretch()
        return w

    @staticmethod
    def _hint(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        return lbl

    # -- pages ------------------------------------------------------------
    def _appearance_page(self) -> QWidget:
        g = QGroupBox("Theme")
        f = QFormLayout(g)
        f.setSpacing(8)

        self.cmb_theme = QComboBox()
        self.cmb_theme.addItem("Dark", "dark")
        self.cmb_theme.addItem("Light", "light")
        idx = self.cmb_theme.findData(settings.theme_mode())
        self.cmb_theme.setCurrentIndex(max(idx, 0))
        self.cmb_theme.currentIndexChanged.connect(self._on_theme_picked)
        f.addRow("Colour scheme:", self.cmb_theme)

        f.addRow("", self._hint(
            "Applies immediately, including charts and 3D views."
        ))
        return self._page(g)

    def _console_page(self) -> QWidget:
        g = QGroupBox("Log output")
        f = QFormLayout(g)
        f.setSpacing(8)

        self.cmb_log = QComboBox()
        self.cmb_log.addItems(settings.LOG_LEVELS)
        current = str(settings.get("console/log_level") or "INFO").upper()
        self.cmb_log.setCurrentText(current if current in settings.LOG_LEVELS else "INFO")
        f.addRow("Minimum level:", self.cmb_log)
        f.addRow("", self._hint(
            "DEBUG is verbose: solver iterations, mesh sizing, per-step state."
        ))
        return self._page(g)

    def _projects_page(self) -> QWidget:
        g = QGroupBox("Default project folder")
        lo = QVBoxLayout(g)
        lo.setSpacing(8)

        row = QHBoxLayout()
        self.edit_dir = QLineEdit(str(settings.get("projects/default_dir") or ""))
        self.edit_dir.setPlaceholderText(str(settings.project_dir()))
        row.addWidget(self.edit_dir, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_dir)
        row.addWidget(btn)
        lo.addLayout(row)
        lo.addWidget(self._hint("Leave blank to use Documents/K2 AeroSim Projects."))

        g2 = QGroupBox("Exit")
        f2 = QFormLayout(g2)
        self.chk_confirm_exit = QCheckBox("Ask before closing with unsaved changes")
        self.chk_confirm_exit.setChecked(bool(settings.get("sim/confirm_on_exit")))
        f2.addRow(self.chk_confirm_exit)
        return self._page(g, g2)

    def _updates_page(self) -> QWidget:
        g = QGroupBox("Automatic checks")
        f = QFormLayout(g)
        self.chk_updates = QCheckBox("Check for updates on launch")
        self.chk_updates.setChecked(bool(settings.get("startup/check_updates")))
        f.addRow(self.chk_updates)
        if not getattr(sys, "frozen", False):
            f.addRow("", self._hint(
                "Update checks only run in installed builds, not when running "
                "from source."
            ))
        return self._page(g)

    # -- actions ----------------------------------------------------------
    def _browse_dir(self):
        start = self.edit_dir.text().strip() or str(settings.project_dir())
        chosen = QFileDialog.getExistingDirectory(self, "Default project folder", start)
        if chosen:
            self.edit_dir.setText(chosen)

    def _on_theme_picked(self):
        mode = self.cmb_theme.currentData()
        self._apply_theme(mode)

    @staticmethod
    def _apply_theme(mode: str):
        theme.set_mode(mode)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.stylesheet())
        theme.apply_matplotlib_theme()

    def _restore_defaults(self):
        self.cmb_theme.setCurrentIndex(self.cmb_theme.findData(
            settings.DEFAULTS["appearance/theme"]))
        self.cmb_log.setCurrentText(settings.DEFAULTS["console/log_level"])
        self.edit_dir.setText("")
        self.chk_updates.setChecked(bool(settings.DEFAULTS["startup/check_updates"]))
        self.chk_confirm_exit.setChecked(bool(settings.DEFAULTS["sim/confirm_on_exit"]))

    def _on_accept(self):
        mode = self.cmb_theme.currentData()
        settings.set("appearance/theme", mode)
        settings.set("console/log_level", self.cmb_log.currentText())
        settings.set("projects/default_dir", self.edit_dir.text().strip())
        settings.set("startup/check_updates", self.chk_updates.isChecked())
        settings.set("sim/confirm_on_exit", self.chk_confirm_exit.isChecked())
        settings.apply_log_level()
        if mode != self._initial_theme:
            self.theme_changed.emit(mode)
        logger.info("Settings saved (theme=%s, log=%s)", mode, self.cmb_log.currentText())
        self.accept()

    def _on_reject(self):
        # roll back the live theme preview
        if self.cmb_theme.currentData() != self._initial_theme:
            self._apply_theme(self._initial_theme)
        self.reject()


def _has_icon(key: str) -> bool:
    from ui.icons import _MAP
    return key in _MAP
