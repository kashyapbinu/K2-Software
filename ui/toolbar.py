"""
K2 AeroSim — Toolbar
========================
Main application toolbar with essential actions:
New, Open, Save, Run Simulation, Settings.
"""

import logging
from PyQt6.QtWidgets import QToolBar, QWidget, QSizePolicy
from PyQt6.QtGui import QAction, QIcon, QFont
from PyQt6.QtCore import Qt

from ui.icons import icon

logger = logging.getLogger("K2.Toolbar")


class MainToolbar(QToolBar):
    """
    Primary application toolbar with file operations, simulation controls,
    and settings access.
    """
    
    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)
        self.setFloatable(False)
        self.setIconSize(parent.iconSize() if parent else self.iconSize())
        
        self._create_actions()
        self._build_toolbar()
    
    def _create_actions(self):
        """Create all toolbar actions."""
        # ── File actions ──
        self.action_new = QAction(icon("new"), "New", self)
        self.action_new.setToolTip("Create a new rocket project (Ctrl+N)")
        self.action_new.setShortcut("Ctrl+N")

        self.action_open = QAction(icon("open"), "Open", self)
        self.action_open.setToolTip("Open an existing project (Ctrl+O)")
        self.action_open.setShortcut("Ctrl+O")

        self.action_save = QAction(icon("save"), "Save", self)
        self.action_save.setToolTip("Save current project (Ctrl+S)")
        self.action_save.setShortcut("Ctrl+S")

        self.action_save_as = QAction(icon("save_as"), "Save As", self)
        self.action_save_as.setToolTip("Save project to a new file (Ctrl+Shift+S)")
        self.action_save_as.setShortcut("Ctrl+Shift+S")

        self.action_import_ork = QAction(icon("import"), "Import .ork", self)
        self.action_import_ork.setToolTip("Import an OpenRocket design file (Ctrl+I)")
        self.action_import_ork.setShortcut("Ctrl+I")

        # ── Simulation actions ──
        self.action_run_sim = QAction(icon("run", color="#3fb950"), "Run Sim", self)
        self.action_run_sim.setToolTip("Run flight simulation (F5)")
        self.action_run_sim.setShortcut("F5")

        self.action_stop_sim = QAction(icon("stop", color="#f85149"), "Stop", self)
        self.action_stop_sim.setToolTip("Stop simulation")
        self.action_stop_sim.setEnabled(False)

        self.action_reset = QAction(icon("reset"), "Reset", self)
        self.action_reset.setToolTip("Reset simulation state")

        # ── View actions ──
        self.action_reset_view = QAction(icon("reset_view"), "Reset View", self)
        self.action_reset_view.setToolTip("Reset 3D camera to default view")

        # ── Settings ──
        self.action_settings = QAction(icon("settings"), "Settings", self)
        self.action_settings.setToolTip("Application settings")

        self.action_check_updates = QAction(icon("update"), "Update", self)
        self.action_check_updates.setToolTip("Check for software updates")
    
    def _build_toolbar(self):
        """Add actions to toolbar with separators."""
        # File group
        self.addAction(self.action_new)
        self.addAction(self.action_open)
        self.addAction(self.action_save)
        self.addAction(self.action_save_as)
        self.addAction(self.action_import_ork)
        
        self.addSeparator()
        
        # View group
        self.addAction(self.action_reset_view)
        
        # Spacer to push settings to the right
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        self.addWidget(spacer)

        self.addAction(self.action_check_updates)
        self.addAction(self.action_settings)
