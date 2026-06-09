"""
K2 Aerospace — Main Window
=============================
Tabbed workspace layout with 7 engineering workspaces.
Global toolbar and console dock.
"""

import sys, logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QFileDialog, QTabWidget,
    QMessageBox, QApplication, QLabel, QStatusBar
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from core.rocket_state import RocketStateEngine
from core.simulation_engine import SimulationEngine
from core.project_manager import save_project, load_project, get_default_project_dir
from core.event_manager import SimEvent
from avionics.flight_computer.flight_computer import FlightComputer
from ui.toolbar import MainToolbar
from ui.console_panel import ConsolePanel
from ui.workspaces.design_workspace import DesignWorkspace
from ui.workspaces.propulsion_workspace import PropulsionWorkspace
from ui.workspaces.structures_workspace import StructuresWorkspace
from ui.workspaces.avionics_workspace import AvionicsWorkspace
from ui.workspaces.simulation_workspace import SimulationWorkspace
from ui.workspaces.mission_visualizer_workspace import MissionVisualizerWorkspace
from ui.workspaces.results_workspace import ResultsWorkspace
from ui.workspaces.cfd_workspace import CFDWorkspace
from ui.workspaces.monte_carlo_workspace import MonteCarloWorkspace
from ui.workspaces.optimization_workspace import OptimizationWorkspace

logger = logging.getLogger("K2.MainWindow")


class MainWindow(QMainWindow):
    """K2 Aerospace main application window with tabbed workspaces."""

    TAB_ICONS = ["🛠", "🔥", "🌊", "🏗", "🔬", "📡", "🚀", "🌍", "📊", "🎲", "🎯"]
    TAB_NAMES = ["Design", "Propulsion", "CFD", "Structures", "Dynamics", "Avionics", "Simulation", "Mission Visualizer", "Results", "Monte Carlo", "Optimization"]

    def __init__(self):
        super().__init__()
        self._current_file = None

        # ── State Engine ──
        self.engine = RocketStateEngine()

        # ── Simulation Engine ──
        self.sim_engine = SimulationEngine(self.engine)

        # ── Flight Computer ──
        self.flight_computer = FlightComputer(self.sim_engine.event_mgr)

        # ── Window setup ──
        self.setWindowTitle("K2 Aerospace — Rocket Simulation Platform")
        self.setMinimumSize(1200, 750)
        self.resize(1600, 950)

        # ── Build UI ──
        self._setup_toolbar()
        self._setup_tabs()
        self._setup_bottom_dock()
        self._setup_status_bar()
        self._connect_actions()
        self._connect_sim_signals()

        # ── Initial state push ──
        QTimer.singleShot(200, self._initial_state_push)

        logger.info("K2 Aerospace initialized — 11 workspaces ready")

    def _setup_toolbar(self):
        self.toolbar = MainToolbar(self)
        self.addToolBar(self.toolbar)

    def _setup_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self.tab_widget.setDocumentMode(True)

        # Create workspaces
        from ui.workspaces.dynamics_workspace import DynamicsWorkspace
        self.design_ws = DesignWorkspace(self.engine, self)
        self.propulsion_ws = PropulsionWorkspace(self.engine, self)
        self.structures_ws = StructuresWorkspace(self.engine, self)
        self.dynamics_ws = DynamicsWorkspace(self.engine, self)
        self.avionics_ws = AvionicsWorkspace(self.engine, self)
        self.avionics_ws.set_flight_computer(self.flight_computer)
        self.simulation_ws = SimulationWorkspace(self.engine, self.sim_engine, self)
        self.mission_viz_ws = MissionVisualizerWorkspace(self.engine, self.sim_engine, self)
        self.results_ws = ResultsWorkspace(self.engine, self.sim_engine, self)
        self.monte_carlo_ws = MonteCarloWorkspace(self.engine, self.sim_engine, self)
        self.optimization_ws = OptimizationWorkspace(self.engine, self.sim_engine, self)

        self.cfd_ws = CFDWorkspace(
            self.engine,
            assembly_provider=lambda: getattr(self.design_ws, 'assembly', None),
            parent=self
        )

        workspaces = [
            self.design_ws, self.propulsion_ws, self.cfd_ws,
            self.structures_ws, self.dynamics_ws, self.avionics_ws, self.simulation_ws,
            self.mission_viz_ws, self.results_ws, self.monte_carlo_ws, self.optimization_ws
        ]

        for icon, name, ws in zip(self.TAB_ICONS, self.TAB_NAMES, workspaces):
            self.tab_widget.addTab(ws, f"{icon}  {name}")

        self.setCentralWidget(self.tab_widget)

    def _setup_bottom_dock(self):
        dock = QDockWidget("Console", self)
        dock.setMaximumHeight(200)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.console_panel = ConsolePanel(self)
        dock.setWidget(self.console_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        
        # Connect engine log messages to the console
        def _route_log(msg):
            level = "INFO"
            if "⚠" in msg or "Warning" in msg:
                level = "WARNING"
            elif "❌" in msg or "Error" in msg or "Failed" in msg:
                level = "ERROR"
            self.console_panel.log(msg, level)
            
        self.engine.log_message.connect(_route_log)

    def _setup_status_bar(self):
        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label, 1)
        self.status_motor = QLabel("Motor: None")
        self.status_motor.setStyleSheet("color: #8b949e; padding-right: 8px;")
        status.addPermanentWidget(self.status_motor)
        self.status_sim = QLabel("SIM: Idle")
        self.status_sim.setStyleSheet("color: #8b949e; padding-right: 12px;")
        status.addPermanentWidget(self.status_sim)

    def _connect_actions(self):
        tb = self.toolbar
        tb.action_new.triggered.connect(self._on_new)
        tb.action_open.triggered.connect(self._on_open)
        tb.action_save.triggered.connect(self._on_save)
        tb.action_save_as.triggered.connect(self._on_save_as)
        tb.action_import_ork.triggered.connect(self._on_import_ork)
        tb.action_run_sim.triggered.connect(self._on_run_sim)
        tb.action_stop_sim.triggered.connect(self._on_stop_sim)
        tb.action_reset.triggered.connect(self._on_reset)
        tb.action_reset_view.triggered.connect(self._on_reset_view)

        self.engine.state_changed.connect(self._on_state_changed)

    def _connect_sim_signals(self):
        self.sim_engine.sim_started.connect(lambda: self._set_sim_status("RUNNING", "#7ee787"))
        self.sim_engine.sim_started.connect(
            lambda: QTimer.singleShot(300, lambda: self.tab_widget.setCurrentIndex(7))
        )
        self.sim_engine.sim_paused.connect(lambda: self._set_sim_status("PAUSED", "#d29922"))
        self.sim_engine.sim_resumed.connect(lambda: self._set_sim_status("RUNNING", "#7ee787"))
        self.sim_engine.sim_finished.connect(self._on_sim_finished)

        # Auto-switch to results when sim finishes
        self.sim_engine.sim_finished.connect(self._show_results)

        # Tick the flight computer on every telemetry update
        self.engine.telemetry_tick.connect(self._tick_flight_computer)

        # Wire event manager to console
        em = self.sim_engine.event_mgr
        em.subscribe(SimEvent.MOTOR_BURNOUT, lambda d: self.engine.log_message.emit(
            f"🔥 Motor burnout at T+{d.get('time',0):.2f}s, alt={d.get('altitude',0):.1f}m"))
        em.subscribe(SimEvent.APOGEE, lambda d: self.engine.log_message.emit(
            f"🎯 Apogee reached at T+{d.get('time',0):.2f}s, alt={d.get('altitude',0):.1f}m"))
        em.subscribe(SimEvent.DROGUE_DEPLOY, lambda d: self.engine.log_message.emit(
            f"🪂 Drogue deployed at T+{d.get('time',0):.2f}s, alt={d.get('altitude',0):.1f}m"))
        em.subscribe(SimEvent.MAIN_DEPLOY, lambda d: self.engine.log_message.emit(
            f"🪂 Main chute deployed at T+{d.get('time',0):.2f}s, alt={d.get('altitude',0):.1f}m"))
        em.subscribe(SimEvent.LANDING, lambda d: self.engine.log_message.emit(
            f"🛬 Landing at T+{d.get('time',0):.2f}s — Apogee: {d.get('max_altitude',0):.1f}m"))
        em.subscribe(SimEvent.MAX_Q, lambda d: self.engine.log_message.emit(
            f"💨 Max-Q: {d.get('max_q',0):.0f} Pa at Mach {d.get('mach',0):.3f}"))

    def _set_sim_status(self, text, color):
        self.status_sim.setText(f"SIM: {text}")
        self.status_sim.setStyleSheet(f"color: {color}; padding-right: 12px; font-weight: 600;")
        self.toolbar.action_run_sim.setEnabled(text != "RUNNING")
        self.toolbar.action_stop_sim.setEnabled(text == "RUNNING")

    def _tick_flight_computer(self, s):
        """Update flight computer with latest simulation state."""
        if not s.sim_running:
            return
            
        # Tick FC with true values; it will add its own sensor noise
        self.flight_computer.tick(
            true_accel=s.acceleration,
            true_pressure=s.atm_pressure,
            true_altitude=s.altitude,
            true_velocity=s.velocity,
            t=s.sim_time,
            gyro_rates=(s.gyro_x, s.gyro_y, s.gyro_z)
        )
        
        # Optionally: you could feed FC estimates back to the state here
        # For now, we just ensure the FC internal state machine progresses
        # (Implementation of FC state logic would go in FlightComputer.tick)

    def _on_sim_finished(self):
        self._set_sim_status("COMPLETE", "#58a6ff")
        self.toolbar.action_stop_sim.setEnabled(False)
        self.toolbar.action_run_sim.setEnabled(True)

    def _show_results(self):
        """Auto-refresh results and switch to Results tab."""
        def _do():
            self.results_ws.refresh_plots()
            self.tab_widget.setCurrentIndex(8)
        QTimer.singleShot(300, _do)

    def _initial_state_push(self):
        self.engine.update(name="Untitled Rocket")

    def _on_state_changed(self, state):
        self.status_motor.setText(f"Motor: {state.motor_designation}")

    # ── File operations ──

    def _on_new(self):
        reply = QMessageBox.question(self, "New Project",
            "Create a new project? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.sim_engine.is_running:
                self.sim_engine.stop()
            self.engine.reset()
            from core.components import RocketAssembly
            self.design_ws.set_assembly(RocketAssembly())
            self._current_file = None
            self._update_title()
            logger.info("New project created")

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project",
            str(Path.home() / "OneDrive" / "Desktop"),
            "All Supported (*.k2proj *.ork *.json);;K2 Projects (*.k2proj);;OpenRocket Files (*.ork);;JSON Files (*.json);;All Files (*)")
        if path:
            if path.lower().endswith('.ork'):
                self._on_import_ork_path(path)
                return
            state = load_project(path)
            if state:
                self.engine.set_state(state)
                self._current_file = path
                self._update_title()
                logger.info(f"Opened: {path}")
            else:
                QMessageBox.critical(self, "Error", f"Failed to load project:\n{path}")

    def _on_save(self):
        if self._current_file:
            self._save_to(self._current_file)
        else:
            self._on_save_as()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As",
            str(get_default_project_dir() / f"{self.engine.state.name}.k2proj"),
            "K2 Projects (*.k2proj);;JSON Files (*.json)")
        if path:
            self._save_to(path)

    def _save_to(self, path):
        if save_project(self.engine.state, path):
            self._current_file = path
            self._update_title()
            self.status_label.setText(f"Saved: {Path(path).name}")
            logger.info(f"Saved: {path}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save project.")

    # ── Import ──

    def _on_import_ork(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import OpenRocket Design",
            str(Path.home() / "OneDrive" / "Desktop"),
            "OpenRocket Files (*.ork);;XML Files (*.xml);;All Files (*)")
        if not path:
            return
        self._on_import_ork_path(path)

    def _on_import_ork_path(self, path):
        try:
            from import_export.ork_importer import import_ork
            assembly = import_ork(path)
            self.design_ws.set_assembly(assembly)
            self._current_file = None
            self._update_title()
            self.tab_widget.setCurrentIndex(0)  # Switch to Design tab
            self.status_label.setText(f"Imported: {Path(path).name}")
            comp_count = sum(1 for _ in assembly.all_components()) - len(assembly.stages)
            self.engine.log_message.emit(
                f"✅ Imported OpenRocket file: {assembly.name} — "
                f"{len(assembly.stages)} stage(s), {comp_count} components")
            logger.info(f"Imported ORK: {path}")
        except Exception as e:
            logger.error(f"ORK import failed: {e}")
            QMessageBox.critical(self, "Import Error",
                f"Failed to import OpenRocket file:\n\n{e}")

    # ── Simulation controls ──

    def _on_run_sim(self):
        self.tab_widget.setCurrentIndex(6)  # Switch to Simulation tab
        self.sim_engine.start()

    def _on_stop_sim(self):
        self.sim_engine.stop()

    def _on_reset(self):
        if self.sim_engine.is_running:
            self.sim_engine.stop()
        self.engine.reset()
        self._set_sim_status("Idle", "#8b949e")
        logger.info("State reset")

    def _on_reset_view(self):
        self.design_ws.reset_camera()

    def _update_title(self):
        name = self.engine.state.name
        file_info = f" — {Path(self._current_file).name}" if self._current_file else ""
        self.setWindowTitle(f"K2 Aerospace — {name}{file_info}")

    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Quit K2 Aerospace",
            "Are you sure you want to quit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            if self.sim_engine.is_running:
                self.sim_engine.stop()
            self.mission_viz_ws.shutdown()
            self.design_ws.closeEvent(event)
            event.accept()
        else:
            event.ignore()
