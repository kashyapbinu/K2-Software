"""
K2 AeroSim — Simulation Workspace
Mission control: Run/Pause/Stop, environment, recovery config, live readouts.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLabel, QPushButton, QDoubleSpinBox, QComboBox,
    QFrame, QGridLayout, QProgressBar, QScrollArea, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView)
from PyQt6.QtCore import Qt
from core.flight_phases import FlightPhase, PHASE_COLORS
from ui.icons import icon

logger = logging.getLogger("K2.SimWS")


class BigReadout(QLabel):
    def __init__(self, t="—", parent=None):
        super().__init__(t, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; "
            "font-size: 22px; font-weight: 700; padding: 8px; "
            "background-color: #161b22; border: 1px solid #21262d; border-radius: 8px;")


class PhaseLight(QLabel):
    """Status light for a flight phase."""
    def __init__(self, phase: FlightPhase, parent=None):
        super().__init__(phase.value, parent)
        self.phase = phase
        self._color = PHASE_COLORS.get(phase, "#484f58")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_active(False)

    def set_active(self, active: bool):
        if active:
            self.setStyleSheet(
                f"color: {self._color}; font-weight: 700; font-size: 11px; "
                f"padding: 4px 8px; background-color: #0d1117; "
                f"border: 2px solid {self._color}; border-radius: 6px;"
            )
        else:
            self.setStyleSheet(
                "color: #484f58; font-weight: 600; font-size: 11px; "
                "padding: 4px 8px; background-color: #161b22; "
                "border: 1px solid #21262d; border-radius: 6px;"
            )


class SimulationWorkspace(QWidget):
    def __init__(self, engine, sim_engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._setup_ui()
        self.engine.telemetry_tick.connect(self._on_tick)
        self.engine.state_changed.connect(self._on_state_changed)
        self.sim_engine.sim_started.connect(self._on_sim_started)
        self.sim_engine.sim_finished.connect(self._on_sim_finished)

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Title
        title = QLabel("MISSION CONTROL")
        title.setStyleSheet("color: #58a6ff; font-size: 18px; font-weight: 700; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Controls row
        ctrl_group = QGroupBox("Simulation Controls")
        cl = QHBoxLayout(); cl.setSpacing(12)

        self.btn_run = QPushButton(icon("run", color="#3fb950"), "RUN")
        self.btn_run.setProperty("primary", True)
        self.btn_run.setMinimumHeight(44)
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.btn_pause = QPushButton(icon("pause"), "PAUSE")
        self.btn_pause.setMinimumHeight(44)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        cl.addWidget(self.btn_pause)

        self.btn_stop = QPushButton(icon("stop", color="#f85149"), "STOP")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.setMinimumHeight(44)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        cl.addWidget(self.btn_stop)

        self.btn_reset = QPushButton(icon("reset"), "RESET")
        self.btn_reset.setMinimumHeight(44)
        self.btn_reset.clicked.connect(self._on_reset)
        cl.addWidget(self.btn_reset)

        ctrl_group.setLayout(cl)
        layout.addWidget(ctrl_group)

        # Middle: settings + readouts
        mid = QHBoxLayout()
        mid.setSpacing(14)

        # Left column: Environment + Recovery + Integrator
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # Environment settings
        env_group = QGroupBox("Environment")
        ef = QFormLayout(); ef.setSpacing(6)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 90); self.angle_spin.setValue(90); self.angle_spin.setSuffix("°")
        self.angle_spin.valueChanged.connect(lambda v: self.engine.update(launch_angle=v))
        ef.addRow("Launch Angle:", self.angle_spin)

        self.rod_spin = QDoubleSpinBox()
        self.rod_spin.setRange(0.3, 15.0); self.rod_spin.setValue(1.0)
        self.rod_spin.setDecimals(2); self.rod_spin.setSingleStep(0.1)
        self.rod_spin.setSuffix(" m")
        self.rod_spin.setToolTip("Launch rod/rail guided length — attitude is "
                                 "held until the rocket clears it")
        self.rod_spin.valueChanged.connect(
            lambda v: self.engine.update(launch_rod_length=v))
        ef.addRow("Launch Rod:", self.rod_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 330); self.temp_spin.setValue(288.15); self.temp_spin.setSuffix(" K")
        self.temp_spin.setDecimals(1)
        self.temp_spin.valueChanged.connect(
            lambda v: self.engine.update(ground_temperature=v))
        ef.addRow("Temperature:", self.temp_spin)

        env_group.setLayout(ef)
        left_col.addWidget(env_group)

        # Wind settings (Average / Multi-Level)
        wind_group = QGroupBox("Wind")
        wl = QVBoxLayout(); wl.setSpacing(6)

        self.wind_mode_combo = QComboBox()
        self.wind_mode_combo.addItems(["Average Wind", "Multi-Level Wind"])
        self.wind_mode_combo.currentIndexChanged.connect(self._on_wind_mode_changed)
        wl.addWidget(self.wind_mode_combo)

        self.wind_stack = QStackedWidget()

        # ── Page 0: Average wind ──
        avg_page = QWidget()
        af = QFormLayout(avg_page); af.setSpacing(6)
        af.setContentsMargins(0, 4, 0, 0)

        self.wind_spin = QDoubleSpinBox()
        self.wind_spin.setRange(0, 50); self.wind_spin.setValue(0); self.wind_spin.setSuffix(" m/s")
        self.wind_spin.setDecimals(1)
        self.wind_spin.valueChanged.connect(self._on_avg_wind_changed)
        af.addRow("Avg Speed:", self.wind_spin)

        self.wind_stddev_spin = QDoubleSpinBox()
        self.wind_stddev_spin.setRange(0, 25); self.wind_stddev_spin.setValue(0)
        self.wind_stddev_spin.setSuffix(" m/s"); self.wind_stddev_spin.setDecimals(2)
        self.wind_stddev_spin.valueChanged.connect(self._on_wind_stddev_changed)
        af.addRow("Std Deviation:", self.wind_stddev_spin)

        self.wind_turb_spin = QDoubleSpinBox()
        self.wind_turb_spin.setRange(0, 100); self.wind_turb_spin.setValue(0)
        self.wind_turb_spin.setSuffix(" %"); self.wind_turb_spin.setDecimals(1)
        self.wind_turb_spin.valueChanged.connect(self._on_wind_turb_changed)
        af.addRow("Turbulence:", self.wind_turb_spin)

        self.wind_dir_spin = QDoubleSpinBox()
        self.wind_dir_spin.setRange(0, 360); self.wind_dir_spin.setValue(0)
        self.wind_dir_spin.setSuffix("°"); self.wind_dir_spin.setDecimals(0)
        self.wind_dir_spin.setWrapping(True)
        self.wind_dir_spin.setToolTip("Bearing the wind blows FROM (0=N, 90=E)")
        self.wind_dir_spin.valueChanged.connect(
            lambda v: self.engine.update(wind_direction=v))
        af.addRow("Direction:", self.wind_dir_spin)

        self.wind_stack.addWidget(avg_page)

        # ── Page 1: Multi-level wind ──
        ml_page = QWidget()
        ml = QVBoxLayout(ml_page); ml.setSpacing(6)
        ml.setContentsMargins(0, 4, 0, 0)

        self.wind_table = QTableWidget(0, 3)
        self.wind_table.setHorizontalHeaderLabels(["Alt (m)", "Speed (m/s)", "Dir (°)"])
        self.wind_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.wind_table.verticalHeader().setVisible(False)
        self.wind_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.wind_table.setMinimumHeight(120)
        self.wind_table.setMaximumHeight(180)
        self.wind_table.cellChanged.connect(self._on_wind_table_changed)
        ml.addWidget(self.wind_table)

        ml_btns = QHBoxLayout()
        btn_add_layer = QPushButton(icon("add"), "Add Layer")
        btn_add_layer.clicked.connect(self._on_add_wind_layer)
        ml_btns.addWidget(btn_add_layer)
        btn_del_layer = QPushButton(icon("delete", color="#f85149"), "Delete")
        btn_del_layer.clicked.connect(self._on_delete_wind_layer)
        ml_btns.addWidget(btn_del_layer)
        ml.addLayout(ml_btns)

        ml_turb = QFormLayout(); ml_turb.setSpacing(6)
        self.wind_ml_turb_spin = QDoubleSpinBox()
        self.wind_ml_turb_spin.setRange(0, 100); self.wind_ml_turb_spin.setValue(0)
        self.wind_ml_turb_spin.setSuffix(" %"); self.wind_ml_turb_spin.setDecimals(1)
        self.wind_ml_turb_spin.valueChanged.connect(
            lambda v: self.engine.update(wind_gust_intensity=v / 100.0))
        ml_turb.addRow("Turbulence:", self.wind_ml_turb_spin)
        ml.addLayout(ml_turb)

        self.wind_stack.addWidget(ml_page)
        wl.addWidget(self.wind_stack)
        wind_group.setLayout(wl)
        left_col.addWidget(wind_group)

        # Simulation settings
        sim_group = QGroupBox("Simulation Settings")
        sf = QFormLayout(); sf.setSpacing(6)

        self.speed_combo = QComboBox()
        for s in ["0.25x", "0.5x", "1x", "2x", "5x", "10x"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        sf.addRow("Sim Speed:", self.speed_combo)

        self.integrator_combo = QComboBox()
        self.integrator_combo.addItems(["RK4", "Euler"])
        self.integrator_combo.setCurrentText("RK4")
        self.integrator_combo.currentTextChanged.connect(self._on_integrator_changed)
        sf.addRow("Integrator:", self.integrator_combo)

        sim_group.setLayout(sf)
        left_col.addWidget(sim_group)

        # Recovery configuration
        rec_group = QGroupBox("Recovery System")
        rf = QFormLayout(); rf.setSpacing(6)

        self.drogue_delay_spin = QDoubleSpinBox()
        self.drogue_delay_spin.setRange(0, 30); self.drogue_delay_spin.setValue(1.0)
        self.drogue_delay_spin.setSuffix(" s"); self.drogue_delay_spin.setDecimals(1)
        self.drogue_delay_spin.valueChanged.connect(
            lambda v: self.engine.update(drogue_deploy_delay=v))
        rf.addRow("Drogue Delay:", self.drogue_delay_spin)

        self.main_alt_spin = QDoubleSpinBox()
        self.main_alt_spin.setRange(50, 5000); self.main_alt_spin.setValue(300)
        self.main_alt_spin.setSuffix(" m"); self.main_alt_spin.setDecimals(0)
        self.main_alt_spin.valueChanged.connect(
            lambda v: self.engine.update(main_deploy_altitude=v))
        rf.addRow("Main Deploy Alt:", self.main_alt_spin)

        self.drogue_cda_spin = QDoubleSpinBox()
        self.drogue_cda_spin.setRange(0.01, 10); self.drogue_cda_spin.setValue(0.5)
        self.drogue_cda_spin.setSuffix(" m²"); self.drogue_cda_spin.setDecimals(2)
        self.drogue_cda_spin.valueChanged.connect(
            lambda v: self.engine.update(drogue_cd_area=v))
        rf.addRow("Drogue Cd×A:", self.drogue_cda_spin)

        self.main_cda_spin = QDoubleSpinBox()
        self.main_cda_spin.setRange(0.1, 50); self.main_cda_spin.setValue(3.0)
        self.main_cda_spin.setSuffix(" m²"); self.main_cda_spin.setDecimals(1)
        self.main_cda_spin.valueChanged.connect(
            lambda v: self.engine.update(main_cd_area=v))
        rf.addRow("Main Cd×A:", self.main_cda_spin)

        rec_group.setLayout(rf)
        left_col.addWidget(rec_group)

        mid.addLayout(left_col, 1)

        # Right column: readouts
        ro_group = QGroupBox("Live Flight Data")
        rg = QGridLayout(); rg.setSpacing(8)

        readout_defs = [
            ("Altitude", "m"), ("Velocity", "m/s"), ("Acceleration", "m/s²"),
            ("Thrust", "N"), ("Drag", "N"), ("Mach", ""),
            ("Mass", "kg"), ("Dyn. Pressure", "Pa"),
        ]
        self.readouts = {}
        for i, (name, unit) in enumerate(readout_defs):
            row, col = divmod(i, 4)
            header = QLabel(f"{name}")
            header.setStyleSheet("color: #58a6ff; font-weight: 600; font-size: 11px;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            rg.addWidget(header, row * 2, col)
            val = BigReadout("—")
            self.readouts[name] = val
            rg.addWidget(val, row * 2 + 1, col)

        ro_group.setLayout(rg)
        mid.addWidget(ro_group, 2)

        layout.addLayout(mid)

        # Phase status lights (all 8 phases)
        phase_group = QGroupBox("Flight Phase Timeline")
        pl = QHBoxLayout(); pl.setSpacing(6)

        self.phase_lights = {}
        active_phases = [
            FlightPhase.PRELAUNCH, FlightPhase.IGNITION, FlightPhase.BOOST,
            FlightPhase.COAST, FlightPhase.APOGEE, FlightPhase.DROGUE_DESCENT,
            FlightPhase.MAIN_DESCENT, FlightPhase.LANDED,
        ]
        for phase in active_phases:
            light = PhaseLight(phase)
            self.phase_lights[phase] = light
            pl.addWidget(light)

        phase_group.setLayout(pl)
        layout.addWidget(phase_group)

        # Timeline / progress
        time_group = QGroupBox("Mission Timeline")
        tl = QVBoxLayout()
        self.time_label = QLabel("T+ 0.00 s")
        self.time_label.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; "
            "font-size: 28px; font-weight: 700;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tl.addWidget(self.time_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        tl.addWidget(self.progress)

        self.phase_label = QLabel("Pre-Launch")
        self.phase_label.setStyleSheet("color: #8b949e; font-size: 14px; font-weight: 600;")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tl.addWidget(self.phase_label)

        time_group.setLayout(tl)
        layout.addWidget(time_group)

        layout.addStretch()

        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Actions ──

    def _on_run(self):
        if self.sim_engine.is_paused:
            self.sim_engine.resume()
            return

        # Ask which view should host the flight before starting
        from PyQt6.QtWidgets import QMessageBox
        from PyQt6.QtCore import QTimer
        box = QMessageBox(self)
        box.setWindowTitle("Launch View")
        box.setText("Where do you want to watch this flight?")
        box.setInformativeText(
            "Normal: stay here with the live readouts.\n"
            "Advanced Visualizer: switch to the 3D view, then launch.")
        normal_btn = box.addButton("Normal", QMessageBox.ButtonRole.AcceptRole)
        cine_btn = box.addButton("Advanced Visualizer", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cine_btn)
        box.exec()
        clicked = box.clickedButton()

        if clicked is normal_btn:
            self.sim_engine.start()
        elif clicked is cine_btn:
            main = self.window()
            cine = getattr(main, "cinematic_ws", None)
            tabs = getattr(main, "tab_widget", None)
            if cine is not None and tabs is not None:
                tabs.setCurrentWidget(cine)
                # Let the WebGL view come to the front before igniting
                QTimer.singleShot(350, self.sim_engine.start)
            else:
                self.sim_engine.start()
        # Cancel → do nothing

    def _on_pause(self):
        self.sim_engine.pause()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  RESUME")

    def _on_stop(self):
        self.sim_engine.stop()

    def _on_reset(self):
        if self.sim_engine.is_running:
            self.sim_engine.stop()
        self.engine.reset()
        # Re-derive geometry from the design assembly — reset() blanks it and
        # a zero-diameter rocket flies dragless (absurd apogees).
        main = self.window()
        design = getattr(main, "design_ws", None)
        if design is not None and getattr(design, "assembly", None) is not None:
            try:
                design._sync_to_engine()
            except Exception:
                pass
        self._reset_ui()

    def _on_speed_changed(self, text):
        speed = float(text.replace("x", ""))
        self.sim_engine.set_speed(speed)

    def _on_integrator_changed(self, text):
        name = text.lower()
        self.engine.update(integrator_name=name)

    # ── Wind handlers ─────────────────────────────────────────────

    def _on_wind_mode_changed(self, index):
        self.wind_stack.setCurrentIndex(index)
        mode = "multi_level" if index == 1 else "average"
        self.engine.update(wind_mode=mode)
        if mode == "multi_level":
            self._push_wind_layers()
            self.engine.update(
                wind_gust_intensity=self.wind_ml_turb_spin.value() / 100.0)
        else:
            self.engine.update(
                wind_speed=self.wind_spin.value(),
                wind_direction=self.wind_dir_spin.value(),
                wind_gust_intensity=self.wind_turb_spin.value() / 100.0)

    def _on_avg_wind_changed(self, speed):
        self.engine.update(wind_speed=speed)
        # Keep σ consistent with TI% at the new mean speed
        self.wind_stddev_spin.blockSignals(True)
        self.wind_stddev_spin.setValue(speed * self.wind_turb_spin.value() / 100.0)
        self.wind_stddev_spin.blockSignals(False)

    def _on_wind_turb_changed(self, pct):
        self.engine.update(wind_gust_intensity=pct / 100.0)
        self.wind_stddev_spin.blockSignals(True)
        self.wind_stddev_spin.setValue(self.wind_spin.value() * pct / 100.0)
        self.wind_stddev_spin.blockSignals(False)

    def _on_wind_stddev_changed(self, sigma):
        # σ and TI% are two views of the same setting: TI = σ / mean
        speed = self.wind_spin.value()
        pct = (sigma / speed * 100.0) if speed > 0 else 0.0
        self.wind_turb_spin.blockSignals(True)
        self.wind_turb_spin.setValue(pct)
        self.wind_turb_spin.blockSignals(False)
        self.engine.update(wind_gust_intensity=pct / 100.0)

    def _on_add_wind_layer(self):
        row = self.wind_table.rowCount()
        # Default the new layer to 500m above the previous one
        prev_alt = 0.0
        if row > 0:
            try:
                prev_alt = float(self.wind_table.item(row - 1, 0).text())
            except (ValueError, AttributeError):
                pass
        self.wind_table.blockSignals(True)
        self.wind_table.insertRow(row)
        for col, val in enumerate([f"{prev_alt + 500.0:g}", "5.0", "0"]):
            self.wind_table.setItem(row, col, QTableWidgetItem(val))
        self.wind_table.blockSignals(False)
        self._push_wind_layers()

    def _on_delete_wind_layer(self):
        rows = sorted({i.row() for i in self.wind_table.selectedIndexes()},
                      reverse=True)
        if not rows and self.wind_table.rowCount() > 0:
            rows = [self.wind_table.rowCount() - 1]
        self.wind_table.blockSignals(True)
        for r in rows:
            self.wind_table.removeRow(r)
        self.wind_table.blockSignals(False)
        self._push_wind_layers()

    def _on_wind_table_changed(self, *_):
        self._push_wind_layers()

    def _push_wind_layers(self):
        """Read the layer table and push it into simulation state."""
        layers = []
        for r in range(self.wind_table.rowCount()):
            try:
                alt = float(self.wind_table.item(r, 0).text())
                spd = float(self.wind_table.item(r, 1).text())
                drn = float(self.wind_table.item(r, 2).text())
                layers.append((alt, max(0.0, spd), drn % 360.0))
            except (ValueError, AttributeError):
                continue  # skip incomplete/non-numeric rows
        self.engine.update(wind_layers=layers)

    def _on_sim_started(self):
        self.btn_run.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.angle_spin.setEnabled(False)
        self.integrator_combo.setEnabled(False)
        # Light up prelaunch
        for light in self.phase_lights.values():
            light.set_active(False)
        self.phase_lights[FlightPhase.PRELAUNCH].set_active(True)

    def _on_sim_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  RUN")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.angle_spin.setEnabled(True)
        self.integrator_combo.setEnabled(True)
        self.progress.setValue(100)

    def reset_workspace(self):
        """Blank the live readouts + phase lights (called on New Project)."""
        if self.sim_engine.is_running:
            try:
                self.sim_engine.stop()
            except Exception:
                pass
        self._reset_ui()

    def _reset_ui(self):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  RUN")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.angle_spin.setEnabled(True)
        self.integrator_combo.setEnabled(True)
        self.progress.setValue(0)
        self.time_label.setText("T+ 0.00 s")
        self.phase_label.setText("Pre-Launch")
        for v in self.readouts.values():
            v.setText("—")
        for light in self.phase_lights.values():
            light.set_active(False)

    def _on_tick(self, state):
        self._update_readouts(state)

    def _on_state_changed(self, state):
        self._update_readouts(state)

    def _update_readouts(self, s):
        self.time_label.setText(f"T+ {s.sim_time:.2f} s")
        self.phase_label.setText(s.sim_phase)

        self.readouts["Altitude"].setText(f"{s.altitude:.1f}")
        self.readouts["Velocity"].setText(f"{s.velocity:.1f}")
        self.readouts["Acceleration"].setText(f"{s.acceleration:.1f}")
        self.readouts["Thrust"].setText(f"{s.thrust:.1f}")
        self.readouts["Drag"].setText(f"{s.drag:.1f}")
        self.readouts["Mach"].setText(f"{s.mach_number:.3f}")
        self.readouts["Mass"].setText(f"{s.total_mass():.3f}")
        self.readouts["Dyn. Pressure"].setText(f"{s.dynamic_pressure:.0f}")

        # Update phase lights
        try:
            current_phase = FlightPhase(s.sim_phase)
        except ValueError:
            current_phase = FlightPhase.PRELAUNCH

        # Light all phases up to and including current
        phase_order = [
            FlightPhase.PRELAUNCH, FlightPhase.IGNITION, FlightPhase.BOOST,
            FlightPhase.COAST, FlightPhase.APOGEE, FlightPhase.DROGUE_DESCENT,
            FlightPhase.MAIN_DESCENT, FlightPhase.LANDED,
        ]
        reached = False
        for phase in reversed(phase_order):
            if phase == current_phase:
                reached = True
            if phase in self.phase_lights:
                self.phase_lights[phase].set_active(reached)

        # Progress
        phase_progress = {
            FlightPhase.PRELAUNCH: 0, FlightPhase.IGNITION: 5,
            FlightPhase.BOOST: 20, FlightPhase.COAST: 40,
            FlightPhase.APOGEE: 55, FlightPhase.DROGUE_DESCENT: 70,
            FlightPhase.MAIN_DESCENT: 85, FlightPhase.LANDED: 100,
        }
        self.progress.setValue(phase_progress.get(current_phase, 0))
