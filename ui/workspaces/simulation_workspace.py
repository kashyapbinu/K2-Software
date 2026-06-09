"""
K2 Aerospace — Simulation Workspace
Mission control: Run/Pause/Stop, environment, recovery config, live readouts.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLabel, QPushButton, QDoubleSpinBox, QComboBox,
    QFrame, QGridLayout, QProgressBar, QScrollArea)
from PyQt6.QtCore import Qt
from core.flight_phases import FlightPhase, PHASE_COLORS

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

        self.btn_run = QPushButton("▶  RUN")
        self.btn_run.setProperty("primary", True)
        self.btn_run.setMinimumHeight(44)
        self.btn_run.clicked.connect(self._on_run)
        cl.addWidget(self.btn_run)

        self.btn_pause = QPushButton("⏸  PAUSE")
        self.btn_pause.setMinimumHeight(44)
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        cl.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹  STOP")
        self.btn_stop.setProperty("danger", True)
        self.btn_stop.setMinimumHeight(44)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        cl.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("🔄  RESET")
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

        self.wind_spin = QDoubleSpinBox()
        self.wind_spin.setRange(0, 50); self.wind_spin.setValue(0); self.wind_spin.setSuffix(" m/s")
        self.wind_spin.valueChanged.connect(lambda v: self.engine.update(wind_speed=v))
        ef.addRow("Wind Speed:", self.wind_spin)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 330); self.temp_spin.setValue(288.15); self.temp_spin.setSuffix(" K")
        self.temp_spin.setDecimals(1)
        ef.addRow("Temperature:", self.temp_spin)

        env_group.setLayout(ef)
        left_col.addWidget(env_group)

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
        else:
            self.sim_engine.start()

    def _on_pause(self):
        self.sim_engine.pause()
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶  RESUME")

    def _on_stop(self):
        self.sim_engine.stop()

    def _on_reset(self):
        self.sim_engine.stop() if self.sim_engine.is_running else None
        self.engine.reset()
        self._reset_ui()

    def _on_speed_changed(self, text):
        speed = float(text.replace("x", ""))
        self.sim_engine.set_speed(speed)

    def _on_integrator_changed(self, text):
        name = text.lower()
        self.engine.update(integrator_name=name)

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
