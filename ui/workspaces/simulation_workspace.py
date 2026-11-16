"""
K2 AeroSim — Simulation Workspace
Transport bar, launch setup on the left, live readouts and a live
altitude / velocity trace on the right.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QFormLayout, QLabel, QPushButton, QDoubleSpinBox, QComboBox,
    QFrame, QGridLayout, QProgressBar, QScrollArea, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QMessageBox,
    QCheckBox, QInputDialog)
from PyQt6.QtCore import Qt
from core.flight_phases import FlightPhase, PHASE_COLORS
from core import launch_sites
from ui.icons import icon

from ui import theme
from ui.widgets.readout import Readout
from ui import settings

CUSTOM_SITE = "Custom"

logger = logging.getLogger("K2.SimWS")


class PhaseLight(QLabel):
    """One segment of the flight-phase strip."""
    def __init__(self, phase: FlightPhase, parent=None):
        super().__init__(phase.value, parent)
        self.phase = phase
        self._color = PHASE_COLORS.get(phase, theme.LINE_STRONG)
        self._active = False
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.set_active(False)

    def retheme(self):
        self.set_active(self._active)

    def set_active(self, active: bool):
        self._active = active
        bar = self._color if active else theme.LINE
        ink = theme.TEXT_BRIGHT if active else theme.TEXT_FAINT
        self.setStyleSheet(
            f"color: {ink}; font-size: 11px; font-weight: {600 if active else 400}; "
            f"padding: 6px 0px 2px 0px; background: transparent; "
            f"border: none; border-top: 2px solid {bar};")


class LiveTrace(QWidget):
    """Altitude and velocity against time, drawn while the flight runs.

    Redraws are limited to a few per second and skipped while the tab is
    hidden, so the trace does not slow the integrator's timer.
    """

    MIN_REDRAW_S = 0.2

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.figure = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.wheelEvent = lambda e: e.ignore()
        self.ax = self.figure.add_subplot(111)
        self.ax2 = self.ax.twinx()
        self._alt, = self.ax.plot([], [], linewidth=1.6)
        self._vel, = self.ax2.plot([], [], linewidth=1.2)
        self._last_draw = 0.0
        self._unit_key = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.retheme()
        self.set_data([], [], [])

    def retheme(self):
        self.figure.patch.set_facecolor(theme.BG)
        self._alt.set_color(theme.ACCENT)
        self._vel.set_color(theme.INFO)
        for ax in (self.ax, self.ax2):
            ax.set_facecolor("none")
            ax.tick_params(colors=theme.TEXT_DIM, labelsize=9, length=3)
            for side, spine in ax.spines.items():
                spine.set_visible(side != "top")
                spine.set_color(theme.LINE)
        self.ax.grid(True, color=theme.LINE, linewidth=0.6, alpha=0.7)
        self.ax2.grid(False)
        self.ax.set_xlabel("Time (s)", color=theme.TEXT_DIM, fontsize=9)
        self.ax.yaxis.label.set_color(theme.ACCENT)
        self.ax2.yaxis.label.set_color(theme.INFO)
        self.canvas.draw_idle()

    def set_data(self, t, altitude, velocity, force=True):
        """Replace the traces. Values are SI; shown in the display units."""
        import time as _time
        now = _time.monotonic()
        if not force and (now - self._last_draw < self.MIN_REDRAW_S
                          or not self.isVisible()):
            return
        self._last_draw = now
        from ui import units
        alt_u, vel_u = units.unit_label("altitude"), units.unit_label("velocity")
        k_alt, k_vel = units.from_si(1.0, "altitude"), units.from_si(1.0, "velocity")
        n = min(len(t), len(altitude), len(velocity))
        t = list(t[:n])
        self._alt.set_data(t, [a * k_alt for a in altitude[:n]])
        self._vel.set_data(t, [v * k_vel for v in velocity[:n]])
        self.ax.set_ylabel(f"Altitude ({alt_u})", fontsize=9)
        self.ax2.set_ylabel(f"Velocity ({vel_u})", fontsize=9)
        if n >= 2:
            for ax in (self.ax, self.ax2):
                # set_ylim on the empty plot switches autoscaling off
                ax.set_autoscaley_on(True)
                ax.relim()
                ax.autoscale_view(scalex=False)
            self.ax.set_xlim(0, max(t[-1], 1.0))
        else:
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 1000 * k_alt)
            self.ax2.set_ylim(0, 100 * k_vel)
        try:
            self.figure.tight_layout(pad=1.2)
        except Exception:
            pass
        self.canvas.draw_idle()


class _FitStack(QStackedWidget):
    """A stacked widget as tall as the page it shows.

    QStackedWidget is as tall as its tallest page, which left a blank band
    under the four average-wind fields the height of the wind-layer table.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(lambda _i: self.updateGeometry())

    def sizeHint(self):
        w = self.currentWidget()
        return w.sizeHint() if w is not None else super().sizeHint()

    def minimumSizeHint(self):
        w = self.currentWidget()
        return w.minimumSizeHint() if w is not None else super().minimumSizeHint()


class SimulationWorkspace(QWidget):
    def __init__(self, engine, sim_engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._setup_ui()
        self._sync_inputs_from_state(self.engine.state)
        self.engine.telemetry_tick.connect(self._on_tick)
        self.engine.state_changed.connect(self._on_state_changed)
        self.sim_engine.sim_started.connect(self._on_sim_started)
        self.sim_engine.sim_finished.connect(self._on_sim_finished)
        self.sim_engine.sim_failed.connect(self._on_sim_failed)

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Transport bar: flight controls, clock, phase, speed ──
        # Run / Pause / Stop / Reset were four 44 px-high buttons stretched
        # across the whole tab (about 470 px each) inside their own group box.
        bar = QFrame()
        bar.setObjectName("transportBar")
        bar.setStyleSheet(
            f"QFrame#transportBar {{ background-color: {theme.PANEL}; "
            f"border-bottom: 1px solid {theme.LINE}; }}")
        cl = QHBoxLayout(bar)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(6)

        def _transport(key, text, color=None, role=None):
            b = QPushButton(icon(key, color=color) if color else icon(key), text)
            if role:
                b.setProperty(role, True)
            b.setFixedHeight(30)
            b.setMinimumWidth(84)
            cl.addWidget(b)
            return b

        self.btn_run = _transport("run", "Run", theme.ACCENT, "primary")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_pause = _transport("pause", "Pause")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause)
        self.btn_stop = _transport("stop", "Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_reset = _transport("reset", "Reset")
        self.btn_reset.clicked.connect(self._on_reset)

        cl.addSpacing(18)
        self.time_label = QLabel("T+ 0.00 s")
        self.time_label.setStyleSheet(
            f"color: {theme.TEXT_BRIGHT}; font-family: {theme.MONO}; "
            "font-size: 18px; font-weight: 600;")
        self.time_label.setMinimumWidth(130)
        cl.addWidget(self.time_label)
        self.phase_label = QLabel("Pre-Launch")
        self.phase_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 13px;")
        cl.addWidget(self.phase_label)
        cl.addStretch(1)

        speed_caption = QLabel("Speed")
        speed_caption.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        cl.addWidget(speed_caption)
        self.speed_combo = QComboBox()
        for s in ["0.25x", "0.5x", "1x", "2x", "5x", "10x"]:
            self.speed_combo.addItem(s)
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.setFixedHeight(30)
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        cl.addWidget(self.speed_combo)
        cl.addSpacing(8)
        self.btn_apogee_tools = QPushButton(icon("target"), "Target Apogee…")
        self.btn_apogee_tools.setFixedHeight(30)
        self.btn_apogee_tools.setToolTip(
            "Ballast or airbrake area to hit a target apogee, and how the "
            "apogee moves with launch-day unknowns.")
        self.btn_apogee_tools.clicked.connect(self.open_apogee_tools)
        cl.addWidget(self.btn_apogee_tools)
        outer.addWidget(bar)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(2)
        self.progress.setStyleSheet(
            "QProgressBar { border: none; border-radius: 0px; background: transparent; }"
            f"QProgressBar::chunk {{ border-radius: 0px; background-color: {theme.ACCENT}; }}")
        outer.addWidget(self.progress)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, 1)

        # ── Left: launch setup, a fixed-width scrolling column ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(348)
        inner = QWidget()
        left_col = QVBoxLayout(inner)
        left_col.setContentsMargins(14, 2, 14, 14)
        left_col.setSpacing(0)

        # Launch site
        left_col.addWidget(self._build_site_group())

        # Environment settings
        env_group = QGroupBox("Environment")
        ef = QFormLayout(); ef.setSpacing(6)

        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0, 90); self.angle_spin.setValue(90); self.angle_spin.setSuffix("°")
        self.angle_spin.valueChanged.connect(lambda v: self.engine.update(launch_angle=v))
        ef.addRow("Launch Angle:", self.angle_spin)

        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(0, 360); self.azimuth_spin.setValue(0)
        self.azimuth_spin.setSuffix("°"); self.azimuth_spin.setDecimals(0)
        self.azimuth_spin.setWrapping(True)
        self.azimuth_spin.setToolTip(
            "Compass bearing the rail leans toward (0 = N, 90 = E). Matters "
            "only when the launch angle is below 90°.")
        self.azimuth_spin.valueChanged.connect(
            lambda v: self.engine.update(launch_azimuth=v % 360.0))
        ef.addRow("Rail Azimuth:", self.azimuth_spin)

        self.rod_spin = QDoubleSpinBox()
        self.rod_spin.setRange(0.3, 15.0); self.rod_spin.setValue(1.0)
        self.rod_spin.setDecimals(2); self.rod_spin.setSingleStep(0.1)
        self.rod_spin.setSuffix(" m")
        self.rod_spin.setToolTip("Launch rod/rail guided length — attitude is "
                                 "held until the rocket clears it")
        self.rod_spin.valueChanged.connect(
            lambda v: self.engine.update(launch_rod_length=v))
        ef.addRow("Launch Rod:", self.rod_spin)

        self.isa_check = QCheckBox("Standard day (ISA)")
        self.isa_check.setChecked(True)
        self.isa_check.setToolTip(
            "Use the standard-atmosphere temperature at the site's elevation. "
            "Untick to enter the air temperature measured at the pad.")
        self.isa_check.toggled.connect(self._on_isa_toggled)
        ef.addRow("Air:", self.isa_check)

        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(200, 330); self.temp_spin.setValue(288.15); self.temp_spin.setSuffix(" K")
        self.temp_spin.setDecimals(1)
        self.temp_spin.setEnabled(False)
        self.temp_spin.setToolTip("Air temperature at the pad.")
        self.temp_spin.valueChanged.connect(
            lambda v: self.engine.update(ground_temperature=v))
        ef.addRow("Pad Temperature:", self.temp_spin)

        env_group.setLayout(ef)
        left_col.addWidget(env_group)

        # Wind settings (Average / Multi-Level)
        wind_group = QGroupBox("Wind")
        wl = QVBoxLayout(); wl.setSpacing(6)

        self.wind_mode_combo = QComboBox()
        self.wind_mode_combo.addItems(["Average Wind", "Multi-Level Wind"])
        self.wind_mode_combo.currentIndexChanged.connect(self._on_wind_mode_changed)
        wl.addWidget(self.wind_mode_combo)

        self.wind_stack = _FitStack()

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
        btn_del_layer = QPushButton(icon("delete", color=theme.ERR), "Delete")
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

        snd_row = QHBoxLayout()
        self.btn_sounding = QPushButton(icon("import"), "Import Sounding…")
        self.btn_sounding.setToolTip(
            "Load a measured atmosphere: University of Wyoming 'Text: List' "
            "or CSV (height_m, pressure_hPa, temperature_C, wind_speed_kt, "
            "wind_dir_deg). Sets the air temperature/pressure profile and, "
            "when the file has winds, the multi-level wind.")
        self.btn_sounding.clicked.connect(self._on_import_sounding)
        snd_row.addWidget(self.btn_sounding)
        self.btn_sounding_clear = QPushButton("Clear")
        self.btn_sounding_clear.setToolTip("Go back to the standard atmosphere")
        self.btn_sounding_clear.clicked.connect(self._on_clear_sounding)
        snd_row.addWidget(self.btn_sounding_clear)
        wl.addLayout(snd_row)
        self.sounding_label = QLabel("")
        self.sounding_label.setWordWrap(True)
        self.sounding_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        wl.addWidget(self.sounding_label)
        wind_group.setLayout(wl)
        left_col.addWidget(wind_group)

        # Simulation settings
        sim_group = QGroupBox("Solver")
        sf = QFormLayout(); sf.setSpacing(6)

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
        self.drogue_cda_spin.setRange(0, 50); self.drogue_cda_spin.setValue(0.5)
        self.drogue_cda_spin.setSuffix(" m²"); self.drogue_cda_spin.setDecimals(2)
        self.drogue_cda_spin.setToolTip(
            "Drag area of the canopy that opens at apogee. Taken from the "
            "Design tab's parachutes when the design has any (a single "
            "parachute goes here). 0 = nothing opens at apogee.")
        self.drogue_cda_spin.valueChanged.connect(
            lambda v: self.engine.update(drogue_cd_area=v))
        rf.addRow("Drogue Cd×A:", self.drogue_cda_spin)

        self.main_cda_spin = QDoubleSpinBox()
        self.main_cda_spin.setRange(0, 50); self.main_cda_spin.setValue(3.0)
        self.main_cda_spin.setSuffix(" m²"); self.main_cda_spin.setDecimals(2)
        self.main_cda_spin.setToolTip(
            "Drag area added at the main-deploy altitude. Taken from the "
            "Design tab when it has two or more parachutes (the largest). "
            "0 = no second canopy.")
        self.main_cda_spin.valueChanged.connect(
            lambda v: self.engine.update(main_cd_area=v))
        rf.addRow("Main Cd×A:", self.main_cda_spin)

        self.inflation_spin = QDoubleSpinBox()
        self.inflation_spin.setRange(0.05, 5.0); self.inflation_spin.setValue(0.6)
        self.inflation_spin.setSuffix(" s"); self.inflation_spin.setDecimals(2)
        self.inflation_spin.setToolTip(
            "Canopy fill time. Main canopy is scaled 2×. Drives the opening-"
            "shock transient — shorter fill = sharper snatch load.")
        self.inflation_spin.valueChanged.connect(
            lambda v: self.engine.update(recovery_inflation_time=v))
        rf.addRow("Inflation Time:", self.inflation_spin)

        self.recovery_note = QLabel("")
        self.recovery_note.setWordWrap(True)
        self.recovery_note.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        self.recovery_note.setVisible(False)
        rf.addRow(self.recovery_note)

        rec_group.setLayout(rf)
        left_col.addWidget(rec_group)

        left_col.addStretch(1)
        scroll.setWidget(inner)
        body.addWidget(scroll)

        divider = QFrame()
        divider.setFixedWidth(1)
        divider.setStyleSheet(f"background-color: {theme.LINE};")
        body.addWidget(divider)

        # ── Right: live data ──
        right = QVBoxLayout()
        right.setContentsMargins(16, 14, 16, 12)
        right.setSpacing(12)

        rg = QGridLayout()
        rg.setSpacing(8)
        readout_names = ["Altitude", "Velocity", "Acceleration", "Mach",
                         "Thrust", "Drag", "Mass", "Dyn. Pressure"]
        self.readouts = {}
        for i, name in enumerate(readout_names):
            row, col = divmod(i, 4)
            val = Readout(name)
            self.readouts[name] = val
            rg.addWidget(val, row, col)
            rg.setColumnStretch(col, 1)
        right.addLayout(rg)

        # Flight phases, one segment each, lit up to the current one
        pl = QHBoxLayout(); pl.setSpacing(4)
        self.phase_lights = {}
        active_phases = [
            FlightPhase.PRELAUNCH, FlightPhase.IGNITION, FlightPhase.BOOST,
            FlightPhase.COAST, FlightPhase.APOGEE, FlightPhase.DROGUE_DESCENT,
            FlightPhase.MAIN_DESCENT, FlightPhase.LANDED,
        ]
        for phase in active_phases:
            light = PhaseLight(phase)
            self.phase_lights[phase] = light
            pl.addWidget(light, 1)
        right.addLayout(pl)

        self.trace = LiveTrace()
        right.addWidget(self.trace, 1)
        body.addLayout(right, 1)

    # ── Launch site ───────────────────────────────────────────────

    def _build_site_group(self):
        group = QGroupBox("Launch Site")
        form = QFormLayout(); form.setSpacing(6)

        row = QHBoxLayout()
        self.site_combo = QComboBox()
        self.site_combo.setToolTip("Saved launch sites. Editing a value below makes it 'Custom'.")
        self.site_combo.activated.connect(self._on_site_chosen)
        row.addWidget(self.site_combo, 1)
        self.btn_site_save = QPushButton(icon("save"), "")
        self.btn_site_save.setToolTip("Save these values as a named site")
        self.btn_site_save.clicked.connect(self._on_site_save)
        row.addWidget(self.btn_site_save)
        self.btn_site_delete = QPushButton(icon("delete", color=theme.ERR), "")
        self.btn_site_delete.setToolTip("Delete the selected saved site")
        self.btn_site_delete.clicked.connect(self._on_site_delete)
        row.addWidget(self.btn_site_delete)
        form.addRow("Site:", row)

        self.site_note = QLabel("")
        self.site_note.setWordWrap(True)
        self.site_note.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        form.addRow(self.site_note)

        def _spin(lo, hi, decimals, suffix, field):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi); sp.setDecimals(decimals); sp.setSuffix(suffix)
            sp.valueChanged.connect(
                lambda v, f=field: self.engine.update(**{f: v, "launch_site_name": ""}))
            return sp

        self.site_alt_spin = _spin(-500.0, 9000.0, 0, " m", "launch_altitude")
        self.site_alt_spin.setToolTip("Pad elevation above sea level. The flight's "
                                      "altitudes are above the pad.")
        form.addRow("Elevation:", self.site_alt_spin)
        self.site_lat_spin = _spin(-90.0, 90.0, 5, "°", "launch_latitude")
        self.site_lat_spin.setToolTip("Latitude, + north")
        form.addRow("Latitude:", self.site_lat_spin)
        self.site_lon_spin = _spin(-180.0, 180.0, 5, "°", "launch_longitude")
        self.site_lon_spin.setToolTip("Longitude, + east")
        form.addRow("Longitude:", self.site_lon_spin)

        group.setLayout(form)
        self._reload_sites()
        return group

    def _sites(self):
        return launch_sites.all_sites(settings.get("sim/launch_sites") or "")

    def _reload_sites(self, select: str = None):
        self._site_list = self._sites()
        self.site_combo.blockSignals(True)
        self.site_combo.clear()
        for site in self._site_list:
            self.site_combo.addItem(site.name)
        self.site_combo.addItem(CUSTOM_SITE)
        self.site_combo.blockSignals(False)
        self._select_site(select if select is not None
                          else self.engine.state.launch_site_name)

    def _select_site(self, name: str):
        idx = self.site_combo.findText(name) if name else -1
        if idx < 0:
            idx = self.site_combo.findText(CUSTOM_SITE)
        self.site_combo.blockSignals(True)
        self.site_combo.setCurrentIndex(idx)
        self.site_combo.blockSignals(False)
        site = self._site_by_name(self.site_combo.currentText())
        self.site_note.setText(site.note if site else "")
        self.site_note.setVisible(bool(site and site.note))
        self.btn_site_delete.setEnabled(bool(site and not site.builtin))

    def _site_by_name(self, name: str):
        return next((s for s in getattr(self, "_site_list", []) if s.name == name), None)

    def _on_site_chosen(self, index: int):
        site = self._site_by_name(self.site_combo.itemText(index))
        if site is None:            # "Custom": keep the values as they are
            self.engine.update(launch_site_name="")
        else:
            self.engine.update(**site.state_fields())
        self._select_site(site.name if site else CUSTOM_SITE)

    def _on_site_save(self):
        s = self.engine.state
        name, ok = QInputDialog.getText(self, "Save Launch Site", "Site name:",
                                        text=s.launch_site_name or "")
        if not ok or not name.strip():
            return
        site = launch_sites.LaunchSite(
            name=name.strip(), altitude=s.launch_altitude,
            latitude=s.launch_latitude, longitude=s.launch_longitude,
            note="Saved by you.")
        problems = site.problems()
        if problems:
            QMessageBox.warning(self, "Save Launch Site", "\n".join(problems))
            return
        settings.set("sim/launch_sites", launch_sites.with_site(
            settings.get("sim/launch_sites") or "", site))
        self.engine.update(launch_site_name=site.name)
        self._reload_sites(select=site.name)

    def _on_site_delete(self):
        site = self._site_by_name(self.site_combo.currentText())
        if site is None or site.builtin:
            return
        settings.set("sim/launch_sites", launch_sites.without_site(
            settings.get("sim/launch_sites") or "", site.name))
        self.engine.update(launch_site_name="")
        self._reload_sites(select=CUSTOM_SITE)

    def open_apogee_tools(self):
        from ui.dialogs.apogee_tools_dialog import ApogeeToolsDialog
        ApogeeToolsDialog(self.engine, self).exec()

    # ── Weather sounding ──────────────────────────────────────────

    def _on_import_sounding(self, path: str = None):
        from PyQt6.QtWidgets import QFileDialog
        from environment.sounding import SoundingError, read_sounding_file
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import Sounding", "",
                "Soundings (*.txt *.csv *.tsv *.dat);;All Files (*)")
            if not path:
                return
        try:
            snd = read_sounding_file(path)
        except (OSError, SoundingError) as exc:
            QMessageBox.warning(self, "Import Sounding", f"Could not read the sounding:\n\n{exc}")
            return
        s = self.engine.state
        site = s.launch_altitude
        update = {"atmosphere_profile": snd.atmosphere_profile(),
                  "atmosphere_source": snd.source}
        layers = snd.wind_layers(site)
        if layers:
            update.update(wind_mode="multi_level", wind_layers=layers)
        self.engine.update(**update)

        lo, hi = snd.levels[0].altitude, snd.levels[-1].altitude
        p, t = snd.surface(site)
        notes = []
        if site < lo:
            notes.append(f"The pad ({site:.0f} m) is below the lowest level "
                         f"({lo:.0f} m); the air there is extrapolated.")
        if hi < site + 3000.0:
            notes.append(f"The sounding stops at {hi:.0f} m; standard-atmosphere "
                         f"shape is used above it.")
        if not layers:
            notes.append("No wind in the file: the wind setting is unchanged.")
        self.engine.log_message.emit(
            f"Sounding '{snd.source}': {len(snd.levels)} levels, "
            f"{lo:.0f}–{hi:.0f} m; pad {t - 273.15:.1f} °C, {p / 100:.1f} hPa"
            + (f"; {len(layers)} wind layers" if layers else ""))
        for n in notes:
            self.engine.log_message.emit("Sounding: " + n)

    def _on_clear_sounding(self):
        self.engine.update(atmosphere_profile=[], atmosphere_source="")

    def _show_sounding(self, s):
        """While a sounding is loaded it is the air (launch_site_atmosphere
        ignores ground_temperature). The Air box used to stay ticked as
        "Standard day (ISA)" and Pad Temperature kept its old value, so the
        tab said one atmosphere while the flight used another. Now the box
        shows unticked and the spin box shows the sounding's pad temperature,
        both greyed out and with signals blocked: nothing is written to the
        state, and clearing the sounding brings back the saved setting."""
        active = bool(getattr(s, "atmosphere_profile", None))
        self.btn_sounding_clear.setEnabled(active)
        self.isa_check.setEnabled(not active)
        if active:
            from environment.atmosphere_model import launch_site_atmosphere
            self.isa_check.blockSignals(True)
            self.isa_check.setChecked(False)
            self.isa_check.blockSignals(False)
            try:
                pad_t = launch_site_atmosphere(
                    s.launch_altitude, 0.0, s.atmosphere_profile).temperature(0.0)
                self.temp_spin.blockSignals(True)
                self.temp_spin.setValue(pad_t)
                self.temp_spin.blockSignals(False)
            except Exception:           # malformed profile: config_problems reports it
                pass
            self.temp_spin.setEnabled(False)
            self.sounding_label.setText(
                f"Air from sounding: {s.atmosphere_source or 'imported profile'} "
                f"({len(s.atmosphere_profile)} levels). Wind layers are heights "
                f"above the pad elevation at import; re-import after moving "
                f"the pad.")
        else:
            self.sounding_label.setText("")
        self.sounding_label.setVisible(active)

    def _on_isa_toggled(self, standard_day: bool):
        """Standard day stores 0 (ISA at the site). Unticking starts the pad
        temperature from the standard value at this elevation."""
        self.temp_spin.setEnabled(not standard_day)
        if standard_day:
            self.engine.update(ground_temperature=0.0)
        else:
            from environment.atmosphere_model import isa_temperature
            t = round(isa_temperature(self.engine.state.launch_altitude), 1)
            self.temp_spin.blockSignals(True)
            self.temp_spin.setValue(t)
            self.temp_spin.blockSignals(False)
            self.engine.update(ground_temperature=t)

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
        self.btn_run.setText("Resume")

    def _on_stop(self):
        self.sim_engine.stop()

    def _on_reset(self):
        if self.sim_engine.is_running:
            self.sim_engine.stop()
        # Back on the pad with the same rocket, motor and launch setup.
        # This called engine.reset(), which wiped all three; only the
        # geometry was put back, so the next RUN had no motor and flew the
        # default wind and parachutes under the values still shown here.
        self.engine.reset_flight()
        # A multistage flight leaves the dropped stages' mass and length on
        # the state; the design puts the whole stack back.
        design = getattr(self.window(), "design_ws", None)
        if design is not None and getattr(design, "assembly", None) is not None:
            try:
                design._sync_to_engine()
            except Exception:
                logger.warning("Post-reset design re-sync failed", exc_info=True)
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

    def _table_wind_layers(self):
        """The layer table as [(alt, speed, dir)], skipping unfinished rows."""
        layers = []
        for r in range(self.wind_table.rowCount()):
            try:
                alt = float(self.wind_table.item(r, 0).text())
                spd = float(self.wind_table.item(r, 1).text())
                drn = float(self.wind_table.item(r, 2).text())
                layers.append((alt, max(0.0, spd), drn % 360.0))
            except (ValueError, AttributeError):
                continue  # skip incomplete/non-numeric rows
        return layers

    def _push_wind_layers(self):
        """Read the layer table and push it into simulation state."""
        self.engine.update(wind_layers=self._table_wind_layers())

    # ── State → inputs ────────────────────────────────────────────

    def _sync_inputs_from_state(self, s):
        """Make every input on this tab show what the state holds.

        The inputs only ever wrote to the state. Opening a project that was
        saved with a 10 m/s wind flew the 10 m/s wind while this tab showed
        0, and the first spin of any box here then overwrote the saved value
        with whatever the box happened to hold. Signals are blocked, so
        showing a value never writes it back.
        """
        def _show(spin, value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return
            if abs(spin.value() - value) > 0.5 * 10 ** (-spin.decimals()):
                spin.blockSignals(True)
                spin.setValue(value)
                spin.blockSignals(False)

        gust = getattr(s, "wind_gust_intensity", 0.0) or 0.0
        _show(self.angle_spin, s.launch_angle)
        _show(self.azimuth_spin, s.launch_azimuth)
        _show(self.rod_spin, s.launch_rod_length)
        _show(self.site_alt_spin, s.launch_altitude)
        _show(self.site_lat_spin, s.launch_latitude)
        _show(self.site_lon_spin, s.launch_longitude)
        if self.site_combo.currentText() != (s.launch_site_name or CUSTOM_SITE):
            self._select_site(s.launch_site_name)
        standard_day = not (s.ground_temperature and s.ground_temperature > 0)
        if self.isa_check.isChecked() != standard_day:
            self.isa_check.blockSignals(True)
            self.isa_check.setChecked(standard_day)
            self.isa_check.blockSignals(False)
        self.temp_spin.setEnabled(not standard_day)
        if not standard_day:
            _show(self.temp_spin, s.ground_temperature)
        else:                           # greyed out: the ISA value it stands for
            from environment.atmosphere_model import isa_temperature
            _show(self.temp_spin, isa_temperature(s.launch_altitude))
        self._show_sounding(s)
        _show(self.wind_spin, s.wind_speed)
        _show(self.wind_dir_spin, s.wind_direction)
        _show(self.wind_turb_spin, gust * 100.0)
        _show(self.wind_ml_turb_spin, gust * 100.0)
        _show(self.wind_stddev_spin, s.wind_speed * gust)
        _show(self.drogue_delay_spin, s.drogue_deploy_delay)
        _show(self.main_alt_spin, s.main_deploy_altitude)
        _show(self.drogue_cda_spin, s.drogue_cd_area)
        _show(self.main_cda_spin, s.main_cd_area)
        _show(self.inflation_spin, s.recovery_inflation_time)

        mode_index = 1 if s.wind_mode == "multi_level" else 0
        if self.wind_mode_combo.currentIndex() != mode_index:
            self.wind_mode_combo.blockSignals(True)
            self.wind_mode_combo.setCurrentIndex(mode_index)
            self.wind_mode_combo.blockSignals(False)
            self.wind_stack.setCurrentIndex(mode_index)

        integrator = {"rk4": "RK4", "euler": "Euler"}.get(
            str(s.integrator_name).lower())
        if integrator and self.integrator_combo.currentText() != integrator:
            self.integrator_combo.blockSignals(True)
            self.integrator_combo.setCurrentText(integrator)
            self.integrator_combo.blockSignals(False)

        try:
            layers = [(float(a), float(v), float(d)) for a, v, d in s.wind_layers]
        except (TypeError, ValueError):
            layers = None
        if layers is not None and layers != self._table_wind_layers():
            self.wind_table.blockSignals(True)
            self.wind_table.setRowCount(0)
            for row, layer in enumerate(layers):
                self.wind_table.insertRow(row)
                for col, val in enumerate(layer):
                    self.wind_table.setItem(row, col, QTableWidgetItem(f"{val:.10g}"))
            self.wind_table.blockSignals(False)

        self._show_recovery_source(s)

    def _show_recovery_source(self, s):
        """The Design tab's parachutes, when there are any, are the recovery;
        the drogue/main slots here are then not flown, so they are locked."""
        rows = getattr(s, "recovery_canopies", None) or []
        for w in (self.drogue_delay_spin, self.main_alt_spin,
                  self.drogue_cda_spin, self.main_cda_spin):
            w.setEnabled(not rows)
        if rows:
            parts = []
            for r in rows:
                when = ("apogee" if r.get("event") == "apogee"
                        else f"{r.get('altitude', 0):.0f} m")
                if r.get("delay"):
                    when += f" + {r['delay']:.1f} s"
                parts.append(f"{r.get('name', 'Parachute')} "
                             f"({r.get('cd_area', 0):.2f} m², {when})")
            self.recovery_note.setText(
                "From the Design tab: " + "; ".join(parts)
                + ". Edit the parachutes there.")
        self.recovery_note.setVisible(bool(rows))

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
        self.btn_run.setText("Run")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.angle_spin.setEnabled(True)
        self.integrator_combo.setEnabled(True)
        self.progress.setValue(100)
        self._update_trace()

    def _on_sim_failed(self, title, detail):
        """Flight diverged / unstable / bad config — restore controls and show
        the diagnosed problem so the user knows what to fix."""
        self._on_sim_finished()              # restore Run/Pause/Stop buttons
        # Light the ABORTED phase if it has a lamp; otherwise just flag prelaunch.
        for light in self.phase_lights.values():
            light.set_active(False)
        if FlightPhase.ABORTED in self.phase_lights:
            self.phase_lights[FlightPhase.ABORTED].set_active(True)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Simulation Aborted")
        box.setText(f"<b>{title}</b>")
        box.setInformativeText(detail)
        box.exec()

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
        self.btn_run.setText("Run")
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.angle_spin.setEnabled(True)
        self.integrator_combo.setEnabled(True)
        self.progress.setValue(0)
        self.time_label.setText("T+ 0.00 s")
        self.phase_label.setText("Pre-Launch")
        for v in self.readouts.values():
            v.setText("—")
        self.trace.set_data([], [], [])
        for light in self.phase_lights.values():
            light.set_active(False)

    def _on_tick(self, state):
        self._update_readouts(state)
        self._update_trace(force=False)

    def _update_trace(self, force=True):
        """Draw the recorded flight so far on the live trace."""
        hist = getattr(self.sim_engine, "history", None)
        if hist is None or not hasattr(self, "trace"):
            return
        times, alt = hist.get_series("altitude")
        self.trace.set_data(times, alt, hist.get_values("velocity"), force=force)

    def showEvent(self, event):
        super().showEvent(event)
        self._update_trace()

    def _on_state_changed(self, state):
        self._sync_inputs_from_state(state)
        self._update_readouts(state)

    def _update_readouts(self, s):
        self.time_label.setText(f"T+ {s.sim_time:.2f} s")
        self.phase_label.setText(s.sim_phase)

        from ui.units import format_value as fv
        self.readouts["Altitude"].setText(fv(s.altitude, "altitude"))
        self.readouts["Velocity"].setText(fv(s.velocity, "velocity"))
        self.readouts["Acceleration"].setText(fv(s.acceleration, "acceleration"))
        self.readouts["Thrust"].setText(fv(s.thrust, "force"))
        self.readouts["Drag"].setText(fv(s.drag, "force"))
        self.readouts["Mach"].setText(f"{s.mach_number:.3f}")
        self.readouts["Mass"].setText(fv(s.flight_mass(), "mass"))
        self.readouts["Dyn. Pressure"].setText(fv(s.dynamic_pressure, "pressure"))

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
