"""
K2 Aerospace - Mission Visualizer Workspace
============================================
Real-time 3D mission visualization synchronized with the simulation engine.
Subscribes to engine.telemetry_tick for live rendering while the solver runs
on the main thread. Enters replay mode automatically when simulation completes.

Rendering is decoupled from the physics tick:
  - telemetry_tick handler only buffers data + updates cheap text readouts
  - a 30 fps QTimer rebuilds the 3D scene from the latest buffered state
  - graphs refresh on a slower timer
So the solver is never blocked by rendering.
"""

import logging
import math
from collections import deque
from time import perf_counter

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton,
    QComboBox, QSlider, QTabWidget, QFrame, QScrollArea, QListWidget,
    QListWidgetItem, QSizePolicy, QCheckBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    _PYVISTA_OK = True
except Exception:
    _PYVISTA_OK = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except Exception:
    _MPL_OK = False

from core.event_manager import SimEvent

try:
    from visualization.mission.recovery_visualizer import RecoveryVisualizer
    from visualization.mission.altitude_reference import AltitudeReferenceSystem
    from visualization.mission.flight_envelope import FlightEnvelope, estimate_landing
    _OVERLAYS_OK = True
except Exception:
    _OVERLAYS_OK = False

logger = logging.getLogger("K2.MissionViz")

_PHASE_COLORS = {
    "Pre-Launch":     "#58a6ff",
    "Boost":          "#ffa657",
    "Coast":          "#7ee787",
    "Apogee":         "#f0883e",
    "Drogue Descent": "#79c0ff",
    "Main Descent":   "#56d364",
    "Landed":         "#3fb950",
    "Timeout":        "#f85149",
    "Terminated":     "#f85149",
}

_EVENT_LABELS = {
    "sim_start":      ("Ignition",            "#58a6ff"),
    "motor_ignition": ("Motor Ignition",      "#ffa657"),
    "motor_burnout":  ("Burnout",             "#ffa657"),
    "apogee":         ("Apogee",              "#f0883e"),
    "drogue_deploy":  ("Drogue Deploy",       "#79c0ff"),
    "main_deploy":    ("Main Chute Deploy",   "#56d364"),
    "landing":        ("Landing",             "#3fb950"),
    "max_q":          ("Max-Q",               "#d29922"),
    "sim_end":        ("Sim End",             "#3fb950"),
}


class _Readout(QWidget):
    """Compact telemetry readout: label + value + unit."""

    def __init__(self, label, unit="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 4)
        lay.setSpacing(1)

        lbl = QLabel(label.upper())
        lbl.setStyleSheet("color:#8b949e;font-size:9px;font-weight:600;letter-spacing:1px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._val = QLabel("--")
        self._val.setStyleSheet(
            "color:#e6edf3;font-family:'Cascadia Code',monospace;"
            "font-size:14px;font-weight:700;"
        )
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay.addWidget(lbl)
        lay.addWidget(self._val)
        if unit:
            u = QLabel(unit)
            u.setStyleSheet("color:#484f58;font-size:8px;")
            u.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(u)

        self.setStyleSheet("background:#161b22;border:1px solid #21262d;border-radius:6px;")
        self.setMinimumWidth(85)

        self._last_text = None
        self._last_col = None

    def set_value(self, v, fmt="{:.1f}", color=None):
        text = v if isinstance(v, str) else fmt.format(v)
        if text != self._last_text:
            self._val.setText(text)
            self._last_text = text
        col = color or "#e6edf3"
        # setStyleSheet reparses CSS — only touch it when the colour changes.
        if col != self._last_col:
            self._val.setStyleSheet(
                f"color:{col};font-family:'Cascadia Code',monospace;"
                "font-size:14px;font-weight:700;"
            )
            self._last_col = col


class MissionVisualizerWorkspace(QWidget):
    """
    Live 3D mission visualizer. Renders the rocket flying in real time as the
    simulation runs, then enters replay mode when the flight finishes.
    """

    _CAMERA_MODES = ["Chase", "Launch Pad", "Side View", "FPV", "Free", "Recovery"]
    _TRAIL_MODES  = ["Mach", "Altitude", "Velocity", "Flight Phase"]

    # Rocket model drawn this many times real size so it stays visible at altitude
    _VIS_SCALE = 30.0

    def __init__(self, engine, sim_engine, parent=None):
        super().__init__(parent)
        self.engine     = engine
        self.sim_engine = sim_engine

        # Live state
        self._latest_state    = None
        self._needs_3d_update = False
        self._failure_active  = False

        # Trail buffers (capped at 100k points)
        self._trail_pts  = deque(maxlen=100_000)
        self._trail_mach = deque(maxlen=100_000)
        self._trail_alt  = deque(maxlen=100_000)
        self._trail_vel  = deque(maxlen=100_000)

        # Downsampled graph data
        self._g_time  = deque(maxlen=5_000)
        self._g_alt   = deque(maxlen=5_000)
        self._g_vel   = deque(maxlen=5_000)
        self._g_mach  = deque(maxlen=5_000)
        self._g_accel = deque(maxlen=5_000)
        self._g_dynq  = deque(maxlen=5_000)
        self._tick_counter = 0

        # Scene actor handles
        self._plotter       = None
        self._actor_rocket  = None
        self._actor_trail   = None
        self._actor_landing = None
        self._actor_failure = None
        self._trail_rebuild_counter = 0
        self._scene_ready = False
        self._closing = False

        # Rocket geometry (updated from state)
        self._rocket_length   = 2.0
        self._rocket_diameter = 0.1

        # Camera / trail settings
        self._camera_mode      = "Chase"
        self._trail_color_mode = "Mach"

        # Replay state
        self._replay_mode       = False
        self._replay_index      = 0
        self._replay_is_playing = False
        self._replay_pts   = []
        self._replay_mach  = []
        self._replay_alt   = []
        self._replay_vel   = []
        self._replay_times = []

        # Scene overlays (recovery / altitude reference / flight envelope)
        self._recovery   = None
        self._alt_ref    = None
        self._envelope   = None
        self._show_alt_planes = True
        self._show_scalebar   = False
        self._show_envelope   = True

        # Mission capture (positions/times grabbed live, reused in replay)
        self._apogee_xyz   = None
        self._apogee_value = 0.0
        self._drogue_info  = None   # (time, altitude)
        self._main_info    = None
        self._landing_xy   = None
        self._alt_ref_apogee = 0.0  # apogee the alt reference is currently sized to

        # Rocket geometry caching (build mesh once, transform per frame)
        self._rocket_dims_key = None

        # Camera smoothing (interpolate toward target instead of snapping)
        self._cam_target_pos   = None
        self._cam_target_focal = None
        self._cam_snap         = True   # next apply is instant (mode switch/seek)
        self._CAM_EASE         = 0.22   # lerp factor per render tick

        # Adaptive quality + profiling
        self._quality       = "Balanced"
        self._trail_cap     = 2000      # max rendered trail points
        self._label_lod     = False     # distance-cull labels (off: always show)
        self._labels_hidden = False
        self._show_stats    = False
        self._fps           = 0.0
        self._frame_ms      = 0.0
        self._render_ms     = 0.0
        self._graph_ms      = 0.0
        self._panel_ms      = 0.0
        self._last_frame_t  = None

        self._build_ui()
        self._subscribe_signals()

        # Render timer (interval set by quality mode; default 30 fps)
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(33)
        self._render_timer.timeout.connect(self._on_render_tick)
        self._render_timer.start()

        # Telemetry panel timer - 10 Hz (decoupled from sim tick)
        self._panel_timer = QTimer(self)
        self._panel_timer.setInterval(100)
        self._panel_timer.timeout.connect(self._update_readouts)
        self._panel_timer.start()

        # Graph update timer - interval set by quality mode
        self._graph_timer = QTimer(self)
        self._graph_timer.setInterval(300)
        self._graph_timer.timeout.connect(self._update_graphs)
        self._graph_timer.start()

        # Replay playback timer
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(50)
        self._replay_timer.timeout.connect(self._on_replay_tick)

        self._apply_quality("Balanced")

    # ===============================================================
    # UI CONSTRUCTION
    # ===============================================================

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_title_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background: #21262d; }")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([920, 280])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_controls_bar())

    def _build_title_bar(self):
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet("background:#0d1117;border-bottom:1px solid #21262d;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        title = QLabel("MISSION VISUALIZER")
        title.setStyleSheet(
            "color:#58a6ff;font-size:13px;font-weight:700;letter-spacing:3px;"
        )
        lay.addWidget(title)
        lay.addStretch()

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(
            "color:#7ee787;font-family:'Cascadia Code',monospace;font-size:10px;"
            "padding:2px 8px;margin-right:8px;"
        )
        self._stats_lbl.setVisible(False)
        lay.addWidget(self._stats_lbl)

        self._status_lbl = QLabel("STANDBY")
        self._status_lbl.setStyleSheet(
            "color:#484f58;font-size:11px;font-weight:600;"
            "padding:2px 10px;border:1px solid #21262d;border-radius:4px;"
        )
        lay.addWidget(self._status_lbl)

        self._phase_lbl = QLabel("--")
        self._phase_lbl.setStyleSheet(
            "color:#8b949e;font-size:11px;font-weight:600;"
            "padding:2px 10px;border:1px solid #21262d;border-radius:4px;margin-left:8px;"
        )
        lay.addWidget(self._phase_lbl)
        return bar

    def _build_left_panel(self):
        w = QWidget()
        w.setStyleSheet("background:#0d1117;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 3D view
        view_frame = QFrame()
        view_frame.setStyleSheet("background:#0d1117;")
        vfl = QVBoxLayout(view_frame)
        vfl.setContentsMargins(0, 0, 0, 0)

        if _PYVISTA_OK:
            try:
                self._plotter = QtInteractor(view_frame)
                vfl.addWidget(self._plotter.interactor)
                self._plotter.set_background("#0d1117")
                self._init_3d_scene()
                self._scene_ready = True
            except Exception as exc:
                logger.warning(f"PyVista init failed: {exc}")
                import traceback
                traceback.print_exc()
                self._plotter = None
                vfl.addWidget(self._fallback_label(
                    "3D view unavailable - PyVista error (see console)"
                ))
        else:
            vfl.addWidget(self._fallback_label(
                "PyVista not installed.\npip install pyvista pyvistaqt"
            ))

        lay.addWidget(view_frame, stretch=4)

        # Live graph strip
        lay.addWidget(self._build_graph_strip(), stretch=0)
        return w

    def _fallback_label(self, text):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color:#484f58;font-size:12px;")
        return lbl

    def _build_graph_strip(self):
        frame = QFrame()
        frame.setFixedHeight(125)
        frame.setStyleSheet("background:#0d1117;border-top:1px solid #21262d;")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        self._graph_axes     = []
        self._graph_lines    = []
        self._graph_canvases = []

        specs = [
            ("Altitude", "m"),
            ("Velocity", "m/s"),
            ("Mach", ""),
            ("Accel", "m/s2"),
            ("Dyn-Q", "Pa"),
        ]

        if _MPL_OK:
            for label, unit in specs:
                fig = Figure(figsize=(2, 1), dpi=70)
                fig.patch.set_facecolor("#0d1117")
                ax = fig.add_subplot(111)
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors="#484f58", labelsize=6)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#21262d")
                title_str = f"{label} ({unit})" if unit else label
                ax.set_title(title_str, color="#8b949e", fontsize=7, pad=2)
                fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.22)
                line, = ax.plot([], [], color="#58a6ff", linewidth=1.0)
                canvas = FigureCanvas(fig)
                canvas.setMinimumWidth(110)
                canvas.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
                )
                lay.addWidget(canvas, stretch=1)
                self._graph_axes.append(ax)
                self._graph_lines.append(line)
                self._graph_canvases.append(canvas)
        else:
            lay.addWidget(self._fallback_label("matplotlib unavailable"))
        return frame

    def _build_right_panel(self):
        tabs = QTabWidget()
        tabs.setMinimumWidth(250)
        tabs.setStyleSheet("""
            QTabWidget::pane { border:none; background:#0d1117; }
            QTabBar::tab {
                color:#8b949e; background:#161b22; padding:6px 10px;
                border:1px solid #21262d; border-bottom:none; font-size:10px;
            }
            QTabBar::tab:selected {
                color:#e6edf3; background:#0d1117;
                border-top:2px solid #58a6ff;
            }
        """)
        tabs.addTab(self._build_telemetry_tab(), "Telemetry")
        tabs.addTab(self._build_events_tab(),    "Flight Events")
        tabs.addTab(self._build_timeline_tab(),  "Timeline")
        return tabs

    def _build_telemetry_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background:#0d1117;")
        inner = QWidget()
        inner.setStyleSheet("background:#0d1117;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        def row(*widgets):
            r = QWidget()
            rl = QHBoxLayout(r)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(5)
            for wdg in widgets:
                rl.addWidget(wdg, stretch=1)
            return r

        self._rd_time     = _Readout("Mission Time", "s")
        self._rd_alt      = _Readout("Altitude", "m")
        self._rd_vel      = _Readout("Velocity", "m/s")
        self._rd_accel    = _Readout("Acceleration", "m/s2")
        self._rd_mach     = _Readout("Mach", "")
        self._rd_mass     = _Readout("Mass", "kg")
        self._rd_thrust   = _Readout("Thrust", "N")
        self._rd_drag     = _Readout("Drag", "N")
        self._rd_dynq     = _Readout("Dyn Pressure", "Pa")
        self._rd_stab     = _Readout("Stability", "cal")
        self._rd_phase    = _Readout("Phase", "")
        self._rd_recovery = _Readout("Recovery", "")

        lay.addWidget(row(self._rd_time, self._rd_alt))
        lay.addWidget(row(self._rd_vel, self._rd_accel))
        lay.addWidget(row(self._rd_mach, self._rd_mass))
        lay.addWidget(row(self._rd_thrust, self._rd_drag))
        lay.addWidget(row(self._rd_dynq, self._rd_stab))
        lay.addWidget(row(self._rd_phase, self._rd_recovery))

        # ── Recovery section ──
        hdr = QLabel("RECOVERY")
        hdr.setStyleSheet(
            "color:#56d364;font-size:9px;font-weight:700;letter-spacing:2px;"
            "padding:6px 2px 2px 2px;"
        )
        lay.addWidget(hdr)

        self._rd_rec_state   = _Readout("Recovery State", "")
        self._rd_rec_area    = _Readout("Canopy Area", "m2")
        self._rd_rec_descent = _Readout("Descent Rate", "m/s")
        self._rd_rec_depalt  = _Readout("Deploy Alt", "m")
        self._rd_rec_since   = _Readout("Since Deploy", "s")

        lay.addWidget(row(self._rd_rec_state, self._rd_rec_area))
        lay.addWidget(row(self._rd_rec_descent, self._rd_rec_depalt))
        lay.addWidget(row(self._rd_rec_since))
        lay.addStretch()
        scroll.setWidget(inner)
        return scroll

    def _build_events_tab(self):
        w = QWidget()
        w.setStyleSheet("background:#0d1117;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._events_list = QListWidget()
        self._events_list.setStyleSheet("""
            QListWidget { background:#0d1117; border:none; color:#e6edf3; font-size:11px; }
            QListWidget::item { padding:5px 6px; border-bottom:1px solid #161b22; }
            QListWidget::item:selected { background:#161b22; color:#58a6ff; }
        """)
        self._events_list.itemClicked.connect(self._on_event_clicked)
        lay.addWidget(self._events_list)

        hint = QLabel("Click an event to jump replay")
        hint.setStyleSheet("color:#484f58;font-size:9px;padding:4px 2px;")
        lay.addWidget(hint)
        return w

    def _build_timeline_tab(self):
        w = QWidget()
        w.setStyleSheet("background:#0d1117;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._timeline_list = QListWidget()
        self._timeline_list.setStyleSheet("""
            QListWidget {
                background:#0d1117; border:none; color:#e6edf3;
                font-family:'Cascadia Code',monospace; font-size:10px;
            }
            QListWidget::item { padding:4px 6px; border-bottom:1px solid #161b22; }
            QListWidget::item:selected { background:#161b22; color:#58a6ff; }
        """)
        lay.addWidget(self._timeline_list)
        return w

    def _build_controls_bar(self):
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet("background:#161b22;border-top:1px solid #21262d;")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(8)

        lay.addWidget(self._styled_label("Camera:"))
        self._cam_combo = self._styled_combo(self._CAMERA_MODES)
        self._cam_combo.currentTextChanged.connect(self._on_camera_mode_changed)
        lay.addWidget(self._cam_combo)

        lay.addWidget(self._styled_label("Trail:"))
        self._trail_combo = self._styled_combo(self._TRAIL_MODES)
        self._trail_combo.currentTextChanged.connect(self._on_trail_mode_changed)
        lay.addWidget(self._trail_combo)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.VLine)
        sep0.setStyleSheet("color:#30363d;")
        lay.addWidget(sep0)

        # Overlay toggles
        self._chk_planes = self._styled_check("Alt Planes", True)
        self._chk_planes.toggled.connect(self._on_toggle_planes)
        lay.addWidget(self._chk_planes)

        self._chk_scalebar = self._styled_check("Scale Bar", False)
        self._chk_scalebar.toggled.connect(self._on_toggle_scalebar)
        lay.addWidget(self._chk_scalebar)

        self._chk_envelope = self._styled_check("Envelope", True)
        self._chk_envelope.toggled.connect(self._on_toggle_envelope)
        lay.addWidget(self._chk_envelope)

        lay.addWidget(self._styled_label("Target:"))
        self._spin_target = self._styled_spin(0, 50000, 0, " m")
        self._spin_target.editingFinished.connect(self._on_envelope_config_changed)
        lay.addWidget(self._spin_target)

        lay.addWidget(self._styled_label("Recov R:"))
        self._spin_recov = self._styled_spin(50, 50000, 1000, " m")
        self._spin_recov.editingFinished.connect(self._on_envelope_config_changed)
        lay.addWidget(self._spin_recov)

        lay.addWidget(self._styled_label("Quality:"))
        self._quality_combo = self._styled_combo(["Performance", "Balanced", "Quality"])
        self._quality_combo.setCurrentText("Balanced")
        self._quality_combo.currentTextChanged.connect(self._apply_quality)
        lay.addWidget(self._quality_combo)

        self._chk_stats = self._styled_check("Stats", False)
        self._chk_stats.toggled.connect(self._on_toggle_stats)
        lay.addWidget(self._chk_stats)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#30363d;")
        lay.addWidget(sep)

        # Replay controls (shown only in replay mode)
        self._pb_group = QWidget()
        pb_lay = QHBoxLayout(self._pb_group)
        pb_lay.setContentsMargins(0, 0, 0, 0)
        pb_lay.setSpacing(5)
        pb_lay.addWidget(self._styled_label("Replay:"))

        btn_style = (
            "QPushButton { background:#21262d; color:#e6edf3; border:1px solid #30363d;"
            "border-radius:4px; padding:2px 10px; font-size:12px; font-weight:600; }"
            "QPushButton:hover { background:#30363d; }"
        )
        self._btn_restart = QPushButton("Restart")
        self._btn_play    = QPushButton("Play")
        self._btn_pause   = QPushButton("Pause")
        for btn in (self._btn_restart, self._btn_play, self._btn_pause):
            btn.setFixedHeight(30)
            btn.setStyleSheet(btn_style)

        self._replay_slider = QSlider(Qt.Orientation.Horizontal)
        self._replay_slider.setRange(0, 1000)
        self._replay_slider.setFixedWidth(180)
        self._replay_slider.setStyleSheet(
            "QSlider::groove:horizontal{background:#21262d;height:4px;border-radius:2px;}"
            "QSlider::handle:horizontal{background:#58a6ff;width:12px;height:12px;"
            "border-radius:6px;margin:-4px 0;}"
        )
        self._replay_time_lbl = QLabel("T+0.00s")
        self._replay_time_lbl.setStyleSheet(
            "color:#8b949e;font-family:'Cascadia Code',monospace;font-size:10px;"
        )

        self._btn_restart.clicked.connect(self._on_replay_restart)
        self._btn_play.clicked.connect(self._on_replay_play)
        self._btn_pause.clicked.connect(self._on_replay_pause)
        self._replay_slider.sliderMoved.connect(self._on_replay_seek)

        pb_lay.addWidget(self._btn_restart)
        pb_lay.addWidget(self._btn_play)
        pb_lay.addWidget(self._btn_pause)
        pb_lay.addWidget(self._replay_slider)
        pb_lay.addWidget(self._replay_time_lbl)
        self._pb_group.setVisible(False)
        lay.addWidget(self._pb_group)

        lay.addStretch()

        reset_btn = QPushButton("Reset Camera")
        reset_btn.setStyleSheet(
            "QPushButton{background:#21262d;color:#8b949e;border:1px solid #30363d;"
            "border-radius:4px;padding:4px 10px;font-size:11px;}"
            "QPushButton:hover{background:#30363d;color:#e6edf3;}"
        )
        reset_btn.clicked.connect(self._reset_camera)
        lay.addWidget(reset_btn)
        return bar

    def _styled_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#8b949e;font-size:11px;")
        return lbl

    def _styled_combo(self, items):
        cb = QComboBox()
        cb.addItems(items)
        cb.setStyleSheet(
            "QComboBox{background:#0d1117;color:#e6edf3;border:1px solid #30363d;"
            "border-radius:4px;padding:3px 8px;font-size:11px;min-width:100px;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#161b22;color:#e6edf3;"
            "selection-background-color:#21262d;}"
        )
        return cb

    def _styled_check(self, text, checked):
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setStyleSheet(
            "QCheckBox{color:#8b949e;font-size:11px;spacing:4px;}"
            "QCheckBox::indicator{width:12px;height:12px;border:1px solid #30363d;"
            "border-radius:3px;background:#0d1117;}"
            "QCheckBox::indicator:checked{background:#58a6ff;border-color:#58a6ff;}"
        )
        return cb

    def _styled_spin(self, lo, hi, val, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(0)
        sp.setSingleStep(100)
        sp.setValue(val)
        if suffix:
            sp.setSuffix(suffix)
        sp.setFixedWidth(86)
        sp.setStyleSheet(
            "QDoubleSpinBox{background:#0d1117;color:#e6edf3;border:1px solid #30363d;"
            "border-radius:4px;padding:2px 4px;font-size:11px;}"
        )
        return sp

    # ===============================================================
    # 3D SCENE
    # ===============================================================

    def _init_3d_scene(self):
        if not self._plotter:
            return
        p = self._plotter

        # Ground plane
        ground = pv.Plane(
            center=(0, 0, 0), direction=(0, 0, 1),
            i_size=8000, j_size=8000, i_resolution=20, j_resolution=20
        )
        p.add_mesh(ground, color="#111820", opacity=0.95,
                   show_edges=True, edge_color="#1e2530", line_width=0.4,
                   name="ground")

        # Launch rail (scaled for visibility)
        rail = pv.Cylinder(
            center=(0, 0, 30), direction=(0, 0, 1), radius=0.6, height=60.0
        )
        p.add_mesh(rail, color="#484f58", name="rail", smooth_shading=True)

        # Launch pad
        pad = pv.Disc(center=(0, 0, 0.5), normal=(0, 0, 1),
                      inner=0.0, outer=12.0, r_res=2, c_res=24)
        p.add_mesh(pad, color="#21262d", name="pad")

        # Initial camera: framing the pad, ready for liftoff
        p.camera.position    = (300, -350, 180)
        p.camera.focal_point = (0, 0, 120)
        p.camera.up = (0, 0, 1)

        # Scene overlay systems (share this plotter)
        if _OVERLAYS_OK:
            try:
                self._recovery = RecoveryVisualizer(p, vis_scale=self._VIS_SCALE)
                self._alt_ref  = AltitudeReferenceSystem(p)
                self._envelope = FlightEnvelope(p)
            except Exception as exc:
                logger.warning(f"Overlay init failed: {exc}")
                self._recovery = self._alt_ref = self._envelope = None

        try:
            p.enable_anti_aliasing()
        except Exception:
            pass
        try:
            p.render()
        except Exception:
            pass

    def _ensure_rocket_actor(self):
        """Build the rocket mesh ONCE (real size at origin). Pose set per frame
        via cheap VTK actor transforms — no geometry regen each frame."""
        if not self._plotter:
            return
        key = (round(self._rocket_length, 4), round(self._rocket_diameter, 4))
        if self._actor_rocket is not None and key == self._rocket_dims_key:
            return
        if self._actor_rocket is not None:
            try:
                self._plotter.remove_actor(self._actor_rocket)
            except Exception:
                pass
            self._actor_rocket = None

        L = max(0.5, self._rocket_length)
        D = max(0.05, self._rocket_diameter)
        body = pv.Cylinder(
            center=(0, 0, L * 0.4), direction=(0, 0, 1),
            radius=D / 2, height=L * 0.8, resolution=12
        )
        nose = pv.Cone(
            center=(0, 0, L * 0.9), direction=(0, 0, 1),
            height=L * 0.2, radius=D / 2, resolution=12
        )
        rocket = body.merge(nose)
        try:
            self._actor_rocket = self._plotter.add_mesh(
                rocket, color="#58a6ff", name="rocket",
                smooth_shading=True, ambient=0.35, diffuse=0.65
            )
            self._rocket_dims_key = key
            self._rocket_color = None
        except Exception as exc:
            logger.debug(f"Rocket build error: {exc}")

    def _update_rocket_transform(self, x, y, z, pitch):
        """Cheap per-frame pose update: scale + tilt + position + colour."""
        self._ensure_rocket_actor()
        a = self._actor_rocket
        if a is None:
            return
        tilt_deg = math.degrees(math.pi / 2.0 - max(0.0, min(math.pi, pitch)))
        try:
            a.SetScale(self._VIS_SCALE)
            a.SetOrientation(0.0, tilt_deg, 0.0)
            a.SetPosition(x, y, z)
            color = (0.973, 0.318, 0.286) if self._failure_active else (0.345, 0.651, 1.0)
            if color != getattr(self, "_rocket_color", None):
                a.GetProperty().SetColor(*color)
                self._rocket_color = color
        except Exception as exc:
            logger.debug(f"Rocket transform error: {exc}")

    def _rebuild_trail_actor(self, pts, mach_arr, alt_arr, vel_arr):
        if not self._plotter or len(pts) < 2:
            return
        if self._actor_trail is not None:
            try:
                self._plotter.remove_actor(self._actor_trail)
            except Exception:
                pass
            self._actor_trail = None

        pts_arr  = np.array(pts, dtype=float)
        mach_arr = np.asarray(mach_arr, dtype=float)
        alt_arr  = np.asarray(alt_arr, dtype=float)
        vel_arr  = np.asarray(vel_arr, dtype=float)
        n = len(pts_arr)

        # Decimate to the quality-mode point cap for rendering
        cap = max(200, self._trail_cap)
        if n > cap:
            step = max(1, n // cap)
            pts_arr  = pts_arr[::step]
            mach_arr = mach_arr[::step]
            alt_arr  = alt_arr[::step]
            vel_arr  = vel_arr[::step]
            n = len(pts_arr)

        if n < 2:
            return

        # Build polyline connectivity: [2, i, i+1] per segment
        idx = np.arange(n - 1)
        cells = np.column_stack(
            [np.full(n - 1, 2), idx, idx + 1]
        ).ravel()

        poly = pv.PolyData(pts_arr)
        poly.lines = cells

        mode = self._trail_color_mode
        if mode == "Altitude":
            scalars = alt_arr
            cmap, clim = "viridis", [0.0, float(alt_arr.max()) + 1.0]
        elif mode == "Velocity":
            scalars = vel_arr
            cmap, clim = "hot", [0.0, float(vel_arr.max()) + 1.0]
        elif mode == "Flight Phase":
            scalars = mach_arr
            cmap, clim = "cool", [0.0, 2.0]
        else:  # Mach
            scalars = mach_arr
            cmap, clim = "plasma", [0.0, 2.0]

        try:
            self._actor_trail = self._plotter.add_mesh(
                poly, scalars=scalars, cmap=cmap, clim=clim,
                line_width=3.0, name="trail", show_scalar_bar=False,
                render_lines_as_tubes=False,
            )
        except Exception as exc:
            logger.debug(f"Trail actor error: {exc}")

    def _show_landing_marker(self, x, y):
        if not self._plotter:
            return
        if self._actor_landing is not None:
            try:
                self._plotter.remove_actor(self._actor_landing)
            except Exception:
                pass
        marker = pv.Disc(center=(x, y, 1.0), normal=(0, 0, 1),
                         inner=20.0, outer=120.0, r_res=2, c_res=32)
        try:
            self._actor_landing = self._plotter.add_mesh(
                marker, color="#3fb950", opacity=0.7, name="landing"
            )
        except Exception:
            pass

    def _show_failure_sphere(self, x, y, z):
        if not self._plotter:
            return
        if self._actor_failure is not None:
            try:
                self._plotter.remove_actor(self._actor_failure)
            except Exception:
                pass
        sphere = pv.Sphere(radius=self._rocket_length * self._VIS_SCALE * 1.5,
                           center=(x, y, z))
        try:
            self._actor_failure = self._plotter.add_mesh(
                sphere, color="#f85149", opacity=0.30, name="failure"
            )
        except Exception:
            pass

    def _apply_camera(self, x, y, z, pitch):
        """Compute the camera *target* pose. Actual camera eases toward it in
        _ease_camera so motion is cinematic rather than snapping each frame."""
        mode = self._camera_mode
        if mode == "Free":
            self._cam_target_pos = None  # user controls camera
            return

        # Framing distance scales with the (exaggerated) rocket size
        rsize = max(60.0, self._rocket_length * self._VIS_SCALE)

        if mode == "Chase":
            d = rsize * 4.0
            pos = (x + d * 0.5, y - d * 0.85, z + d * 0.18)
            foc = (x, y, z)
        elif mode == "Launch Pad":
            pos = (300, -350, 180)
            foc = (x * 0.3, y * 0.3, max(z * 0.5, 120))
        elif mode == "Side View":
            d = rsize * 5.0
            pos = (x, y - d, z)
            foc = (x, y, z)
        elif mode == "FPV":
            d = rsize * 0.6
            pos = (x - math.cos(pitch) * d, y, z - math.sin(pitch) * d)
            foc = (x + math.cos(pitch) * 300, y, z + math.sin(pitch) * 300)
        elif mode == "Recovery":
            d = rsize * 4.0
            pos = (x + d * 0.3, y + d * 0.3, z + d)
            foc = (x, y, z)
        else:
            return
        self._cam_target_pos   = pos
        self._cam_target_focal = foc

    def _ease_camera(self):
        """Lerp the live camera toward its target. Returns True if it moved."""
        if not self._plotter or self._cam_target_pos is None:
            return False
        cam = self._plotter.camera
        tp, tf = self._cam_target_pos, self._cam_target_focal
        if self._cam_snap:
            cam.position = tp
            cam.focal_point = tf
            cam.up = (0, 0, 1)
            self._cam_snap = False
            return True
        cp, cf = cam.position, cam.focal_point
        a = self._CAM_EASE
        npos = tuple(cp[i] + (tp[i] - cp[i]) * a for i in range(3))
        nfoc = tuple(cf[i] + (tf[i] - cf[i]) * a for i in range(3))
        cam.position = npos
        cam.focal_point = nfoc
        cam.up = (0, 0, 1)
        # Converged? (squared distance target<->new position, in m^2)
        d2 = sum((tp[i] - npos[i]) ** 2 for i in range(3))
        return d2 > 1.0

    # ===============================================================
    # TIMERS
    # ===============================================================

    def _on_render_tick(self):
        """Fixed-rate frame: update scene from newest buffered state if dirty,
        ease the camera, render only when something changed. Skipped if hidden
        so a backgrounded tab costs nothing."""
        if self._closing or not self._plotter or not self.isVisible():
            return

        t0 = perf_counter()
        dirty = self._needs_3d_update
        if dirty:
            self._needs_3d_update = False
            if self._replay_mode:
                self._render_replay_scene()
            else:
                self._render_live_scene()

        cam_moved = self._ease_camera()

        if dirty or cam_moved:
            if self._label_lod:
                self._apply_label_lod()
            # Recompute near/far clip planes to fit the whole scene, else far
            # geometry (top altitude planes, recovery ring) gets culled when the
            # user zooms out to inspect the full envelope.
            try:
                self._plotter.reset_camera_clipping_range()
            except Exception:
                pass
            try:
                self._plotter.render()
            except Exception:
                pass
            self._record_frame_timing(t0)

    def _render_live_scene(self):
        """Update rocket/trail/recovery/camera target from latest live state."""
        s = self._latest_state
        if s is None:
            return
        pitch = getattr(s, 'pitch', math.pi / 2)

        self._update_rocket_transform(s.x_position, s.y_position, s.altitude, pitch)

        # Trail: rebuild throttled (every ~1s of render ticks, or while short)
        self._trail_rebuild_counter += 1
        if self._trail_rebuild_counter >= 30 or len(self._trail_pts) < 30:
            self._trail_rebuild_counter = 0
            if len(self._trail_pts) >= 2:
                self._rebuild_trail_actor(
                    list(self._trail_pts), list(self._trail_mach),
                    list(self._trail_alt), list(self._trail_vel),
                )

        if self._camera_mode != "Free":
            self._apply_camera(s.x_position, s.y_position, s.altitude, pitch)

        if s.sim_phase in ("Landed", "Terminated") and self._actor_landing is None:
            self._show_landing_marker(s.x_position, s.y_position)

        descent = abs(s.velocity) if s.velocity < 0 else 0.0
        self._update_recovery(s.x_position, s.y_position, s.altitude,
                              s.sim_time, descent)
        self._ensure_alt_reference(s.altitude)

    def _record_frame_timing(self, t0):
        now = perf_counter()
        self._render_ms = (now - t0) * 1000.0
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if dt > 0:
                inst = 1.0 / dt
                self._fps = inst if self._fps == 0 else self._fps * 0.9 + inst * 0.1
                fm = dt * 1000.0
                self._frame_ms = fm if self._frame_ms == 0 else self._frame_ms * 0.9 + fm * 0.1
        self._last_frame_t = now

    def _apply_label_lod(self):
        """Distance-cull overlay labels: hide text when zoomed far out (#4/#5)."""
        if not self._plotter:
            return
        cam = self._plotter.camera
        cp, cf = cam.position, cam.focal_point
        dist = math.sqrt(sum((cp[i] - cf[i]) ** 2 for i in range(3)))
        thr = max(2500.0, self._alt_ref_apogee * 0.9)
        hide = dist > thr
        if hide != self._labels_hidden:
            self._set_overlay_labels(not hide)
            self._labels_hidden = hide

    def _set_overlay_labels(self, visible):
        if self._alt_ref:
            self._alt_ref.set_labels_visible(visible)
        if self._envelope:
            self._envelope.set_labels_visible(visible)

    def _update_graphs(self):
        if not _MPL_OK or not self._graph_lines or not self._g_time:
            return
        if not self.isVisible():
            return
        t0 = perf_counter()
        times = list(self._g_time)
        series = [
            list(self._g_alt),
            list(self._g_vel),
            list(self._g_mach),
            list(self._g_accel),
            list(self._g_dynq),
        ]
        for ax, line, canvas, data in zip(
            self._graph_axes, self._graph_lines, self._graph_canvases, series
        ):
            if not data:
                continue
            line.set_xdata(times)
            line.set_ydata(data)
            ax.relim()
            ax.autoscale_view()
            try:
                canvas.draw_idle()
            except Exception:
                pass
        self._graph_ms = (perf_counter() - t0) * 1000.0

    # ===============================================================
    # SIGNAL HANDLERS
    # ===============================================================

    @pyqtSlot(object)
    def _on_telemetry_tick(self, s):
        """Called every sim step. Fast path only - buffer + cheap text updates."""
        self._latest_state    = s
        self._needs_3d_update = True
        self._tick_counter   += 1

        # Buffer trail
        self._trail_pts.append((s.x_position, s.y_position, s.altitude))
        self._trail_mach.append(s.mach_number)
        self._trail_alt.append(s.altitude)
        self._trail_vel.append(abs(s.velocity))

        # Downsample graph data (every 5 ticks)
        if self._tick_counter % 5 == 0:
            self._g_time.append(s.sim_time)
            self._g_alt.append(s.altitude)
            self._g_vel.append(abs(s.velocity))
            self._g_mach.append(s.mach_number)
            self._g_accel.append(abs(s.acceleration))
            self._g_dynq.append(s.dynamic_pressure)

        # Failure check (cheap — only acts on state transitions)
        self._check_failures(s)

        # Keep rocket dims current (mesh rebuilt only when these actually change)
        if s.length > 0:
            self._rocket_length = s.length
        if s.diameter > 0:
            self._rocket_diameter = s.diameter

    def _update_readouts(self):
        """10 Hz: refresh telemetry panel from latest state. Decoupled from the
        sim tick so 500 Hz physics never repaints the panel 500 times/sec."""
        if self._closing or not self.isVisible():
            return
        t0 = perf_counter()
        s = self._latest_state
        if s is None:
            return
        self._rd_time.set_value(s.sim_time,          "{:.2f}")
        self._rd_alt.set_value(s.altitude,           "{:.1f}")
        self._rd_vel.set_value(abs(s.velocity),      "{:.1f}")
        self._rd_accel.set_value(abs(s.acceleration),"{:.2f}")
        self._rd_mach.set_value(s.mach_number,       "{:.3f}")
        self._rd_mass.set_value(s.dry_mass + s.propellant_mass, "{:.2f}")
        self._rd_thrust.set_value(s.thrust,          "{:.1f}")
        self._rd_drag.set_value(s.drag,              "{:.1f}")
        self._rd_dynq.set_value(s.dynamic_pressure,  "{:.0f}")

        stab = s.stability_margin
        sc = "#3fb950" if stab >= 1.5 else ("#ffa657" if stab >= 0.5 else "#f85149")
        self._rd_stab.set_value(stab, "{:.2f}", sc)

        pc = _PHASE_COLORS.get(s.sim_phase, "#8b949e")
        self._rd_phase.set_value(s.sim_phase, color=pc)
        if s.sim_phase != getattr(self, "_phase_lbl_text", None):
            self._phase_lbl.setText(s.sim_phase)
            self._phase_lbl.setStyleSheet(
                f"color:{pc};font-size:11px;font-weight:600;"
                "padding:2px 10px;border:1px solid #21262d;border-radius:4px;margin-left:8px;"
            )
            self._phase_lbl_text = s.sim_phase

        if s.parachute_deployed:
            self._rd_recovery.set_value("DEPLOYED", color="#7ee787")
        else:
            self._rd_recovery.set_value(str(s.flight_computer_state))

        self._panel_ms = (perf_counter() - t0) * 1000.0
        if self._show_stats:
            self._update_stats_label()

    def _check_failures(self, s):
        fail, reason = False, ""
        if s.sim_time > 0.5 and s.sim_phase not in ("Landed", "Terminated"):
            if s.stability_margin < 0.5:
                fail, reason = True, "Instability"
            elif 0 < s.safety_factor < 0.8:
                fail, reason = True, "Structural Failure"
            elif s.mach_number > 3.5:
                fail, reason = True, "Mach Limit Exceeded"
            elif 0 < getattr(s, 'flutter_margin', 99) < 1.0:
                fail, reason = True, "Flutter"

        if fail and not self._failure_active:
            self._failure_active = True
            self._add_timeline_entry(s.sim_time, f"FAILURE: {reason}", "#f85149", events=True)
            self._update_status("FAILURE", "#f85149")
            self._show_failure_sphere(s.x_position, s.y_position, s.altitude)
        elif not fail and self._failure_active:
            self._failure_active = False
            if self._actor_failure and self._plotter:
                try:
                    self._plotter.remove_actor(self._actor_failure)
                except Exception:
                    pass
                self._actor_failure = None

    @pyqtSlot()
    def _on_sim_started(self):
        self._replay_mode = False
        self._replay_timer.stop()
        self._pb_group.setVisible(False)
        self._failure_active = False
        self._tick_counter   = 0
        self._trail_rebuild_counter = 0

        # Clear buffers
        for buf in (self._trail_pts, self._trail_mach, self._trail_alt, self._trail_vel,
                    self._g_time, self._g_alt, self._g_vel, self._g_mach,
                    self._g_accel, self._g_dynq):
            buf.clear()

        self._events_list.clear()
        self._timeline_list.clear()

        # Clear 3D actors
        for attr in ('_actor_trail', '_actor_rocket', '_actor_landing', '_actor_failure'):
            actor = getattr(self, attr, None)
            if actor and self._plotter:
                try:
                    self._plotter.remove_actor(actor)
                except Exception:
                    pass
            setattr(self, attr, None)

        # Reset mission overlays + capture
        self._apogee_xyz   = None
        self._apogee_value = 0.0
        self._drogue_info  = None
        self._main_info    = None
        self._landing_xy   = None
        self._alt_ref_apogee = 0.0
        st = self.engine.state if hasattr(self.engine, "state") else None
        if self._recovery:
            self._recovery.reset()
            if st is not None:
                self._recovery.set_chute_config(
                    getattr(st, "drogue_cd_area", 0.5),
                    getattr(st, "main_cd_area", 3.0),
                )
        if self._alt_ref:
            self._alt_ref.reset()
        if self._envelope:
            target = float(self._spin_target.value())
            recov  = float(self._spin_recov.value())
            self._envelope.build_static(
                launch_xy=(0.0, 0.0), recovery_radius=recov,
                target_apogee=target,
            )
            self._envelope.set_visible(self._show_envelope)

        self._update_status("LIVE", "#7ee787")
        self._add_timeline_entry(0.0, "Simulation started", "#58a6ff")

    @pyqtSlot()
    def _on_sim_finished(self):
        t = self._latest_state.sim_time if self._latest_state else 0.0
        self._update_status("COMPLETE - REPLAY", "#58a6ff")
        self._add_timeline_entry(t, "Simulation complete", "#3fb950")

        # Final full-trail render
        if self._plotter and len(self._trail_pts) >= 2:
            self._rebuild_trail_actor(
                list(self._trail_pts),
                list(self._trail_mach),
                list(self._trail_alt),
                list(self._trail_vel),
            )
            try:
                self._plotter.render()
            except Exception:
                pass

        # Capture landing point if the engine didn't fire a landing event
        if self._landing_xy is None and self._latest_state is not None:
            self._landing_xy = (self._latest_state.x_position,
                                self._latest_state.y_position)

        # Build final altitude reference + flight envelope (persist in replay)
        self._finalize_mission_overlays()
        try:
            self._plotter.render()
        except Exception:
            pass

        self._enter_replay_mode()

    def _on_flight_event(self, data):
        event_name = data.get("event", "")
        if event_name == "phase_change":
            return

        # ── Mission capture (positions grabbed from live state) ──
        s = self._latest_state
        t   = data.get("time", 0.0)
        alt = data.get("altitude", None)
        if event_name == "apogee":
            if s is not None:
                self._apogee_xyz = (s.x_position, s.y_position, s.altitude)
                self._apogee_value = s.altitude
            elif alt is not None:
                self._apogee_xyz = (0.0, 0.0, alt)
                self._apogee_value = alt
            if self._envelope and self._apogee_xyz:
                self._envelope.on_apogee(*self._apogee_xyz, self._apogee_value)
        elif event_name == "drogue_deploy":
            self._drogue_info = (t, alt)
            if self._recovery:
                self._recovery.on_drogue_deploy(t, alt)
        elif event_name == "main_deploy":
            self._main_info = (t, alt)
            if self._recovery:
                self._recovery.on_main_deploy(t, alt)
        elif event_name == "landing":
            if s is not None:
                self._landing_xy = (s.x_position, s.y_position)
            if self._recovery:
                self._recovery.on_landing(t)

        label_color = _EVENT_LABELS.get(event_name)
        if not label_color:
            return
        label, color = label_color
        t   = data.get("time", 0.0)
        alt = data.get("altitude", None)
        text = f"[T+{t:7.2f}s]  {label}"
        if alt is not None:
            text += f"  @ {alt:.0f} m"
        self._add_timeline_entry(t, text, color, events=True)

    def _add_timeline_entry(self, t, text, color="#8b949e", events=False):
        targets = [self._timeline_list]
        if events:
            targets.append(self._events_list)
        for lst in targets:
            item = QListWidgetItem(text)
            item.setForeground(QColor(color))
            item.setData(Qt.ItemDataRole.UserRole, float(t))
            lst.addItem(item)
            lst.scrollToBottom()

    def _on_event_clicked(self, item):
        """Jump the replay to the clicked event's time."""
        t = item.data(Qt.ItemDataRole.UserRole)
        if t is None or not self._replay_mode or not self._replay_times:
            return
        # Nearest recorded frame to the event time
        idx = min(
            range(len(self._replay_times)),
            key=lambda i: abs(self._replay_times[i] - t),
        )
        self._on_replay_pause()
        self._replay_index    = idx
        self._needs_3d_update = True

    def _update_status(self, text, color):
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(
            f"color:{color};font-size:11px;font-weight:600;"
            f"padding:2px 10px;border:1px solid {color}55;border-radius:4px;"
        )

    # ===============================================================
    # CAMERA / TRAIL CONTROLS
    # ===============================================================

    def _on_camera_mode_changed(self, mode):
        self._camera_mode = mode
        self._cam_snap = True   # jump to the new framing, then ease afterwards
        self._needs_3d_update = True

    def _on_trail_mode_changed(self, mode):
        self._trail_color_mode = mode
        if self._actor_trail and self._plotter:
            try:
                self._plotter.remove_actor(self._actor_trail)
            except Exception:
                pass
            self._actor_trail = None
        if self._replay_mode and self._replay_pts:
            # Replay trail is static — rebuild it now in the new colour mode.
            self._rebuild_trail_actor(
                self._replay_pts, self._replay_mach,
                self._replay_alt, self._replay_vel,
            )
            self._safe_render()
        else:
            self._trail_rebuild_counter = 99  # force rebuild next tick
        self._needs_3d_update = True

    def _on_toggle_planes(self, on):
        self._show_alt_planes = on
        if self._alt_ref:
            self._alt_ref.set_planes_visible(on)
        self._safe_render()

    def _on_toggle_scalebar(self, on):
        self._show_scalebar = on
        if self._alt_ref:
            self._alt_ref.set_scalebar_visible(on)
        self._safe_render()

    def _on_toggle_envelope(self, on):
        self._show_envelope = on
        if self._envelope:
            self._envelope.set_visible(on)
        self._safe_render()

    def _apply_quality(self, mode):
        """Adaptive quality: trail resolution, label LOD, render + graph rates."""
        self._quality = mode
        # Label LOD culling disabled — labels/envelope stay visible at any zoom.
        self._label_lod = False
        if mode == "Performance":
            self._trail_cap = 800
            render_int, graph_int, aa = 40, 500, False
        elif mode == "Quality":
            self._trail_cap = 5000
            render_int, graph_int, aa = 16, 150, True   # 60 fps target
        else:  # Balanced
            self._trail_cap = 2000
            render_int, graph_int, aa = 33, 300, True   # 30 fps
        if hasattr(self, "_render_timer"):
            self._render_timer.setInterval(render_int)
        if hasattr(self, "_graph_timer"):
            self._graph_timer.setInterval(graph_int)
        if self._plotter:
            try:
                if aa:
                    self._plotter.enable_anti_aliasing()
                else:
                    self._plotter.disable_anti_aliasing()
            except Exception:
                pass
        if not self._label_lod and self._labels_hidden:
            self._set_overlay_labels(True)
            self._labels_hidden = False
        self._needs_3d_update = True

    def _on_toggle_stats(self, on):
        self._show_stats = on
        self._stats_lbl.setVisible(on)
        if not on:
            self._stats_lbl.setText("")

    def _update_stats_label(self):
        self._stats_lbl.setText(
            f"FPS {self._fps:4.0f} | frame {self._frame_ms:4.1f}ms | "
            f"render {self._render_ms:4.1f}ms | panel {self._panel_ms:3.1f}ms | "
            f"graph {self._graph_ms:3.1f}ms"
        )

    def _on_envelope_config_changed(self):
        """Target apogee / recovery radius changed — rebuild static envelope."""
        if not self._envelope:
            return
        target = float(self._spin_target.value())
        recov  = float(self._spin_recov.value())
        apogee = self._apogee_value or self._alt_ref_apogee
        self._envelope.build_static(
            launch_xy=(0.0, 0.0), recovery_radius=recov,
            target_apogee=target, extent_hint=max(apogee, recov * 2.5),
        )
        # Re-apply any captured mission markers on top of the rebuilt base
        self._rebuild_envelope_markers()
        self._envelope.set_visible(self._show_envelope)
        self._safe_render()

    def _safe_render(self):
        if self._plotter:
            try:
                self._plotter.render()
            except Exception:
                pass

    # ===============================================================
    # MISSION OVERLAY LOGIC (recovery / altitude / envelope)
    # ===============================================================

    def _wind_params(self):
        """(wind_speed, wind_direction_deg) from current state."""
        s = self._latest_state
        if s is None and hasattr(self.engine, "state"):
            s = self.engine.state
        ws = getattr(s, "wind_speed", 0.0) if s else 0.0
        wd = getattr(s, "wind_direction", 0.0) if s else 0.0
        return ws, wd

    def _terminal_descent(self):
        """Estimate main-chute terminal descent rate (m/s) for landing prediction."""
        s = self._latest_state
        if s is None and hasattr(self.engine, "state"):
            s = self.engine.state
        if s is None:
            return 5.0
        m = max(0.1, s.dry_mass + getattr(s, "propellant_mass", 0.0))
        cd_area = max(0.1, getattr(s, "main_cd_area", 1.5))
        v = math.sqrt(2.0 * m * 9.81 / (1.225 * cd_area))
        return max(1.0, min(v, 60.0))

    def _ensure_alt_reference(self, apogee, x_landing=0.0, force=False):
        """(Re)build altitude reference, growing it as the rocket climbs."""
        if not self._alt_ref or apogee <= 0:
            return
        # Grow coarsely (2x steps) so planes are rebuilt only a handful of
        # times during ascent, not every frame the rocket climbs.
        if not force and apogee <= self._alt_ref_apogee * 2.0:
            return
        self._alt_ref_apogee = apogee
        try:
            self._alt_ref.build(apogee, x_center=0.0, y_center=0.0,
                                x_landing=x_landing)
            self._alt_ref.set_planes_visible(self._show_alt_planes)
            self._alt_ref.set_scalebar_visible(self._show_scalebar)
        except Exception as exc:
            logger.debug(f"alt reference build failed: {exc}")

    def _rebuild_envelope_markers(self):
        """Re-apply captured apogee + landing markers onto the static envelope."""
        if not self._envelope:
            return
        if self._apogee_xyz:
            self._envelope.on_apogee(*self._apogee_xyz, self._apogee_value)
            ws, wd = self._wind_params()
            dr = self._terminal_descent()
            px, py, drift = estimate_landing(
                self._apogee_xyz[0], self._apogee_xyz[1],
                self._apogee_value, ws, wd, dr,
            )
            recov = float(self._spin_recov.value())
            a = max(recov * 0.2, drift * 0.25, 50.0)
            b = a * 0.6
            self._envelope.set_predicted_landing(
                px, py, apogee_xyz=self._apogee_xyz,
                ellipse_axes=(a, b), wind_dir_deg=wd,
            )
        if self._landing_xy:
            self._envelope.set_actual_landing(*self._landing_xy)

    def _finalize_mission_overlays(self):
        """Build final altitude reference + envelope when the flight completes."""
        # Determine apogee + landing downrange
        apogee = self._apogee_value
        if apogee <= 0:
            try:
                apogee = max(self._trail_alt) if self._trail_alt else 0.0
            except Exception:
                apogee = 0.0
        x_landing = self._landing_xy[0] if self._landing_xy else 0.0

        self._ensure_alt_reference(apogee, x_landing=x_landing, force=True)

        if self._envelope:
            target = float(self._spin_target.value())
            recov  = float(self._spin_recov.value())
            self._envelope.build_static(
                launch_xy=(0.0, 0.0), recovery_radius=recov,
                target_apogee=target, extent_hint=max(apogee, recov * 2.5),
            )
            self._rebuild_envelope_markers()
            self._envelope.set_visible(self._show_envelope)

    def _update_recovery(self, x, y, z, sim_time, descent_rate):
        """Drive the parachute visualizer + recovery readouts for one frame."""
        if not self._recovery:
            return
        self._recovery.update(
            x, y, z, sim_time,
            self._rocket_length * self._VIS_SCALE,
            descent_rate=descent_rate,
        )
        tel = self._recovery.telemetry(sim_time)
        st = tel["state"]
        col = _PHASE_COLORS.get(
            {"Drogue Descending": "Drogue Descent",
             "Drogue Deploying": "Drogue Descent",
             "Main Descending": "Main Descent",
             "Main Deploying": "Main Descent",
             "Landed": "Landed"}.get(st, ""), "#8b949e",
        )
        self._rd_rec_state.set_value(st, color=col)
        self._rd_rec_area.set_value(tel["canopy_area"], "{:.2f}")
        self._rd_rec_descent.set_value(tel["descent_rate"], "{:.1f}")
        if tel["deploy_altitude"] is not None:
            self._rd_rec_depalt.set_value(tel["deploy_altitude"], "{:.0f}")
        else:
            self._rd_rec_depalt.set_value("--")
        if tel["time_since_deploy"] is not None:
            self._rd_rec_since.set_value(tel["time_since_deploy"], "{:.1f}")
        else:
            self._rd_rec_since.set_value("--")

    def _fit_scene(self):
        """Frame the entire scene (trajectory + envelope + altitude scale) and
        release the follow-camera so the user can orbit/zoom freely."""
        if not self._plotter:
            return
        self._cam_target_pos = None
        try:
            self._plotter.reset_camera()
            self._plotter.reset_camera_clipping_range()
            self._plotter.render()
        except Exception:
            pass

    def _reset_camera(self):
        if not self._plotter:
            return
        # In replay, "Reset Camera" frames the whole scene for inspection.
        if self._replay_mode and not self._replay_is_playing:
            self._fit_scene()
            return
        self._cam_snap = True
        s = self._latest_state
        if s is not None:
            self._apply_camera(s.x_position, s.y_position, s.altitude,
                               getattr(s, 'pitch', math.pi / 2))
            self._ease_camera()
        else:
            self._plotter.camera.position    = (300, -350, 180)
            self._plotter.camera.focal_point = (0, 0, 120)
            self._plotter.camera.up = (0, 0, 1)
        try:
            self._plotter.render()
        except Exception:
            pass

    # ===============================================================
    # REPLAY MODE
    # ===============================================================

    def _enter_replay_mode(self):
        if not self._trail_pts:
            return
        self._replay_pts  = list(self._trail_pts)
        self._replay_mach = list(self._trail_mach)
        self._replay_alt  = list(self._trail_alt)
        self._replay_vel  = list(self._trail_vel)

        try:
            self._replay_times = self.sim_engine.history.get_values("time") or []
        except Exception:
            self._replay_times = []

        if not self._replay_pts:
            return

        n = len(self._replay_pts)
        self._replay_index = n - 1

        # Build the full trajectory trail ONCE — static for the whole replay.
        if n >= 2:
            self._rebuild_trail_actor(
                self._replay_pts, self._replay_mach,
                self._replay_alt, self._replay_vel,
            )

        self._replay_slider.setRange(0, n - 1)
        self._replay_slider.setValue(n - 1)
        self._replay_mode = True
        self._pb_group.setVisible(True)
        self._update_status("REPLAY", "#d29922")
        # Place the rocket at touchdown, then frame the whole scene so the user
        # can immediately see the full trajectory + envelope + altitude scale.
        self._needs_3d_update = False
        self._render_replay_scene()
        self._fit_scene()

    def _render_replay_scene(self):
        """Per-frame replay update. The full trajectory trail is a STATIC actor
        built once on replay entry — here we only move the rocket + camera, so
        scrubbing is instant regardless of flight length."""
        if not self._replay_pts or not self._plotter:
            return

        idx = max(0, min(self._replay_index, len(self._replay_pts) - 1))
        x, y, z = self._replay_pts[idx]

        self._update_rocket_transform(x, y, z, math.pi / 2)

        # Follow the rocket only while actively playing. When paused/seeking the
        # camera is released so the user can freely orbit + zoom out to inspect
        # the full envelope and altitude scale.
        if self._camera_mode != "Free" and self._replay_is_playing:
            self._apply_camera(x, y, z, math.pi / 2)

        # Time label + recovery animation synced to the replay frame
        frame_t = 0.0
        if self._replay_times and idx < len(self._replay_times):
            frame_t = self._replay_times[idx]
        self._replay_time_lbl.setText(f"T+{frame_t:.2f}s")
        descent = self._replay_vel[idx] if idx < len(self._replay_vel) else 0.0
        self._update_recovery(x, y, z, frame_t, descent)

        self._replay_slider.blockSignals(True)
        self._replay_slider.setValue(idx)
        self._replay_slider.blockSignals(False)

    def _on_replay_tick(self):
        if not self._replay_is_playing:
            return
        n = len(self._replay_pts)
        if self._replay_index >= n - 1:
            self._replay_is_playing = False
            self._replay_timer.stop()
            return
        self._replay_index = min(self._replay_index + 5, n - 1)
        self._needs_3d_update = True

    def _on_replay_play(self):
        if not self._replay_mode:
            return
        if self._replay_index >= len(self._replay_pts) - 1:
            self._replay_index = 0
        self._replay_is_playing = True
        self._cam_snap = True   # re-grab the rocket when playback resumes
        self._replay_timer.start()

    def _on_replay_pause(self):
        self._replay_is_playing = False
        self._replay_timer.stop()
        self._cam_target_pos = None   # release camera for free inspection

    def _on_replay_restart(self):
        self._replay_is_playing = False
        self._replay_timer.stop()
        self._replay_index = 0
        self._needs_3d_update = True

    def _on_replay_seek(self, value):
        self._replay_index    = value
        self._needs_3d_update = True

    # ===============================================================
    # SHUTDOWN
    # ===============================================================

    def shutdown(self):
        """Stop timers and tear down the VTK render window cleanly.

        Without this, the render timer keeps calling plotter.render() after the
        OpenGL context is destroyed, flooding the log with wglMakeCurrent errors.
        """
        if self._closing:
            return
        self._closing = True
        for tmr in ('_render_timer', '_graph_timer', '_replay_timer', '_panel_timer'):
            t = getattr(self, tmr, None)
            if t is not None:
                try:
                    t.stop()
                except Exception:
                    pass
        # VTK emits benign wglMakeCurrent errors while destroying the GL context
        # on Windows. Mute its global warning display before teardown to keep the
        # crash log clean.
        try:
            import vtk
            vtk.vtkObject.GlobalWarningDisplayOff()
        except Exception:
            pass
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
            self._plotter = None

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    # ===============================================================
    # SIGNAL SUBSCRIPTIONS
    # ===============================================================

    def _subscribe_signals(self):
        self.engine.telemetry_tick.connect(self._on_telemetry_tick)
        self.sim_engine.sim_started.connect(self._on_sim_started)
        self.sim_engine.sim_finished.connect(self._on_sim_finished)
        em = self.sim_engine.event_mgr
        for event in SimEvent:
            em.subscribe(event, self._on_flight_event)
