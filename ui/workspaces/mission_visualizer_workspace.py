"""
K2 AeroSim - Mission Visualizer Workspace
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
    from visualization.mission.flight_effects import (
        FlightEffects, ForceVectors, EventFlags)
    _OVERLAYS_OK = True
except Exception:
    _OVERLAYS_OK = False

try:
    from visualization.mission.rocket_mesh import build_rocket_mesh
    _ROCKET_MESH_OK = True
except Exception:
    _ROCKET_MESH_OK = False

from ui import theme
from ui.replay_clock import clock_for
from ui.widgets.replay_bar import ReplayBar
from ui.widgets.viewport_gate import gate_hidden_rendering

logger = logging.getLogger("K2.MissionViz")

_PHASE_COLORS = {
    "Pre-Launch":     theme.ACCENT,
    "Boost":          theme.ACCENT_HOVER,
    "Coast":          theme.OK,
    "Apogee":         theme.ACCENT,
    "Drogue Descent": theme.ACCENT_HOVER,
    "Main Descent":   theme.OK,
    "Landed":         theme.OK,
    "Timeout":        theme.ERR,
    "Terminated":     theme.ERR,
}

# Categorical phase coloring for the "Flight Phase" trail mode
_PHASE_INDEX = {
    "Pre-Launch": 0, "Ignition": 1, "Boost": 1, "Coast": 2, "Apogee": 3,
    "Drogue Descent": 4, "Main Descent": 5, "Landed": 6,
}
_PHASE_CMAP = [theme.ACCENT, theme.ACCENT_HOVER, theme.OK, theme.ACCENT,
               theme.ACCENT_HOVER, theme.OK, theme.OK]

# Events that get a 3D flag marker on the trajectory
_FLAG_EVENTS = {
    "motor_burnout": ("Burnout",  theme.ACCENT_HOVER),
    "max_q":         ("Max-Q",    theme.WARN),
    "apogee":        ("Apogee",   theme.ACCENT),
    "drogue_deploy": ("Drogue",   theme.ACCENT_HOVER),
    "main_deploy":   ("Main",     theme.OK),
    "landing":       ("Landing",  theme.OK),
}

_EVENT_LABELS = {
    "sim_start":      ("Ignition",            theme.ACCENT),
    "motor_ignition": ("Motor Ignition",      theme.ACCENT_HOVER),
    "motor_burnout":  ("Burnout",             theme.ACCENT_HOVER),
    "apogee":         ("Apogee",              theme.ACCENT),
    "drogue_deploy":  ("Drogue Deploy",       theme.ACCENT_HOVER),
    "main_deploy":    ("Main Chute Deploy",   theme.OK),
    "landing":        ("Landing",             theme.OK),
    "max_q":          ("Max-Q",               theme.WARN),
    "sim_end":        ("Sim End",             theme.OK),
}


class _Readout(QWidget):
    """Compact telemetry readout: label + value + unit."""

    def __init__(self, label, unit="", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(9, 5, 9, 5)
        lay.setSpacing(1)

        # Caption, value and unit left-aligned in one bordered tile. The frame
        # style had no selector, so the border was drawn round the caption,
        # the value and the unit as three separate boxes.
        self.setObjectName("mvReadout")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        cap = label if not unit else f"{label} ({unit})"
        lbl = QLabel(cap)
        lbl.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;background:transparent;border:none;")

        self._val = QLabel("--")
        self._val.setStyleSheet(self._value_qss(theme.TEXT_BRIGHT))

        lay.addWidget(lbl)
        lay.addWidget(self._val)

        self.setStyleSheet(
            f"QWidget#mvReadout {{ background:{theme.PANEL}; border:1px solid {theme.LINE}; "
            "border-radius:3px; }")
        self.setMinimumWidth(85)

        self._last_text = None
        self._last_col = None

    def set_value(self, v, fmt="{:.1f}", color=None):
        text = v if isinstance(v, str) else fmt.format(v)
        if text != self._last_text:
            self._val.setText(text)
            self._last_text = text
        col = color or theme.TEXT_BRIGHT
        # setStyleSheet reparses CSS — only touch it when the colour changes.
        if col != self._last_col:
            self._val.setStyleSheet(self._value_qss(col))
            self._last_col = col

    @staticmethod
    def _value_qss(color):
        return (f"color:{color};font-family:{theme.MONO};font-size:14px;"
                "font-weight:600;background:transparent;border:none;")


class MissionVisualizerWorkspace(QWidget):
    """
    Live 3D mission visualizer. Renders the rocket flying in real time as the
    simulation runs, then enters replay mode when the flight finishes.
    """

    _CAMERA_MODES = ["Chase", "Launch Pad", "Side View", "FPV", "Onboard",
                     "Orbit", "Free", "Recovery"]
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
        self._trail_pts    = deque(maxlen=100_000)
        self._trail_mach   = deque(maxlen=100_000)
        self._trail_alt    = deque(maxlen=100_000)
        self._trail_vel    = deque(maxlen=100_000)
        self._trail_pitch  = deque(maxlen=100_000)
        self._trail_yaw    = deque(maxlen=100_000)
        self._trail_thrust = deque(maxlen=100_000)
        self._trail_drag   = deque(maxlen=100_000)
        self._trail_phase  = deque(maxlen=100_000)   # phase index (ints)

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
        # The flight being replayed and the instant shown are the shared
        # clock's (ui.replay_clock), so the Results tab follows the same one.
        self._clock = clock_for(sim_engine)
        self._replay = None             # core.flight_replay.FlightReplay
        self._replay_mode       = False
        self._replay_index      = 0
        self._replay_shown_t    = None  # instant the panel last drew
        # Loads overlay (core.load_overlay): off unless the user picks one.
        self._loads_profile = None      # LoadProfile being painted
        self._loads_mesh    = None      # the rocket mesh carrying the "load" scalars
        self._actor_loads   = None
        self._loads_stations = None     # each mesh point's distance from the nose
        self._rocket_mesh   = None      # mesh the rocket actor was built from
        self._replay_pts    = []
        self._replay_mach   = []
        self._replay_alt    = []
        self._replay_vel    = []
        self._replay_times  = []
        self._replay_phase  = []

        # Scene overlays (recovery / altitude reference / flight envelope)
        self._recovery   = None
        self._alt_ref    = None
        self._envelope   = None
        self._effects    = None
        self._vectors    = None
        self._flags      = None
        self._hud_actor  = None
        self._show_alt_planes = True
        self._show_scalebar   = False
        self._show_envelope   = True
        self._show_effects    = True
        self._show_vectors    = False
        self._show_flags      = True
        self._show_hud        = True
        self._orbit_t0        = perf_counter()   # wall-clock base for Orbit cam
        self._max_thrust_seen = 1.0              # for thrust_frac normalization

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

        # Replay: the shared clock drives the instant shown
        self._clock.time_changed.connect(self._on_clock_time)
        self._clock.playing_changed.connect(self._on_clock_playing)
        self._clock.flight_changed.connect(self._on_clock_flight)

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
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {theme.RAISED}; }}")
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([920, 280])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        root.addWidget(self._build_replay_row())
        root.addWidget(self._build_controls_bar())

    def _build_title_bar(self):
        bar = QWidget()
        bar.setFixedHeight(36)
        bar.setStyleSheet(f"background:{theme.BG};border-bottom:1px solid {theme.RAISED};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 0, 16, 0)

        title = QLabel("Mission Visualizer")
        title.setStyleSheet(theme.panel_title_qss())
        lay.addWidget(title)
        lay.addStretch()

        self._stats_lbl = QLabel("")
        self._stats_lbl.setStyleSheet(
            f"color:{theme.OK};font-family:'Cascadia Code',monospace;font-size:10px;"
            "padding:2px 8px;margin-right:8px;"
        )
        self._stats_lbl.setVisible(False)
        lay.addWidget(self._stats_lbl)

        self._status_lbl = QLabel("STANDBY")
        self._status_lbl.setStyleSheet(
            f"color:{theme.LINE_STRONG};font-size:11px;font-weight:600;"
            f"padding:2px 10px;border:1px solid {theme.RAISED};border-radius:4px;"
        )
        lay.addWidget(self._status_lbl)

        self._phase_lbl = QLabel("--")
        self._phase_lbl.setStyleSheet(
            f"color:{theme.TEXT_DIM};font-size:11px;font-weight:600;"
            f"padding:2px 10px;border:1px solid {theme.RAISED};border-radius:4px;margin-left:8px;"
        )
        lay.addWidget(self._phase_lbl)
        return bar

    def _build_left_panel(self):
        w = QWidget()
        w.setStyleSheet(f"background:{theme.BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 3D view
        view_frame = QFrame()
        view_frame.setStyleSheet(f"background:{theme.BG};")
        vfl = QVBoxLayout(view_frame)
        vfl.setContentsMargins(0, 0, 0, 0)

        if _PYVISTA_OK:
            try:
                self._plotter = QtInteractor(view_frame)
                vfl.addWidget(self._plotter.interactor)
                # The workspace's own render tick already skips a hidden tab;
                # pyvistaqt's 5 Hz auto-update did not.
                gate_hidden_rendering(self._plotter)
                self._plotter.set_background(theme.BG)
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
        lbl.setStyleSheet(f"color:{theme.LINE_STRONG};font-size:12px;")
        return lbl

    def _build_graph_strip(self):
        frame = QFrame()
        frame.setFixedHeight(125)
        frame.setStyleSheet(f"background:{theme.BG};border-top:1px solid {theme.RAISED};")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(6)

        self._graph_axes     = []
        self._graph_lines    = []
        self._graph_canvases = []
        self._graph_cursors  = []   # the replay instant, drawn on each graph

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
                fig.patch.set_facecolor(theme.BG)
                ax = fig.add_subplot(111)
                ax.set_facecolor(theme.BG)
                ax.tick_params(colors=theme.LINE_STRONG, labelsize=6)
                for sp in ax.spines.values():
                    sp.set_edgecolor(theme.RAISED)
                title_str = f"{label} ({unit})" if unit else label
                ax.set_title(title_str, color=theme.TEXT_DIM, fontsize=7, pad=2)
                fig.subplots_adjust(left=0.18, right=0.97, top=0.78, bottom=0.22)
                line, = ax.plot([], [], color=theme.ACCENT, linewidth=1.0)
                cursor = ax.axvline(0.0, color=theme.TEXT_BRIGHT, linewidth=0.8,
                                    visible=False)
                self._graph_cursors.append(cursor)
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
        tabs.addTab(self._build_telemetry_tab(), "Telemetry")
        tabs.addTab(self._build_events_tab(),    "Flight Events")
        tabs.addTab(self._build_timeline_tab(),  "Timeline")
        return tabs

    def _build_telemetry_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background:{theme.BG};")
        inner = QWidget()
        inner.setStyleSheet(f"background:{theme.BG};")
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
        hdr = QLabel("Recovery")
        hdr.setStyleSheet(
            f"color:{theme.TEXT_DIM};font-size:11px;font-weight:600;"
            "padding:8px 2px 2px 2px;"
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
        w.setStyleSheet(f"background:{theme.BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._events_list = QListWidget()
        self._events_list.setStyleSheet(f"""
            QListWidget {{ background:{theme.BG}; border:none; color:{theme.TEXT_BRIGHT}; font-size:11px; }}
            QListWidget::item {{ padding:5px 6px; border-bottom:1px solid {theme.PANEL}; }}
            QListWidget::item:selected {{ background:{theme.PANEL}; color:{theme.ACCENT}; }}
        """)
        self._events_list.itemClicked.connect(self._on_event_clicked)
        lay.addWidget(self._events_list)

        hint = QLabel("Click an event to jump replay")
        hint.setStyleSheet(f"color:{theme.LINE_STRONG};font-size:9px;padding:4px 2px;")
        lay.addWidget(hint)
        return w

    def _build_timeline_tab(self):
        w = QWidget()
        w.setStyleSheet(f"background:{theme.BG};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        self._timeline_list = QListWidget()
        self._timeline_list.setStyleSheet(f"""
            QListWidget {{
                background:{theme.BG}; border:none; color:{theme.TEXT_BRIGHT};
                font-family:'Cascadia Code',monospace; font-size:10px;
            }}
            QListWidget::item {{ padding:4px 6px; border-bottom:1px solid {theme.PANEL}; }}
            QListWidget::item:selected {{ background:{theme.PANEL}; color:{theme.ACCENT}; }}
        """)
        lay.addWidget(self._timeline_list)
        return w

    def _build_controls_bar(self):
        bar = QWidget()
        bar.setFixedHeight(50)
        bar.setStyleSheet(f"background:{theme.PANEL};border-top:1px solid {theme.RAISED};")
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
        sep0.setStyleSheet(f"color:{theme.LINE};")
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

        self._chk_effects = self._styled_check("Effects", True)
        self._chk_effects.toggled.connect(self._on_toggle_effects)
        lay.addWidget(self._chk_effects)

        self._chk_vectors = self._styled_check("Vectors", False)
        self._chk_vectors.toggled.connect(self._on_toggle_vectors)
        lay.addWidget(self._chk_vectors)

        self._chk_flags = self._styled_check("Flags", True)
        self._chk_flags.toggled.connect(self._on_toggle_flags)
        lay.addWidget(self._chk_flags)

        self._chk_hud = self._styled_check("HUD", True)
        self._chk_hud.toggled.connect(self._on_toggle_hud)
        lay.addWidget(self._chk_hud)

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

        lay.addStretch()

        reset_btn = QPushButton("Reset Camera")
        reset_btn.setStyleSheet(
            f"QPushButton{{background:{theme.RAISED};color:{theme.TEXT_DIM};border:1px solid {theme.LINE};"
            "border-radius:4px;padding:4px 10px;font-size:11px;}"
            f"QPushButton:hover{{background:{theme.LINE};color:{theme.TEXT_BRIGHT};}}"
        )
        reset_btn.clicked.connect(self._reset_camera)
        lay.addWidget(reset_btn)
        return bar

    def _build_replay_row(self):
        """Transport bar for the finished flight (hidden until there is one):
        restart / play / speed, the flight timeline with its events, and the
        key to the force arrows when those are shown."""
        self._pb_group = QWidget()
        self._pb_group.setObjectName("mvReplayRow")
        self._pb_group.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._pb_group.setStyleSheet(
            f"QWidget#mvReplayRow {{ background:{theme.PANEL}; "
            f"border-top:1px solid {theme.RAISED}; }}")
        lay = QHBoxLayout(self._pb_group)
        lay.setContentsMargins(12, 6, 12, 4)
        lay.setSpacing(12)
        self._replay_bar = ReplayBar(self._clock)
        lay.addWidget(self._replay_bar, 1)
        self._vector_key = QLabel()
        self._vector_key.setTextFormat(Qt.TextFormat.RichText)
        self._vector_key.setStyleSheet("font-size:11px; background:transparent;")
        self._vector_key.setVisible(False)
        lay.addWidget(self._vector_key)

        lay.addWidget(self._styled_label("Loads:"))
        self._loads_combo = self._styled_combo(list(self._LOADS_CHOICES))
        self._loads_combo.setToolTip(
            "Paint a solved result on the rocket: CFD surface pressure scaled "
            "to the dynamic pressure of the instant shown, or the FE static "
            "stress. Off by default; needs that analysis to have been run.")
        self._loads_combo.currentTextChanged.connect(self._on_loads_changed)
        lay.addWidget(self._loads_combo)
        self._loads_note = QLabel()
        self._loads_note.setStyleSheet(
            f"color:{theme.TEXT_DIM}; font-size:11px; background:transparent;")
        lay.addWidget(self._loads_note)
        self._pb_group.setVisible(False)
        return self._pb_group

    # ── Loads overlay ──
    _LOADS_CHOICES = {"Off": None, "CFD Pressure": "cfd_pressure", "FE Stress": "fe_stress"}

    def _load_profile_for(self, kind):
        """(LoadProfile or None, note) for the chosen overlay. The note says
        why there is nothing to draw, or that the result is out of date."""
        from core import load_overlay as LO
        from core import readiness as R
        results = getattr(self.engine, "results", {})
        if kind == LO.CFD_PRESSURE:
            res = results.get("cfd")
            stage = "cfd"
            profile = (LO.cfd_pressure_profile(getattr(res, "surface_vtk", None),
                                               getattr(res, "mach", 0.0))
                       if res is not None else None)
            missing = "Run CFD first"
        else:
            res = results.get("structures")
            stage = "structures"
            profile = (LO.fe_stress_profile(res["fem"], res.get("condition", ""))
                       if res else None)
            missing = "Run a static FE analysis first"
        if profile is None:
            return None, missing
        note = profile.source
        try:
            st = self.engine.readiness.status(stage, self.engine.state)
            if st.status == R.STALE:
                note += f" — out of date ({st.reason})"
        except Exception:
            pass
        return profile, note

    def _on_loads_changed(self, text):
        self._remove_loads_actor()
        kind = self._LOADS_CHOICES.get(text)
        if kind is None:
            self._loads_note.setText("")
        else:
            profile, note = self._load_profile_for(kind)
            self._loads_note.setText(note)
            if profile is not None:
                self._build_loads_actor(profile)
        self._needs_3d_update = True
        self._safe_render()

    def _remove_loads_actor(self):
        if self._actor_loads is not None and self._plotter:
            try:
                self._plotter.remove_actor(self._actor_loads)
                self._plotter.remove_scalar_bar("Loads")
            except Exception:
                pass
        self._actor_loads = self._loads_mesh = self._loads_profile = None
        if self._actor_rocket is not None:
            self._actor_rocket.SetVisibility(True)

    def _build_loads_actor(self, profile):
        """A copy of the rocket, coloured by *profile*, drawn in place of the
        plain one while the overlay is on."""
        self._ensure_rocket_actor()
        if not self._plotter or self._rocket_mesh is None:
            return
        mesh = self._rocket_mesh.copy(deep=True)
        # The rocket is modelled tail at z = 0, nose at z = length.
        length = float(mesh.points[:, 2].max())
        self._loads_stations = length - mesh.points[:, 2]
        q_max = 1.0
        if self._replay is not None:
            q_max = float(np.nanmax(self._replay.series("dynamic_pressure")))
        lo, hi = profile.limits(q_max)
        pressure = profile.scales_with_q
        scale, unit = (1e-3, "kPa") if pressure else (1e-6, "MPa")
        self._loads_scale = scale
        mesh["load"] = profile.at(self._loads_stations, q_max if pressure else 1.0) * scale
        try:
            self._actor_loads = self._plotter.add_mesh(
                mesh, scalars="load", name="rocket_loads",
                cmap="coolwarm" if pressure else "inferno",
                clim=(lo * scale, hi * scale), smooth_shading=True,
                ambient=0.5, diffuse=0.6, show_scalar_bar=True,
                scalar_bar_args=dict(
                    title="Loads", n_labels=5, fmt="%.3g", color=theme.TEXT,
                    vertical=True, position_x=0.02, position_y=0.25,
                    width=0.05, height=0.5, title_font_size=12,
                    label_font_size=10))
        except Exception as exc:
            logger.warning(f"Loads overlay failed: {exc}")
            self._actor_loads = None
            return
        self._loads_mesh, self._loads_profile = mesh, profile
        self._loads_note.setText(
            self._loads_note.text() + "  |  "
            + ("gauge pressure" if pressure else "von Mises") + f", {unit}")
        if self._actor_rocket is not None:
            self._actor_rocket.SetVisibility(False)
        if getattr(self, "_last_pose", None):
            self._update_rocket_transform(*self._last_pose)

    def _update_loads(self, q):
        """Re-colour the overlay for the dynamic pressure *q* (Pa) of the
        instant shown. FE stress is one load case and does not change."""
        if self._loads_mesh is None or not self._loads_profile.scales_with_q:
            return
        self._loads_mesh["load"] = (self._loads_profile.at(self._loads_stations, q)
                                    * self._loads_scale)

    def _vector_key_html(self):
        """Name of each arrow and marker, in its own colour."""
        if not _OVERLAYS_OK:
            return ""
        names = (("thrust", "Thrust"), ("drag", "Drag"), ("lift", "Lift"),
                 ("weight", "Weight"), ("velocity", "Velocity"), ("wind", "Wind"),
                 ("cg", "CG"), ("cp", "CP"))
        return "&nbsp;&nbsp;".join(
            f'<span style="color:{ForceVectors.COLORS[k]}; font-weight:600;">{label}</span>'
            for k, label in names)

    def _styled_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;")
        return lbl

    def _styled_combo(self, items):
        cb = QComboBox()
        cb.addItems(items)
        cb.setStyleSheet(
            f"QComboBox{{background:{theme.BG};color:{theme.TEXT_BRIGHT};border:1px solid {theme.LINE};"
            "border-radius:4px;padding:3px 8px;font-size:11px;min-width:100px;}"
            "QComboBox::drop-down{border:none;}"
            f"QComboBox QAbstractItemView{{background:{theme.PANEL};color:{theme.TEXT_BRIGHT};"
            f"selection-background-color:{theme.RAISED};}}"
        )
        return cb

    def _styled_check(self, text, checked):
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setStyleSheet(
            f"QCheckBox{{color:{theme.TEXT_DIM};font-size:11px;spacing:4px;}}"
            f"QCheckBox::indicator{{width:12px;height:12px;border:1px solid {theme.LINE};"
            f"border-radius:3px;background:{theme.BG};}}"
            f"QCheckBox::indicator:checked{{background:{theme.ACCENT};border-color:{theme.ACCENT};}}"
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
            f"QDoubleSpinBox{{background:{theme.BG};color:{theme.TEXT_BRIGHT};border:1px solid {theme.LINE};"
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
        p.add_mesh(rail, color=theme.LINE_STRONG, name="rail", smooth_shading=True)

        # Launch pad
        pad = pv.Disc(center=(0, 0, 0.5), normal=(0, 0, 1),
                      inner=0.0, outer=12.0, r_res=2, c_res=24)
        p.add_mesh(pad, color=theme.RAISED, name="pad")

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
                self._effects = FlightEffects(p, vis_scale=self._VIS_SCALE)
                self._vectors = ForceVectors(p, vis_scale=self._VIS_SCALE)
                self._flags   = EventFlags(p)
                self._vectors.set_visible(self._show_vectors)
            except Exception as exc:
                logger.warning(f"Effects init failed: {exc}")
                self._effects = self._vectors = self._flags = None

        # In-view HUD (alpha / Mach / q) — one text actor, updated at 10 Hz
        try:
            self._hud_actor = p.add_text(
                "", position="upper_left", font_size=9, color=theme.OK,
                name="hud", font="courier")
            self._hud_actor.SetVisibility(self._show_hud)
        except Exception:
            self._hud_actor = None

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
        via cheap VTK actor transforms — no geometry regen each frame.

        Uses the actual design geometry (nose shape / tubes / transitions /
        fins / nozzle, vertex-colored) when an assembly exists on the engine;
        falls back to the generic cylinder+cone otherwise."""
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

        self._rocket_is_assembly = False
        asm = getattr(self.engine, "_assembly", None)
        if _ROCKET_MESH_OK and asm is not None:
            try:
                built = build_rocket_mesh(asm)
            except Exception as exc:
                logger.debug(f"assembly rocket mesh failed: {exc}")
                built = None
            if built is not None:
                mesh, total_len, _ = built
                self._rocket_mesh = mesh
                try:
                    self._actor_rocket = self._plotter.add_mesh(
                        mesh, scalars="rgb", rgb=True, name="rocket",
                        smooth_shading=True, ambient=0.35, diffuse=0.65,
                        show_scalar_bar=False,
                    )
                    self._rocket_dims_key = key
                    self._rocket_is_assembly = True
                    self._rocket_color = None
                    return
                except Exception as exc:
                    logger.debug(f"assembly rocket actor error: {exc}")
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
        self._rocket_mesh = rocket
        try:
            self._actor_rocket = self._plotter.add_mesh(
                rocket, color=theme.ACCENT, name="rocket",
                smooth_shading=True, ambient=0.35, diffuse=0.65
            )
            self._rocket_dims_key = key
            self._rocket_color = None
        except Exception as exc:
            logger.debug(f"Rocket build error: {exc}")

    def _update_rocket_transform(self, x, y, z, pitch, yaw=0.0):
        """Cheap per-frame pose update: scale + tilt + yaw + position + colour."""
        self._ensure_rocket_actor()
        a = self._actor_rocket
        if a is None:
            return
        tilt_deg = math.degrees(math.pi / 2.0 - max(0.0, min(math.pi, pitch)))
        # VTK applies orientation as Rz·Rx·Ry: body tilts about Y (pitch),
        # then the tilted axis swings about world Z (yaw) — matches the
        # engine's thrust direction (cosθcosψ, cosθsinψ, sinθ).
        yaw_deg = math.degrees(yaw)
        try:
            if self._actor_loads is not None:
                self._actor_loads.SetScale(self._VIS_SCALE)
                self._actor_loads.SetOrientation(0.0, tilt_deg, yaw_deg)
                self._actor_loads.SetPosition(x, y, z)
            a.SetScale(self._VIS_SCALE)
            a.SetOrientation(0.0, tilt_deg, yaw_deg)
            a.SetPosition(x, y, z)
            # Vertex-colored assembly mesh ignores the actor property color
            # (failure is still indicated by the red failure sphere).
            if not getattr(self, "_rocket_is_assembly", False):
                color = (0.973, 0.318, 0.286) if self._failure_active else (0.345, 0.651, 1.0)
                if color != getattr(self, "_rocket_color", None):
                    a.GetProperty().SetColor(*color)
                    self._rocket_color = color
        except Exception as exc:
            logger.debug(f"Rocket transform error: {exc}")

    def _rebuild_trail_actor(self, pts, mach_arr, alt_arr, vel_arr,
                             phase_arr=None):
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
        phase_np = None
        if phase_arr is not None and len(phase_arr) == len(pts):
            try:
                phase_np = np.asarray(phase_arr, dtype=float)
            except (TypeError, ValueError):
                # Phase names instead of indices - map them, don't crash.
                phase_np = np.asarray(
                    [_PHASE_INDEX.get(p, 0) if isinstance(p, str) else p
                     for p in phase_arr], dtype=float)
        n = len(pts_arr)

        # Decimate to the quality-mode point cap for rendering
        cap = max(200, self._trail_cap)
        if n > cap:
            step = max(1, n // cap)
            pts_arr  = pts_arr[::step]
            mach_arr = mach_arr[::step]
            alt_arr  = alt_arr[::step]
            vel_arr  = vel_arr[::step]
            if phase_np is not None:
                phase_np = phase_np[::step]
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
        elif mode == "Flight Phase" and phase_np is not None:
            # Categorical: one distinct color per flight phase
            scalars = phase_np
            cmap, clim = _PHASE_CMAP, [-0.5, len(_PHASE_CMAP) - 0.5]
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
                marker, color=theme.OK, opacity=0.7, name="landing"
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
                sphere, color=theme.ERR, opacity=0.30, name="failure"
            )
        except Exception:
            pass

    def _apply_camera(self, x, y, z, pitch, yaw=0.0):
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
        elif mode == "Onboard":
            # Ride just behind the nose, looking out along the body axis
            cp = math.cos(pitch)
            ax = (cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch))
            nose = rsize * 1.05
            back = rsize * 0.9
            pos = (x + ax[0] * nose - back * 0.35,
                   y + ax[1] * nose - back * 0.35,
                   z + ax[2] * nose + back * 0.25)
            foc = (x + ax[0] * 2000, y + ax[1] * 2000, z + ax[2] * 2000)
        elif mode == "Orbit":
            # Slow cinematic orbit around the rocket (wall-clock driven)
            ang = (perf_counter() - self._orbit_t0) * 0.25
            d = rsize * 4.5
            pos = (x + d * math.cos(ang), y + d * math.sin(ang), z + d * 0.3)
            foc = (x, y, z)
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

        # Orbit camera moves on wall-clock even when the scene is static
        if self._camera_mode == "Orbit" and getattr(self, "_last_pose", None):
            px, py, pz, ppitch, pyaw = self._last_pose
            self._apply_camera(px, py, pz, ppitch, pyaw)

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
        yaw   = getattr(s, 'yaw', 0.0)
        x, y, z = s.x_position, s.y_position, s.altitude
        self._last_pose = (x, y, z, pitch, yaw)

        self._update_rocket_transform(x, y, z, pitch, yaw)

        # Trail: rebuild throttled (every ~1s of render ticks, or while short)
        self._trail_rebuild_counter += 1
        if self._trail_rebuild_counter >= 30 or len(self._trail_pts) < 30:
            self._trail_rebuild_counter = 0
            if len(self._trail_pts) >= 2:
                self._rebuild_trail_actor(
                    list(self._trail_pts), list(self._trail_mach),
                    list(self._trail_alt), list(self._trail_vel),
                    list(self._trail_phase),
                )

        if self._camera_mode != "Free":
            self._apply_camera(x, y, z, pitch, yaw)

        if s.sim_phase in ("Landed", "Terminated") and self._actor_landing is None:
            self._show_landing_marker(x, y)

        descent = abs(s.velocity) if s.velocity < 0 else 0.0
        self._update_recovery(x, y, z, s.sim_time, descent)
        self._ensure_alt_reference(z)

        # Flight effects (flame / smoke / Mach cone / dust)
        if self._effects and self._show_effects:
            thrust_frac = s.thrust / self._max_thrust_seen \
                if self._max_thrust_seen > 0 else 0.0
            self._effects.update(
                x, y, z, pitch, yaw, thrust_frac, s.mach_number, s.sim_time,
                self._rocket_length, self._rocket_diameter,
                landed=(s.sim_phase == "Landed"),
            )

        # Force vectors (thrust / velocity / drag)
        if self._vectors and self._show_vectors:
            vel_vec = (getattr(s, 'velocity_x', 0.0),
                       getattr(s, 'velocity_y', 0.0),
                       getattr(s, 'velocity_z', 0.0))
            self._vectors.update(x, y, z, pitch, yaw, vel_vec,
                                 s.thrust, s.drag, self._rocket_length)

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
        if not self.isVisible() or self._replay_mode:
            return      # in replay the graphs hold the whole flight already
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
        self._trail_pitch.append(getattr(s, 'pitch', math.pi / 2))
        self._trail_yaw.append(getattr(s, 'yaw', 0.0))
        self._trail_thrust.append(s.thrust)
        self._trail_drag.append(s.drag)
        self._trail_phase.append(_PHASE_INDEX.get(s.sim_phase, 0))
        if s.thrust > self._max_thrust_seen:
            self._max_thrust_seen = s.thrust

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
        if self._replay_mode and self._replay is not None:
            self._show_replay_instant()
            return
        s = self._latest_state
        if s is None:
            return
        self._rd_time.set_value(s.sim_time,          "{:.2f}")
        self._rd_alt.set_value(s.altitude,           "{:.1f}")
        self._rd_vel.set_value(abs(s.velocity),      "{:.1f}")
        self._rd_accel.set_value(abs(s.acceleration),"{:.2f}")
        self._rd_mach.set_value(s.mach_number,       "{:.3f}")
        # flight_mass(): airframe, motor case and what is left of the
        # propellant. This was dry + propellant, the rocket without its motor
        # case (1.03 kg short for an L1090W), while the Simulation tab and
        # the replay of the same flight showed all of it.
        self._rd_mass.set_value(s.flight_mass(), "{:.2f}")
        self._rd_thrust.set_value(s.thrust,          "{:.1f}")
        self._rd_drag.set_value(s.drag,              "{:.1f}")
        self._rd_dynq.set_value(s.dynamic_pressure,  "{:.0f}")

        stab = s.stability_margin
        sc = theme.OK if stab >= 1.5 else (theme.ACCENT_HOVER if stab >= 0.5 else theme.ERR)
        self._rd_stab.set_value(stab, "{:.2f}", sc)

        pc = _PHASE_COLORS.get(s.sim_phase, theme.TEXT_DIM)
        self._rd_phase.set_value(s.sim_phase, color=pc)
        if s.sim_phase != getattr(self, "_phase_lbl_text", None):
            self._phase_lbl.setText(s.sim_phase)
            self._phase_lbl.setStyleSheet(
                f"color:{pc};font-size:11px;font-weight:600;"
                f"padding:2px 10px;border:1px solid {theme.RAISED};border-radius:4px;margin-left:8px;"
            )
            self._phase_lbl_text = s.sim_phase

        if s.parachute_deployed:
            self._rd_recovery.set_value("DEPLOYED", color=theme.OK)
        else:
            self._rd_recovery.set_value(str(s.flight_computer_state))

        # In-view HUD (10 Hz, single text actor — SetInput, never re-add)
        if self._hud_actor is not None and self._show_hud:
            vx = getattr(s, 'velocity_x', 0.0)
            vy = getattr(s, 'velocity_y', 0.0)
            vz = getattr(s, 'velocity_z', 0.0)
            hspeed = math.hypot(vx, vy)
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            if speed > 1.0:
                aoa = math.degrees(
                    getattr(s, 'pitch', math.pi / 2) - math.atan2(vz, hspeed))
            else:
                aoa = 0.0
            hud = (f"T+{s.sim_time:7.2f} s   {s.sim_phase}\n"
                   f"ALT {s.altitude:8.1f} m    VEL {abs(s.velocity):7.1f} m/s\n"
                   f"MACH {s.mach_number:5.2f}      AoA {aoa:+6.2f} deg\n"
                   f"Q   {s.dynamic_pressure:8.0f} Pa   THR {s.thrust:7.0f} N")
            try:
                self._hud_actor.SetInput(hud)
            except Exception:
                pass

        self._panel_ms = (perf_counter() - t0) * 1000.0
        if self._show_stats:
            self._update_stats_label()

    def _check_failures(self, s):
        fail, reason = False, ""
        if s.sim_time > 0.5 and s.sim_phase not in ("Landed", "Terminated"):
            # Unstable is a margin below zero, as the flight's own warning
            # has it (flight_warnings). This flagged FAILURE below 0.5 cal,
            # on rockets that flew straight and carried no warning.
            if s.stability_margin < 0.0:
                fail, reason = True, "Instability"
            elif 0 < s.safety_factor < 0.8:
                fail, reason = True, "Structural Failure"
            elif s.mach_number > 3.5:
                fail, reason = True, "Mach Limit Exceeded"
            elif 0 < getattr(s, 'flutter_margin', 99) < 1.0:
                fail, reason = True, "Flutter"

        if fail and not self._failure_active:
            self._failure_active = True
            self._add_timeline_entry(s.sim_time, f"FAILURE: {reason}", theme.ERR, events=True)
            self._update_status("FAILURE", theme.ERR)
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
        self._replay = None
        self._replay_shown_t = None
        if self._loads_combo.currentText() != "Off":
            self._loads_combo.setCurrentText("Off")
        self._pb_group.setVisible(False)
        for cursor in getattr(self, "_graph_cursors", []):
            cursor.set_visible(False)
        self._failure_active = False
        self._tick_counter   = 0
        self._trail_rebuild_counter = 0

        # Clear buffers
        for buf in (self._trail_pts, self._trail_mach, self._trail_alt, self._trail_vel,
                    self._trail_pitch, self._trail_yaw, self._trail_thrust,
                    self._trail_drag,
                    self._trail_phase,
                    self._g_time, self._g_alt, self._g_vel, self._g_mach,
                    self._g_accel, self._g_dynq):
            buf.clear()

        # Reset effects / flags / vectors + thrust normalization
        self._max_thrust_seen = 1.0
        self._orbit_t0 = perf_counter()
        self._last_pose = None
        if self._effects:
            self._effects.reset()
        if self._flags:
            self._flags.clear()
        if self._vectors:
            self._vectors.reset()

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

        self._update_status("LIVE", theme.OK)
        self._add_timeline_entry(0.0, "Simulation started", theme.ACCENT)

    @pyqtSlot()
    def _on_sim_finished(self):
        t = self._latest_state.sim_time if self._latest_state else 0.0
        self._update_status("COMPLETE - REPLAY", theme.ACCENT)
        self._add_timeline_entry(t, "Simulation complete", theme.OK)

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

        try:
            self._enter_replay_mode()
        except Exception:
            # A trail/overlay problem must not cost the user their replay
            # controls - log it and still offer playback.
            logger.exception("Replay setup failed; enabling controls anyway")
            self._replay_mode = True
            self._pb_group.setVisible(True)

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

        # 3D flag marker on the trajectory at the event position
        if (self._flags and self._show_flags and s is not None
                and event_name in _FLAG_EVENTS):
            flbl, fcol = _FLAG_EVENTS[event_name]
            self._flags.add(event_name, flbl, fcol,
                            (s.x_position, s.y_position, s.altitude))

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

    def _add_timeline_entry(self, t, text, color=None, events=False):
        color = color or theme.TEXT_DIM
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
        if t is None or not self._replay_mode or self._replay is None:
            return
        self._clock.pause()
        self._clock.seek(float(t))

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

    def _on_toggle_effects(self, on):
        self._show_effects = on
        if self._effects:
            self._effects.set_visible(on)
        self._needs_3d_update = True
        self._safe_render()

    def _on_toggle_vectors(self, on):
        self._show_vectors = on
        if self._vectors:
            self._vectors.set_visible(on)
        self._vector_key.setText(self._vector_key_html())
        self._vector_key.setVisible(bool(on))
        self._needs_3d_update = True
        self._safe_render()

    def _on_toggle_flags(self, on):
        self._show_flags = on
        if self._flags:
            self._flags.set_visible(on)
        self._safe_render()

    def _on_toggle_hud(self, on):
        self._show_hud = on
        if self._hud_actor is not None:
            try:
                self._hud_actor.SetVisibility(on)
                if not on:
                    self._hud_actor.SetInput("")
            except Exception:
                pass
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
        """Descent rate (m/s) for the landing envelope: the burnt-out rocket
        under the canopies it carries (recovery.final_descent_rate, which
        says what this used to work out instead). 60 m/s, the envelope's
        ceiling, for a rocket with no parachute."""
        s = self._latest_state
        if s is None and hasattr(self.engine, "state"):
            s = self.engine.state
        if s is None:
            return 5.0
        from recovery.parachute_dynamics import final_descent_rate
        v = final_descent_rate(s)
        return 60.0 if v is None else max(1.0, min(v, 60.0))

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
             "Landed": "Landed"}.get(st, ""), theme.TEXT_DIM,
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

    def reset_workspace(self):
        """Reset the visualizer + plots (called on New Project)."""
        self._last_text = None
        self._last_col = None
        from ui.workspace_reset import clear_visuals
        clear_visuals(self)
        for m in ("_clear_flight", "_reset_flight", "clear_scene"):
            fn = getattr(self, m, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

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

    @property
    def _replay_is_playing(self):
        return self._clock.playing

    def _enter_replay_mode(self):
        """Switch from the live view to replaying the flight just recorded.
        The flight and the instant shown are the shared clock's."""
        self._clock.ensure_loaded()
        self._replay = self._clock.replay
        if self._replay is None:
            return
        # The trail drawn during the flight stays as the path; it is static
        # for the whole replay.
        self._replay_pts    = list(self._trail_pts)
        self._replay_mach   = list(self._trail_mach)
        self._replay_alt    = list(self._trail_alt)
        self._replay_vel    = list(self._trail_vel)
        self._replay_phase  = list(self._trail_phase)
        self._replay_times  = [float(t) for t in self._replay.times]
        if len(self._replay_pts) >= 2:
            self._rebuild_trail_actor(
                self._replay_pts, self._replay_mach,
                self._replay_alt, self._replay_vel,
                self._replay_phase,
            )

        self._replay_mode = True
        self._replay_shown_t = None
        self._replay_index = self._replay.index_at(self._clock.time)
        self._plot_whole_flight()
        self._pb_group.setVisible(True)
        logger.info("Replay ready: %d recorded points, %.1f s of flight",
                    self._replay.count, self._replay.duration)
        self._update_status("REPLAY", theme.WARN)
        # Place the rocket at touchdown, then frame the whole scene so the user
        # can immediately see the full trajectory + envelope + altitude scale.
        self._needs_3d_update = False
        self._render_replay_scene()
        self._fit_scene()

    def _render_replay_scene(self):
        """Per-frame replay update. The full trajectory trail is a STATIC actor
        built once on replay entry — here we only move the rocket + camera, so
        scrubbing is instant regardless of flight length."""
        if self._replay is None or not self._plotter:
            return

        frame_t = self._clock.time
        f = self._replay.frame(frame_t)
        self._replay_index = self._replay.index_at(frame_t)
        x, y, z = f["x"], f["y"], f["altitude"]
        pitch, yaw = f["pitch"], f["yaw"]
        self._last_pose = (x, y, z, pitch, yaw)

        self._update_rocket_transform(x, y, z, pitch, yaw)

        # Follow the rocket only while actively playing. When paused/seeking the
        # camera is released so the user can freely orbit + zoom out to inspect
        # the full envelope and altitude scale.
        if self._camera_mode != "Free" and self._clock.playing:
            self._apply_camera(x, y, z, pitch, yaw)

        self._update_recovery(x, y, z, frame_t, abs(f["velocity"]))

        # Flame + Mach cone track the replay frame (smoke/dust live-only)
        if self._effects and self._show_effects:
            thrust_frac = (f["thrust"] / self._max_thrust_seen
                           if self._max_thrust_seen > 0 else 0.0)
            self._effects.update(x, y, z, pitch, yaw, thrust_frac, f["mach"],
                                 frame_t, self._rocket_length,
                                 self._rocket_diameter, landed=False)

        self._update_loads(f["dynamic_pressure"])

        # The free-body picture at this instant, as recorded by the engine.
        if self._vectors and self._show_vectors:
            self._vectors.update_forces((x, y, z), self._replay.forces(frame_t),
                                        self._rocket_length)

    def _plot_whole_flight(self):
        """Draw the graph strip from the full record. The live buffers keep
        only the most recent 5,000 samples, so after a long descent the
        strip showed the end of the flight and none of the boost."""
        if not _MPL_OK or not self._graph_lines:
            return
        r = self._replay
        step = max(1, r.count // 2000)
        t = r.times[::step]
        series = (r.series("altitude"), abs(r.series("velocity")), r.series("mach"),
                  abs(r.series("acceleration")), r.series("dynamic_pressure"))
        for ax, line, canvas, data in zip(self._graph_axes, self._graph_lines,
                                          self._graph_canvases, series):
            line.set_xdata(t)
            line.set_ydata(data[::step])
            ax.relim()
            ax.autoscale_view()
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _show_replay_instant(self):
        """Telemetry panel, HUD and graph cursors for the replayed instant.
        Runs on the 10 Hz panel timer; does nothing while the instant has
        not moved."""
        t = self._clock.time
        if t == self._replay_shown_t:
            return
        self._replay_shown_t = t
        f = self._replay.frame(t)

        def show(readout, value, fmt):
            readout.set_value("--" if value != value else fmt.format(value))

        show(self._rd_time, t, "{:.2f}")
        show(self._rd_alt, f["altitude"], "{:.1f}")
        show(self._rd_vel, abs(f["velocity"]), "{:.1f}")
        show(self._rd_accel, abs(f["acceleration"]), "{:.2f}")
        show(self._rd_mach, f["mach"], "{:.3f}")
        show(self._rd_mass, f["mass"], "{:.2f}")
        show(self._rd_thrust, f["thrust"], "{:.1f}")
        show(self._rd_drag, f["drag"], "{:.1f}")
        show(self._rd_dynq, f["dynamic_pressure"], "{:.0f}")
        stab = f["stability_margin"]
        if stab != stab:            # NaN: no static margin under a canopy
            self._rd_stab.set_value("--")
        else:
            self._rd_stab.set_value(
                stab, "{:.2f}",
                theme.OK if stab >= 1.5 else (theme.ACCENT_HOVER if stab >= 0.5 else theme.ERR))
        phase = f["phase"]
        pc = _PHASE_COLORS.get(phase, theme.TEXT_DIM)
        self._rd_phase.set_value(phase, color=pc)
        if phase != getattr(self, "_phase_lbl_text", None):
            self._phase_lbl.setText(phase)
            self._phase_lbl.setStyleSheet(
                f"color:{pc};font-size:11px;font-weight:600;"
                f"padding:2px 10px;border:1px solid {theme.RAISED};border-radius:4px;margin-left:8px;"
            )
            self._phase_lbl_text = phase
        # A canopy is out from its recorded deployment onward.
        out = [e for e in self._replay.events if e.key == "deploy" and e.time <= t]
        if out:
            self._rd_recovery.set_value("DEPLOYED", color=theme.OK)
        else:
            self._rd_recovery.set_value("STOWED")

        if self._hud_actor is not None and self._show_hud:
            hud = (f"T+{t:7.2f} s   {phase}   REPLAY\n"
                   f"ALT {f['altitude']:8.1f} m    VEL {abs(f['velocity']):7.1f} m/s\n"
                   f"MACH {f['mach']:5.2f}      AoA {math.degrees(f['aoa']):+6.2f} deg\n"
                   f"Q   {f['dynamic_pressure']:8.0f} Pa   THR {f['thrust']:7.0f} N")
            try:
                self._hud_actor.SetInput(hud)
            except Exception:
                pass

        for cursor, canvas in zip(self._graph_cursors, self._graph_canvases):
            cursor.set_xdata([t, t])
            cursor.set_visible(True)
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _on_clock_time(self, _t):
        if self._replay_mode:
            self._needs_3d_update = True

    def _on_clock_playing(self, playing):
        if playing:
            self._cam_snap = True          # re-grab the rocket when playback resumes
        else:
            self._cam_target_pos = None    # release the camera for free inspection

    def _on_clock_flight(self):
        """The clock dropped its flight (a new one is starting) or was given
        one by another view."""
        if self._clock.replay is None:
            self._replay = None
            self._replay_mode = False

    def _on_replay_play(self):
        if not self._replay_mode:
            logger.warning("Play ignored - not in replay mode (no flight loaded yet)")
            return
        self._clock.play()

    def _on_replay_pause(self):
        self._clock.pause()

    def _on_replay_restart(self):
        self._clock.restart()

    def _on_replay_seek(self, index):
        """Move to recorded point *index*."""
        self._clock.seek_index(index)

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
        for tmr in ('_render_timer', '_graph_timer', '_panel_timer'):
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
