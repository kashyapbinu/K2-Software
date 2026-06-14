"""
K2 AeroSim — Cinematic Flight View
======================================
A second, purely visual flight view rendered with three.js inside a
QWebEngineView: PBR rocket, HDR-style bloom, GPU exhaust/smoke sprites,
velocity-coloured trail, transonic vapour cone, parachute inflation, and
cinematic camera modes — driven by the same telemetry feed the Mission
Visualizer uses, so the two tabs can coexist.

Data flow (one-way, render-only):
    RocketStateEngine.telemetry_tick ──(throttled ~30 Hz)──▶ _Bridge.tickSig(json)
    SimulationEngine.sim_started      ──▶ _Bridge.initSig(json: geometry/motor/recovery)
                                          + _Bridge.resetSig()
All physics stays in Python; the web page never sends anything back except
a one-shot "ready" handshake.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, QUrl, QElapsedTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtWebChannel import QWebChannel

logger = logging.getLogger("K2.CinematicWS")

_WEB_DIR = Path(__file__).resolve().parents[2] / "visualization" / "cinematic"


class _Bridge(QObject):
    """QWebChannel object exposed to the page as ``k2``."""
    initSig = pyqtSignal(str)    # one-shot scene setup (geometry, motor, recovery)
    tickSig = pyqtSignal(str)    # telemetry sample
    resetSig = pyqtSignal()      # clear trail/smoke/HUD maxima

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page_ready = False

    @pyqtSlot()
    def ready(self):
        """Page finished wiring the channel — safe to (re)send init."""
        self.page_ready = True
        p = self.parent()
        if p is not None and hasattr(p, "_send_init"):
            p._send_init()


class CinematicWorkspace(QWidget):
    """Cinematic 3D flight view (three.js / WebGL)."""

    def __init__(self, engine, sim_engine=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.sim_engine = sim_engine
        self._throttle = QElapsedTimer()
        self._throttle.start()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
        except Exception as e:
            logger.error(f"QtWebEngine unavailable: {e}")
            msg = QLabel(
                "Cinematic view requires PyQt6-WebEngine.\n"
                "pip install PyQt6-WebEngine (version-matched to PyQt6)."
            )
            msg.setStyleSheet("color:#8b949e; font-size:14px; padding:40px;")
            lay.addWidget(msg)
            self._view = None
            self._bridge = None
            return

        self._view = QWebEngineView()
        self._bridge = _Bridge(self)
        self._channel = QWebChannel(self._view.page())
        self._channel.registerObject("k2", self._bridge)
        self._view.page().setWebChannel(self._channel)

        index = _WEB_DIR / "index.html"
        if not index.is_file():
            logger.error(f"Cinematic assets missing: {index}")
        self._view.load(QUrl.fromLocalFile(str(index)))
        # Page zoom must stay fixed — wheel/pinch zoom is the 3D camera's job.
        # JS blocks ctrl+wheel; this catches anything that still slips through.
        self._view.loadFinished.connect(lambda ok: self._view.setZoomFactor(1.0))
        self._zoom_guard = QTimer(self)
        self._zoom_guard.setInterval(1000)
        self._zoom_guard.timeout.connect(self._pin_zoom)
        self._zoom_guard.start()
        lay.addWidget(self._view)

        # Telemetry feed (shared with the rest of the app)
        self.engine.telemetry_tick.connect(self._on_tick)
        if self.sim_engine is not None:
            self.sim_engine.sim_started.connect(self._on_sim_started)
            # Sim end updates state via state_changed (not telemetry_tick), so
            # push one final sample — JS needs running:false to offer replay
            # and the Landed phase to fire touchdown/envelope.
            self.sim_engine.sim_finished.connect(self._on_sim_finished)

    # ── Init payload (geometry + motor + recovery) ───────────────────────────

    def _assembly(self):
        main = self.window()
        if hasattr(main, "design_ws"):
            return getattr(main.design_ws, "assembly", None)
        return None

    def _send_init(self):
        if self._bridge is None:
            return
        s = self.engine.state
        geometry = {}
        asm = self._assembly()
        if asm is not None:
            try:
                from cfd.geometry_exporter import extract_cfd_geometry
                geometry = extract_cfd_geometry(asm)
            except Exception as e:
                logger.warning(f"Cinematic geometry extraction failed: {e}")
        if not geometry:
            # Fallback: rebuild rough dims from state so the scene still works
            geometry = {
                "length": getattr(s, "length", 1.2) or 1.2,
                "body_radius": (getattr(s, "diameter", 0.066) or 0.066) / 2,
                "nose_length": getattr(s, "nose_length", 0.0) or (s.length * 0.25 if s.length else 0.3),
                "fin_count": getattr(s, "fin_count", 4) or 4,
                "fin_height": getattr(s, "fin_span", 0.0) or 0.05,
                "fin_root": getattr(s, "fin_root_chord", 0.0) or 0.1,
                "fin_tip": getattr(s, "fin_tip_chord", 0.0) or 0.05,
                "fin_sweep_deg": math.degrees(getattr(s, "fin_sweep_angle", 0.0) or 0.5),
                "fin_thick": getattr(s, "fin_thickness", 0.003) or 0.003,
            }
        payload = {
            "geometry": geometry,
            "max_thrust": getattr(s, "motor_max_thrust", 0.0)
                          or getattr(s, "motor_avg_thrust", 0.0) or 200.0,
            "recovery": {
                "drogue_cd_area": getattr(s, "drogue_cd_area", 0.5) or 0.5,
                "main_cd_area": getattr(s, "main_cd_area", 3.0) or 3.0,
            },
            "envelope": self._envelope_params(s),
        }
        self._bridge.initSig.emit(json.dumps(payload))

    def _envelope_params(self, s):
        """Mission-envelope inputs, mirroring the Mission Visualizer's sources:
        target/recovery from its control spinboxes, wind from state, terminal
        descent from main-chute CdA."""
        target, recov = 0.0, 1000.0
        mv = getattr(self.window(), "mission_viz_ws", None)
        try:
            if mv is not None and hasattr(mv, "_spin_target"):
                target = float(mv._spin_target.value())
                recov = float(mv._spin_recov.value())
        except Exception:
            pass
        m = max(0.1, (getattr(s, "dry_mass", 0.0) or 0.0)
                + (getattr(s, "propellant_mass", 0.0) or 0.0))
        cd_area = max(0.1, getattr(s, "main_cd_area", 1.5) or 1.5)
        descent = math.sqrt(2.0 * m * 9.81 / (1.225 * cd_area))
        return {
            "target_apogee": target,
            "recovery_radius": recov,
            "wind_speed": getattr(s, "wind_speed", 0.0) or 0.0,
            "wind_dir_deg": getattr(s, "wind_direction", 0.0) or 0.0,
            "descent_rate": max(1.0, min(descent, 60.0)),
        }

    def reset_workspace(self):
        """Clear the 3D replay/flight on New Project (resetSig → resetFlight)."""
        if self._bridge is not None:
            try:
                self._bridge.resetSig.emit()
            except Exception:
                pass

    def _on_sim_started(self):
        if self._bridge is None:
            return
        self._send_init()
        self._bridge.resetSig.emit()

    def _on_sim_finished(self):
        self._on_tick(self.engine.state)

    def _pin_zoom(self):
        if self._view is not None and self._view.zoomFactor() != 1.0:
            self._view.setZoomFactor(1.0)

    # ── Telemetry relay (throttled — JS interpolates between samples) ────────

    def _on_tick(self, state):
        if self._bridge is None or not self._bridge.page_ready:
            return
        if self._throttle.elapsed() < 33 and getattr(state, "sim_running", False):
            return
        self._throttle.restart()
        sample = {
            "t": state.sim_time,
            "x": getattr(state, "x_position", 0.0),
            "y": getattr(state, "y_position", 0.0),
            "alt": state.altitude,
            "vx": getattr(state, "velocity_x", 0.0),
            "vy": getattr(state, "velocity_y", 0.0),
            "vz": getattr(state, "velocity_z", 0.0),
            "pitch": getattr(state, "pitch", math.pi / 2),
            "yaw": getattr(state, "yaw", 0.0),
            "roll": getattr(state, "roll", 0.0),
            "thrust": state.thrust,
            "mach": state.mach_number,
            "acc": abs(getattr(state, "acceleration", 0.0) or 0.0),
            "q": getattr(state, "dynamic_pressure", 0.0),
            "phase": getattr(state, "sim_phase", "Pre-Launch"),
            "running": getattr(state, "sim_running", False),
        }
        self._bridge.tickSig.emit(json.dumps(sample))

    def showEvent(self, event):
        super().showEvent(event)
        if self._bridge is not None and self._bridge.page_ready:
            self._send_init()
