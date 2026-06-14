"""
K2 AeroSim — Recovery (Parachute) Visualizer
================================================
Renders an animated recovery system on the shared Mission Visualizer plotter.

Owns its own named PyVista actors:
    recov_canopy   — inflating drogue / main canopy (hemispherical dome)
    recov_lines    — suspension lines from canopy rim to the rocket nose

The visualizer is a pure state machine driven by deployment *times* (sim
seconds), so it works identically for the live flight and for scrubbed replay
— `update()` is called every frame with the current sim_time and rocket pose,
and it derives the recovery state + canopy inflation from the recorded
deployment times.

Recovery states (exposed as strings):
    Packed, Drogue Deploying, Drogue Descending,
    Main Deploying, Main Descending, Landed

PyVista 0.48 notes (see project memory):
    - every pv call is wrapped: an unhandled throw blanks the whole view.
    - never call plotter.render() here; the workspace owns render timing.
"""

import logging
import math

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:  # pragma: no cover - pyvista optional
    _PV_OK = False

logger = logging.getLogger("K2.MissionViz.Recovery")

# Recovery states
PACKED            = "Packed"
DROGUE_DEPLOYING  = "Drogue Deploying"
DROGUE_DESCENDING = "Drogue Descending"
MAIN_DEPLOYING    = "Main Deploying"
MAIN_DESCENDING   = "Main Descending"
LANDED            = "Landed"

# Canopy colours
_DROGUE_COLOR = "#79c0ff"
_MAIN_COLOR   = "#56d364"
_LINE_COLOR   = "#8b949e"


def _ease_out(p: float) -> float:
    """Ease-out cubic for a realistic 'snap then settle' inflation."""
    p = max(0.0, min(1.0, p))
    return 1.0 - (1.0 - p) ** 3


class RecoveryVisualizer:
    """Animated drogue + main parachute rendering for the Mission Visualizer."""

    # Inflation durations (sim seconds)
    _DROGUE_INFLATE = 0.8
    _MAIN_INFLATE   = 1.3

    # Assumed drag coefficients to turn Cd*A back into a displayed canopy area
    _DROGUE_CD = 1.5
    _MAIN_CD   = 1.4

    def __init__(self, plotter, vis_scale: float = 30.0):
        self._p = plotter
        self._vis_scale = vis_scale
        self._visible = True
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self):
        """Clear actors and deployment state for a fresh flight."""
        self._remove("recov_canopy")
        self._remove("recov_lines")
        self._drogue_t = None
        self._drogue_alt = None
        self._main_t = None
        self._main_alt = None
        self._landed_t = None
        self._drogue_cd_area = 0.5
        self._main_cd_area = 3.0
        # Last-computed telemetry snapshot
        self._state = PACKED
        self._inflation = 0.0
        self._canopy_area = 0.0
        self._descent_rate = 0.0

    def set_chute_config(self, drogue_cd_area: float, main_cd_area: float):
        if drogue_cd_area > 0:
            self._drogue_cd_area = drogue_cd_area
        if main_cd_area > 0:
            self._main_cd_area = main_cd_area

    def set_visible(self, visible: bool):
        self._visible = visible
        if not visible:
            self._remove("recov_canopy")
            self._remove("recov_lines")

    # ── event hooks (absolute sim times) ─────────────────────────────

    def on_drogue_deploy(self, t: float, altitude: float = None):
        self._drogue_t = t
        self._drogue_alt = altitude

    def on_main_deploy(self, t: float, altitude: float = None):
        self._main_t = t
        self._main_alt = altitude

    def on_landing(self, t: float):
        self._landed_t = t

    # ── per-frame update ─────────────────────────────────────────────

    def update(self, x, y, z, sim_time, rocket_vis_length, descent_rate=0.0):
        """Rebuild canopy/line actors for the current frame.

        Args:
            x, y, z:            rocket base position (world metres).
            sim_time:           current sim time (live or replay frame).
            rocket_vis_length:  exaggerated rocket length in world units
                                (rocket_length * vis_scale) — sizes the canopy.
            descent_rate:       |downward velocity| for telemetry (m/s).
        """
        self._descent_rate = abs(descent_rate)
        state, inflation, active = self._resolve_state(sim_time)
        self._state = state
        self._inflation = inflation

        if not self._visible or self._p is None or active is None or inflation <= 0.0:
            self._remove("recov_canopy")
            self._remove("recov_lines")
            self._canopy_area = 0.0
            return

        if active == "drogue":
            base_r = max(2.0, rocket_vis_length * 0.7)
            color = _DROGUE_COLOR
            cd_area = self._drogue_cd_area
            cd = self._DROGUE_CD
            line_gap = rocket_vis_length * 1.0
        else:  # main
            base_r = max(2.0, rocket_vis_length * 2.2)
            color = _MAIN_COLOR
            cd_area = self._main_cd_area
            cd = self._MAIN_CD
            line_gap = rocket_vis_length * 1.6

        infl = _ease_out(inflation)
        radius = base_r * (0.12 + 0.88 * infl)   # starts as a small bud
        gap = line_gap * (0.3 + 0.7 * infl)
        self._canopy_area = (cd_area / cd) * infl

        nose_z = z + rocket_vis_length
        canopy_z = nose_z + gap
        self._draw_canopy(x, y, canopy_z, radius, color)
        self._draw_lines(x, y, nose_z, canopy_z, radius)

    # ── geometry ─────────────────────────────────────────────────────

    def _draw_canopy(self, x, y, cz, radius, color):
        self._remove("recov_canopy")
        try:
            # Top hemisphere: phi 0..90 (from +z pole down to equator) gives a
            # dome that opens downward toward the rocket.
            dome = pv.Sphere(
                radius=radius, center=(x, y, cz),
                start_phi=0, end_phi=95,
                theta_resolution=24, phi_resolution=12,
            )
            actor = self._p.add_mesh(
                dome, color=color, name="recov_canopy",
                opacity=0.85, smooth_shading=True,
                ambient=0.4, diffuse=0.6, show_edges=False,
            )
            self._actor_set_culling(actor)
        except Exception as exc:
            logger.debug(f"canopy draw failed: {exc}")

    def _draw_lines(self, x, y, nose_z, canopy_z, radius):
        self._remove("recov_lines")
        try:
            n = 8
            rim_z = canopy_z - radius * 0.15
            pts = []
            cells = []
            apex = np.array([x, y, nose_z])
            for i in range(n):
                ang = 2.0 * math.pi * i / n
                rim = np.array([
                    x + radius * math.cos(ang),
                    y + radius * math.sin(ang),
                    rim_z,
                ])
                base = len(pts)
                pts.append(apex)
                pts.append(rim)
                cells.extend([2, base, base + 1])
            poly = pv.PolyData(np.array(pts, dtype=float))
            poly.lines = np.array(cells)
            self._p.add_mesh(
                poly, color=_LINE_COLOR, name="recov_lines",
                line_width=1.2, show_scalar_bar=False,
            )
        except Exception as exc:
            logger.debug(f"suspension lines draw failed: {exc}")

    def _actor_set_culling(self, actor):
        # Show the inside of the dome too, so the canopy reads from below.
        try:
            actor.GetProperty().SetBackfaceCulling(False)
            actor.GetProperty().SetFrontfaceCulling(False)
        except Exception:
            pass

    # ── state machine ────────────────────────────────────────────────

    def _resolve_state(self, t):
        """Return (state_str, inflation 0..1, active_chute|None)."""
        if self._landed_t is not None and t >= self._landed_t:
            return LANDED, 0.0, None

        # Main takes over once its deploy time passes
        if self._main_t is not None and t >= self._main_t:
            p = (t - self._main_t) / self._MAIN_INFLATE
            if p < 1.0:
                return MAIN_DEPLOYING, p, "main"
            return MAIN_DESCENDING, 1.0, "main"

        if self._drogue_t is not None and t >= self._drogue_t:
            p = (t - self._drogue_t) / self._DROGUE_INFLATE
            if p < 1.0:
                return DROGUE_DEPLOYING, p, "drogue"
            return DROGUE_DESCENDING, 1.0, "drogue"

        return PACKED, 0.0, None

    # ── telemetry ────────────────────────────────────────────────────

    def telemetry(self, sim_time):
        """Recovery telemetry dict for the side panel."""
        deploy_alt = None
        time_since = None
        if self._state in (MAIN_DEPLOYING, MAIN_DESCENDING) and self._main_t is not None:
            deploy_alt = self._main_alt
            time_since = max(0.0, sim_time - self._main_t)
        elif self._state in (DROGUE_DEPLOYING, DROGUE_DESCENDING) and self._drogue_t is not None:
            deploy_alt = self._drogue_alt
            time_since = max(0.0, sim_time - self._drogue_t)
        return {
            "state": self._state,
            "canopy_area": self._canopy_area,
            "descent_rate": self._descent_rate,
            "deploy_altitude": deploy_alt,
            "time_since_deploy": time_since,
        }

    # ── helpers ──────────────────────────────────────────────────────

    def _remove(self, name):
        if self._p is None:
            return
        try:
            self._p.remove_actor(name)
        except Exception:
            pass
