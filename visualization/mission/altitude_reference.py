"""
K2 AeroSim — Altitude Reference System
==========================================
Renders altitude situational-awareness layers on the Mission Visualizer
plotter so the user can judge height at a glance:

    * semi-transparent horizontal reference planes
    * floating billboard altitude labels (always face the camera)
    * an optional vertical scale bar / ruler at the scene corner

Spacing auto-scales to the flight: a 500 m hop gets ~100 m increments, a
10 km flight gets ~1000 m increments. All actors are static once built, so
they persist untouched through replay.

PyVista 0.48 notes: pv.Plane for planes, add_point_labels for billboards
(point_size=0 → label only). Every call defensive; no render() here.
"""

import logging

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:  # pragma: no cover
    _PV_OK = False

logger = logging.getLogger("K2.MissionViz.AltRef")

# "Nice" increments (m) to choose spacing from
_NICE_STEPS = [50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]

_PLANE_COLOR = "#3b5168"
_LABEL_COLOR = "#8b949e"
_BAR_COLOR   = "#58a6ff"


def nice_step(apogee: float, target_divisions: int = 8) -> float:
    """Pick a clean altitude increment giving ~target_divisions planes."""
    if apogee <= 0:
        return 100.0
    raw = apogee / max(1, target_divisions)
    for s in _NICE_STEPS:
        if s >= raw:
            return float(s)
    # Beyond the table: round up to the next 10 km
    return float(int(raw / 10000 + 1) * 10000)


class AltitudeReferenceSystem:
    """Reference planes, floating labels, and an optional scale bar."""

    def __init__(self, plotter):
        self._p = plotter
        self._plane_actors = []
        self._bar_actors = []
        self._planes_visible = True
        self._bar_visible = False
        self._levels = []
        self._extent = 4000.0
        self._cx = 0.0
        self._cy = 0.0

    # ── build ────────────────────────────────────────────────────────

    def build(self, apogee: float, x_center: float = 0.0, y_center: float = 0.0,
              x_landing: float = 0.0):
        """(Re)build reference layers sized to the flight's apogee + range."""
        self.reset()
        if self._p is None or apogee <= 0:
            return

        step = nice_step(apogee)
        self._cx = x_center
        self._cy = y_center

        # Horizontal extent: span the downrange travel, with a sane minimum.
        horiz = max(abs(x_landing - x_center), abs(y_center), 2.0 * step, 2000.0)
        self._extent = horiz * 1.4

        # Levels from one step up to just past apogee
        levels = []
        h = step
        while h <= apogee * 1.05:
            levels.append(h)
            h += step
        self._levels = levels

        if self._planes_visible:
            self._draw_planes()
        self._build_scale_bar(apogee, step)
        if not self._bar_visible:
            self._set_bar_actor_visibility(False)

    def _draw_planes(self):
        size = self._extent
        for i, h in enumerate(self._levels):
            try:
                plane = pv.Plane(
                    center=(self._cx, self._cy, h), direction=(0, 0, 1),
                    i_size=size, j_size=size, i_resolution=1, j_resolution=1,
                )
                self._p.add_mesh(
                    plane, color=_PLANE_COLOR, opacity=0.07,
                    name=f"alt_plane_{i}", show_edges=True,
                    edge_color=_PLANE_COLOR, line_width=0.6, pickable=False,
                )
                self._plane_actors.append(f"alt_plane_{i}")
            except Exception as exc:
                logger.debug(f"alt plane {h} failed: {exc}")
        # One label set for all levels (cheaper, single billboard actor)
        try:
            pts = np.array([[self._cx + size * 0.5, self._cy, h] for h in self._levels])
            labels = [f"{h:.0f} m" for h in self._levels]
            if len(labels):
                self._p.add_point_labels(
                    pts, labels, font_size=10, text_color=_LABEL_COLOR,
                    point_size=0, name="alt_labels", shape=None,
                    always_visible=True,
                )
                self._plane_actors.append("alt_labels")
        except Exception as exc:
            logger.debug(f"alt labels failed: {exc}")

    def _build_scale_bar(self, apogee, step):
        # Vertical ruler at a back corner of the scene.
        bx = self._cx - self._extent * 0.5
        by = self._cy - self._extent * 0.5
        top = (int(apogee / step) + 1) * step
        try:
            line = pv.Line((bx, by, 0.0), (bx, by, top))
            self._p.add_mesh(line, color=_BAR_COLOR, line_width=2.5,
                             name="alt_scalebar", pickable=False)
            self._bar_actors.append("alt_scalebar")
        except Exception as exc:
            logger.debug(f"scale bar line failed: {exc}")

        ticks = []
        h = 0.0
        while h <= top + 1:
            ticks.append(h)
            h += step
        try:
            pts = np.array([[bx, by, t] for t in ticks])
            labels = [f"{t:.0f} m" for t in ticks]
            self._p.add_point_labels(
                pts, labels, font_size=10, text_color=_BAR_COLOR,
                point_size=4, name="alt_scalebar_labels", shape=None,
                always_visible=True,
            )
            self._bar_actors.append("alt_scalebar_labels")
        except Exception as exc:
            logger.debug(f"scale bar labels failed: {exc}")

    # ── visibility toggles ───────────────────────────────────────────

    def set_planes_visible(self, visible: bool):
        self._planes_visible = visible
        for name in self._plane_actors:
            self._set_actor_visibility(name, visible)

    def set_scalebar_visible(self, visible: bool):
        self._bar_visible = visible
        self._set_bar_actor_visibility(visible)

    def set_labels_visible(self, visible: bool):
        """LOD hook: hide text labels (keep planes/bar lines) when zoomed out."""
        self._set_actor_visibility("alt_labels", visible and self._planes_visible)
        self._set_actor_visibility("alt_scalebar_labels",
                                   visible and self._bar_visible)

    def _set_bar_actor_visibility(self, visible):
        for name in self._bar_actors:
            self._set_actor_visibility(name, visible)

    def _set_actor_visibility(self, name, visible):
        if self._p is None:
            return
        try:
            actor = self._p.renderer.actors.get(name)
            if actor is not None:
                actor.SetVisibility(bool(visible))
        except Exception:
            pass

    # ── teardown ─────────────────────────────────────────────────────

    def reset(self):
        for name in self._plane_actors + self._bar_actors:
            self._remove(name)
        self._plane_actors = []
        self._bar_actors = []
        self._levels = []

    def _remove(self, name):
        if self._p is None:
            return
        try:
            self._p.remove_actor(name)
        except Exception:
            pass
