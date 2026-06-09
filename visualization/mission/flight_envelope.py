"""
K2 Aerospace — Flight Envelope Visualization
==============================================
Mission-planning and recovery-safety overlay for the Mission Visualizer.
This is the layer that pushes K2 past a plain trajectory viewer toward a
professional flight-analysis tool.

Renders (all as independent, persistent named actors):
    * target apogee plane (green) + label
    * actual apogee marker + altitude / delta-to-target
    * safe recovery radius ring + label, colour-coded by outcome
    * predicted landing zone (confidence ellipse) + label
    * wind drift envelope (translucent cone apogee -> landing) + label
    * launch (white) / predicted (green) / actual (blue) landing markers
      with downrange distances

Monte-Carlo ready: `set_trajectory_cloud`, `set_landing_points`, and
`set_landing_ellipse` accept aggregate inputs so a future MC engine can feed
trajectory bundles, landing heatmaps, and confidence ellipses without any
redesign — the single-run path is just the N=1 case of the same API.

PyVista 0.48 notes: pv.Disc for rings, pv.Plane for planes, custom PolyData
for the ellipse + cone, add_point_labels for billboard text. All defensive.
"""

import logging
import math

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:  # pragma: no cover
    _PV_OK = False

logger = logging.getLogger("K2.MissionViz.Envelope")

_TARGET_COLOR    = "#3fb950"
_LAUNCH_COLOR    = "#f0f6fc"
_PREDICTED_COLOR = "#56d364"
_ACTUAL_COLOR    = "#58a6ff"
_DRIFT_COLOR     = "#d29922"

# Outcome colours (success zone)
_OK_COLOR   = "#3fb950"   # well inside recovery radius
_WARN_COLOR = "#d29922"   # near boundary
_BAD_COLOR  = "#f85149"   # outside safe area


def estimate_landing(apogee_x, apogee_y, apogee_alt,
                     wind_speed, wind_dir_deg, descent_rate):
    """Rough downwind landing estimate from apogee.

    Wind direction is the bearing the wind blows FROM (0=N, 90=E), matching
    the project WindModel; the rocket drifts toward dir+180.
    Returns (x, y, drift_distance).
    """
    if descent_rate <= 0.01 or apogee_alt <= 0:
        return apogee_x, apogee_y, 0.0
    descent_time = apogee_alt / descent_rate
    drift = wind_speed * descent_time
    blow_to = math.radians(wind_dir_deg + 180.0)
    dx = drift * math.cos(blow_to)
    dy = drift * math.sin(blow_to)
    return apogee_x + dx, apogee_y + dy, drift


def _ellipse_points(cx, cy, a, b, angle_rad, n=48, z=1.0):
    """Rim points of an ellipse (semi-axes a,b) rotated by angle_rad on ground."""
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        ex, ey = a * math.cos(t), b * math.sin(t)
        pts.append([cx + ex * ca - ey * sa, cy + ex * sa + ey * ca, z])
    return np.array(pts, dtype=float)


class FlightEnvelope:
    """Mission-planning / safety overlay for the Mission Visualizer."""

    def __init__(self, plotter):
        self._p = plotter
        self._visible = True
        self._actors = set()
        self._launch = (0.0, 0.0)
        self._recovery_radius = 1000.0
        self._target_apogee = 0.0
        self._extent = 3000.0

    # ── configuration / static build ─────────────────────────────────

    def build_static(self, launch_xy=(0.0, 0.0), recovery_radius=1000.0,
                     target_apogee=0.0, extent_hint=3000.0):
        """Build launch-site-fixed geometry: recovery ring + target plane."""
        self.reset()
        if self._p is None:
            return
        self._launch = launch_xy
        self._recovery_radius = max(1.0, recovery_radius)
        self._target_apogee = max(0.0, target_apogee)
        self._extent = max(extent_hint, self._recovery_radius * 2.5, 3000.0)

        self._draw_recovery_ring(_OK_COLOR)
        self._draw_launch_marker()
        if self._target_apogee > 0:
            self._draw_target_plane()

    def _draw_recovery_ring(self, color):
        lx, ly = self._launch
        r = self._recovery_radius
        try:
            ring = pv.Disc(center=(lx, ly, 1.0), normal=(0, 0, 1),
                           inner=r * 0.97, outer=r, r_res=2, c_res=64)
            self._add(ring, "env_recovery_ring", color=color, opacity=0.9)
            fill = pv.Disc(center=(lx, ly, 0.5), normal=(0, 0, 1),
                           inner=0.0, outer=r, r_res=2, c_res=64)
            self._add(fill, "env_recovery_fill", color=color, opacity=0.05)
            self._label([[lx, ly + r, 5.0]], ["SAFE RECOVERY ZONE"],
                        "env_recovery_label", color)
        except Exception as exc:
            logger.debug(f"recovery ring failed: {exc}")

    def _draw_launch_marker(self):
        lx, ly = self._launch
        r = max(self._recovery_radius * 0.012, 8.0)
        try:
            # Flat low-profile pad marker (no floating ball)
            disc = pv.Disc(center=(lx, ly, 1.0), normal=(0, 0, 1),
                           inner=r * 0.45, outer=r, r_res=2, c_res=28)
            self._add(disc, "env_launch_marker", color=_LAUNCH_COLOR, opacity=0.9)
            self._label([[lx, ly, r * 3]], ["LAUNCH"], "env_launch_label",
                        _LAUNCH_COLOR)
        except Exception as exc:
            logger.debug(f"launch marker failed: {exc}")

    def _draw_target_plane(self):
        lx, ly = self._launch
        size = self._extent
        try:
            plane = pv.Plane(center=(lx, ly, self._target_apogee),
                             direction=(0, 0, 1), i_size=size, j_size=size,
                             i_resolution=1, j_resolution=1)
            self._add(plane, "env_target_plane", color=_TARGET_COLOR,
                      opacity=0.12, show_edges=True, edge_color=_TARGET_COLOR,
                      line_width=1.0)
            self._label([[lx - size * 0.5, ly, self._target_apogee]],
                        [f"TARGET APOGEE  {self._target_apogee:.0f} m"],
                        "env_target_label", _TARGET_COLOR)
        except Exception as exc:
            logger.debug(f"target plane failed: {exc}")

    # ── apogee ───────────────────────────────────────────────────────

    def on_apogee(self, x, y, z, actual_apogee):
        """Mark apogee with a text label only (no floating sphere)."""
        if self._p is None:
            return
        mr = max(self._recovery_radius * 0.02, 12.0)
        try:
            txt = f"* APOGEE  {actual_apogee:.0f} m"
            if self._target_apogee > 0:
                delta = actual_apogee - self._target_apogee
                txt += f"  ({delta:+.0f} m vs target)"
            self._label([[x, y, z + mr * 3]], [txt], "env_apogee_label",
                        _TARGET_COLOR)
        except Exception as exc:
            logger.debug(f"apogee marker failed: {exc}")

    # ── predicted landing + drift envelope ───────────────────────────

    def set_predicted_landing(self, x, y, apogee_xyz=None,
                              ellipse_axes=None, wind_dir_deg=0.0):
        """Predicted landing marker, confidence ellipse, and wind drift cone.

        ellipse_axes: (a, b) semi-axes (m). If None, a small default is used.
        apogee_xyz:   if given, draws the translucent drift cone apogee->zone.
        """
        if self._p is None:
            return
        lx, ly = self._launch
        mr = max(self._recovery_radius * 0.02, 12.0)

        if ellipse_axes is None:
            ellipse_axes = (self._recovery_radius * 0.25,
                            self._recovery_radius * 0.15)
        a, b = ellipse_axes
        # Major axis aligned downwind
        angle = math.radians(wind_dir_deg + 180.0)

        try:
            rim = _ellipse_points(x, y, a, b, angle, n=48, z=1.5)
            # Filled translucent ellipse (triangle fan)
            center = np.array([[x, y, 1.5]])
            pts = np.vstack([center, rim])
            n = len(rim)
            faces = []
            for i in range(n):
                faces.extend([3, 0, 1 + i, 1 + (i + 1) % n])
            poly = pv.PolyData(pts, np.array(faces))
            self._add(poly, "env_landing_zone", color=_PREDICTED_COLOR,
                      opacity=0.12)
            # Ellipse outline
            line_pts = np.vstack([rim, rim[:1]])
            outline = pv.lines_from_points(line_pts)
            self._add(outline, "env_landing_outline", color=_PREDICTED_COLOR,
                      opacity=0.8, line_width=2.0)
            self._label([[x, y + b, 5.0]], ["EXPECTED LANDING AREA"],
                        "env_landing_label", _PREDICTED_COLOR)

            # Predicted marker
            sph = pv.Sphere(radius=mr, center=(x, y, mr))
            self._add(sph, "env_predicted_marker", color=_PREDICTED_COLOR)
            dist = math.hypot(x - lx, y - ly)
            self._label([[x, y, mr * 3]], [f"PREDICTED  {dist:.0f} m"],
                        "env_predicted_dist", _PREDICTED_COLOR)
        except Exception as exc:
            logger.debug(f"predicted landing failed: {exc}")

        if apogee_xyz is not None:
            self._draw_drift_cone(apogee_xyz, x, y, a, b, angle)

    def _draw_drift_cone(self, apogee_xyz, lx_pred, ly_pred, a, b, angle):
        ax, ay, az = apogee_xyz
        try:
            rim = _ellipse_points(lx_pred, ly_pred, a, b, angle, n=32, z=1.5)
            apex = np.array([[ax, ay, az]])
            pts = np.vstack([apex, rim])
            n = len(rim)
            faces = []
            for i in range(n):
                faces.extend([3, 0, 1 + i, 1 + (i + 1) % n])
            cone = pv.PolyData(pts, np.array(faces))
            self._add(cone, "env_drift_cone", color=_DRIFT_COLOR, opacity=0.10)
            mid = [(ax + lx_pred) * 0.5, (ay + ly_pred) * 0.5, az * 0.5]
            self._label([mid], ["WIND DRIFT ENVELOPE"], "env_drift_label",
                        _DRIFT_COLOR)
        except Exception as exc:
            logger.debug(f"drift cone failed: {exc}")

    # ── actual landing + outcome ─────────────────────────────────────

    def set_actual_landing(self, x, y):
        """Actual landing marker + distance, and colour the success zone."""
        if self._p is None:
            return
        lx, ly = self._launch
        mr = max(self._recovery_radius * 0.02, 12.0)
        dist = math.hypot(x - lx, y - ly)
        outcome = self.classify_outcome(dist)

        try:
            sph = pv.Sphere(radius=mr * 1.05, center=(x, y, mr))
            self._add(sph, "env_actual_marker", color=_ACTUAL_COLOR)
            self._label([[x, y, mr * 4]],
                        [f"LANDING  {dist:.0f} m  ({outcome.upper()})"],
                        "env_actual_dist", self._outcome_color(outcome))
        except Exception as exc:
            logger.debug(f"actual landing failed: {exc}")

        # Recolour recovery ring + fill to reflect outcome
        self._recolor_recovery(self._outcome_color(outcome))

    def classify_outcome(self, dist):
        if dist <= self._recovery_radius * 0.8:
            return "safe"
        if dist <= self._recovery_radius:
            return "marginal"
        return "outside"

    def _outcome_color(self, outcome):
        return {"safe": _OK_COLOR, "marginal": _WARN_COLOR,
                "outside": _BAD_COLOR}.get(outcome, _OK_COLOR)

    def _recolor_recovery(self, color):
        lx, ly = self._launch
        r = self._recovery_radius
        try:
            ring = pv.Disc(center=(lx, ly, 1.0), normal=(0, 0, 1),
                           inner=r * 0.97, outer=r, r_res=2, c_res=64)
            self._add(ring, "env_recovery_ring", color=color, opacity=0.9)
            fill = pv.Disc(center=(lx, ly, 0.5), normal=(0, 0, 1),
                           inner=0.0, outer=r, r_res=2, c_res=64)
            self._add(fill, "env_recovery_fill", color=color, opacity=0.05)
        except Exception:
            pass

    # ── Monte-Carlo aggregate API ────────────────────────────────────

    def set_trajectory_cloud(self, trajectories, color="#3b5168", opacity=0.25):
        """Render a bundle of trajectories (list of Nx3 arrays).

        Single-run is just len==1. Heavy decimation keeps it cheap; designed
        so an MC engine can drop its run bundle straight in.
        """
        if self._p is None:
            return
        self._remove("env_mc_cloud")
        try:
            blocks = []
            for traj in trajectories:
                arr = np.asarray(traj, dtype=float)
                if arr.ndim == 2 and len(arr) >= 2:
                    blocks.append(pv.lines_from_points(arr))
            if blocks:
                merged = blocks[0]
                for b in blocks[1:]:
                    merged = merged.merge(b)
                self._add(merged, "env_mc_cloud", color=color, opacity=opacity,
                          line_width=1.0)
        except Exception as exc:
            logger.debug(f"trajectory cloud failed: {exc}")

    def set_landing_points(self, points, color="#58a6ff"):
        """Scatter of landing points (MC landing heatmap input)."""
        if self._p is None:
            return
        self._remove("env_mc_landings")
        try:
            arr = np.asarray(points, dtype=float)
            if arr.ndim == 2 and len(arr):
                if arr.shape[1] == 2:
                    arr = np.column_stack([arr, np.full(len(arr), 2.0)])
                cloud = pv.PolyData(arr)
                self._add(cloud, "env_mc_landings", color=color,
                          opacity=0.6, point_size=6, render_points_as_spheres=True)
        except Exception as exc:
            logger.debug(f"landing points failed: {exc}")

    def set_landing_ellipse(self, cx, cy, a, b, angle_deg=0.0,
                            color="#bc8cff", label="CONFIDENCE ELLIPSE"):
        """Explicit confidence ellipse (e.g. MC 1-sigma)."""
        if self._p is None:
            return
        self._remove("env_conf_ellipse")
        self._remove("env_conf_label")
        try:
            rim = _ellipse_points(cx, cy, a, b, math.radians(angle_deg),
                                  n=48, z=2.0)
            line_pts = np.vstack([rim, rim[:1]])
            outline = pv.lines_from_points(line_pts)
            self._add(outline, "env_conf_ellipse", color=color,
                      opacity=0.9, line_width=2.0)
            self._label([[cx, cy + b, 5.0]], [label], "env_conf_label", color)
        except Exception as exc:
            logger.debug(f"confidence ellipse failed: {exc}")

    # ── visibility / teardown ────────────────────────────────────────

    def set_visible(self, visible: bool):
        self._visible = visible
        for name in list(self._actors):
            self._set_actor_visibility(name, visible)

    def set_labels_visible(self, visible: bool):
        """LOD hook: hide text labels (keep geometry) when zoomed far out."""
        show = visible and self._visible
        for name in list(self._actors):
            if "label" in name or name.endswith("_dist"):
                self._set_actor_visibility(name, show)

    def _set_actor_visibility(self, name, visible):
        if self._p is None:
            return
        try:
            actor = self._p.renderer.actors.get(name)
            if actor is not None:
                actor.SetVisibility(bool(visible))
        except Exception:
            pass

    def reset(self):
        for name in list(self._actors):
            self._remove(name)
        self._actors.clear()

    # ── low-level helpers ────────────────────────────────────────────

    def _add(self, mesh, name, **kw):
        try:
            self._p.add_mesh(mesh, name=name, show_scalar_bar=False, **kw)
            self._actors.add(name)
            if not self._visible:
                self._set_actor_visibility(name, False)
        except Exception as exc:
            logger.debug(f"add_mesh {name} failed: {exc}")

    def _label(self, pts, labels, name, color):
        try:
            self._p.add_point_labels(
                np.array(pts, dtype=float), labels, font_size=11,
                text_color=color, point_size=0, name=name, shape=None,
                always_visible=True,
            )
            self._actors.add(name)
            if not self._visible:
                self._set_actor_visibility(name, False)
        except Exception as exc:
            logger.debug(f"label {name} failed: {exc}")

    def _remove(self, name):
        if self._p is None:
            return
        try:
            self._p.remove_actor(name)
        except Exception:
            pass
        self._actors.discard(name)
