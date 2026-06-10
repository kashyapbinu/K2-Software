"""
K2 Aerospace — Mission Visualizer Flight Effects
=================================================
Scene-overlay effects for the live 3D flight view, following the established
overlay pattern (RecoveryVisualizer / FlightEnvelope): each class owns named
PyVista actors on the shared workspace plotter and is driven per-frame from
both the live and replay render paths.

    FlightEffects — exhaust flame, smoke trail, Mach cone, touchdown dust
    ForceVectors  — thrust / velocity / drag arrows on the rocket
    EventFlags    — 3D markers + labels at flight events on the trajectory

All geometry is built once (or per state-bucket) and posed per frame with
cheap VTK actor transforms — no per-frame mesh regeneration on hot paths.
"""

import math
import logging
from collections import deque
from time import perf_counter

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:
    _PV_OK = False

logger = logging.getLogger("K2.FlightEffects")


def _orient_for_axis(direction):
    """VTK SetOrientation tuple (x, y, z) that rotates local +z onto a world
    direction vector. VTK applies RotY first, then RotX, then RotZ; we use
    (0, tilt_about_y, azimuth_about_z): Rz(az)·Ry(tilt)·ẑ = direction."""
    dx, dy, dz = direction
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        return (0.0, 0.0, 0.0)
    dx, dy, dz = dx / n, dy / n, dz / n
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, dz))))
    az = math.degrees(math.atan2(dy, dx))
    return (0.0, tilt, az)


def _body_axis(pitch, yaw):
    """Unit body axis (nose direction) from engine pitch/yaw convention:
    (cosθ·cosψ, cosθ·sinψ, sinθ)."""
    cp = math.cos(pitch)
    return (cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch))


class FlightEffects:
    """Exhaust flame + smoke trail + Mach cone + touchdown dust."""

    _SMOKE_CAP = 400          # max smoke puffs kept
    _SMOKE_DT = 0.08          # sim-seconds between puffs
    _DUST_DURATION = 1.6      # wall-seconds for the touchdown dust ring

    def __init__(self, plotter, vis_scale=30.0):
        self._p = plotter
        self._scale = vis_scale
        self._visible = True

        self._flame_core = None
        self._flame_glow = None
        self._smoke_actor = None
        self._mach_actor = None
        self._mach_bucket = None
        self._dust_actor = None

        self._smoke = deque(maxlen=self._SMOKE_CAP)   # (x, y, z, t_birth)
        self._smoke_last_t = -1e9
        self._smoke_dirty = False
        self._dust_start = None
        self._dust_xy = None
        self._dust_done = False

        self._build_flame()

    # ── flame ────────────────────────────────────────────────────────────────

    def _build_flame(self):
        """Unit flame cones pointing local -z (apex away from the nozzle).
        Posed + scaled per frame; never rebuilt."""
        if not _PV_OK or self._p is None:
            return
        try:
            core = pv.Cone(center=(0, 0, -0.5), direction=(0, 0, -1),
                           height=1.0, radius=0.30, resolution=16)
            glow = pv.Cone(center=(0, 0, -0.55), direction=(0, 0, -1),
                           height=1.25, radius=0.50, resolution=16)
            self._flame_core = self._p.add_mesh(
                core, color="#ffd166", opacity=0.95, name="fx_flame_core",
                smooth_shading=True, ambient=0.9, diffuse=0.1)
            self._flame_glow = self._p.add_mesh(
                glow, color="#ff7b29", opacity=0.35, name="fx_flame_glow",
                smooth_shading=True, ambient=0.9, diffuse=0.1)
            self._flame_core.SetVisibility(False)
            self._flame_glow.SetVisibility(False)
        except Exception as exc:
            logger.debug(f"flame build failed: {exc}")
            self._flame_core = self._flame_glow = None

    def _update_flame(self, x, y, z, orient, thrust_frac, sim_time,
                      rocket_len, rocket_diam):
        on = self._visible and thrust_frac > 0.02
        for a in (self._flame_core, self._flame_glow):
            if a is None:
                continue
            a.SetVisibility(on)
        if not on or self._flame_core is None:
            return
        # Flicker: fast multi-sine so it looks alive at any frame rate
        flick = 1.0 + 0.18 * math.sin(37.0 * sim_time) * math.cos(23.0 * sim_time)
        length = self._scale * rocket_len * (0.35 + 0.85 * thrust_frac) * flick
        width = self._scale * rocket_diam * (1.6 + 0.8 * thrust_frac)
        for a, k in ((self._flame_core, 1.0), (self._flame_glow, 1.25)):
            try:
                a.SetScale(width * k, width * k, length * k)
                a.SetOrientation(*orient)
                a.SetPosition(x, y, z)
            except Exception:
                pass

    # ── smoke ────────────────────────────────────────────────────────────────

    def _update_smoke(self, x, y, z, thrust_frac, sim_time):
        if thrust_frac > 0.02 and sim_time - self._smoke_last_t >= self._SMOKE_DT:
            self._smoke_last_t = sim_time
            self._smoke.append((x, y, z, sim_time))
            self._smoke_dirty = True
        if not self._smoke_dirty or not self._smoke:
            return
        self._smoke_dirty = False
        if not _PV_OK or self._p is None:
            return
        pts = np.array([(sx, sy, sz) for sx, sy, sz, _ in self._smoke], dtype=float)
        births = np.array([b for *_, b in self._smoke], dtype=float)
        age = sim_time - births
        # young → light grey, old → fades into the dark background
        scal = np.clip(age / 8.0, 0.0, 1.0)
        try:
            cloud = pv.PolyData(pts)
            cloud["age"] = scal
            actor = self._p.add_mesh(
                cloud, scalars="age", cmap="Greys", clim=[-0.6, 1.4],
                name="fx_smoke", point_size=13.0,
                render_points_as_spheres=True, opacity=0.35,
                show_scalar_bar=False,
            )
            actor.SetVisibility(self._visible)
            self._smoke_actor = actor
        except Exception as exc:
            logger.debug(f"smoke update failed: {exc}")

    # ── Mach cone ────────────────────────────────────────────────────────────

    def _update_mach_cone(self, x, y, z, orient, mach, rocket_len):
        show = self._visible and mach > 1.02
        if not show:
            if self._mach_actor is not None:
                self._mach_actor.SetVisibility(False)
            return
        bucket = round(mach * 10.0)   # rebuild geometry per 0.1-Mach step
        if bucket != self._mach_bucket or self._mach_actor is None:
            self._mach_bucket = bucket
            if self._mach_actor is not None:
                try:
                    self._p.remove_actor(self._mach_actor)
                except Exception:
                    pass
                self._mach_actor = None
            mu = math.asin(min(1.0, 1.0 / max(mach, 1.0001)))
            H = 3.0 * rocket_len
            try:
                # Apex at local nose tip (0,0,L), opening backward (down -z)
                cone = pv.Cone(center=(0, 0, rocket_len - H / 2.0),
                               direction=(0, 0, 1), height=H,
                               radius=H * math.tan(mu), resolution=28, capping=False)
                self._mach_actor = self._p.add_mesh(
                    cone, color="#9ecbff", opacity=0.10, name="fx_machcone",
                    smooth_shading=True)
            except Exception as exc:
                logger.debug(f"mach cone build failed: {exc}")
                return
        a = self._mach_actor
        if a is None:
            return
        try:
            a.SetVisibility(True)
            a.SetScale(self._scale)
            a.SetOrientation(*orient)
            a.SetPosition(x, y, z)
        except Exception:
            pass

    # ── touchdown dust ───────────────────────────────────────────────────────

    def trigger_dust(self, x, y):
        """Start the expanding dust ring at the landing point (idempotent)."""
        if self._dust_done or self._dust_start is not None:
            return
        self._dust_start = perf_counter()
        self._dust_xy = (x, y)

    def _update_dust(self):
        if self._dust_start is None or not _PV_OK or self._p is None:
            return
        frac = (perf_counter() - self._dust_start) / self._DUST_DURATION
        if frac >= 1.0:
            if self._dust_actor is not None:
                try:
                    self._p.remove_actor(self._dust_actor)
                except Exception:
                    pass
                self._dust_actor = None
            self._dust_start = None
            self._dust_done = True
            return
        x, y = self._dust_xy
        r = 30.0 + 260.0 * frac
        try:
            ring = pv.Disc(center=(x, y, 2.0), normal=(0, 0, 1),
                           inner=r * 0.55, outer=r, r_res=2, c_res=32)
            actor = self._p.add_mesh(
                ring, color="#c9b896", opacity=0.5 * (1.0 - frac),
                name="fx_dust", show_edges=False)
            actor.SetVisibility(self._visible)
            self._dust_actor = actor
        except Exception:
            pass

    # ── public API ───────────────────────────────────────────────────────────

    def update(self, x, y, z, pitch, yaw, thrust_frac, mach, sim_time,
               rocket_len, rocket_diam, landed=False):
        """Per-frame effects update (live or replay)."""
        orient = _orient_for_axis(_body_axis(pitch, yaw))
        self._update_flame(x, y, z, orient, thrust_frac, sim_time,
                           rocket_len, rocket_diam)
        self._update_smoke(x, y, z, thrust_frac, sim_time)
        self._update_mach_cone(x, y, z, orient, mach, rocket_len)
        if landed:
            self.trigger_dust(x, y)
        self._update_dust()

    def reset(self):
        self._smoke.clear()
        self._smoke_last_t = -1e9
        self._smoke_dirty = False
        self._dust_start = None
        self._dust_done = False
        self._mach_bucket = None
        for name in ("fx_smoke", "fx_machcone", "fx_dust"):
            actor = self._p.renderer.actors.get(name) if self._p else None
            if actor is not None:
                try:
                    self._p.remove_actor(actor)
                except Exception:
                    pass
        self._smoke_actor = self._mach_actor = self._dust_actor = None
        for a in (self._flame_core, self._flame_glow):
            if a is not None:
                a.SetVisibility(False)

    def set_visible(self, on):
        self._visible = bool(on)
        if not self._visible:
            # flame/mach/smoke re-evaluate visibility on next update();
            # force-hide everything now
            for a in (self._flame_core, self._flame_glow, self._smoke_actor,
                      self._mach_actor, self._dust_actor):
                if a is not None:
                    try:
                        a.SetVisibility(False)
                    except Exception:
                        pass


class ForceVectors:
    """Thrust / velocity / drag arrows posed on the rocket each frame.

    Unit pv.Arrow meshes (along +x) built once; per-frame pose via actor
    transforms. Arrow length = base + magnitude-relative growth.
    """

    _SPECS = [
        ("thrust", "#ffa657"),
        ("velocity", "#58d0ff"),
        ("drag", "#f85149"),
    ]

    def __init__(self, plotter, vis_scale=30.0):
        self._p = plotter
        self._scale = vis_scale
        self._visible = False
        self._actors = {}
        if not _PV_OK or plotter is None:
            return
        try:
            arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0),
                             tip_length=0.22, tip_radius=0.06,
                             shaft_radius=0.025)
            for name, color in self._SPECS:
                a = self._p.add_mesh(arrow, color=color, name=f"vec_{name}",
                                     smooth_shading=True, ambient=0.6)
                a.SetVisibility(False)
                self._actors[name] = a
        except Exception as exc:
            logger.debug(f"vector build failed: {exc}")
            self._actors = {}

    @staticmethod
    def _orient_for_vector(v):
        """SetOrientation tuple rotating local +x onto world vector v."""
        vx, vy, vz = v
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        if n < 1e-9:
            return None
        az = math.degrees(math.atan2(vy, vx))
        elev = math.degrees(math.asin(max(-1.0, min(1.0, vz / n))))
        # RotY applied first (pitch the +x arrow up/down), then RotZ (azimuth)
        return (0.0, -elev, az)

    def _pose(self, name, pos, direction, length):
        a = self._actors.get(name)
        if a is None:
            return
        orient = self._orient_for_vector(direction)
        if orient is None or length <= 1e-3:
            a.SetVisibility(False)
            return
        try:
            a.SetVisibility(self._visible)
            a.SetScale(length, length * 0.35, length * 0.35)
            a.SetOrientation(*orient)
            a.SetPosition(*pos)
        except Exception:
            pass

    def update(self, x, y, z, pitch, yaw, vel_vec, thrust_n, drag_n,
               rocket_len):
        """Pose all three arrows. Magnitudes scale length logarithmically so
        a 10x force difference reads as a visibly longer (not 10x) arrow."""
        if not self._actors:
            return
        if not self._visible:
            for a in self._actors.values():
                a.SetVisibility(False)
            return
        base = self._scale * rocket_len
        axis = _body_axis(pitch, yaw)
        pos = (x, y, z)

        def vis_len(mag, ref):
            if mag <= ref * 1e-3:
                return 0.0
            return base * (0.6 + 0.45 * math.log10(1.0 + 9.0 * mag / ref))

        speed = math.sqrt(sum(c * c for c in vel_vec))
        self._pose("thrust", pos, axis, vis_len(thrust_n, 100.0))
        self._pose("velocity", pos, vel_vec, vis_len(speed, 100.0))
        drag_dir = tuple(-c for c in vel_vec)
        self._pose("drag", pos, drag_dir, vis_len(drag_n, 100.0))

    def set_visible(self, on):
        self._visible = bool(on)
        if not on:
            for a in self._actors.values():
                try:
                    a.SetVisibility(False)
                except Exception:
                    pass

    def reset(self):
        for a in self._actors.values():
            try:
                a.SetVisibility(False)
            except Exception:
                pass


class EventFlags:
    """3D markers + billboard labels at flight events along the trajectory."""

    def __init__(self, plotter):
        self._p = plotter
        self._visible = True
        self._names = []   # actor name suffixes added this flight

    def add(self, key, label, color, xyz):
        """Drop a marker sphere + label at the event position."""
        if not _PV_OK or self._p is None or key in self._names:
            return
        x, y, z = xyz
        try:
            marker = pv.Sphere(radius=22.0, center=(x, y, z),
                               theta_resolution=12, phi_resolution=12)
            a = self._p.add_mesh(marker, color=color, name=f"flag_{key}",
                                 opacity=0.85, smooth_shading=True)
            a.SetVisibility(self._visible)
            la = self._p.add_point_labels(
                np.array([(x, y, z)], dtype=float), [f"  {label}"],
                name=f"flag_lbl_{key}", font_size=11, text_color=color,
                shape=None, point_size=0, always_visible=True)
            try:
                la.SetVisibility(self._visible)
            except Exception:
                pass
            self._names.append(key)
        except Exception as exc:
            logger.debug(f"event flag '{key}' failed: {exc}")

    def _actor_names(self, key):
        """All actor names for one flag. add_point_labels registers suffixed
        actors (e.g. 'flag_lbl_apogee-points' / '-labels'), so match by prefix."""
        try:
            names = list(self._p.renderer.actors.keys())
        except Exception:
            return []
        return [n for n in names
                if n == f"flag_{key}" or n.startswith(f"flag_lbl_{key}")]

    def clear(self):
        for key in self._names:
            for name in self._actor_names(key):
                try:
                    self._p.remove_actor(name)
                except Exception:
                    pass
        self._names = []

    def set_visible(self, on):
        self._visible = bool(on)
        for key in self._names:
            for name in self._actor_names(key):
                actor = self._p.renderer.actors.get(name)
                if actor is not None:
                    try:
                        actor.SetVisibility(self._visible)
                    except Exception:
                        pass
