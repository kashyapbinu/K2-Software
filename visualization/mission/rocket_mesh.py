"""
K2 AeroSim — Assembly Rocket Mesh Builder
============================================
Builds ONE merged PolyData of the actual rocket design (nose shape, body
tubes, transitions, fins, nozzle) for the Mission Visualizer, with per-vertex
RGB colors so a single actor renders the multi-color vehicle and can be posed
per frame with cheap VTK transforms.

Geometry convention matches visualization.viewer_3d (and the visualizer's
generic rocket actor): rocket along local +z, nozzle at z=0, nose tip at
z=total_length, real-size units (the actor applies the visual scale).
"""

import logging
import math

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:
    _PV_OK = False

from visualization.viewer_3d import (
    _ogive_profile, _make_surface_of_revolution, _make_tube, _make_frustum)
from core.components import (NoseCone, BodyTube, Transition,
                             TrapezoidalFinSet, LaunchLug, Nozzle)

logger = logging.getLogger("K2.RocketMesh")

_RES = 32   # revolution resolution (visualizer rocket is small on screen)

_COLORS = {
    "nosecone":   (58, 143, 214),
    "bodytube":   (140, 160, 178),
    "transition": (45, 138, 165),
    "fins":       (217, 79, 59),
    "nozzle":     (90, 90, 95),
    "lug":        (136, 136, 136),
}


def _nose_profile(shape, length, radius, n=40):
    """(zs, rs) profile from base (z=0, r=radius) to tip (z=length, r≈0)."""
    shape = (shape or "Ogive").lower()
    zs = np.linspace(0.0, length, n)
    if shape.startswith("conic"):
        rs = radius * (1.0 - zs / length)
    elif shape.startswith("ellip"):
        rs = radius * np.sqrt(np.maximum(0.0, 1.0 - (zs / length) ** 2))
    elif shape.startswith("parab"):
        rs = radius * (1.0 - (zs / length) ** 2)
    else:  # ogive / haack — reuse the design-viewer ogive
        return _ogive_profile(length, radius, n=n)
    rs = np.clip(rs, 0.0, radius)
    return zs, rs


def _colored(mesh, key):
    """Attach a per-point RGB array so merged meshes keep component colors."""
    rgb = np.tile(np.array(_COLORS[key], dtype=np.uint8), (mesh.n_points, 1))
    mesh.point_data["rgb"] = rgb
    return mesh


def _make_fin(body_r, height, root_chord, tip_chord, sweep_deg, thickness,
              angle_rad, z_start):
    """One solid trapezoidal fin (same convention as the design viewer):
    root TE at z_start (aft), root LE at z_start+root_chord, tip swept back."""
    sweep = height * math.tan(math.radians(sweep_deg)) if sweep_deg > 0 else 0.0
    pts = np.array([
        [body_r,          0, z_start],
        [body_r,          0, z_start + root_chord],
        [body_r + height, 0, z_start + root_chord - sweep],
        [body_r + height, 0, z_start + root_chord - sweep - tip_chord],
    ], dtype=float)
    t = max(thickness, 0.002)
    outer = pts.copy(); outer[:, 1] += t / 2
    inner = pts.copy(); inner[:, 1] -= t / 2
    all_pts = np.vstack([outer, inner])
    faces = np.array([
        4, 0, 1, 2, 3,
        4, 7, 6, 5, 4,
        4, 0, 4, 5, 1,
        4, 1, 5, 6, 2,
        4, 2, 6, 7, 3,
        4, 3, 7, 4, 0,
    ])
    fin = pv.PolyData(all_pts, faces=faces)
    return fin.rotate_z(math.degrees(angle_rad), point=(0, 0, 0))


def build_rocket_mesh(assembly):
    """Merged, vertex-colored PolyData of the assembly, or None on failure.

    Returns (mesh, total_length, max_radius)."""
    if not _PV_OK or assembly is None:
        return None
    try:
        total_len = assembly.total_length()
    except Exception:
        return None
    if total_len <= 0:
        return None

    parts = []
    z = total_len
    last_r = 0.03
    max_r = 0.0

    try:
        for stage in assembly.stages:
            for comp in stage.children:
                if isinstance(comp, NoseCone):
                    r = comp.diameter / 2
                    L_n = comp.length
                    L_sh = getattr(comp, "shoulder_length", 0.0)
                    z_base = z - L_n - L_sh
                    if L_sh > 0:
                        r_sh = getattr(comp, "shoulder_diameter", comp.diameter) / 2
                        r_sh = r_sh if r_sh > 0 else r * 0.95
                        parts.append(_colored(
                            _make_tube(z_base, L_sh, r_sh, n_theta=_RES),
                            "nosecone"))
                    zs, rs = _nose_profile(getattr(comp, "shape", "Ogive"), L_n, r)
                    parts.append(_colored(
                        _make_surface_of_revolution(zs + z_base + L_sh, rs,
                                                    n_theta=_RES), "nosecone"))
                    z, last_r = z_base, r
                elif isinstance(comp, BodyTube):
                    r = comp.outer_diameter_val / 2
                    z_base = z - comp.length
                    parts.append(_colored(
                        _make_tube(z_base, comp.length, r, n_theta=_RES),
                        "bodytube"))
                    for child in comp.children:
                        if isinstance(child, TrapezoidalFinSet):
                            for i in range(child.fin_count):
                                ang = 2 * math.pi * i / child.fin_count
                                parts.append(_colored(_make_fin(
                                    r, child.height, child.root_chord,
                                    child.tip_chord, child.sweep_angle,
                                    getattr(child, "thickness", 0.003),
                                    ang, z_base), "fins"))
                            max_r = max(max_r, r + child.height)
                    z, last_r = z_base, r
                elif isinstance(comp, Transition):
                    z_base = z - comp.length
                    r_top = comp.fore_diameter / 2
                    r_bot = comp.aft_diameter / 2
                    parts.append(_colored(
                        _make_frustum(z_base, comp.length, r_bot, r_top,
                                      n_theta=_RES), "transition"))
                    z, last_r = z_base, r_bot
                elif isinstance(comp, TrapezoidalFinSet):
                    for i in range(comp.fin_count):
                        ang = 2 * math.pi * i / comp.fin_count
                        parts.append(_colored(_make_fin(
                            last_r, comp.height, comp.root_chord,
                            comp.tip_chord, comp.sweep_angle,
                            getattr(comp, "thickness", 0.003), ang, z),
                            "fins"))
                    max_r = max(max_r, last_r + comp.height)
                elif isinstance(comp, LaunchLug):
                    z_base = z - comp.length
                    lug = pv.Cylinder(
                        center=(last_r + 0.004, 0, z_base + comp.length / 2),
                        direction=(0, 0, 1),
                        radius=comp.outer_diameter_val / 2,
                        height=comp.length, resolution=10)
                    parts.append(_colored(lug, "lug"))
                    z = z_base
                elif isinstance(comp, Nozzle):
                    z_base = z - comp.length
                    r_in = comp.inlet_diameter / 2
                    r_th = comp.throat_diameter / 2
                    r_ex = comp.exit_diameter / 2
                    if getattr(comp, "nozzle_type", "") == "Boat-Tail":
                        parts.append(_colored(
                            _make_frustum(z_base, comp.length, r_ex, r_in,
                                          n_theta=_RES), "nozzle"))
                    else:
                        L_div = comp.length * 0.6
                        parts.append(_colored(
                            _make_frustum(z_base, L_div, r_ex, r_th,
                                          n_theta=_RES), "nozzle"))
                        parts.append(_colored(
                            _make_frustum(z_base + L_div, comp.length * 0.4,
                                          r_th, r_in, n_theta=_RES), "nozzle"))
                    z, last_r = z_base, r_ex
                # inner/recovery components don't affect the outer mold line
                max_r = max(max_r, last_r)
    except Exception as exc:
        logger.warning(f"assembly mesh build failed: {exc}")
        return None

    if not parts:
        return None
    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    return mesh, total_len, max_r
