"""
K2 AeroSim — Geometry Exporter
==================================
Converts a K2 RocketAssembly (or an external CAD file) into a watertight
triangulated STL surface suitable for CFD meshing.

For K2 assemblies we reconstruct the surfaces from the same geometry
functions used in Viewer3D, then merge and export via PyVista.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv

logger = logging.getLogger("K2.CFD.GeoExport")

# ── Reuse viewer geometry helpers ────────────────────────────────────────────
# We import the same private helpers used in viewer_3d so geometry is
# guaranteed to match the visual representation exactly.

try:
    from visualization.viewer_3d import (
        _ogive_profile, _make_surface_of_revolution, _make_tube, _make_frustum,
        nose_profile as _shared_nose_profile,
    )
except ImportError:
    # Fallback stubs if viewer is not importable (headless environment)
    def _make_tube(z_base, length, radius, n_theta=64):
        return pv.Cylinder(
            center=(0, 0, z_base + length / 2),
            direction=(0, 0, 1),
            radius=radius,
            height=length,
            resolution=n_theta,
        )

    def _ogive_profile(length, radius, n=50):
        if radius <= 0 or length <= 0:
            return np.array([0, length]), np.array([radius, 0])
        rho = (radius ** 2 + length ** 2) / (2 * radius)
        zs = np.linspace(0, length, n)
        rs = np.sqrt(np.maximum(rho ** 2 - zs ** 2, 0)) - (rho - radius)
        rs = np.clip(rs, 0, radius)
        return zs, rs

    def _shared_nose_profile(shape, length, radius, n=50):
        """Headless twin of visualization.viewer_3d.nose_profile.

        Must stay shape-aware: the CFD mesh is built from this in a subprocess
        where the viewer (and Qt) may not import, and a cone silently becoming
        an ogive there would move the geometry the solver actually solves.
        """
        if radius <= 0 or length <= 0:
            return np.array([0.0, length]), np.array([radius, 0.0])
        key = (shape or "Ogive").strip().lower()
        if key.startswith("conic"):
            return np.array([0.0, length]), np.array([radius, 0.0])
        zs = np.linspace(0.0, length, max(int(n), 2))
        t = zs / length
        if key.startswith("ellip"):
            rs = radius * np.sqrt(np.maximum(1.0 - t ** 2, 0.0))
        elif key.startswith("parab"):
            rs = radius * (1.0 - t ** 2)
        elif key.startswith("haack") or "karman" in key or "kármán" in key:
            x = np.clip(1.0 - t, 0.0, 1.0)
            theta = np.arccos(np.clip(1.0 - 2.0 * x, -1.0, 1.0))
            rs = (radius / np.sqrt(np.pi)) * np.sqrt(
                np.maximum(theta - 0.5 * np.sin(2.0 * theta), 0.0)
            )
        elif key.startswith("agard"):
            # AGARD-B ogive (AGARD AG-4/M3); see visualization.viewer_3d.
            s = np.clip(1.0 - t, 0.0, 1.0)
            rs = radius * (2.0 * s - 2.0 * s ** 3 + s ** 4)
        else:
            return _ogive_profile(length, radius, n)
        return zs, np.clip(rs, 0.0, radius)

    def _make_surface_of_revolution(zs, rs, n_theta=64):
        thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        pts = []
        for z, r in zip(zs, rs):
            for t in thetas:
                pts.append([r * np.cos(t), r * np.sin(t), z])
        pts = np.array(pts)
        faces = []
        n_z, n_t = len(zs), n_theta
        for i in range(n_z - 1):
            for j in range(n_t):
                j1 = (j + 1) % n_t
                a = i * n_t + j
                b = i * n_t + j1
                c = (i + 1) * n_t + j1
                d = (i + 1) * n_t + j
                faces.extend([4, a, b, c, d])
        return pv.PolyData(pts, np.array(faces))

    def _make_frustum(z_base, length, r_bot, r_top, n_theta=64):
        return _make_tube(z_base, length, max(r_bot, r_top), n_theta)


from core.components import (
    NoseCone, BodyTube, Transition, TrapezoidalFinSet,
    InnerTube, Stage
)


def extract_cfd_geometry(assembly) -> dict:
    """
    Walk the K2 RocketAssembly component tree and extract EXACT geometry
    parameters for CFD meshing — no STL estimation needed.

    Returns a dict compatible with build_wind_tunnel_mesh(geometry_dict=...).
    All lengths in metres.  Fin dimensions are from the TrapezoidalFinSet.
    """
    total_L = assembly.total_length()
    if total_L <= 0:
        raise ValueError("Assembly has zero length.")

    # K2 stacks components from nose tip downward (z decreasing)
    z_cursor = total_L   # start at nose tip

    nose_L  = 0.0
    nose_r  = 0.03    # fallback
    body_r  = 0.03
    body_L  = 0.0
    fins    = []      # list of fin parameter dicts

    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                nose_r = comp.diameter / 2
                nose_L = comp.length + getattr(comp, "shoulder_length", 0.0)
                z_cursor -= nose_L

            elif isinstance(comp, BodyTube):
                body_r = comp.outer_diameter_val / 2
                body_L += comp.length
                z_cursor -= comp.length

                # Fins attached to this tube (z_cursor = bottom of tube = nozzle end)
                for child in comp.children:
                    if isinstance(child, TrapezoidalFinSet):
                        sweep_deg = getattr(child, "sweep_angle", 0.0)
                        fins.append({
                            "count":      child.fin_count,
                            "height":     child.height,
                            "root_chord": child.root_chord,
                            "tip_chord":  getattr(child, "tip_chord", child.root_chord * 0.5),
                            "sweep_deg":  sweep_deg,
                            "thick":      max(0.002, getattr(child, "thickness", 0.003)),
                            # Square / Rounded / Airfoil. The mesher used to
                            # extrude every fin as a square-edged plank whatever
                            # this said, which puts a blunt face across the whole
                            # leading edge — 4% of chord on AGARD-B.
                            "cross_section": getattr(child, "cross_section", "Square"),
                            # z_cursor here = nozzle end of this body tube
                            "z_base_k2":  z_cursor,
                        })

            elif isinstance(comp, Transition):
                body_r = max(comp.fore_diameter, comp.aft_diameter) / 2
                body_L += comp.length
                z_cursor -= comp.length

    # Nose-only / pure-cone geometry has no body tube, so body_r is still the
    # fallback — use the nose base radius so the CFD reference area is correct.
    if body_L <= 0:
        body_r = nose_r

    # Pick the largest fin set (most aerodynamically significant).
    #
    # A finless design reports fin_count 0 — it must NOT invent four fins. The
    # old fallback did exactly that, so a finless airframe was meshed and solved
    # with four fins it does not have; and on a body-less design (a pure cone,
    # e.g. the Taylor-Maccoll case) the fabricated root chord came out as
    # body_L*0.25 = 0 and the mesher died on "Degenerate box". The benchmark
    # only got through by passing fin_count=0 explicitly.
    fin_data = max(fins, key=lambda f: f["height"]) if fins else {
        "count": 0, "height": 0.0,
        "root_chord": 0.0, "tip_chord": 0.0,
        "sweep_deg": 0.0, "thick": 0.0, "z_base_k2": 0.0,
        "cross_section": "Square",
    }
    if not fins:
        logger.info("No fin set in the assembly — meshing a finless body.")

    # Nose length falls back to 30% of total if not parsed
    if nose_L <= 0:
        nose_L = total_L * 0.30
    actual_body_L = total_L - nose_L

    # Full outer mold line, so the mesher can build transitions/boattails and a
    # shape-correct nose instead of inferring cone + cylinder from the scalars
    # below (which silently straightened every transition into body tube).
    try:
        profile = cfd_profile(assembly)
    except Exception as e:
        profile = []
        logger.warning(f"Axial profile extraction failed ({e}) — the mesher will "
                       f"fall back to the cone + cylinder approximation.")

    nose_shape = "Ogive"
    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                nose_shape = getattr(comp, "shape", "Ogive")
                break

    if len(fins) > 1:
        logger.warning(
            f"{len(fins)} fin sets found; the CFD mesh carries only the largest "
            f"(h={fin_data['height']:.3f} m). Canards/secondary sets are omitted."
        )

    logger.info(
        f"CFD geometry from assembly: L={total_L:.3f} m  "
        f"body_r={body_r:.4f} m  nose_L={nose_L:.3f} m ({nose_shape})  "
        f"profile: {len(profile)} stations  "
        f"fins: {fin_data['count']}× h={fin_data['height']:.3f} m  "
        f"Cr={fin_data['root_chord']:.3f} m  Ct={fin_data['tip_chord']:.3f} m  "
        f"sweep={fin_data['sweep_deg']:.1f}°"
    )

    return {
        "length":       total_L,
        # [(x_from_nose, radius), …] ascending in x. Empty ⇒ mesher falls back.
        "profile":      [[float(x), float(r)] for x, r in profile],
        "nose_shape":   nose_shape,
        # Body (max) diameter drives the CFD reference area. Provide it
        # explicitly so SU2 normalises forces by the true body frontal area
        # instead of guessing from the STL bounding box — which wrongly picks up
        # the fin span on a finned rocket and inflates the reference area ~10×.
        "max_diameter": 2.0 * body_r,
        "body_radius":  body_r,
        "nose_radius":  body_r,
        "nose_length":  nose_L,
        "body_length":  actual_body_L,
        "fin_count":    fin_data["count"],
        "fin_height":   fin_data["height"],
        "fin_root":     fin_data["root_chord"],
        "fin_tip":      fin_data["tip_chord"],
        "fin_sweep_deg": fin_data["sweep_deg"],
        "fin_thick":    fin_data["thick"],
        "fin_cross_section": fin_data.get("cross_section", "Square"),
        "fin_z_base_k2": fin_data["z_base_k2"],  # K2 z of fin root bottom
    }


def export_assembly_to_stl(assembly, output_path: Path) -> Path:
    """
    Build a watertight 3D surface of a K2 RocketAssembly and export to STL.
    Returns the path to the exported STL file.

    The axisymmetric stack (nose + body tubes + transitions) is built as ONE
    capped surface of revolution — closed at the nose apex and the aft base — so
    it is watertight by construction. Previously each component was a separate
    primitive (an *open* nose shell merged with a capped body cylinder), which
    left the nose base ring as ~64 open boundary edges that ``fill_holes`` could
    not close because the body cap sat inside it. Fins are added as closed solids
    and boolean-unioned into the body.
    """
    total_len = assembly.total_length()
    if total_len <= 0:
        raise ValueError("Assembly has zero total length — cannot export geometry.")

    zs, rs = _assembly_profile(assembly)
    if len(zs) < 2:
        raise ValueError("No renderable axisymmetric geometry found in assembly.")
    combined = _revolve_watertight(zs, rs)

    # Fins: closed solids, unioned into the body so the result stays watertight.
    fin_meshes: list = []
    z_cursor = total_len
    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, BodyTube):
                z_base = z_cursor - comp.length
                for child in comp.children:
                    if isinstance(child, TrapezoidalFinSet):
                        _fin_set_to_meshes(child, z_base, comp.outer_diameter_val / 2,
                                           fin_meshes)
            if isinstance(comp, TrapezoidalFinSet):
                _fin_set_to_meshes(comp, z_cursor, _profile_radius_at(zs, rs, z_cursor),
                                   fin_meshes)
            z_cursor -= _component_axial_length(comp)

    # Fins are appended as individual closed solids. A boolean union with the
    # body is unreliable here — the fin root lies exactly on the body surface
    # (no volumetric overlap), which makes VTK's boolean collapse the mesh — so
    # we merge instead. Each solid is closed (no open boundary), so the surface
    # stays hole-free; the only artefact is coincident faces at the fin root,
    # which the analytic gmsh path (geometry_dict) fuses properly anyway.
    for fin in fin_meshes:
        combined = combined.merge(fin.triangulate().clean())
    combined = combined.clean(tolerance=1e-6).triangulate()

    boundary = combined.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False,
    )
    nonmanifold = combined.extract_feature_edges(
        boundary_edges=False, non_manifold_edges=True,
        feature_edges=False, manifold_edges=False,
    )
    if boundary.n_cells > 0:
        logger.warning(
            f"STL has {boundary.n_cells} open boundary edges (holes) — not "
            f"watertight.")
    elif nonmanifold.n_cells > 0:
        logger.info(
            f"STL is hole-free; {nonmanifold.n_cells} non-manifold edges at "
            f"fin/body joints (coincident faces) — fused analytically for CFD.")
    else:
        logger.info("STL is watertight.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(output_path), binary=False)
    logger.info(f"Assembly exported to STL: {output_path}  ({output_path.stat().st_size:,} bytes)")
    return output_path


def _component_axial_length(comp) -> float:
    """Axial length a component consumes along the body axis."""
    if isinstance(comp, NoseCone):
        return comp.length + getattr(comp, "shoulder_length", 0.0)
    if isinstance(comp, (BodyTube, Transition)):
        return comp.length
    return 0.0


def _nose_profile(shape: str, length: float, radius: float, n: int = 50):
    """Meridian (zs, rs) of a nose cone of the given shape.

    Thin wrapper over the shared builder in ``visualization.viewer_3d`` so the
    exported STL, the CFD solid and every on-screen view use one curve. Kept as
    a module-level name because the headless fallback below has to be able to
    substitute it when the viewer is unimportable.
    """
    return _shared_nose_profile(shape, length, radius, n)


def _assembly_profile(assembly):
    """Walk the stack nose→tail and return (zs, rs) of the outer mold line.

    z runs from the nose tip (z = total_length) down to the base (z = 0). The
    nose contributes a tip point at r=0 so the revolution closes to an apex.
    """
    z = assembly.total_length()
    zs: list = []
    rs: list = []
    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                r = comp.diameter / 2
                # pz: 0(base,r=R)→L(tip,r≈0)
                pz, pr = _nose_profile(getattr(comp, "shape", "Ogive"),
                                       comp.length, r)
                for zz, rr in zip(pz[::-1], pr[::-1]):    # tip first so z descends
                    zs.append(z - (comp.length - zz))
                    rs.append(float(rr))
                z -= comp.length
                L_sh = getattr(comp, "shoulder_length", 0.0)
                if L_sh > 0:                                # internal shoulder
                    z -= L_sh
            elif isinstance(comp, BodyTube):
                r = comp.outer_diameter_val / 2
                zs.append(z);            rs.append(r)
                zs.append(z - comp.length); rs.append(r)
                z -= comp.length
            elif isinstance(comp, Transition):
                zs.append(z);            rs.append(comp.fore_diameter / 2)
                zs.append(z - comp.length); rs.append(comp.aft_diameter / 2)
                z -= comp.length
    return zs, rs


def _simplify_profile(prof: list, tol: float) -> list:
    """Drop profile stations that lie within ``tol`` of the chord they sit on.

    Douglas–Peucker. A straight cone described by 50 sampled stations collapses
    back to its two endpoints, so the CFD solid is one exact frustum rather than
    a 50-facet polyline — which both keeps the Taylor–Maccoll cone geometry
    exact and stops the mesher from resolving 48 imaginary creases.
    """
    if len(prof) < 3:
        return list(prof)

    keep = [False] * len(prof)
    keep[0] = keep[-1] = True
    stack = [(0, len(prof) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, r0 = prof[i0]
        x1, r1 = prof[i1]
        dx, dr = x1 - x0, r1 - r0
        seg = math.hypot(dx, dr)
        worst, worst_i = -1.0, -1
        for i in range(i0 + 1, i1):
            x, r = prof[i]
            if seg < 1e-15:
                d = math.hypot(x - x0, r - r0)
            else:
                d = abs(dr * (x - x0) - dx * (r - r0)) / seg
            if d > worst:
                worst, worst_i = d, i
        if worst > tol and worst_i > 0:
            keep[worst_i] = True
            stack.append((i0, worst_i))
            stack.append((worst_i, i1))
    return [p for p, k in zip(prof, keep) if k]


def cfd_profile(assembly, n_max: int = 60) -> list:
    """Outer mold line as ``[(x_from_nose, radius), …]`` in the CFD frame.

    x runs 0 (nose tip) → total_length (base), the frame ``cfd/meshing.py``
    meshes in. This is what lets the mesher build transitions, boattails and
    shaped noses instead of the cone + cylinder it previously assumed — a
    boattailed rocket was meshed as a straight tube, which turns its boattail
    into a flat base and inflates base drag.

    Radii are floored just above zero: an exactly-zero station makes the OCC
    revolve degenerate, the same reason the old cone was built with a 1 mm tip.
    Coincident stations (a diameter step between two tubes) are separated by a
    hair so the step becomes a very short frustum rather than a zero-length edge.
    """
    zs, rs = _assembly_profile(assembly)
    if len(zs) < 2:
        return []

    total_L = float(assembly.total_length())
    if total_L <= 0:
        return []

    prof = sorted(((total_L - z, max(float(r), 0.0)) for z, r in zip(zs, rs)),
                  key=lambda p: p[0])

    r_min = max(total_L * 1e-4, 1e-6)
    eps_x = max(total_L * 1e-6, 1e-9)

    out: list = []
    for x, r in prof:
        x = min(max(x, 0.0), total_L)
        r = max(r, r_min)
        if out and x - out[-1][0] < eps_x:
            if abs(r - out[-1][1]) <= r_min:
                continue                      # true duplicate station
            x = out[-1][0] + eps_x            # diameter step → tiny frustum
        out.append((x, r))

    if len(out) < 2:
        return []

    out = _simplify_profile(out, tol=r_min)
    if len(out) > n_max:
        # Tighten until it fits: cheaper than meshing 200 near-collinear facets.
        tol = r_min
        while len(out) > n_max and tol < total_L:
            tol *= 2.0
            out = _simplify_profile(out, tol=tol)
    return out


def _profile_radius_at(zs, rs, z_query: float) -> float:
    """Nearest profile radius at an axial station (for top-level fin roots)."""
    if not zs:
        return 0.03
    return rs[min(range(len(zs)), key=lambda i: abs(zs[i] - z_query))]


def _revolve_watertight(zs, rs, n_theta: int = 96) -> "pv.PolyData":
    """Closed surface of revolution about Z for profile (zs, rs).

    A profile point with r≈0 collapses to a single axis vertex (apex), so a
    nose tip closes naturally; the first/last rings with r>0 are capped with a
    centre-fan so open ends (e.g. the aft base) are sealed.
    """
    pts: list = []
    rings: list = []          # (kind, base_index) per profile station
    for z, r in zip(zs, rs):
        if r <= 1e-9:
            pts.append([0.0, 0.0, z])
            rings.append(("point", len(pts) - 1))
        else:
            start = len(pts)
            for j in range(n_theta):
                t = 2.0 * math.pi * j / n_theta
                pts.append([r * math.cos(t), r * math.sin(t), z])
            rings.append(("ring", start))

    faces: list = []

    def ridx(i, j):
        return rings[i][1] + (j % n_theta)

    for i in range(len(zs) - 1):
        ka, kb = rings[i][0], rings[i + 1][0]
        if ka == "ring" and kb == "ring":
            for j in range(n_theta):
                faces += [4, ridx(i, j), ridx(i, j + 1), ridx(i + 1, j + 1), ridx(i + 1, j)]
        elif ka == "point" and kb == "ring":
            ap = rings[i][1]
            for j in range(n_theta):
                faces += [3, ap, ridx(i + 1, j), ridx(i + 1, j + 1)]
        elif ka == "ring" and kb == "point":
            bp = rings[i + 1][1]
            for j in range(n_theta):
                faces += [3, bp, ridx(i, j + 1), ridx(i, j)]
        # point→point: degenerate axis segment, no surface

    def cap(i, flip):
        if rings[i][0] != "ring":
            return
        c = len(pts)
        pts.append([0.0, 0.0, zs[i]])
        for j in range(n_theta):
            if flip:
                faces.extend([3, c, ridx(i, j + 1), ridx(i, j)])
            else:
                faces.extend([3, c, ridx(i, j), ridx(i, j + 1)])

    cap(0, True)                 # forward end (truncated nose, if any)
    cap(len(zs) - 1, False)      # aft base

    mesh = pv.PolyData(np.asarray(pts, dtype=float), np.asarray(faces))
    return mesh.clean(tolerance=1e-9).triangulate()


def _component_to_mesh(comp, z_top, parent_r, meshes):
    """Render one component into a PyVista mesh and append to meshes list."""
    if isinstance(comp, NoseCone):
        r = comp.diameter / 2
        L_nose = comp.length
        L_sh = getattr(comp, "shoulder_length", 0.0)
        z_base = z_top - (L_nose + L_sh)
        z_og = z_base + L_sh

        if L_sh > 0:
            r_sh = getattr(comp, "shoulder_diameter", comp.diameter) / 2 or r * 0.95
            meshes.append(_make_tube(z_base, L_sh, r_sh))

        pz, pr = _nose_profile(getattr(comp, "shape", "Ogive"), L_nose, r)
        pz = pz + z_og
        meshes.append(_make_surface_of_revolution(pz, pr))
        return z_base, r

    elif isinstance(comp, BodyTube):
        r = comp.outer_diameter_val / 2
        L = comp.length
        z_base = z_top - L
        meshes.append(_make_tube(z_base, L, r))

        # Fins attached to this tube
        for child in comp.children:
            if isinstance(child, TrapezoidalFinSet):
                _fin_set_to_meshes(child, z_base, r, meshes)

        return z_base, r

    elif isinstance(comp, Transition):
        L = comp.length
        z_base = z_top - L
        r_top = comp.fore_diameter / 2
        r_bot = comp.aft_diameter / 2
        meshes.append(_make_frustum(z_base, L, r_bot, r_top))
        return z_base, r_bot

    elif isinstance(comp, TrapezoidalFinSet):
        # Top-level fin sets (not inside a body tube)
        _fin_set_to_meshes(comp, z_top, parent_r, meshes)
        return z_top, parent_r

    return z_top, parent_r


def _fin_set_to_meshes(finset, z_base, body_r, meshes):
    """Convert a trapezoidal fin set into PyVista meshes."""
    n = finset.fin_count
    h = finset.height
    Cr = finset.root_chord
    Ct = finset.tip_chord
    sweep_deg = finset.sweep_angle
    sweep_offset = h * math.tan(math.radians(sweep_deg)) if sweep_deg > 0 else 0
    thick = max(0.002, getattr(finset, "thickness", 0.003))

    for i in range(n):
        angle = 2 * math.pi * i / n
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        # Local fin corners (body_r offset in radial direction)
        pts_local = np.array([
            [body_r, 0, z_base],
            [body_r, 0, z_base + Cr],
            [body_r + h, 0, z_base + Cr - sweep_offset],
            [body_r + h, 0, z_base + Cr - sweep_offset - Ct],
        ])

        pts_fwd = pts_local.copy()
        pts_aft = pts_local.copy()
        pts_fwd[:, 1] -= thick / 2
        pts_aft[:, 1] += thick / 2

        all_pts = np.vstack([pts_fwd, pts_aft])  # 8 points

        # Rotate around Z axis
        rot = np.array([[cos_a, -sin_a, 0],
                         [sin_a,  cos_a, 0],
                         [0,       0,    1]])
        all_pts = (rot @ all_pts.T).T

        faces = np.array([
            4, 0, 1, 2, 3,
            4, 7, 6, 5, 4,
            4, 0, 4, 5, 1,
            4, 1, 5, 6, 2,
            4, 2, 6, 7, 3,
            4, 3, 7, 4, 0,
        ])
        meshes.append(pv.PolyData(all_pts, faces=faces).triangulate())


def load_external_cad(filepath: Path) -> pv.PolyData:
    """
    Load an external CAD file (.stl, .obj, .ply, .step, .iges) via PyVista.
    Returns a cleaned PolyData mesh.
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in {".stl", ".obj", ".ply", ".vtk", ".vtu"}:
        mesh = pv.read(str(filepath))
    elif suffix in {".step", ".stp", ".iges", ".igs", ".brep"}:
        # Try CadQuery / OCC bridge if available
        try:
            import cadquery as cq
            result = cq.importers.importStep(str(filepath))
            # Export to tmp STL then re-import
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
                tmp_path = tmp.name
            cq.exporters.export(result, tmp_path)
            mesh = pv.read(tmp_path)
            os.unlink(tmp_path)
        except ImportError:
            raise ImportError(
                "STEP/IGES import requires the 'cadquery' package.\n"
                "Install it with: pip install cadquery"
            )
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    mesh = mesh.clean().triangulate()
    logger.info(f"Loaded external CAD: {filepath} ({mesh.n_points} points)")
    return mesh
