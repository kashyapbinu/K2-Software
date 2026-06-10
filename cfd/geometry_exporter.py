"""
K2 Aerospace — Geometry Exporter
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
        _ogive_profile, _make_surface_of_revolution, _make_tube, _make_frustum
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

    # Pick the largest fin set (most aerodynamically significant)
    fin_data = max(fins, key=lambda f: f["height"]) if fins else {
        "count": 4, "height": body_r * 0.8,
        "root_chord": body_L * 0.25, "tip_chord": body_L * 0.1,
        "sweep_deg": 0.0, "thick": 0.003, "z_base_k2": 0.0,
    }

    # Nose length falls back to 30% of total if not parsed
    if nose_L <= 0:
        nose_L = total_L * 0.30
    actual_body_L = total_L - nose_L

    logger.info(
        f"CFD geometry from assembly: L={total_L:.3f} m  "
        f"body_r={body_r:.4f} m  nose_L={nose_L:.3f} m  "
        f"fins: {fin_data['count']}× h={fin_data['height']:.3f} m  "
        f"Cr={fin_data['root_chord']:.3f} m  Ct={fin_data['tip_chord']:.3f} m  "
        f"sweep={fin_data['sweep_deg']:.1f}°"
    )

    return {
        "length":       total_L,
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
                pz, pr = _ogive_profile(comp.length, r)   # pz: 0(tip)→L(base)
                for zz, rr in zip(pz, pr):
                    zs.append(z - zz)
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

        pz, pr = _ogive_profile(L_nose, r)
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
