"""
K2 AeroSim — CFD Meshing (Gmsh)
====================================
Generates a 3D volumetric SU2 mesh for RANS simulation.

Approach:
  1. Reconstruct rocket OCC geometry (cone nose + cylinder body + fins)
  2. Boolean subtract from wind-tunnel domain → watertight fluid volume
  3. Classify surfaces (rocket_wall vs farfield)
  4. Apply curvature-based + tiered distance refinement fields
  5. Generate 2D surface mesh, then 3D tetrahedra
  6. Run quality checks and export SU2

Boundary Layer Strategy:
  Tet-only with aggressive near-wall refinement, from the tiered distance
  fields alone. Prism layers remain out of reach on this domain, and the reason
  is more specific than "extrudeBoundaryLayer crashes" — that claim was
  re-tested on Gmsh 4.15.2 and is no longer what happens:

  * The extrusion itself RUNS. On the OCC boolean-cut domain, at coarse, medium
    and fine, on the finned cone-tipped rocket, it produces attached prisms
    (151k at coarse) with the physical groups intact, and generate(3) then
    completes without error. Element counts, quality metrics and the exported
    .su2 all look correct.
  * The mesh is nevertheless invalid, and silently so. The original OCC fluid
    volume still spans the boundary-layer region, so the tet pass fills it too:
    the prisms and tets OVERLAP. Measured on the coarse mesh, all 6066 wall
    triangles had a 3D element on both sides — the wall is not a boundary at
    all. SU2 accepts the file, reports plausible mesh-quality metrics, and then
    cannot converge: the residual never drops below its iteration-0 value and
    the run ends in NaN. That was reproduced across five numerics variants
    (gradient scheme, CFL, linear solver, limiter), which is how it was traced
    to the mesh rather than the settings.
  * Fixing it properly means rebuilding the volume after the extrusion, so the
    tets are bounded by the stack's outer face. That is where Gmsh stops:
    "Pyramid top vertex already classified ... non-manifold quad boundaries not
    supported yet". The prism stack's quad faces cannot bound a tet region.

  So the conclusion the module reached before still holds, but the useful
  version of it is: extrusion is not the blocker, the volume rebuild is, and an
  extruded mesh that is never volume-rebuilt is WORSE than no prisms because
  nothing downstream reports it as broken. _check_mesh_quality now audits wall
  manifoldness for exactly this, so a future attempt fails loudly.

  Consequences, unchanged: the first cell sits near y+ 3500, far outside the
  30 < y+ < 300 band a wall function can invert, so neither a low-Re model nor
  a wall model has a valid first cell. Skin friction is under-resolved either
  way — a known accuracy limit, and the reason the sweep offers a hybrid Euler
  + flat-plate-friction mode instead of trusting RANS friction here.

  There are NO wall functions to fall back on, and this was tested rather than
  assumed. SU2's STANDARD_WALL_FUNCTION diverges from a freestream cold start
  (T_Wall < 0 -> NaN); from a converged restart it runs, but its wall-coefficient
  solve fails on 88% of wall points, pinning y+ at exactly 30 and leaving those
  points with no wall shear at all. Tuning the wall-model solver made it worse.
  Full measured numbers are in cfd/solvers/su2_solver.py — read them before
  re-enabling it.

Coordinate convention (CFD frame): +X = freestream flow direction,
nose tip at x=0, nozzle at x=total_L.

Requires: pip install gmsh
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Optional

logger = logging.getLogger("K2.CFD.Meshing")

# Refinement levels: mesh size near rocket = fraction of body radius
_REFINEMENT_FACTORS = {
    "coarse":     {"wall_frac": 0.50,  "far_frac": 6.0},
    "medium":     {"wall_frac": 0.28,  "far_frac": 10.0},
    "fine":       {"wall_frac": 0.18,  "far_frac": 15.0},
    "very_fine":  {"wall_frac": 0.10,  "far_frac": 20.0},
    "ultra_fine": {"wall_frac": 0.05,  "far_frac": 30.0},
}


def _estimate_sizes_from_count(
    target_count: int,
    body_r: float,
    total_L: float,
    tun_radius: float,
    tun_len: float,
) -> tuple[float, float]:
    """
    Estimate (wall_size, far_size) from a target element count.

    Uses the heuristic: average tet volume ≈ lc³/6,
    so N ≈ V_domain / (lc³/6)  →  lc ≈ (6·V/N)^(1/3).
    Wall size is scaled down and far-field size scaled up from lc_avg.
    """
    domain_volume = tun_len * (2 * tun_radius) ** 2  # box approximation
    lc_avg = (6.0 * domain_volume / max(target_count, 1000)) ** (1.0 / 3.0)
    lc_wall = lc_avg * 0.15
    lc_far = lc_avg * 3.0
    # Clamp to reasonable bounds
    lc_wall = max(lc_wall, body_r * 0.005)   # floor: 0.5% of body radius
    lc_wall = min(lc_wall, body_r * 0.5)     # ceiling: 50% of body radius
    lc_far = max(lc_far, body_r * 2.0)
    lc_far = min(lc_far, total_L * 10.0)
    logger.info(
        f"Size from target count {target_count:,}: "
        f"lc_avg={lc_avg:.5f}  wall={lc_wall:.5f}  far={lc_far:.4f}"
    )
    return lc_wall, lc_far


# ── Main Entry Point ──────────────────────────────────────────────────────────

def build_wind_tunnel_mesh(
    stl_path: Path,
    output_path: Path,
    refinement: str = "medium",
    domain_length_scale: float = 10.0,
    domain_radius_scale: float = 20.0,
    bl_layers: int = 15,
    bl_growth: float = 1.2,
    geometry_dict: dict = None,
    custom_wall_size: float | None = None,
    target_element_count: int | None = None,
    external_cad: Path | None = None,
    flow_axis: str = "auto",
    cad_info: dict | None = None,
    cad_units: str = "auto",
    cad_wrap: bool = False,
    cad_wrap_resolution: str = "medium",
) -> Path:
    """
    Generate a volumetric SU2 mesh with prism boundary layers using Gmsh.
    Returns the path to the .su2 mesh file.

    When ``external_cad`` is given the parametric rocket is bypassed entirely
    and the imported body is meshed instead (see
    :func:`build_external_cad_mesh`); ``stl_path`` and ``geometry_dict`` are
    then ignored.
    """
    if external_cad is not None:
        return build_external_cad_mesh(
            cad_path=external_cad,
            output_path=output_path,
            refinement=refinement,
            domain_length_scale=domain_length_scale,
            domain_radius_scale=domain_radius_scale,
            custom_wall_size=custom_wall_size,
            target_element_count=target_element_count,
            flow_axis=flow_axis,
            cad_info=cad_info,
            cad_units=cad_units,
            cad_wrap=cad_wrap,
            cad_wrap_resolution=cad_wrap_resolution,
        )

    try:
        import gmsh
    except ImportError:
        raise ImportError("Install Gmsh: pip install gmsh")

    stl_path    = Path(stl_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not stl_path.is_file():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    ref_f = _REFINEMENT_FACTORS.get(refinement, _REFINEMENT_FACTORS["medium"])

    # Parse rocket dimensions
    if geometry_dict is not None:
        rocket = geometry_dict
        logger.info(
            f"Using exact assembly geometry: L={rocket['length']:.3f} m  "
            f"body_r={rocket['body_radius']:.4f} m  "
            f"fins {rocket['fin_count']}× Cr={rocket['fin_root']:.3f} m"
        )
    else:
        rocket = _parse_rocket_geometry(stl_path)
        logger.info(
            f"Rocket geometry (STL estimate): L={rocket['length']:.3f} m  "
            f"r_body={rocket['body_radius']:.4f} m"
        )

    body_r   = rocket["body_radius"]
    # Medium preset as a safety ceiling (custom must be at least as fine)
    lc_medium = body_r * _REFINEMENT_FACTORS["medium"]["wall_frac"]

    # ── Compute mesh sizes: target count > custom wall size > preset ──────
    # When the UI is in custom mode it sends both target_element_count and
    # custom_wall_size (they're bidirectionally synced).  Prefer the count-
    # based estimator because it produces a coherent (wall, far) pair.
    # Only use custom_wall_size alone when target_element_count is absent.
    if target_element_count is not None and target_element_count > 0:
        tun_len_est = rocket["length"] * domain_length_scale
        tun_r_est = body_r * max(domain_radius_scale, 20.0)
        lc_rocket, lc_far = _estimate_sizes_from_count(
            target_element_count, body_r, rocket["length"], tun_r_est, tun_len_est
        )
        # Safety: never coarser than medium preset
        if lc_rocket > lc_medium:
            logger.info(f"Target count estimate wall={lc_rocket:.5f} m too coarse; "
                        f"clamping to medium preset ({lc_medium:.5f} m)")
            lc_rocket = lc_medium
        logger.info(f"Target element count override ({target_element_count:,}): "
                     f"lc_rocket={lc_rocket:.5f} m  lc_far={lc_far:.4f} m")
    elif custom_wall_size is not None and custom_wall_size > 0:
        lc_rocket = custom_wall_size
        # Safety: never coarser than medium preset
        if lc_rocket > lc_medium:
            logger.info(f"Custom wall size {lc_rocket:.5f} m too coarse; "
                        f"clamping to medium preset ({lc_medium:.5f} m)")
            lc_rocket = lc_medium
        # Compute far-field size proportionally — keep the preset ratio
        wall_frac = ref_f["wall_frac"]
        far_frac = ref_f["far_frac"]
        lc_far = lc_rocket * (far_frac / wall_frac) if wall_frac > 0 else lc_rocket * 35.0
        lc_far = max(lc_far, body_r * 6.0)  # minimum far-field
        logger.info(f"Custom wall size override: lc_rocket={lc_rocket:.5f} m  lc_far={lc_far:.4f} m")
    else:
        lc_rocket = body_r * ref_f["wall_frac"]
        lc_far    = body_r * ref_f["far_frac"]

    tun_len    = rocket["length"] * domain_length_scale
    tun_radius = body_r * max(domain_radius_scale, 20.0)

    # Safety: clear any stale Gmsh session. Guard with isInitialized() — calling
    # finalize() on a fresh session makes Gmsh's C++ logger print
    # "Error : Gmsh has not been initialized" before raising (the exception is
    # swallowed, but the stderr line still leaks).
    try:
        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.model.add("K2_CFD")

    try:
        _build_mesh(
            gmsh, rocket, tun_len, tun_radius,
            lc_far, lc_rocket, output_path,
            bl_layers, bl_growth,
        )
    finally:
        gmsh.finalize()

    su2_path = output_path.with_suffix(".su2")
    size_mb = su2_path.stat().st_size / 1e6
    logger.info(f"SU2 mesh ready: {su2_path}  ({size_mb:.2f} MB)")
    return su2_path


# ── Core meshing logic ────────────────────────────────────────────────────────

def _build_mesh(
    gmsh, rocket, tun_len, tun_radius, lc_far, lc_rocket, output_path,
    bl_layers, bl_growth,
):
    """
    Build wind-tunnel fluid volume and mesh it with prism boundary layers.

    Coordinate convention (CFD frame):
        +X  = freestream flow direction
        Nose tip  at x = 0          (faces the incoming flow)
        Nozzle    at x = total_L    (in the wake)
        Body axis = X axis
    """
    occ = gmsh.model.occ

    body_r  = rocket["body_radius"]
    body_L  = rocket["body_length"]
    nose_r  = rocket["nose_radius"]
    nose_L  = rocket["nose_length"]
    total_L = rocket["length"]

    # ── 1. Build rocket solid ─────────────────────────────────────────────────
    rocket_parts = []

    # Preferred: revolve the real outer mold line. It reproduces the shape the
    # assembly actually declares — shaped nose, diameter steps, transitions,
    # boattails — none of which the cone+cylinder fallback below can express.
    profile = rocket.get("profile") or []
    profile = [(float(p[0]), float(p[1])) for p in profile if len(p) >= 2]
    body_built = False
    if len(profile) >= 2:
        try:
            rocket_parts.extend(_revolve_profile_solid(occ, profile))
            body_built = True
        except Exception as e:
            logger.warning(
                f"Profile revolve failed ({e}) — falling back to the "
                f"cone + cylinder approximation (transitions will be lost)."
            )

    if not body_built:
        # Fallback: straight cone + constant-radius tube. Used when no profile
        # is available (STL-only runs with no geometry_dict) or if the revolve
        # could not be built. Any transition/boattail is straightened here.
        if profile:
            logger.warning("Approximating the body as cone + cylinder; any "
                           "transition or boattail is NOT represented.")
        # Nose cone: tip at x=0 (upstream), base at x=nose_L
        nose_tag = occ.addCone(
            0.0,    0, 0,
            nose_L, 0, 0,
            0.001,          # near-zero tip radius (avoids degenerate vertex)
            nose_r,
        )
        rocket_parts.append((3, nose_tag))

        # Body tube: from x=nose_L to x=total_L. Skip a degenerate (≈0-length)
        # body so a nose-only / pure-cone geometry meshes instead of crashing in
        # addCylinder ("Cannot build cylinder of zero height").
        if body_L > 1e-6:
            body_tag = occ.addCylinder(nose_L, 0, 0, body_L, 0, 0, body_r)
            rocket_parts.append((3, body_tag))
        else:
            logger.info("Body length ≈ 0 — building nose-only (cone) geometry.")

    # Fins at the aft end
    fin_parts = _add_fins(occ, rocket, total_L)
    rocket_parts.extend(fin_parts)

    # Fuse all rocket parts into one solid
    if len(rocket_parts) > 1:
        fused, _ = occ.fuse(
            [rocket_parts[0]], rocket_parts[1:],
            removeObject=True, removeTool=True
        )
        rocket_solid = fused
    else:
        rocket_solid = rocket_parts
    occ.synchronize()
    logger.info(f"Rocket solid created: {len(rocket_solid)} volume(s)  "
                f"[nose@x=0, nozzle@x={total_L:.3f}]")

    # ── 2. Wind tunnel domain ─────────────────────────────────────────────────
    upstream_x     = -5.0 * total_L
    downstream_x   = total_L + 15.0 * total_L
    domain_len     = downstream_x - upstream_x

    tunnel_tag = occ.addBox(
        upstream_x, -tun_radius, -tun_radius,
        domain_len,  tun_radius * 2, tun_radius * 2,
    )
    occ.synchronize()
    logger.info(
        f"Wind tunnel: upstream={upstream_x:.2f} m, downstream={downstream_x:.2f} m, "
        f"radial=±{tun_radius:.2f} m  (domain {domain_len:.1f} m long)"
    )

    # ── 3. Boolean cut: fluid = tunnel − rocket ───────────────────────────────
    fluid, _ = occ.cut(
        [(3, tunnel_tag)],
        rocket_solid,
        removeObject=True,
        removeTool=True,
    )
    occ.synchronize()
    logger.info(f"Boolean cut complete: {len(fluid)} fluid volume(s)")

    if not fluid:
        raise RuntimeError(
            "Boolean subtraction failed — no fluid volume created. "
            "This usually means the rocket solid extends outside the wind tunnel."
        )

    # ── 4. Identify boundary surfaces (CRITICAL for BL extrusion) ─────────────
    fluid_vol_tags = [v[1] for v in fluid]

    # Take the fluid volume's OWN boundary, not every surface in the model.
    #
    # Revolving a meridian through 2*pi leaves seam faces behind in the OCC
    # model. They are interior to the (subtracted) body, so they bound no fluid
    # — but they are still 2D entities, and classifying off getEntities(2) swept
    # them into rocket_wall. SU2 then aborts with "The surface element (0, 0)
    # doesn't have an associated volume element". Everything returned by
    # getBoundary is by construction adjacent to a fluid cell.
    _bnd = gmsh.model.getBoundary(
        [(3, t) for t in fluid_vol_tags],
        combined=True, oriented=False, recursive=False,
    )
    all_surfs = [(2, abs(tag)) for dim, tag in _bnd if dim == 2]
    if not all_surfs:
        logger.warning("Fluid volume reported no boundary surfaces — falling "
                       "back to every 2D entity in the model.")
        all_surfs = gmsh.model.getEntities(2)
    else:
        _n_all = len(gmsh.model.getEntities(2))
        if _n_all > len(all_surfs):
            logger.info(
                f"Ignoring {_n_all - len(all_surfs)} surface(s) not on the fluid "
                f"boundary (revolve seams / internal faces)."
            )

    rocket_wall_surfs = []
    farfield_surfs    = []

    far_dist_thresh  = tun_radius * 0.80
    inlet_x_thresh   = upstream_x   * 0.90
    outlet_x_thresh  = downstream_x * 0.90

    for _, stag in all_surfs:
        cx, cy, cz = occ.getCenterOfMass(2, stag)
        radial_dist = math.sqrt(cy**2 + cz**2)

        is_farfield = (
            radial_dist > far_dist_thresh or
            cx < inlet_x_thresh          or
            cx > outlet_x_thresh
        )
        if is_farfield:
            farfield_surfs.append(stag)
        else:
            rocket_wall_surfs.append(stag)

    logger.info(
        f"Surface classification: {len(rocket_wall_surfs)} rocket_wall, "
        f"{len(farfield_surfs)} farfield"
    )

    # ── VALIDATION: ensure no farfield surfaces leaked into rocket_wall ────────
    # A rocket wall surface should have a centroid within the rocket's
    # bounding envelope (radial < body_r + fin_span + margin)
    # Envelope must bound the REVOLVED body, not just body_r: a flared
    # transition can carry a larger radius than the nominal tube, and a wall
    # surface outside the envelope gets reclassified as farfield — which would
    # silently drop the flare from the monitored wall.
    r_body_max = max([body_r] + [p[1] for p in profile]) if profile else body_r
    fin_span = rocket.get("fin_height", body_r) + r_body_max
    max_wall_r = fin_span * 1.5
    validated_wall = []
    for stag in rocket_wall_surfs:
        cx, cy, cz = occ.getCenterOfMass(2, stag)
        r = math.sqrt(cy**2 + cz**2)
        if r < max_wall_r and -0.01 <= cx <= total_L * 1.01:
            validated_wall.append(stag)
        else:
            farfield_surfs.append(stag)
            logger.warning(
                f"Surface {stag} reclassified: centroid ({cx:.3f}, {cy:.3f}, {cz:.3f}) "
                f"r={r:.4f} outside rocket envelope — moved to farfield"
            )
    rocket_wall_surfs = validated_wall

    if not rocket_wall_surfs:
        logger.error("No rocket wall surfaces found — boundary layer cannot be created!")

    if rocket_wall_surfs:
        gmsh.model.addPhysicalGroup(2, rocket_wall_surfs, name="rocket_wall")
    if farfield_surfs:
        gmsh.model.addPhysicalGroup(2, farfield_surfs,    name="farfield")
    if fluid_vol_tags:
        gmsh.model.addPhysicalGroup(3, fluid_vol_tags,    name="fluid")

    # ── 5. Mesh size fields ────────────────────────────────────────────────────

    nose_L  = rocket.get("nose_length", total_L * 0.3)
    fin_Cr  = rocket.get("fin_root", total_L * 0.15)
    fin_h   = rocket.get("fin_height", body_r)
    # Size the fin/wake boxes off the widest station of the REVOLVED body, the
    # same envelope the classification above used — not the nominal body_r,
    # which under-sizes the refinement region on a flared airframe.
    fin_span = fin_h + r_body_max

    # ── 5a. Distance-based near-wall refinement ───────────────────────────────
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_dist, "SurfacesList",
        rocket_wall_surfs if rocket_wall_surfs else [s[1] for s in all_surfs[:5]]
    )

    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_rocket)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  body_r * 0.1)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  body_r * 5.0)

    # ── 5b. Nose tip refinement ───────────────────────────────────────────────
    lc_nose = lc_rocket * 0.6
    f_nose = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_nose, "XMin",  -body_r)
    gmsh.model.mesh.field.setNumber(f_nose, "XMax",   nose_L * 0.3)
    gmsh.model.mesh.field.setNumber(f_nose, "YMin",  -body_r * 2)
    gmsh.model.mesh.field.setNumber(f_nose, "YMax",   body_r * 2)
    gmsh.model.mesh.field.setNumber(f_nose, "ZMin",  -body_r * 2)
    gmsh.model.mesh.field.setNumber(f_nose, "ZMax",   body_r * 2)
    gmsh.model.mesh.field.setNumber(f_nose, "VIn",    lc_nose)
    gmsh.model.mesh.field.setNumber(f_nose, "VOut",   lc_far)

    # ── 5c. Fin-region refinement ─────────────────────────────────────────────
    # Centre the box on the actual fin axial station (fins sit at
    # x_TE = total_L - fin_z_base_k2 in the CFD frame; 0 = tail-mounted).
    fin_x_te = total_L - rocket.get("fin_z_base_k2", 0.0)
    fin_x_te = min(total_L, max(fin_Cr, fin_x_te))
    lc_fin = lc_rocket * 0.7
    f_fin = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_fin, "XMin",  fin_x_te - fin_Cr * 1.2)
    gmsh.model.mesh.field.setNumber(f_fin, "XMax",  fin_x_te + body_r)
    gmsh.model.mesh.field.setNumber(f_fin, "YMin", -fin_span * 1.5)
    gmsh.model.mesh.field.setNumber(f_fin, "YMax",  fin_span * 1.5)
    gmsh.model.mesh.field.setNumber(f_fin, "ZMin", -fin_span * 1.5)
    gmsh.model.mesh.field.setNumber(f_fin, "ZMax",  fin_span * 1.5)
    gmsh.model.mesh.field.setNumber(f_fin, "VIn",   lc_fin)
    gmsh.model.mesh.field.setNumber(f_fin, "VOut",  lc_far)

    # ── 5d. Wake refinement ───────────────────────────────────────────────────
    lc_wake = body_r * 2.0
    f_wake = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_wake, "XMin",  total_L)
    gmsh.model.mesh.field.setNumber(f_wake, "XMax",  total_L + 3.0 * total_L)
    gmsh.model.mesh.field.setNumber(f_wake, "YMin", -body_r * 3.0)
    gmsh.model.mesh.field.setNumber(f_wake, "YMax",  body_r * 3.0)
    gmsh.model.mesh.field.setNumber(f_wake, "ZMin", -body_r * 3.0)
    gmsh.model.mesh.field.setNumber(f_wake, "ZMax",  body_r * 3.0)
    gmsh.model.mesh.field.setNumber(f_wake, "VIn",   lc_wake)
    gmsh.model.mesh.field.setNumber(f_wake, "VOut",  lc_far)

    # Combine: take minimum size from all fields
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(
        f_min, "FieldsList", [f_thr, f_nose, f_fin, f_wake]
    )
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    logger.info(
        f"Mesh fields: wall={lc_rocket:.4f}  nose={lc_nose:.4f}  "
        f"fin={lc_fin:.4f}  wake={lc_wake:.4f}  far={lc_far:.4f}"
    )

    # ── 6. Generate 2D surface mesh ───────────────────────────────────────────
    gmsh.option.setNumber("Mesh.Algorithm",   6)   # Frontal-Delaunay 2D
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # Curvature-based sizing MUST stay on: the near-zero cone tip (r=0.001) and
    # the body cylinder are high-curvature surfaces. With it off, coarse/medium
    # presets size facets only from the distance/box fields — which don't know
    # about the tip — so triangles at the tip span wider than the tip itself and
    # fold over each other, producing "Invalid boundary mesh (overlapping
    # facets) on surface 1" at generate(3). 20 = min elements per 2*pi of arc.
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

    # Size floor MUST be set before generate(2), not just before generate(3).
    #
    # It used to be applied only ahead of the 3D pass, which left the surface
    # mesh entirely unconstrained: curvature sizing at the cone tip (radius
    # ~2e-4 m, 20 elements per 2*pi of arc) drove facets down to 3.1e-5 m
    # against a body median of 8.6e-3 m. Measured on the shipped mesh that is
    # a 275:1 spread in edge length and 3.6e5:1 in cell AREA, with 852 cells
    # crammed into x < 0.02 m holding 0.15% of the wetted area.
    #
    # Those slivers are where the solution goes non-physical — wall pressure
    # 10.6% above the isentropic stagnation limit and wall temperature 9.6%
    # above T0 both peak there — and because they are so numerous they also
    # captured every point-counted statistic downstream.
    #
    # The floor is deliberately mild: at 5% of lc_rocket it sits just under
    # the tip's own median facet, so it removes the degenerate tail without
    # coarsening the tip. Curvature sizing stays ON — turning it off makes
    # coarse/medium meshes fold facets over the tip and fail generate(3).
    _lc_min = lc_rocket * 0.05
    gmsh.option.setNumber("Mesh.MeshSizeMin", _lc_min)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", _lc_min)
    logger.info(f"Surface size floor: {_lc_min:.3e} m (5% of lc_rocket={lc_rocket:.5f})")

    gmsh.model.mesh.generate(2)
    logger.info("2D surface mesh generated")

    # ── 7. Prism boundary layer extrusion — NOT DONE, and not for the reason
    #       this comment used to give ────────────────────────────────────────
    #
    # geo.extrudeBoundaryLayer does run on an OCC boolean-cut domain (Gmsh
    # 4.15.2, measured at coarse/medium/fine on the finned rocket). What it
    # cannot do is leave a valid mesh behind: the original fluid volume still
    # covers the boundary-layer region, so generate(3) fills it with tets on top
    # of the prisms. All 6066 wall triangles came out with an element on BOTH
    # sides — the wall stops being a boundary — and SU2 takes the file, reports
    # healthy mesh metrics, then fails to converge from iteration 0 and NaNs.
    # Rebuilding the volume against the stack's outer face is the correct fix
    # and is where Gmsh actually stops: "non-manifold quad boundaries not
    # supported yet". Full write-up in the module docstring.
    #
    # The danger is that the broken version looks like it works, so
    # _check_mesh_quality audits wall manifoldness on every mesh. If prisms are
    # attempted again, that check is what will catch a silent overlap.
    #
    # The tiered distance-based refinement fields (step 5) give the finest
    # near-wall tets this route can produce, and that is still not close enough:
    # the first cell lands near y+ 3500. There is no wall model behind them
    # either — wall functions are disabled in the SU2 config, having been
    # measured to leave 88% of wall points with no wall shear (see
    # cfd/solvers/su2_solver.py) — so wall shear is under-resolved either way.
    #
    # bl_layers / bl_growth are accepted for call compatibility and are NOT used.
    n_prisms = 0
    logger.info(
        "Tet-only mesh — BL prism extrusion disabled (leaves prisms and tets "
        f"overlapping; see module docstring); bl_layers={bl_layers}/"
        f"bl_growth={bl_growth} ignored. Near-wall refinement is from the "
        "distance fields only, with no wall model."
    )

    # ── 8. Generate 3D volume mesh ────────────────────────────────────────────
    # (size floor already applied before generate(2) — see step 6)
    gmsh.option.setNumber("Mesh.Optimize",    1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Extra optimization for tet quality

    # Delaunay 3D (Algorithm3D=1) is primary: it gives the proven full-fidelity
    # "fine" mesh (~826k tets). HXT (the multithreaded alternative) only "wins"
    # by coarsening the mesh, which hurts transonic shock resolution, so it is
    # NOT used as primary. It stays only as the robustness fallback: if Delaunay
    # intermittently aborts (exit-1) on thin fin-TE / nose-tip slivers, retry
    # with HXT + a coarser size floor so a valid mesh still comes out.
    try:
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)   # Delaunay 3D (full-fidelity fine)
        gmsh.model.mesh.generate(3)
    except Exception as e:
        logger.warning(
            f"3D mesh (Delaunay) failed: {e}. Retrying with HXT + coarser floor."
        )
        gmsh.option.setNumber("Mesh.MeshSizeMin", lc_rocket * 0.15)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_rocket * 0.15)
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT — robust on bad boundaries
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.generate(3)
        logger.info("3D mesh recovered via HXT fallback.")

    # ── 8b. Post-generation mesh optimization ─────────────────────────────────
    # Smooth and untangle distorted elements (especially BL prisms at junctions)
    try:
        gmsh.model.mesh.optimize("", force=True)
        logger.info("Mesh optimization pass 1 (Gmsh default) complete")
    except Exception as e:
        logger.warning(f"Mesh optimization pass 1 failed: {e}")
    try:
        gmsh.model.mesh.optimize("Netgen", force=True)
        logger.info("Mesh optimization pass 2 (Netgen) complete")
    except Exception as e:
        logger.warning(f"Mesh optimization pass 2 (Netgen) failed: {e}")

    # ── 9. Quality checks ─────────────────────────────────────────────────────
    _check_mesh_quality(gmsh, body_r, n_prisms, wall_surfs=rocket_wall_surfs)

    # ── 10. Export ────────────────────────────────────────────────────────────
    # Update physical groups to include any new BL volumes
    all_vols = gmsh.model.getEntities(3)
    if len(all_vols) > len(fluid_vol_tags):
        all_vol_tags = [v[1] for v in all_vols]
        # Remove old physical group and re-add with all volumes
        try:
            gmsh.model.removePhysicalGroups([(3, g) for g in
                gmsh.model.getPhysicalGroups(3)])
        except Exception:
            pass
        gmsh.model.addPhysicalGroup(3, all_vol_tags, name="fluid")

    su2_path = output_path.with_suffix(".su2")
    gmsh.write(str(su2_path))


# ── External CAD meshing ─────────────────────────────────────────────────────

def _geo_box(gmsh, x0, y0, z0, x1, y1, z1) -> tuple[list, int]:
    """
    Build an axis-aligned box in the ``geo`` kernel and return
    ``(surface_tags, surface_loop_tag)``.

    The OCC kernel has ``addBox``, the geo kernel does not — and the discrete
    (STL) import route is stuck in geo because ``createGeometry`` produces geo
    entities. Hence the manual 8 points / 12 lines / 6 faces construction.
    """
    geo = gmsh.model.geo
    p = [
        geo.addPoint(x0, y0, z0), geo.addPoint(x1, y0, z0),
        geo.addPoint(x1, y1, z0), geo.addPoint(x0, y1, z0),
        geo.addPoint(x0, y0, z1), geo.addPoint(x1, y0, z1),
        geo.addPoint(x1, y1, z1), geo.addPoint(x0, y1, z1),
    ]
    bottom = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    top    = [geo.addLine(p[4 + i], p[4 + (i + 1) % 4]) for i in range(4)]
    vert   = [geo.addLine(p[i], p[4 + i]) for i in range(4)]

    # Every face must wind so its normal points OUT of the box, or the surface
    # loop is inconsistently oriented and addVolume can produce an inverted or
    # unmeshable region. `bottom` and `top` are built with the same in-plane
    # winding, which is outward (+Z) only for the top — so the bottom loop is
    # traversed in reverse.
    faces = [
        geo.addPlaneSurface([geo.addCurveLoop([-bottom[3], -bottom[2],
                                               -bottom[1], -bottom[0]])]),
        geo.addPlaneSurface([geo.addCurveLoop(top)]),
    ]
    for i in range(4):
        j = (i + 1) % 4
        # p[i] → p[j] → p[j+4] → p[i+4] → p[i]
        loop = geo.addCurveLoop([bottom[i], vert[j], -top[i], -vert[i]])
        faces.append(geo.addPlaneSurface([loop]))

    surf_loop = geo.addSurfaceLoop(faces)
    geo.synchronize()
    return faces, surf_loop


def build_external_cad_mesh(
    cad_path: Path,
    output_path: Path,
    refinement: str = "medium",
    domain_length_scale: float = 10.0,
    domain_radius_scale: float = 20.0,
    custom_wall_size: float | None = None,
    target_element_count: int | None = None,
    flow_axis: str = "auto",
    cad_info: dict | None = None,
    cad_units: str = "auto",
    cad_wrap: bool = False,
    cad_wrap_resolution: str = "medium",
    _force_discrete: bool = False,
) -> Path:
    """
    Generate a wind-tunnel SU2 mesh around an arbitrary imported CAD body.

    Unlike :func:`build_wind_tunnel_mesh`, nothing about the shape is assumed —
    no cone, no body tube, no fins. The imported solid itself is subtracted
    from the domain, so the mesh follows the real surfaces.

    Two routes (see ``cfd/external_geometry.py``):
      * exact B-Rep (.step/.iges/.brep) → OCC import + boolean cut
      * discrete (.stl/.obj/.ply)       → re-topologised shell used as an inner
        hole of the domain volume (the geo kernel has no booleans)

    Physical group names are deliberately identical to the rocket path
    (``rocket_wall`` / ``farfield`` / ``fluid``) so the SU2 config writer,
    force integration and post-processing need no special case.
    """
    from cfd.external_geometry import (
        CADImportError, analyze_cad, cad_info_from_dict, gmsh_session,
        import_brep_into_occ, import_stl_into_geo,
    )

    cad_path    = Path(cad_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cad_path.is_file():
        raise FileNotFoundError(f"CAD file not found: {cad_path}")

    # Reuse the analysis the UI already ran when it still matches this request;
    # re-measure when it is missing, stale, or was taken about a different axis.
    info = cad_info_from_dict(cad_info)
    stale = (
        info is None
        or Path(info.source) != cad_path
        or not Path(info.preview_stl).is_file()
        or (flow_axis in ("x", "y", "z") and info.flow_axis != flow_axis)
        or (cad_units not in (None, "", "auto") and info.units != cad_units)
        or (bool(cad_wrap) != bool(info.wrapped))
    )
    if stale:
        info = analyze_cad(cad_path, flow_axis=flow_axis,
                           work_dir=output_path.parent, units=cad_units or "auto",
                           wrap=cad_wrap, wrap_resolution=cad_wrap_resolution)

    if info.length <= 0:
        raise CADImportError(f"{cad_path.name} has zero extent along the flow axis.")

    # Caller (or the B-Rep fallback below) can force the discrete route.
    if _force_discrete and info.is_brep:
        info = replace(info, is_brep=False)

    if not info.watertight and not info.is_brep:
        logger.warning(
            f"{cad_path.name} has {info.open_edges} free edges — the discrete "
            "import route needs a closed shell and will likely fail."
        )

    L       = info.length
    cross_r = max(info.cross_radius, L * 0.02)
    # Characteristic near-wall scale — the arbitrary-body analogue of body_r.
    #
    # Derived from the measured frontal area, not the bounding-box diagonal:
    # for a circular cross-section r_eq recovers the true radius exactly, while
    # cross_radius (half the bbox diagonal) overshoots it by √2 and coarsens
    # the whole mesh. That mattered — a 40 mm cone-cylinder meshed off the
    # diagonal gave a wake so under-resolved it reported Cd=0.86 and a spurious
    # Cl=0.08 at zero AoA; off r_eq it lands at Cd=0.41, Cl≈0.
    #
    # Capped at L/4 so a stubby body (sphere, capsule) is not sized by a radius
    # comparable to its own length. Floored at an eighth of the bbox radius so
    # a thin, wide body (wing, plate) — whose frontal area is tiny next to its
    # span — does not demand an unmeshably fine element size.
    r_eq = math.sqrt(info.frontal_area / math.pi) if info.frontal_area > 0 else cross_r
    char = max(min(r_eq, L / 4.0), cross_r / 8.0, L / 500.0)

    ref_f = _REFINEMENT_FACTORS.get(refinement, _REFINEMENT_FACTORS["medium"])
    lc_medium = char * _REFINEMENT_FACTORS["medium"]["wall_frac"]

    # Domain: same proportions as the rocket tunnel (5 L upstream, 15 L wake).
    # domain_length_scale is deliberately not applied here — the rocket path
    # hardcodes the same 5/15 for its box and only uses the scale to pre-estimate
    # element sizes, and here the true extent is passed to that estimate instead.
    # Kept in the signature so the two mesh entry points stay call-compatible.
    up_x   = -5.0 * L
    down_x = L + 15.0 * L
    rad    = max(domain_radius_scale, 10.0) * cross_r
    rad    = max(rad, 6.0 * max(info.cross_width, info.cross_height, char))

    if target_element_count is not None and target_element_count > 0:
        lc_wall, lc_far = _estimate_sizes_from_count(
            target_element_count, char, L, rad, down_x - up_x
        )
        if lc_wall > lc_medium:
            logger.info(f"Target count estimate wall={lc_wall:.5f} m too coarse; "
                        f"clamping to medium preset ({lc_medium:.5f} m)")
            lc_wall = lc_medium
    elif custom_wall_size is not None and custom_wall_size > 0:
        lc_wall = min(custom_wall_size, lc_medium)
        wall_frac, far_frac = ref_f["wall_frac"], ref_f["far_frac"]
        lc_far = lc_wall * (far_frac / wall_frac) if wall_frac > 0 else lc_wall * 35.0
        lc_far = max(lc_far, char * 6.0)
    else:
        lc_wall = char * ref_f["wall_frac"]
        lc_far  = char * ref_f["far_frac"]

    logger.info(
        f"External CAD mesh: L={L:.4f} m  char={char:.5f} m  "
        f"domain x[{up_x:.2f}, {down_x:.2f}] r=±{rad:.2f} m  "
        f"lc_wall={lc_wall:.5f}  lc_far={lc_far:.4f}  "
        f"({'exact B-Rep' if info.is_brep else 'discrete shell'})"
    )

    with gmsh_session("K2_CFD_CAD", verbosity=3) as gmsh:
        reparametrised = True   # B-Rep always has an exact parametrisation
        if info.is_brep:
            occ = gmsh.model.occ
            body_vols = import_brep_into_occ(gmsh, cad_path, info.flow_axis,
                                             unit_scale=info.unit_scale)
            box = occ.addBox(up_x, -rad, -rad, down_x - up_x, 2 * rad, 2 * rad)
            occ.synchronize()

            fluid, _ = occ.cut(
                [(3, box)], body_vols, removeObject=True, removeTool=True
            )
            occ.synchronize()
            if not fluid:
                raise RuntimeError(
                    "Boolean subtraction produced no fluid volume. The CAD body "
                    "is probably not a closed solid, or it extends past the "
                    "domain — check the model in your CAD tool."
                )
            # Keep only the fluid region connected to the freestream.
            #
            # An assembly with several solids traps sealed pockets between them
            # (a jet-engine STEP produced a 40x40x40 void at x[126,166]). The cut
            # returns those as extra "fluid" volumes. They touch no boundary
            # condition, so SU2 would either diverge on them or — once they are
            # excluded from the fluid group — abort because their faces have no
            # adjacent volume element. The external region is the one that
            # reaches the upstream face of the domain box.
            box_tol = max(abs(up_x), rad) * 1e-6
            external, trapped = [], []
            for _, t in fluid:
                bb = occ.getBoundingBox(3, t)
                (external if bb[0] <= up_x + box_tol else trapped).append(t)
            if trapped:
                total_trapped = sum(abs(occ.getMass(3, t)) for t in trapped)
                logger.warning(
                    f"Discarding {len(trapped)} sealed internal void(s) "
                    f"(total {total_trapped:.4g} volume units) enclosed inside the "
                    f"CAD assembly — they carry no boundary condition and are not "
                    f"part of the external flow."
                )
            if not external:
                raise RuntimeError(
                    "No fluid region reaches the domain inlet — the CAD body "
                    "probably extends past the wind tunnel."
                )
            fluid_tags = external

            # Farfield = the six domain-box planes; everything else is the body.
            # A plane test is used instead of the rocket path's radius heuristic
            # because an arbitrary body has no "expected envelope".
            tol = min(rad, L) * 1e-4
            # Only surfaces that actually bound the fluid — an imported solid can
            # carry internal faces, and a marker element with no adjacent volume
            # element makes SU2 abort at mesh load.
            _bnd = gmsh.model.getBoundary(
                [(3, t) for t in fluid_tags],
                combined=True, oriented=False, recursive=False,
            )
            _surfs = [abs(t) for d, t in _bnd if d == 2] or \
                     [s[1] for s in gmsh.model.getEntities(2)]
            wall_surfs, far_surfs = [], []
            for stag in _surfs:
                cx, cy, cz = occ.getCenterOfMass(2, stag)
                on_box = (
                    abs(cx - up_x) < tol or abs(cx - down_x) < tol
                    or abs(abs(cy) - rad) < tol or abs(abs(cz) - rad) < tol
                )
                (far_surfs if on_box else wall_surfs).append(stag)
        else:
            # Discrete route: shell first (classifySurfaces would otherwise also
            # chew on the box faces), then the box, then a volume with a hole.
            # A wrap is a dense organic triangulation by construction: it has
            # no analytic patches to recover, and asking gmsh to find some can
            # hang it. Go straight to the triangulation.
            shell_loops, reparametrised = import_stl_into_geo(
                gmsh, Path(info.preview_stl),
                allow_reparametrization=not info.wrapped,
            )
            wall_surfs = [s[1] for s in gmsh.model.getEntities(2)]
            far_surfs, box_loop = _geo_box(gmsh, up_x, -rad, -rad, down_x, rad, rad)
            # Every body shell is a hole in the domain volume.
            fluid_tags = [gmsh.model.geo.addVolume([box_loop] + list(shell_loops))]
            gmsh.model.geo.synchronize()

        if not wall_surfs:
            raise RuntimeError(
                "No body surfaces survived classification — the CAD import "
                "produced an empty or fully-degenerate solid."
            )
        logger.info(
            f"Surface classification: {len(wall_surfs)} rocket_wall, "
            f"{len(far_surfs)} farfield, {len(fluid_tags)} fluid volume(s)"
        )

        gmsh.model.addPhysicalGroup(2, wall_surfs, name="rocket_wall")
        gmsh.model.addPhysicalGroup(2, far_surfs,  name="farfield")
        gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")

        # ── Feature-size clamp ────────────────────────────────────────────────
        # char comes from the frontal area, which describes how BIG the body is,
        # not how FINE it is. On a bluff, many-part assembly the two diverge
        # hard: the jet-engine STEP has a 202-unit equivalent radius but a
        # 35-unit median edge, so the frontal-area estimate asked for wall
        # elements ~3x longer than a typical feature. Facets then span across
        # thin parts and cut through each other, and the 3D pass dies with
        # "PLC Error: A segment and a facet intersect".
        #
        # Cap the wall size at the model's own median edge length (robust: the
        # degenerate short edges sit far below it, the domain box edges far
        # above). Only for exact B-Rep, where real topology exists to measure.
        if info.is_brep:
            try:
                _wall_curves = set()
                for _s in wall_surfs:
                    for _d, _t in gmsh.model.getBoundary(
                        [(2, _s)], combined=False, oriented=False, recursive=False
                    ):
                        if _d == 1:
                            _wall_curves.add(abs(_t))
                _lens = sorted(c for c in
                               (occ.getMass(1, t) for t in _wall_curves) if c > 0)
                if _lens:
                    _feature = _lens[len(_lens) // 2]          # median edge
                    if _feature < lc_wall:
                        logger.info(
                            f"Feature-size clamp: median wall edge {_feature:.4g} "
                            f"< lc_wall {lc_wall:.4g} — reducing wall size to match "
                            f"the CAD's own detail ({len(_lens)} edges)."
                        )
                        lc_wall = max(_feature, L * 1e-4)
                        lc_far = max(lc_far, lc_wall * 6.0)
            except Exception as e:
                logger.debug(f"Feature-size clamp unavailable: {e}")

        # ── Size fields ───────────────────────────────────────────────────────
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", wall_surfs)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", lc_wall)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", lc_far)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", char * 0.1)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", char * 8.0)

        # Keep the whole body envelope fine even where the distance field has
        # already relaxed (concave pockets, gaps between separate solids).
        hw = max(info.cross_width * 0.75, char)
        hh = max(info.cross_height * 0.75, char)
        f_body = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_body, "XMin", -0.25 * L)
        gmsh.model.mesh.field.setNumber(f_body, "XMax",  1.25 * L)
        gmsh.model.mesh.field.setNumber(f_body, "YMin", -hw)
        gmsh.model.mesh.field.setNumber(f_body, "YMax",  hw)
        gmsh.model.mesh.field.setNumber(f_body, "ZMin", -hh)
        gmsh.model.mesh.field.setNumber(f_body, "ZMax",  hh)
        gmsh.model.mesh.field.setNumber(f_body, "VIn",   lc_wall * 2.0)
        gmsh.model.mesh.field.setNumber(f_body, "VOut",  lc_far)

        f_wake = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_wake, "XMin",  L)
        gmsh.model.mesh.field.setNumber(f_wake, "XMax",  L + 3.0 * L)
        gmsh.model.mesh.field.setNumber(f_wake, "YMin", -2.0 * hw)
        gmsh.model.mesh.field.setNumber(f_wake, "YMax",  2.0 * hw)
        gmsh.model.mesh.field.setNumber(f_wake, "ZMin", -2.0 * hh)
        gmsh.model.mesh.field.setNumber(f_wake, "ZMax",  2.0 * hh)
        gmsh.model.mesh.field.setNumber(f_wake, "VIn",   char * 2.0)
        gmsh.model.mesh.field.setNumber(f_wake, "VOut",  lc_far)

        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_thr, f_body, f_wake])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

        # ── Generate ──────────────────────────────────────────────────────────
        gmsh.option.setNumber("Mesh.Algorithm", 6)          # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        # Curvature sizing needs a parametrisation. Exact B-Rep has one, so use
        # the same strength as the rocket path; a re-topologised STL only has an
        # approximate one, so ask for less and lean on the distance field.
        # Curvature sizing reads the surface parametrisation. A raw
        # triangulation has none, so asking for it there is meaningless at
        # best and destabilising at worst.
        gmsh.option.setNumber(
            "Mesh.MeshSizeFromCurvature",
            (20 if info.is_brep else 12) if reparametrised else 0,
        )
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)

        # Absolute size floor BEFORE the 2D pass. Real CAD carries degenerate
        # edges — this jet-engine STEP has a 2.2e-14-long curve among its 602 —
        # and curvature sizing on one of those asks for an element size near
        # zero, which crashes gmsh outright:
        #     OSError: exception: access violation reading 0x0000000000000008
        # The floor was previously applied only between the 2D and 3D passes,
        # so the 2D pass ran unprotected. Clamping here keeps curvature sizing
        # ON (24k triangles on this model vs 4.6k with curvature disabled)
        # while making the degenerate edge harmless.
        _lc_floor = lc_wall * 0.05
        gmsh.option.setNumber("Mesh.MeshSizeMin", _lc_floor)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
        try:
            _clens = [occ.getMass(1, t) for _, t in gmsh.model.getEntities(1)] \
                     if info.is_brep else []
            _clens = [c for c in _clens if c > 0]
            if _clens and min(_clens) < _lc_floor * 1e-3:
                logger.warning(
                    f"CAD contains near-degenerate edges (shortest "
                    f"{min(_clens):.3g} vs size floor {_lc_floor:.3g}). Clamped; "
                    f"features below the floor will not be resolved."
                )
        except Exception:
            pass

        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        # ── Meshing ladder ────────────────────────────────────────────────────
        # Imported CAD fails in ways a parametric body never does: overlapping
        # facets on one bad surface, segments piercing facets, degenerate edges.
        # A single "try Delaunay, else HXT" retry is not enough — on a real
        # jet-engine assembly both completed without raising and left ZERO tets.
        #
        # Each rung changes the thing that actually matters: first the 3D
        # kernel, then the 2D algorithm (MeshAdapt is far more forgiving of
        # dirty surfaces than Frontal-Delaunay), then the element size. Every
        # rung is verified by element count, because gmsh reports success on an
        # empty mesh.
        _algo2d = 6 if info.is_brep else 6
        ladder = [
            ("Delaunay 3D",            _algo2d, 1,  1.00, 0.05),
            ("HXT",                    _algo2d, 10, 1.00, 0.15),
            ("MeshAdapt 2D + Delaunay", 1,      1,  1.00, 0.05),
            ("MeshAdapt 2D + HXT",      1,      10, 0.50, 0.05),
            ("MeshAdapt 2D, half size", 1,      1,  0.25, 0.05),
        ]

        def _n_tets() -> int:
            try:
                _t, _g, _ = gmsh.model.mesh.getElements(3)
                return sum(len(g) for g in _g)
            except Exception:
                return 0

        meshed = False
        for i, (label, a2d, a3d, scale, floor_frac) in enumerate(ladder):
            try:
                if i > 0:
                    gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm", a2d)
                gmsh.option.setNumber("Mesh.Algorithm3D", a3d)
                gmsh.option.setNumber("Mesh.MeshSizeMin", lc_wall * scale * 0.05)
                gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMin",
                                      lc_wall * scale * floor_frac)
                if scale != 1.0:
                    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", lc_wall * scale)
                    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
                gmsh.model.mesh.generate(2)
                gmsh.model.mesh.generate(3)
                n = _n_tets()
                if n > 0:
                    logger.info(f"3D mesh built with {label}: {n:,} elements"
                                + ("" if i == 0 else f" (attempt {i + 1})"))
                    meshed = True
                    break
                logger.warning(f"{label} completed but produced no elements — "
                               f"trying the next strategy.")
            except Exception as e:
                logger.warning(f"{label} failed: {e}")

        # Exact B-Rep can be unmeshable for reasons no size setting fixes — one
        # self-overlapping face is enough ("Invalid boundary mesh (overlapping
        # facets) on surface 161"). The tessellation analyze_cad already made is
        # a clean watertight triangle soup of the same shape, and
        # classifySurfaces re-topologises it without ever consulting the bad
        # face. Lower fidelity, but a mesh instead of nothing. Signalled out of
        # the session so gmsh is finalized before the retry re-initialises it.
        fall_back_to_discrete = (
            not meshed and info.is_brep and not _force_discrete
        )
        if not meshed:
            logger.error("Every meshing strategy failed to produce elements.")

        if not fall_back_to_discrete:
            for label, opt in (("Gmsh default", ""), ("Netgen", "Netgen")):
                try:
                    gmsh.model.mesh.optimize(opt, force=True)
                    logger.info(f"Mesh optimization ({label}) complete")
                except Exception as e:
                    logger.warning(f"Mesh optimization ({label}) failed: {e}")

            _check_mesh_quality(gmsh, char, 0)

            # An empty volume mesh is a FAILURE, not a result. The ladder above
            # can exhaust itself without raising — on a jet-engine STEP every
            # rung "completed" and left zero tets. The old code then wrote a
            # 0.00 MB .su2, logged "SU2 mesh ready" and printed MESH_OK, so the
            # failure only surfaced later as an obscure SU2 load error.
            if not meshed:
                raise RuntimeError(
                    "3D meshing produced no elements. The geometry is "
                    "self-intersecting, or the assembly's parts interpenetrate "
                    "so the fluid volume is not closed."
                    + ("" if cad_wrap else
                       " Tick 'Wrap geometry' in the CAD panel — it rebuilds "
                       "the body as one clean shell and is the route that "
                       "survives exactly this. Note it gives an outer mold line "
                       "only: the surface is offset outward, fine detail is "
                       "rounded off and internal passages are sealed.")
                )

            su2_path = output_path.with_suffix(".su2")
            gmsh.write(str(su2_path))

    if fall_back_to_discrete:
        logger.warning(
            f"Exact B-Rep meshing of {cad_path.name} failed at every setting — "
            f"retrying through the discrete (tessellated) route. Fine CAD "
            f"features will be approximated by the triangulation."
        )
        return build_external_cad_mesh(
            cad_path=cad_path, output_path=output_path, refinement=refinement,
            domain_length_scale=domain_length_scale,
            domain_radius_scale=domain_radius_scale,
            custom_wall_size=custom_wall_size,
            target_element_count=target_element_count,
            flow_axis=flow_axis, cad_info=info.as_dict(),
            cad_units=info.units,
            cad_wrap=cad_wrap, cad_wrap_resolution=cad_wrap_resolution,
            _force_discrete=True,
        )

    su2_path = output_path.with_suffix(".su2")
    size_mb = su2_path.stat().st_size / 1e6
    logger.info(f"SU2 mesh ready: {su2_path}  ({size_mb:.2f} MB)")
    return su2_path


# ── Mesh Quality Checks ──────────────────────────────────────────────────────

def _revolve_profile_solid(occ, profile: list) -> list:
    """
    Build the axisymmetric body as an OCC solid of revolution about +X.

    ``profile`` is ``[(x, r), …]`` ascending in x, in the CFD frame (nose tip at
    x=0), as produced by ``cfd.geometry_exporter.cfd_profile``.

    Why a revolve rather than the old ``addCone`` + ``addCylinder`` pair: those
    two primitives can only express "straight cone, then constant-radius tube".
    A transition or boattail has no representation there, so it was silently
    dropped — the aft taper became body tube and the model grew a flat base it
    does not have, which lands directly on the base-drag integral. A revolved
    meridian carries whatever the assembly actually declares.

    The meridian is closed through the axis at both ends, so the result is a
    genuine solid (a nose that tapers to r≈0 still closes, because cfd_profile
    floors the radius just above zero rather than at it).

    Returns the volume dimTags.
    """
    if len(profile) < 2:
        raise ValueError("Profile needs at least two stations.")

    pts = [occ.addPoint(float(x), 0.0, float(r)) for x, r in profile]
    x_first = float(profile[0][0])
    x_last = float(profile[-1][0])
    p_axis_aft = occ.addPoint(x_last, 0.0, 0.0)
    p_axis_fwd = occ.addPoint(x_first, 0.0, 0.0)

    lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    lines.append(occ.addLine(pts[-1], p_axis_aft))    # close the base
    lines.append(occ.addLine(p_axis_aft, p_axis_fwd))  # along the axis
    lines.append(occ.addLine(p_axis_fwd, pts[0]))     # close the tip

    face = occ.addPlaneSurface([occ.addCurveLoop(lines)])
    occ.synchronize()

    # Full 2*pi in one shot where OCC allows it; two half turns otherwise
    # (some OCC builds refuse a 2*pi revolve of a face touching the axis).
    try:
        out = occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, 2.0 * math.pi)
        occ.synchronize()
        vols = [d for d in out if d[0] == 3]
        if not vols:
            raise RuntimeError("revolve produced no volume")
    except Exception as e:
        logger.warning(f"Full revolve failed ({e}) — retrying as two half turns.")
        occ.synchronize()
        half_a = occ.revolve([(2, face)], 0, 0, 0, 1, 0, 0, math.pi)
        occ.synchronize()
        va = [d for d in half_a if d[0] == 3]
        # The first revolve consumes the source face; rebuild it for the second.
        pts2 = [occ.addPoint(float(x), 0.0, float(r)) for x, r in profile]
        pa2 = occ.addPoint(x_last, 0.0, 0.0)
        pf2 = occ.addPoint(x_first, 0.0, 0.0)
        l2 = [occ.addLine(pts2[i], pts2[i + 1]) for i in range(len(pts2) - 1)]
        l2 += [occ.addLine(pts2[-1], pa2), occ.addLine(pa2, pf2),
               occ.addLine(pf2, pts2[0])]
        face2 = occ.addPlaneSurface([occ.addCurveLoop(l2)])
        occ.synchronize()
        half_b = occ.revolve([(2, face2)], 0, 0, 0, 1, 0, 0, -math.pi)
        occ.synchronize()
        vb = [d for d in half_b if d[0] == 3]
        if not (va and vb):
            raise RuntimeError("Profile revolve failed in both full and half form.")
        vols, _ = occ.fuse(va, vb, removeObject=True, removeTool=True)
        occ.synchronize()

    r_max = max(r for _, r in profile)
    logger.info(
        f"Body built by revolution: {len(profile)} profile stations, "
        f"x[{x_first:.4f}, {x_last:.4f}] m, r_max={r_max:.4f} m "
        f"→ {len(vols)} solid(s)"
    )
    return vols


def _audit_wall_is_manifold(gmsh, wall_surfs) -> Optional[int]:
    """Check that every wall triangle has exactly ONE 3D element behind it.

    Returns the number of wall triangles with something meshed on both sides,
    or None if the audit could not run. Zero is the only acceptable answer: the
    wall is the edge of the fluid, so a second element on the far side means
    two regions of the mesh occupy the same space.

    This exists because a prism-extrusion attempt produced exactly that and
    nothing else noticed. Element counts, Gmsh's own quality metrics, SU2's
    mesh-quality table and the .su2 file were all clean; the only symptom was a
    solve that would not converge. Cheap topology audits catch that class of
    fault, and expensive geometric ones do not.
    """
    if not wall_surfs:
        return None
    try:
        from collections import Counter

        # Face -> number of 3D elements using it. Only triangular faces matter:
        # a wall boundary is triangles either way.
        faces: Counter = Counter()
        for _, vol in gmsh.model.getEntities(3):
            types, _, nodes = gmsh.model.mesh.getElements(3, vol)
            for et, nd in zip(types, nodes):
                _, _, _, nn, _, _ = gmsh.model.mesh.getElementProperties(et)
                if nn == 4:      # tet: four triangular faces
                    combos = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
                elif nn == 6:    # prism: triangular top and bottom
                    combos = ((0, 1, 2), (3, 4, 5))
                else:
                    continue
                for i in range(0, len(nd), nn):
                    el = nd[i:i + nn]
                    for c in combos:
                        faces[frozenset(int(el[k]) for k in c)] += 1

        buried = 0
        checked = 0
        for s in wall_surfs:
            types, _, nodes = gmsh.model.mesh.getElements(2, s)
            for et, nd in zip(types, nodes):
                _, _, _, nn, _, _ = gmsh.model.mesh.getElementProperties(et)
                if nn != 3:
                    continue
                for i in range(0, len(nd), 3):
                    checked += 1
                    if faces[frozenset(int(v) for v in nd[i:i + 3])] > 1:
                        buried += 1
        if not checked:
            return None
        return buried
    except Exception as e:
        logger.warning(f"Wall manifoldness audit could not run: {e}")
        return None


def _check_mesh_quality(gmsh, body_r: float, n_bl_entities: int, wall_surfs=None):
    """
    Post-generation mesh quality validation.
    Checks element types, counts, quality metrics and wall manifoldness.
    """
    n_prisms = 0
    n_tets = 0
    n_pyramids = 0
    n_hexas = 0
    tags_3d: list[int] = []

    # Count 3D elements by type
    try:
        types, tags_per_type, _ = gmsh.model.mesh.getElements(3)
        for i, t in enumerate(types):
            tags_3d.extend(int(x) for x in tags_per_type[i])
        for i, t in enumerate(types):
            name, dim, order, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(t)
            n_elems = len(tags_per_type[i])

            if n_nodes == 6:       # Prism/Wedge
                n_prisms += n_elems
            elif n_nodes == 4:     # Tetrahedron
                n_tets += n_elems
            elif n_nodes == 5:     # Pyramid
                n_pyramids += n_elems
            elif n_nodes == 8:     # Hexahedron
                n_hexas += n_elems
    except Exception as e:
        logger.warning(f"Element count failed: {e}")

    total_3d = n_prisms + n_tets + n_pyramids + n_hexas
    prism_pct = (n_prisms / max(total_3d, 1)) * 100

    logger.info(
        f"Mesh quality report:\n"
        f"  Total 3D elements: {total_3d:,}\n"
        f"  Prisms (BL):       {n_prisms:,}  ({prism_pct:.1f}%)\n"
        f"  Tetrahedra:        {n_tets:,}\n"
        f"  Pyramids (trans):  {n_pyramids:,}\n"
        f"  Hexahedra:         {n_hexas:,}"
    )

    if n_bl_entities > 0 and n_prisms == 0:
        logger.error(
            "BL extrusion returned entities but no prisms in final mesh! "
            "Possible normal inversion or topology disconnect."
        )
    elif n_prisms > 0:
        logger.info(f"[OK] Prism boundary layer confirmed: {n_prisms:,} elements")

    # ── Quality metrics via Gmsh ──────────────────────────────────────────────
    # Sample the REAL 3D element tags. This used to ask for tags 1..10000, which
    # are whatever Gmsh numbered first — the 1D and 2D elements — so the figure
    # reported was never the quality of the volume mesh being exported. That was
    # survivable while every cell was a tet from a single algorithm; it is not
    # now, because the prism stack's worst cells are exactly what this is for
    # (fin-root and nose-tip junctions, where the extrusion fronts converge).
    if tags_3d:
        try:
            # SICN = Scaled Inverse Condition Number (1 = perfect, 0 = degenerate)
            sample = tags_3d if len(tags_3d) <= 200_000 else tags_3d[::max(1, len(tags_3d) // 200_000)]
            # getElementQualities returns a numpy array, so this must test
            # length — `if sicn_data:` raises "truth value of an array ... is
            # ambiguous", which the except below then swallowed as "quality
            # check unavailable".
            sicn_data = gmsh.model.mesh.getElementQualities(sample, "minSICN")
            if len(sicn_data):
                import statistics
                min_q = min(sicn_data)
                avg_q = statistics.mean(sicn_data)
                n_negative = sum(1 for q in sicn_data if q < 0)
                n_poor = sum(1 for q in sicn_data if 0 <= q < 0.01)
                logger.info(
                    f"  Quality (SICN): min={min_q:.4f}  avg={avg_q:.4f}  "
                    f"negative={n_negative}  near-degenerate(<0.01)={n_poor}  "
                    f"(sampled {len(sicn_data):,} of {len(tags_3d):,} 3D elements)"
                )
                if n_negative > 0:
                    logger.warning(
                        f"  {n_negative} elements have negative Jacobians — "
                        f"SU2 may produce poor convergence"
                    )
        except Exception as e:
            # getElementQualities may not be available in all Gmsh builds
            logger.warning(f"Element quality check unavailable: {e}")

    # ── Wall manifoldness ─────────────────────────────────────────────────────
    buried = _audit_wall_is_manifold(gmsh, wall_surfs)
    if buried is None:
        pass
    elif buried:
        logger.error(
            f"MESH INVALID: {buried:,} wall triangles have a 3D element on both "
            f"sides, so the rocket wall is not a boundary of the fluid. Two "
            f"parts of the mesh occupy the same space and SU2 will not converge "
            f"on it — the residual will not fall below its starting value. This "
            f"is the failure mode prism extrusion produces (see the module "
            f"docstring); do not trust any result from this mesh."
        )
    else:
        logger.info("[OK] Wall is manifold: every wall face bounds exactly one cell")


# ── Fin geometry ──────────────────────────────────────────────────────────────

def _profile_r_at(profile: list, x: float) -> float:
    """Linearly interpolated body radius at axial station ``x``."""
    if not profile:
        return 0.0
    if x <= profile[0][0]:
        return float(profile[0][1])
    if x >= profile[-1][0]:
        return float(profile[-1][1])
    for i in range(1, len(profile)):
        x0, r0 = profile[i - 1]
        x1, r1 = profile[i]
        if x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            return float(r0 + t * (r1 - r0))
    return float(profile[-1][1])


def _add_fins(occ, rocket: dict, total_L: float) -> list:
    """
    Create accurate trapezoidal fins using OCC wire → face → extrude.
    """
    parts   = []
    n_fins  = int(rocket.get("fin_count", 4))
    fin_h   = rocket.get("fin_height", rocket["body_radius"] * 1.0)
    fin_Cr  = rocket.get("fin_root",   rocket["body_length"] * 0.25)
    # Finless / degenerate fin definition — build nothing rather than let OCC
    # fail on a zero-extent box further down.
    if n_fins <= 0 or fin_h <= 1e-9 or fin_Cr <= 1e-9:
        if n_fins > 0:
            logger.warning(
                f"Fin set has zero extent (count={n_fins}, h={fin_h:.4g}, "
                f"Cr={fin_Cr:.4g}) — skipping fins."
            )
        return parts
    fin_Ct  = rocket.get("fin_tip",    fin_Cr * 0.5)
    sweep   = math.radians(rocket.get("fin_sweep_deg", 0.0))
    fin_t   = rocket.get("fin_thick",  max(0.002, rocket["body_radius"] * 0.04))
    body_r  = rocket["body_radius"]

    sweep_offset = fin_h * math.tan(sweep)

    # Axial placement: fin_z_base_k2 is the K2-frame z of the fin root
    # trailing edge (distance from the rocket base; 0 = at the nozzle).
    # CFD frame measures from the nose, so x_TE = total_L - z_base_k2.
    # Default 0 keeps the legacy tail-mounted behaviour for STL-estimated
    # geometry that carries no fin position.
    z_base_k2 = rocket.get("fin_z_base_k2", 0.0)
    x_root_TE = min(total_L, max(fin_Cr, total_L - z_base_k2))
    x_root_LE = x_root_TE - fin_Cr
    x_tip_LE  = x_root_LE + sweep_offset
    x_tip_TE  = x_tip_LE  + fin_Ct

    if x_tip_TE > total_L + 1e-4:
        scale   = (total_L - x_tip_LE) / fin_Ct if fin_Ct > 1e-6 else 1.0
        fin_Ct  = fin_Ct * max(0.1, scale)
        x_tip_TE = x_tip_LE + fin_Ct

    # Root radius must follow the body the fin is actually mounted on.
    #
    # The root edge used to sit at the nominal body_r. That was exact while the
    # body was a plain cylinder, but the body is now revolved from the real
    # profile, so a fin overlapping a boattail or flare had its root floating
    # off (or buried in) the surface — and the fuse then left the fin as a
    # detached solid. Root below the SMALLEST local radius under the chord so
    # the fin always penetrates, and let the boolean union trim the buried part;
    # the exposed span then follows the taper automatically.
    profile = rocket.get("profile") or []
    profile = [(float(p[0]), float(p[1])) for p in profile if len(p) >= 2]
    if profile:
        xs_chk = [x_root_LE + (x_tip_TE - x_root_LE) * k / 8.0 for k in range(9)]
        r_local_min = min(_profile_r_at(profile, x) for x in xs_chk)
        r_local_mid = _profile_r_at(profile, 0.5 * (x_root_LE + x_root_TE))
    else:
        r_local_min = r_local_mid = body_r
    r_root = max(r_local_min * 0.90, 1e-6)   # sink 10% for a clean intersection
    r_tip = r_local_mid + fin_h

    logger.info(
        f"Fins: n={n_fins}  h={fin_h:.3f}  Cr={fin_Cr:.3f}  Ct={fin_Ct:.3f}  "
        f"sweep={math.degrees(sweep):.1f} deg  t={fin_t:.3f}  "
        f"x=[{x_root_LE:.3f}->{x_tip_TE:.3f}]  "
        f"r=[{r_root:.4f}->{r_tip:.4f}] (local body r={r_local_mid:.4f})"
    )

    for i in range(n_fins):
        angle = 2.0 * math.pi * i / n_fins

        try:
            p0 = occ.addPoint(x_root_LE, r_root, 0.0)
            p1 = occ.addPoint(x_root_TE, r_root, 0.0)
            p2 = occ.addPoint(x_tip_TE,  r_tip,  0.0)
            p3 = occ.addPoint(x_tip_LE,  r_tip,  0.0)

            l0 = occ.addLine(p0, p1)
            l1 = occ.addLine(p1, p2)
            l2 = occ.addLine(p2, p3)
            l3 = occ.addLine(p3, p0)

            loop = occ.addCurveLoop([l0, l1, l2, l3])
            face = occ.addPlaneSurface([loop])

            extruded = occ.extrude([(2, face)], 0, 0, fin_t)
            vol_tags  = [e[1] for e in extruded if e[0] == 3]
            if not vol_tags:
                raise RuntimeError("Extrude returned no 3D volume")
            vol_tag = vol_tags[0]

            occ.translate([(3, vol_tag)], 0, 0, -fin_t / 2)
            occ.rotate([(3, vol_tag)], 0, 0, 0, 1, 0, 0, angle)
            parts.append((3, vol_tag))

        except Exception as e:
            logger.warning(f"Fin {i} OCC construction failed ({e}); using box fallback")
            tag = occ.addBox(
                x_root_LE, r_root, -fin_t / 2,
                fin_Cr, max(r_tip - r_root, fin_h), fin_t,
            )
            occ.rotate([(3, tag)], 0, 0, 0, 1, 0, 0, angle)
            parts.append((3, tag))

    return parts


# ── Geometry parsing ──────────────────────────────────────────────────────────

def _parse_rocket_geometry(stl_path: Path) -> dict:
    """
    Extract rocket geometry from the STL using point-cloud statistics.
    """
    import pyvista as pv
    import numpy as np

    m   = pv.read(str(stl_path))
    pts = np.array(m.points)
    b   = m.bounds

    total_L = max(abs(b[5] - b[4]), 0.1)
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)

    body_r = float(np.percentile(radii, 90))
    body_r = max(body_r, 0.01)

    fin_tip_r = max(abs(b[1]), abs(b[0]), abs(b[3]), abs(b[2]))
    fin_h     = max(fin_tip_r - body_r, body_r * 0.3)

    nose_L = min(6.0 * body_r, total_L * 0.40)
    body_L = total_L - nose_L

    fin_cr = body_L * 0.30
    fin_t  = max(0.002, body_r * 0.04)

    logger.info(
        f"STL geometry: total_L={total_L:.3f} m  body_r={body_r:.4f} m  "
        f"fin_tip_r={fin_tip_r:.4f} m  fin_h={fin_h:.4f} m  "
        f"nose_L={nose_L:.3f} m  fineness={(total_L/(2*body_r)):.1f}"
    )

    return {
        "length":       total_L,
        "body_radius":  body_r,
        "nose_radius":  body_r,
        "nose_length":  nose_L,
        "body_length":  body_L,
        "fin_count":    4,
        "fin_height":   fin_h,
        "fin_root":     fin_cr,
        "fin_thick":    fin_t,
    }


def _stl_bounds(stl_path: Path) -> tuple[float, float]:
    try:
        import pyvista as pv
        m = pv.read(str(stl_path))
        b = m.bounds
        L = abs(b[5] - b[4])
        r = max(abs(b[1] - b[0]), abs(b[3] - b[2])) / 2
        return max(L, 0.1), max(r, 0.02)
    except Exception as e:
        logger.warning(f"STL bounds error ({e}) — using defaults")
        return 1.0, 0.05
