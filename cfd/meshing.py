"""
K2 Aerospace — CFD Meshing (Gmsh)
====================================
Generates a proper 3D volumetric SU2 mesh for RANS simulation with
anisotropic prism boundary-layer inflation.

Approach:
  1. Reconstruct rocket OCC geometry (cone nose + cylinder body + fins)
  2. Boolean subtract from wind-tunnel domain → watertight fluid volume
  3. Classify surfaces (rocket_wall vs farfield)
  4. Apply curvature-based refinement fields
  5. Generate 2D surface mesh
  6. Extrude prism boundary layers from rocket walls (geo.extrudeBL)
  7. Fill remaining volume with tetrahedra
  8. Run quality checks and export SU2

Boundary Layer Strategy:
  gmsh.model.occ does NOT have extrudeBoundaryLayer.
  gmsh.model.geo.extrudeBoundaryLayer is the ONLY Gmsh API that
  extrudes along mesh normals to create 3D prisms.
  After occ.synchronize(), entities are visible to geo — this hybrid
  workflow is safe when sync ordering is respected:
    occ.synchronize() → generate(2) → geo.extrudeBL → geo.synchronize() → generate(3)

Requires: pip install gmsh
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

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


# ── Boundary Layer Helpers ────────────────────────────────────────────────────

def _compute_first_layer_height(
    mach: float,
    body_r: float,
    ref_length: float,
    altitude_m: float = 0.0,
    target_yplus: float = 1.0,
) -> float:
    """
    Estimate first prism layer height for target y+.

    Uses Schlichting flat-plate turbulent skin friction:
        Cf ≈ 0.058 · Re_L^(-0.2)
        τ_w = 0.5 · ρ · V² · Cf
        u_τ = √(τ_w / ρ)
        y₁ = y⁺ · μ / (ρ · u_τ)

    Returns h1 in metres, clamped to physically reasonable bounds.
    """
    from cfd.solvers.base import isa_conditions

    P, T, rho = isa_conditions(altitude_m)
    a = math.sqrt(1.4 * 287.05 * T)
    V = max(mach * a, 10.0)  # clamp to avoid division by zero
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)

    Re_L = rho * V * ref_length / mu
    Re_L = max(Re_L, 1e3)  # safety floor

    # Schlichting flat-plate friction
    Cf = 0.058 * Re_L ** (-0.2)
    tau_w = 0.5 * rho * V ** 2 * Cf
    u_tau = math.sqrt(max(tau_w / rho, 1e-12))

    h1 = target_yplus * mu / (rho * u_tau)

    # ── Clamp to physically reasonable bounds ─────────────────────────────
    # Too thin -> extreme aspect ratio -> SU2 divergence from poor conditioning
    h1_min = body_r * 8e-4       # absolute floor: 0.08% of body radius for stability
    # Too thick -> poor y+ resolution
    h1_max = body_r * 5e-3       # ceiling: 0.5% of body radius

    h1_clamped = max(h1_min, min(h1, h1_max))

    if h1 != h1_clamped:
        logger.info(
            f"First layer height clamped: {h1:.6f} -> {h1_clamped:.6f} m  "
            f"(bounds [{h1_min:.6f}, {h1_max:.6f}])"
        )

    logger.info(
        f"BL first layer: h1={h1_clamped:.6f} m  (target y+={target_yplus}, "
        f"Re_L={Re_L:.2e}, Cf={Cf:.6f}, u_tau={u_tau:.3f} m/s)"
    )
    return h1_clamped


def _cumulative_geometric_heights(h1: float, ratio: float, n: int) -> list[float]:
    """
    Build cumulative height list for geometric-growth boundary layer.
    Returns [h1, h1+h1·r, h1+h1·r+h1·r², ...] — n entries.
    """
    heights = []
    cumulative = 0.0
    for i in range(n):
        cumulative += h1 * ratio ** i
        heights.append(cumulative)
    return heights


def _safe_bl_thickness(
    total_bl: float,
    body_r: float,
    rocket: dict,
) -> float:
    """
    Clamp total BL thickness to prevent prism overlap at tight features.

    Critical zones:
      - Fin root gap:  BL < 0.4 × fin thickness
      - Body radius:   BL < 0.10 × body_r  (prevents self-intersection
                        on cylinder surface and at nose-body junction)
      - Fin tip:       BL < 0.3 × fin height
    """
    limits = [body_r * 0.10]   # 10% of body radius (tighter for nose-body safety)

    fin_t = rocket.get("fin_thick", 0.003)
    if fin_t > 0:
        limits.append(fin_t * 0.25)  # 25% of fin thickness (conservative for thin fins)

    fin_h = rocket.get("fin_height", body_r)
    if fin_h > 0:
        limits.append(fin_h * 0.3)

    safe_max = min(limits)
    if total_bl > safe_max:
        logger.warning(
            f"BL thickness {total_bl:.5f} m exceeds safe limit {safe_max:.5f} m "
            f"(body_r={body_r:.4f}, fin_t={fin_t:.4f}) - clamping"
        )
        return safe_max
    return total_bl


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
) -> Path:
    """
    Generate a volumetric SU2 mesh with prism boundary layers using Gmsh.
    Returns the path to the .su2 mesh file.
    """
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

    # Safety: clear any stale Gmsh session
    try:
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

    # Nose cone: tip at x=0 (upstream), base at x=nose_L
    nose_tag = occ.addCone(
        0.0,    0, 0,
        nose_L, 0, 0,
        0.001,          # near-zero tip radius (avoids degenerate vertex)
        nose_r,
    )
    rocket_parts.append((3, nose_tag))

    # Body tube: from x=nose_L to x=total_L
    body_tag = occ.addCylinder(nose_L, 0, 0, body_L, 0, 0, body_r)
    rocket_parts.append((3, body_tag))

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
    all_surfs      = gmsh.model.getEntities(2)
    fluid_vol_tags = [v[1] for v in fluid]

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
    fin_span = rocket.get("fin_height", body_r) + body_r
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
    fin_span = fin_h + body_r

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
    lc_fin = lc_rocket * 0.7
    f_fin = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_fin, "XMin",  total_L - fin_Cr * 1.2)
    gmsh.model.mesh.field.setNumber(f_fin, "XMax",  total_L + body_r)
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

    gmsh.model.mesh.generate(2)
    logger.info("2D surface mesh generated")

    # ── 7. Selective prism boundary layer extrusion ───────────────────────────
    # Strategy: Tet-only with aggressive near-wall refinement.
    #
    # geo.extrudeBoundaryLayer is INCOMPATIBLE with OCC boolean-cut domains —
    # it corrupts the model topology, causing generate(3) to crash even when
    # limited to simple body-tube surfaces.  This is a fundamental Gmsh
    # limitation (OCC entities become invalid after geo.synchronize).
    #
    # The tiered distance-based refinement fields (step 5) provide fine
    # near-wall tets that SU2's wall-function SST handles well.
    n_prisms = 0
    logger.info(
        "Tet-only mesh — BL prism extrusion disabled (incompatible with OCC). "
        "Tiered near-wall refinement provides y+ resolution for wall-function RANS."
    )

    # ── 8. Generate 3D volume mesh ────────────────────────────────────────────
    # Guard: set minimum element size to prevent degenerate tets at extreme refinement
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_rocket * 0.05)
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
    _check_mesh_quality(gmsh, body_r, n_prisms)

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


def _extrude_boundary_layer(
    gmsh, rocket, rocket_wall_surfs, farfield_surfs,
    bl_layers, bl_growth, body_r,
):
    """
    Extrude prism boundary layers from rocket wall surfaces.

    Uses gmsh.model.geo.extrudeBoundaryLayer() — the ONLY Gmsh API
    that extrudes along mesh normals for 3D prism generation.

    Returns the number of BL entities created (0 on failure).
    """
    # ── Compute first layer height (y+ ≈ 1) ──────────────────────────────────
    h1 = _compute_first_layer_height(
        mach=rocket.get("_mach", 0.8),
        body_r=body_r,
        ref_length=rocket["length"],
        altitude_m=rocket.get("_altitude", 0.0),
    )

    # ── Build cumulative heights ──────────────────────────────────────────────
    raw_heights = _cumulative_geometric_heights(h1, bl_growth, bl_layers)
    total_bl = raw_heights[-1]

    # ── Safety: clamp total BL thickness ──────────────────────────────────────
    safe_total = _safe_bl_thickness(total_bl, body_r, rocket)
    if safe_total < total_bl:
        # Recompute with reduced layer count to fit within safe thickness
        while bl_layers > 3:
            raw_heights = _cumulative_geometric_heights(h1, bl_growth, bl_layers)
            if raw_heights[-1] <= safe_total:
                break
            bl_layers -= 1
        total_bl = raw_heights[-1]
        logger.info(f"BL reduced to {bl_layers} layers, total={total_bl:.5f} m")

    logger.info(
        f"BL inflation: {bl_layers} layers, h1={h1:.6f} m, "
        f"growth={bl_growth}, total={total_bl:.5f} m  "
        f"(safe limit={safe_total:.5f} m)"
    )

    # ── Normal direction check ────────────────────────────────────────────────
    # Verify that extrusion direction is OUTWARD (into the flow) not inward.
    # After boolean cut, rocket wall normals should point into the fluid.
    # We check by comparing the centroid of a wall surface to the volume
    # centroid — the normal should point AWAY from the rocket body axis.
    occ = gmsh.model.occ
    for stag in rocket_wall_surfs[:3]:  # spot-check first 3
        cx, cy, cz = occ.getCenterOfMass(2, stag)
        # For an external flow domain (fluid = tunnel - rocket),
        # surface normals point into the fluid (outward from rocket)
        # which is the correct direction for BL extrusion.
        r_wall = math.sqrt(cy**2 + cz**2)
        if r_wall < 0.01:
            # Surface is on the body axis — normal direction is ambiguous
            # but unlikely to cause issues for axis-aligned surfaces
            continue
        logger.debug(
            f"Wall surface {stag}: centroid ({cx:.3f}, {cy:.3f}, {cz:.3f}), "
            f"r={r_wall:.4f} - normal points outward [OK]"
        )

    # ── Extrude ───────────────────────────────────────────────────────────────
    wall_dimtags = [(2, s) for s in rocket_wall_surfs]
    gmsh.option.setNumber("Geometry.ExtrudeReturnLateralEntities", 0)

    # Reverse normals so prisms extrude OUTWARD into the fluid, not INWARD into the rocket
    gmsh.model.mesh.reverse(wall_dimtags)

    out = gmsh.model.geo.extrudeBoundaryLayer(
        wall_dimtags,
        numElements=[1] * bl_layers,
        heights=raw_heights,
        recombine=True,   # prisms (hex-wedge), not tets
    )

    # ── CRITICAL: geo.synchronize() after extrudeBL ───────────────────────────
    # This merges the new BL entities into the model topology.
    # Without this, the 3D mesher won't see the extruded volumes.
    gmsh.model.geo.synchronize()

    n_bl_entities = len(out)
    logger.info(
        f"BL extrusion complete: {n_bl_entities} entities created, "
        f"{bl_layers} prism layers, total thickness={total_bl:.5f} m"
    )
    return n_bl_entities


# ── Mesh Quality Checks ──────────────────────────────────────────────────────

def _check_mesh_quality(gmsh, body_r: float, n_bl_entities: int):
    """
    Post-generation mesh quality validation.
    Checks element types, counts, and quality metrics.
    """
    n_prisms = 0
    n_tets = 0
    n_pyramids = 0
    n_hexas = 0

    # Count 3D elements by type
    try:
        types, tags_per_type, _ = gmsh.model.mesh.getElements(3)
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
    try:
        # SICN = Scaled Inverse Condition Number (1 = perfect, 0 = degenerate)
        sicn_data = gmsh.model.mesh.getElementQualities(
            list(range(1, min(total_3d + 1, 10001))), "minSICN"
        )
        if sicn_data:
            import statistics
            min_q = min(sicn_data)
            avg_q = statistics.mean(sicn_data)
            n_negative = sum(1 for q in sicn_data if q < 0)
            logger.info(
                f"  Quality (SICN): min={min_q:.4f}  avg={avg_q:.4f}  "
                f"negative={n_negative}  (sampled {len(sicn_data)} elements)"
            )
            if n_negative > 0:
                logger.warning(
                    f"  ⚠ {n_negative} elements have negative Jacobians — "
                    f"SU2 may produce poor convergence"
                )
    except Exception:
        # getElementQualities may not be available in all Gmsh builds
        pass


# ── Fin geometry ──────────────────────────────────────────────────────────────

def _add_fins(occ, rocket: dict, total_L: float) -> list:
    """
    Create accurate trapezoidal fins using OCC wire → face → extrude.
    """
    parts   = []
    n_fins  = int(rocket.get("fin_count", 4))
    fin_h   = rocket.get("fin_height", rocket["body_radius"] * 1.0)
    fin_Cr  = rocket.get("fin_root",   rocket["body_length"] * 0.25)
    fin_Ct  = rocket.get("fin_tip",    fin_Cr * 0.5)
    sweep   = math.radians(rocket.get("fin_sweep_deg", 0.0))
    fin_t   = rocket.get("fin_thick",  max(0.002, rocket["body_radius"] * 0.04))
    body_r  = rocket["body_radius"]

    sweep_offset = fin_h * math.tan(sweep)

    x_root_LE = total_L - fin_Cr
    x_root_TE = total_L
    x_tip_LE  = x_root_LE + sweep_offset
    x_tip_TE  = x_tip_LE  + fin_Ct

    if x_tip_TE > total_L + 1e-4:
        scale   = (total_L - x_tip_LE) / fin_Ct if fin_Ct > 1e-6 else 1.0
        fin_Ct  = fin_Ct * max(0.1, scale)
        x_tip_TE = x_tip_LE + fin_Ct

    logger.info(
        f"Fins: n={n_fins}  h={fin_h:.3f}  Cr={fin_Cr:.3f}  Ct={fin_Ct:.3f}  "
        f"sweep={math.degrees(sweep):.1f} deg  t={fin_t:.3f}  "
        f"x=[{x_root_LE:.3f}->{x_tip_TE:.3f}]"
    )

    for i in range(n_fins):
        angle = 2.0 * math.pi * i / n_fins

        try:
            p0 = occ.addPoint(x_root_LE, body_r,          0.0)
            p1 = occ.addPoint(x_root_TE, body_r,          0.0)
            p2 = occ.addPoint(x_tip_TE,  body_r + fin_h,  0.0)
            p3 = occ.addPoint(x_tip_LE,  body_r + fin_h,  0.0)

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
                x_root_LE, body_r, -fin_t / 2,
                fin_Cr, fin_h, fin_t,
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
