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
  TWO paths. ``bl_prisms=False`` (default) is the tet-only mesh described below.
  ``bl_prisms=True`` builds real prism layers via :func:`_build_bl_mesh`.

  Prism layers ARE reachable on this domain — measured 2026-08-19, against the
  conclusion the rest of this docstring used to draw. What was missing was not a
  Gmsh capability but a sequence:

  * The extrusion source must be a **reparametrised** surface
    (``classifySurfaces(forReparametrization=True)`` + ``createGeometry()``).
    Raw discrete surfaces fail with "Could not find extruded node", and so do
    exact OCC surfaces — which corrects the claim below that extrusion "runs"
    on the OCC domain. It runs; it never yields a meshable layer.
  * The farfield box must be built **after** the extrusion. Built before, its
    entities make ``extrudeBoundaryLayer`` fail "Could not replace surface N in
    Coherence" on some tessellations and pass on others, which looks exactly
    like a fragile dependence on STL resolution and is not.
  * The outer region must be ``addVolume([box_loop, top_loop])`` — the box with
    the stack's outer shell as a HOLE. That is what makes the overlap failure
    below structurally impossible rather than merely detected after the fact.

  Measured on the finned test rocket at M=0.5, Re=1.16e7: y+ median 0.41 with
  100% of wall points below 1 (tet-only: ~3500), Cf median 2.9e-3 against a
  1.9e-3 flat-plate reference, wall manifoldness 0 buried of 15,244, and SU2
  reaching Exit Success with rms[Rho] falling 2.7 decades and CD stationary.

  Known limit, not yet fixed: the workable STL-carrier resolution is not
  contiguous. Too fine fails ``classifySurfaces``; some values fail
  "PLC Error: A segment and a facet intersect", which is the stack
  self-intersecting where extrusion fronts converge at fin trailing edges and
  tips (Gmsh's BL has no convex-corner treatment). Both raise rather than
  falling back, deliberately — see _build_bl_mesh.

  ---- The tet-only path, and why it is what it is ----

  Tet-only with aggressive near-wall refinement, from the tiered distance
  fields alone. Prism layers were long believed out of reach on this domain,
  and the reason is more specific than "extrudeBoundaryLayer crashes" — that
  claim was re-tested on Gmsh 4.15.2 and is no longer what happens:

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

  The useful version of that: an extruded mesh that is never volume-rebuilt is
  WORSE than no prisms, because nothing downstream reports it as broken.
  _check_mesh_quality audits wall manifoldness for exactly this, so a failed
  attempt fails loudly. (The rebuild is no longer the blocker it looks like
  here — bounding the tets with a surface-loop hole sidesteps the quad problem
  entirely. See the top of this docstring.)

  Consequences on the tet-only path: the first cell sits near y+ 3500, far
  outside the
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

# Bump whenever a change here would produce a different mesh from identical
# inputs. Callers that cache meshes (the validation sweeps) mix this into their
# cache key, so an old mesh is never silently reused across a mesher change —
# which is how an entire AGARD-B sweep once "re-ran" and reproduced the previous
# numbers to six digits after the size fields had been rewritten.
MESHER_REVISION = "2026-08-30-farfield-and-growth"

# Refinement levels: mesh size near rocket = fraction of body radius
_REFINEMENT_FACTORS = {
    "coarse":     {"wall_frac": 0.50,  "far_frac": 6.0},
    "medium":     {"wall_frac": 0.28,  "far_frac": 10.0},
    "fine":       {"wall_frac": 0.18,  "far_frac": 15.0},
    "very_fine":  {"wall_frac": 0.10,  "far_frac": 20.0},
    "ultra_fine": {"wall_frac": 0.05,  "far_frac": 30.0},
}

# -- Domain extent and cell grading -------------------------------------------
#
# Both of these were measured wrong on the shipped mesher and both distorted the
# forces, so the numbers below are the load-bearing part of this module.
#
# FARFIELD PLACEMENT. The tunnel radius used to be ``body_r * 20`` alone. On the
# canonical 2 m rocket that is 1.02 m -- HALF a body length from the axis, and
# 6x the fin semi-span. A boundary that close is a tunnel wall, not free air:
# the blockage it imposes shows up directly in the coefficients, and it was
# measured doing so -- a symmetric body at M=0.8 and EXACTLY zero angle of
# attack reported CL = -0.0406, which at a lift slope of ~0.05/deg is 0.8
# degrees of angle the vehicle does not have. External aerodynamics wants the
# boundary 15-50 reference lengths out; the floors below are the low end of
# that, applied as a FLOOR on top of the caller's own scale so no existing
# configuration ever gets a smaller domain than it asked for.
_FARFIELD_REF_LENGTHS  = 15.0   # lateral half-width
_UPSTREAM_REF_LENGTHS  = 15.0   # ahead of the nose
_WAKE_REF_LENGTHS      = 25.0   # behind the tail -- longer, the wake convects
#
# CELL GRADING. For a gmsh Threshold field the size varies linearly with wall
# distance, s(d) = SizeMin + alpha*d, and the ratio between neighbouring cells
# is then exactly 1 + alpha -- constant everywhere in the band. The shipped
# settings (SizeMin=lc_wall at 0.1*body_r, SizeMax=lc_far at 5*body_r) gave
# alpha = 3.0 on the "fine" preset, i.e. every cell FOUR TIMES its neighbour,
# with the whole wall-to-farfield transition crossed in 1.5 cells.
#
# Measured on a shipped 231k-cell solution mesh, cell size by radius:
#     r=[0.00,0.50)  n=222,945  h_med=0.018 m
#     r=[0.50,0.80)  n=  3,444  h_med=0.032 m
#     r=[0.80,1.10)  n=    525  h_med=0.846 m   <- 47x in one band
# i.e. 97% of the cells inside a third of the radius and essentially nothing
# outside it. A Roe/MUSCL flux across a 47x size jump produces spurious entropy
# and reflects disturbances back onto the body, and Green-Gauss gradients are
# not even first-order consistent there.
#
# 1.2 is the standard external-aero ceiling. Everything downstream -- DistMax on
# the wall field, the Thickness on each box field -- is DERIVED from it rather
# than set independently, so the grading cannot be broken by tuning one field.
_MAX_GROWTH_RATIO = 1.2
#
# Fraction of the tunnel radius over which the growth is allowed to run before
# the size saturates at lc_far. Kept below 1 so the outer domain is uniform
# (cheap) rather than still growing when it reaches the boundary.
_GROWTH_SPAN_FRAC = 0.35


def _growth_alpha() -> float:
    """Size gradient d(size)/d(distance) that yields ``_MAX_GROWTH_RATIO``."""
    return _MAX_GROWTH_RATIO - 1.0


def _graded_far_size(lc_wall: float, tun_radius: float, lc_far_preset: float) -> float:
    """Far-field cell size consistent with the growth ratio and the domain.

    The preset far size is a *floor*: honouring it literally on a domain 15
    reference lengths across would mean 0.77 m cells filling a 30 m radius --
    hundreds of thousands of cells describing undisturbed air. The size the
    grading actually reaches after ``_GROWTH_SPAN_FRAC`` of the radius is used
    when that is coarser, which is what keeps the count tractable once the
    domain is the size external aerodynamics needs.
    """
    reach = lc_wall + _growth_alpha() * tun_radius * _GROWTH_SPAN_FRAC
    return max(lc_far_preset, reach)


def _graded_dist_max(lc_wall: float, lc_far: float) -> float:
    """Distance at which a Threshold band reaches ``lc_far`` at the growth cap."""
    return max((lc_far - lc_wall) / _growth_alpha(), lc_wall * 4.0)


def _box_thickness(lc_in: float, lc_far: float) -> float:
    """Transition thickness that lets a Box field relax out at the growth cap.

    A gmsh Box field without ``Thickness`` is a step: VIn inside, VOut outside,
    nothing between. That is where the 47x jump measured above came from -- the
    fin box held 0.7*lc_wall over a +/-1.5 fin-span cube and then fell straight
    to the far-field size at its face. Giving every box the same relaxation the
    wall field uses removes the discontinuity by construction.
    """
    return max((lc_far - lc_in) / _growth_alpha(), lc_in * 4.0)


def _count_from_wall_size(
    lc_wall: float, body_r: float, total_L: float, tun_radius: float,
) -> float:
    """Predicted tet count for a wall size, under the graded field above.

    Integrates the shell-by-shell cell count outward from the body,

        N = integral over d of  A(d) / s(d)^3

    with ``s(d) = lc_wall + alpha*d`` saturating at ``lc_far``, and ``A(d)`` the
    area of the surface offset ``d`` from a cylinder of radius ``body_r`` and
    length ``total_L``. Crude about the body's real shape and exact about the
    grading, which is the right way round: the count is dominated by the first
    few centimetres, where every body looks like its own offset surface.

    Replaces a domain-volume estimate (``N = V_domain / (lc^3/6)``) that could
    not survive this module's real domain. Spread over a tunnel 15 reference
    lengths across, that formula answers "coarser than the medium preset" for
    any target a user would type -- on the ONERA M6 wing it asked for a coarser
    wall at 900,000 elements than at the default, and got clamped back.
    """
    alpha = _growth_alpha()
    lc_far = _graded_far_size(lc_wall, tun_radius, 0.0)
    d_sat = _graded_dist_max(lc_wall, lc_far)
    n = 0.0
    steps = 400
    d_prev = 0.0
    for i in range(1, steps + 1):
        # Cubic spacing: the integrand falls off fast and the first millimetre
        # carries as much of the count as the last ten metres.
        d = tun_radius * ((i / steps) ** 3)
        dd = d - d_prev
        dm = 0.5 * (d + d_prev)
        s = lc_far if dm >= d_sat else min(lc_wall + alpha * dm, lc_far)
        area = (2.0 * math.pi * (body_r + dm) * total_L
                + 4.0 * math.pi * (body_r + dm) ** 2)
        n += area * dd / (s ** 3)
        d_prev = d
    return n


def _body_ref_length(rocket: dict) -> float:
    """Reference length the farfield floors are measured in.

    The larger of the body's flow-wise length and its full cross-stream span.
    A rocket is set by its length; a stubby or wide body (a capsule, a finned
    stage whose fins span more than it is long) is set by its span, and using
    the length alone there would place the boundary closer than it looks.
    """
    prof = [p for p in (rocket.get("profile") or []) if len(p) >= 2]
    r_env = max([float(rocket.get("body_radius", 0.0))]
                + [float(p[1]) for p in prof])
    span = 2.0 * (r_env + float(rocket.get("fin_height", 0.0) or 0.0))
    return max(float(rocket.get("length", 0.0)), span, 1e-6)


def _estimate_sizes_from_count(
    target_count: int,
    body_r: float,
    total_L: float,
    tun_radius: float,
    tun_len: float,
) -> tuple[float, float]:
    """
    Estimate (wall_size, far_size) from a target element count.

    Bisects :func:`_count_from_wall_size` -- the same graded field the mesher
    will actually build -- instead of dividing a domain volume by an average
    cell. ``tun_len`` is accepted for call compatibility and unused: the count
    is set by the near field, which the tunnel length does not touch.
    """
    lo, hi = body_r * 0.002, body_r * 0.60
    target = float(max(target_count, 1000))
    # Monotone decreasing in lc_wall, so a plain bisection converges.
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if _count_from_wall_size(mid, body_r, total_L, tun_radius) > target:
            lo = mid
        else:
            hi = mid
    lc_wall = math.sqrt(lo * hi)
    lc_wall = min(max(lc_wall, body_r * 0.005), body_r * 0.5)
    lc_far = _graded_far_size(lc_wall, tun_radius, body_r * 2.0)
    predicted = _count_from_wall_size(lc_wall, body_r, total_L, tun_radius)
    logger.info(
        f"Size from target count {target_count:,}: wall={lc_wall:.5f} m  "
        f"far={lc_far:.4f} m  (predicted {predicted:,.0f} tets)"
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
    bl_prisms: bool = False,
    bl_first_height: float | None = None,
    bl_tessellation: float | None = None,
    bl_max_apex_radius: float | None = None,
    geometry_dict: dict = None,
    custom_wall_size: float | None = None,
    target_element_count: int | None = None,
    external_cad: Path | None = None,
    flow_axis: str = "auto",
    cad_info: dict | None = None,
    cad_units: str = "auto",
    cad_wrap: bool = False,
    cad_wrap_resolution: str = "medium",
    cad_curvature_elements: int | None = None,
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
            cad_curvature_elements=cad_curvature_elements,
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
    # Farfield radius: the caller's own scale, floored at _FARFIELD_REF_LENGTHS
    # reference lengths. max(), never min() -- a caller asking for a wider
    # domain still gets it, and one asking for the old body_r*20 gets the floor
    # that makes the result free-air rather than a wind tunnel with walls.
    _ref_L = _body_ref_length(rocket)
    tun_radius = max(body_r * max(domain_radius_scale, 20.0),
                     _FARFIELD_REF_LENGTHS * _ref_L)
    logger.info(
        f"Farfield radius {tun_radius:.2f} m "
        f"({tun_radius / _ref_L:.1f} reference lengths of {_ref_L:.3f} m)"
    )

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
        if bl_prisms:
            # UNSUPPORTED. One deterministic attempt, and an honest refusal if
            # it does not work. Read _BL_PRISM_STATUS before changing this.
            #
            # There was a five-rung ladder here that retried at 1.0, 1.4, 0.7,
            # 2.0 and 0.5 x lc_rocket until one happened not to crash. It was
            # removed on measurement, not on taste:
            #
            #   * It is a lottery. The workable carrier resolution is not a
            #     property anything computes, so the ladder is a search with no
            #     model behind it and no guarantee for the next geometry.
            #   * What it wins does not solve. On the canonical rocket rung 2
            #     produced a structurally VALID mesh -- 5,317,942 cells,
            #     141,024 prisms, wall manifold, zero inverted -- and SU2 then
            #     stalled on it: rms[Rho] started at -3.559, best -3.791, and
            #     sat at -3.227 after 98 iterations, i.e. it fell nothing at
            #     all, while CD read 0.898 against 0.343 from the supported
            #     path. Seven minutes of retrying buys a mesh that cannot be
            #     solved.
            #
            # Failing in ~30 s with a reason is strictly better than that.
            logger.warning(
                "Prism boundary layer is UNSUPPORTED (see _BL_PRISM_STATUS in "
                "cfd/meshing.py). One attempt; no retries. If it fails, the "
                "supported path for trustworthy drag is Euler + flat-plate "
                "friction on the tet-only mesh."
            )
            try:
                _build_bl_mesh(
                    gmsh, rocket, tun_len, tun_radius,
                    lc_far, lc_rocket, output_path,
                    bl_layers, bl_growth,
                    bl_first_height=bl_first_height,
                    bl_tessellation=bl_tessellation,
                    bl_max_apex_radius=bl_max_apex_radius,
                )
            except Exception as e:                       # noqa: BLE001
                raise RuntimeError(
                    f"Prism boundary-layer meshing failed: {e}\n\n"
                    f"This is a limitation of the mesher, not of these "
                    f"settings. gmsh's extrudeBoundaryLayer has no corner "
                    f"treatment, so where advancing fronts converge -- at a "
                    f"sharp nose tip, a fin root or a thin trailing edge -- the "
                    f"stack self-intersects and the tet pass rejects it. The "
                    f"binding feature is usually the NOSE TIP, not the fins: "
                    f"the opposing-wall distance there was measured at 0.665 mm "
                    f"on a coarse carrier and 0.292 mm on a fine one, against "
                    f"1.528 mm at the fins.\n\n"
                    f"There is deliberately no retry and no silent fallback to "
                    f"the tet-only mesh: a tet-only wall sits at y+ in the "
                    f"hundreds with no wall model, and nothing downstream would "
                    f"say so. Turn the prism layer off and use Euler + "
                    f"flat-plate friction, which is the supported path and "
                    f"reports itself under its own name."
                ) from e
        else:
            _build_mesh(
                gmsh, rocket, tun_len, tun_radius,
                lc_far, lc_rocket, output_path,
                bl_layers, bl_growth,
            )
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass

    su2_path = output_path.with_suffix(".su2")
    size_mb = su2_path.stat().st_size / 1e6
    logger.info(f"SU2 mesh ready: {su2_path}  ({size_mb:.2f} MB)")
    return su2_path


# ── Core meshing logic ────────────────────────────────────────────────────────

def _build_rocket_solid(gmsh, rocket, lc_rocket):
    """Build the fused rocket solid in OCC; return ``(volume_dimtags, profile)``.

    The clamped profile comes back with the solid because callers size their
    refinement regions off the true maximum body radius, which the profile
    carries and ``rocket["body_radius"]`` does not (a boattail or a flare makes
    them differ).

    Shared by the tet-only path and the prism-boundary-layer path so the two
    always mesh the identical outer mold line. Leaves the solid in the model,
    synchronized; the caller decides whether to cut it out of a tunnel or to
    tessellate it as a standalone shell.
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
    profile = _clamp_profile_tip(profile, lc_rocket)
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
    return rocket_solid, profile


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
    rocket_solid, profile = _build_rocket_solid(gmsh, rocket, lc_rocket)

    # ── 2. Wind tunnel domain ─────────────────────────────────────────────────
    # Floored at _UPSTREAM_REF_LENGTHS / _WAKE_REF_LENGTHS reference lengths.
    # min()/max() so this can only ever enlarge the domain a caller asked for.
    ref_L          = _body_ref_length(rocket)
    upstream_x     = min(-5.0 * total_L, -_UPSTREAM_REF_LENGTHS * ref_L)
    downstream_x   = max(total_L + 15.0 * total_L,
                         total_L + _WAKE_REF_LENGTHS * ref_L)
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
    #
    # SizeMax/DistMax are DERIVED from _MAX_GROWTH_RATIO, not chosen. A gmsh
    # Threshold band is linear in wall distance, so the ratio between adjacent
    # cells is exactly 1 + (SizeMax-SizeMin)/(DistMax-DistMin) everywhere in the
    # band; the shipped DistMax of 5*body_r made that ratio 4.0 and crossed the
    # entire wall-to-farfield transition in 1.5 cells. See the module constants.
    #
    # DistMin is 0, not 0.1*body_r: a flat first band is a place where the mesh
    # does not grade at all, which on a body this size is most of the near
    # field. The growth starts at the wall.
    lc_far = _graded_far_size(lc_rocket, tun_radius, lc_far)
    dist_max = _graded_dist_max(lc_rocket, lc_far)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(
        f_dist, "SurfacesList",
        rocket_wall_surfs if rocket_wall_surfs else [s[1] for s in all_surfs[:5]]
    )
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 40)

    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField",  f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin",  lc_rocket)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax",  lc_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax",  dist_max)

    # ── 5b. Nose tip refinement ───────────────────────────────────────────────
    # Every Box below carries a Thickness. Without one a Box field is a step
    # function -- VIn inside the faces, VOut immediately outside -- and since
    # the background field is the minimum over all fields, that step IS a cell
    # size discontinuity wherever the box is finer than the wall band. It is
    # where the measured 47x jump came from.
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
    gmsh.model.mesh.field.setNumber(f_nose, "Thickness",
                                    _box_thickness(lc_nose, lc_far))

    # ── 5c. Fin-region refinement ─────────────────────────────────────────────
    # Centre the box on the actual fin axial station (fins sit at
    # x_TE = total_L - fin_z_base_k2 in the CFD frame; 0 = tail-mounted).
    #
    # The lateral half-width is 1.15 fin spans, not 1.5. At 1.5 this box held a
    # uniform 0.7*lc_wall over a cube reaching well past the fin tips, and on
    # the measured mesh that single field owned 222,945 of 230,946 cells -- 97%
    # of the mesh spent on a block of air the fins do not touch. 1.15 still
    # covers every fin with a margin, and the Thickness below carries the size
    # outward at the growth cap instead of holding it flat and then dropping it.
    fin_x_te = total_L - rocket.get("fin_z_base_k2", 0.0)
    fin_x_te = min(total_L, max(fin_Cr, fin_x_te))
    lc_fin = lc_rocket * 0.7
    f_fin = gmsh.model.mesh.field.add("Box")
    gmsh.model.mesh.field.setNumber(f_fin, "XMin",  fin_x_te - fin_Cr * 1.2)
    gmsh.model.mesh.field.setNumber(f_fin, "XMax",  fin_x_te + body_r)
    gmsh.model.mesh.field.setNumber(f_fin, "YMin", -fin_span * 1.15)
    gmsh.model.mesh.field.setNumber(f_fin, "YMax",  fin_span * 1.15)
    gmsh.model.mesh.field.setNumber(f_fin, "ZMin", -fin_span * 1.15)
    gmsh.model.mesh.field.setNumber(f_fin, "ZMax",  fin_span * 1.15)
    gmsh.model.mesh.field.setNumber(f_fin, "VIn",   lc_fin)
    gmsh.model.mesh.field.setNumber(f_fin, "VOut",  lc_far)
    gmsh.model.mesh.field.setNumber(f_fin, "Thickness",
                                    _box_thickness(lc_fin, lc_far))

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
    gmsh.model.mesh.field.setNumber(f_wake, "Thickness",
                                    _box_thickness(lc_wake, lc_far))

    # ── 5e. Curvature-driven edge refinement ──────────────────────────────────
    # The boxes above are blunt instruments: f_fin asks for 0.7·lc_rocket
    # everywhere inside the fin envelope, which sizes the panel faces but says
    # nothing about the *edge*. A fin leading edge is where the suction peak
    # that carries the lift forms, and on a swept panel it is the one feature
    # whose radius is orders of magnitude below the wall size.
    #
    # Measured on AGARD-B (a delta wing spanning 4 body diameters): with the box
    # fields alone the SU2 lift-curve slope came out 10% low at M=0.5 and 18%
    # low at M=0.8 against the AEDC tunnel data, and the deficit grew when the
    # mesh was refined uniformly — shape converging, circulation not.
    #
    # This path refines off the model's own edge curves rather than off measured
    # STL curvature (which is what the external-CAD path has to do, having no
    # exact geometry). Tried the STL route here first and it is the wrong tool
    # for a B-Rep: on AGARD-B it selected 1,388 of 4,818 surface points — the
    # whole ogive as well as the wing — and seeding those into the OCC model left
    # gmsh still inside generate(2) after 20 minutes. The wall curves are exactly
    # the sharp features, cost nothing to enumerate, and gmsh samples them
    # directly.
    f_feature = _edge_refinement_field(gmsh, rocket_wall_surfs, lc_rocket, lc_far)

    # Combine: take minimum size from all fields
    _fields = [f_thr, f_nose, f_fin, f_wake]
    if f_feature is not None:
        _fields.append(f_feature)
    f_min = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", _fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

    logger.info(
        f"Mesh fields: wall={lc_rocket:.4f}  nose={lc_nose:.4f}  "
        f"fin={lc_fin:.4f}  wake={lc_wake:.4f}  far={lc_far:.4f}"
        + ("" if f_feature is None else "  + curvature edge refinement")
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
    # Ceiling as well as floor. gmsh defaults MeshSizeMax to 1e22, so before
    # this the only thing bounding a cell was whichever field happened to be
    # smallest -- and the domain is now large enough that "whichever field"
    # is not a bound anyone should rely on.
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_far)
    logger.info(f"Surface size floor: {_lc_min:.3e} m (5% of lc_rocket={lc_rocket:.5f})"
                f"  ceiling: {lc_far:.4g} m")

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
    # NOT used as primary. It stays as the first robustness rung: if Delaunay
    # aborts on thin fin-TE / nose-tip slivers, retry with a coarser size floor
    # so a valid mesh still comes out.
    #
    # "Try Delaunay, else HXT" was not enough, and the way it failed is the
    # reason this is a verified ladder now: on the sharp 10° validation cone
    # Delaunay raised "Invalid boundary mesh (overlapping facets) on surface 2",
    # HXT then *completed without raising and produced zero elements*, and a
    # 40-byte .su2 was written and announced as ready. The failure only surfaced
    # inside SU2 as "0 grid points ... doesn't have any definition for marker".
    #
    # Each rung changes the thing that actually matters: first the size floor,
    # then the 2D algorithm (MeshAdapt is far more forgiving of self-overlapping
    # facets than Frontal-Delaunay), then the element size itself. Every rung is
    # verified by element count, because gmsh reports success on an empty mesh.
    # Rung 0 reuses the 2D mesh generated in step 6.
    ladder = [
        ("Delaunay 3D",             None, 1,  1.00, 0.05),
        ("HXT + coarser floor",     6,    10, 1.00, 0.15),
        ("MeshAdapt 2D + Delaunay", 1,    1,  1.00, 0.05),
        ("MeshAdapt 2D + HXT",      1,    10, 1.00, 0.15),
        ("MeshAdapt 2D, finer",     1,    1,  0.25, 0.05),
    ]

    def _n_tets() -> int:
        try:
            _t, _g, _ = gmsh.model.mesh.getElements(3)
            return sum(len(g) for g in _g)
        except Exception:                                # noqa: BLE001
            return 0

    def _optimize():
        """Smooth and untangle distorted elements. Never fatal on its own."""
        for _label, _opt in (("Gmsh default", ""), ("Netgen", "Netgen")):
            try:
                gmsh.model.mesh.optimize(_opt, force=True)
                logger.info(f"Mesh optimization ({_label}) complete")
            except Exception as e:                       # noqa: BLE001, PERF203
                logger.warning(f"Mesh optimization ({_label}) failed: {e}")

    meshed = False
    problems: list = []
    for i, (label, a2d, a3d, scale, floor_frac) in enumerate(ladder):
        try:
            if i > 0:
                gmsh.model.mesh.clear()
                gmsh.option.setNumber("Mesh.Algorithm", a2d)
            gmsh.option.setNumber("Mesh.Algorithm3D", a3d)
            _floor = lc_rocket * scale * floor_frac
            gmsh.option.setNumber("Mesh.MeshSizeMin", _floor)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", _floor)
            if scale != 1.0:
                # DistMax has to move with SizeMin or the growth ratio changes:
                # the band is linear, so the neighbour ratio is
                # 1 + (SizeMax-SizeMin)/DistMax and shrinking only SizeMin
                # steepens it. Both come from the same helper for that reason.
                _sm = lc_rocket * scale
                gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", _sm)
                gmsh.model.mesh.field.setNumber(
                    f_thr, "DistMax", _graded_dist_max(_sm, lc_far))
                gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
            if i > 0:
                gmsh.model.mesh.generate(2)
            gmsh.model.mesh.generate(3)
            n = _n_tets()
            if n > 0:
                # Optimise and audit INSIDE the rung, so "this strategy worked"
                # means the mesh it produced is usable — not merely non-empty.
                # Both passes used to run once after the loop and the audit only
                # logged, so a rung that produced an inverted or non-manifold
                # mesh ended the search and that mesh was exported.
                _optimize()
                problems = _check_mesh_quality(
                    gmsh, body_r, n_prisms, wall_surfs=rocket_wall_surfs)
                if not problems:
                    logger.info(f"3D mesh built with {label}: {n:,} elements"
                                + ("" if i == 0 else f" (attempt {i + 1})"))
                    meshed = True
                    break
                logger.warning(
                    f"{label} produced {n:,} elements but the mesh is invalid "
                    f"({'; '.join(problems)}) — trying the next strategy."
                )
                continue
            _t2, _g2, _ = gmsh.model.mesh.getElements(2)
            logger.warning(f"{label} completed but produced no elements — "
                           f"trying the next strategy. "
                           f"(2D elements present: {sum(len(g) for g in _g2):,})")
        except Exception as e:                           # noqa: BLE001, PERF203
            logger.warning(f"{label} failed: {e}")

    if not meshed and problems:
        # Every strategy produced elements and every one of them was invalid.
        # Refusing is the only honest outcome: SU2 loads such a mesh, reports
        # healthy metrics and returns numbers that look like an answer.
        raise RuntimeError(
            "Mesh generation produced an INVALID mesh at every refinement "
            "strategy: " + "; ".join(problems) + ". Do not trust any result "
            "from this geometry until it meshes cleanly — try a coarser "
            "refinement level, or check the body for self-intersections and "
            "features thinner than the element size."
        )
    if not meshed:
        logger.error("Every meshing strategy failed to produce elements.")

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

    # An empty volume mesh is a FAILURE, not a result — see the ladder above.
    # Writing it produced a 0.00 MB .su2 that SU2 loaded as "0 grid points" and
    # then rejected with an unrelated-looking marker error.
    if not meshed:
        raise RuntimeError(
            "3D meshing produced no elements after every strategy. The fluid "
            "volume is not closed — the body surface self-intersects, or a "
            "feature (fin trailing edge, nose tip) is too thin to mesh at this "
            "refinement. Try a coarser refinement level."
        )

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


# ── Prism boundary-layer meshing ──────────────────────────────────────────────

# ── Prism boundary layer: measured status ────────────────────────────────────
#
# UNSUPPORTED as of 2026-08-30. It is reachable (CFDConfig.bl_prisms, and a
# checkbox in the CFD workspace) because it works on some geometries, but it is
# not a capability you can rely on, and the reason is not a missing setting.
#
# What works: on the canonical rocket the mesh BUILDS and is structurally
# sound -- 5,317,942 cells of which 141,024 are prisms, wall manifold, zero
# inverted elements, neighbour size ratio p99 1.39.
#
# What does not: SU2 will not solve it. SST on that mesh went
#     rms[Rho]  start -3.559   best -3.791   iter 98 -3.227
# i.e. zero decades of convergence, with CD at 0.898 against 0.343 from the
# supported Euler + flat-plate-friction path. Suspects are the 5,062
# near-degenerate cells (SICN < 0.01) and a first-layer aspect ratio near
# 1400:1 against numerics tuned for isotropic tets. Not yet diagnosed.
#
# Why a global stack cap cannot fix the meshing half, measured rather than
# argued. The correct general collision criterion is the OPPOSING-WALL
# DISTANCE: for wall node p marching along +n_p, the nearest wall q that lies
# ahead of p and faces back at it. Two fronts then meet at |q-p|/2. (Plain
# nearest-non-neighbour LFS is wrong here -- the 2*pi revolve seam puts
# coincident, topologically distant nodes together and it reports 0.02 mm on a
# smooth cylinder.) On the canonical rocket:
#
#     carrier 14.28 mm : min 0.665 mm, at the NOSE TIP; fins 1.528 mm
#     carrier  5.00 mm : min 0.292 mm, at the NOSE TIP
#
# Two consequences. The binding feature is the tip, not the fins -- so a
# fin-thickness cap was never measuring the right thing. And the value HALVES
# when the carrier is refined, because the tip is a singularity where the true
# distance goes to zero: a cap derived from it is a function of tessellation,
# not of geometry. That is why the code below caps on fins and blunts the apex
# instead; it is a workaround, but a rational one given that
# extrudeBoundaryLayer accepts only ONE global height list.
#
# The permanent fix is per-node layer termination -- thin the stack only where
# it must thin, from the opposing-wall distance at that node -- which is what
# every production BL mesher does.
#
# Whether gmsh can express that is OPEN, not settled. extrudeBoundaryLayer takes
# a trailing `viewIndex`, and a scalar view is documented to scale the extrusion
# normals, which is exactly the per-node thickness field this needs. Two earlier
# probes reported "no effect" and BOTH were broken (one never called generate(3)
# so no layer was ever built; the other reparametrised its test sphere down to
# ~18 triangles and built 144 prisms). So the mechanism is untested, not
# disproven -- try it before writing an extruder of our own. Either way the
# solver half is separate work: a valid prism mesh here still stalls.
#
# Until then the supported path for trustworthy drag is Euler + flat-plate
# friction on the tet-only mesh, which reports itself by name and carries a
# wall_resolved verdict on every result.
_BL_PRISM_STATUS = "unsupported: mesh builds, solver stalls (2026-08-30)"


# 25 degrees, NOT the 40 that cfd/external_geometry.py uses for CAD import.
#
# Measured on the parametric rocket: at 40 deg, classifySurfaces produces one
# patch carrying the whole aft body AND all four fin roots. Remeshing that
# single patch takes over 417 SECONDS (every other patch finishes in under 1.5)
# and it is what made the BL path look like a hang. At 25 deg the same shell
# splits into 65 patches instead of 39 and the entire 2D remesh takes 1.2 s.
# The curve angle makes no measurable difference either way, so it stays at the
# don't-split-curves default.
_BL_CLASSIFY_ANGLE_DEG = 25.0
_BL_CLASSIFY_CURVE_ANGLE_DEG = 180.0


def _bl_heights(first: float, growth: float, n: int) -> list:
    """Cumulative offsets for ``n`` geometrically growing layers."""
    h, cum, out = float(first), 0.0, []
    for _ in range(int(n)):
        cum += h
        out.append(cum)
        h *= float(growth)
    return out


def _bl_thickness_cap(rocket, profile) -> float:
    """Largest total stack thickness this geometry can carry, or inf.

    Gmsh's boundary layer has no convex-corner treatment: where two extrusion
    fronts advance toward each other they simply pass through one another, and
    the failure surfaces later as "PLC Error: A segment and a facet intersect"
    with nothing pointing at the cause.

    Two places on a rocket close on themselves:

    The one this caps is the FIN: both faces extrude toward the mid-plane, so
    each side may use less than half the thickness. Measured - a 2.0 mm fin with
    a 1.05 mm stack per side fails exactly here. The 0.4 margin leaves room for
    the stack to be non-uniform where fronts merge; it is not derived.

    The nose apex is the other such place and is deliberately NOT capped here.
    Capping by the apex throttles the stack over the entire body to suit a few
    square millimetres, and it does not even work: inversions at the apex are not
    monotonic in the cap (see _bl_apex_clamp). The apex is handled by blunting it
    just enough instead.
    """
    limits = []
    # "fin_thick", not "fin_thickness". The geometry dict is built by
    # cfd.geometry_exporter.extract_cfd_geometry and every other reader in the
    # codebase uses "fin_thick"; this one asked for a key that has never
    # existed, so .get() returned None, fin_t came out 0.0, `limits` stayed
    # empty and the cap was inf. The one guard against the exact failure it
    # documents was switched off by the name.
    #
    # Measured on the canonical rocket (4.0 mm fins, so a 0.8 mm cap): the
    # stack ran at 1.029 mm and every tessellation in the ladder died on
    # "PLC Error: A segment and a facet intersect".
    fin_t = float(rocket.get("fin_thick") or rocket.get("fin_thickness") or 0.0)
    if fin_t > 0.0 and int(rocket.get("fin_count") or 0) > 0:
        limits.append(0.4 * fin_t / 2.0)
    return min(limits) if limits else float("inf")


# Apex radius needed per unit of stack thickness. Measured by sweeping the apex
# against a fixed 1.736e-4 m stack and counting inverted prisms:
#     apex 0.37 mm (ratio 0.47) -> 20 inverted
#     apex 0.80 mm (ratio 0.22) ->  0
#     apex 1.50 mm (ratio 0.12) ->  5      <- NOT monotonic
#     apex 3.00 mm (ratio 0.06) ->  0
#     apex 6.00 mm (ratio 0.03) ->  0
# So no ratio is provably safe and this is a heuristic that happens to clear the
# measured cases; _count_inverted is the actual guarantee. 5 is the smallest
# multiple that worked, chosen to keep the geometric change small.
_BL_APEX_RADIUS_PER_THICKNESS = 5.0


def _bl_apex_clamp(rocket: dict, total_thickness: float) -> dict:
    """Return a copy of ``rocket`` whose nose apex is blunt enough to wrap.

    A stack advancing off a cone tip converges circumferentially, and once the
    tip is finer than the stack the innermost prisms turn inside out. Widening
    the apex is a REAL geometric change to the body being validated - it is
    logged, and kept as small as the measurements allow - but it is a smaller
    lie than a boundary layer with inverted cells in it, and the tet path
    already blunts the same apex (``_clamp_profile_tip``) for its own reasons.
    """
    profile = rocket.get("profile") or []
    if not profile:
        return rocket
    need = _BL_APEX_RADIUS_PER_THICKNESS * total_thickness
    apex_r = float(profile[0][1])
    if apex_r >= need:
        return rocket
    out = dict(rocket)
    prof = [list(p) for p in profile]
    prof[0][1] = need
    out["profile"] = [tuple(p) for p in prof]
    logger.warning(
        f"Prism BL blunted the nose apex {apex_r:.3e} m -> {need:.3e} m "
        f"({_BL_APEX_RADIUS_PER_THICKNESS:.0f}x the {total_thickness:.3e} m "
        f"stack). A tip finer than the stack inverts the innermost prisms. "
        f"THIS CHANGES THE GEOMETRY BEING SOLVED - the body is no longer the one "
        f"that was designed. Measured on the parametric test rocket at fixed "
        f"stack and flow, blunting 0.80 -> 3.37 mm moved CD by +5.17%, so the "
        f"error scales with how much tip is removed. Pass bl_max_apex_radius to "
        f"bound it (costs boundary-layer coverage), and do not trust this mesh "
        f"for a sharp-tipped validation case such as a Taylor-Maccoll cone."
    )
    return out


def _count_inverted(gmsh) -> int:
    """Number of 3D elements with a negative scaled Jacobian (SICN < 0)."""
    import numpy as np

    n = 0
    try:
        for _, vol in gmsh.model.getEntities(3):
            types, tags, _ = gmsh.model.mesh.getElements(3, vol)
            for _et, tg in zip(types, tags):
                if not len(tg):
                    continue
                q = np.asarray(
                    gmsh.model.mesh.getElementQualities(list(tg), "minSICN")
                )
                n += int((q < 0).sum())
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"Inverted-element count could not run: {e}")
        return 0
    return n


def _audit_bl_direction(gmsh, wall_surfs, top_surfs) -> Optional[bool]:
    """True when the stack grew into the fluid, False when it grew into the body.

    Measures the volume each closed shell encloses (divergence theorem over its
    triangles). The BL top must enclose MORE than the wall it grew from.

    This is not a formality. Every topology check — element counts, wall
    manifoldness, Gmsh's own quality metrics — passes identically for a stack
    extruded the wrong way, because an inward stack is still a perfectly valid
    mesh of the wrong region. Only the geometry tells them apart.
    """
    import numpy as np

    try:
        ntags, ncoord, _ = gmsh.model.mesh.getNodes()
        pos = {int(t): ncoord[3 * i:3 * i + 3] for i, t in enumerate(ntags)}

        def enclosed(surfs):
            v = 0.0
            for s in surfs:
                types, _, nodes = gmsh.model.mesh.getElements(2, s)
                for et, nd in zip(types, nodes):
                    _, _, _, nn, _, _ = gmsh.model.mesh.getElementProperties(et)
                    if nn != 3:
                        continue
                    tri = np.asarray(nd, dtype=np.int64).reshape(-1, 3)
                    a = np.array([pos[int(i)] for i in tri[:, 0]])
                    b = np.array([pos[int(i)] for i in tri[:, 1]])
                    c = np.array([pos[int(i)] for i in tri[:, 2]])
                    v += float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))))
            return abs(v) / 6.0

        v_wall, v_top = enclosed(wall_surfs), enclosed(top_surfs)
        logger.info(
            f"BL direction: wall encloses {v_wall:.6g} m^3, stack top encloses "
            f"{v_top:.6g} m^3"
        )
        return v_top > v_wall
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"BL direction audit could not run: {e}")
        return None


def _build_bl_mesh(
    gmsh, rocket, tun_len, tun_radius, lc_far, lc_rocket, output_path,
    bl_layers, bl_growth, bl_first_height=None, bl_tessellation=None,
    bl_max_apex_radius=None,
):
    """Mesh the wind tunnel with real prism boundary layers.

    Spiked and measured 2026-08-19. The route is narrow and every step in it is
    load-bearing; the module docstring records what was tried and rejected.

    1. Build the rocket in OCC and tessellate it to an STL. The STL is a
       TOPOLOGY CARRIER — its resolution is not the wall resolution, which comes
       from the size field in step 6. It only has to be coarse enough for
       ``classifySurfaces`` to parametrise and fine enough to keep the shape.
    2. Re-import it and re-topologise with ``forReparametrization=True``.
       Extruding a boundary layer needs analytic patches: raw discrete surfaces
       AND exact OCC surfaces both fail with "Could not find extruded node".
    3. Extrude the stack off the reparametrised wall.
    4. ONLY THEN build the farfield box. Built first, its entities make
       ``extrudeBoundaryLayer`` fail "Could not replace surface N in Coherence"
       on some tessellations and pass on others.
    5. Outer volume = box loop with the BL-top loop as a HOLE. The tets are
       therefore structurally unable to refill the boundary layer — which is the
       failure the previous attempt shipped, undetected, for months.
    """
    import tempfile

    total_L = rocket["length"]
    body_r = rocket["body_radius"]

    tess = bl_tessellation or lc_rocket
    first_h = bl_first_height
    if first_h is None or first_h <= 0.0:
        # No flow state here to target a y+ with. Callers that want a resolved
        # wall must pass bl_first_height, normally
        # cfd.boundary_layer.predict_wall_yplus(...)["spacing_for_yplus_1"].
        first_h = lc_rocket * 1e-3
        logger.warning(
            f"No bl_first_height given — defaulting to {first_h:.3e} m "
            f"(lc_rocket/1000). This is a geometric guess, NOT a y+ target; "
            f"pass spacing_for_yplus_1 from predict_wall_yplus() to resolve the "
            f"wall on purpose."
        )

    # ── 1. Size the stack, then build a solid that can carry it ───────────────
    #
    # Order matters: how blunt the nose has to be depends on the final stack
    # thickness, so the layer budget is settled BEFORE the geometry is built.
    #
    # Trim the stack to what the thinnest feature can carry. Dropping whole
    # layers keeps the requested first-layer height — which is what sets y+ and
    # is the entire point — and gives up only the outer, coarsest layers.
    n_layers = int(bl_layers)
    heights = _bl_heights(first_h, bl_growth, n_layers)
    cap = _bl_thickness_cap(rocket, rocket.get("profile") or [])
    if heights[-1] > cap:
        kept = [h for h in heights if h <= cap]
        if not kept:
            raise RuntimeError(
                f"Prism BL cannot fit this geometry: even one layer of "
                f"{first_h:.3e} m exceeds the {cap:.3e} m the thinnest feature "
                f"can carry (0.4 x fin half-thickness). Reduce "
                f"bl_first_height, or thicken the fins."
            )
        logger.warning(
            f"Prism BL trimmed {n_layers} layers -> {len(kept)}: the full stack "
            f"({heights[-1]:.3e} m) exceeds the {cap:.3e} m this geometry can "
            f"carry before extrusion fronts collide at the fin mid-plane. "
            f"First-layer height is unchanged, so the y+ target still "
            f"holds; the stack just spans less of the boundary layer."
        )
        heights, n_layers = kept, len(kept)

    logger.info(
        f"Prism BL: {n_layers} layers, first={first_h:.3e} m, growth={bl_growth}, "
        f"total={heights[-1]:.4e} m (fin cap {cap:.3e} m); "
        f"STL carrier at {tess:.4g} m"
    )

    # Bound the geometric damage, if the caller asked for a bound.
    #
    # The apex has to be blunted to about 5x the stack thickness, so the stack
    # is what sets how much of the nose is thrown away, and the two trade
    # directly. Measured on the parametric rocket at fixed stack and flow:
    # blunting 0.80 -> 3.37 mm moved CD by +5.17%. A sharp-nosed validation case
    # wants that bound tight and will pay for it in boundary-layer coverage; a
    # real airframe usually does not care. Trimming layers is the currency.
    if bl_max_apex_radius and bl_max_apex_radius > 0:
        while (len(heights) > 1
               and _BL_APEX_RADIUS_PER_THICKNESS * heights[-1] > bl_max_apex_radius):
            heights = heights[:-1]
        n_layers = len(heights)
        need = _BL_APEX_RADIUS_PER_THICKNESS * heights[-1]
        if need > bl_max_apex_radius:
            raise RuntimeError(
                f"Prism BL cannot honour bl_max_apex_radius={bl_max_apex_radius:.3e} m: "
                f"even a single {first_h:.3e} m layer needs the apex blunted to "
                f"{need:.3e} m. Lower bl_first_height or raise the bound."
            )
        logger.info(
            f"Prism BL held to bl_max_apex_radius={bl_max_apex_radius:.3e} m: "
            f"{n_layers} layers, stack {heights[-1]:.3e} m, apex {need:.3e} m"
        )

    rocket = _bl_apex_clamp(rocket, heights[-1])
    rocket_solid, profile = _build_rocket_solid(gmsh, rocket, lc_rocket)

    # Export the solid's OWN boundary, not every surface in the model.
    #
    # Revolving a meridian through 2*pi leaves seam faces inside the OCC solid.
    # They are interior, so they bound nothing — but they are still 2D entities,
    # and tessellating them into the carrier makes it non-manifold: the STL
    # comes back with an edge incident to three triangles and classifySurfaces
    # rejects it ("Wrong topology of triangulation for parametrization"). The
    # tet-only path dodges this by taking the FLUID volume's boundary after the
    # boolean cut (step 4 there); there is no cut here, so take the solid's.
    _bnd = gmsh.model.getBoundary(
        [(3, v[1]) for v in rocket_solid],
        combined=True, oriented=False, recursive=False,
    )
    shell_surfs = [abs(tag) for dim, tag in _bnd if dim == 2]
    if not shell_surfs:
        raise RuntimeError("Prism BL meshing failed: rocket solid has no boundary.")
    gmsh.model.addPhysicalGroup(2, shell_surfs, name="body_shell")
    logger.info(f"STL carrier: {len(shell_surfs)} outer surface(s) of the solid")

    gmsh.option.setNumber("Mesh.MeshSizeMin", tess * 0.3)
    gmsh.option.setNumber("Mesh.MeshSizeMax", tess)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
    gmsh.model.mesh.generate(2)
    stl_dir = tempfile.mkdtemp(prefix="k2_bl_")
    stl_path = str(Path(stl_dir) / "bl_carrier.stl")
    gmsh.option.setNumber("Mesh.SaveAll", 0)   # physical groups only
    gmsh.write(stl_path)

    # ── 2. Re-import and reparametrise ────────────────────────────────────────
    gmsh.model.remove()
    gmsh.model.add("K2_CFD_BL")
    gmsh.merge(stl_path)
    gmsh.model.mesh.removeDuplicateNodes()
    ang = math.radians(_BL_CLASSIFY_ANGLE_DEG)
    cang = math.radians(_BL_CLASSIFY_CURVE_ANGLE_DEG)
    try:
        gmsh.model.mesh.classifySurfaces(ang, True, True, cang)
        gmsh.model.mesh.createGeometry()
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(
            f"Prism BL meshing failed: could not build a parametrisation for the "
            f"wall ({e}). The STL carrier at {tess:.4g} m is probably too dense — "
            f"classifySurfaces throws 'Wrong topology of boundary mesh for "
            f"parametrization' on fine tessellations. Retry with a coarser "
            f"bl_tessellation, or use the tet-only mesher. There is deliberately "
            f"no silent fallback: a tet-only mesh leaves the wall at y+ in the "
            f"thousands and nothing downstream would say so."
        ) from e
    gmsh.model.geo.synchronize()
    wall_surfs = [s[1] for s in gmsh.model.getEntities(2)]
    if not wall_surfs:
        raise RuntimeError("Prism BL meshing failed: no wall surfaces recovered.")
    logger.info(f"Wall reparametrised into {len(wall_surfs)} patch(es)")

    # ── 3. Extrude the stack ──────────────────────────────────────────────────
    try:
        ext = gmsh.model.geo.extrudeBoundaryLayer(
            [(2, s) for s in wall_surfs], [1] * n_layers, heights, True
        )
    except Exception as e:                                   # noqa: BLE001
        raise RuntimeError(
            f"Prism BL extrusion failed: {e}. A 'PLC Error: A segment and a facet "
            f"intersect' here is the stack self-intersecting where extrusion "
            f"fronts converge — fin trailing edges and tips. Reduce bl_layers or "
            f"bl_growth so the stack is thinner than the local feature."
        ) from e
    tops = [ext[i - 1][1] for i in range(1, len(ext)) if ext[i][0] == 3]
    bl_vols = [d[1] for d in ext if d[0] == 3]
    gmsh.model.geo.synchronize()

    # ── 4. Farfield box — AFTER the extrusion, see the docstring ──────────────
    upstream_x = -5.0 * total_L
    downstream_x = total_L + 15.0 * total_L
    box_faces, box_loop = _geo_box(
        gmsh, upstream_x, -tun_radius, -tun_radius,
        downstream_x, tun_radius, tun_radius,
    )

    # ── 5. Outer volume = box with the stack as a hole ────────────────────────
    top_loop = gmsh.model.geo.addSurfaceLoop(tops)
    outer_vol = gmsh.model.geo.addVolume([box_loop, top_loop])
    gmsh.model.geo.synchronize()

    # ── 6. Size field — measured from the WALL, not the stack top ─────────────
    # The tops carry no mesh when the surfaces are meshed, so a Distance field
    # on them evaluates to nothing and the wall comes out several times coarser
    # than lc_rocket without any warning.
    f_d = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_d, "SurfacesList", wall_surfs)
    gmsh.model.mesh.field.setNumber(f_d, "Sampling", 200)
    f_t = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_t, "InField", f_d)
    gmsh.model.mesh.field.setNumber(f_t, "SizeMin", lc_rocket)
    gmsh.model.mesh.field.setNumber(f_t, "SizeMax", lc_far)
    gmsh.model.mesh.field.setNumber(f_t, "DistMin", body_r)
    gmsh.model.mesh.field.setNumber(f_t, "DistMax", body_r * 10.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_t)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    # Undo the carrier's size clamps from step 1. Gmsh options are global and
    # survive model.remove(), so the tessellation ceiling (Mesh.MeshSizeMax =
    # tess) would otherwise cap the ENTIRE wind tunnel at the carrier's element
    # size — a 21 m domain at 1 cm is order 1e8 cells, which reads as a hang
    # rather than an error. Same class as the Threshold SizeMax trap in
    # _curvature_feature_field.
    gmsh.option.setNumber("Mesh.MeshSizeMin", lc_rocket * 0.05)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
    logger.info(
        f"Volume size clamps reset: min={lc_rocket * 0.05:.4g} m  max={lc_far:.4g} m"
    )

    # ── 7. Volume mesh ────────────────────────────────────────────────────────
    gmsh.model.mesh.generate(3)

    # ── 8. Markers ────────────────────────────────────────────────────────────
    gmsh.model.addPhysicalGroup(2, wall_surfs, name="rocket_wall")
    gmsh.model.addPhysicalGroup(2, box_faces, name="farfield")
    gmsh.model.addPhysicalGroup(3, bl_vols + [outer_vol], name="fluid")

    # ── 9. Audits ─────────────────────────────────────────────────────────────
    n_prisms = 0
    try:
        types, tags, _ = gmsh.model.mesh.getElements(3)
        for t, tg in zip(types, tags):
            _, _, _, nn, _, _ = gmsh.model.mesh.getElementProperties(t)
            if nn == 6:
                n_prisms += len(tg)
    except Exception:                                        # noqa: BLE001
        pass
    if n_prisms == 0:
        raise RuntimeError(
            "Prism BL meshing produced no prisms — the extrusion silently "
            "collapsed. Refusing to write a mesh that claims a resolved wall."
        )

    outward = _audit_bl_direction(gmsh, wall_surfs, tops)
    if outward is False:
        raise RuntimeError(
            "Prism BL grew INTO the body, not into the fluid: every prism sits "
            "inside the solid and the fluid region is short by the stack's "
            "thickness. Flip the sign of the layer heights."
        )

    _bl_problems = _check_mesh_quality(gmsh, body_r, n_prisms,
                                       wall_surfs=wall_surfs)
    if _bl_problems:
        raise RuntimeError(
            "Prism BL mesh is invalid: " + "; ".join(_bl_problems) + ". This "
            "is the overlap failure the module docstring describes - the tets "
            "filled the boundary-layer region as well as the prisms - and SU2 "
            "will accept the file and then fail to converge on it."
        )

    # Inverted cells are a hard stop on this path, not a warning.
    #
    # _check_mesh_quality only logs them, which is right for the tet path where
    # a stray sliver is survivable. Here they are concentrated (measured: all of
    # them prisms within 2 mm of the nose apex) and they sit in the layer whose
    # whole purpose is to carry the wall shear. Worse, the count is NOT monotonic
    # in how blunt the apex is - 0.37 mm gave 20, 0.80 mm gave 0, 1.50 mm gave 5,
    # 3.00 mm gave 0 - so no clamp ratio can be trusted to prevent them and the
    # mesh has to be checked rather than argued about.
    n_negative = _count_inverted(gmsh)
    if n_negative:
        raise RuntimeError(
            f"Prism BL mesh has {n_negative} inverted element(s) (negative "
            f"Jacobian), concentrated at the nose apex where the stack wraps a "
            f"tip finer than itself. SU2 may silently re-orient some and still "
            f"produce wrong wall shear there, so this is refused rather than "
            f"written. Blunt the nose apex, or lower bl_first_height/bl_layers "
            f"so the stack is thinner than the tip it has to wrap."
        )

    # ── 10. Export ────────────────────────────────────────────────────────────
    su2_path = output_path.with_suffix(".su2")
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write(str(su2_path))
    logger.info(f"Prism BL mesh written: {su2_path}")


def _curvature_feature_field(gmsh, surface_stl, lc_wall: float, lc_far: float,
                             is_brep: bool, max_points: int = 400):
    """Refine where the body is sharply curved. Returns a field tag, or None.

    Measures the principal curvature of the imported tessellation, keeps the
    points whose radius of curvature is finer than the wall element about to be
    used — those are exactly the features the mesh would otherwise flatten — and
    asks for elements of about half that radius near them.

    Why not gmsh's ``Mesh.MeshSizeFromCurvature``: that reads a surface
    parametrisation. An exact B-Rep has one; a re-topologised STL has only an
    approximate one and the option measurably does nothing there.

    Returns None when the body has no such features (a cylinder, a box) or when
    the curvature computation is unavailable — in both cases the caller's other
    fields already describe the mesh.
    """
    import numpy as np

    try:
        import pyvista as pv
        import vtk

        surf = pv.read(str(surface_stl)).extract_surface().triangulate().clean()
        if surf.n_points == 0:
            return None
        surf = surf.compute_normals(point_normals=True, cell_normals=False,
                                    auto_orient_normals=True)
        # VTK warns per point on degenerate triangles — a knife-edge trailing
        # edge produces thousands of them and buries the mesh log. The affected
        # points report an over-large curvature, which this function treats as a
        # sharp feature; that is the right answer for a knife edge anyway.
        vtk.vtkObject.GlobalWarningDisplayOff()
        try:
            k = np.abs(np.asarray(surf.curvature("maximum")))
        finally:
            vtk.vtkObject.GlobalWarningDisplayOn()
    except Exception as e:                                   # noqa: BLE001
        logger.debug(f"Curvature feature field unavailable: {e}")
        return None

    with np.errstate(divide="ignore"):
        radius = np.where(k > 1e-9, 1.0 / np.maximum(k, 1e-9), np.inf)

    # A feature is anything the wall element cannot wrap around: it takes several
    # elements to represent a curve, so the test is against a small multiple of
    # the element size, not against the element itself. (Testing r < lc/2 would
    # call a 7 mm leading edge "resolved" by a 10 mm element — one facet across
    # the whole nose, which is the failure this exists to fix.)
    sel = radius < 2.0 * lc_wall
    n_sel = int(sel.sum())
    if n_sel < 8:
        return None

    pts = np.asarray(surf.points)[sel]
    radii = radius[sel]
    try:
        normals = np.asarray(surf.point_data["Normals"])[sel]
    except Exception:                                        # noqa: BLE001
        normals = None

    # Thin to a few hundred seeds — a Distance field is evaluated against every
    # one of them at every candidate mesh point, and the leading edge of a wing
    # can easily contribute tens of thousands of near-identical points. Strided
    # in surface order, which walks the geometry section by section: taking the
    # sharpest instead would spend every seed on the trailing edge and leave the
    # leading edge unrefined.
    if len(pts) > max_points:
        step = len(pts) // max_points + 1
        pts, radii = pts[::step][:max_points], radii[::step][:max_points]
        if normals is not None:
            normals = normals[::step][:max_points]

    # Element size at the feature: about a third of the radius, so roughly three
    # elements wrap the curve instead of one facet chording across it. Driven by
    # the sharper quartile rather than the median — the selection also catches
    # the mildly-curved skin either side of an edge, and sizing off the middle of
    # that population leaves the edge itself under-resolved.
    # Floored at a tenth of the wall size: a knife-edge trailing edge has a
    # radius near zero and would otherwise ask for an unmeshable element.
    r_feature = float(np.percentile(radii, 25))
    target = min(max(r_feature / 3.0, lc_wall * 0.1), lc_wall * 0.9)

    # Lift the seeds off the wall into the fluid. A free point sitting exactly on
    # the shell hangs the volume mesher — measured: gmsh ran past a 9-minute
    # timeout with the seeds on the surface and meshes in ~2 minutes with them
    # offset. The field only needs to be near the edge, not on it.
    #
    # DistMin below MUST then cover the offset, and used to not. With seeds
    # 1.5*target off the wall and DistMin=target, a point ON the wall sat at
    # d=1.5*target — already past the flat band — so the field returned
    # target + (lc_far-target)/6 there. With a farfield size of order 1 m that
    # is ~170 mm against a wall size of ~10 mm, and since the background field
    # is the MINIMUM over all fields, this one never won at the surface. The
    # refinement existed only in a shell hanging just off the skin, which is
    # not where the leading edge is.
    offset = 0.0
    if normals is not None and len(normals) == len(pts):
        offset = 1.5 * target
        pts = pts + normals * offset
    kernel = gmsh.model.occ if is_brep else gmsh.model.geo
    tags = []
    for x, y, z in pts:
        try:
            tags.append(kernel.addPoint(float(x), float(y), float(z), target))
        except Exception:                          # noqa: BLE001, PERF203
            pass
    if not tags:
        return None
    kernel.synchronize()

    f_pts = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_pts, "PointsList", tags)

    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_pts)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", target)
    # SizeMax is the FARFIELD size: this is what the field returns outside the
    # band, and the background field is the minimum over all fields, so a
    # wall-sized ceiling here caps the entire domain instead of just the edge.
    # Measured on the equivalent rocket-path field: it put 1.78M of 1.8M surface
    # triangles on the six farfield faces.
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", lc_far)
    # The band grows back to the wall size over a few feature widths, NOT over a
    # multiple of the wall size. Tied to lc_wall it reached 60 mm either side of
    # every edge on the M6 wing and pushed the mesh past a million cells for
    # refinement nobody asked for; the suction peak lives within a chord percent
    # or two of the edge.
    #
    # DistMin carries the seed offset so the flat SizeMin band reaches the wall
    # itself — see the offset comment above.
    dist_min = offset + target
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", dist_min + target * 4.0)

    logger.info(
        f"Curvature feature refinement: {n_sel:,} of {surf.n_points:,} surface "
        f"points are curved tighter than the wall size ({lc_wall:.4g} m); "
        f"seeding {len(tags)} of them at {target:.4g} m "
        f"(feature radius p25 {r_feature:.4g} m, median "
        f"{float(np.median(radii)):.4g} m)."
    )
    return f_thr


def _surface_mesh_resolution(stl_path) -> Optional[float]:
    """Median triangle edge length of a tessellation, or None."""
    try:
        import numpy as np
        import pyvista as pv
        m = pv.read(str(stl_path)).extract_surface().triangulate()
        f = m.faces.reshape(-1, 4)[:, 1:]
        p = np.asarray(m.points)
        e = np.concatenate([
            np.linalg.norm(p[f[:, 1]] - p[f[:, 0]], axis=1),
            np.linalg.norm(p[f[:, 2]] - p[f[:, 1]], axis=1),
            np.linalg.norm(p[f[:, 0]] - p[f[:, 2]], axis=1),
        ])
        e = e[np.isfinite(e) & (e > 0)]
        return float(np.median(e)) if e.size else None
    except Exception as e:                                    # noqa: BLE001
        logger.debug(f"Could not measure surface resolution of {stl_path}: {e}")
        return None


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
    cad_curvature_elements: int | None = None,
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
    # Same reference-length floors the rocket path uses (see the module
    # constants): the caller's own scale, but never closer than
    # _FARFIELD_REF_LENGTHS reference lengths, where "reference length" is the
    # largest extent the body actually has. min()/max() only ever enlarge.
    _ref_L = max(L, info.cross_width, info.cross_height, 1e-6)
    up_x   = min(-5.0 * L, -_UPSTREAM_REF_LENGTHS * _ref_L)
    down_x = max(L + 15.0 * L, L + _WAKE_REF_LENGTHS * _ref_L)
    rad    = max(domain_radius_scale, 10.0) * cross_r
    rad    = max(rad, 6.0 * max(info.cross_width, info.cross_height, char))
    rad    = max(rad, _FARFIELD_REF_LENGTHS * _ref_L)
    logger.info(
        f"Farfield radius {rad:.2f} m ({rad / _ref_L:.1f} reference lengths of "
        f"{_ref_L:.3f} m); domain x[{up_x:.1f}, {down_x:.1f}] m"
    )

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
            if not reparametrised:
                # The wall mesh IS the imported triangulation from here on:
                # without a parametrisation gmsh cannot remesh those surfaces,
                # so generate(2) keeps them exactly as supplied and the size
                # field controls the VOLUME only.
                #
                # This was silent, and it invalidated the thing it was silent
                # about: the ONERA M6 benchmark solved the same case at a 20 mm
                # and a 10 mm "wall size" and got byte-identical surface meshes
                # (16,900 points / 33,796 cells, same median cell area) with a
                # 238k vs 1.47M volume. Its refinement study was measuring the
                # volume mesh and reporting it as surface convergence.
                _res = _surface_mesh_resolution(info.preview_stl)
                _msg = (
                    "Imported tessellation is too dense to reparametrise, so "
                    "the WALL MESH IS THE IMPORTED TRIANGULATION and cannot be "
                    "refined. The wall size setting controls the volume mesh "
                    "only."
                )
                if _res:
                    _msg += (f" Effective wall resolution is the file's own "
                             f"median edge, {_res * 1000:.2f} mm")
                    if _res > lc_wall * 1.25:
                        _msg += (f" — {_res / lc_wall:.1f}x COARSER than the "
                                 f"{lc_wall * 1000:.2f} mm requested. Surface "
                                 f"quantities (Cp, suction peaks, sectional "
                                 f"loads) are limited by the import, not by "
                                 f"this setting; supply a finer tessellation "
                                 f"or a STEP/IGES file to refine them.")
                    else:
                        _msg += (f", against {lc_wall * 1000:.2f} mm requested.")
                logger.warning(_msg)
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

        # ── Curvature-driven feature refinement ───────────────────────────────
        # gmsh's own MeshSizeFromCurvature reads a surface parametrisation, so on
        # a re-topologised STL it does nothing at all — measured on the ONERA M6
        # wing, raising it from 12 to 60 changed the cell count by 0.8% and the
        # lift not at all. Curvature is therefore measured directly off the
        # triangulation here and turned into an explicit distance field.
        #
        # What it buys: a wing leading edge of radius ~7 mm meshed with 20 mm
        # elements is a flat facet, and the suction peak that carries the lift
        # forms on exactly that radius.
        f_feature = _curvature_feature_field(
            gmsh, info.preview_stl, lc_wall, lc_far, is_brep=info.is_brep,
        )

        # ── Edge refinement ───────────────────────────────────────────────────
        # The curvature field seeds at most a few hundred discrete points and
        # relaxes back over four feature widths, so on a long edge its bands are
        # islands: on the M6 wing the seeds land ~12 mm apart along a 2.4 m
        # leading edge while each band is ~4 mm wide, leaving most of the edge at
        # the wall size. The curves bounding the wall patches ARE those edges —
        # exact on a B-Rep, and on the discrete route classifySurfaces splits the
        # shell at exactly the feature angle — so the rocket path's continuous
        # band applies here unchanged. It is what took AGARD-B from a 10-18% lift
        # deficit to inside tolerance.
        # No `reparametrised` guard. classifySurfaces splits the shell at the
        # feature angle on BOTH routes, so the discrete route has real bounding
        # curves too; the guard was excluding exactly the case that needed it
        # most (an STL over the reparametrisation limit, which is where the wall
        # mesh is frozen and every bit of volume refinement at the edge counts).
        # _edge_refinement_field returns None when a body genuinely has no
        # curves, so the worst case here is the previous behaviour.
        f_edge = _edge_refinement_field(gmsh, wall_surfs, lc_wall, lc_far)

        # ── Size fields ───────────────────────────────────────────────────────
        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", wall_surfs)
        gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)

        # SizeMax/DistMax derived from _MAX_GROWTH_RATIO, exactly as on the
        # rocket path — char*8.0 was an independent guess and on a body whose
        # frontal radius is small next to its span (a wing) it relaxed to the
        # farfield size within a fraction of a chord.
        lc_far = _graded_far_size(lc_wall, rad, lc_far)
        _dist_max = _graded_dist_max(lc_wall, lc_far)

        f_thr = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", lc_wall)
        gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", lc_far)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thr, "DistMax", _dist_max)

        # Keep the whole body envelope fine even where the distance field has
        # already relaxed (concave pockets, gaps between separate solids).
        # Thickness on every box, for the reason given on the rocket path: a
        # gmsh Box field without one is a step, and the background field is the
        # minimum over all fields, so the step is a size discontinuity.
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
        gmsh.model.mesh.field.setNumber(f_body, "Thickness",
                                        _box_thickness(lc_wall * 2.0, lc_far))

        f_wake = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f_wake, "XMin",  L)
        gmsh.model.mesh.field.setNumber(f_wake, "XMax",  L + 3.0 * L)
        gmsh.model.mesh.field.setNumber(f_wake, "YMin", -2.0 * hw)
        gmsh.model.mesh.field.setNumber(f_wake, "YMax",  2.0 * hw)
        gmsh.model.mesh.field.setNumber(f_wake, "ZMin", -2.0 * hh)
        gmsh.model.mesh.field.setNumber(f_wake, "ZMax",  2.0 * hh)
        gmsh.model.mesh.field.setNumber(f_wake, "VIn",   char * 2.0)
        gmsh.model.mesh.field.setNumber(f_wake, "VOut",  lc_far)
        gmsh.model.mesh.field.setNumber(f_wake, "Thickness",
                                        _box_thickness(char * 2.0, lc_far))

        _fields = [f_thr, f_body, f_wake]
        for _f in (f_feature, f_edge):
            if _f is not None:
                _fields.append(_f)
        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", _fields)
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
        _curv_default = (20 if info.is_brep else 12) if reparametrised else 0
        _curv = _curv_default if cad_curvature_elements is None \
            else (int(cad_curvature_elements) if reparametrised else 0)
        if _curv != _curv_default:
            logger.info(f"Curvature sizing: {_curv} elements per 2π "
                        f"(default for this route {_curv_default})")
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", _curv)
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

        def _optimize():
            for _label, _opt in (("Gmsh default", ""), ("Netgen", "Netgen")):
                try:
                    gmsh.model.mesh.optimize(_opt, force=True)
                    logger.info(f"Mesh optimization ({_label}) complete")
                except Exception as e:
                    logger.warning(f"Mesh optimization ({_label}) failed: {e}")

        meshed = False
        problems: list = []
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
                    # See the rocket path: DistMax moves with SizeMin so the
                    # growth ratio stays at the cap on the fallback rungs too.
                    _sm = lc_wall * scale
                    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", _sm)
                    gmsh.model.mesh.field.setNumber(
                        f_thr, "DistMax", _graded_dist_max(_sm, lc_far))
                    gmsh.model.mesh.field.setAsBackgroundMesh(f_min)
                gmsh.model.mesh.generate(2)
                gmsh.model.mesh.generate(3)
                n = _n_tets()
                if n > 0:
                    # Audit inside the rung — see the rocket path. wall_surfs is
                    # passed here now; it never was, so the manifoldness check
                    # (the one that catches two regions of mesh occupying the
                    # same space) simply did not run on imported CAD at all.
                    _optimize()
                    problems = _check_mesh_quality(gmsh, char, 0,
                                                   wall_surfs=wall_surfs)
                    if not problems:
                        logger.info(f"3D mesh built with {label}: {n:,} elements"
                                    + ("" if i == 0 else f" (attempt {i + 1})"))
                        meshed = True
                        break
                    logger.warning(
                        f"{label} produced {n:,} elements but the mesh is "
                        f"invalid ({'; '.join(problems)}) — trying the next "
                        f"strategy."
                    )
                    continue
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
            # Optimisation and the quality audit already ran inside the winning
            # ladder rung. Refuse an invalid mesh rather than export it.
            if not meshed and problems:
                raise RuntimeError(
                    "Mesh generation produced an INVALID mesh at every "
                    "refinement strategy: " + "; ".join(problems) + ". Do not "
                    "trust any result from this geometry until it meshes "
                    "cleanly — try a coarser refinement level, or repair the "
                    "CAD (self-intersecting faces, features thinner than the "
                    "element size)."
                )

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

# Element size on a body edge, as a fraction of the wall size. See
# _edge_refinement_field. Measured on the AGARD-B medium mesh, surface triangles
# against this factor: none 10.6k, 0.50 13.8k, 0.35 18.8k, 0.25 24.0k, 0.15
# 36.2k. 0.25 buys a 4x finer leading edge for 2.3x the surface mesh.
_EDGE_REFINE_FACTOR = 0.25


def _edge_refinement_field(gmsh, wall_surfs: list, lc_rocket: float, lc_far: float,
                           factor: float = _EDGE_REFINE_FACTOR):
    """Refine a band around every sharp edge of the body. Returns a field tag.

    The box fields size the fin *region*; they cannot size the fin *edge*. A
    constant-thickness panel has a knife leading edge, and the suction peak that
    carries most of its lift forms within a percent or two of chord behind it —
    on a 14 mm wall element that peak is one facet wide and the circulation it
    should have produced is simply absent.

    Uses the B-Rep's own curves, so "sharp edge" needs no detection: the fin
    outline, the fin-root junction, the nose/base rims and the diameter steps
    are precisely the curves bounding the wall surfaces. Returns None when the
    body has none of them (it cannot, in practice, but the caller treats the
    field as optional).

    The band is deliberately narrow — ``factor`` of the wall size at the edge,
    growing back over four element widths. Wider bands buy cell count, not
    circulation.

    ``SizeMax`` is the FARFIELD size, not the wall size. It is the value this
    field returns everywhere outside the band, and the background field is the
    minimum over all fields, so a wall-sized ceiling here silently caps the
    entire tunnel: measured on AGARD-B, 14 mm elements out to the 1 m farfield
    wall turned a 10.6k-triangle surface mesh into 1.8M, of which 1.78M were on
    the six farfield faces.
    """
    curves = set()
    for s in wall_surfs:
        try:
            for (dim, tag) in gmsh.model.getBoundary([(2, s)], combined=False,
                                                     oriented=False, recursive=False):
                if dim == 1:
                    curves.add(abs(int(tag)))
        except Exception as e:                                # noqa: BLE001, PERF203
            logger.debug(f"Edge refinement: boundary of surface {s} unavailable: {e}")
    if not curves:
        return None

    lc_edge = lc_rocket * factor
    f_d = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_d, "CurvesList", sorted(curves))
    gmsh.model.mesh.field.setNumber(f_d, "Sampling", 200)

    f_t = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_t, "InField", f_d)
    gmsh.model.mesh.field.setNumber(f_t, "SizeMin", lc_edge)
    gmsh.model.mesh.field.setNumber(f_t, "SizeMax", lc_far)
    gmsh.model.mesh.field.setNumber(f_t, "DistMin", lc_edge)
    gmsh.model.mesh.field.setNumber(f_t, "DistMax", lc_edge * 4.0)

    logger.info(
        f"Edge refinement: {len(curves)} wall curves seeded at {lc_edge:.4g} m "
        f"({factor:.0%} of the {lc_rocket:.4g} m wall size), relaxing back over "
        f"{lc_edge * 4.0:.4g} m."
    )
    return f_t


def _clamp_profile_tip(profile: list, lc_rocket: float) -> list:
    """Widen a nose tip that is finer than the mesh can ever represent.

    ``cfd_profile`` floors the apex radius just above zero (1e-4 m) so the
    revolve closes on a disc rather than a degenerate vertex. That disc is a
    real B-Rep face with a real bounding circle, and gmsh meshes the circle at
    its own scale no matter what ``Mesh.MeshSizeMin`` says — a 1e-4 m tip
    against a 2.2e-3 m size floor gives 3e-5 m facets sitting next to 5e-3 m
    ones. The transition band folds, and generate(3) dies with

        Invalid boundary mesh (overlapping facets) on surface 2

    which is what skipped the SU2 cone benchmark at coarse and fine (medium
    survived by luck: the failure is a sliver fold, not a clean size threshold).

    So the apex is snapped out to half the surface size floor. On the 10°
    validation cone that is a 1.1 mm disc on an 88 mm base radius — under 1.3%
    of the radius, blunting the first 6 mm of a 500 mm cone, and the Cp
    comparison window starts at 30 mm. Anything the mesh could have resolved is
    left alone: the clamp only ever raises a radius the mesher was going to
    misrepresent anyway, and it says so in the log when it fires.
    """
    if len(profile) < 2:
        return profile
    r_floor = lc_rocket * 0.05 * 0.5
    x0, r0 = profile[0]
    if r0 >= r_floor:
        return profile
    # Only the apex station moves, so the body keeps its length and every other
    # station its declared radius. Guard the case where station 1 is itself
    # below the floor (a very long, very fine taper): then the whole leading run
    # of stations under the floor is lifted, otherwise the profile goes backwards.
    out = [(x, max(r, r_floor)) for x, r in profile]
    logger.info(
        f"Nose tip clamped: apex radius {r0:.3e} m -> {r_floor:.3e} m "
        f"(half the {lc_rocket * 0.05:.3e} m surface size floor). A tip finer "
        f"than the mesh folds the facets around it."
    )
    return out


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


def _audit_growth_ratio(gmsh) -> Optional[dict]:
    """Size ratio between face-adjacent 3D cells — the real grading metric.

    Returns percentiles of max(h_a/h_b, h_b/h_a) over every interior face, or
    None if it could not run.

    Reported because nothing else in the pipeline could see it. The UI's mesh
    "quality rating" is built from mean aspect ratio and max skew, and a mesh
    whose cell size collapses over one band scores "Good" on both: they measure
    the shape of individual cells, not how a cell relates to its neighbour. A
    Roe/MUSCL flux and a Green-Gauss gradient are both evaluated ACROSS the
    face, so the neighbour ratio is what bounds their accuracy.
    """
    try:
        import numpy as np
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        idx = {int(t): i for i, t in enumerate(node_tags)}
        pts = np.asarray(coords, dtype=float).reshape(-1, 3)

        tets = []
        for _, vol in gmsh.model.getEntities(3):
            types, _, nodes = gmsh.model.mesh.getElements(3, vol)
            for et, nd in zip(types, nodes):
                _, _, _, nn, _, _ = gmsh.model.mesh.getElementProperties(et)
                if nn != 4:
                    continue
                arr = np.fromiter((idx.get(int(v), -1) for v in nd),
                                  dtype=np.int64, count=len(nd))
                tets.append(arr.reshape(-1, 4))
        if not tets:
            return None
        tets = np.concatenate(tets)
        tets = tets[(tets >= 0).all(axis=1)]
        if len(tets) < 100:
            return None

        vol6 = np.abs(np.einsum(
            "ij,ij->i",
            pts[tets[:, 1]] - pts[tets[:, 0]],
            np.cross(pts[tets[:, 2]] - pts[tets[:, 0]],
                     pts[tets[:, 3]] - pts[tets[:, 0]])))
        h = np.maximum(vol6, 1e-30) ** (1.0 / 3.0)

        combos = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        faces = np.concatenate([np.sort(tets[:, c], axis=1) for c in combos])
        owner = np.tile(np.arange(len(tets)), 4)
        order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
        faces, owner = faces[order], owner[order]
        same = np.all(faces[1:] == faces[:-1], axis=1)
        a, b = owner[:-1][same], owner[1:][same]
        if a.size == 0:
            return None
        ratio = np.maximum(h[a] / h[b], h[b] / h[a])
        return {
            "p50": float(np.percentile(ratio, 50)),
            "p99": float(np.percentile(ratio, 99)),
            "max": float(ratio.max()),
            "pct_over_2": float((ratio > 2.0).mean() * 100.0),
            "n_faces": int(ratio.size),
        }
    except Exception as e:
        logger.debug(f"Growth-ratio audit could not run: {e}")
        return None


def _check_mesh_quality(gmsh, body_r: float, n_bl_entities: int,
                        wall_surfs=None) -> list:
    """
    Post-generation mesh quality validation.
    Checks element types, counts, quality metrics, grading and wall
    manifoldness.

    Returns a list of FATAL problem descriptions — empty means the mesh is
    usable. The caller is expected to act on a non-empty list (retry at another
    setting, or refuse to export); this used to log the same findings and
    return, which meant a mesh known to be invalid was written, loaded by SU2
    and solved anyway. "MESH INVALID" in a log nobody reads is not a check.
    """
    problems: list = []
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
                    problems.append(
                        f"{n_negative} of {len(sicn_data):,} sampled elements "
                        f"have a negative Jacobian (inverted cells)"
                    )
        except Exception as e:
            # getElementQualities may not be available in all Gmsh builds
            logger.warning(f"Element quality check unavailable: {e}")

    # ── Grading ───────────────────────────────────────────────────────────────
    grow = _audit_growth_ratio(gmsh)
    if grow:
        logger.info(
            f"  Neighbour size ratio: p50={grow['p50']:.2f}  "
            f"p99={grow['p99']:.2f}  max={grow['max']:.2f}  "
            f"({grow['pct_over_2']:.3f}% of {grow['n_faces']:,} interior faces "
            f"above 2x)"
        )
        if grow["p99"] > 2.0:
            logger.warning(
                f"  Cell size changes by {grow['p99']:.1f}x between neighbours "
                f"at the 99th percentile (target <= {_MAX_GROWTH_RATIO}). "
                f"Gradients and fluxes are evaluated across those faces; "
                f"expect elevated numerical entropy and a drag/lift error that "
                f"refinement will not remove."
            )

    # ── Wall manifoldness ─────────────────────────────────────────────────────
    buried = _audit_wall_is_manifold(gmsh, wall_surfs)
    if buried is None:
        pass
    elif buried:
        problems.append(
            f"{buried:,} wall triangles have a 3D element on both sides, so the "
            f"wall is not a boundary of the fluid (two parts of the mesh occupy "
            f"the same space; SU2 cannot converge on it)"
        )
    else:
        logger.info("[OK] Wall is manifold: every wall face bounds exactly one cell")

    for p in problems:
        logger.error(f"MESH INVALID: {p}")
    return problems


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


def _fin_section_2d(chord: float, thick: float, kind: str) -> list:
    """Chordwise section of a fin panel as a closed polygon of ``(s, z)``.

    ``s`` runs 0 (leading edge) to ``chord`` (trailing edge); ``z`` is the
    half-thickness direction. Points are ordered around the section.

    Why this exists: the mesher used to extrude the planform into a slab and
    call it a fin, regardless of what the component declared. A slab presents a
    blunt face across the entire leading edge — on the AGARD-B wing that face is
    4% of chord, which in an Euler solve stagnates the flow where the suction
    peak should form. The declared section is now built.

      Square    the slab, unchanged — this is what a square-edged fin is.
      Airfoil   a symmetric double wedge: sharp LE and TE, full thickness over
                the middle half. Not a NACA section, but it has the property
                that matters here — the flow attaches at a point instead of
                across a step.
      Rounded   the slab with LE and TE capped by half-cylinders of radius t/2,
                which is what a filed-round fin edge is.
    """
    k = (kind or "Square").strip().lower()
    # Thickness is clamped against the chord rather than the section being
    # downgraded to a slab on a short chord: a loft needs the SAME polygon at
    # both ends, and a root that is an airfoil lofted to a tip that fell back to
    # a rectangle is not a solid — OCC reports it downstream as
    # "Difference failed - BOPAlgo_AlertTooFewArguments".
    t = min(0.5 * thick, 0.2 * chord)
    if k not in ("airfoil", "rounded") or t <= 0:
        return [(0.0, -t), (chord, -t), (chord, t), (0.0, t)]

    if k == "airfoil":
        a, b = 0.25 * chord, 0.75 * chord
        return [(0.0, 0.0), (a, -t), (b, -t), (chord, 0.0), (b, t), (a, t)]

    # Rounded: walk the lower surface LE→TE, round the TE, walk the upper
    # surface back, round the LE. Three interior points per cap is enough at the
    # element sizes this mesher works at and keeps the wire cheap.
    def _arc(cx, a0, a1, n=5):
        return [(cx + t * math.cos(a0 + (a1 - a0) * i / (n - 1)),
                 t * math.sin(a0 + (a1 - a0) * i / (n - 1)))
                for i in range(n)]

    te_cap = _arc(chord - t, -0.5 * math.pi, 0.5 * math.pi)
    le_cap = _arc(t, 0.5 * math.pi, 1.5 * math.pi)
    return te_cap + le_cap


def _fin_solid_lofted(occ, section: str, x_root_LE: float, c_root: float,
                      r_root: float, x_tip_LE: float, c_tip: float,
                      r_tip: float, thick: float):
    """Loft a shaped fin between its root and tip sections. Returns a volume tag.

    The panel lies in the x–y plane (span along +y, chord along +x) and is
    thick in z, matching the extruded construction it replaces. Both sections
    are the same polygon — see :func:`_fin_section_2d` — so a nearly-pointed
    delta tip comes out as a scaled-down section rather than a different shape.
    """
    wires = []
    for x_le, chord, span_r in (
        (x_root_LE, c_root, r_root),
        (x_tip_LE, c_tip, r_tip),
    ):
        pts = [occ.addPoint(x_le + s, span_r, z)
               for s, z in _fin_section_2d(chord, thick, section)]
        lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                 for i in range(len(pts))]
        wires.append(occ.addWire(lines))

    out = occ.addThruSections(wires, makeSolid=True, makeRuled=True)
    vols = [tag for dim, tag in out if dim == 3]
    if not vols:
        raise RuntimeError("addThruSections produced no volume")
    return vols[0]


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
        f"section={rocket.get('fin_cross_section', 'Square')}  "
        f"x=[{x_root_LE:.3f}->{x_tip_TE:.3f}]  "
        f"r=[{r_root:.4f}->{r_tip:.4f}] (local body r={r_local_mid:.4f})"
    )

    section = str(rocket.get("fin_cross_section", "Square"))
    # A shaped section needs two real sections to loft between. A true delta has
    # no tip chord at all, so it keeps the extruded planform — a sharp-edged
    # triangle is still the right planform, it just has square edges.
    want_loft = (section.strip().lower() in ("airfoil", "rounded")
                 and fin_Ct > max(1e-6, 1e-4 * fin_Cr))
    if section.strip().lower() != "square" and not want_loft:
        logger.info(f"Fin cross-section '{section}' requested but the tip chord "
                    f"({fin_Ct:.4g} m) is degenerate — extruding a square "
                    f"section instead.")

    for i in range(n_fins):
        angle = 2.0 * math.pi * i / n_fins

        try:
            if want_loft:
                try:
                    vol_tag = _fin_solid_lofted(
                        occ, section, x_root_LE, fin_Cr, r_root,
                        x_tip_LE, fin_Ct, r_tip, fin_t)
                    occ.rotate([(3, vol_tag)], 0, 0, 0, 1, 0, 0, angle)
                    parts.append((3, vol_tag))
                    continue
                except Exception as e:              # noqa: BLE001
                    # Fall through to the extruded planform — a square-edged fin
                    # of the right planform beats no fin, and beats the box.
                    logger.warning(
                        f"Lofting the '{section}' fin section failed ({e}); "
                        f"extruding a square section instead.")
                    want_loft = False

            # A true delta fin has tip_chord = 0, so the two tip corners are the
            # SAME point and addLine() between them fails — which used to drop
            # the fin into the box fallback below. A box has far more planform
            # area than the delta it replaced (it inflated lift ~25% on the
            # AGARD-B validation case) and the only trace was a log warning, so
            # build the triangle the geometry actually describes instead.
            sharp_tip = fin_Ct <= max(1e-6, 1e-4 * max(fin_Cr, 1e-9))

            p0 = occ.addPoint(x_root_LE, r_root, 0.0)
            p1 = occ.addPoint(x_root_TE, r_root, 0.0)
            p2 = occ.addPoint(x_tip_TE,  r_tip,  0.0)

            if sharp_tip:
                lines = [occ.addLine(p0, p1),
                         occ.addLine(p1, p2),
                         occ.addLine(p2, p0)]
            else:
                p3 = occ.addPoint(x_tip_LE, r_tip, 0.0)
                lines = [occ.addLine(p0, p1),
                         occ.addLine(p1, p2),
                         occ.addLine(p2, p3),
                         occ.addLine(p3, p0)]

            loop = occ.addCurveLoop(lines)
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
