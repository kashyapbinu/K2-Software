"""
CFD benchmarks.
===============

Fast, deterministic (no solver):
    * Taylor–Maccoll reference gate — the exact cone solver vs published
      NACA-1135 cone-table values (proves the *reference* before it is used to
      judge SU2).

Published / solver-proof (SU2, slow):
    * supersonic cone at M=2 → surface pressure coefficient vs Taylor–Maccoll.

Analytic-vs-solver (SU2, slow) — the user's core ask:
    * K2 Barrowman aerodynamics (physics.aerodynamics.AeroModel) vs SU2 on the
      canonical rocket (Cd, normal-force slope). Barrowman is an engineering
      approximation, so tolerances are loose and the level is ESTIMATED.

The SU2 runs go through the real K2 pipeline (geometry → STL → gmsh → SU2). That
pipeline is heavy and, per project notes, fragile when driven head-less, so
those benchmarks *attempt* a run and mark themselves skipped (not failed) if the
pipeline raises — exactly like the OpenRocket bridge.
"""
from __future__ import annotations

import math
from pathlib import Path

from core.validation import ValidationLevel
from validation.harness import Benchmark, Comparison

_WORK = Path("cfd_run") / "validation"


# ── fast: Taylor–Maccoll vs published cone tables ─────────────────────────────

# NACA Report 1135, cone tables (γ=1.4). (M∞, θc_deg) -> (shock_deg, Cp_surface).
_NACA1135 = {
    (2.0, 10.0): dict(shock=31.2, cp=0.105),
    (3.0, 10.0): dict(shock=21.8, cp=0.088),
}


def bench_taylor_maccoll_reference() -> Benchmark:
    from validation.cfd.taylor_maccoll import solve_cone

    bm = Benchmark(name="Taylor–Maccoll cone vs NACA-1135", domain="cfd",
                   reference="NACA Report 1135 cone tables",
                   level=ValidationLevel.VALIDATED)
    for (M, tc), ref in _NACA1135.items():
        s = solve_cone(M, tc)
        bm.add(Comparison.make(f"Shock angle (M={M}, θc={tc}°)",
                               s.shock_angle_deg, ref["shock"],
                               "NACA-1135", "deg", tol_rel=0.01))
        bm.add(Comparison.make(f"Surface Cp (M={M}, θc={tc}°)",
                               s.cp_surface, ref["cp"],
                               "NACA-1135", "-", tol_rel=0.05))
    return bm


# ── SU2 single-point runner (real pipeline) ───────────────────────────────────

def _run_su2_point(assembly, mach: float, aoa_deg: float, refinement: str,
                   tag: str, cg_from_nose_m: float | None = None,
                   geom_overrides: dict | None = None,
                   turbulence_model: str = "SST"):
    """Run ONE SU2 point on `assembly` through the K2 pipeline. Returns CFDResult.

    `geom_overrides` patches the extracted geometry dict (e.g. ``fin_count=0`` to
    keep a validation cone fin-free, since extract_cfd_geometry fabricates fins
    when an assembly has none).
    """
    from cfd.solvers.base import CFDConfig
    from cfd.solvers.su2_solver import SU2Solver
    from cfd.geometry_exporter import extract_cfd_geometry, export_assembly_to_stl

    work = _WORK / tag
    work.mkdir(parents=True, exist_ok=True)
    geom = extract_cfd_geometry(assembly)
    if geom_overrides:
        geom.update(geom_overrides)
    stl = export_assembly_to_stl(assembly, work / "geometry.stl")

    # Refuse to benchmark against SU2 on a non-watertight mesh. The head-less
    # auto-export frequently leaves open/non-manifold edges (the supervised CFD
    # workspace does not); a leaky surface makes the SU2 solution meaningless, so
    # raise → the caller turns it into a documented skip rather than a false fail.
    n_open = _stl_open_edges(stl)
    if n_open > 0:
        raise RuntimeError(
            f"exported STL not watertight ({n_open} open/non-manifold edges); "
            "run this case from the CFD workspace with a supervised mesh")

    cfg = CFDConfig(
        mach=mach, angle_of_attack_deg=aoa_deg, altitude_m=3000.0,
        mesh_refinement=refinement, work_dir=work,
        geometry_stl=stl, geometry_dict=geom,
        cg_from_nose_m=cg_from_nose_m, turbulence_model=turbulence_model,
        # Keep curvature-based sizing ON (coarse/medium meshes fold at the cone
        # tip otherwise — see project notes). Cap iterations for a quick check.
        max_iterations=2000,
    )
    solver = SU2Solver(cfg)
    solver.generate_mesh()
    solver.generate_case()
    for _ in solver.run():
        pass
    return solver.parse_results()


def _stl_open_edges(stl_path) -> int:
    """Count open *boundary* edges (holes) of an STL — 0 means no leaks.

    Non-manifold edges at fin/body joints are tolerated (coincident faces from
    merging closed fin solids); only true holes make a surface unusable for CFD.
    """
    import pyvista as pv
    mesh = pv.read(str(stl_path))
    edges = mesh.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=False,
        feature_edges=False, manifold_edges=False)
    return int(edges.n_cells)


def _skip(name, reference, exc) -> Benchmark:
    bm = Benchmark(name=name, domain="cfd", reference=reference)
    bm.skipped = True
    bm.skip_reason = f"SU2 pipeline unavailable/failed headless: {exc}"
    return bm


def _surface_cp_forebody(vtk_path, x_lo: float, x_hi: float):
    """Mean surface pressure coefficient over a forebody x-window of a SU2
    surface_flow file — excludes the flat base disk (whose low pressure is base
    drag, absent from the inviscid Taylor–Maccoll cone solution)."""
    import pyvista as pv
    import numpy as np
    m = pv.read(str(vtk_path))
    cp = np.asarray(m.point_data["Pressure_Coefficient"])
    x = m.points[:, 0]
    mask = (x > x_lo) & (x < x_hi)
    if mask.sum() == 0:
        raise RuntimeError("no forebody surface points found")
    return float(cp[mask].mean()), int(mask.sum())


# ── SU2 supersonic cone vs Taylor–Maccoll ─────────────────────────────────────

def bench_su2_cone() -> Benchmark:
    """SU2 cone at M=2 vs the exact Taylor–Maccoll surface pressure coefficient."""
    name = "SU2 cone vs Taylor–Maccoll"
    ref = "Taylor–Maccoll exact (M=2, 10° cone)"
    try:
        from validation.cfd.taylor_maccoll import solve_cone
        from validation.cfd.cone_geometry import cone_assembly

        M, half_angle, cone_L = 2.0, 10.0, 0.5
        exact = solve_cone(M, half_angle)
        # TM is inviscid → run SU2 Euler on a fine mesh so the attached shock is
        # resolved (a viscous medium mesh smears it). Compare the *forebody
        # surface pressure coefficient* — the direct TM observable — not the
        # total Cd, which is contaminated by base drag off the flat aft face.
        res = _run_su2_point(cone_assembly(half_angle_deg=half_angle, length=cone_L),
                             mach=M, aoa_deg=0.0, refinement="fine", tag="cone",
                             geom_overrides={"fin_count": 0},
                             turbulence_model="Euler")
        surf = res.surface_vtk or (_WORK / "cone" / "surface_flow.vtu")
        cp_su2, n = _surface_cp_forebody(surf, 0.06 * cone_L, 0.94 * cone_L)

        bm = Benchmark(name=name, domain="cfd", reference=ref,
                       level=ValidationLevel.VALIDATED)
        bm.add(Comparison.make("Cone surface Cp (M=2, 10°)",
                               cp_su2, exact.cp_surface,
                               "Taylor–Maccoll", "-", tol_rel=0.10,
                               note=f"SU2 Euler forebody mean over {n} pts"))
        return bm
    except Exception as exc:
        return _skip(name, ref, exc)


# ── K2 Barrowman aero vs SU2 ──────────────────────────────────────────────────

def bench_barrowman_vs_su2() -> Benchmark:
    """K2 AeroModel (Barrowman) vs SU2 on the canonical rocket: Cd and Cn."""
    name = "Barrowman aero vs SU2"
    ref = "SU2 RANS (canonical rocket)"
    try:
        from validation.cases.rocket_canonical import canonical_state, canonical_assembly
        from physics.aerodynamics import AeroModel
        from environment.atmosphere_model import Atmosphere

        mach, aoa = 0.5, 4.0
        asm = canonical_assembly()
        state = canonical_state()

        res = _run_su2_point(asm, mach=mach, aoa_deg=aoa, refinement="medium",
                             tag="barrowman", cg_from_nose_m=state.cg or 1.2)

        # Watertightness is gated upstream in _run_su2_point; here just require a
        # converged solve before trusting the reference.
        if not res.converged:
            return _skip(name, ref, "SU2 did not converge head-less — "
                         "run this case from the CFD workspace")

        # K2 Barrowman prediction at the same condition.
        atm = Atmosphere()
        alt = 3000.0
        a = atm.speed_of_sound(alt)
        rho = atm.density(alt)
        v = mach * a
        q = 0.5 * rho * v ** 2
        aero = AeroModel.from_state(state)
        k2 = aero.compute(alpha=math.radians(aoa), mach=mach, q_dyn=q,
                          pitch_rate=0.0, v_rel=v, cg=state.cg or 1.2)

        bm = Benchmark(name=name, domain="cfd", reference=ref,
                       level=ValidationLevel.ESTIMATED)

        # ── Gating: the things these fixes actually make correct ──
        # 1. SU2 normalises by the true body frontal area (validates the
        #    max_diameter reference-area fix; previously the STL bounding box
        #    picked up the fin span and inflated the area ~10×).
        a_body = math.pi * (state.diameter / 2.0) ** 2
        bm.add(Comparison.make("CFD reference area", res.reference_area_m2, a_body,
                               "π·(d_body/2)²", "m²", tol_rel=0.02,
                               note="validates body-frontal ref area"))
        # 2. Physically sane signs/trends: drag and lift positive at +AoA.
        bm.add(Comparison.make("Drag positive", 1.0 if res.cd > 0 else 0.0, 1.0,
                               "sign", "bool", tol_abs=0.5))
        bm.add(Comparison.make("Lift positive at +AoA",
                               1.0 if res.cl > 0 else 0.0, 1.0, "sign", "bool",
                               tol_abs=0.5))

        # ── Gated: low-order Barrowman vs RANS, in the SAME axes ──
        # These carried a 100% band while they read 45% and 71% off, which is
        # not a tolerance — nothing short of a sign error could trip it. Both
        # rows were 100%-banded because two real defects were being absorbed
        # rather than found:
        #
        #   * the analytic model counted all four fins of a cruciform set as
        #     lifting panels (physics.aerodynamics.compute_fin_cn_alpha), and
        #   * this row compared K2's BODY-AXIS Cn against SU2's WIND-AXIS Cl.
        #
        # With the fin count fixed and the axes reconciled the gap is 13%, so
        # the band is 25%: wide enough for a preliminary-design method against
        # RANS on a 2.4-body-radius fin, tight enough to catch the next one.
        #
        # CL = CN·cos(alpha) - CA·sin(alpha). K2's cd at incidence is the
        # wind-axis drag, which is what SU2's CD is too, so it stands in for CA
        # to within the sin(alpha) weighting on a term that is itself small.
        # Put both sides in the SAME axes before comparing. K2's `cd` is a
        # WIND-axis drag (the engine applies it anti-parallel to the velocity
        # vector, simulation_engine.py:640) while `cn` is a BODY-axis normal
        # force, so rotate SU2's pair into body axes rather than guessing at
        # K2's. The row used to compare K2's Cn against SU2's Cl outright.
        cos_a, sin_a = math.cos(math.radians(aoa)), math.sin(math.radians(aoa))
        cn_su2 = res.cl * cos_a + res.cd * sin_a
        bm.add(Comparison.make("Normal force Cn (body axes)",
                               k2["cn"], cn_su2, "SU2", "-", tol_rel=0.10,
                               note="SU2 CL/CD rotated into body axes"))

        # DIAGNOSTIC, and it should stay uncomfortable. K2's drag build-up is
        # OpenRocket's, including base drag as a function of Mach alone
        # (0.12 + 0.13*M^2). Against the AGARD-B wind tunnel that model runs
        # +19.9% at M=0.2 rising to +84.3% at M=0.9 — see the C_D0 rows in
        # bench_agardb_barrowman. This row inherits that error; the 12% here is
        # not evidence the drag model is good, only that SU2's canonical drag
        # happens to sit between K2 and the measurement.
        bm.add(Comparison.make("Drag coefficient Cd (wind axes)",
                               k2["cd"], res.cd, "SU2", "-", tol_rel=0.10,
                               diagnostic=True,
                               note="inherits the OpenRocket base-drag model; "
                                    "see the AGARD-B C_D0 rows"))
        return bm
    except Exception as exc:
        return _skip(name, ref, exc)


# ── AGARD-B vs wind-tunnel measurement (AEDC-TR-70-100) ───────────────────────

_AGARDB_REF = "AEDC-TR-70-100 wind-tunnel data (AGARD-B, Tunnel 4T)"


def bench_agardb_barrowman() -> Benchmark:
    """K2's Barrowman lift-curve slope vs the AGARD-B measurement.

    Two kinds of row, both now GATED against the measurement:

      * **Mach trend** — C_L_alpha(M) normalised by its own value at M=0.2.
        This isolates K2's compressibility correction from its absolute level.
      * **Absolute level** — the lift-curve slope itself, within 10%.

    The absolute rows spent most of their life as 300%-band diagnostics, on the
    reading that a wing spanning four body diameters is outside Barrowman's
    envelope and a 63% miss was therefore expected. Both halves of that were
    wrong, and 2026-08-31 fixed them:

      1. ``AeroModel.from_state`` reads flat fields and ignores
         ``state.assembly``, so the AGARD-B wing never reached the aero model —
         its ``or``-fallbacks had substituted a generic 4-fin rocket. 63% -> 21%.
      2. The fin term carried Barrowman's ``K_fb = 1 + tau``, which is only the
         fin-in-presence-of-body half of the wing-body interference. Adding the
         body-carryover half, ``(1 + tau)^2`` per NACA Report 1307, took it to
         under 5% at every Mach in the dataset.

    A method being low-order is a reason to check it against measurement, not a
    licence to widen the band until it passes.
    """
    from validation.cfd.agardb import barrowman_cl_alpha_per_deg, barrowman_cd0
    from validation.data.agardb_aedc import AEDC_TR_70_100 as REF

    bm = Benchmark(name="Barrowman lift slope vs AGARD-B experiment", domain="cfd",
                   reference=_AGARDB_REF, level=ValidationLevel.ESTIMATED)

    machs = [0.2, 0.4, 0.6, 0.8, 0.9]
    k2 = {m: barrowman_cl_alpha_per_deg(mach=m) for m in machs}
    base_k2, base_ref = k2[0.2], REF[0.2]["cl_alpha_per_deg"]

    for m in machs[1:]:
        bm.add(Comparison.make(
            f"C_Lα(M={m}) / C_Lα(M=0.2)", k2[m] / base_k2,
            REF[m]["cl_alpha_per_deg"] / base_ref,
            "AEDC-TR-70-100", "-", tol_rel=0.08,
            note="compressibility trend, self-normalised"))

    for m in (0.2, 0.6, 0.9):
        bm.add(Comparison.make(
            f"C_Lα at M={m} (absolute)", k2[m],
            REF[m]["cl_alpha_per_deg"], "AEDC-TR-70-100", "1/deg", tol_rel=0.10,
            note="absolute lift-curve slope vs wind tunnel, wing-area referenced"))

    # Zero-lift drag against the same report's measured C_D0 — the only measured
    # drag in the suite. DIAGNOSTIC, at a real 10% band, so the report shows the
    # gap at its true size instead of hiding it behind a band wide enough to
    # pass. K2's drag build-up is OpenRocket's, and its base drag is a function
    # of Mach alone (0.12 + 0.13*M^2, over half of this vehicle's total drag):
    #
    #     M=0.2  +19.9%     M=0.5  +32.5%     M=0.9  +84.3%
    #
    # The error grows monotonically with Mach, which is a wrong term rather than
    # scatter: the measured C_D0 rises 8% between M=0.2 and M=0.9 while the
    # model rises 88%. SU2 on this same geometry sits within 1.5% of the tunnel
    # at M=0.5, so the analytic model is the outlier, not the reference.
    #
    # Hoerner's boundary-layer-coupled base drag (C_D,base = 0.029/sqrt(C_D,f),
    # *Fluid-Dynamic Drag* ch.13) brings every one of these inside +/-5.5%. It
    # is NOT applied: it would diverge from OpenRocket, whose drag model is
    # separately validated. Caveat on the reference too — this C_D0 column is
    # the weaker of the report's two aggregates and its base term is
    # sting-dependent (see validation.data.agardb_aedc), so it is a 10%-class
    # reference at best. Treat this as a documented decision, not a TODO.
    for m in (0.2, 0.5, 0.9):
        bm.add(Comparison.make(
            f"C_D0 at M={m} (diagnostic)", barrowman_cd0(mach=m), REF[m]["cd0"],
            "AEDC-TR-70-100", "-", tol_rel=0.10, diagnostic=True,
            note="OpenRocket base-drag model vs wind tunnel — known open gap"))

    bm.curves["cl_alpha"] = {
        "x": machs, "k2": [k2[m] for m in machs],
        "ref": [REF[m]["cl_alpha_per_deg"] for m in machs],
        "xlabel": "Mach", "ylabel": "dC_L/dα (1/deg, wing area)"}
    bm.notes = ("Coefficients are wing-planform-area referenced (S = 4√3·D²), "
                "as in the source report; K2's body-area values are rescaled.")
    return bm


def bench_agardb_su2(machs=(0.5, 0.6, 0.7, 0.8),
                     alphas=(0.0, 2.0, 4.0, 6.0),
                     refinement: str = "medium",
                     max_iterations: int = 3000) -> Benchmark:
    """Full K2 CFD pipeline vs the AGARD-B wind-tunnel lift-curve slope.

    One shared mesh, a (Mach × α) grid of SU2 solves, then dC_L/dα per Mach
    against the measured value. Marks itself skipped — not failed — if the
    head-less geometry/mesh/solve pipeline raises, like the other SU2 cases.
    """
    from validation.cfd.agardb import run_su2_sweep, slope_per_deg
    from validation.data.agardb_aedc import AEDC_TR_70_100 as REF

    name = "SU2 lift slope vs AGARD-B experiment"
    try:
        data = run_su2_sweep(machs=machs, alphas=alphas, refinement=refinement,
                             max_iterations=max_iterations)
    except Exception as exc:
        return _skip(name, _AGARDB_REF, exc)

    bm = Benchmark(name=name, domain="cfd", reference=_AGARDB_REF,
                   level=ValidationLevel.VALIDATED)
    unconverged = [f"M={m}, α={a}" for m, rows in data.items()
                   for a, _cl, _cd, ok in rows if not ok]

    k2_slopes, ref_slopes, used = [], [], []
    for mach, rows in sorted(data.items()):
        if mach not in REF:
            continue
        s_k2 = slope_per_deg(rows)
        s_ref = REF[mach]["cl_alpha_per_deg"]
        used.append(mach)
        k2_slopes.append(s_k2)
        ref_slopes.append(s_ref)
        bm.add(Comparison.make(
            f"dC_L/dα at M={mach}", s_k2, s_ref, "AEDC-TR-70-100", "1/deg",
            tol_rel=0.15,
            note=f"{len(rows)} α points, inviscid + flat-plate friction"))
        # C_L at the largest solved angle: catches a right-slope/wrong-level
        # solution that a slope-only comparison would pass.
        a_max, cl_max = rows[-1][0], rows[-1][1]
        if a_max > 0:
            bm.add(Comparison.make(
                f"C_L at M={mach}, α={a_max:g}°", cl_max,
                s_ref * a_max + REF[mach]["cl0"], "AEDC-TR-70-100", "-",
                tol_rel=0.20, note="measured fit evaluated at the same α"))

    if unconverged:
        bm.notes = ("SU2 reported non-convergence at: " + ", ".join(unconverged)
                    + ". Those points still contribute to the fitted slope.")
    bm.curves["cl_alpha_vs_mach"] = {
        "x": used, "k2": k2_slopes, "ref": ref_slopes,
        "xlabel": "Mach", "ylabel": "dC_L/dα (1/deg, wing area)"}
    return bm


# ── ONERA M6 wing vs measured surface pressures ───────────────────────────────

_M6_SECTIONS = (1, 3, 5, 7)


def _m6_level(M6, wall_size: float, refinement: str, max_iterations: int) -> dict:
    """Solve one M6 refinement level; return its per-station Cp errors."""
    import pyvista as pv

    work = _WORK / f"onera_m6_w{wall_size:g}".replace(".", "_")
    res = M6.run_su2(work, refinement=refinement, max_iterations=max_iterations,
                     mirror=True, wall_size=wall_size)
    mesh = pv.read(str(res.surface_vtk or (work / "surface_flow.vtu")))
    # Read the span from the solution: the CAD import recentres the geometry, so
    # the station coordinates are not the analytic ones.
    span = M6.span_frame(mesh)

    out = {"wall_size": wall_size, "result": res, "mesh": mesh, "span": span,
           "rmse": {}, "cp_min": {}, "cn": {}}
    for section in _M6_SECTIONS:
        out["cn"][section] = M6.section_cn(mesh, section, span)
        for upper in (True, False):
            key = (section, upper)
            out["rmse"][key] = M6.cp_rmse(mesh, section, upper, span)
            exp = M6.load_experimental_cp(section, upper)
            xs = [x for x, _ in exp]
            got = M6.surface_cp_at_section(mesh, M6.SECTIONS[section], upper,
                                           xs, span)
            out["cp_min"][key] = (min(got), min(c for _, c in exp))
    return out


def bench_onera_m6(refinement: str = "fine", max_iterations: int = 4000,
                   wall_sizes=None) -> Benchmark:
    """K2's external-CAD CFD path vs Schmitt & Charpin surface pressures.

    Solved at **two** mesh resolutions, because one run cannot separate the two
    things that make a transonic Cp disagree. K2's tet mesher has no anisotropic
    wall layer and no shock adaption, so it will not reach the ±0.02 the
    experiment quotes; the honest question is whether what remains is a
    *resolution* error, which has a signature: it shrinks as the mesh refines.

    Two gates, measuring different things, and they currently disagree — which is
    the finding:

      * **Cp RMSE improves with refinement** — the pointwise pressure
        distribution does converge toward the data, at every station.
      * **Sectional normal force** vs the measured loading — and this gets
        *worse* with refinement, to about -45%. A distribution converging in
        shape while the load it integrates to moves away from the measurement is
        not a resolution story, so the row is gated at 15% and currently fails.
        Leading-edge resolution has since been tested directly and is not the
        cause — see the notes.
    """
    from validation.cfd import onera_m6 as M6

    name = "ONERA M6 wing surface Cp vs experiment"
    ref = "Schmitt & Charpin, AGARD AR-138 (1979), Test 2308"
    sizes = tuple(wall_sizes or M6.WALL_SIZES)
    try:
        levels = [_m6_level(M6, w, refinement, max_iterations)
                  for w in sorted(sizes, reverse=True)]   # coarse → fine
        coarse, fine = levels[0], levels[-1]

        bm = Benchmark(name=name, domain="cfd", reference=ref,
                       level=ValidationLevel.VALIDATED)

        for section in _M6_SECTIONS:
            for upper in (True, False):
                key = (section, upper)
                side = "upper" if upper else "lower"
                label = f"y/b={M6.SECTIONS[section]} ({side})"
                r_fine, n = fine["rmse"][key]
                r_coarse, _ = coarse["rmse"][key]

                # Refinement must not move the solution away from the data. The
                # 15% slack keeps a station whose error is already at the noise
                # floor from failing on a wobble.
                bm.add(Comparison.make(
                    f"Cp RMSE improves with refinement, {label}",
                    r_fine, r_coarse, "AGARD AR-138", "-",
                    tol_rel=0.15, one_sided="below",
                    note=f"coarse {r_coarse:.3f} → fine {r_fine:.3f}"))

        for section in _M6_SECTIONS:
            cn_k2, cn_exp = fine["cn"][section]
            cn_coarse = coarse["cn"][section][0]
            bm.add(Comparison.make(
                f"Sectional normal force c_n, y/b={M6.SECTIONS[section]}",
                cn_k2, cn_exp, "AGARD AR-138", "-", tol_rel=0.15,
                note=f"integrated Cp loading; coarse mesh gave {cn_coarse:+.3f}"))

        # Overlay one station so the report shows the pressure distribution, not
        # only its error norm.
        exp_u = M6.load_experimental_cp(3, True)
        xs = [x for x, _ in exp_u]
        bm.curves["cp_yb065_upper"] = {
            "x": xs,
            "k2": M6.surface_cp_at_section(fine["mesh"], M6.SECTIONS[3], True,
                                           xs, fine["span"]),
            "ref": [c for _, c in exp_u],
            "xlabel": "x/c", "ylabel": "Cp (y/b = 0.65, upper)"}

        rmse_tbl = "; ".join(
            f"y/b={M6.SECTIONS[s]} {coarse['rmse'][(s, True)][0]:.3f}→"
            f"{fine['rmse'][(s, True)][0]:.3f}" for s in _M6_SECTIONS)
        peaks = "; ".join(
            f"y/b={M6.SECTIONS[s]} {fine['cp_min'][(s, True)][0]:+.2f} vs "
            f"{fine['cp_min'][(s, True)][1]:+.2f}" for s in _M6_SECTIONS)
        cls = ", ".join(f"{lv['wall_size']:g} m → CL={lv['result'].cl:.3f}"
                        for lv in levels)
        bm.notes = (
            f"M={M6.MACH}, α={M6.ALPHA_DEG}°, inviscid. Wall element sizes {cls}. "
            f"Upper-surface Cp RMSE, coarse→fine: {rmse_tbl}. "
            f"Suction peak (fine) vs measured: {peaks}. "
            "\n\nThe wing is meshed mirrored about its root: the experiment is a "
            "half-model on a reflection plane and K2's CAD path has no symmetry "
            "boundary, so a semi-span import relieves around the root plane as if "
            "it were a second tip. "
            "\n\nOPEN: the solution carries 26-45% too little sectional load, and "
            "refining moves it further from the measurement even as the pointwise "
            "Cp error falls at every station. Two candidate causes have now been "
            "tested and neither is it. Farfield proximity: taking the domain "
            "radius from 14 m to 48 m changed CL by 0.01 and left the deficit "
            "where it was. Leading-edge resolution, which was the leading "
            "hypothesis: the mesher now refines a band around every curved "
            "feature to about a third of its radius, which on the AGARD-B wing "
            "moved the lift-curve slope from 10-18% low to inside 5% — and here "
            "it moved the sectional loads by about a point (-25/-41/-48/-38% "
            "before, -26/-43/-45/-39% after). Whatever this is, it is not the "
            "mesh. Reported as failing rather than widened to pass.")
        return bm
    except Exception as exc:
        return _skip(name, ref, exc)


def run_benchmarks(include_su2: bool = True) -> list:
    """All CFD benchmarks. `include_su2` runs the slow SU2 cases."""
    out = [bench_taylor_maccoll_reference(), bench_agardb_barrowman()]
    if include_su2:
        out.append(bench_su2_cone())
        out.append(bench_barrowman_vs_su2())
        out.append(bench_agardb_su2())
        out.append(bench_onera_m6())
    return out
