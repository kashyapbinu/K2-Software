"""
CFD post-processing gates.

These cover the numbers the visualization views auto-scale and threshold on,
which are easy to get wrong in ways that never raise: a colour bar that
describes the nose-tip slivers instead of the airframe, a "Gaussian" filter
that is really a box mean, a stagnation pressure computed with the
incompressible formula. All of them render something plausible-looking.

Everything here is analytic or built from a synthetic mesh, so it is fast and
needs no solver run.
"""
import numpy as np
import pyvista as pv
import pytest

from cfd.post_processing import (
    compute_derived_fields,
    field_percentiles,
    smooth_volume_field,
)


def _graded_surface():
    """A strip whose left end is meshed ~100x finer than its right end.

    Mimics the real failure mode: a nose tip carved into slivers while the
    body stays coarse. The field is 1.0 over the fine end and 0.0 over the
    coarse end, and the fine end holds only ~1% of the total AREA.
    """
    fine = np.linspace(0.0, 0.01, 101)
    coarse = np.linspace(0.01, 1.0, 11)[1:]
    xs = np.concatenate([fine, coarse])
    grid = pv.StructuredGrid(
        *np.meshgrid(xs, np.array([0.0, 0.05]), np.array([0.0]), indexing="ij")
    )
    surf = grid.extract_surface(algorithm="dataset_surface")
    surf["f"] = (surf.points[:, 0] <= 0.01).astype(np.float64)
    return surf


@pytest.mark.cfd
def test_field_percentiles_weighs_by_area_not_point_count():
    surf = _graded_surface()

    # The fine end is ~1% of the area but ~90% of the points, so the two
    # estimators must disagree, and only the weighted one tracks the area.
    unweighted_median = field_percentiles(surf, "f", 50.0, weighted=False)[0]
    weighted_median = field_percentiles(surf, "f", 50.0, weighted=True)[0]

    assert unweighted_median == pytest.approx(1.0), (
        "point-counted median should be captured by the refined end"
    )
    assert weighted_median == pytest.approx(0.0), (
        "area-weighted median should follow the bulk of the surface"
    )


@pytest.mark.cfd
def test_field_percentiles_matches_numpy_on_a_uniform_mesh():
    """With equal-area cells the weighting must not change the answer.

    The field has to be spatially smooth for this comparison to mean
    anything: the weighted path averages point data onto cells first, which
    would legitimately narrow the tails of an uncorrelated per-point field.
    """
    surf = pv.Plane(i_resolution=40, j_resolution=40).triangulate()
    surf["f"] = surf.points[:, 0].astype(np.float64)

    for q in (5.0, 50.0, 95.0):
        weighted = field_percentiles(surf, "f", q, weighted=True)[0]
        plain = field_percentiles(surf, "f", q, weighted=False)[0]
        assert weighted == pytest.approx(plain, abs=0.02)


@pytest.mark.cfd
def test_field_percentiles_degrades_safely():
    surf = pv.Plane().triangulate()
    surf["vec"] = np.zeros((surf.n_points, 3))
    assert np.isnan(field_percentiles(surf, "missing", 50.0)[0])
    assert np.isnan(field_percentiles(surf, "vec", 50.0)[0])
    assert np.isnan(field_percentiles(None, "f", 50.0)[0])


@pytest.mark.cfd
def test_smoothing_kernel_scales_with_mesh_spacing():
    """sigma is a multiple of local spacing, so it must actually bite.

    As an absolute length it silently did nothing: every neighbour of every
    point sat far inside a 1.5-unit kernel, all weights came out 1.0, and the
    filter degenerated to an unweighted box mean no matter what was passed.
    """
    grid = pv.ImageData(dimensions=(21, 21, 21), spacing=(0.001,) * 3)
    spike = np.zeros(grid.n_points)
    spike[grid.n_points // 2] = 1.0
    grid["f"] = spike

    peaks = {}
    for sigma in (0.3, 1.5):
        work = grid.copy()
        smooth_volume_field(work, "f", sigma=sigma, k=12)
        peaks[sigma] = float(np.asarray(work["f"]).max())

    assert peaks[0.3] > peaks[1.5], "a tighter kernel must preserve more peak"
    assert peaks[1.5] < 0.5, "a wide kernel must actually smooth"


@pytest.mark.cfd
def test_total_pressure_uses_the_compressible_relation():
    """P_total must be isentropic, not P + 0.5*rho*V^2."""
    mesh = pv.ImageData(dimensions=(4, 4, 4))
    n = mesh.n_points
    gamma, R, T, M = 1.4, 287.05, 268.65, 0.8
    p = np.full(n, 70108.27)
    rho = p / (R * T)
    speed = M * np.sqrt(gamma * R * T)

    mesh["Pressure"] = p
    mesh["Density"] = rho
    mesh["Temperature"] = np.full(n, T)
    mesh["Velocity"] = np.tile([speed, 0.0, 0.0], (n, 1))
    mesh["Speed"] = np.full(n, speed)
    mesh["Mach"] = np.full(n, M)

    mesh = compute_derived_fields(mesh, gamma=gamma, r_gas=R)

    expected = p * (1.0 + (gamma - 1.0) / 2.0 * M ** 2) ** (gamma / (gamma - 1.0))
    incompressible = p + 0.5 * rho * speed ** 2
    got = np.asarray(mesh["P_total"], dtype=float)

    assert got == pytest.approx(expected, rel=1e-5)
    # The two formulas must be distinguishable here, or the test proves nothing.
    assert abs(expected[0] - incompressible[0]) > 0.01 * p[0]


@pytest.mark.cfd
def test_solver_q_criterion_is_not_overwritten():
    """Missing Lambda2 must not cause SU2's own Q_Criterion to be replaced."""
    mesh = pv.ImageData(dimensions=(6, 6, 6))
    rng = np.random.default_rng(1)
    mesh["Velocity"] = rng.normal(size=(mesh.n_points, 3))
    sentinel = np.arange(mesh.n_points, dtype=np.float32)
    mesh["Q_Criterion"] = sentinel.copy()

    mesh = compute_derived_fields(mesh)

    assert np.array_equal(np.asarray(mesh["Q_Criterion"]), sentinel), (
        "solver-supplied Q_Criterion must survive"
    )
    assert "Lambda2" in mesh.array_names, "the missing field should still be filled in"


# ── Wall resolution and solver configuration ─────────────────────────────────

@pytest.mark.cfd
def test_predict_wall_yplus_separates_the_regimes():
    """The predictor has to tell a resolved wall from an unresolved one."""
    from cfd.boundary_layer import predict_wall_yplus

    # Measured M=0.8 case: rho 0.909, V 262.9, mu 1.694e-5, L 1.68.
    flow = dict(velocity=262.86, density=0.9091, viscosity=1.694e-5,
                ref_length=1.68)

    coarse = predict_wall_yplus(wall_spacing=0.0149, **flow)   # tet-only mesh
    assert coarse["y_plus"] > 300
    assert coarse["regime"] == "under-resolved"
    assert not coarse["wall_resolved"]

    # Meshing to the reported requirement must actually land at y+ ~ 1.
    resolved = predict_wall_yplus(
        wall_spacing=coarse["spacing_for_yplus_1"], **flow)
    assert resolved["y_plus"] == pytest.approx(1.0, rel=0.05)
    assert resolved["wall_resolved"]

    # y+ scales linearly with spacing.
    half = predict_wall_yplus(wall_spacing=0.0149 / 2, **flow)
    assert half["y_plus"] == pytest.approx(coarse["y_plus"] / 2, rel=1e-6)


@pytest.mark.cfd
def test_predict_wall_yplus_rejects_nonsense_input():
    from cfd.boundary_layer import predict_wall_yplus
    bad = predict_wall_yplus(wall_spacing=0.0, velocity=100.0, density=1.0,
                             viscosity=1e-5, ref_length=1.0)
    assert np.isnan(bad["y_plus"])
    assert bad["regime"] == "unknown"


@pytest.mark.cfd
def test_su2_config_activates_the_cauchy_criterion(tmp_path):
    """CONV_CAUCHY_* is inert unless a Cauchy field is named in CONV_FIELD.

    The pair sat in the template for a long time doing nothing because
    CONV_FIELD was never written and SU2 defaulted to RMS_DENSITY alone.
    """
    from cfd.solvers.base import CFDConfig
    from cfd.solvers.su2_solver import SU2Solver

    cfg = CFDConfig(mach=0.8, altitude_m=3000.0, work_dir=tmp_path,
                    geometry_dict={"max_diameter": 0.155, "length": 1.68})
    solver = SU2Solver(cfg)
    text = solver.generate_case().read_text()

    assert "CONV_FIELD=" in text, "no CONV_FIELD means the Cauchy settings do nothing"
    conv_line = next(l for l in text.splitlines()
                     if l.strip().startswith("CONV_FIELD="))
    assert "DRAG" in conv_line, "a Cauchy field must be monitored, not just a residual"
    assert "CONV_CAUCHY_ELEMS=" in text and "CONV_CAUCHY_EPS=" in text


@pytest.mark.cfd
def test_su2_viscous_config_does_not_enable_wall_functions(tmp_path):
    """Wall functions must stay off, and the reason is measured, not cautious.

    STANDARD_WALL_FUNCTION is the obvious-looking fix for an unresolved wall
    and it fails quietly here, which is exactly why this needs a guard rather
    than a comment. From a converged restart on a measured M=0.8 case SU2's
    wall-coefficient solve failed on 5554 of 6318 wall points (88%), pinning
    y+ at exactly 30.0 and dropping skin friction to a median 3.1e-8 there —
    no wall shear at all. On the 12% that converged, Cf was 1.46e-3 against a
    flat-plate 1.9e-3, so the surface ends up part correct and part blank with
    nothing in the output saying which is which. The headline numbers move the
    right way (drag 0.079 -> 0.137, wall-temperature violation 19.1% -> 4.2%)
    partly *because* the viscous heating is missing on most of the body.

    Tuning made it worse: WALLMODEL_MAXITER=1000 / RELFAC=0.1 / MINYPLUS=2.0
    raised the failure rate to 6222/6318 (98.5%) and held it there. The first
    cell sits near y+ 3500, outside the 30 < y+ < 300 band a wall function can
    invert at all.

    If prism layers ever land (cfd/meshing.py step 7), re-measure before
    deleting this test.
    """
    from cfd.solvers.base import CFDConfig
    from cfd.solvers.su2_solver import SU2Solver

    cfg = CFDConfig(mach=0.8, altitude_m=3000.0, work_dir=tmp_path,
                    turbulence_model="SST",
                    geometry_dict={"max_diameter": 0.155, "length": 1.68})
    text = SU2Solver(cfg).generate_case().read_text()

    assert "SOLVER= RANS" in text, "this guard is only meaningful for a viscous run"
    assert "MARKER_HEATFLUX=" in text, "viscous walls must stay no-slip"

    active = [l for l in text.splitlines()
              if l.strip().startswith("MARKER_WALL_FUNCTIONS")]
    assert not active, (
        "wall functions are enabled — on this tet-only mesh that leaves 88% of "
        "the wall with zero skin friction while still reporting a confident "
        "drag number. See the docstring above before changing this."
    )
