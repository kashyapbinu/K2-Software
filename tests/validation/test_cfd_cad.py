"""
Regression gates for the external-CAD CFD pipeline and the surface post-processing.

Every check here corresponds to a bug that shipped. They are cheap on purpose —
the expensive SU2 gates live in test_cfd.py — because the failures they catch
are silent ones: a mesh that comes out empty, coefficients normalised by a body
a thousand times too big, force arrows drawn backwards. None of those raise, and
none were caught by the SU2 benchmarks, which only ever exercised the parametric
rocket.
"""
import math
from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cylinder_surface():
    """Capped cylinder along +X carrying a physically-signed pressure field.

    High pressure upstream, low aft, so the net pressure force is a DRAG along
    +X. That known sign is what the force-vector tests key off.
    """
    s = (pv.Cylinder(direction=(1, 0, 0), radius=0.05, height=0.4,
                     resolution=120, capping=True)
         .triangulate().extract_surface())
    x = s.points[:, 0]
    span = x.max() - x.min()
    s.point_data["Pressure"] = (
        101325.0 + 4000.0 * (1.0 - (x - x.min()) / span) - 1000.0
    )
    return s


P_INF = 101325.0
Q_INF = 5000.0
REF_A = math.pi * 0.05 ** 2


# ── units ────────────────────────────────────────────────────────────────────

@pytest.mark.cfd
def test_resolve_units_auto_detects_millimetres():
    """A model 1000x too large is millimetres, not a 816 m object."""
    from cfd.external_geometry import resolve_units

    name, scale, auto = resolve_units(816.0, "auto")
    assert (name, scale, auto) == ("mm", 0.001, True)

    name, scale, auto = resolve_units(0.8, "auto")
    assert (name, scale, auto) == ("m", 1.0, True)


@pytest.mark.cfd
@pytest.mark.parametrize("unit,scale", [
    ("m", 1.0), ("mm", 0.001), ("cm", 0.01), ("in", 0.0254), ("ft", 0.3048),
])
def test_resolve_units_explicit_wins(unit, scale):
    from cfd.external_geometry import resolve_units

    name, got, auto = resolve_units(816.0, unit)
    assert name == unit and got == pytest.approx(scale) and auto is False


@pytest.mark.cfd
def test_analyze_cad_scales_to_metres(tmp_path):
    """Measurements must be SI: a mm-authored file is scaled before measuring.

    Left unscaled, reference area is out by 1e6 and Reynolds by 1e3, silently.
    """
    from cfd.external_geometry import analyze_cad

    stl = tmp_path / "mm_sphere.stl"
    # radius 200 "units"; as millimetres that is a 0.4 m diameter sphere.
    pv.Sphere(radius=200.0, theta_resolution=40,
              phi_resolution=40).triangulate().save(stl)

    info = analyze_cad(stl, flow_axis="x", work_dir=tmp_path, units="mm")
    assert info.units == "mm" and info.unit_scale == pytest.approx(0.001)
    assert info.length == pytest.approx(0.4, rel=0.02)
    assert info.frontal_area == pytest.approx(math.pi * 0.2 ** 2, rel=0.02)


# ── nose shapes ──────────────────────────────────────────────────────────────

@pytest.mark.cfd
@pytest.mark.parametrize("shape", [
    "Conical", "Ogive", "Elliptical", "Parabolic", "Haack (LD)",
])
def test_nose_profile_is_a_valid_meridian(shape):
    from visualization.viewer_3d import nose_profile

    zs, rs = nose_profile(shape, 0.2, 0.05, n=101)
    assert rs[0] == pytest.approx(0.05)         # base at full radius
    assert rs[-1] == pytest.approx(0.0, abs=1e-9)   # closes at the tip
    assert np.all(np.diff(rs) <= 1e-12)         # monotone, no bulges
    assert zs[0] == 0.0 and zs[-1] == pytest.approx(0.2)


@pytest.mark.cfd
def test_nose_profile_shapes_actually_differ():
    """Every shape used to render as an ogive — the parameter was ignored."""
    from visualization.viewer_3d import nose_profile

    mids = {}
    for shape in ("Conical", "Ogive", "Elliptical", "Parabolic", "Haack (LD)"):
        zs, rs = nose_profile(shape, 0.2, 0.05, n=101)
        mids[shape] = float(np.interp(0.1, zs, rs))
    assert len(set(round(v, 5) for v in mids.values())) == len(mids)
    # A cone is exactly linear: half way along, half the radius.
    assert mids["Conical"] == pytest.approx(0.025, abs=1e-6)


@pytest.mark.cfd
def test_nose_profile_agrees_across_modules():
    """Viewer, mission view and CFD exporter must draw one shape."""
    from visualization.viewer_3d import nose_profile
    from visualization.mission.rocket_mesh import _nose_profile as mission
    from cfd.geometry_exporter import _nose_profile as cfd

    for shape in ("Conical", "Ogive", "Elliptical", "Parabolic", "Haack (LD)"):
        a = nose_profile(shape, 0.2, 0.05, n=40)
        b = mission(shape, 0.2, 0.05, n=40)
        c = cfd(shape, 0.2, 0.05, 40)
        assert np.allclose(a[1], b[1]), f"{shape}: mission view disagrees"
        assert np.allclose(a[1], c[1]), f"{shape}: CFD exporter disagrees"


# ── axial profile → CFD geometry ─────────────────────────────────────────────

@pytest.mark.cfd
def test_cfd_profile_keeps_a_cone_exact():
    """The Taylor-Maccoll gate depends on a conical nose staying a true cone."""
    from validation.cfd.cone_geometry import cone_assembly
    from cfd.geometry_exporter import cfd_profile

    prof = cfd_profile(cone_assembly(10.0, 0.5))
    assert len(prof) == 2, "a cone must reduce to two stations, not a polyline"
    (x0, _), (x1, r1) = prof
    assert x0 == pytest.approx(0.0, abs=1e-9)
    assert x1 == pytest.approx(0.5)
    assert r1 == pytest.approx(0.5 * math.tan(math.radians(10.0)), rel=1e-6)


@pytest.mark.cfd
def test_cfd_profile_preserves_a_boattail():
    """A transition used to be flattened into body tube, inventing a flat base.

    That lands straight on the base-drag integral, so it has to survive.
    """
    from core.components import RocketAssembly, NoseCone, BodyTube, Transition
    from cfd.geometry_exporter import cfd_profile

    asm = RocketAssembly()
    nose = NoseCone(); nose.shape = "Ogive"; nose.length = 0.20
    nose.diameter = 0.10
    tube = BodyTube(); tube.length = 0.60; tube.outer_diameter_val = 0.10
    boat = Transition(); boat.length = 0.10
    boat.fore_diameter = 0.10; boat.aft_diameter = 0.06
    for comp in (nose, tube, boat):
        asm.add_component(asm.stages[0], comp)

    prof = cfd_profile(asm)
    assert prof[-1][0] == pytest.approx(0.9, rel=1e-3)
    assert prof[-1][1] == pytest.approx(0.03, rel=1e-3), "boattail was flattened"
    # Mid-body still full radius.
    xs = np.array([p[0] for p in prof])
    rs = np.array([p[1] for p in prof])
    assert float(np.interp(0.5, xs, rs)) == pytest.approx(0.05, rel=1e-3)


@pytest.mark.cfd
def test_extract_cfd_geometry_does_not_invent_fins():
    """A finless design was given four fabricated fins, and a bodyless one
    produced a zero-extent fin that killed the mesher outright."""
    from validation.cfd.cone_geometry import cone_assembly
    from cfd.geometry_exporter import extract_cfd_geometry

    geo = extract_cfd_geometry(cone_assembly(10.0, 0.5))
    assert geo["fin_count"] == 0
    assert geo["fin_height"] == 0.0 and geo["fin_root"] == 0.0


# ── projected area ───────────────────────────────────────────────────────────

@pytest.mark.cfd
def test_projected_frontal_area_matches_analytic():
    from cfd.external_geometry import projected_frontal_area

    cyl = pv.Cylinder(direction=(1, 0, 0), radius=0.05, height=0.5,
                      resolution=200).triangulate()
    assert projected_frontal_area(cyl, 0) == pytest.approx(
        math.pi * 0.05 ** 2, rel=0.01)


@pytest.mark.cfd
def test_projected_frontal_area_takes_the_union_not_the_sum():
    """Overlapping bodies must project as their union.

    The cheap 0.5*sum(|n.x|*A) formula counts hidden rear-facing area too and
    reports two overlapping cylinders as two separate discs.
    """
    from cfd.external_geometry import projected_frontal_area

    a = pv.Cylinder(center=(0, 0, 0), direction=(1, 0, 0), radius=0.05,
                    height=0.4, resolution=120).triangulate()
    b = pv.Cylinder(center=(0, 0.03, 0), direction=(1, 0, 0), radius=0.05,
                    height=0.4, resolution=120).triangulate()
    union = projected_frontal_area((a + b).triangulate(), 0)
    two_discs = 2.0 * math.pi * 0.05 ** 2
    assert union < 0.8 * two_discs


# ── surface sampling helpers ─────────────────────────────────────────────────

@pytest.mark.cfd
def test_point_areas_matches_the_reference_loop():
    """Vectorised replacement for a per-cell Python loop; must be identical."""
    from cfd.post_processing import _point_areas

    for mesh in (pv.Sphere(theta_resolution=20, phi_resolution=20).triangulate(),
                 pv.Cube().extract_surface()):
        got = _point_areas(mesh)
        cell_areas = np.asarray(
            mesh.compute_cell_sizes(length=False, area=True,
                                    volume=False)["Area"])
        acc = np.zeros(mesh.n_points)
        cnt = np.zeros(mesh.n_points)
        for ci in range(mesh.n_cells):
            for pid in mesh.get_cell(ci).point_ids:
                acc[pid] += cell_areas[ci]
                cnt[pid] += 1
        want = acc / np.where(cnt == 0, 1, cnt)
        assert np.allclose(got, want)


@pytest.mark.cfd
def test_voxel_representatives_are_unique_and_in_range():
    """The old multiplicative voxel hash aliased once an index reached 1009."""
    from cfd.post_processing import _voxel_representatives

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, size=(5000, 3))
    bounds = (-1, 1, -1, 1, -1, 1)
    sel = _voxel_representatives(pts, bounds, 0.1)
    assert len(sel) == len(set(sel.tolist())), "duplicate representatives"
    assert sel.min() >= 0 and sel.max() < len(pts)


# ── drag decomposition ───────────────────────────────────────────────────────

@pytest.mark.cfd
@pytest.mark.parametrize("aoa", [0.0, 5.0, 15.0, 30.0])
def test_base_drag_uses_the_same_axis_as_the_total(cylinder_surface, tmp_path, aoa):
    """cd_base is subtracted from a wind-axis cd_pressure, so it must be a
    wind-axis quantity too. Projecting it on the body axis instead made the
    forebody/wave split drift with angle of attack."""
    from cfd.drag_decomposition import base_drag_from_surface
    from cfd.solvers.su2_solver import _integrate_surface_forces

    f = tmp_path / "surf.vtk"
    cylinder_surface.save(f)

    total = _integrate_surface_forces(f, p_inf=P_INF, q_inf=Q_INF,
                                      ref_area=REF_A, aoa_deg=aoa)
    # Widen the base mask to the whole body: it must then reproduce the total.
    everything = base_drag_from_surface(f, p_inf=P_INF, q_inf=Q_INF,
                                        ref_area=REF_A, aoa_deg=aoa,
                                        base_angle_deg=180.0)
    assert everything["cd_base"] == pytest.approx(total["cd_pressure"], abs=1e-12)


@pytest.mark.cfd
@pytest.mark.parametrize("cd_p,cd_b,mach", [
    (0.2, 0.05, 2.0), (0.2, 0.05, 0.3), (-0.01, -0.004, 2.0),
])
def test_pressure_split_is_exact(cd_p, cd_b, mach):
    """cd_base + cd_forebody_pressure == cd_pressure, always."""
    from cfd.drag_decomposition import split_pressure_drag

    out = split_pressure_drag(cd_p, cd_b, mach)
    assert cd_b + out["cd_forebody_pressure"] == pytest.approx(cd_p, abs=1e-15)
    assert out["cd_wave"] >= 0.0
    if mach < 0.8:
        assert out["cd_wave"] == 0.0


# ── force glyphs ─────────────────────────────────────────────────────────────

@pytest.mark.cfd
def test_force_glyphs_point_into_the_surface(cylinder_surface):
    """Pressure pushes: dF = -(P - P_inf)*n_outward*dA.

    The minus was missing, so every arrow was drawn reversed — a stagnation
    point appeared to blow the nose forward and the glyphs summed to thrust.
    The scalar coefficients were unaffected, so only the picture was wrong.
    """
    from cfd.post_processing import compute_force_vectors

    fv = compute_force_vectors(cylinder_surface, P_INF, Q_INF, n_samples=400)
    assert fv is not None and fv.n_points > 0

    net = np.asarray(fv["ForceVectorTrue"]).sum(axis=0)
    assert net[0] > 0, "net pressure force must be drag (+X), not thrust"

    vec = np.asarray(fv["ForceVectorTrue"])
    nrm = np.asarray(fv["NormalDirection"])
    gauge = np.asarray(fv["GaugePressure"])
    pushing = gauge > 0
    if pushing.any():
        dots = np.einsum("ij,ij->i", vec[pushing], nrm[pushing])
        assert np.all(dots <= 0), "above-ambient pressure must act inward"


@pytest.mark.cfd
def test_pressure_shear_decomposition_has_the_same_sign(cylinder_surface):
    from cfd.post_processing import compute_pressure_shear_vectors

    press, _shear = compute_pressure_shear_vectors(
        cylinder_surface, P_INF, Q_INF, n_samples=400)
    assert press is not None
    assert np.asarray(press["ForceVector"]).sum(axis=0)[0] > 0


# ── convergence reporting ────────────────────────────────────────────────────

def _solver_with_history(tmp_path, rows, max_iterations=100):
    """Build an SU2Solver whose work_dir holds a synthetic history.csv."""
    from cfd.solvers.base import CFDConfig
    from cfd.solvers.su2_solver import SU2Solver

    cfg = CFDConfig(work_dir=tmp_path, max_iterations=max_iterations)
    lines = ["\"Inner_Iter\",\"rms[Rho]\",\"CD\",\"CL\",\"CMy\""]
    lines += [f"{i},{rho},{cd},{cl},{cm}" for i, rho, cd, cl, cm in rows]
    (tmp_path / "history.csv").write_text("\n".join(lines), encoding="utf-8")
    return SU2Solver(cfg)


@pytest.mark.cfd
def test_diverged_run_is_not_reported_converged(tmp_path):
    """Residual climbing away from its best is divergence, wherever it stops.

    'Stopped before the iteration cap' used to be sufficient on its own, and
    that flag gates injection into the sim engine.
    """
    rows = [(i, -6.0 + 0.5 * i, 0.4, 0.0, 0.0) for i in range(20)]
    result = _solver_with_history(tmp_path, rows).parse_results()
    assert result.converged is False


@pytest.mark.cfd
def test_nonphysical_run_is_not_reported_converged(tmp_path):
    rows = [(i, -3.0, 0.4, 0.0, 0.0) for i in range(10)]
    rows.append((10, float("nan"), float("nan"), 0.0, 0.0))
    result = _solver_with_history(tmp_path, rows).parse_results()
    assert result.converged is False


@pytest.mark.cfd
def test_residual_floor_still_counts_as_converged(tmp_path):
    rows = [(i, -1.0 - 0.6 * i, 0.4, 0.0, 0.0) for i in range(20)]
    result = _solver_with_history(tmp_path, rows).parse_results()
    assert result.converged is True


# ── geometry repair (wrap) ───────────────────────────────────────────────────

@pytest.mark.cfd
def test_wrap_produces_a_single_watertight_shell():
    """The wrap exists to survive geometry the normal routes cannot mesh, so
    the one thing it must guarantee is a closed manifold shell."""
    from cfd.external_geometry import wrap_surface, _count_open_edges

    a = pv.Sphere(center=(0, 0, 0), radius=0.05,
                  theta_resolution=30, phi_resolution=30).triangulate()
    b = pv.Sphere(center=(0.06, 0, 0), radius=0.05,
                  theta_resolution=30, phi_resolution=30).triangulate()
    dirty = (a + b).triangulate()          # two interpenetrating bodies

    wrap = wrap_surface(dirty, cell=0.006, offset=0.003)
    assert _count_open_edges(wrap) == 0
    assert wrap.n_cells > 0
    assert wrap.connectivity("all").n_points == wrap.n_points
    assert wrap.volume > 0


@pytest.mark.cfd
def test_wrap_rejects_an_unaffordable_grid():
    from cfd.external_geometry import wrap_surface, CADImportError

    sphere = pv.Sphere(radius=1.0).triangulate()
    with pytest.raises(CADImportError, match="coarser"):
        wrap_surface(sphere, cell=1e-4, offset=1e-4)


# ── end-to-end mesh ──────────────────────────────────────────────────────────

@pytest.mark.cfd
@pytest.mark.slow
def test_external_cad_mesh_end_to_end(tmp_path):
    """A watertight STL must reach a real SU2 mesh with both markers.

    Guards the whole discrete route: an empty mesh used to be written as a
    0-byte .su2 and reported as success.
    """
    from cfd.external_geometry import analyze_cad
    from cfd.meshing import build_wind_tunnel_mesh

    stl = tmp_path / "body.stl"
    pv.Sphere(radius=0.05, theta_resolution=60,
              phi_resolution=60).triangulate().save(stl)

    info = analyze_cad(stl, flow_axis="x", work_dir=tmp_path)
    out = build_wind_tunnel_mesh(
        stl_path=None, output_path=tmp_path / "m.su2", refinement="coarse",
        external_cad=stl, flow_axis="x", cad_info=info.as_dict(),
    )
    assert out.is_file() and out.stat().st_size > 100_000
    text = out.read_text(errors="replace")
    assert "rocket_wall" in text and "farfield" in text
    assert "NELEM= 0" not in text
