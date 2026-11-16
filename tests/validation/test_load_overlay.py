"""
The replay's Loads overlay paints a solved result on the flying rocket.

It must show the field where it is on the vehicle (station from the nose tip,
the frame both solvers use), keep a fin's pressure out of the body's, scale
CFD pressure with the dynamic pressure of the instant shown, and draw nothing
at all for an analysis that was not run.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from core import load_overlay as LO


def _body_points(n_x=60, n_theta=24, length=2.0, radius=0.05):
    x = np.repeat(np.linspace(0.0, length, n_x), n_theta)
    th = np.tile(np.linspace(0, 2 * np.pi, n_theta, endpoint=False), n_x)
    return np.column_stack([x, radius * np.cos(th), radius * np.sin(th)])


def test_a_pressure_profile_follows_the_field_along_the_body():
    pts = _body_points()
    cp = 1.0 - pts[:, 0]                      # +1 at the nose, -1 at the tail
    prof = LO.profile_from_points(pts, cp, mach=0.8, bins=20)
    assert prof.kind == LO.CFD_PRESSURE and prof.scales_with_q
    assert prof.source == "CFD at Mach 0.8"
    assert prof.at([0.5])[0] == pytest.approx(0.5, abs=0.06)
    assert prof.at([1.5])[0] == pytest.approx(-0.5, abs=0.06)
    assert np.all(np.diff(prof.x) > 0)


def test_cfd_pressure_scales_with_the_dynamic_pressure_of_the_instant():
    pts = _body_points()
    prof = LO.profile_from_points(pts, np.full(len(pts), 0.4), bins=10)
    assert prof.at([1.0], q=20_000.0)[0] == pytest.approx(8_000.0)   # Cp * q, Pa
    assert prof.at([1.0], q=0.0)[0] == 0.0                            # on the pad
    assert prof.limits(q_max=20_000.0) == pytest.approx((-8_000.0, 8_000.0))


def test_fin_surfaces_do_not_leak_into_the_body_pressure():
    """Body Cp is 0.1 everywhere; the fins (outside the tube, aft) carry -2.
    Averaged together the aft body would read about -0.5."""
    body = _body_points()
    fx = np.repeat(np.linspace(1.7, 1.95, 12), 30)
    fr = np.tile(np.linspace(0.06, 0.20, 30), 12)
    fins = np.column_stack([fx, fr, np.zeros_like(fr)])
    pts = np.vstack([body, fins])
    cp = np.concatenate([np.full(len(body), 0.1), np.full(len(fins), -2.0)])
    prof = LO.profile_from_points(pts, cp, bins=40)
    assert prof.at([1.8])[0] == pytest.approx(0.1, abs=0.02)


def test_a_profile_is_held_at_its_ends_not_extrapolated():
    pts = _body_points(length=1.0)
    prof = LO.profile_from_points(pts, pts[:, 0].copy(), bins=10)
    assert prof.at([5.0])[0] == pytest.approx(prof.value[-1])
    assert prof.at([-1.0])[0] == pytest.approx(prof.value[0])


def test_fe_stress_keeps_the_highest_element_at_each_station():
    fem = SimpleNamespace(element_stresses=[(1.0, 30e6), (0.0, 5e6), (1.0, 80e6),
                                            (2.0, 10e6), (0.5, float("nan"))])
    prof = LO.fe_stress_profile(fem, "Max-Q")
    assert prof.kind == LO.FE_STRESS and not prof.scales_with_q
    assert prof.source == "FE static, Max-Q"
    assert list(prof.x) == [0.0, 1.0, 2.0]
    assert prof.at([1.0])[0] == pytest.approx(80e6)
    assert prof.at([1.0], q=12345.0)[0] == pytest.approx(80e6)   # not a function of q
    assert prof.limits() == pytest.approx((0.0, 80e6))


def test_an_analysis_that_was_not_run_gives_nothing_to_draw():
    assert LO.cfd_pressure_profile("") is None
    assert LO.cfd_pressure_profile("no/such/surface_flow.vtu") is None
    assert LO.fe_stress_profile(None) is None
    assert LO.fe_stress_profile(SimpleNamespace(element_stresses=[])) is None
    assert LO.profile_from_points(np.zeros((3, 3)), np.zeros(3)) is None


def test_a_surface_file_is_read_and_measured_from_the_nose(tmp_path):
    pv = pytest.importorskip("pyvista")
    pts = _body_points(length=1.5)
    pts[:, 0] += 0.25                         # a file whose nose is not at x = 0
    cloud = pv.PolyData(pts)
    cloud["Pressure_Coefficient"] = (pts[:, 0] - 0.25) / 1.5      # 0 nose -> 1 tail
    path = tmp_path / "surface_flow.vtp"
    cloud.save(str(path))
    prof = LO.cfd_pressure_profile(path, mach=1.2, bins=15)
    assert prof is not None and prof.source == "CFD at Mach 1.2"
    assert prof.x[0] < 0.1 and prof.x[-1] == pytest.approx(1.5, abs=0.1)
    assert prof.at([0.75])[0] == pytest.approx(0.5, abs=0.06)


def test_the_engine_keeps_a_result_until_another_project_is_opened():
    from core.rocket_state import RocketState, RocketStateEngine
    e = RocketStateEngine()
    e.analysis_done("cfd", "Cd 0.4", result="the result")
    assert e.results["cfd"] == "the result"
    e.analysis_done("cfd", "Cd 0.4")                 # no object given: keeps the last
    assert e.results["cfd"] == "the result"
    e.set_state(RocketState())
    assert e.results == {}
