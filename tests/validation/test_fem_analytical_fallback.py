"""
The no-CalculiX fallback must analyse the fins the rocket has.

Before: with the ccx binary missing, the fin-root stress came from a fin the
fallback made up (span 0.8 D, chord 0.12 L, three of them), with a root
section sized from the AIRFRAME wall thickness; with no airspeed it was 30 %
of the axial stress. A finless rocket got the same fin stress as a finned
one, changing the fin thickness changed nothing, the fin stress was added to
the airframe's bending number, and the fin was judged against the airframe's
yield strength.
"""
import math
from dataclasses import replace

import pytest

from core.components import BodyTube, NoseCone, RocketAssembly, TrapezoidalFinSet
from structures.solvers.base import FEMConfig, FEMResult, LoadCase, get_structural_material
from structures.solvers.ccx_solver import CalculiXSolver
from structures.workstation import fin_root_bending
from validation.cases.rocket_canonical import canonical_assembly

_KT, _DAF = 1.8, 1.3          # the fallback's detail factor and Max-Q gust factor
_Q, _AOA = 31.4e3, 3.0


def _max_q():
    return LoadCase.max_q(thrust=500.0, q_dyn=_Q, mach=0.8, alt=3000.0, aoa=_AOA)


def _max_thrust(mach=0.3):
    return LoadCase.max_thrust(thrust=1500.0, angle_of_attack_deg=2.0, mach=mach,
                               altitude_m=500.0)


def _fins(asm):
    return next(c for c in asm.all_components() if isinstance(c, TrapezoidalFinSet))


def _finless():
    """The canonical airframe with no fin set."""
    asm = RocketAssembly()
    stage = asm.stages[0]
    nose = NoseCone()
    nose.shape, nose.length, nose.diameter, nose.wall_thickness = "Ogive", 0.40, 0.102, 0.0025
    asm.add_component(stage, nose)
    tube = BodyTube()
    tube.length, tube.outer_diameter_val, tube.inner_diameter = 1.60, 0.102, 0.097
    asm.add_component(stage, tube)
    return asm


def _fallback(tmp_path, asm, lc, material="Aluminum 6061-T6"):
    """A static result as the Structures workspace gets it without ccx."""
    solver = CalculiXSolver(FEMConfig(material_name=material, load_case=lc,
                                      work_dir=tmp_path, assembly=asm))
    solver._ccx_exe = None
    solver._material = get_structural_material(material)
    return solver.parse_results()


@pytest.mark.parametrize("case", [_max_q, _max_thrust])
def test_a_finless_rocket_has_no_fin_stress(tmp_path, case):
    r = _fallback(tmp_path, _finless(), case())
    assert "Fins" not in r.component_results
    assert r.max_von_mises == pytest.approx(r.component_results["Airframe"]["max_stress"])


def test_fin_stress_is_that_of_the_fins_drawn(tmp_path):
    asm = canonical_assembly()
    f = _fins(asm)
    r = _fallback(tmp_path, asm, _max_q())
    _force, root = fin_root_bending(f.height, f.root_chord, f.tip_chord, f.thickness,
                                    _Q, math.radians(_AOA))
    assert root > 1e6
    assert r.component_results["Fins"]["max_stress"] == pytest.approx(root * _KT * _DAF)


def test_fin_thickness_changes_the_fin_stress(tmp_path):
    thin, thick = canonical_assembly(), canonical_assembly()
    _fins(thick).thickness = 2 * _fins(thin).thickness
    a = _fallback(tmp_path, thin, _max_q()).component_results["Fins"]["max_stress"]
    b = _fallback(tmp_path, thick, _max_q()).component_results["Fins"]["max_stress"]
    assert a == pytest.approx(4.0 * b)          # root section modulus ∝ t²


def test_fin_is_judged_against_its_own_material(tmp_path):
    """Plywood fins on an aluminium tube fail as plywood."""
    asm = canonical_assembly()
    f = _fins(asm)
    f.thickness, f.material = 0.0015, "Plywood (Birch)"
    r = _fallback(tmp_path, asm, _max_q())
    fin = r.component_results["Fins"]
    plywood = get_structural_material("Plywood (Birch)")
    assert fin["sf"] == pytest.approx(plywood.yield_strength / fin["max_stress"])
    assert fin["sf"] < r.component_results["Airframe"]["sf"]
    assert r.safety_factor == pytest.approx(fin["sf"])
    assert r.yield_utilization == pytest.approx(1.0 / fin["sf"])
    # the airframe's aluminium would have passed the same stress easily
    assert r.safety_factor < 0.5 * get_structural_material(
        "Aluminum 6061-T6").yield_strength / r.max_von_mises


def test_no_airspeed_means_no_fin_load(tmp_path):
    """It was 30 % of the axial stress, whatever the fins. With no airflow
    the only bending left is the airframe's own inertia, which scales with
    its density; the made-up fin term did not."""
    lc = LoadCase.max_thrust(thrust=1500.0, angle_of_attack_deg=0.0, mach=0.0)
    r = _fallback(tmp_path, canonical_assembly(), lc)
    assert "Fins" not in r.component_results
    assert r.max_axial_stress > 0

    solver = CalculiXSolver(FEMConfig(load_case=lc, work_dir=tmp_path,
                                      assembly=canonical_assembly()))
    alu = get_structural_material("Aluminum 6061-T6")
    light = solver._analytical_fallback(FEMResult(), alu)
    heavy = solver._analytical_fallback(FEMResult(), replace(alu, density=2 * alu.density))
    assert light.max_bending_stress > 0
    assert heavy.max_bending_stress == pytest.approx(2.0 * light.max_bending_stress)


def test_bending_is_the_airframes_alone(tmp_path):
    """The fin-root stress is not part of the tube's bending stress (it fed
    the body contour, the buckling check and the tube deflection)."""
    thin, thick = canonical_assembly(), canonical_assembly()
    _fins(thick).thickness = 2 * _fins(thin).thickness
    for case in (_max_q, _max_thrust):
        a, b = _fallback(tmp_path, thin, case()), _fallback(tmp_path, thick, case())
        assert a.component_results["Fins"]["max_stress"] > b.component_results["Fins"]["max_stress"]
        assert a.max_bending_stress == pytest.approx(b.max_bending_stress)
        assert a.max_displacement_mm == pytest.approx(b.max_displacement_mm)


@pytest.mark.parametrize("count, panels", [(1, 1.0), (2, 2.0), (3, 1.5), (4, 2.0), (6, 3.0)])
def test_only_the_fins_in_the_crossflow_load_the_body(tmp_path, count, panels):
    asm = canonical_assembly()
    _fins(asm).fin_count = count
    solver = CalculiXSolver(FEMConfig(load_case=_max_q(), work_dir=tmp_path, assembly=asm))
    assert solver._fin_loads(_Q, math.radians(_AOA))[2] == panels


def test_an_earlier_runs_results_are_not_read_back(tmp_path):
    """With ccx missing, the static solve read the last run's analysis.frd
    from the work folder and showed it, labelled CalculiX, as this rocket's
    result (and run_modal parsed the last analysis.dat)."""
    for name in ("analysis.frd", "analysis.dat"):
        (tmp_path / name).write_text("left over from another rocket\n")
    solver = CalculiXSolver(FEMConfig(material_name="Aluminum 6061-T6", load_case=_max_q(),
                                      work_dir=tmp_path, assembly=canonical_assembly()))
    solver._ccx_exe = None
    solver.generate_mesh()
    solver.generate_case()
    for _stage in solver.run():
        pass
    r = solver.parse_results()
    assert not (tmp_path / "analysis.frd").exists()
    assert r.fe_field is None and r.converged
    assert set(r.component_results) == {"Airframe", "Fins"}

    (tmp_path / "analysis.dat").write_text("left over from another rocket\n")
    modal = solver.run_modal()
    assert not (tmp_path / "analysis.dat").exists()
    assert modal.converged and modal.damping_source.startswith("Material estimate")


def test_more_fin_area_bends_the_body_more(tmp_path):
    small, big = canonical_assembly(), canonical_assembly()
    _fins(big).height = 2 * _fins(small).height
    a, b = _fallback(tmp_path, small, _max_q()), _fallback(tmp_path, big, _max_q())
    assert b.max_bending_stress > a.max_bending_stress
    assert _fallback(tmp_path, _finless(), _max_q()).max_bending_stress < a.max_bending_stress
