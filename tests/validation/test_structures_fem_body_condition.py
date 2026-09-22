"""
A CalculiX static result must be able to drive the structural workstation.

StructuresWorkspace._on_result handed full_analysis() the FEM stresses as a
bare dict with no "safety_factor", and full_analysis indexes that key, so
every static FEM run raised KeyError('safety_factor'). The workstation tabs
(safety score, failure map, buckling, report export) then never updated.
Before 0.1.9 the same dict was patched onto the report after the analysis,
and the PDF printed the FEM run's safety factor as 0.00.
"""
import ast
import math
from pathlib import Path

import pytest

import structures.workstation as wks
from structures.solvers.base import FEMResult
from validation.cases.rocket_canonical import canonical_assembly, canonical_state

_MAT = "Aluminum 6061-T6"
_YIELD = 276e6


def _fem(vm=21.6e6):
    """A CalculiX-shaped static result. The margin uses the FEM convention
    (SF_req = FEMConfig.safety_factor_required = 2)."""
    sf = _YIELD / vm
    return FEMResult(max_von_mises=vm, max_axial_stress=1.2e6,
                     max_hoop_stress=0.2e6, max_bending_stress=8.0e6,
                     max_shear_stress=0.1e6, safety_factor=sf,
                     margin_of_safety=sf / 2.0 - 1.0,
                     yield_utilization=vm / _YIELD)


def _analyse(bc):
    return wks.full_analysis(canonical_state(), canonical_assembly(), None,
                             _MAT, "Max-Q", body_condition=bc)


@pytest.mark.structures
def test_fem_result_drives_the_workstation():
    rep = _analyse(wks.body_condition_from_fem(_fem()))
    assert rep.body_condition["safety_factor"] == pytest.approx(_YIELD / 21.6e6)
    assert rep.body_condition["von_mises"] == pytest.approx(21.6e6)
    assert rep.verdict != "—"


@pytest.mark.structures
def test_fem_margin_follows_the_workstation_convention():
    # SF 1.5 against the FEM's SF_req of 2 is a margin of -0.25, which the
    # physics checks report as "negative margin while SF > 1".
    bc = wks.body_condition_from_fem(_fem(vm=_YIELD / 1.5))
    assert bc["margin_of_safety"] == pytest.approx(0.5)
    rep = _analyse(bc)
    assert not any("margin convention" in w.message for w in rep.warnings)


@pytest.mark.structures
def test_unloaded_fem_result_is_not_a_failure():
    # The analytical FEM fallback leaves SF at its 0.0 default when nothing is
    # stressed — that must read as "no failure", not the worst possible one.
    bc = wks.body_condition_from_fem(FEMResult())
    assert math.isinf(bc["safety_factor"])
    assert _analyse(bc).verdict != "FAIL"


def test_structures_workspace_builds_the_body_condition_from_the_helper():
    src = (Path(__file__).resolve().parents[2]
           / "ui" / "workspaces" / "structures_workspace.py").read_text(encoding="utf-8")
    on_result = next(n for n in ast.walk(ast.parse(src))
                     if isinstance(n, ast.FunctionDef) and n.name == "_on_result")
    calls = {c.func.attr for c in ast.walk(on_result)
             if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
    assert "body_condition_from_fem" in calls
