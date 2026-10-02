"""
The optimisation results must describe the whole rocket, not just what moved.

A candidate's ``variables`` dict holds only the parameters the optimiser was
asked to vary. The Best Design panel read every row from it with a default of
zero, so a run that varied fin span alone reported a best design 0 m in
diameter, 0 m long, with a dry mass of 0 kg and "Motor N/A". The same pattern
gave "Best Mass: 999.00 kg", a 0 kg trade-study row and zeros in the PDF.

The engine had already been corrected for this (``_geom_value``); the display
had not. ``design_value`` is the display's counterpart.
"""
import inspect
import types

import pytest

from core.batch_simulation import BatchSimConfig
from core.optimization_engine import (CandidateDesign, DesignVariable,
                                      ObjectiveFunction, design_value,
                                      evaluate_candidate)
from validation.cases.rocket_canonical import canonical_state


def _design(variables, **config):
    cfg = BatchSimConfig(**config) if config else None
    return CandidateDesign(variables=dict(variables), batch_config=cfg)


def test_an_optimised_parameter_is_the_optimisers_value():
    d = _design({"fin_span": 0.09}, fin_span=0.12, diameter=0.102)
    assert design_value(d, "fin_span") == 0.09


def test_a_parameter_left_alone_is_the_rockets_own():
    d = _design({"fin_span": 0.09}, diameter=0.102, length=2.0,
                motor_designation="L1090W")
    assert design_value(d, "diameter") == 0.102
    assert design_value(d, "length") == 2.0
    assert design_value(d, "motor_designation") == "L1090W"


def test_an_unevaluated_design_falls_back_to_the_rocket_state():
    state = types.SimpleNamespace(diameter=0.102)
    failed = CandidateDesign(variables={"fin_span": 0.09})   # no batch_config

    assert design_value(failed, "diameter", fallback=state) == 0.102
    assert design_value(failed, "fin_span", fallback=state) == 0.09


def test_an_unknown_value_is_none_not_zero():
    assert design_value(CandidateDesign(), "diameter") is None
    assert design_value(None, "diameter") is None


@pytest.mark.sim
@pytest.mark.optimization
def test_an_evaluated_candidate_reports_the_base_rockets_dimensions():
    base = BatchSimConfig.from_rocket_state(canonical_state())
    span = DesignVariable("fin_span", "Fin Span", "Geometry", 0.05, 0.20, 0.12)
    design = evaluate_candidate(
        base, {"fin_span": 0.09}, [span],
        [ObjectiveFunction("max_apogee", "Max Apogee")], [], [], 1, seed=1)

    assert "diameter" not in design.variables
    assert design_value(design, "diameter") == pytest.approx(base.diameter)
    assert design_value(design, "length") == pytest.approx(base.length)
    assert design_value(design, "dry_mass") == pytest.approx(base.dry_mass)
    assert design_value(design, "fin_span") == pytest.approx(0.09)


def test_the_best_design_panel_does_not_default_missing_rows_to_zero():
    """The panel is a QWidget, so its wiring is read off the source: no row
    may come from ``variables.get(..., <default>)``."""
    from ui.workspaces.optimization_workspace import OptimizationWorkspace

    src = inspect.getsource(
        OptimizationWorkspace._update_results_panel_from_design)
    parameters = src.split("# Performance")[0]
    assert "v.get(" not in parameters
    assert "_design_text(design, \"diameter\"" in parameters
