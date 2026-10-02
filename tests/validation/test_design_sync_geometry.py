"""
The rocket drawn in the Design tab must be the rocket the simulation flies.

``DesignWorkspace._sync_to_engine`` copied length, diameter, fin count, CG and
CP into the rocket state and stopped there. Fin span, fin chords, fin sweep and
nose length stayed at zero, and ``AeroModel.from_state`` fills a zero with a
generic fin sized from the body (span 0.6 D, root chord 0.08 L). The flight
sim, Monte Carlo, the optimizer and the structures fin check all read those
flat fields, so every one of them analysed a rocket nobody drew:

* the validation rocket's 0.12 m x 0.20 m fins became 0.061 m x 0.16 m, its
  CP moved ahead of the CG and the flight aborted 0.4 s after liftoff;
* Monte Carlo has no such abort, so it reported a mean apogee of 237 m for a
  rocket that reaches 2700 m;
* on a heavier airframe nothing aborted at all: the apogee simply came out
  12% high and the stability margin read 0.5 cal instead of 3.3.

Every other test in this suite hands the sim a hand-written state. These fly
the state the Design tab produces.
"""
import math
import types

import pytest

from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.components import (BodyTube, NoseCone, RocketAssembly,
                             TrapezoidalFinSet)
from core.rocket_state import RocketStateEngine
from core.simulation_engine import SimulationEngine
from physics.aerodynamics import AeroModel
from validation.cases.rocket_canonical import MOTOR, canonical_assembly
from validation.sim.headless_runner import _ensure_qapp


def _design_sync(assembly, engine=None) -> RocketStateEngine:
    """Run the real Design-tab sync without building the widget.

    _sync_to_engine touches only ``assembly`` and ``engine``, so the method is
    bound to a plain object; the workspace itself needs an OpenGL viewport.
    """
    from ui.workspaces.design_workspace import DesignWorkspace

    _ensure_qapp()
    engine = engine or RocketStateEngine()
    engine._assembly = assembly
    shim = types.SimpleNamespace(assembly=assembly, engine=engine)
    DesignWorkspace._sync_to_engine(shim)
    return engine


def _load_motor(engine):
    engine.update(
        motor_designation=MOTOR["designation"],
        motor_avg_thrust=MOTOR["avg_thrust"],
        motor_max_thrust=MOTOR["max_thrust"],
        motor_total_impulse=MOTOR["total_impulse"],
        motor_burn_time=MOTOR["burn_time"],
        propellant_mass=MOTOR["propellant_mass"],
        propellant_mass_initial=MOTOR["propellant_mass"],
    )


def _fins(assembly):
    return next(c for c in assembly.all_components()
                if isinstance(c, TrapezoidalFinSet))


def _nose(assembly):
    return next(c for c in assembly.all_components() if isinstance(c, NoseCone))


def test_design_sync_carries_the_drawn_fins_and_nose():
    asm = canonical_assembly()
    state = _design_sync(asm).state
    fins, nose = _fins(asm), _nose(asm)

    assert state.fin_span == pytest.approx(fins.height)
    assert state.fin_height == pytest.approx(fins.height)
    assert state.fin_root_chord == pytest.approx(fins.root_chord)
    assert state.fin_tip_chord == pytest.approx(fins.tip_chord)
    assert state.fin_thickness == pytest.approx(fins.thickness)
    assert state.fin_cross_section == fins.cross_section
    assert state.nose_length == pytest.approx(nose.length)
    # Degrees on the component, radians on the state.
    assert state.fin_sweep_angle == pytest.approx(math.radians(fins.sweep_angle))
    assert math.tan(state.fin_sweep_angle) > 0.0


def test_the_sim_models_the_drawn_fins_not_generic_ones():
    asm = canonical_assembly()
    state = _design_sync(asm).state
    fins = _fins(asm)
    model = AeroModel.from_state(state)

    assert model.fin_span == pytest.approx(fins.height)
    assert model.fin_root_chord == pytest.approx(fins.root_chord)
    assert model.fin_tip_chord == pytest.approx(fins.tip_chord)
    assert model.fin_sweep == pytest.approx(math.radians(fins.sweep_angle))
    # What from_state substitutes for a missing fin: these must not come back.
    assert model.fin_span != pytest.approx(state.diameter * 0.6)
    assert model.fin_root_chord != pytest.approx(state.length * 0.08)


def test_loaded_validation_rocket_is_stable_in_the_sim():
    engine = _design_sync(canonical_assembly())
    _load_motor(engine)
    state = engine.state

    margin = (AeroModel.from_state(state).cp_subsonic() - state.cg) / state.diameter
    assert margin > 1.0, f"sim sees {margin:+.2f} cal for the validation rocket"


def test_a_new_design_replaces_the_previous_rockets_fins():
    engine = _design_sync(canonical_assembly())

    finless = RocketAssembly()
    stage = finless.stages[0]
    finless.add_component(stage, NoseCone())
    finless.add_component(stage, BodyTube())
    state = _design_sync(finless, engine).state

    assert state.fin_count == 0
    assert state.fin_span == state.fin_height == 0.0
    assert state.fin_root_chord == state.fin_tip_chord == 0.0
    assert state.fin_sweep_angle == 0.0


@pytest.mark.sim
def test_design_tab_rocket_flies_and_monte_carlo_flies_the_same_one():
    """End to end on a Design-produced state: the interactive engine reaches
    the ground without aborting, and the batch run Monte Carlo is built from
    agrees with it."""
    engine = _design_sync(canonical_assembly())
    _load_motor(engine)
    state = engine.state
    batch_cfg = BatchSimConfig.from_rocket_state(state)

    sim = SimulationEngine(engine)
    aborts = []
    sim.sim_failed.connect(lambda title, detail: aborts.append(title))
    sim.start()
    steps = 0
    while sim._running and steps < 2_000_000:
        sim._step()
        steps += 1

    assert aborts == []
    assert state.sim_phase == "Landed"
    assert state.max_altitude > 1000.0

    batch = run_batch_simulation(batch_cfg, seed=3)
    assert batch.final_phase == "Landed"
    assert batch.apogee == pytest.approx(state.max_altitude, rel=0.015)


def test_a_finless_design_reaches_the_sim_without_fins():
    """Design sync writes zeros for a rocket with no fin set, and the sim
    used to answer zeros with four fins of its own."""
    finless = RocketAssembly()
    stage = finless.stages[0]
    finless.add_component(stage, NoseCone())
    finless.add_component(stage, BodyTube())
    state = _design_sync(finless).state

    model = AeroModel.from_state(state)
    assert state.fin_count == 0
    assert model.fin_count == 0 and model.fin_span == 0.0

