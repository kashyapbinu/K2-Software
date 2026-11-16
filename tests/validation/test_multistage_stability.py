"""
A staged rocket balances where the design puts it, and nothing flies it on one
motor.

Found 2026-11-16 on a two-stage rocket (0.08 m sustainer over a 0.10 m
booster, a motor in each):

* The staged flight placed each stage's airframe at the middle of its length
  and its motor at 85 % of it, whatever the design said. Its static margin at
  liftoff was 2.19 cal; the Design tab showed 1.34 cal, and that one left the
  sustainer's motor out of the liftoff mass and CG altogether.
* The stage list the flight reads (``stages_config``) was rebuilt only when
  the Propulsion tab was shown, so a fin changed in the Design tab was not
  the fin that flew until that tab had been opened again.
* Monte Carlo, the optimiser and the apogee tools build their rocket from the
  flat state fields, which hold one motor. They flew the stack on that one
  motor: 1,256 m for a rocket the Simulation tab takes to 2,536 m.
"""
import types

import pytest

from core.batch_simulation import BatchSimConfig, StagedRocketNotSupported
from core.components import (BodyTube, MassComponent, NoseCone, RocketAssembly,
                             TrapezoidalFinSet)
from core.constants import G_EARTH
from core.rocket_state import RocketStateEngine
from core.staging import StageConfig, StageManager, build_stages_config
from physics.aerodynamics import AeroModel
from validation.sim.headless_runner import _ensure_qapp

SUSTAINER_MOTOR = dict(motor_designation="S", motor_avg_thrust=300.0, motor_max_thrust=380.0,
                       motor_total_impulse=600.0, motor_burn_time=2.0, propellant_mass=0.30,
                       motor_dry_mass=0.25, motor_length=0.30, motor_diameter=0.054)
BOOSTER_MOTOR = dict(motor_designation="B", motor_avg_thrust=900.0, motor_max_thrust=1100.0,
                     motor_total_impulse=1800.0, motor_burn_time=2.0, propellant_mass=0.90,
                     motor_dry_mass=0.60, motor_length=0.40, motor_diameter=0.075)


def _tube(asm, stage, d, length):
    t = BodyTube()
    t.length, t.outer_diameter_val, t.inner_diameter = length, d, d - 0.004
    asm.add_component(stage, t)
    return t


def _fins(asm, tube, span, root, tip):
    f = TrapezoidalFinSet()
    f.fin_count, f.height, f.root_chord, f.tip_chord, f.sweep_angle = 4, span, root, tip, 30.0
    asm.add_component(tube, f)
    return f


def two_stage():
    """Sustainer with a 1.2 kg payload just behind the nose (so its airframe
    balances well ahead of mid-length) over a finned booster."""
    asm = RocketAssembly()
    top = asm.stages[0]
    nose = NoseCone()
    nose.shape, nose.length, nose.diameter = "Ogive", 0.25, 0.08
    asm.add_component(top, nose)
    upper = _tube(asm, top, 0.08, 0.80)
    payload = MassComponent("Payload")
    payload.mass, payload.mass_length = 1.2, 0.10
    payload._ork_pos, payload._ork_rel = 0.02, "top"
    asm.add_component(upper, payload)
    _fins(asm, upper, 0.07, 0.11, 0.05)
    booster = asm.add_stage("Booster")
    _fins(asm, _tube(asm, booster, 0.10, 0.60), 0.12, 0.18, 0.09)
    top.motor, booster.motor = dict(SUSTAINER_MOTOR), dict(BOOSTER_MOTOR)
    asm._recompute_positions()
    return asm


def _sync(asm) -> RocketStateEngine:
    """The Design tab's sync, without the widget (see test_design_sync_geometry)."""
    from ui.workspaces.design_workspace import DesignWorkspace
    _ensure_qapp()
    engine = RocketStateEngine()
    engine._assembly = asm
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=asm, engine=engine))
    return engine


def _true_cg(asm, stages):
    """Liftoff CG of *stages* from the design itself: every component at its
    own place, each motor (case and propellant) centred half a motor length
    ahead of its stage's aft end. Measured from the top of the first stage."""
    origin = stages[0].position
    moment = mass = 0.0
    for st in stages:
        for c in asm._iter_children(st):
            m = c.computed_mass()
            moment += m * (c.cg_position() - origin)
            mass += m
        motor = st.motor["motor_dry_mass"] + st.motor["propellant_mass"]
        aft = st.position + st.component_length() - origin
        moment += motor * (aft - st.motor["motor_length"] / 2.0)
        mass += motor
    return moment / mass, mass


def test_a_staged_rocket_balances_in_flight_where_the_design_puts_it():
    asm = two_stage()
    mgr = StageManager([StageConfig.from_dict(d) for d in build_stages_config(asm)])
    cg, mass = _true_cg(asm, asm.stages)
    assert mgr.total_mass() == pytest.approx(mass, rel=1e-9)
    assert mgr.active_cg() == pytest.approx(cg, abs=1e-6)

    # The payload puts the stack's CG well ahead of where stage mid-lengths
    # and 85 % motor stations would: that estimate was 85 mm (0.85 cal) aft.
    L1, L2 = asm.stages[0].component_length(), asm.stages[1].component_length()
    m1 = asm.stages[0].total_mass()
    m2 = asm.stages[1].total_mass()
    s_motor = SUSTAINER_MOTOR["motor_dry_mass"] + SUSTAINER_MOTOR["propellant_mass"]
    b_motor = BOOSTER_MOTOR["motor_dry_mass"] + BOOSTER_MOTOR["propellant_mass"]
    guessed = (m1 * L1 * 0.5 + s_motor * L1 * 0.85
               + m2 * (L1 + L2 * 0.5) + b_motor * (L1 + L2 * 0.85)) / mass
    assert guessed - cg > 0.05, (guessed, cg)

    # After separation the sustainer flies with its own balance.
    mgr.bottom_index = 1
    cg_s, mass_s = _true_cg(asm, asm.stages[:1])
    assert mgr.total_mass() == pytest.approx(mass_s, rel=1e-9)
    assert mgr.active_cg() == pytest.approx(cg_s, abs=1e-6)


def test_a_stage_list_saved_without_mass_centres_still_flies_as_it_did():
    """Projects saved before the stage list carried dry_cg and motor_position
    keep the mid-length and 85 % stations they were flown with."""
    old = [{k: v for k, v in d.items() if k not in ("dry_cg", "motor_position")}
           for d in build_stages_config(two_stage())]
    mgr = StageManager([StageConfig.from_dict(d) for d in old])
    booster, sustainer = mgr.stages
    expected = (sustainer.dry_mass * sustainer.length * 0.5
                + (sustainer.motor_dry_mass + sustainer.propellant_mass) * sustainer.length * 0.85
                + booster.dry_mass * (sustainer.length + booster.length * 0.5)
                + (booster.motor_dry_mass + booster.propellant_mass)
                * (sustainer.length + booster.length * 0.85)) / mgr.total_mass()
    assert mgr.active_cg() == pytest.approx(expected, rel=1e-9)


def test_the_design_tab_shows_the_staged_rockets_liftoff_stack():
    asm = two_stage()
    engine = _sync(asm)
    # The Propulsion tab mirrors the stage being edited in the flat motor
    # fields: here the sustainer's, as after choosing its motor last.
    flat = dict(SUSTAINER_MOTOR)
    flat.pop("motor_diameter")
    engine.update(propellant_mass_initial=flat["propellant_mass"], **flat)
    s = engine.state
    assert len(s.stages_config) == 2, "the design sync carries the stage list"

    mgr = StageManager([StageConfig.from_dict(d) for d in s.stages_config])
    assert s.liftoff_mass() == pytest.approx(mgr.total_mass(), rel=1e-9)
    assert s.liftoff_cg() == pytest.approx(mgr.active_cg(), rel=1e-9)
    flown = AeroModel.from_state(mgr.aero_config()).static_margin(0.0, mgr.active_cg())
    assert s.pad_stability_margin() == pytest.approx(flown, rel=1e-9)

    # Liftoff thrust-to-weight is the first stage to light, over the stack.
    assert s.liftoff_avg_thrust() == BOOSTER_MOTOR["motor_avg_thrust"]
    expected_twr = BOOSTER_MOTOR["motor_avg_thrust"] / (mgr.total_mass() * G_EARTH)
    from core import rules as R
    from core.technical_report import build_content, fmt
    assert R.context_from_state(s).liftoff_thrust_to_weight == pytest.approx(expected_twr)
    rep = build_content(s, None)
    assert rep.value("Motor", "Liftoff thrust-to-weight (average thrust)") == fmt(expected_twr, "", 2)
    assert rep.value("Vehicle", "Liftoff mass") == fmt(mgr.total_mass(), "kg", 3)


def test_a_design_change_reaches_the_staged_flight():
    asm = two_stage()
    engine = _sync(asm)
    booster_fins = next(c for c in asm._iter_children(asm.stages[1])
                        if isinstance(c, TrapezoidalFinSet))
    assert engine.state.stages_config[0]["fin_span"] == pytest.approx(0.12)
    booster_fins.height = 0.16
    from ui.workspaces.design_workspace import DesignWorkspace
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=asm, engine=engine))
    assert engine.state.stages_config[0]["fin_span"] == pytest.approx(0.16)

    # One stage again: no stage list is left behind to fly.
    asm.remove_stage(asm.stages[1])
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=asm, engine=engine))
    assert engine.state.stages_config == []


def test_monte_carlo_refuses_a_staged_rocket_instead_of_flying_one_stage():
    engine = _sync(two_stage())
    with pytest.raises(StagedRocketNotSupported) as err:
        BatchSimConfig.from_rocket_state(engine.state)
    assert isinstance(err.value, ValueError)
    assert "2 stages" in str(err.value) and "Simulation tab" in str(err.value)

    from core.monte_carlo_engine import MonteCarloConfig, MonteCarloEngine
    mc = MonteCarloEngine(engine)
    told = []
    mc.analysis_failed.connect(told.append)
    mc.start(MonteCarloConfig(num_simulations=4))
    assert told and "2 stages" in told[0], told
    assert not mc.is_running

    # A single-stage rocket is flown as before.
    single = two_stage()
    single.remove_stage(single.stages[1])
    assert BatchSimConfig.from_rocket_state(_sync(single).state).length == pytest.approx(1.05)
