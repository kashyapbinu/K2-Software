"""
The sim must fly every part the Design tab's stability readout counts.

RocketAssembly.compute_cp, the Design tab's CP, sums the nose, EVERY fin set
and every diameter change (transitions and boat-tail nozzles). The flight
sim's AeroModel knew one fin set on one constant-diameter tube. Measured on
2026-10-03, sim CP minus Design CP:

* four 50 mm canards ahead of the tail fins: +2.0 cal, so the sim flew a rocket
  two calibers MORE stable than the one on screen;
* a two-stage rocket as one state: -4.5 cal (the sustainer's fins only); a
  multistage flight took the booster's fins alone at liftoff;
* a 70 -> 100 mm shoulder: -0.50 cal; fins on a 70 mm can below a
  100 mm body: -0.25 cal;
* a boat-tail nozzle: +0.32 cal. The Design tab also placed the boat-tail's
  normal force at L/3 instead of at the frustum's CP.
"""
import copy
import json
import math
import types

import pytest

from core.batch_simulation import BatchSimConfig, _StateProxy, run_batch_simulation
from core.components import (BodyTube, NoseCone, Nozzle, RocketAssembly, Transition,
                             TrapezoidalFinSet)
from core.rocket_state import RocketState, RocketStateEngine
from core.staging import StageConfig, StageManager, build_stages_config
from physics.aerodynamics import AeroModel
from validation.cases.rocket_canonical import MOTOR
from validation.sim.headless_runner import _ensure_qapp


def _sync(assembly):
    """The Design tab's sync, without the widget (see test_design_sync_geometry)."""
    from ui.workspaces.design_workspace import DesignWorkspace
    _ensure_qapp()
    engine = RocketStateEngine()
    engine._assembly = assembly
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=assembly, engine=engine))
    return engine


def _nose(asm, d=0.10, length=0.30):
    n = NoseCone()
    n.shape, n.length, n.diameter = "Ogive", length, d
    asm.add_component(asm.stages[0], n)


def _tube(asm, d=0.10, length=1.0, stage=None):
    t = BodyTube()
    t.length, t.outer_diameter_val, t.inner_diameter = length, d, d - 0.004
    asm.add_component(stage or asm.stages[0], t)
    return t


def _fins(asm, parent, n=4, span=0.10, root=0.15, tip=0.07, sweep=30.0, at=None):
    f = TrapezoidalFinSet()
    f.fin_count, f.height, f.root_chord, f.tip_chord, f.sweep_angle = n, span, root, tip, sweep
    if at is not None:                       # from the parent tube's fore end
        f._ork_pos, f._ork_rel = at, "top"
    asm.add_component(parent, f)
    return f


def _transition(asm, fore, aft, length, stage=None):
    tr = Transition()
    tr.length, tr.fore_diameter, tr.aft_diameter = length, fore, aft
    asm.add_component(stage or asm.stages[0], tr)


def plain():
    asm = RocketAssembly()
    _nose(asm)
    _fins(asm, _tube(asm, length=1.4))
    return asm


def shoulder():
    """Narrow payload section and nose, shoulder out to a wider motor tube."""
    asm = RocketAssembly()
    _nose(asm, d=0.07, length=0.25)
    _tube(asm, d=0.07, length=0.5)
    _transition(asm, 0.07, 0.10, 0.10)
    _fins(asm, _tube(asm, d=0.10, length=0.8))
    return asm


def fin_can():
    """Wide body tapering to a narrower motor section that carries the fins."""
    asm = RocketAssembly()
    _nose(asm)
    _tube(asm, length=0.5)
    _transition(asm, 0.10, 0.07, 0.12)
    _fins(asm, _tube(asm, d=0.07, length=0.8))
    return asm


def boat_tail():
    asm = RocketAssembly()
    _nose(asm)
    _fins(asm, _tube(asm, length=1.3))
    bt = Nozzle()
    bt.nozzle_type, bt.length, bt.inlet_diameter, bt.exit_diameter = "Boat-Tail", 0.10, 0.10, 0.06
    asm.add_component(asm.stages[0], bt)
    return asm


def canards(with_canards=True):
    asm = RocketAssembly()
    _nose(asm)
    tube = _tube(asm, length=1.4)
    _fins(asm, tube)
    if with_canards:
        _fins(asm, tube, span=0.05, root=0.06, tip=0.03, sweep=20.0, at=0.10)
    asm._recompute_positions()
    return asm


def two_stage(motors=False, interstage=False, canards=False):
    asm = RocketAssembly()
    _nose(asm, d=0.08, length=0.25)
    upper = _tube(asm, d=0.08, length=0.8)
    _fins(asm, upper, span=0.07, root=0.11, tip=0.05)
    if canards:
        _fins(asm, upper, n=3, span=0.04, root=0.05, tip=0.03, at=0.05)
    booster = asm.add_stage("Booster")
    if interstage:
        _transition(asm, 0.08, 0.10, 0.08, stage=booster)
    _fins(asm, _tube(asm, d=0.10, length=0.6, stage=booster), span=0.12, root=0.18, tip=0.09)
    if motors:
        for st in asm.stages:
            st.motor = dict(motor_designation="X", motor_avg_thrust=400.0,
                            motor_max_thrust=500.0, motor_total_impulse=800.0,
                            motor_burn_time=2.0, propellant_mass=0.4, motor_dry_mass=0.3,
                            motor_length=0.3, motor_diameter=0.054)
    asm._recompute_positions()
    return asm


_ROCKETS = [plain, shoulder, fin_can, boat_tail, canards, two_stage]


@pytest.mark.parametrize("build", _ROCKETS, ids=lambda f: f.__name__)
def test_design_tab_and_sim_place_the_cp_identically(build):
    state = _sync(build()).state
    sim_cp = AeroModel.from_state(state).cp_subsonic()
    assert state.cp == pytest.approx(sim_cp, rel=1e-9)
    assert state.stability_margin == pytest.approx(
        (sim_cp - state.cg) / state.diameter, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("kw", [{}, dict(interstage=True), dict(interstage=True, canards=True)],
                         ids=["plain", "interstage", "interstage+canards"])
def test_multistage_stack_flies_the_design_tabs_cp(kw):
    """At liftoff the stack carries both stages' fins; after separation the
    sustainer flies as the Design tab shows the sustainer alone."""
    asm = two_stage(motors=True, **kw)
    mgr = StageManager([StageConfig.from_dict(d) for d in build_stages_config(asm)])
    assert AeroModel.from_state(mgr.aero_config()).cp_subsonic() == \
        pytest.approx(_sync(asm).state.cp, rel=1e-9)

    sustainer = copy.deepcopy(asm)
    sustainer.remove_stage(sustainer.stages[1])
    mgr.bottom_index = 1
    assert AeroModel.from_state(mgr.aero_config()).cp_subsonic() == \
        pytest.approx(_sync(sustainer).state.cp, rel=1e-9)


def test_the_boat_tail_is_a_transition():
    from physics.aerodynamics import compute_conical_transition_cp
    asm = boat_tail()
    bt = next(c for c in asm.all_components() if isinstance(c, Nozzle))
    tr = Transition()
    tr._position, tr.length = bt.position, bt.length
    tr.fore_diameter, tr.aft_diameter = bt.inlet_diameter, bt.exit_diameter
    d_ref = asm.get_reference_diameter()
    assert bt.cp_contribution(d_ref) == pytest.approx(tr.cp_contribution(d_ref))
    cn, cp = bt.cp_contribution(d_ref)
    assert cn < 0
    # frustum CP, not L/3: (L/3)(df + 2 da)/(df + da)
    assert cp - bt.position == pytest.approx(0.1 / 3.0 * (0.10 + 0.12) / 0.16)
    assert compute_conical_transition_cp(0.3, 0.1, 0.1) == pytest.approx(0.15)


def test_a_full_diameter_part_follows_the_body():
    """Fins on the body tube and a nose as wide as it are stored as 0 ("the
    body's"), so a diameter edited on the state (the optimizer's diameter
    variable) moves them with it. Only a different tube keeps its radius."""
    state = _sync(plain()).state
    assert state.nose_diameter == 0.0 and state.fin_body_radius == 0.0
    assert state.extra_fin_sets == [] and state.transitions == []

    state = _sync(fin_can()).state
    assert state.fin_body_radius == pytest.approx(0.035)
    assert state.nose_diameter == 0.0
    assert [t["fore_diameter"] for t in state.transitions] == [0.10]


def test_canards_add_their_drag_and_damping():
    with_c = AeroModel.from_state(_sync(canards()).state)
    without = AeroModel.from_state(_sync(canards(False)).state)
    assert len(with_c.fin_sets) == 2 and len(without.fin_sets) == 1
    for mach in (0.3, 0.8, 1.6):
        v = mach * 340.0
        a = with_c.compute(0.05, mach, 0.5 * 1.1 * v * v, 0.5, v, 0.9)
        b = without.compute(0.05, mach, 0.5 * 1.1 * v * v, 0.5, v, 0.9)
        assert a["cd"] > b["cd"]
        assert a["cmq"] < b["cmq"]
        assert a["cn_total"] > b["cn_total"]


def test_batch_runs_carry_the_whole_rocket_without_sharing_it():
    state = _sync(canards()).state
    cfg = BatchSimConfig.from_rocket_state(state)
    assert AeroModel.from_state(_StateProxy(cfg)).cp_subsonic() == pytest.approx(state.cp, rel=1e-9)
    cfg.extra_fin_sets[0]["fin_span"] = 1.0                 # a perturbed copy...
    assert state.extra_fin_sets[0]["fin_span"] == pytest.approx(0.05)   # ...not the state


def test_canards_change_the_flight():
    """The batch sim (Monte Carlo, the optimizer) flew a canard rocket and a
    canard-free one identically: same CP, same drag."""
    def apogee(asm):
        engine = _sync(asm)
        engine.update(motor_designation=MOTOR["designation"],
                      motor_avg_thrust=MOTOR["avg_thrust"], motor_max_thrust=MOTOR["max_thrust"],
                      motor_total_impulse=MOTOR["total_impulse"], motor_burn_time=MOTOR["burn_time"],
                      propellant_mass=MOTOR["propellant_mass"],
                      propellant_mass_initial=MOTOR["propellant_mass"])
        return run_batch_simulation(BatchSimConfig.from_rocket_state(engine.state), seed=3).apogee
    assert apogee(canards()) < 0.995 * apogee(canards(False))


def test_optimizer_keeps_drawn_diameters_in_proportion():
    from core.optimization_engine import DesignVariable, build_candidate_config
    base = BatchSimConfig.from_rocket_state(_sync(fin_can()).state)
    dvs = [DesignVariable("diameter", "Body Diameter", "Geometry", 0.05, 0.2, 0.1),
           DesignVariable("length", "Body Length", "Geometry", 1.0, 4.0, base.length)]
    cfg = build_candidate_config(base, {"diameter": 0.2, "length": 2 * base.length}, dvs)
    assert cfg.fin_body_radius == pytest.approx(2 * base.fin_body_radius)
    assert cfg.transitions[0]["fore_diameter"] == pytest.approx(0.2)
    assert cfg.transitions[0]["aft_diameter"] == pytest.approx(0.14)
    assert cfg.transitions[0]["position"] == pytest.approx(2 * base.transitions[0]["position"])
    assert base.transitions[0]["fore_diameter"] == pytest.approx(0.10)  # base untouched

    # A plain rocket's fins stay on the body whatever its diameter becomes.
    base = BatchSimConfig.from_rocket_state(_sync(plain()).state)
    cfg = build_candidate_config(base, {"diameter": 0.2}, dvs[:1])
    model = AeroModel.from_state(_StateProxy(cfg))
    assert model.fin_sets[0].body_radius == pytest.approx(0.1)


def test_a_saved_project_reloads_the_whole_rocket():
    state = _sync(two_stage(interstage=True, canards=True)).state
    reloaded = RocketState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert reloaded.extra_fin_sets == state.extra_fin_sets
    assert reloaded.transitions == state.transitions
    assert AeroModel.from_state(reloaded).cp_subsonic() == pytest.approx(state.cp, rel=1e-9)
