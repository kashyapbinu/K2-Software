"""
What the rocket is on the pad does not depend on whether it has flown.

A flight burns the propellant down on the rocket state and leaves it there:
after landing ``propellant_mass`` is the last gram of residue, and the CG,
static margin and weight that follow it are the burnt-out rocket's. Code that
describes the *design* read those live values. So for the canonical rocket
(7.400 kg, CG 1.051 m, 5.30 cal on the pad), once it had flown:

* the technical report printed "Liftoff mass 6.001 kg", "CG at liftoff
  0.900 m", "Static margin on the pad 6.78 cal" and a liftoff
  thrust-to-weight of 18.52 where the rocket lifts off at 15.02;
* the rules check judged liftoff thrust-to-weight with the burnt-out mass;
* the Design tab, the Properties panel, the Readiness board and the AI
  assistant gave 6.78 cal as the rocket's stability;
* the Structures tab took the burnt-out mass as the mass at peak thrust.

All of them now read RocketState.liftoff_mass / liftoff_cg /
pad_stability_margin.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import rules as R
from core.constants import G_EARTH
from core.rocket_state import RocketStateEngine
from core.technical_report import build_content
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import _ensure_qapp

ROOT = Path(__file__).resolve().parents[2]


def _engine() -> RocketStateEngine:
    _ensure_qapp()
    engine = RocketStateEngine()
    engine.set_state(canonical_state())
    return engine


def _fly(engine) -> None:
    """Leave the state as a landed flight leaves it: the motor burnt out but
    for the residue the mass-flow cut-off keeps."""
    engine.update(propellant_mass=0.0008, sim_time=140.0, sim_phase="Landed")


def test_the_pad_numbers_do_not_change_when_the_rocket_has_flown():
    engine = _engine()
    s = engine.state
    mass, cg, margin = s.liftoff_mass(), s.liftoff_cg(), s.pad_stability_margin()
    assert mass == pytest.approx(s.dry_mass + s.motor_dry_mass + s.propellant_mass_initial)
    # On the pad they are the live values.
    assert cg == pytest.approx(s.cg) and margin == pytest.approx(s.stability_margin)

    _fly(engine)
    assert s.cg < cg - 0.1, "the live CG is the burnt-out rocket's"
    assert s.liftoff_mass() == pytest.approx(mass)
    assert s.liftoff_cg() == pytest.approx(cg)
    assert s.pad_stability_margin() == pytest.approx(margin)


def test_the_report_describes_the_rocket_on_the_pad_after_it_has_flown():
    engine = _engine()
    before = build_content(engine.state, None)
    assert before.value("Vehicle", "Liftoff mass") == "7.400 kg"
    _fly(engine)
    after = build_content(engine.state, None)
    for section, label in (("Vehicle", "Liftoff mass"),
                           ("Vehicle", "CG at liftoff (from nose)"),
                           ("Vehicle", "Static margin on the pad"),
                           ("Motor", "Liftoff thrust-to-weight (average thrust)")):
        assert after.value(section, label) == before.value(section, label), label


def test_liftoff_thrust_to_weight_is_judged_with_the_liftoff_mass():
    engine = _engine()
    s = engine.state
    expected = s.motor_avg_thrust / (7.4 * G_EARTH)
    assert R.context_from_state(s).liftoff_thrust_to_weight == pytest.approx(expected)
    _fly(engine)
    assert R.context_from_state(s).liftoff_thrust_to_weight == pytest.approx(expected)


def test_the_assistant_is_told_the_pad_balance_after_a_flight():
    from ai.context import liftoff_mass, summarize_state
    engine = _engine()
    before = summarize_state(engine.state)["mass_and_stability"]
    _fly(engine)
    after = summarize_state(engine.state)["mass_and_stability"]
    for key in ("liftoff_mass_kg", "cg_from_nose_m", "stability_margin_cal",
                "stability_verdict"):
        assert after[key] == before[key], key
    assert liftoff_mass(engine.state) == pytest.approx(engine.state.liftoff_mass())


def test_structures_takes_the_same_vehicle_mass_before_and_after_a_flight():
    from structures import workstation as wks
    engine = _engine()
    s = engine.state
    before = wks.modal_estimate(s, s.material_name).total_mass_kg
    assert before == pytest.approx(7.4)
    _fly(engine)
    assert wks.modal_estimate(s, s.material_name).total_mass_kg == pytest.approx(before)


def test_one_stability_verdict_for_every_screen():
    """Three readings of one number: at 0.3 cal the Design tab said UNSTABLE,
    the Properties panel beside it "MARGINAL–" and the assistant "marginal";
    at 2.8 cal both panels said OVERSTABLE and the assistant "stable". The
    Mission Visualizer declared "FAILURE: Instability" below 0.5 cal, on a
    rocket whose flight carried no stability warning (that needs a margin
    below zero)."""
    from core.stability import (MARGINAL, OVERSTABLE, STABLE, UNSTABLE,
                                stability_verdict)
    assert [stability_verdict(m) for m in (-0.2, 0.0, 0.3, 0.99, 1.0, 2.5, 2.51, 6.0)] == [
        UNSTABLE, MARGINAL, MARGINAL, MARGINAL, STABLE, STABLE, OVERSTABLE, OVERSTABLE]
    assert stability_verdict(None) is None and stability_verdict(float("nan")) is None

    from ai.context import _stability_verdict
    assert "UNSTABLE" in _stability_verdict(-0.2)
    assert "marginal" in _stability_verdict(0.3)
    assert _stability_verdict(2.0) == "stable"
    assert "over-stable" in _stability_verdict(2.8)
    assert _stability_verdict(None) == "unknown"

    # The visualizer's in-flight failure flag is the flight's own warning.
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from ui.workspaces.mission_visualizer_workspace import MissionVisualizerWorkspace
    view = MagicMock()
    view._failure_active = False
    flying = dict(sim_time=3.0, sim_phase="Coast", safety_factor=5.0, mach_number=0.8,
                  flutter_margin=9.0, x_position=0.0, y_position=0.0, altitude=500.0)
    MissionVisualizerWorkspace._check_failures(view, SimpleNamespace(stability_margin=0.3, **flying))
    assert view._failure_active is False, "0.3 cal is marginal, not a failure"
    MissionVisualizerWorkspace._check_failures(view, SimpleNamespace(stability_margin=-0.3, **flying))
    assert view._failure_active is True


def test_the_optimiser_starts_from_the_loaded_motor_after_a_flight():
    """The "Propellant Mass" variable took its current value from the live
    propellant_mass: the 0.8 g of residue once the rocket had flown."""
    from core.optimization_engine import get_default_design_variables
    engine = _engine()
    before = {v.name: v.current_val for v in get_default_design_variables(engine.state)}
    assert before["propellant_mass"] == pytest.approx(1.4)
    _fly(engine)
    after = {v.name: v.current_val for v in get_default_design_variables(engine.state)}
    assert after == before


def test_the_live_mass_readout_includes_the_motor_case():
    """Mission Visualizer telemetry showed dry + propellant: the rocket
    without its motor case (1.03 kg short for an L1090W), while the
    Simulation tab and the replay of the same flight showed all of it."""
    from unittest.mock import MagicMock
    from ui.workspaces.mission_visualizer_workspace import MissionVisualizerWorkspace
    engine = _engine()
    engine.update(motor_dry_mass=1.032, propellant_mass=0.9, sim_time=1.0)
    s = engine.state
    view = MagicMock()
    view._closing, view._replay_mode, view._latest_state = False, False, s
    view._hud_actor, view._show_stats = None, False
    MissionVisualizerWorkspace._update_readouts(view)
    shown = view._rd_mass.set_value.call_args[0][0]
    assert shown == pytest.approx(s.dry_mass + 1.032 + 0.9)
    assert shown == pytest.approx(s.flight_mass())


def test_the_landing_envelope_descends_under_the_canopies_the_rocket_has():
    """The envelope's descent rate was sqrt(2 m g / (rho x main slot)), with
    m = dry + propellant and the area floored at 0.1 m². A rocket with one
    parachute has no main slot, so it was drawn falling under an invented
    0.1 m² canopy (34 m/s for this rocket, which comes down at 6.7 m/s)."""
    import math
    from recovery.parachute_dynamics import final_descent_rate
    engine = _engine()
    engine.update(motor_dry_mass=1.032, recovery_canopies=[
        {"name": "Main", "cd_area": 2.5, "event": "apogee", "altitude": 0.0, "delay": 1.0}],
        drogue_cd_area=2.5, main_cd_area=0.0)
    s = engine.state
    burnt_out = s.dry_mass + s.motor_dry_mass
    expected = math.sqrt(2.0 * burnt_out * G_EARTH / (1.225 * 2.5))
    assert final_descent_rate(s) == pytest.approx(expected)
    # Every canopy is open at the end.
    engine.update(recovery_canopies=[
        {"name": "Drogue", "cd_area": 0.4, "event": "apogee", "altitude": 0.0, "delay": 1.0},
        {"name": "Main", "cd_area": 3.5, "event": "altitude", "altitude": 250.0, "delay": 0.0}])
    assert final_descent_rate(s) == pytest.approx(
        math.sqrt(2.0 * burnt_out * G_EARTH / (1.225 * 3.9)))
    # The same before and after the motor has burnt.
    _fly(engine)
    assert final_descent_rate(s) == pytest.approx(
        math.sqrt(2.0 * burnt_out * G_EARTH / (1.225 * 3.9)))
    # No parachute: no descent rate, not one under a made-up canopy.
    engine.update(recovery_canopies=[], drogue_cd_area=0.0, main_cd_area=0.0)
    assert final_descent_rate(s) is None

    # Both 3D views take it from there.
    import types
    from ui.workspaces.mission_visualizer_workspace import MissionVisualizerWorkspace
    engine.update(recovery_canopies=[
        {"name": "Main", "cd_area": 2.5, "event": "apogee", "altitude": 0.0, "delay": 1.0}])
    view = types.SimpleNamespace(_latest_state=s, engine=engine)
    assert MissionVisualizerWorkspace._terminal_descent(view) == pytest.approx(expected)


def test_an_empty_motor_weighs_and_balances_as_an_empty_motor():
    """With no design tree the engine estimates the balance. It divided the
    moments of the propellant actually left by total_mass(), which reads an
    exactly empty tank as "not loaded yet" and puts the full load back: a
    burnt-out rocket weighed its liftoff weight and its CG moved 0.17 m
    toward the nose."""
    engine = _engine()
    s = engine.state
    engine.update(propellant_mass=0.0, sim_time=10.0)
    assert s.weight == pytest.approx((s.dry_mass + s.motor_dry_mass) * G_EARTH)
    assert s.cg == pytest.approx(s.cg_with(0.0))
    # Not yet flown and the tank not synced: the load is the rated one.
    engine.update(sim_time=0.0)
    assert s.weight == pytest.approx(s.liftoff_mass() * G_EARTH)
    assert s.cg == pytest.approx(s.liftoff_cg())


_WIDGETS = textwrap.dedent("""
    import sys, types
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication, QLabel
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from ui.properties_panel import PropertiesPanel
    from ui.workspaces.design_workspace import DesignWorkspace
    from ui.workspaces.readiness_workspace import ReadinessWorkspace
    from validation.cases.rocket_canonical import canonical_state

    engine = RocketStateEngine()
    board = ReadinessWorkspace(engine)
    panel = PropertiesPanel(engine)
    # The Design tab's read-out without its OpenGL viewport.
    design = types.SimpleNamespace(engine=engine, _stab_cp=QLabel(), _stab_cg=QLabel(),
                                   _stab_margin=QLabel(), _stab_status=QLabel())
    engine.state_changed.connect(lambda s: DesignWorkspace._refresh_stability(design, s))
    engine.set_state(canonical_state())

    def shown():
        return (board.design_tiles["stability"].text(), board.design_tiles["mass"].text(),
                panel.stability_indicator.value_label.text(), panel.cg_label.text(),
                panel.total_mass_label.text(), panel.weight_label.text(),
                design._stab_cg.text(), design._stab_margin.text(), design._stab_status.text())

    pad = shown()
    assert pad[0] == "5.30 cal" and pad[2] == "5.30 cal" and pad[7] == "5.30 cal", pad
    assert pad[4] == "7.400 kg", pad

    # As a landed flight leaves the state.
    engine.update(propellant_mass=0.0008, sim_time=140.0, sim_phase="Landed")
    assert shown() == pad, (shown(), pad)

    # One verdict for one number: the Design read-out and the Properties
    # panel sit side by side and used to disagree below 0.5 cal.
    def set_margin(m):
        s = engine.state
        total, motor = s.liftoff_mass(), s.liftoff_mass() - s.dry_mass
        motor_cg = max(0.0, s.motor_position - 0.5 * s.motor_length)
        cg = s.cp - m * s.diameter
        engine.update(dry_cg=(total * cg - motor * motor_cg) / s.dry_mass, sim_time=0.0,
                      propellant_mass=s.propellant_mass_initial, sim_phase="Pre-Launch")
    for m, word in ((-0.4, "UNSTABLE"), (0.3, "MARGINAL"), (1.6, "STABLE"), (3.1, "OVERSTABLE")):
        set_margin(m)
        d, p = design._stab_status.text(), panel.stability_indicator.status_label.text()
        assert d == p and word in d, (m, d, p, design._stab_margin.text())
    print("OK")
""")


def test_the_design_readouts_show_the_pad_rocket_after_a_flight():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    r = subprocess.run([sys.executable, "-c", _WIDGETS.format(root=str(ROOT))],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
