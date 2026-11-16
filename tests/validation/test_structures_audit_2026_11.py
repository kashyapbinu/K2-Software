"""
Structures audit, November 2026: the analytical body check read the wrong
material, and the recovery loads hung the wrong mass after a flight.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from physics.structures import compute_for_condition, get_material
from structures.solvers.base import STRUCTURAL_MATERIALS, get_structural_material
from structures.workstation import recovery_loads
from validation.cases.rocket_canonical import canonical_state


@pytest.mark.parametrize("name", list(STRUCTURAL_MATERIALS))
def test_every_material_the_structures_tab_lists_is_analysed_as_itself(name):
    """Kraft Phenolic, Plywood and ABS were analysed with aluminium's yield,
    modulus and expansion while the panel showed their own."""
    sm = STRUCTURAL_MATERIALS[name]
    mat = get_material(name)
    assert mat["yield"] == sm.yield_strength
    assert mat["E"] == sm.E
    assert mat["density"] == sm.density
    assert mat["poisson"] == sm.nu
    assert mat["cte"] == sm.cte


def test_a_phenolic_airframe_is_judged_on_its_own_yield():
    args = ("Max-Q", 0.102, 0.0025, 2.0)
    kw = dict(force=1400.0, mach=0.97, altitude_m=400.0, moment_arm_m=0.7)
    phenolic = compute_for_condition(*args, "Kraft Phenolic", **kw)
    aluminium = compute_for_condition(*args, "Aluminum 6061-T6", **kw)
    # Same loads, same section: the stress is the same and the safety factor
    # scales with the yield strength (60 MPa against 276 MPa).
    assert phenolic["von_mises"] == pytest.approx(aluminium["von_mises"])
    assert phenolic["safety_factor"] == pytest.approx(60e6 / phenolic["von_mises"])
    assert phenolic["safety_factor"] == pytest.approx(
        aluminium["safety_factor"] * 60e6 / 276e6)


def test_a_design_tab_material_name_reaches_its_structural_analogue():
    assert get_material("Cardboard")["yield"] == \
        get_structural_material("Cardboard").yield_strength
    assert get_material("Cardboard")["yield"] != get_material("Aluminum 6061-T6")["yield"]


def test_recovery_loads_hang_the_same_mass_before_and_after_a_flight():
    """After a flight the tank is empty, and the propellant was subtracted
    from the burnt-out mass a second time."""
    state = canonical_state()
    before = recovery_loads(state)
    state.sim_time = 230.0
    state.propellant_mass = 0.0008          # what the motor leaves behind
    after = recovery_loads(state)
    assert after.main_shock_N == pytest.approx(before.main_shock_N, rel=1e-6)
    assert after.main_deploy_velocity == pytest.approx(before.main_deploy_velocity, rel=1e-6)


# ── No invented numbers ─────────────────────────────────────────────────────

def _report(assembly=None):
    from structures.workstation import full_analysis
    return full_analysis(canonical_state(), assembly, None, "Aluminum 6061-T6", "Max-Q")


def test_the_failure_map_shows_only_what_was_analysed():
    """Couplers, Avionics Bay, Nose Cone, Motor Mount and Bulkheads carried
    safety factors scaled from the body's (x0.9, x1.2 ...)."""
    fm = _report().failure
    assert [c.name for c in fm.components] == ["Airframe", "Fins"]
    for name in ("Couplers", "Avionics Bay", "Nose Cone", "Motor Mount", "Bulkheads"):
        assert name in fm.not_modelled
    assert fm.weakest.name in ("Airframe", "Fins")


def test_a_finless_rocket_has_no_fin_tile():
    from structures.workstation import full_analysis
    state = canonical_state()
    state.fin_count = 0
    fm = full_analysis(state, None, None, "Aluminum 6061-T6").failure
    assert [c.name for c in fm.components] == ["Airframe"]


def test_recovery_reports_loads_and_no_safety_factor():
    """The safety factor was the load against a 5300 N eyebolt K2 assumed."""
    rl = recovery_loads(canonical_state())
    assert rl.peak_force_N > 0
    assert rl.safety_factor is None
    assert rl.eyebolt_load_N == rl.harness_tension_N
    assert not hasattr(rl, "allowable_N")


def test_an_unassessed_recovery_margin_is_left_out_of_the_score():
    from structures.workstation import safety_score
    scored = safety_score(3.0, 3.0, None, 1.0, 3.0)
    assert "recovery" not in scored.subscores
    # yield .30, buckling .25, thermal .15 at 100 and fin .15 at 40, over .85
    assert scored.score == round((100 * 0.70 + 40 * 0.15) / 0.85)
    assert safety_score(3.0, 3.0, 0.5, 3.0, 3.0).score < 100   # a real one counts


def test_the_load_path_is_the_designs_own_masses():
    from validation.cases.rocket_canonical import canonical_assembly
    assembly = canonical_assembly()
    state = canonical_state()
    lp = _report(assembly).loads
    nose = next(c for c in assembly.all_components() if type(c).__name__ == "NoseCone")
    total = assembly.total_mass() + state.propellant_mass_initial + state.motor_dry_mass
    joint = lp.stations[0]
    assert joint.position == pytest.approx(nose.component_length())
    assert joint.force_N == pytest.approx(lp.peak_force_N * nose.computed_mass() / total)
    assert lp.stations[-1].force_N == lp.peak_force_N == state.motor_max_thrust

    # A heavier nose puts more force through the joint behind it.
    nose.override_mass = nose.computed_mass() + 1.0
    assert _report(assembly).loads.stations[0].force_N > joint.force_N


def test_no_design_means_no_load_path():
    assert _report(None).loads.stations == []


def test_the_optimiser_checks_the_rockets_own_wall_and_material():
    """It checked a 2 mm aluminium tube whatever the rocket was."""
    from core.batch_simulation import BatchSimConfig
    from core.optimization_engine import airframe_safety_factor
    from physics.structures import compute_all
    state = canonical_state()
    cfg = BatchSimConfig.from_rocket_state(state)
    assert cfg.wall_thickness == state.wall_thickness
    assert cfg.material_name == state.material_name
    own = compute_all(state.motor_max_thrust, state.diameter, state.wall_thickness,
                      state.length, state.material_name)["safety_factor"]
    assert airframe_safety_factor(cfg) == pytest.approx(own)
    cfg.material_name = "Kraft Phenolic"
    assert airframe_safety_factor(cfg) == pytest.approx(own * 60e6 / 276e6)
    # A thinner candidate wall is a weaker one.
    assert airframe_safety_factor(cfg, {"wall_thickness": 0.001}) < airframe_safety_factor(cfg)


@pytest.mark.parametrize("field,value", [("wall_thickness", 0.0), ("material_name", ""),
                                         ("motor_max_thrust", 0.0)])
def test_an_airframe_that_cannot_be_checked_does_not_pass(field, value):
    from core.batch_simulation import BatchSimConfig
    from core.optimization_engine import airframe_safety_factor
    cfg = BatchSimConfig.from_rocket_state(canonical_state())
    setattr(cfg, field, value)
    assert airframe_safety_factor(cfg) == 0.0


def test_structures_knows_when_there_is_no_flight_to_take_loads_from():
    """Without a flight the tabs were filled in for Mach 0.6 at 3000 m."""
    from structures.workstation import flight_known
    state = canonical_state()
    assert not flight_known(state, None)
    state.max_velocity, state.max_mach = 330.0, 0.97     # what a flight leaves
    assert flight_known(state, None)


def test_a_fin_with_no_thickness_is_not_analysed_as_three_millimetres():
    from structures.workstation import FlightLoads, fin_analysis
    state = canonical_state()
    state.fin_thickness = 0.0
    fa = fin_analysis(state, FlightLoads(max_dynamic_pressure=60000.0, max_velocity=300.0),
                      "Aluminum 6061-T6")
    assert not fa.present and fa.root_bending_MPa == 0.0


def test_a_finless_tube_has_no_panel_between_fins_to_buckle():
    from structures.workstation import FlightLoads, buckling_analysis
    state = canonical_state()
    fl = FlightLoads(max_thrust=1400.0, max_dynamic_pressure=60000.0)
    assert "Panel Buckling" in [m.name for m in buckling_analysis(state, fl, "Aluminum 6061-T6").modes]
    state.fin_count = 0
    assert "Panel Buckling" not in [m.name for m in buckling_analysis(state, fl, "Aluminum 6061-T6").modes]


# ── Recovery hardware rating (entered by the user) ──────────────────────────

def test_the_recovery_safety_factor_is_the_entered_rating_over_the_harness_tension():
    state = canonical_state()
    assert recovery_loads(state).safety_factor is None
    state.recovery_hardware_rating = 2000.0
    rl = recovery_loads(state)
    assert rl.rating_N == 2000.0
    assert rl.safety_factor == pytest.approx(2000.0 / rl.harness_tension_N)
    state.recovery_hardware_rating = 2.0 * rl.harness_tension_N
    assert recovery_loads(state).status == "SAFE"
    state.recovery_hardware_rating = 0.5 * rl.harness_tension_N
    assert recovery_loads(state).status == "FAILURE"


def test_the_hardware_rating_is_saved_and_marks_structures_out_of_date():
    from core.readiness import INPUT_GROUPS
    from core.rocket_state import RocketState
    state = canonical_state()
    assert state.recovery_hardware_rating == 0.0          # old projects: not entered
    state.recovery_hardware_rating = 3500.0
    assert RocketState.from_dict(state.to_dict()).recovery_hardware_rating == 3500.0
    assert any("recovery_hardware_rating" in names for names in INPUT_GROUPS.values())


# ── Max-Q: the flight's drag, not Cd = 0.5 ──────────────────────────────────

def test_max_q_compression_is_the_larger_of_thrust_and_the_given_drag():
    import math
    d, t = 0.102, 0.0025
    area = math.pi * d * t
    kw = dict(mach=0.97, altitude_m=400.0, moment_arm_m=0.7)
    args = ("Max-Q", d, t, 2.0, "Aluminum 6061-T6")
    assert compute_for_condition(*args, force=421.0, drag_force_N=342.0, **kw)["axial"] == \
        pytest.approx(421.0 / area)
    assert compute_for_condition(*args, force=0.0, drag_force_N=342.0, **kw)["axial"] == \
        pytest.approx(342.0 / area)             # coasting: drag alone, once
    assert compute_for_condition(*args, force=421.0, **kw)["axial"] == pytest.approx(421.0 / area)
    # The recorded dynamic pressure drives the bending when it is given.
    lo = compute_for_condition(*args, force=421.0, q_dyn=30000.0, **kw)["bending"]
    hi = compute_for_condition(*args, force=421.0, q_dyn=60000.0, **kw)["bending"]
    assert hi == pytest.approx(2.0 * lo)


@pytest.fixture(scope="module")
def flown():
    """The canonical rocket, flown: (state, history)."""
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.sim.headless_runner import _ensure_qapp
    _ensure_qapp()
    eng = RocketStateEngine(canonical_state())
    sim = SimulationEngine(eng)
    sim.start()
    while sim._running:
        sim._step()
    return eng.state, sim.history


@pytest.mark.sim
def test_the_panel_and_the_worst_case_table_agree_on_max_q(flown):
    """7.5 MPa on the panel against 5.3 MPa in the table, for one event."""
    from structures.workstation import find_worst_case, full_analysis
    state, history = flown
    panel = full_analysis(state, None, history, "Aluminum 6061-T6", "Max-Q").body_condition
    table = next(e for e in find_worst_case(state, history, "Aluminum 6061-T6").events
                 if e.name == "Max-Q")
    assert panel["von_mises"] == pytest.approx(table.von_mises, rel=1e-3)


@pytest.mark.sim
def test_the_aero_model_drag_is_the_drag_the_flight_recorded(flown):
    import math
    from structures.workstation import FlightLoads, aero_drag
    state, history = flown
    fl = FlightLoads.from_history(history, state)
    assert fl.maxq_drag > 0
    assert aero_drag(state, fl.maxq_mach, fl.maxq_altitude) == pytest.approx(fl.maxq_drag, rel=0.02)
    # and it is not half the dynamic pressure on the frontal area
    assert fl.maxq_drag != pytest.approx(
        0.5 * fl.max_dynamic_pressure * math.pi * (state.diameter / 2) ** 2, rel=0.1)


@pytest.mark.sim
def test_a_parachute_does_not_buckle_the_airframe(flown):
    from structures.workstation import find_worst_case
    state, history = flown
    events = {e.name: e for e in find_worst_case(state, history, "Aluminum 6061-T6").events}
    assert events["Parachute Deployment"].buckling_margin == 0.0
    assert events["Max-Q"].buckling_margin > 0.0


# ── Thermal: the wall is heated over the flight ─────────────────────────────

def _steady(seconds, n=400, speed=600.0, mach=2.0, air_T=250.0, rho=0.6):
    times = [seconds * i / (n - 1) for i in range(n)]
    return times, [speed] * n, [mach] * n, [air_T] * n, [rho] * n


@pytest.mark.parametrize("name", ["Aluminum 6061-T6", "Fiberglass (G10)"])
def test_the_heat_that_enters_the_wall_is_the_heat_it_holds(name):
    import numpy as np
    from structures.workstation import wall_heating
    mat = STRUCTURAL_MATERIALS[name]
    wall = 0.003
    times, *air = _steady(5.0)
    w = wall_heating(times, *air, x_station=0.3, thickness=wall, mat=mat)
    dt = times[1] - times[0]
    entered = float(np.sum(w["flux"][1:]) * dt)                  # J/m^2
    held = mat.density * mat.specific_heat * wall * (w["mean"][-1] - w["mean"][0])
    assert held == pytest.approx(entered, rel=1e-6)
    assert w["outer"][-1] > w["inner"][-1] > 250.0 - 1e-9


def test_a_wall_held_at_speed_long_enough_reaches_the_recovery_temperature():
    from structures.thermal_analysis import recovery_temperature
    from structures.workstation import wall_heating
    times, *air = _steady(2000.0, n=2000)
    w = wall_heating(times, *air, x_station=0.3, thickness=0.002,
                     mat=STRUCTURAL_MATERIALS["Aluminum 6061-T6"])
    assert w["outer"][-1] == pytest.approx(recovery_temperature(250.0, 2.0), abs=1.0)
    assert w["outer"][-1] - w["inner"][-1] < 0.05      # soaked through: no gradient


@pytest.mark.sim
def test_a_short_burn_warms_the_skin_far_less_than_the_recovery_temperature(flown):
    """The skin was put at the steady recovery temperature, 333.6 K here."""
    from structures.thermal_analysis import recovery_temperature
    from structures.workstation import FlightLoads, thermal_margin, thermal_profile
    state, history = flown
    fl = FlightLoads.from_history(history, state)
    tp = thermal_profile(state, fl, history, "Aluminum 6061-T6")
    assert tp.available
    recovery = recovery_temperature(history.get_values("atm_temperature")[0], fl.max_mach)
    rise = tp.skin_temp_K - tp.initial_temp_K
    assert 2.0 < rise < 0.4 * (recovery - tp.initial_temp_K)
    # Nearest the nose is hottest, and the peak stress is in the boost, not
    # in the cold air at apogee (it was reported at 2594 m of a 2612 m flight).
    temps = [T for _, T, _ in tp.body_station_temps]
    assert temps == sorted(temps, reverse=True)
    _, apogee, _ = history.find_max("altitude")
    assert tp.max_stress_altitude < 0.5 * apogee
    assert tp.max_thermal_stress_MPa < 5.0             # was 62 MPa
    assert thermal_margin(tp) == pytest.approx(
        (tp.service_limit_K - tp.initial_temp_K) / rise)
    # Asked again for the same flight, the kept answer comes back.
    assert thermal_profile(state, fl, history, "Aluminum 6061-T6") is tp


def test_no_flight_history_means_no_thermal_result_and_no_thermal_score():
    from structures.workstation import (FlightLoads, safety_score, thermal_margin,
                                        thermal_profile)
    tp = thermal_profile(canonical_state(), FlightLoads(), None, "Aluminum 6061-T6")
    assert not tp.available and tp.body_station_temps == [] and tp.profile == []
    assert thermal_margin(tp) is None
    assert "thermal" not in safety_score(3.0, 3.0, None, 3.0, None).subscores


def test_the_thermal_condition_is_the_stress_of_the_wall_temperature_difference():
    args = ("Thermal", 0.102, 0.0025, 2.0, "Aluminum 6061-T6")
    mat = get_material("Aluminum 6061-T6")
    r = compute_for_condition(*args, mach=2.0, altitude_m=5000.0, wall_delta_T=10.0)
    assert r["thermal"] == pytest.approx(mat["E"] * mat["cte"] * 10.0 / (1 - mat["poisson"]))
    # No known difference through the wall: no stress, whatever the Mach.
    still = compute_for_condition(*args, mach=2.0, altitude_m=5000.0)
    assert still["thermal"] == 0.0 and still["von_mises"] == 0.0


# ── The Structures tab itself ───────────────────────────────────────────────

_UI = textwrap.dedent("""
    import sys, os, tempfile
    sys.path.insert(0, {root!r})
    import pyvistaqt
    class _NoGL:                                   # no OpenGL off-screen
        def __init__(self, *a, **k): raise RuntimeError("no GL")
    pyvistaqt.QtInteractor = _NoGL
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    app = QApplication([])
    import ui.settings as st
    ini = os.path.join(tempfile.mkdtemp(), "k2.ini")
    st._store = lambda: QSettings(ini, QSettings.Format.IniFormat)
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.cases.rocket_canonical import canonical_state
    from ui.workspaces.structures_workspace import StructuresWorkspace

    eng = RocketStateEngine(canonical_state())
    sim = SimulationEngine(eng)
    ws = StructuresWorkspace(eng)
    ws.sim_engine = sim                 # what _get_history looks for on the window
    ws._has_run = True

    # No flight yet: nothing is filled in for a stand-in one.
    ws._refresh()
    assert "Fly the rocket" in ws._status.text(), ws._status.text()
    assert ws._tab_recovery.currentIndex() == 0

    sim.start()
    while sim._running:
        sim._step()
    ws._refresh()
    assert ws._tab_recovery.currentIndex() == 1
    assert ws.lbl_rec_sf.text() == "Enter rating", ws.lbl_rec_sf.text()

    # The load factor shown is the sensed peak, the one the Results tab and
    # the report give. It was the peak of d|v|/dt from the recorder, about
    # 1 g less on the way up.
    ws._refresh_flight_indicator()
    sensed_g = sim.summary.max_acceleration / 9.80665
    assert ws.lbl_fl_a.text() == f"{{sensed_g:.1f}} G", (ws.lbl_fl_a.text(), sensed_g)
    kinematic_g = sim.history.find_max("acceleration")[1] / 9.80665
    assert sensed_g - kinematic_g > 0.5, (sensed_g, kinematic_g)

    # The load-factor estimate under the Mach / altitude boxes is this
    # rocket's drag there, from the aero model the flight uses. It was
    # q x state.cd x frontal area: 0.45 for every rocket, and the last
    # instant's drag coefficient once the rocket had flown.
    from structures import workstation as wks
    ws.chk_cfd_map.setChecked(False)
    ws.sp_mach.setValue(1.5); ws.sp_alt.setValue(3000.0)
    ws._update_q()
    expected_g = wks.aero_drag(eng.state, 1.5, 3000.0) / (wks.vehicle_mass(eng.state) * 9.81)
    assert ws.lbl_lf.text() == f"{{expected_g:.1f}} G", (ws.lbl_lf.text(), expected_g)
    import math
    flat_g = (0.5 * 0.9093 * (1.5 * 328.6) ** 2 * 0.45 * math.pi * (eng.state.diameter / 2) ** 2
              / (wks.vehicle_mass(eng.state) * 9.81))
    assert abs(expected_g - flat_g) > 0.5, (expected_g, flat_g)
    assert ws.sp_rec_rating.value() == 0.0

    # Widget -> state -> result.
    ws.sp_rec_rating.setValue(2000.0)
    assert eng.state.recovery_hardware_rating == 2000.0
    tension = float(ws.lbl_rec_harness.text().split()[0])
    assert abs(float(ws.lbl_rec_sf.text()) - 2000.0 / tension) < 0.02, ws.lbl_rec_sf.text()

    # State -> widget (an opened project shows its rating).
    eng.update(recovery_hardware_rating=3500.0)
    assert ws.sp_rec_rating.value() == 3500.0

    # The failure map shows only what was analysed.
    assert sorted(ws._fail_buttons) == ["Airframe", "Fins"], ws._fail_buttons
    assert "Not modelled" in ws._fail_detail.text()
    print("OK")
""")


_NO_FLIGHT = _UI.split("ws._has_run = True")[0] + textwrap.dedent("""
    ws._has_run = True

    # Before any flight a flight load case gives no stress and no verdict.
    # It used to show "Safe", SF 207.92, from the 500 N default force.
    for cond in ("Max Thrust", "Max-Q", "Recovery Shock", "Thermal"):
        ws.lc_combo.setCurrentText(cond)
        ws._refresh()
        assert ws.lbl_verdict.text() == "No Flight", (cond, ws.lbl_verdict.text())
        assert ws.lbl_sf.text() == "—", (cond, ws.lbl_sf.text())
        assert ws.lbl_vm.text() == "—", (cond, ws.lbl_vm.text())
        assert "Fly the rocket" in ws._status.text()

    # The FE run refuses too, so Readiness is never marked current on it.
    started = []
    eng.analysis_started = lambda key: started.append(key)
    ws.lc_combo.setCurrentText("Max Thrust")
    ws._get_assembly = lambda: object()
    ws._run_static()
    assert started == [], started

    # Custom judges only what the user typed, so it still runs.
    ws.lc_combo.setCurrentText("Custom")
    ws._refresh()
    assert ws.lbl_verdict.text() != "No Flight"
    assert ws.lbl_sf.text() != "—"

    # After a flight the flight cases are judged again.
    sim.start()
    while sim._running:
        sim._step()
    ws._get_assembly = lambda: None     # no design: the load path has no joints
    ws.lc_combo.setCurrentText("Max-Q")
    ws._refresh()
    assert ws.lbl_verdict.text() in ("Safe", "Adequate", "Marginal", "Failure"), ws.lbl_verdict.text()

    # Reset drops the flight, and the margins worked from its loads go with
    # it. The workstation report stayed, so the technical report went on
    # printing a structural safety factor for a flight that no longer existed.
    assert ws._wks_report is not None
    assert ws._tab_fin.currentIndex() == 1 and ws._tab_recovery.currentIndex() == 1
    eng.reset_flight()
    assert ws._wks_report is None
    assert ws.lbl_verdict.text() == "No Flight", ws.lbl_verdict.text()
    # ...and the tabs worked from that flight go back to "No Results
    # Available". They kept the dropped flight's fin loads, opening shocks
    # and buckling margins on screen until the next flight landed.
    for name in ("_tab_fin", "_tab_recovery", "_tab_buckling", "_tab_failure",
                 "_tab_loadpath", "_tab_deform"):
        assert getattr(ws, name).currentIndex() == 0, name
    assert ws.lbl_score.text() == "Not run", ws.lbl_score.text()
    print("OK")
""")


@pytest.mark.sim
def test_no_flight_means_no_structural_verdict_for_a_flight_load_case():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    proc = subprocess.run([sys.executable, "-c", _NO_FLIGHT.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]


@pytest.mark.sim
def test_structures_tab_asks_for_the_hardware_rating_and_waits_for_a_flight():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
