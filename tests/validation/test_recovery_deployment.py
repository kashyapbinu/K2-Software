"""
Recovery: every parachute opens on its own trigger, in both engines.

Before this the flight had two fixed slots (a drogue at apogee + delay, a
main at one altitude), parachutes from the Design tab were squeezed into
them by size whatever each was set to do, OpenRocket deploy settings were
not imported, and the structures recovery check read the two slots through
``or 0.5`` / ``or 3.0`` — so a one-parachute design was checked against a
drogue and a main it does not have.
"""
import math
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.components import Parachute, RocketAssembly
from core.flight_summary import FlightSummary
from core.rocket_state import RocketStateEngine
from recovery.parachute_dynamics import (ALTITUDE, APOGEE, Canopy, RecoverySystem,
                                         recovery_phase_parameters)
from validation.cases.rocket_canonical import canonical_assembly, canonical_state
from validation.sim.headless_runner import _ensure_qapp, run_flight

DROGUE = {"name": "Drogue", "cd_area": 0.4, "event": "apogee", "altitude": 0.0, "delay": 1.0}
MAIN = {"name": "Main", "cd_area": 3.5, "event": "altitude", "altitude": 250.0, "delay": 0.0}


def _state(canopies):
    s = canonical_state()
    s.recovery_canopies = [dict(c) for c in canopies]
    return s


def _batch(canopies):
    return run_batch_simulation(BatchSimConfig.from_rocket_state(_state(canopies)), seed=1)


# ── The model ───────────────────────────────────────────────────────────────

def _fly_profile(system, apogee=1000.0, rate=20.0, dt=0.1):
    """Up at 50 m/s to apogee, then down at `rate`; returns the deploy log."""
    t, h, log = 0.0, 0.0, []
    while h < apogee:
        t += dt; h += 50.0 * dt
        log += [(c.name, t, h) for c in system.step(t, h, 50.0, 1.0, 50.0)]
    while h > 0:
        t += dt; h -= rate * dt
        log += [(c.name, t, h) for c in system.step(t, h, -rate, 1.0, rate)]
    return log


def test_dual_deploy_opens_each_canopy_on_its_trigger():
    sys_ = RecoverySystem.from_canopies([DROGUE, MAIN])
    log = _fly_profile(sys_)
    names = [n for n, _, _ in log]
    assert names == ["Drogue", "Main"]
    (_, t_d, _), (_, _, h_m) = log
    assert t_d == pytest.approx(sys_.apogee_time + 1.0, abs=0.11)
    assert 248.0 <= h_m <= 250.0
    drogue = sys_.canopies[0]
    assert drogue.descent_rate == pytest.approx(20.0)        # measured as the main opened


def test_a_single_main_at_apogee_and_an_altitude_only_main():
    at_apogee = RecoverySystem.from_canopies([{**MAIN, "event": "apogee"}])
    assert [n for n, _, _ in _fly_profile(at_apogee)] == ["Main"]
    assert at_apogee.canopies[0].deploy_altitude > 990.0

    low = RecoverySystem.from_canopies([MAIN])
    (_, _, h), = _fly_profile(low)
    assert 248.0 <= h <= 250.0


def test_an_altitude_trigger_above_apogee_opens_at_apogee():
    sys_ = RecoverySystem.from_canopies([{**MAIN, "altitude": 5000.0}])
    (_, _, h), = _fly_profile(sys_, apogee=1000.0)
    assert h > 990.0


def test_zero_area_canopies_are_not_canopies():
    sys_ = RecoverySystem.from_canopies([{**DROGUE, "cd_area": 0.0}, MAIN])
    assert [c.name for c in sys_.canopies] == ["Main"]
    assert recovery_phase_parameters(sys_) == (0.0, 250.0)


def test_unknown_event_is_refused():
    with pytest.raises(ValueError):
        Canopy("x", 1.0, "ejection")
    cfg = BatchSimConfig.from_rocket_state(_state([{**DROGUE, "event": "ejection"}]))
    res = run_batch_simulation(cfg, seed=1)
    assert not res.success and "deploy event" in res.failure_reasons[0]


def test_the_two_slots_still_fly_when_the_design_has_no_parachute():
    s = canonical_state()
    s.recovery_canopies = []
    sys_ = RecoverySystem.for_state(s)
    assert [(c.name, c.event) for c in sys_.canopies] == [("Drogue", APOGEE), ("Main", ALTITUDE)]
    assert sys_.canopies[1].altitude == s.main_deploy_altitude


# ── Flight ──────────────────────────────────────────────────────────────────

@pytest.mark.sim
@pytest.mark.parametrize("canopies", [
    [DROGUE, MAIN],
    [{**MAIN, "event": "apogee", "delay": 2.0}],
    [MAIN],
], ids=["dual-deploy", "main-at-apogee", "main-at-altitude-only"])
def test_both_engines_deploy_the_same_way(canopies):
    a = run_flight(_state(canopies)).summary
    b = FlightSummary.from_batch(_batch(canopies))
    assert len(a.canopies) == len(b.canopies) == len(canopies)
    for ca, cb in zip(a.canopies, b.canopies):
        assert ca["name"] == cb["name"]
        assert ca["deploy_altitude"] == pytest.approx(cb["deploy_altitude"], rel=0.02, abs=3.0)
        assert ca["deploy_time"] == pytest.approx(cb["deploy_time"], rel=0.01, abs=0.2)
        assert ca["peak_force"] == pytest.approx(cb["peak_force"], rel=0.10)
    assert a.landing_velocity == pytest.approx(b.landing_velocity, rel=0.03)


@pytest.mark.sim
def test_a_main_only_altitude_deploy_falls_fast_then_lands_soft():
    res = _batch([MAIN])
    main = res.canopies[0]
    assert main["deploy_speed"] > 50.0                    # free fall from apogee
    assert res.landing_velocity < 10.0
    assert main["peak_force"] > 500.0
    assert res.drogue_descent_rate is None                # nothing above it


@pytest.mark.sim
def test_dual_deploy_summary_reports_each_canopy():
    s = FlightSummary.from_batch(_batch([DROGUE, MAIN]))
    drogue, main = s.canopies
    assert s.drogue_deployed and s.main_deployed
    assert drogue["deploy_altitude"] > 2000.0
    assert 240.0 <= main["deploy_altitude"] <= 251.0
    assert s.drogue_descent_rate == pytest.approx(drogue["descent_rate"])
    assert main["descent_rate"] == pytest.approx(s.landing_velocity)
    assert s.recovery_shock_force >= max(drogue["peak_force"], main["peak_force"])


def test_canopies_that_open_together_measure_no_descent_rate_at_apogee():
    """Two canopies opening in one step: neither came down alone, so neither
    has a descent rate until touchdown. The first used to be stamped with
    |vz| at the instant the second opened, which at apogee is about zero."""
    sys_ = RecoverySystem.from_canopies([{**DROGUE, "delay": 0.0},
                                         {**MAIN, "event": "apogee"}])
    sys_.step(1.0, 1000.0, 1.0, 1.0, 1.0)
    assert len(sys_.step(1.1, 1000.0, -0.3, 1.0, 0.3)) == 2     # both open at apogee
    assert [c.descent_rate for c in sys_.canopies] == [None, None]
    sys_.touchdown(5.0)
    assert [c.descent_rate for c in sys_.canopies] == [5.0, 5.0]


@pytest.mark.sim
@pytest.mark.parametrize("canopies", [
    [{**DROGUE, "delay": 0.0}, {**MAIN, "event": "apogee"}],
    [{**DROGUE, "delay": 0.0}, {**MAIN, "altitude": 99999.0}],
], ids=["both-at-apogee", "main-trigger-above-apogee"])
def test_a_flight_with_no_drogue_only_descent_reports_no_drogue_rate(canopies):
    """The summary's drogue descent rate was 0.3 m/s here (the vertical speed
    at apogee), which a rules check then judged as the drogue's descent."""
    s = FlightSummary.from_batch(_batch(canopies))
    first, second = s.canopies
    assert first["deploy_time"] == second["deploy_time"]
    assert s.drogue_descent_rate is None
    assert first["descent_rate"] == pytest.approx(s.landing_velocity)
    assert second["descent_rate"] == pytest.approx(s.landing_velocity)


# ── Design, save/load, import ───────────────────────────────────────────────

def test_parachute_deploy_settings_save_and_reload():
    asm = canonical_assembly()
    chute = Parachute("Main")
    chute.deploy_event, chute.deploy_altitude, chute.deploy_delay = "altitude", 220.0, 0.5
    asm.add_component(asm.stages[0].children[1], chute)
    again = RocketAssembly.from_dict(asm.to_dict())
    back = next(c for c in again.all_components() if isinstance(c, Parachute))
    assert (back.deploy_event, back.deploy_altitude, back.deploy_delay) == ("altitude", 220.0, 0.5)


_ORK = """<?xml version='1.0' encoding='utf-8'?>
<openrocket version="1.9" creator="test">
 <rocket><name>R</name><subcomponents><stage><name>S</name><subcomponents>
  <nosecone><name>Nose</name><length>0.3</length><aftradius>0.05</aftradius>
   <shape>ogive</shape><thickness>0.002</thickness></nosecone>
  <bodytube><name>Body</name><length>1.0</length><radius>0.05</radius>
   <thickness>0.002</thickness><subcomponents>
    {chutes}
   </subcomponents></bodytube>
 </subcomponents></stage></subcomponents></rocket>
</openrocket>"""


def _chute(name, inner):
    return (f"<parachute><name>{name}</name><diameter>1.0</diameter><cd>1.5</cd>"
            f"{inner}</parachute>")


def test_openrocket_deploy_settings_are_imported(tmp_path, caplog):
    from import_export.ork_importer import import_ork

    chutes = "".join([
        _chute("Drogue", "<deployevent>apogee</deployevent><deploydelay>2</deploydelay>"),
        _chute("Main", "<deployevent>altitude</deployevent><deployaltitude>180</deployaltitude>"),
        _chute("Nested", "<deploymentconfiguration configid='a'><deployevent>altitude"
                         "</deployevent><deployaltitude>90</deployaltitude>"
                         "</deploymentconfiguration>"),
        _chute("Charge", "<deployevent>ejection</deployevent><deploydelay>1</deploydelay>"),
        _chute("Spare", "<deployevent>never</deployevent>"),
    ])
    p = tmp_path / "r.ork"
    p.write_text(_ORK.format(chutes=chutes), encoding="utf-8")
    asm = import_ork(str(p))
    got = {c.name: c for c in asm.all_components() if isinstance(c, Parachute)}
    assert (got["Drogue"].deploy_event, got["Drogue"].deploy_delay) == ("apogee", 2.0)
    assert (got["Main"].deploy_event, got["Main"].deploy_altitude) == ("altitude", 180.0)
    assert (got["Nested"].deploy_event, got["Nested"].deploy_altitude) == ("altitude", 90.0)
    assert got["Charge"].deploy_event == "apogee" and got["Charge"].deploy_delay == 1.0
    assert got["Spare"].cd_area() == 0.0 and got["Spare"].computed_mass() > 0.0
    from core.staging import assembly_recovery
    names = [r["name"] for r in assembly_recovery(asm)["recovery_canopies"]]
    assert names == ["Drogue", "Main", "Nested", "Charge"]          # "never" flies no drag
    assert any("ejection" in r.message for r in caplog.records)


# ── Structures ──────────────────────────────────────────────────────────────

def test_structures_checks_the_canopies_the_vehicle_has():
    from structures.workstation import recovery_loads

    s = canonical_state()
    s.max_altitude = 2600.0
    s.recovery_canopies = []
    s.drogue_cd_area = s.main_cd_area = 0.0
    none = recovery_loads(s)
    assert none.peak_force_N == 0.0 and none.canopies == [] and none.status == "No parachute"

    s.recovery_canopies = [MAIN]
    single = recovery_loads(s)
    assert [c["name"] for c in single.canopies] == ["Main"]
    assert single.main_shock_N == 0.0                       # no second canopy invented
    # Opens after a free fall from apogee to 250 m: a hard opening.
    assert single.canopies[0]["inflow_velocity"] > 100.0

    s.recovery_canopies = [MAIN, DROGUE]
    dual = recovery_loads(s)
    assert [c["name"] for c in dual.canopies] == ["Drogue", "Main"]   # opening order
    assert dual.canopies[1]["inflow_velocity"] < single.canopies[0]["inflow_velocity"]
    assert dual.peak_force_N < single.peak_force_N


# ── Simulation tab ──────────────────────────────────────────────────────────

_UI = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from validation.cases.rocket_canonical import canonical_state

    engine = RocketStateEngine(canonical_state())
    ws = SimulationWorkspace(engine, SimulationEngine(engine))
    assert ws.main_cda_spin.isEnabled() and ws.recovery_note.isHidden()
    engine.update(recovery_canopies=[{main!r}])
    assert not ws.main_cda_spin.isEnabled() and not ws.drogue_delay_spin.isEnabled()
    assert "Main" in ws.recovery_note.text() and "250 m" in ws.recovery_note.text()
    engine.update(recovery_canopies=[])
    assert ws.main_cda_spin.isEnabled()
    print("OK")
""")


def test_simulation_tab_locks_the_slots_when_the_design_has_parachutes():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root), main=MAIN)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
