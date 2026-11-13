"""
Technical report (core.technical_report): every flight number in it is the
FlightSummary's, an analysis that was not run says so, and the PDF builds
headless.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import rules as R
from core.landing_map import fit_dispersion
from core.technical_report import NOT_RUN, build_content, fmt, render_pdf
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import run_flight


@pytest.fixture(scope="module")
def flight():
    state = canonical_state()
    state.recovery_canopies = [
        {"name": "Drogue", "cd_area": 0.4, "event": "apogee", "altitude": 0.0, "delay": 1.0},
        {"name": "Main", "cd_area": 3.5, "event": "altitude", "altitude": 250.0, "delay": 0.0}]
    res = run_flight(state)
    return state, res.summary, res.history


@pytest.mark.sim
def test_flight_numbers_are_the_summary(flight):
    state, s, _ = flight
    rep = build_content(state, s)
    v = lambda label: rep.value("Flight", label)
    assert v("Apogee above pad") == fmt(s.apogee, "m", 0)
    assert v("Rail exit velocity") == fmt(s.rail_exit_velocity, "m/s", 1)
    assert v("Stability at rail exit") == fmt(s.stability_at_rail_exit, "cal", 2)
    assert v("Minimum stability (rail exit → apogee)") == fmt(s.min_stability, "cal", 2)
    assert v("Max sensed acceleration") == fmt(s.max_acceleration / 9.80665, "g", 1)
    assert v("Touchdown speed") == fmt(s.landing_velocity, "m/s", 1)
    rec = rep.section("Recovery")
    assert [row[0] for row in rec.table[1:]] == ["Drogue", "Main"]
    assert rec.table[2][1] == "250 m"
    assert rep.value("Recovery", "Peak opening shock (all canopies)") == \
        fmt(s.recovery_shock_force, "N", 0)


def test_what_was_not_run_says_so():
    rep = build_content(canonical_state(), None)
    assert NOT_RUN in rep.section("Flight").notes[0]
    assert NOT_RUN in rep.section("Landing dispersion (Monte Carlo)").notes[0]
    assert NOT_RUN in rep.value("Structural and flutter margins", "Fin flutter")
    assert NOT_RUN in rep.value("Structural and flutter margins", "Airframe structure")
    assert NOT_RUN in rep.section("Rules check").notes[0]
    assert rep.section("Assumptions and model limits").notes


def test_vehicle_and_launch_sections_read_the_state():
    s = canonical_state()
    s.launch_site_name, s.launch_altitude, s.launch_azimuth = "Pad 2", 1401.0, 45.0
    s.ground_temperature = 305.0
    rep = build_content(s, None)
    assert rep.value("Launch conditions", "Pad elevation") == "1,401 m"
    assert rep.value("Launch conditions", "Site") == "Pad 2"
    assert "305.0 K" in rep.value("Launch conditions", "Air")
    assert rep.value("Motor", "Designation") == s.motor_designation
    assert rep.value("Vehicle", "Liftoff mass") == fmt(s.total_mass(), "kg", 3)


_UI = textwrap.dedent("""
    import sys, os, tempfile
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QSettings
    app = QApplication([])
    import ui.settings as st
    ini = os.path.join(tempfile.mkdtemp(), "k2.ini")
    st._store = lambda: QSettings(ini, QSettings.Format.IniFormat)
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.results_workspace import ResultsWorkspace
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from validation.cases.rocket_canonical import canonical_state

    eng = RocketStateEngine(canonical_state())
    sim = SimulationEngine(eng)
    res = ResultsWorkspace(eng, sim)
    sw = SimulationWorkspace(eng, sim)
    sim.start()
    while sim._running:
        sim._step()
    res.refresh_plots()
    inp = res.report_inputs()
    assert inp["summary"] is sim.summary and "dispersion" not in inp
    path = os.path.join(tempfile.mkdtemp(), "r.pdf")
    res._on_report(path)
    assert open(path, "rb").read(4) == b"%PDF"

    st.set("display/units", "imperial")
    res.refresh_plots()
    sw._update_readouts(eng.state)
    assert res.summary_tiles["apogee"].text().endswith(" ft"), res.summary_tiles["apogee"].text()
    assert sw.readouts["Mass"].text().endswith(" lb")
    st.set("display/units", "SI")
    sw._update_readouts(eng.state)
    assert sw.readouts["Mass"].text().endswith(" kg")
    print("OK")
""")


@pytest.mark.sim
def test_results_tab_writes_the_report_and_follows_units():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]


@pytest.mark.sim
def test_full_pdf_builds(tmp_path, flight):
    state, s, history = flight
    pts = [(100.0 * i % 700 - 300, 37.0 * i % 500 - 200) for i in range(40)]
    rs = R.RuleSet("Ours", source="Rulebook 2027", confirmed=True,
                   rules=[R.Rule("Rail exit", "rail_exit_velocity", ">=", 10.0)])
    rep = build_content(state, s, history=history, rule_set=rs,
                        rule_results=R.evaluate_all(rs, s),
                        dispersion=fit_dispersion(pts), landings=pts,
                        margins={"flutter_speed": 450.0, "structural_sf": 2.4,
                                 "structures_source": "CalculiX"})
    assert rep.section("Rules check").table[1][3] == R.PASS
    assert rep.value("Structural and flutter margins", "Fin flutter speed (governing)") == "450 m/s"
    path = tmp_path / "report.pdf"
    render_pdf(rep, str(path))
    data = path.read_bytes()
    assert data[:4] == b"%PDF" and len(data) > 30_000
