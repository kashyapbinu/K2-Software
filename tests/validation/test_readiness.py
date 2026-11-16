"""
A result is only a result for the rocket it was computed on.

Every tab kept showing its last answer after the design changed: a flutter
speed for the old fin, an apogee for the old motor, and a technical report
that printed both. Nothing said which rocket a number belonged to.

core.readiness stamps each analysis with a fingerprint of the inputs it read.
These tests pin what makes a result stale, what does not, and that the
Readiness board shows it.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import readiness as R
from core.rocket_state import FLIGHT_FIELDS, RocketState
from validation.cases.rocket_canonical import canonical_state

ROOT = Path(__file__).resolve().parents[2]


def _status(tracker, key, state):
    return tracker.status(key, state).status


def test_an_analysis_that_was_never_run_says_so():
    t = R.ReadinessTracker()
    assert _status(t, "flight", canonical_state()) == R.NOT_RUN


def test_a_result_is_current_until_an_input_it_read_changes():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s, "Apogee 2,610 m")
    assert _status(t, "flight", s) == R.CURRENT
    s.fin_root_chord *= 1.1
    st = t.status("flight", s)
    assert st.status == R.STALE
    assert st.changed == ["geometry"]
    assert "geometry changed" in st.reason
    assert st.headline == "Apogee 2,610 m"       # still shown, as the old result


def test_putting_the_input_back_makes_the_result_current_again():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flutter", s)
    old = s.fin_thickness
    s.fin_thickness = old * 2
    assert _status(t, "flutter", s) == R.STALE
    s.fin_thickness = old
    assert _status(t, "flutter", s) == R.CURRENT


def test_wind_makes_the_flight_stale_but_not_cfd_or_flutter():
    s, t = canonical_state(), R.ReadinessTracker()
    for key in ("flight", "monte_carlo", "cfd", "flutter", "structures"):
        t.complete(key, s)
    s.wind_speed += 5.0
    assert _status(t, "flight", s) == R.STALE
    assert _status(t, "monte_carlo", s) == R.STALE
    assert _status(t, "cfd", s) == R.CURRENT
    assert _status(t, "flutter", s) == R.CURRENT
    # Structures reads no wind itself, but its loads are that flight's.
    st = t.status("structures", s)
    assert st.status == R.STALE and st.changed == [] and st.upstream == ["Flight"]


def test_a_material_change_reaches_structures_and_flutter_only():
    s, t = canonical_state(), R.ReadinessTracker()
    for key in ("flight", "cfd", "flutter", "structures"):
        t.complete(key, s)
    s.elastic_modulus *= 0.5
    assert _status(t, "structures", s) == R.STALE
    assert _status(t, "flutter", s) == R.STALE
    assert _status(t, "flight", s) == R.CURRENT
    assert _status(t, "cfd", s) == R.CURRENT


def test_a_long_run_is_stamped_with_the_inputs_it_started_on():
    """A CFD run takes minutes. Editing the nose while it runs must leave the
    finished result stale, not stamped with the nose it never saw."""
    s, t = canonical_state(), R.ReadinessTracker()
    t.begin("cfd", s)
    assert t.status("cfd", s).running
    s.nose_length *= 1.5
    t.complete("cfd", s, "Cd 0.412")
    st = t.status("cfd", s)
    assert st.status == R.STALE and not st.running


def test_a_failed_run_leaves_the_previous_result_standing():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("cfd", s, "Cd 0.412")
    t.begin("cfd", s)
    t.abandon("cfd")
    st = t.status("cfd", s)
    assert st.status == R.CURRENT and st.headline == "Cd 0.412" and not st.running


def test_the_rules_check_goes_stale_with_the_flight_it_judged():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s)
    t.complete("rules", s, "5 of 5 rules pass", verdict=True)
    assert _status(t, "rules", s) == R.CURRENT
    s.launch_rod_length += 1.0
    st = t.status("rules", s)
    assert st.status == R.STALE and st.upstream == ["Flight"]


def test_a_report_is_out_of_date_once_an_analysis_in_it_is_re_run():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s)
    t.complete("flutter", s)
    t.complete("report", s, "rocket_report.pdf")
    assert _status(t, "report", s) == R.CURRENT
    t.complete("flutter", s)                     # same inputs, newer result
    st = t.status("report", s)
    assert st.status == R.STALE and st.upstream == ["Flutter"]


def test_structures_goes_out_of_date_when_the_flight_it_took_loads_from_is_re_flown():
    """The Structures load cases are the flight's (peak thrust, max-Q, the
    opening shocks). The stage had no link to the flight, so a wind or
    launch-angle change followed by a new flight left it 'current'."""
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s)
    t.complete("structures", s, "Safety factor 3.10", verdict=True)
    assert _status(t, "structures", s) == R.CURRENT
    s.wind_speed += 6.0                           # the flight is now stale
    st = t.status("structures", s)
    assert st.status == R.STALE and st.upstream == ["Flight"]
    t.complete("flight", s)                       # re-flown: newer than the FE run
    st = t.status("structures", s)
    assert st.status == R.STALE and st.upstream == ["Flight"]
    t.complete("structures", s)
    assert _status(t, "structures", s) == R.CURRENT


def test_a_parachute_change_reaches_structures():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("structures", s)
    s.main_cd_area += 1.0                         # a bigger opening shock
    st = t.status("structures", s)
    assert st.status == R.STALE and st.changed == ["recovery"]


def test_a_result_built_on_a_flight_is_out_of_date_once_that_flight_is_dropped():
    """Reset drops the flight. What was built from it used to stay 'current'
    (a result missing upstream was read as 'built without it')."""
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s)
    t.complete("structures", s)
    t.complete("report", s, "rocket_report.pdf")
    t.forget("flight")
    for key in ("structures", "report"):
        st = t.status(key, s)
        assert st.status == R.STALE and "Flight" in st.upstream, (key, st.upstream)
    # One run with no flight at all (typed-in loads) is not stale for it.
    t2 = R.ReadinessTracker()
    t2.complete("structures", s)
    assert _status(t2, "structures", s) == R.CURRENT


def test_a_report_written_without_an_analysis_is_not_stale_for_lacking_it():
    s, t = canonical_state(), R.ReadinessTracker()
    t.complete("flight", s)
    t.complete("report", s)
    assert _status(t, "report", s) == R.CURRENT   # it says "Not run" inside


def test_no_input_group_holds_a_field_a_flight_writes():
    """Otherwise a flight would mark itself stale as it flew."""
    names = [n for group in R.INPUT_GROUPS.values() for n in group]
    assert not set(names) & set(FLIGHT_FIELDS)
    assert not {"propellant_mass", "cg", "cp", "stability_margin"} & set(names)
    assert len(names) == len(set(names)), "a field is in two groups"
    known = set(RocketState.__dataclass_fields__)
    assert set(names) <= known, set(names) - known


def test_a_value_that_went_through_a_spin_box_is_the_same_value():
    a, b = canonical_state(), canonical_state()
    b.diameter = float(f"{a.diameter:.12f}") * (1 + 1e-13)
    assert R.fingerprint(a) == R.fingerprint(b)


def test_an_unknown_stage_is_an_error_not_a_silent_new_one():
    with pytest.raises(KeyError, match="unknown readiness stage"):
        R.ReadinessTracker().complete("fluter", canonical_state())


def test_the_next_step_names_what_is_missing_then_what_is_stale():
    t = R.ReadinessTracker()
    assert "no motor" in t.next_step(RocketState()).lower()
    s = canonical_state()
    assert t.next_step(s).startswith("Run Flight")
    t.complete("flight", s)
    assert t.next_step(s).startswith("Run Monte Carlo")
    s.dry_mass += 0.5
    step = t.next_step(s)
    assert step.startswith("Re-run Flight") and "mass changed" in step


# ── The engines and the board (needs Qt; own process, see CLAUDE.md §3) ──

_FLIGHT_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core import readiness as R
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.cases.rocket_canonical import canonical_state
    from ui.workspaces.readiness_workspace import ReadinessWorkspace, status_word

    engine = RocketStateEngine()
    engine.set_state(canonical_state())
    sim = SimulationEngine(engine)
    board = ReadinessWorkspace(engine)
    tile = board.tiles["flight"]
    assert tile.lbl_status.text() == "Not Run"
    assert board.lbl_next.text().startswith("Next: Run Flight"), board.lbl_next.text()

    # Fly it to the ground, stepping the engine directly.
    sim.start()
    assert engine.readiness.status("flight", engine.state).running
    assert tile.lbl_status.text() == "Running"
    sim._timer.stop()
    for _ in range(400000):
        if not sim._running:
            break
        sim._step()
    assert sim.summary is not None and sim.summary.final_phase == "Landed", sim.summary

    # The finished flight is current: flying did not change its own inputs.
    st = engine.readiness.status("flight", engine.state)
    assert st.status == R.CURRENT, (st.status, st.changed)
    assert st.headline.startswith("Apogee "), st.headline
    assert tile.lbl_status.text() == "Current", tile.lbl_status.text()
    assert tile.lbl_headline.text() == st.headline

    # Change a fin: the board says the flight is out of date, and why.
    engine.update(fin_root_chord=engine.state.fin_root_chord * 1.2)
    assert tile.lbl_status.text() == "Out of Date", tile.lbl_status.text()
    assert "eometry changed" in tile.lbl_reason.text(), tile.lbl_reason.text()
    assert "Re-run Flight" in board.lbl_next.text(), board.lbl_next.text()

    # Putting the rocket back on the pad removes the flight, so there is no
    # result left to be current or stale.
    engine.reset_flight()
    assert engine.readiness.status("flight", engine.state).status == R.NOT_RUN
    assert tile.lbl_status.text() == "Not Run"

    # Opening another project starts with nothing run.
    engine.analysis_done("flutter", "Flutter at 400 m/s", True)
    engine.set_state(canonical_state())
    assert all(s.status == R.NOT_RUN for s in engine.readiness.all(engine.state))

    # A tile's button asks for the tab that produces the result.
    asked = []
    board.open_tab.connect(asked.append)
    board.tiles["flutter"].btn_open.click()
    assert asked == ["Dynamics"], asked
    print("OK")
""")


def _run(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    return subprocess.run([sys.executable, "-c", script.format(root=str(ROOT))],
                          capture_output=True, text=True, timeout=300, env=env)


def test_a_flight_is_stamped_shown_and_goes_out_of_date_on_the_board():
    r = _run(_FLIGHT_SCRIPT)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_every_stage_opens_a_tab_the_main_window_has():
    from ui.main_window import MainWindow
    assert {s.tab for s in R.STAGES} <= set(MainWindow.TAB_NAMES)
    assert MainWindow.TAB_NAMES[0] == "Readiness"
    assert len(MainWindow.TAB_NAMES) == len(MainWindow.TAB_ICONS)
