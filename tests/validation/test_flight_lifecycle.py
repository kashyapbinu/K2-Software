"""
A flight belongs to the rocket that flew it, and ends when that rocket goes.

The flight lived in three places: the flight fields on the state, and on the
simulation engine its recorder (``history``), its ``summary`` and the replay
clock. Opening a project or pressing Reset cleared only the first. So after
Open Project the Results tab offered the previous rocket's 28,642 points, the
Mission Visualizer replayed its flight, and the technical report printed the
new rocket with the old rocket's apogee, while the Readiness board said the
flight was not run.

Each test runs in a subprocess: widgets need a GUI QApplication, and the
headless sim tests put a bare QCoreApplication in this process.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_PRELUDE = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.cases.rocket_canonical import canonical_state
    from ui.replay_clock import clock_for
    from ui.workspaces.results_workspace import ResultsWorkspace

    engine = RocketStateEngine()
    engine.set_state(canonical_state())
    sim = SimulationEngine(engine)
    clock = clock_for(sim)
    results = ResultsWorkspace(engine, sim)

    def fly():
        sim.start()
        sim._timer.stop()
        for _ in range(400000):
            if not sim._running:
                break
            sim._step()
        assert sim.summary.final_phase == "Landed", sim.summary.final_phase
        assert sim.history.count > 100 and clock.replay is not None

    def flight_stage():
        return engine.readiness.status("flight", engine.state)

    def assert_no_flight(where):
        assert sim.summary is None, where + ": the summary outlived the flight"
        assert sim.history.count == 0, where + ": the recorder outlived the flight"
        assert clock.replay is None, where + ": the replay outlived the flight"
        st = flight_stage()
        assert st.status == "not_run" and not st.running, (where, st.status, st.running)
        assert results.report_inputs()["summary"] is None, where
        assert results.report_inputs()["history"] is None, where
""")

_OPEN_PROJECT = _PRELUDE + textwrap.dedent("""
    fly()
    results.refresh_plots()
    assert "Landed" in results.summary_label.text()

    other = canonical_state()
    other.name, other.dry_mass = "Other", other.dry_mass * 1.5
    engine.set_state(other)
    assert_no_flight("open project")
    assert "No simulation data" in results.summary_label.text(), results.summary_label.text()

    # A blank project (engine.reset) drops it too.
    engine.set_state(canonical_state())
    fly()
    engine.reset()
    assert_no_flight("blank project")
    print("OK")
""")

_RESET = _PRELUDE + textwrap.dedent("""
    import dataclasses
    pad = dataclasses.asdict(engine.state)
    pad_file = engine.state.to_dict()
    fly()
    results.refresh_plots()

    # A flight writes its own fields and the live balance, and nothing else.
    # It used to leave its last instant's drag coefficient in state.cd (1.57
    # under the canopy; this rocket's is 0.45). Monte Carlo then drew its drag
    # perturbation around 1.57, the optimiser took it as the "Cd Correction"
    # baseline, and the project file saved it with the last instant's stress,
    # safety factor and wall temperature.
    assert engine.state.cd == pad["cd"], engine.state.cd
    # Saved now, the project is the file it was before the flight: the
    # loaded rocket, with its own balance (not the burnt-out CG and margin).
    saved = engine.state.to_dict()
    assert saved == pad_file, sorted(k for k in pad_file if saved.get(k) != pad_file[k])

    # Stop with nothing running changes nothing. It used to rebuild the
    # summary as a stopped flight: Landed became Terminated and the landing
    # velocity None, with the board still saying Current.
    finished = []
    sim.sim_finished.connect(lambda: finished.append(1))
    landing_v = sim.summary.landing_velocity
    sim.stop()
    assert sim.summary.final_phase == "Landed" and sim.summary.landing_velocity == landing_v
    assert engine.state.sim_phase == "Landed", engine.state.sim_phase
    assert not finished, "a flight that was not running did not finish again"
    assert flight_stage().status == "current"

    engine.reset_flight()
    assert_no_flight("reset")
    assert "No simulation data" in results.summary_label.text(), results.summary_label.text()
    assert engine.state.motor_designation != "None", "Reset keeps the rocket"
    # Reset puts back exactly the rocket that stood on the pad.
    now = dataclasses.asdict(engine.state)
    assert now == pad, sorted(k for k in pad if now[k] != pad[k])
    print("OK")
""")

_REFUSED_START = _PRELUDE + textwrap.dedent("""
    fly()

    # A start that is refused (here: too heavy to leave the pad) has already
    # wiped the recorder. The board used to go on showing the old apogee.
    failed = []
    sim.sim_failed.connect(lambda title, detail: failed.append(title))
    mass = engine.state.dry_mass
    engine.update(dry_mass=5000.0)
    sim.start()
    assert failed and not sim.is_running, failed
    assert_no_flight("refused start")

    # Stopped before the first step: there is no flight, and the Flight stage
    # is not left "running" for the rest of the session.
    engine.update(dry_mass=mass)
    sim.start()
    sim._timer.stop()
    assert flight_stage().running
    sim.stop()
    st = flight_stage()
    assert st.status == "not_run" and not st.running, (st.status, st.running)
    assert sim.summary is None
    print("OK")
""")


def _run(script):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    r = subprocess.run([sys.executable, "-c", script.format(root=str(ROOT))],
                       capture_output=True, text=True, timeout=400, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


def test_opening_a_project_leaves_no_flight_from_the_previous_rocket():
    _run(_OPEN_PROJECT)


def test_reset_drops_the_flight_and_stop_does_not_rewrite_a_finished_one():
    _run(_RESET)


def test_a_refused_or_unstarted_flight_is_not_shown_as_a_result():
    _run(_REFUSED_START)
