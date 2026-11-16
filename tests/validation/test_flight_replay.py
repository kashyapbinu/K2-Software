"""
One flight, one clock, every view on the same instant.

The Mission Visualizer replayed from buffers of its own, five recorded frames
per timer tick, so playback ran fast through the coast and slow through the
boost (the integrator's step is not constant); its telemetry panel stayed on
touchdown while the rocket on screen was at max-Q; and the Results cursor was
a separate slider that moved nothing else.

core.flight_replay answers "what was happening at time t" from the flight
recorder; ui.replay_clock holds the instant, plays it in real time, and is
shared by both tabs.
"""
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.flight_replay import FlightReplay, body_axis
from core.history_manager import HISTORY_FIELDS, HistoryManager

ROOT = Path(__file__).resolve().parents[2]


def _recorded(points):
    h = HistoryManager()
    for row in points:
        h.record(**row)
    return h


def _simple_flight():
    """Straight up at 100 m/s for 2 s of thrust, then coasting; irregular steps."""
    rows = []
    for t in (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 2.01, 3.0, 5.0, 8.0):
        rows.append(dict(time=t, altitude=100.0 * t, vz=100.0, velocity=100.0,
                         pitch=math.pi / 2, yaw=0.0, mass=10.0 - min(t, 2.0),
                         thrust=500.0 if t < 2.005 else 0.0, drag=20.0,
                         dynamic_pressure=1000.0 * (5.0 - abs(t - 2.0)),
                         phase="Boost" if t < 2.005 else "Coast",
                         cg=1.2, cp=1.5, stability_margin=3.0))
    return _recorded(rows)


def test_a_frame_between_two_recorded_points_is_interpolated():
    r = FlightReplay.from_history(_simple_flight())
    f = r.frame(0.75)                      # halfway between 0.5 s and 1.0 s
    assert f["altitude"] == pytest.approx(75.0)
    assert f["mass"] == pytest.approx(9.25)
    assert f["phase"] == "Boost"


def test_time_outside_the_flight_is_held_at_its_ends():
    r = FlightReplay.from_history(_simple_flight())
    assert r.frame(-5.0)["altitude"] == pytest.approx(0.0)
    assert r.frame(99.0)["altitude"] == pytest.approx(800.0)
    assert (r.start, r.end, r.duration) == (0.0, 8.0, 8.0)


def test_the_nearest_recorded_point_is_found_on_uneven_steps():
    r = FlightReplay.from_history(_simple_flight())
    assert r.times[r.index_at(0.34)] == pytest.approx(0.2)
    assert r.times[r.index_at(0.36)] == pytest.approx(0.5)
    assert r.index_at(1e9) == r.count - 1


def test_a_yaw_that_wraps_is_interpolated_the_short_way_round():
    h = _recorded([dict(time=0.0, yaw=math.radians(179)),
                   dict(time=1.0, yaw=math.radians(-179))])
    yaw = FlightReplay.from_history(h).frame(0.5)["yaw"]
    assert abs(abs(math.degrees(yaw)) - 180.0) < 1e-6      # not 0 degrees


def test_an_undefined_stability_margin_stays_undefined():
    h = _recorded([dict(time=0.0, stability_margin=2.0),
                   dict(time=1.0, stability_margin=float("nan")),
                   dict(time=2.0, stability_margin=float("nan"))])
    r = FlightReplay.from_history(h)
    assert math.isnan(r.frame(1.5)["stability_margin"])
    assert r.frame(0.2)["stability_margin"] == pytest.approx(2.0)


def test_two_records_at_one_time_do_not_break_interpolation():
    h = _recorded([dict(time=0.0, altitude=0.0), dict(time=1.0, altitude=10.0),
                   dict(time=1.0, altitude=10.0), dict(time=2.0, altitude=30.0)])
    r = FlightReplay.from_history(h)
    assert r.count == 3
    assert r.frame(1.5)["altitude"] == pytest.approx(20.0)


def test_a_recorder_with_no_flight_gives_no_replay():
    assert FlightReplay.from_history(HistoryManager()) is None
    assert FlightReplay.from_history(None) is None


def test_events_come_from_the_record_and_the_summary_in_time_order():
    summary = SimpleNamespace(
        rail_exit_time=0.15, apogee_time=7.0,
        canopies=[{"name": "Drogue", "deploy_time": 7.5},
                  {"name": "Main", "deploy_time": None}])
    r = FlightReplay.from_history(_simple_flight(), summary)
    assert [(e.key, e.label) for e in r.events] == [
        ("liftoff", "Liftoff"), ("rail_exit", "Rail Exit"), ("max_q", "Max-Q"),
        ("burnout", "Burnout"), ("apogee", "Apogee"), ("deploy", "Drogue Deploy")]
    times = {e.key: e.time for e in r.events}
    assert times["max_q"] == pytest.approx(2.0)
    assert times["burnout"] == pytest.approx(2.01)     # first point with no thrust
    assert r.event_near(7.02, 0.1).key == "apogee"
    assert r.event_near(4.0, 0.1) is None


def test_forces_point_where_the_physics_says():
    """Thrust along the nose, weight down, drag against the air-relative
    velocity (not the ground-relative one), lift as recorded."""
    pitch, yaw = math.radians(80), math.radians(30)
    row = dict(pitch=pitch, yaw=yaw, vx=10.0, vy=0.0, vz=50.0, mass=8.0,
               thrust=400.0, drag=30.0, wind_x=10.0, wind_y=5.0,
               lift_x=1.0, lift_y=-2.0, lift_z=0.5, aoa=0.05, cg=1.1, cp=1.6)
    r = FlightReplay.from_history(_recorded([dict(time=0.0, **row), dict(time=1.0, **row)]))
    f = r.forces(0.5)
    axis = body_axis(pitch, yaw)
    assert f["thrust"] == pytest.approx(tuple(400.0 * c for c in axis))
    assert f["weight"] == pytest.approx((0.0, 0.0, -8.0 * 9.80665))
    rel = np.array([10.0 - 10.0, 0.0 - 5.0, 50.0])
    assert f["drag"] == pytest.approx(tuple(-30.0 * rel / np.linalg.norm(rel)))
    assert f["lift"] == (1.0, -2.0, 0.5)
    assert f["wind"] == (10.0, 5.0, 0.0)
    assert (f["cg"], f["cp"]) == (1.1, 1.6)


def test_the_recorder_keeps_what_the_force_arrows_need():
    assert {"aoa", "lift_x", "lift_y", "lift_z", "wind_x", "wind_y"} <= set(HISTORY_FIELDS)


def test_the_force_arrows_module_imports_and_names_every_arrow():
    """The visualizer imports its overlays inside a try/except: a mistake in
    this module once turned every overlay off (arrows, envelope, recovery,
    flags) with nothing but a missing picture to show for it."""
    from visualization.mission.flight_effects import ForceVectors as F
    assert [name for name, _ in F._SPECS] == list(F.ARROWS)
    assert set(F.ARROWS) | set(F.MARKERS) == set(F.COLORS)
    import ui.workspaces.mission_visualizer_workspace as mv
    assert mv._OVERLAYS_OK


def test_an_arrow_grows_with_the_force_but_not_in_proportion():
    from visualization.mission.flight_effects import ForceVectors as F
    one, ten = F.arrow_length(50.0, 50.0, 1.0), F.arrow_length(500.0, 50.0, 1.0)
    assert F.arrow_length(0.01, 50.0, 1.0) == 0.0          # negligible: not drawn
    assert one < ten < 2.0 * one


def test_cg_and_cp_markers_sit_at_their_distance_from_the_nose():
    from visualization.mission.flight_effects import ForceVectors as F
    v = F(None, vis_scale=30.0)                             # no plotter: geometry only
    tail, up = (10.0, 0.0, 100.0), (0.0, 0.0, 1.0)
    assert v.station(tail, up, 2.0, 2.0) == pytest.approx(tail)            # the tail
    assert v.station(tail, up, 2.0, 0.0) == pytest.approx((10, 0, 160))    # the nose
    assert v.station(tail, up, 2.0, 1.5) == pytest.approx((10, 0, 115))


# ── A real flight, the clock, and the two tabs (Qt; own process) ──

_SCRIPT = textwrap.dedent("""
    import math, sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.cases.rocket_canonical import canonical_state
    from ui.replay_clock import clock_for
    from ui.widgets.replay_bar import ReplayBar
    from ui.workspaces.results_workspace import ResultsWorkspace

    engine = RocketStateEngine()
    s = canonical_state()
    s.wind_speed, s.wind_direction = 6.0, 90.0       # so there is lift to record
    engine.set_state(s)
    sim = SimulationEngine(engine)
    clock = clock_for(sim)
    assert clock is clock_for(sim), "one clock per engine"
    results = ResultsWorkspace(engine, sim)
    bar = ReplayBar(clock)
    assert not bar.btn_play.isEnabled() and bar.lbl_time.text() == "No flight"

    sim.start()
    sim._timer.stop()
    for _ in range(400000):
        if not sim._running:
            break
        sim._step()
    assert sim.summary.final_phase == "Landed"

    # The clock took the flight when it ended, parked at touchdown.
    r = clock.replay
    assert r is not None and clock.time == r.end
    assert abs(r.end - sim.summary.flight_time) < 0.1, (r.end, sim.summary.flight_time)
    keys = [e.key for e in r.events]
    for k in ("liftoff", "rail_exit", "max_q", "burnout", "apogee", "landing"):
        assert k in keys, (k, keys)
    assert "deploy" in keys, keys
    assert keys == [e.key for e in sorted(r.events, key=lambda e: e.time)]

    # The replay's apogee is the summary's: one flight, one record.
    apogee_t = next(e.time for e in r.events if e.key == "apogee")
    assert abs(r.frame(apogee_t)["altitude"] - sim.summary.apogee) < 0.5

    # The engine recorded real aerodynamics: lift appears off the rail in a
    # crosswind, and the wind was recorded (it grows with height above the
    # 6 m/s set at the ground, so only a floor is checked).
    lift = (r.series("lift_x") ** 2 + r.series("lift_y") ** 2 + r.series("lift_z") ** 2) ** 0.5
    assert lift.max() > 1.0, lift.max()
    wind = (r.series("wind_x") ** 2 + r.series("wind_y") ** 2) ** 0.5
    assert 5.0 < wind.max() < 40.0, wind.max()
    burn = r.forces(next(e.time for e in r.events if e.key == "burnout") - 0.2)
    up = sum(a * b for a, b in zip(burn["thrust"], burn["axis"]))
    assert up > 100.0, "thrust points along the nose during the burn"

    # Playback is in flight seconds per wall second, whatever the step size.
    clock.seek(1.0)
    clock.set_speed(2.0)
    clock.advance(0.5)
    assert abs(clock.time - 2.0) < 1e-9, clock.time
    clock.advance(1e6)
    assert clock.time == r.end and not clock.playing

    # Play from the end starts over; the bar shows the state.
    clock.play()
    assert clock.playing and clock.time == r.start
    assert bar.btn_play.text() == "Pause"
    clock.pause()
    assert bar.btn_play.text() == "Play"

    # The timeline: a click on an event's mark lands exactly on the event.
    bar.resize(900, 40); bar.show(); app.processEvents()
    tl = bar.timeline
    x = tl.x_of(apogee_t)
    assert tl.time_at(x + 3) == apogee_t
    assert abs(tl.time_at(tl.x_of(apogee_t + 9.0)) - (apogee_t + 9.0)) < r.duration / 400

    # Results follows the clock, and its own slider moves the clock.
    results.refresh_plots()
    results.show(); app.processEvents()
    clock.seek(apogee_t)
    snap = results._history.get_snapshot(results.scrub_slider.value())
    assert abs(snap["time"] - apogee_t) < 0.2, (snap["time"], apogee_t)
    assert "Apogee" in results.cursor_readouts["Phase"][0].text() or \\
        abs(snap["altitude"] - sim.summary.apogee) < 5.0
    results.scrub_slider.setValue(10)
    assert abs(clock.time - results._history.get_snapshot(10)["time"]) < 1e-9

    # A new flight clears the old one from every view.
    engine.reset_flight()
    sim.start()
    assert clock.replay is None and not bar.btn_play.isEnabled()
    sim.stop()
    print("OK")
""")


def test_a_flown_flight_replays_in_real_time_and_both_tabs_share_the_instant():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    r = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=str(ROOT))],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr
