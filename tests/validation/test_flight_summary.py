"""
One flight, one set of headline numbers, whichever engine flew it.

core.flight_summary.FlightSummary is what the Results tab, the rules checker
and the technical report read. These tests pin that the interactive engine
(Simulation tab) and the batch runner (Monte Carlo, optimiser) produce the
same summary for the same rocket. They did not:

* the two engines built different trapezoid thrust curves from the same
  catalogue motor (batch kept the rated peak, the interactive engine scaled
  it), so peak acceleration differed by 4 % and rail-exit speed by 1.3 %;
* "minimum stability" used the Mach-0 CP in one and the angle-of-attack CP
  in the other;
* the interactive engine did not report rail-exit speed at all, and reported
  a flight with no parachute as touching down at 0 m/s.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.flight_summary import FlightSummary, flight_warnings
from core.rocket_state import RocketStateEngine
from core.simulation_engine import SimulationEngine
from core.staging import StageConfig, build_thrust_curve, trapezoid_thrust_curve
from physics.aerodynamics import AeroModel
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import _ensure_qapp, run_flight


def _interactive(**over) -> FlightSummary:
    state = canonical_state()
    for key, value in over.items():
        setattr(state, key, value)
    return run_flight(state).summary


def _batch(**over) -> FlightSummary:
    state = canonical_state()
    for key, value in over.items():
        setattr(state, key, value)
    return FlightSummary.from_batch(
        run_batch_simulation(BatchSimConfig.from_rocket_state(state), seed=1))


_SAME = ("apogee", "max_velocity", "max_mach", "max_acceleration", "max_q",
         "rail_exit_velocity", "stability_at_rail_exit", "min_stability",
         "drogue_descent_rate", "landing_velocity", "flight_time")


@pytest.mark.sim
@pytest.mark.parametrize("over", [
    {},
    {"launch_rod_length": 4.0, "wind_speed": 6.0},
    {"drogue_cd_area": 0.0, "main_cd_area": 0.0},
], ids=["canonical", "rod-and-wind", "no-parachute"])
def test_both_engines_give_the_same_summary(over):
    a, b = _interactive(**over), _batch(**over)
    assert a.source == "interactive" and b.source == "batch"
    assert a.final_phase == b.final_phase == "Landed"
    for name in _SAME:
        x, y = getattr(a, name), getattr(b, name)
        if y is None:
            assert x is None, name
            continue
        # Touchdown speed in a crosswind settles over the last seconds of
        # descent and is sampled at a single step; the interactive engine's
        # gust phases are also unseeded. 3 % there, 1 % everywhere else.
        rel = 0.03 if name == "landing_velocity" else 0.01
        assert x == pytest.approx(y, rel=rel), name
    assert (a.drogue_deployed, a.main_deployed) == (b.drogue_deployed, b.main_deployed)
    assert a.warnings == b.warnings


@pytest.mark.sim
def test_interactive_reports_rail_exit_and_a_real_touchdown():
    short = _interactive(launch_rod_length=1.0)
    long = _interactive(launch_rod_length=6.0)
    assert 10.0 < short.rail_exit_velocity < long.rail_exit_velocity
    assert 0.0 < short.rail_exit_time < long.rail_exit_time

    dart = _interactive(drogue_cd_area=0.0, main_cd_area=0.0)
    assert dart.landing_velocity > 100.0          # was reported as 0 m/s
    assert dart.drogue_descent_rate is None
    assert dart.recovery_shock_force is None
    assert any("Ballistic impact" in w for w in dart.warnings)


@pytest.mark.sim
def test_the_state_carries_the_real_touchdown_rate():
    """Results' landing plot reads RocketState.main_descent_rate."""
    _ensure_qapp()
    state = canonical_state()
    state.drogue_cd_area = state.main_cd_area = 0.0
    eng = RocketStateEngine(state)
    sim = SimulationEngine(eng)
    sim.start()
    while sim._running:
        sim._step()
    assert eng.state.main_descent_rate == pytest.approx(sim.summary.landing_velocity)
    assert eng.state.main_descent_rate > 100.0


@pytest.mark.sim
def test_a_stopped_flight_has_no_landing_numbers():
    _ensure_qapp()
    eng = RocketStateEngine(canonical_state())
    sim = SimulationEngine(eng)
    sim.start()
    while sim._running and eng.state.sim_time < 5.0:
        sim._step()
    sim.stop()
    s = sim.summary
    assert s is not None and not s.landed and s.final_phase == "Terminated"
    assert s.landing_velocity is None and s.landing_drift is None
    assert s.descent_time is None
    assert s.rail_exit_velocity is not None      # it did clear the rod
    assert s.apogee > 0.0


def test_batch_summary_of_a_refused_config_is_empty():
    state = canonical_state()
    cfg = BatchSimConfig.from_rocket_state(state)
    cfg.dry_mass = 0.0
    s = FlightSummary.from_batch(run_batch_simulation(cfg, seed=1))
    assert s.rail_exit_velocity is None and s.min_stability is None
    assert s.landing_velocity is None and not s.landed


# ── Results tab ─────────────────────────────────────────────────────────────

_RESULTS_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.results_workspace import ResultsWorkspace, summary_text
    from validation.cases.rocket_canonical import canonical_state

    state = canonical_state()
    state.launch_rod_length = 3.0
    eng = RocketStateEngine(state)
    sim = SimulationEngine(eng)
    ws = ResultsWorkspace(eng, sim)
    sim.start()
    while sim._running:
        sim._step()
    ws.refresh_plots()
    s = sim.summary
    tiles = {{k: w.text() for k, w in ws.summary_tiles.items()}}
    assert tiles["rail_exit_velocity"] == f"{{s.rail_exit_velocity:.1f}} m/s", tiles
    assert tiles["apogee"] == f"{{s.apogee:.0f}} m", tiles
    assert tiles["landing_velocity"] == f"{{s.landing_velocity:.1f}} m/s", tiles
    assert tiles["min_stability"].endswith(" cal"), tiles
    assert "Landed" in ws.summary_label.text()
    assert summary_text(None, "apogee") == "—"
    ws.reset_workspace()
    sim.history.clear()
    ws.refresh_plots()
    assert all(w.text() == "—" for w in ws.summary_tiles.values())
    print("OK")
""")


def test_results_tab_shows_the_flight_summary():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _RESULTS_SCRIPT.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]


# ── One thrust curve ────────────────────────────────────────────────────────

def _area(curve):
    return sum(0.5 * (f0 + f1) * (t1 - t0)
               for (t0, f0), (t1, f1) in zip(curve, curve[1:]))


def test_trapezoid_keeps_the_rated_peak_and_impulse():
    c = trapezoid_thrust_curve(3.0, 1000.0, 1500.0, 3100.0)
    assert max(f for _, f in c) == pytest.approx(1500.0)
    assert _area(c) == pytest.approx(3100.0)
    # No peak given: 1.4 x average. No impulse given: average x burn time.
    c = trapezoid_thrust_curve(2.0, 500.0)
    assert max(f for _, f in c) == pytest.approx(700.0)
    assert _area(c) == pytest.approx(1000.0)
    # A peak the impulse cannot pay for: a triangle with the same area.
    c = trapezoid_thrust_curve(2.0, 500.0, 5000.0)
    assert len(c) == 3 and _area(c) == pytest.approx(1000.0)
    assert trapezoid_thrust_curve(0.0, 500.0) == []


def test_every_flight_path_builds_the_same_curve():
    state = canonical_state()
    stage = StageConfig(motor_burn_time=state.motor_burn_time,
                        motor_avg_thrust=state.motor_avg_thrust,
                        motor_max_thrust=state.motor_max_thrust,
                        motor_total_impulse=state.motor_total_impulse)
    from core.batch_simulation import _build_thrust_curve

    _ensure_qapp()
    sim = SimulationEngine(RocketStateEngine(state))
    sim._build_thrust_curve()
    ref = build_thrust_curve(stage)
    assert sim._thrust_curve == ref
    assert _build_thrust_curve(BatchSimConfig.from_rocket_state(state)) == ref


# ── One stability definition ────────────────────────────────────────────────

def test_static_margin_at_mach_zero_is_the_subsonic_margin():
    model = AeroModel.from_state(canonical_state())
    assert model.cp_at_mach(0.0) == pytest.approx(model.cp_subsonic())
    cg = 1.1
    assert model.static_margin(0.0, cg) == pytest.approx(
        model.stability_margin(model.cp_subsonic(), cg))
    # It moves with Mach (fin CN-alpha correction) and stays on the body.
    for mach in (0.5, 1.2, 2.5, 4.0):
        cp = model.cp_at_mach(mach)
        assert 0.0 <= cp <= model.body_length
    assert model.cp_at_mach(2.0) != pytest.approx(model.cp_at_mach(0.0))


def test_warnings_are_one_function():
    assert flight_warnings(landed=True, apogee=500.0, min_stability=1.5,
                           canopy_opened=True, landing_velocity=5.0) == []
    w = flight_warnings(landed=True, apogee=500.0, min_stability=-0.4,
                        canopy_opened=False, landing_velocity=80.0)
    assert len(w) == 2 and "Unstable" in w[0] and "80 m/s" in w[1]
    # A flight that never left the pad is not a ballistic impact.
    assert flight_warnings(landed=True, apogee=0.0, min_stability=None,
                           canopy_opened=False, landing_velocity=0.0) == []
