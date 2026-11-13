"""
The Simulation tab must show the launch setup the simulation will fly.

Its inputs only ever wrote to the rocket state; nothing wrote the state back
to them. So:

* a project saved with an 8 m/s wind reopened flying 8 m/s under a box that
  said 0, and the first nudge of that box replaced the saved value;
* RESET replaced the whole state with a blank one (no motor, no wind,
  default parachutes) while every box kept the old numbers.

Runs in a subprocess: widgets need a GUI QApplication, and the headless sim
tests put a bare QCoreApplication in this process.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from validation.cases.rocket_canonical import canonical_state

    engine = RocketStateEngine()
    ws = SimulationWorkspace(engine, SimulationEngine(engine))

    # Opening a project: the tab shows what the file holds.
    s = canonical_state()
    s.wind_speed, s.wind_direction, s.wind_gust_intensity = 8.0, 135.0, 0.15
    s.launch_angle, s.launch_rod_length, s.ground_temperature = 84.0, 3.5, 301.0
    s.drogue_deploy_delay, s.main_deploy_altitude = 2.5, 450.0
    s.drogue_cd_area, s.main_cd_area, s.recovery_inflation_time = 0.25, 0.0, 1.2
    s.integrator_name = "euler"
    engine.set_state(s)

    def near(a, b): return abs(a - b) < 1e-6
    assert near(ws.wind_spin.value(), 8.0), ws.wind_spin.value()
    assert near(ws.wind_dir_spin.value(), 135.0)
    assert near(ws.wind_turb_spin.value(), 15.0)
    assert near(ws.wind_stddev_spin.value(), 1.2)
    assert near(ws.angle_spin.value(), 84.0)
    assert near(ws.rod_spin.value(), 3.5)
    assert near(ws.temp_spin.value(), 301.0)
    assert near(ws.drogue_delay_spin.value(), 2.5)
    assert near(ws.main_alt_spin.value(), 450.0)
    assert near(ws.drogue_cda_spin.value(), 0.25)
    assert near(ws.main_cda_spin.value(), 0.0), "a design with one parachute has no main"
    assert near(ws.inflation_spin.value(), 1.2)
    assert ws.integrator_combo.currentText() == "Euler"
    assert ws.wind_mode_combo.currentIndex() == 0

    # Showing a value must not write it back.
    assert engine.state.wind_speed == 8.0 and engine.state.main_cd_area == 0.0
    assert engine.state.ground_temperature == 301.0

    # Multi-level wind from a file fills the table and selects the page.
    engine.update(wind_mode="multi_level",
                  wind_layers=[[0.0, 3.0, 10.0], [1500.0, 12.0, 80.0]])
    assert ws.wind_mode_combo.currentIndex() == 1
    assert ws.wind_stack.currentIndex() == 1
    assert ws._table_wind_layers() == [(0.0, 3.0, 10.0), (1500.0, 12.0, 80.0)]
    assert engine.state.wind_layers == [[0.0, 3.0, 10.0], [1500.0, 12.0, 80.0]]

    # Editing still works the other way.
    ws.wind_mode_combo.setCurrentIndex(0)
    ws.wind_spin.setValue(4.0)
    assert engine.state.wind_mode == "average" and engine.state.wind_speed == 4.0

    # RESET: the flight goes, the rocket and the setup stay, and they agree.
    engine.update_sim_tick(sim_time=200.0, sim_phase="Landed", max_altitude=2500.0,
                           propellant_mass=0.0)
    ws._on_reset()
    st = engine.state
    assert st.motor_designation == s.motor_designation != "None"
    assert st.motor_avg_thrust > 0 and st.propellant_mass == st.propellant_mass_initial > 0
    assert st.sim_time == 0.0 and st.max_altitude == 0.0
    assert st.wind_speed == 4.0 and near(ws.wind_spin.value(), 4.0)
    assert st.launch_angle == 84.0 and near(ws.angle_spin.value(), 84.0)
    assert st.drogue_cd_area == 0.25 and near(ws.drogue_cda_spin.value(), 0.25)
    assert st.length == 2.0 and st.diameter == 0.102
    print("OK")
""")


def test_simulation_tab_shows_and_keeps_the_launch_setup():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=str(ROOT))],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
