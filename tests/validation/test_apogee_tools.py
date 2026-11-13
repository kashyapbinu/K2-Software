"""
Target-apogee tools: ballast, airbrakes, sensitivity (core.apogee_tools).

The check that matters: the solved design, flown again through the batch
runner, lands within 0.5 % of the target, and a rocket that cannot reach the
target is told so rather than given a negative ballast.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import apogee_tools as at
from core.batch_simulation import BatchSimConfig, run_batch_simulation
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import run_flight

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def cfg():
    return BatchSimConfig.from_rocket_state(canonical_state())


@pytest.fixture(scope="module")
def baseline(cfg):
    return run_batch_simulation(cfg, seed=at.SEED).apogee


def test_ballast_hits_the_target(cfg, baseline):
    target = baseline - 200.0
    sol = at.solve_ballast(cfg, target, position=0.2)
    assert sol.reachable and sol.value > 0.0
    again = run_batch_simulation(at.with_ballast(cfg, sol.value, 0.2), seed=at.SEED)
    assert again.apogee == pytest.approx(target, rel=0.005)
    assert sol.apogee == pytest.approx(again.apogee)


def test_ballast_position_moves_stability_not_mass(cfg, baseline):
    target = baseline - 200.0
    nose = at.solve_ballast(cfg, target, position=0.2)
    tail = at.solve_ballast(cfg, target, position=1.9, min_stability=5.0)
    assert nose.value == pytest.approx(tail.value, rel=0.02)
    assert nose.min_stability > nose.baseline_min_stability > tail.min_stability
    assert any("Stability falls" in w for w in tail.warnings)


def test_ballast_moves_the_cg_it_should():
    c = BatchSimConfig(dry_mass=2.0, dry_cg=1.0, cg=1.2, motor_dry_mass=0.5,
                       propellant_mass=0.5)
    b = at.with_ballast(c, 1.0, 0.1)
    assert b.dry_mass == 3.0
    assert b.dry_cg == pytest.approx((2.0 * 1.0 + 0.1) / 3.0)
    assert b.cg == pytest.approx((3.0 * 1.2 + 0.1) / 4.0)


def test_an_unreachable_target_is_reported(cfg, baseline):
    sol = at.solve_ballast(cfg, baseline + 300.0, position=0.2)
    assert not sol.reachable and sol.value == 0.0
    assert "Unreachable" in sol.message and sol.runs == 1


def test_airbrake_hits_the_target_and_both_engines_fly_it(cfg, baseline):
    target = baseline - 150.0
    sol = at.solve_airbrake(cfg, target)
    assert sol.reachable and sol.value > 0.0
    again = run_batch_simulation(at.with_airbrake(cfg, sol.value), seed=at.SEED)
    assert again.apogee == pytest.approx(target, rel=0.005)
    # The Simulation tab flies the same airbrake term.
    state = canonical_state()
    state.airbrake_cd_area = sol.value
    assert run_flight(state).summary.apogee == pytest.approx(again.apogee, rel=0.005)


def test_airbrakes_only_act_between_burnout_and_apogee(cfg):
    plain = run_batch_simulation(cfg, seed=1)
    braked = run_batch_simulation(at.with_airbrake(cfg, 0.01), seed=1)
    assert braked.apogee < plain.apogee
    assert braked.rail_exit_velocity == pytest.approx(plain.rail_exit_velocity)
    assert braked.max_velocity == pytest.approx(plain.max_velocity, rel=1e-3)
    assert braked.landing_velocity == pytest.approx(plain.landing_velocity, rel=0.02)


def test_negative_airbrake_area_is_refused(cfg):
    bad = at.with_airbrake(cfg, 0.0)
    bad.airbrake_cd_area = -1.0
    res = run_batch_simulation(bad, seed=1)
    assert not res.success and "airbrake_cd_area" in res.failure_reasons[0]


def test_sensitivity_signs(cfg):
    by = {s.name: s for s in at.apogee_sensitivity(cfg)}
    assert by["Dry mass"].per_unit < 0
    assert by["Drag (Cd)"].per_unit < 0
    assert by["Motor impulse"].per_unit > 0
    assert by["Pad temperature"].per_unit > 0
    assert by["Launch angle"].per_unit > 0        # more vertical, higher
    assert set(by) == {"Dry mass", "Drag (Cd)", "Motor impulse", "Wind speed",
                       "Launch angle", "Pad temperature"}


def test_airbrake_area_is_saved():
    from core.rocket_state import RocketState
    s = canonical_state()
    s.airbrake_cd_area = 0.002
    assert RocketState.from_dict(s.to_dict()).airbrake_cd_area == 0.002
    assert BatchSimConfig.from_rocket_state(s).airbrake_cd_area == 0.002


_UI = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from ui.dialogs.apogee_tools_dialog import ApogeeToolsDialog, format_result
    from validation.cases.rocket_canonical import canonical_state

    engine = RocketStateEngine(canonical_state())
    dlg = ApogeeToolsDialog(engine)
    dlg.target_spin.setValue(2450)
    dlg.unit_combo.setCurrentText("ft")
    assert abs(dlg.target_m() - 2450) < 1.0          # unit switch keeps the height
    dlg.unit_combo.setCurrentText("m")
    dlg.solve_airbrake()
    dlg._worker.wait(300000)
    app.processEvents()
    r = dlg.last_result
    assert r is not None and r.reachable, dlg.output.toPlainText()
    assert abs(dlg.airbrake_spin.value() - r.value * 1e4) < 0.06
    dlg._apply_airbrake()
    assert abs(engine.state.airbrake_cd_area - r.value) < 1e-5
    assert "cm²" in dlg.output.toPlainText()
    print("OK")
""")


def test_dialog_solves_and_applies_an_airbrake():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=400)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
