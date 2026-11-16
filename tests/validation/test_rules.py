"""
Rules checker (core.rules): a flight, or a Monte Carlo run, against a
competition's rule set.

No built-in rule set may carry a limit without saying which rulebook it came
from, and a rule without a limit is "not evaluated", never a pass.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core import rules as R
from core.flight_summary import FlightSummary


def _flight(**kw):
    base = dict(source="batch", final_phase="Landed", apogee=3000.0, max_velocity=300.0,
                max_mach=0.9, max_acceleration=150.0, max_q=60000.0, flight_time=200.0,
                rail_exit_velocity=25.0, stability_at_rail_exit=1.8, min_stability=1.6,
                drogue_descent_rate=30.0, landing_velocity=6.0, landing_drift=900.0)
    base.update(kw)
    return FlightSummary(**base)


def test_each_comparison():
    f = _flight()
    ge = R.evaluate(R.Rule("rail", "rail_exit_velocity", ">=", 30.0), f)
    assert ge.status == R.FAIL and ge.margin == pytest.approx(-5.0) and ge.value == 25.0
    le = R.evaluate(R.Rule("drift", "landing_drift", "<=", 1000.0), f)
    assert le.status == R.PASS and le.margin == pytest.approx(100.0)
    bt = R.evaluate(R.Rule("drogue", "drogue_descent_rate", "between", 23.0, 38.0), f)
    assert bt.status == R.PASS and bt.margin == pytest.approx(7.0)
    out = R.evaluate(R.Rule("drogue", "drogue_descent_rate", "between", 31.0, 38.0), f)
    assert out.status == R.FAIL and out.margin == pytest.approx(-1.0)


def test_derived_quantities():
    f = _flight(max_acceleration=9.80665 * 12)
    assert R.evaluate(R.Rule("g", "max_acceleration_g", "<=", 15.0), f).value == pytest.approx(12.0)
    ctx = R.Context(target_apogee=3048.0)
    err = R.evaluate(R.Rule("acc", "apogee_error_pct", "<=", 5.0), f, ctx)
    assert err.value == pytest.approx(48.0 / 3048.0 * 100.0) and err.status == R.PASS
    # Without a declared target the apogee error cannot be judged.
    assert R.evaluate(R.Rule("acc", "apogee_error_pct", "<=", 5.0), f).status == R.NOT_EVALUATED
    twr = R.evaluate(R.Rule("twr", "liftoff_thrust_to_weight", ">=", 5.0), f,
                     R.Context(liftoff_thrust_to_weight=6.2))
    assert twr.status == R.PASS


def test_main_descent_rate_reads_the_last_canopy():
    canopies = [{"name": "D", "event": "apogee", "deploy_time": 20.0, "descent_rate": 28.0},
                {"name": "M", "event": "altitude", "deploy_time": 120.0, "descent_rate": 6.5}]
    f = _flight(canopies=canopies, landing_velocity=6.4)
    assert R.evaluate(R.Rule("m", "main_descent_rate", "<=", 7.0), f).value == 6.5


def test_missing_limits_and_values_are_never_a_pass():
    f = _flight(drogue_descent_rate=None)
    no_limit = R.evaluate(R.Rule("rail", "rail_exit_velocity", ">=", None), f)
    assert no_limit.status == R.NOT_EVALUATED and "no limit" in no_limit.note
    assert no_limit.value == 25.0 and no_limit.margin is None   # shown, not judged
    no_value = R.evaluate(R.Rule("drogue", "drogue_descent_rate", "<=", 40.0), f)
    assert no_value.status == R.NOT_EVALUATED
    half = R.evaluate(R.Rule("x", "drogue_descent_rate", "between", 20.0, None), _flight())
    assert half.status == R.NOT_EVALUATED
    bad = R.evaluate(R.Rule("x", "warp_factor", ">=", 1.0), f)
    assert bad.status == R.UNKNOWN


def test_monte_carlo_pass_rate():
    flights = [_flight(rail_exit_velocity=v) for v in (20, 22, 24, 26, 28, 30, 32, 34, 36, 38)]
    r = R.evaluate(R.Rule("rail", "rail_exit_velocity", ">=", 30.0), flights)
    assert r.status == R.FAIL and r.pass_rate == pytest.approx(0.5)
    assert r.evaluated == 10 and r.value == 20.0 and r.margin == pytest.approx(-10.0)
    partial = flights + [_flight(rail_exit_velocity=None)]
    r2 = R.evaluate(R.Rule("rail", "rail_exit_velocity", ">=", 10.0), partial)
    assert r2.status == R.PASS and r2.evaluated == 10 and "1 flight" in r2.note


def test_builtin_rule_sets_cite_their_limits():
    sets = R.builtin_rule_sets()
    assert sets, "no built-in rule sets"
    for rs in sets:
        has_limits = any(r.limit is not None or r.limit_high is not None for r in rs.rules)
        if has_limits:
            assert rs.source.strip(), f"{rs.name}: limits with no source"
        assert not rs.confirmed or rs.source.strip()
        for rule in rs.rules:
            assert rule.problems() == [], (rs.name, rule.name, rule.problems())


def test_user_rule_sets_round_trip():
    rs = R.RuleSet("Ours", source="Rulebook 2027 §10", confirmed=True,
                   rules=[R.Rule("rail", "rail_exit_velocity", ">=", 30.0, citation="10.1")])
    text = R.dump_user_rule_sets([rs, R.RuleSet("builtin", builtin=True)])
    back = R.parse_user_rule_sets(text)
    assert [s.name for s in back] == ["Ours"]
    assert back[0].rules[0].limit == 30.0 and back[0].confirmed
    assert R.parse_user_rule_sets("not json") == []


def test_context_from_state():
    from validation.cases.rocket_canonical import canonical_state
    s = canonical_state()
    ctx = R.context_from_state(s, 3000.0)
    assert ctx.target_apogee == 3000.0
    assert ctx.liftoff_thrust_to_weight == pytest.approx(
        s.motor_avg_thrust / (s.total_mass() * 9.80665))


_UI = textwrap.dedent("""
    import sys, os, tempfile, json
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication, QInputDialog
    from PyQt6.QtCore import QSettings
    app = QApplication([])
    import ui.settings as st
    ini = os.path.join(tempfile.mkdtemp(), "k2.ini")
    st._store = lambda: QSettings(ini, QSettings.Format.IniFormat)
    from core.flight_summary import FlightSummary
    from ui.widgets.rules_panel import RulesPanel, C_LIMIT, C_VALUE, C_RESULT, C_RATE
    p = RulesPanel()
    assert p.set_combo.currentText().startswith("Template")
    assert "not confirmed" in p.source_label.text()
    f = FlightSummary(source="batch", final_phase="Landed", apogee=3000, max_velocity=300,
                      max_mach=0.9, max_acceleration=150, max_q=6e4, flight_time=200,
                      rail_exit_velocity=25.0, stability_at_rail_exit=1.8, min_stability=1.6)
    p.set_flights(f)
    # A rule waiting for its limit says so and still shows the flight's value.
    assert p.table.item(0, C_RESULT).text() == "enter a limit"
    assert p.table.item(0, C_VALUE).text().startswith("25")
    assert "Limit cell" in p.summary_label.text()
    p.table.item(0, C_LIMIT).setText("30")              # rail exit >= 30
    assert p.table.item(0, C_RESULT).text() == "fail"
    p.table.item(0, C_LIMIT).setText("20")
    assert p.table.item(0, C_RESULT).text() == "pass"
    answers = iter([("Our comp 2027", True), ("Rulebook 2027 section 10", True)])
    QInputDialog.getText = staticmethod(lambda *a, **k: next(answers))
    p._save_as()
    assert p.set_combo.currentText() == "Our comp 2027"
    assert "Rulebook 2027" in p.source_label.text()
    # A limit typed into a saved set is still there after a restart.
    p.table.item(1, C_LIMIT).setText("1.5")
    again = RulesPanel()
    again.set_combo.setCurrentIndex(again.set_combo.findText("Our comp 2027"))
    assert again.current_set().rules[1].limit == 1.5
    p.set_flights([f, FlightSummary(**{{**f.__dict__, "rail_exit_velocity": 15.0}})])
    assert "50.0 %" in p.table.item(0, C_RATE).text()
    p._delete()
    assert p.set_combo.findText("Our comp 2027") < 0
    print("OK")
""")


def test_rules_panel():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
