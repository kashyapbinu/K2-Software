"""
Weather soundings: a measured atmosphere and wind for the flight.

A sounding is read once (University of Wyoming text list or CSV), turned
into an atmosphere profile (altitude ASL, pressure, temperature) and wind
layers above the pad, and flown by both engines.
"""
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.flight_summary import FlightSummary
from environment.atmosphere_model import (Atmosphere, ProfileAtmosphere,
                                          launch_site_atmosphere)
from environment.sounding import (SoundingError, parse_csv, parse_sounding,
                                  parse_uwyo)
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import run_flight

# A Wyoming "Text: List" excerpt in the site's fixed-width layout. Rows with
# no temperature (below ground) are skipped.
UWYO = """\
72364 EPZ Santa Teresa Observations at 12Z 08 Jun 2026

-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
-----------------------------------------------------------------------------
 1000.0     91
  925.0    780
  879.0   1252   24.2    3.2     25   5.53    165      9  309.7  327.2  310.7
  850.0   1535   21.4    1.4     26   4.97    175     12  309.7  325.4  310.6
  700.0   3158    8.6   -9.4     27   2.75    230     18  313.1  322.2  313.6
  500.0   5850  -10.1  -25.1     28   0.88    250     31  320.8  324.0  321.0
  300.0   9660  -38.5  -51.5     24   0.10    255     52  330.2  330.6  330.2

Station information and sounding indices
"""


def _isa_levels(top=20000.0, step=500.0):
    atm = Atmosphere()
    h, out = 0.0, []
    while h <= top:
        out.append([h, atm.pressure(h), atm.temperature(h)])
        h += step
    return out


# ── Parsing ─────────────────────────────────────────────────────────────────

def test_wyoming_text_list():
    snd = parse_uwyo(UWYO)
    assert [l.altitude for l in snd.levels] == [1252, 1535, 3158, 5850, 9660]
    first = snd.levels[0]
    assert first.pressure == pytest.approx(87900.0)
    assert first.temperature == pytest.approx(297.35)
    assert first.wind_speed == pytest.approx(9 * 0.514444)
    assert first.wind_from == 165
    assert parse_sounding(UWYO).levels == snd.levels


def test_csv_with_units_in_the_headers():
    text = ("# comment line\n"
            "Height (ft),Pressure (mb),Temperature (F),Wind Speed (kt),Wind Dir (deg)\n"
            "0,1013.25,59,10,270\n"
            "3280.84,898.75,47.3,20,280\n"
            "6561.68,795.0,35.6,,\n")
    snd = parse_csv(text)
    assert snd.levels[1].altitude == pytest.approx(1000.0, abs=1e-3)
    assert snd.levels[0].temperature == pytest.approx(288.15)
    assert snd.levels[1].wind_speed == pytest.approx(20 * 0.514444)
    assert snd.levels[2].wind_speed is None and snd.levels[2].wind_from is None


@pytest.mark.parametrize("text,word", [
    ("height,pressure,temperature\n0,1013,15\n1000,900,8\n", "unit"),
    ("height_m,pressure_hpa,temperature_c\n0,1013,15\n", "two levels"),
    ("height_m,pressure_pa,temperature_c\n0,1013,15\n1000,900,8\n", "pressure"),
    ("height_m,pressure_hpa,temperature_k\n0,1013,15\n1000,900,8\n", "temperature"),
    ("height_m,pressure_hpa,temperature_c\n0,900,15\n1000,1013,8\n", "fall"),
])
def test_a_sounding_that_cannot_be_right_is_refused(text, word):
    with pytest.raises(SoundingError, match=word):
        parse_csv(text)


def test_wind_layers_are_heights_above_the_pad():
    snd = parse_uwyo(UWYO)
    layers = snd.wind_layers(site_altitude=1401.0)
    # The pad gets the wind interpolated between the 1252 m and 1535 m levels.
    h0, s0, d0 = layers[0]
    assert h0 == 0.0
    f = (1401.0 - 1252.0) / (1535.0 - 1252.0)
    assert s0 == pytest.approx((9 + f * 3) * 0.514444)
    assert d0 == pytest.approx(165 + f * 10)
    assert [round(h) for h, _, _ in layers[1:]] == [1535 - 1401, 3158 - 1401,
                                                    5850 - 1401, 9660 - 1401]


# ── Atmosphere ──────────────────────────────────────────────────────────────

def test_a_profile_equal_to_isa_is_isa():
    prof = ProfileAtmosphere(_isa_levels())
    isa = Atmosphere()
    for h in (0.0, 750.0, 5321.0, 11000.0, 14321.0, 30000.0):
        assert prof.temperature(h) == pytest.approx(isa.temperature(h), rel=1e-3)
        assert prof.pressure(h) == pytest.approx(isa.pressure(h), rel=2e-3)
        assert prof.density(h) == pytest.approx(isa.density(h), rel=3e-3)


def test_above_and_below_the_sounding_the_air_is_continuous():
    prof = ProfileAtmosphere(parse_uwyo(UWYO).atmosphere_profile())
    top = 9660.0
    for h in (top, 1252.0):
        for eps in (1e-3,):
            assert prof.temperature(h + eps) == pytest.approx(prof.temperature(h - eps), abs=1e-3)
            assert prof.pressure(h + eps) == pytest.approx(prof.pressure(h - eps), rel=1e-6)
    assert prof.pressure(15000.0) < prof.pressure(top)


def test_a_broken_profile_is_refused():
    with pytest.raises(ValueError):
        ProfileAtmosphere([[0.0, 101325.0, 288.0]])
    with pytest.raises(ValueError):
        ProfileAtmosphere([[0.0, 101325.0, 288.0], [0.0, 90000.0, 280.0]])
    cfg = BatchSimConfig.from_rocket_state(canonical_state())
    cfg.atmosphere_profile = [[0.0, -1.0, 288.0], [1000.0, 90000.0, 280.0]]
    res = run_batch_simulation(cfg, seed=1)
    assert not res.success and "atmosphere profile" in res.failure_reasons[0]


def test_the_sounding_replaces_the_pad_temperature():
    prof = _isa_levels()
    atm = launch_site_atmosphere(0.0, ground_temperature=320.0, profile=prof)
    assert atm.temperature(0.0) == pytest.approx(288.15, abs=0.01)


# ── Flight ──────────────────────────────────────────────────────────────────

def _state(**over):
    s = canonical_state()
    for k, v in over.items():
        setattr(s, k, v)
    return s


@pytest.mark.sim
def test_an_isa_sounding_flies_the_isa_flight():
    a = run_batch_simulation(BatchSimConfig.from_rocket_state(_state()), seed=1)
    b = run_batch_simulation(BatchSimConfig.from_rocket_state(
        _state(atmosphere_profile=_isa_levels())), seed=1)
    assert b.apogee == pytest.approx(a.apogee, rel=0.002)


@pytest.mark.sim
def test_a_hot_day_flies_higher_than_a_cold_one():
    def day(shift):
        return [[h, p, t + shift] for h, p, t in _isa_levels()]
    hot = run_batch_simulation(BatchSimConfig.from_rocket_state(
        _state(atmosphere_profile=day(+20.0))), seed=1)
    cold = run_batch_simulation(BatchSimConfig.from_rocket_state(
        _state(atmosphere_profile=day(-20.0))), seed=1)
    assert hot.apogee > cold.apogee * 1.01


@pytest.mark.sim
def test_both_engines_fly_the_sounding():
    snd = parse_uwyo(UWYO)
    over = dict(launch_altitude=1401.0, atmosphere_profile=snd.atmosphere_profile(),
                wind_mode="multi_level", wind_layers=snd.wind_layers(1401.0))
    a = run_flight(_state(**over)).summary
    b = FlightSummary.from_batch(run_batch_simulation(
        BatchSimConfig.from_rocket_state(_state(**over)), seed=1))
    assert a.apogee == pytest.approx(b.apogee, rel=0.01)
    assert a.landing_drift == pytest.approx(b.landing_drift, rel=0.03)
    assert b.landing_drift > 100.0     # the sounding's wind carried it


def test_interactive_engine_refuses_a_broken_profile():
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from validation.sim.headless_runner import _ensure_qapp

    _ensure_qapp()
    eng = RocketStateEngine(_state(atmosphere_profile=[[0.0, 1.0, 1.0]]))
    sim = SimulationEngine(eng)
    failures = []
    sim.sim_failed.connect(lambda t, d: failures.append(t))
    sim.start()
    assert not sim._running and failures == ["Atmosphere profile"]


def test_profile_is_saved_with_the_project():
    from core.rocket_state import RocketState
    s = _state(atmosphere_profile=_isa_levels(2000.0), atmosphere_source="x.csv")
    d = RocketState.from_dict(s.to_dict())
    assert d.atmosphere_profile == s.atmosphere_profile and d.atmosphere_source == "x.csv"


# ── Simulation tab ──────────────────────────────────────────────────────────

_UI = textwrap.dedent("""
    import sys, os, tempfile
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from validation.cases.rocket_canonical import canonical_state

    path = os.path.join(tempfile.mkdtemp(), "epz.txt")
    with open(path, "w") as f:
        f.write({uwyo!r})
    s = canonical_state(); s.launch_altitude = 1401.0
    engine = RocketStateEngine(s)
    ws = SimulationWorkspace(engine, SimulationEngine(engine))
    assert not ws.btn_sounding_clear.isEnabled()
    ws._on_import_sounding(path)
    st = engine.state
    assert len(st.atmosphere_profile) == 5 and st.atmosphere_source == "epz.txt"
    assert st.wind_mode == "multi_level" and st.wind_layers[0][0] == 0.0
    assert ws.wind_mode_combo.currentIndex() == 1
    assert ws.wind_table.rowCount() == len(st.wind_layers)
    assert not ws.isa_check.isEnabled() and not ws.temp_spin.isEnabled()
    assert "epz.txt" in ws.sounding_label.text()
    ws._on_clear_sounding()
    assert st.atmosphere_profile == [] and ws.isa_check.isEnabled()
    print("OK")
""")


def test_simulation_tab_imports_a_sounding():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root), uwyo=UWYO)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]


_UI_AIR = textwrap.dedent("""
    import sys, os, tempfile
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from environment.atmosphere_model import launch_site_atmosphere
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from validation.cases.rocket_canonical import canonical_state

    path = os.path.join(tempfile.mkdtemp(), "epz.txt")
    with open(path, "w") as f:
        f.write({uwyo!r})

    for ground_t in (0.0, 300.0):            # standard day, then a measured pad
        s = canonical_state(); s.launch_altitude = 1401.0
        s.ground_temperature = ground_t
        engine = RocketStateEngine(s)
        ws = SimulationWorkspace(engine, SimulationEngine(engine))
        ws._on_import_sounding(path)
        st = engine.state
        pad_t = launch_site_atmosphere(1401.0, 0.0, st.atmosphere_profile).temperature(0.0)
        # The sounding is the air: the box must not claim a standard day,
        # and the pad temperature shown is the sounding's.
        assert not ws.isa_check.isChecked(), "ISA still ticked under a sounding"
        assert abs(ws.temp_spin.value() - pad_t) < 0.06, (ws.temp_spin.value(), pad_t)
        assert not ws.temp_spin.isEnabled() and not ws.isa_check.isEnabled()
        # Showing it wrote nothing back.
        assert st.ground_temperature == ground_t, st.ground_temperature
        ws._on_clear_sounding()
        assert ws.isa_check.isChecked() == (ground_t == 0.0)
        assert ws.temp_spin.isEnabled() == (ground_t > 0.0)
        from environment.atmosphere_model import isa_temperature
        shown = ground_t or isa_temperature(1401.0)
        assert abs(ws.temp_spin.value() - shown) < 0.06, (ws.temp_spin.value(), shown)
        assert st.ground_temperature == ground_t
    print("OK")
""")


def test_isa_box_is_not_ticked_while_a_sounding_sets_the_air():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI_AIR.format(root=str(root), uwyo=UWYO)],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
