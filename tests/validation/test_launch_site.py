"""
Launch site: elevation, position, rail azimuth, pad temperature.

Before this every flight started from a sea-level pad, the rail always
leaned north, and the batch runner (Monte Carlo, optimiser) flew a standard
day whatever pad temperature the Simulation tab held. Spaceport America sits
about 1,400 m up: the thinner air there is worth a few percent of apogee.
"""
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import core.batch_simulation as bs
from core import launch_sites
from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.flight_summary import FlightSummary
from core.rocket_state import RocketState
from environment.atmosphere_model import (Atmosphere, isa_temperature,
                                          launch_site_atmosphere)
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import run_flight


def _cfg(**over) -> BatchSimConfig:
    state = canonical_state()
    for key, value in over.items():
        setattr(state, key, value)
    return BatchSimConfig.from_rocket_state(state)


# ── Atmosphere ──────────────────────────────────────────────────────────────

def test_the_pad_breathes_the_air_at_its_elevation():
    isa = Atmosphere()
    site = launch_site_atmosphere(1400.0)
    assert site.density(0.0) == pytest.approx(isa.density(1400.0))
    assert site.pressure(500.0) == pytest.approx(isa.pressure(1900.0))
    assert site.speed_of_sound(0.0) == pytest.approx(isa.speed_of_sound(1400.0))


def test_pad_temperature_is_exactly_what_was_entered():
    site = launch_site_atmosphere(1400.0, ground_temperature=305.0)
    assert site.temperature(0.0) == pytest.approx(305.0)
    # ...and the lapse rate carries on above it.
    assert site.temperature(1000.0) == pytest.approx(305.0 - 6.5)
    # 0 = standard day at the site.
    assert launch_site_atmosphere(1400.0, 0.0).temperature(0.0) == \
        pytest.approx(isa_temperature(1400.0))


def test_old_projects_at_sea_level_fly_the_same_air():
    """Saved projects hold ground_temperature 288.15; new ones hold 0 (ISA).
    At a sea-level pad they are the same atmosphere."""
    a = launch_site_atmosphere(0.0, 288.15)
    b = launch_site_atmosphere(0.0, 0.0)
    for h in (0.0, 1000.0, 9000.0):
        assert a.density(h) == pytest.approx(b.density(h), rel=1e-12)


# ── Flight ──────────────────────────────────────────────────────────────────

@pytest.mark.sim
def test_a_high_pad_flies_higher_above_ground():
    low = run_batch_simulation(_cfg(launch_altitude=0.0), seed=1)
    high = run_batch_simulation(_cfg(launch_altitude=1400.0), seed=1)
    assert high.apogee > low.apogee * 1.01
    # Same airframe and motor: the rail and the time to clear it are unchanged.
    assert high.rail_exit_velocity == pytest.approx(low.rail_exit_velocity, rel=0.01)


@pytest.mark.sim
def test_batch_flies_the_pad_temperature():
    """The batch runner ignored ground_temperature entirely."""
    cold = run_batch_simulation(_cfg(ground_temperature=268.0), seed=1)
    hot = run_batch_simulation(_cfg(ground_temperature=313.0), seed=1)
    assert hot.apogee > cold.apogee * 1.01


@pytest.mark.sim
def test_both_engines_agree_at_a_high_hot_site():
    over = dict(launch_altitude=1400.0, ground_temperature=305.0,
                launch_angle=85.0, launch_azimuth=60.0, wind_speed=5.0,
                wind_direction=200.0)
    state = canonical_state()
    for k, v in over.items():
        setattr(state, k, v)
    a = run_flight(state).summary
    b = FlightSummary.from_batch(run_batch_simulation(_cfg(**over), seed=1))
    for name in ("apogee", "max_velocity", "max_mach", "rail_exit_velocity",
                 "flight_time"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), rel=0.01), name
    assert a.landing_x == pytest.approx(b.landing_x, rel=0.03, abs=5.0)
    assert a.landing_y == pytest.approx(b.landing_y, rel=0.03, abs=5.0)


@pytest.mark.sim
@pytest.mark.parametrize("turn", [90.0, 180.0, 270.0])
def test_turning_rail_and_wind_together_turns_the_landing_point(turn):
    """Rotate the whole problem about the vertical: apogee stays, the
    landing point rotates with it. X is north, Y is east (wind_model)."""
    saved = bs._GUST_AMP_COEFF
    bs._GUST_AMP_COEFF = 0.0
    try:
        base = run_batch_simulation(_cfg(launch_angle=84.0, launch_azimuth=30.0,
                                         wind_speed=6.0, wind_direction=150.0), seed=1)
        rot = run_batch_simulation(_cfg(launch_angle=84.0, launch_azimuth=30.0 + turn,
                                        wind_speed=6.0, wind_direction=150.0 + turn), seed=1)
    finally:
        bs._GUST_AMP_COEFF = saved
    assert rot.apogee == pytest.approx(base.apogee, rel=0.02)
    r = math.radians(turn)
    # Bearing rotation: north/east components rotate clockwise by `turn`.
    exp_x = base.landing_x * math.cos(r) - base.landing_y * math.sin(r)
    exp_y = base.landing_x * math.sin(r) + base.landing_y * math.cos(r)
    reach = math.hypot(base.landing_x, base.landing_y)
    assert math.hypot(rot.landing_x - exp_x, rot.landing_y - exp_y) < 0.05 * reach


@pytest.mark.sim
def test_the_rail_leans_toward_its_azimuth():
    east = run_batch_simulation(_cfg(launch_angle=70.0, launch_azimuth=90.0,
                                     drogue_cd_area=0.0, main_cd_area=0.0), seed=1)
    # Ballistic, no wind: lands downrange along the rail's bearing (east = +Y).
    assert east.landing_y > 100.0
    assert abs(east.landing_x) < 0.1 * east.landing_y


def test_an_impossible_site_is_refused():
    res = run_batch_simulation(_cfg(launch_altitude=40000.0), seed=1)
    assert not res.success and "launch_altitude" in res.failure_reasons[0]
    res = run_batch_simulation(_cfg(launch_azimuth=float("nan")), seed=1)
    assert not res.success and "launch_azimuth" in res.failure_reasons[0]


def test_site_fields_are_saved_and_carried_to_batch():
    s = canonical_state()
    s.launch_altitude, s.launch_latitude, s.launch_longitude = 1401.0, 32.99, -106.97
    s.launch_azimuth, s.launch_site_name, s.ground_temperature = 45.0, "Pad 2", 300.0
    d = RocketState.from_dict(s.to_dict())
    assert (d.launch_altitude, d.launch_latitude, d.launch_longitude,
            d.launch_azimuth, d.launch_site_name, d.ground_temperature) == \
        (1401.0, 32.99, -106.97, 45.0, "Pad 2", 300.0)
    cfg = BatchSimConfig.from_rocket_state(s)
    assert (cfg.launch_altitude, cfg.launch_azimuth, cfg.ground_temperature) == \
        (1401.0, 45.0, 300.0)


def test_structures_reads_the_air_above_sea_level():
    from structures.workstation import FlightLoads, recovery_loads

    state = canonical_state()
    state.altitude = 500.0
    state.max_velocity = 200.0
    low = FlightLoads.from_state(state)
    state.launch_altitude = 1400.0
    high = FlightLoads.from_state(state)
    assert high.maxq_altitude == pytest.approx(low.maxq_altitude + 1400.0)
    # Thinner air at the main-deploy height: a faster drogue descent there.
    state.launch_altitude = 0.0
    sea = recovery_loads(state)
    state.launch_altitude = 1400.0
    up = recovery_loads(state)
    assert up.main_deploy_velocity > sea.main_deploy_velocity


# ── Monte Carlo ─────────────────────────────────────────────────────────────

def test_absolute_wind_sigma_scatters_a_calm_forecast():
    from core.monte_carlo_engine import MonteCarloConfig, MonteCarloEngine

    rng = np.random.default_rng(5)
    base = _cfg(wind_speed=0.0)
    pct_only = MonteCarloConfig(wind_speed_uncertainty_pct=20.0)
    with_abs = MonteCarloConfig(wind_speed_uncertainty_pct=20.0,
                                wind_speed_uncertainty_ms=2.0)
    calm = [MonteCarloEngine._perturb_config(base, pct_only, rng)[0].wind_speed
            for _ in range(200)]
    scattered = [MonteCarloEngine._perturb_config(base, with_abs, rng)[0].wind_speed
                 for _ in range(200)]
    assert max(calm) < 0.05
    # |N(0, 2 m/s)|: mean 1.6 m/s. Clamping negatives to 0 gave about 0.8.
    assert np.mean(scattered) > 1.3 and max(scattered) > 3.0
    assert sum(w == 0.0 for w in scattered) == 0


def test_absolute_wind_sigma_shifts_every_layer_together():
    from core.monte_carlo_engine import MonteCarloConfig, MonteCarloEngine

    rng = np.random.default_rng(9)
    base = _cfg(wind_mode="multi_level",
                wind_layers=[(0.0, 0.0, 0.0), (2000.0, 4.0, 90.0)])
    mc = MonteCarloConfig(wind_speed_uncertainty_pct=0.0,
                          wind_speed_uncertainty_ms=2.0,
                          wind_direction_uncertainty_deg=0.0)
    for _ in range(50):
        cfg, _ = MonteCarloEngine._perturb_config(base, mc, rng)
        (_, s0, _), (_, s1, _) = cfg.wind_layers
        if s0 > 0.0:
            assert s1 - s0 == pytest.approx(4.0)


# ── Site library ────────────────────────────────────────────────────────────

def test_builtin_sites_load_and_say_where_their_numbers_come_from():
    sites = launch_sites.builtin_sites()
    names = [s.name for s in sites]
    assert "Sea level (ISA)" in names
    for s in sites:
        assert s.builtin and s.problems() == []
        assert s.note.strip(), f"{s.name} has no source note"


def test_user_sites_round_trip_and_replace_by_name():
    text = launch_sites.with_site("", launch_sites.LaunchSite("Pad A", 1200.0, 10.0, 20.0))
    text = launch_sites.with_site(text, launch_sites.LaunchSite("Pad B", 50.0))
    text = launch_sites.with_site(text, launch_sites.LaunchSite("Pad A", 1300.0))
    sites = launch_sites.parse_sites(text)
    assert [(s.name, s.altitude) for s in sites] == [("Pad B", 50.0), ("Pad A", 1300.0)]
    assert [s.name for s in launch_sites.parse_sites(
        launch_sites.without_site(text, "Pad B"))] == ["Pad A"]
    # A user site replaces a built-in of the same name; built-ins are never saved.
    allp = launch_sites.all_sites(launch_sites.with_site(
        text, launch_sites.LaunchSite("Sea level (ISA)", 5.0)))
    assert [s.altitude for s in allp if s.name == "Sea level (ISA)"] == [5.0]


def test_bad_site_entries_are_skipped():
    text = ('[{"name": "ok", "altitude": 10}, {"name": "", "altitude": 1},'
            ' {"name": "far", "latitude": 120}, "junk", {"name": "x", "altitude": "abc"}]')
    assert [s.name for s in launch_sites.parse_sites(text)] == ["ok"]
    assert launch_sites.parse_sites("not json") == []


# ── Simulation tab ──────────────────────────────────────────────────────────

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
    from ui.workspaces.simulation_workspace import SimulationWorkspace, CUSTOM_SITE
    from validation.cases.rocket_canonical import canonical_state

    engine = RocketStateEngine(canonical_state())
    ws = SimulationWorkspace(engine, SimulationEngine(engine))
    s = engine.state
    assert ws.isa_check.isChecked() and not ws.temp_spin.isEnabled()

    # Picking a built-in site sets elevation and position.
    i = ws.site_combo.findText("Spaceport America, NM (approx.)")
    assert i >= 0
    ws.site_combo.setCurrentIndex(i); ws._on_site_chosen(i)
    assert s.launch_altitude == 1401.0 and abs(s.launch_latitude - 32.99028) < 1e-9
    assert ws.site_alt_spin.value() == 1401.0
    assert ws.site_note.text().startswith("Reference point")

    # Editing a value makes it Custom.
    ws.site_alt_spin.setValue(1395.0)
    assert s.launch_altitude == 1395.0 and s.launch_site_name == ""
    assert ws.site_combo.currentText() == CUSTOM_SITE

    # Unticking ISA starts from the standard temperature at this elevation.
    ws.isa_check.setChecked(False)
    assert ws.temp_spin.isEnabled() and abs(s.ground_temperature - 279.1) < 0.1
    ws.temp_spin.setValue(305.0)
    assert s.ground_temperature == 305.0
    ws.isa_check.setChecked(True)
    assert s.ground_temperature == 0.0

    # Save a site, see it listed, select it from a loaded project, delete it.
    from PyQt6.QtWidgets import QInputDialog
    QInputDialog.getText = staticmethod(lambda *a, **k: ("Our pad", True))
    ws._on_site_save()
    assert ws.site_combo.currentText() == "Our pad" and s.launch_site_name == "Our pad"
    assert "Our pad" in (st.get("sim/launch_sites") or "")
    engine.update(launch_site_name="", launch_altitude=10.0)
    assert ws.site_combo.currentText() == CUSTOM_SITE and ws.site_alt_spin.value() == 10.0
    engine.update(launch_site_name="Our pad", launch_altitude=1395.0, launch_azimuth=135.0)
    assert ws.site_combo.currentText() == "Our pad" and ws.azimuth_spin.value() == 135.0
    assert ws.btn_site_delete.isEnabled()
    ws._on_site_delete()
    assert ws.site_combo.findText("Our pad") < 0 and s.launch_site_name == ""

    # A project with a pad temperature shows it.
    engine.update(ground_temperature=290.0)
    assert not ws.isa_check.isChecked() and ws.temp_spin.value() == 290.0
    print("OK")
""")


def test_simulation_tab_launch_site_controls():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _UI.format(root=str(root))],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout, \
        (proc.stdout + proc.stderr)[-3000:]
