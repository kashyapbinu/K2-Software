"""
Regressions from the November 2026 audit of v0.2.0.

Each of these was a number or a setting that looked fine and was not the
thing it claimed to be:

* rail-exit velocity that did not depend on the rail;
* a "safe recovery" probability that counted ballistic impacts as recoveries;
* Design-tab parachutes the simulation never heard about;
* a RESET button that removed the motor and the wind while the Simulation
  tab went on showing them;
* a Simulation tab that never showed what a loaded project held;
* a mass readout that climbed back to liftoff mass at burnout;
* a batch runner that would integrate a rocket with no mass, no diameter or
  no time step, and a thrust curve it flew in whatever order it was typed;
* a project loader that accepted a string where a mass belongs, and a save
  that emptied the file before writing it;
* an updater that ran whatever arrived, complete or not.
"""
import json
import math
import types

import pytest

from core.batch_simulation import (BatchSimConfig, config_problems,
                                   is_divergence_reason, run_batch_simulation)
from core.components import Parachute
from core.rocket_state import FLIGHT_FIELDS, RocketState, RocketStateEngine
from core.staging import assembly_recovery, sanitize_thrust_curve
from validation.cases.rocket_canonical import canonical_assembly, canonical_state
from validation.sim.headless_runner import _ensure_qapp


def _cfg(**over) -> BatchSimConfig:
    cfg = BatchSimConfig.from_rocket_state(canonical_state())
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


# ── Rail exit ───────────────────────────────────────────────────────────────

@pytest.mark.sim
def test_rail_exit_velocity_depends_on_the_rail():
    short = run_batch_simulation(_cfg(launch_rod_length=0.5), seed=1)
    mid = run_batch_simulation(_cfg(launch_rod_length=2.0), seed=1)
    long = run_batch_simulation(_cfg(launch_rod_length=8.0), seed=1)

    assert short.rail_exit_velocity < mid.rail_exit_velocity < long.rail_exit_velocity
    # Near-constant acceleration off the pad: v scales with sqrt(length).
    assert long.rail_exit_velocity / mid.rail_exit_velocity == pytest.approx(2.0, rel=0.15)
    # The rail does not change where a windless vertical flight tops out.
    assert long.apogee == pytest.approx(short.apogee, rel=1e-3)


@pytest.mark.sim
def test_rail_exit_velocity_does_not_depend_on_the_time_step():
    fine = run_batch_simulation(_cfg(sim_dt=0.002), seed=1)
    coarse = run_batch_simulation(_cfg(sim_dt=0.05), seed=1)
    # Read at whichever step landed past the mark this was 24.3 vs 31.6 m/s.
    assert coarse.rail_exit_velocity == pytest.approx(fine.rail_exit_velocity, rel=0.03)


# ── Recovery ────────────────────────────────────────────────────────────────

@pytest.mark.sim
def test_a_flight_with_no_parachute_is_not_reported_as_a_soft_landing():
    lawn_dart = run_batch_simulation(
        _cfg(drogue_cd_area=0.0, main_cd_area=0.0), seed=1)
    nominal = run_batch_simulation(_cfg(), seed=1)

    assert lawn_dart.final_phase == "Landed" and nominal.final_phase == "Landed"
    assert nominal.landing_velocity < 10.0
    assert lawn_dart.landing_velocity > 50.0
    assert not lawn_dart.drogue_deployed and not lawn_dart.main_deployed
    assert any("Ballistic impact" in w for w in lawn_dart.warnings)
    assert nominal.warnings == []


@pytest.mark.sim
def test_landing_velocity_matches_the_canopy():
    cfg = _cfg()
    res = run_batch_simulation(cfg, seed=1)
    mass = cfg.dry_mass + cfg.motor_dry_mass
    area = cfg.drogue_cd_area + cfg.main_cd_area   # both are out at touchdown
    terminal = math.sqrt(2.0 * mass * 9.80665 / (1.225 * area))
    assert res.landing_velocity == pytest.approx(terminal, rel=0.05)


@pytest.mark.sim
def test_a_main_opening_at_speed_ramps_in():
    """No drogue: the main opens at several hundred m/s. With the canopy
    switched on in one step this read 13,500 m/s² (1,376 g)."""
    res = run_batch_simulation(_cfg(drogue_cd_area=0.0), seed=1)
    assert res.max_acceleration < 2000.0


@pytest.mark.sim
def test_unstable_flight_carries_a_warning():
    res = run_batch_simulation(_cfg(cg=1.9, dry_cg=1.9), seed=1)
    assert res.min_stability_margin < 0.0
    assert any("Unstable" in w for w in res.warnings)


def test_design_parachutes_reach_the_state():
    from ui.workspaces.design_workspace import DesignWorkspace

    asm = canonical_assembly()
    assert assembly_recovery(asm) == {}          # no parachute: nothing to say

    tube = asm.stages[0].children[1]
    main = Parachute("Main")
    main.diameter, main.cd = 1.2, 1.5
    asm.add_component(tube, main)

    one = assembly_recovery(asm)
    area_main = 1.5 * math.pi * 0.6 ** 2
    assert one["drogue_cd_area"] == pytest.approx(area_main)   # opens at apogee
    assert one["main_cd_area"] == 0.0

    drogue = Parachute("Drogue")
    drogue.diameter, drogue.cd = 0.4, 1.3
    asm.add_component(tube, drogue)
    two = assembly_recovery(asm)
    assert two["drogue_cd_area"] == pytest.approx(1.3 * math.pi * 0.2 ** 2)
    assert two["main_cd_area"] == pytest.approx(area_main)

    # ...and the real Design-tab sync writes them.
    _ensure_qapp()
    engine = RocketStateEngine()
    engine._assembly = asm
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=asm, engine=engine))
    assert engine.state.main_cd_area == pytest.approx(area_main)
    assert engine.state.drogue_cd_area == pytest.approx(two["drogue_cd_area"])


def test_design_sync_leaves_recovery_alone_without_a_parachute():
    from ui.workspaces.design_workspace import DesignWorkspace

    _ensure_qapp()
    asm = canonical_assembly()
    engine = RocketStateEngine()
    engine.update(drogue_cd_area=0.33, main_cd_area=4.4)
    engine._assembly = asm
    DesignWorkspace._sync_to_engine(types.SimpleNamespace(assembly=asm, engine=engine))
    assert engine.state.drogue_cd_area == 0.33
    assert engine.state.main_cd_area == 4.4


# ── Monte Carlo sampling ────────────────────────────────────────────────────

def test_monte_carlo_launch_angle_scatters_both_ways_about_vertical():
    import numpy as np
    from core.monte_carlo_engine import MonteCarloConfig, MonteCarloEngine

    rng = np.random.default_rng(11)
    base = _cfg(launch_angle=90.0)
    mc = MonteCarloConfig(launch_angle_uncertainty_deg=2.0)
    angles = [MonteCarloEngine._perturb_config(base, mc, rng)[0].launch_angle
              for _ in range(400)]

    assert sum(a > 90.0 for a in angles) > 120
    assert sum(a < 90.0 for a in angles) > 120
    assert sum(a == 90.0 for a in angles) < 5      # was about half of them
    assert max(angles) <= 96.0 + 1e-9              # still clamped to 3 sigma


@pytest.mark.sim
def test_a_rail_leaning_past_vertical_flies_the_mirror_image():
    lean = run_batch_simulation(_cfg(launch_angle=85.0), seed=1)
    other = run_batch_simulation(_cfg(launch_angle=95.0), seed=1)
    assert other.apogee == pytest.approx(lean.apogee, rel=0.02)
    assert other.landing_x * lean.landing_x < 0.0


# ── Batch input guard ───────────────────────────────────────────────────────

@pytest.mark.parametrize("field,value", [
    ("dry_mass", 0.0), ("dry_mass", -1.0), ("diameter", 0.0), ("length", 0.0),
    ("sim_dt", 0.0), ("sim_dt", -0.01), ("dry_mass", float("nan")),
    ("propellant_mass", -0.5),
])
def test_an_unflyable_config_is_refused_not_integrated(field, value):
    res = run_batch_simulation(_cfg(**{field: value}), seed=1)
    assert res.success is False
    assert res.apogee == 0.0
    assert len(res.failure_reasons) == 1
    assert res.failure_reasons[0].startswith("Invalid configuration")
    assert field in res.failure_reasons[0]
    # Not a flight: Monte Carlo must keep it out of the statistics.
    assert is_divergence_reason(res.failure_reasons[0])


def test_a_good_config_has_no_problems():
    assert config_problems(_cfg()) == []


@pytest.mark.parametrize("curve,word", [
    ([(0.0, 0.0), (0.5, float("nan")), (1.0, 0.0)], "not numbers"),
    ([(0.0, 500.0)], "at least two"),
])
def test_a_broken_thrust_curve_is_named(curve, word):
    res = run_batch_simulation(_cfg(custom_thrust_curve=curve), seed=1)
    assert res.success is False
    assert word in res.failure_reasons[0]


# ── Thrust curve order ──────────────────────────────────────────────────────

def test_sanitize_thrust_curve_orders_and_filters():
    raw = [(1.0, 500.0), (0.0, 0.0), ("0.5", 1000), (2.0, 0.0),
           (0.7, float("inf")), None, ("x", 1.0)]
    assert sanitize_thrust_curve(raw) == [
        (0.0, 0.0), (0.5, 1000.0), (1.0, 500.0), (2.0, 0.0)]
    assert sanitize_thrust_curve(None) == []


@pytest.mark.sim
def test_a_thrust_curve_flies_the_same_in_any_order():
    ordered = [(0.0, 0.0), (0.5, 1000.0), (1.0, 500.0), (2.0, 0.0)]
    shuffled = [ordered[2], ordered[0], ordered[1], ordered[3]]
    a = run_batch_simulation(_cfg(custom_thrust_curve=ordered), seed=1)
    b = run_batch_simulation(_cfg(custom_thrust_curve=shuffled), seed=1)
    assert a.apogee > 100.0
    assert b.apogee == a.apogee


def test_the_interactive_engine_orders_the_curve_too():
    from core.simulation_engine import SimulationEngine

    _ensure_qapp()
    state = canonical_state()
    state.custom_thrust_curve = [(1.0, 500.0), (0.0, 0.0), (0.5, 1000.0), (2.0, 0.0)]
    sim = SimulationEngine(RocketStateEngine(state))
    sim._build_thrust_curve()
    assert [t for t, _ in sim._thrust_curve] == [0.0, 0.5, 1.0, 2.0]
    assert sim._get_thrust(0.5) == pytest.approx(1000.0)
    assert sim._get_thrust(1.5) == pytest.approx(250.0)


# ── State: reset, mass, save/load ───────────────────────────────────────────

def _flown_engine() -> RocketStateEngine:
    _ensure_qapp()
    engine = RocketStateEngine(canonical_state())
    engine.update(wind_speed=7.0, wind_direction=120.0, launch_angle=84.0,
                  drogue_cd_area=0.3, main_cd_area=4.0)
    # What a finished flight leaves behind.
    engine.update_sim_tick(sim_time=230.0, sim_phase="Landed", altitude=0.0,
                           max_altitude=2610.0, velocity=5.0, propellant_mass=0.0,
                           landing_drift=900.0)
    return engine


def test_reset_flight_keeps_the_rocket_and_the_launch_setup():
    engine = _flown_engine()
    before = engine.state
    motor = (before.motor_designation, before.motor_avg_thrust,
             before.motor_burn_time, before.propellant_mass_initial)

    engine.reset_flight()
    s = engine.state

    assert (s.motor_designation, s.motor_avg_thrust, s.motor_burn_time,
            s.propellant_mass_initial) == motor
    assert s.motor_designation != "None"
    assert (s.wind_speed, s.wind_direction, s.launch_angle) == (7.0, 120.0, 84.0)
    assert (s.drogue_cd_area, s.main_cd_area) == (0.3, 4.0)
    assert s.length == 2.0 and s.diameter == 0.102
    # ...and the flight is gone, with the motor full again.
    assert s.sim_time == 0.0 and s.sim_phase == "Pre-Launch"
    assert s.max_altitude == 0.0 and s.landing_drift == 0.0
    assert s.propellant_mass == s.propellant_mass_initial > 0.0


def test_mass_readout_does_not_jump_back_at_burnout():
    engine = _flown_engine()
    s = engine.state
    burnout = s.dry_mass + s.motor_dry_mass
    assert s.flight_mass() == pytest.approx(burnout)
    assert s.total_mass() > burnout          # the liftoff fallback, as before

    engine.reset_flight()
    assert engine.state.flight_mass() == pytest.approx(
        burnout + engine.state.propellant_mass_initial)


def test_a_saved_project_does_not_carry_the_last_flight():
    state = _flown_engine().state
    saved = state.to_dict()

    assert not set(saved) & set(FLIGHT_FIELDS)
    assert saved["wind_speed"] == 7.0 and saved["motor_designation"] != "None"
    assert saved["propellant_mass"] == state.propellant_mass_initial

    again = RocketState.from_dict(json.loads(json.dumps(saved)))
    assert again.sim_time == 0.0 and again.max_altitude == 0.0
    assert again.wind_speed == 7.0
    assert again.to_dict() == json.loads(json.dumps(saved))


def test_an_old_project_file_loads_on_the_pad():
    """Files written before flight fields stopped being saved still hold them."""
    old = canonical_state()
    from dataclasses import asdict
    data = asdict(old)
    data.update(sim_time=230.0, sim_phase="Landed", max_altitude=2610.0)
    loaded = RocketState.from_dict(data)
    assert loaded.sim_time == 0.0 and loaded.sim_phase == "Pre-Launch"
    assert loaded.max_altitude == 0.0
    assert loaded.length == old.length


def test_project_load_rejects_values_of_the_wrong_kind():
    loaded = RocketState.from_dict({
        "name": "x", "dry_mass": "abc", "length": None, "diameter": 0.1,
        "fin_count": 3.0, "wind_layers": "nope", "wind_speed": float("nan"),
        "motor_designation": 12, "launch_angle": 80, "not_a_field": 1,
    })
    assert loaded.name == "x"
    assert loaded.dry_mass == 0.0 and isinstance(loaded.dry_mass, float)
    assert loaded.length == 0.0
    assert loaded.diameter == 0.1
    assert loaded.fin_count == 3 and isinstance(loaded.fin_count, int)
    assert loaded.wind_layers == []
    assert loaded.wind_speed == 0.0
    assert loaded.motor_designation == "None"
    assert loaded.launch_angle == 80.0 and isinstance(loaded.launch_angle, float)


def test_a_failed_save_leaves_the_old_file_intact(tmp_path, monkeypatch):
    from core import project_manager as pm

    path = tmp_path / "rocket.k2"
    assert pm.save_project(canonical_state(), str(path), canonical_assembly())
    good = path.read_text(encoding="utf-8")
    assert json.loads(good)["rocket_state"]["length"] == 2.0
    assert list(tmp_path.iterdir()) == [path]          # no temp file left over

    def _dies_part_way(obj, f, **kw):
        f.write('{"format_version": "1.0.0", "rocket_')
        raise OSError("disk full")

    monkeypatch.setattr(pm.json, "dump", _dies_part_way)
    assert pm.save_project(canonical_state(), str(path), canonical_assembly()) is False
    assert path.read_text(encoding="utf-8") == good
    assert list(tmp_path.iterdir()) == [path]


# ── Updater ─────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, payload: bytes, content_length=None):
        self._payload = payload
        self._sent = False
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def read(self, _n):
        if self._sent:
            return b""
        self._sent = True
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, payload, content_length=None):
    from core import updater
    monkeypatch.setattr(updater.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(payload, content_length))
    return updater


def test_updater_refuses_a_truncated_installer(tmp_path, monkeypatch):
    updater = _serve(monkeypatch, b"x" * 100)
    dest = tmp_path / "K2-Setup.exe"
    with pytest.raises(RuntimeError, match="incomplete"):
        updater._download("https://example.invalid/a", str(dest), expected_size=5000)
    assert not dest.exists()


def test_updater_refuses_an_installer_with_the_wrong_checksum(tmp_path, monkeypatch):
    updater = _serve(monkeypatch, b"x" * 100)
    dest = tmp_path / "K2-Setup.exe"
    with pytest.raises(RuntimeError, match="checksum"):
        updater._download("https://example.invalid/a", str(dest),
                          expected_size=100, expected_sha256="0" * 64)
    assert not dest.exists()


def test_updater_keeps_a_complete_matching_installer(tmp_path, monkeypatch):
    import hashlib
    payload = b"installer-bytes" * 10
    updater = _serve(monkeypatch, payload, content_length=len(payload))
    dest = tmp_path / "K2-Setup.exe"
    seen = []
    updater._download("https://example.invalid/a", str(dest),
                      progress_cb=seen.append, expected_size=len(payload),
                      expected_sha256=hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload
    assert seen[-1] == 100


def test_updater_reads_the_release_digest():
    from core.updater import _sha256_of
    assert _sha256_of("sha256:" + "AB" * 32) == "ab" * 32
    assert _sha256_of("sha256:tooshort") == ""
    assert _sha256_of("md5:" + "a" * 64) == ""
    assert _sha256_of(None) == ""
