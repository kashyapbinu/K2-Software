"""
Release A exit check: a project saved by v0.2.0 opens and flies the same.

v0.2.0 files carry ground_temperature 288.15 (now 0 means ISA), every flight
field, and none of the launch-site, sounding, canopy-list or airbrake fields.
At a sea-level pad with no parachute settings they must fly within 0.5 % of
what v0.2.0 flew: 2610.4 m for the canonical rocket in the Simulation tab
(memory of the October fixes; the trapezoid thrust-curve unification since
moved it by +0.05 %).
"""
import json
from dataclasses import asdict

import pytest

from core.project_manager import load_project
from core.rocket_state import RocketState
from validation.cases.rocket_canonical import canonical_state
from validation.sim.headless_runner import run_flight

V020_APOGEE = 2610.4

NEW_FIELDS = ("launch_altitude", "launch_latitude", "launch_longitude",
              "launch_azimuth", "launch_site_name", "atmosphere_profile",
              "atmosphere_source", "recovery_canopies", "airbrake_cd_area")


def _v020_file(tmp_path):
    d = asdict(canonical_state())
    for k in NEW_FIELDS:
        d.pop(k, None)
    d["ground_temperature"] = 288.15
    d.update(sim_time=233.1, sim_phase="Landed", max_altitude=2610.4, altitude=0.0)
    p = tmp_path / "old.k2"
    p.write_text(json.dumps({"format_version": "1.0.0", "application": "K2 AeroSim",
                             "rocket_state": d, "assembly": None}), encoding="utf-8")
    return p


@pytest.mark.sim
def test_a_v020_project_flies_the_same(tmp_path):
    state, _ = load_project(str(_v020_file(tmp_path)))
    assert isinstance(state, RocketState)
    assert state.launch_altitude == 0.0 and state.recovery_canopies == []
    assert state.sim_time == 0.0 and state.max_altitude == 0.0     # loads on the pad
    apogee = run_flight(state).summary.apogee
    assert apogee == pytest.approx(V020_APOGEE, rel=0.005)
