"""
Batch trajectories (Monte Carlo, optimizer, DOE, sensitivity, trade study)
must fly the same rocket as the interactive simulation.

Two defects made them disagree:

1. Step size. The batch integrator stretched its coast step to 5× dt (50 ms
   at the default 10 ms) with no attitude-mode limit. Just after burnout
   ω_n·dt was past RK4's stable range, and the numerical tumble moved the
   landing point by up to 5× (85° launch, no wind: 89 m against a converged
   476 m). The interactive engine already capped ω_n·dt ≤ 0.5.

2. Drag. Every run was rescaled so its M0.3 Cd equalled config.cd, which is
   the RocketState default 0.45 and has nothing to do with the geometry.
   The canonical rocket flew with 0.905× its drag (apogee +5.4% against the
   interactive sim), and every optimizer candidate's subsonic Cd was pinned
   to 0.45: doubling the fins changed the batch apogee by 0.0%.
"""
import dataclasses

import numpy as np
import pytest

from core.batch_simulation import BatchSimConfig, run_batch_simulation
from core.monte_carlo_engine import MonteCarloConfig, MonteCarloEngine
from validation.cases.rocket_canonical import canonical_state


def _cfg(**state_overrides) -> BatchSimConfig:
    s = canonical_state()
    for k, v in state_overrides.items():
        setattr(s, k, v)
    return BatchSimConfig.from_rocket_state(s)


def _fly(cfg):
    return run_batch_simulation(cfg, seed=3)


@pytest.mark.sim
def test_default_step_is_converged():
    coarse = _fly(_cfg(launch_angle=85.0, sim_dt=0.01))
    fine = _fly(_cfg(launch_angle=85.0, sim_dt=0.005))
    assert coarse.landing_distance == pytest.approx(fine.landing_distance, rel=0.03)
    assert coarse.apogee == pytest.approx(fine.apogee, rel=0.005)


@pytest.mark.sim
def test_nominal_run_flies_the_geometry_drag():
    # With an aero model, config.cd only carries a perturbation — its nominal
    # value must not change the flight at all.
    assert _fly(_cfg(cd=0.45)).apogee == pytest.approx(_fly(_cfg(cd=0.90)).apogee,
                                                       rel=1e-9)


@pytest.mark.sim
def test_drag_perturbation_still_reaches_the_flight():
    base = _cfg()
    draggier = dataclasses.replace(base, cd=base.cd_nominal * 1.2)
    assert _fly(draggier).apogee < _fly(base).apogee * 0.97


@pytest.mark.sim
def test_candidate_geometry_drag_reaches_the_flight():
    big_fins = _cfg(fin_root_chord=0.40, fin_tip_chord=0.16,
                    fin_height=0.24, fin_span=0.24)
    assert _fly(big_fins).apogee < _fly(_cfg()).apogee * 0.95


def test_monte_carlo_perturbs_around_the_base_cd():
    # A base built without cd_nominal must not silently drop the perturbation.
    base = BatchSimConfig(cd=0.5)
    cfg, params = MonteCarloEngine._perturb_config(
        base, MonteCarloConfig(), np.random.default_rng(1))
    assert cfg.cd_nominal == 0.5
    assert params["cd"] == cfg.cd


@pytest.mark.sim
def test_batch_matches_the_interactive_engine():
    from validation.sim.headless_runner import run_flight
    engine = run_flight(canonical_state())
    batch = _fly(_cfg())
    assert batch.apogee == pytest.approx(engine.apogee_m, rel=0.015)
