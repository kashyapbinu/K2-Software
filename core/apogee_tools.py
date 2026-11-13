"""
K2 AeroSim — Target-apogee tools
==================================
For teams scored on how close they come to a declared apogee:

- :func:`solve_ballast` — how much mass, at a chosen position, brings the
  apogee down to the target, and what it does to stability;
- :func:`solve_airbrake` — how much airbrake drag area (out from burnout to
  apogee) does the same without adding mass;
- :func:`apogee_sensitivity` — how far the apogee moves per unit of each
  thing that is uncertain on launch day.

Every answer is flown through run_batch_simulation (the same physics as
Monte Carlo and the optimiser) with one fixed seed, so two calls with the
same inputs agree and the solved design, flown again, lands where the
solver said. Ballast and airbrakes only take apogee away: a rocket that
falls short with neither is reported as unreachable, not "solved".

No Qt imports.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from core.batch_simulation import (BatchSimConfig, BatchSimResult,
                                   run_batch_simulation)

SEED = 1
REL_TOL = 0.001            # stop when |apogee - target| < 0.1 % of target


@dataclass
class ApogeeSolution:
    reachable: bool
    target: float                       # m above the pad
    apogee: float                       # the solved design, flown (m)
    baseline_apogee: float              # before ballast / airbrakes (m)
    value: float = 0.0                  # kg of ballast or m² of airbrake
    stability_at_rail_exit: Optional[float] = None   # cal, solved design
    min_stability: Optional[float] = None            # cal, solved design
    baseline_min_stability: Optional[float] = None
    runs: int = 0
    message: str = ""
    warnings: List[str] = field(default_factory=list)


def _fly(cfg: BatchSimConfig) -> BatchSimResult:
    return run_batch_simulation(cfg, seed=SEED)


def _min_stab(res: BatchSimResult) -> Optional[float]:
    return res.min_stability_margin if res.rail_exit_time is not None else None


def with_ballast(cfg: BatchSimConfig, mass: float, position: float) -> BatchSimConfig:
    """A copy of *cfg* carrying *mass* kg more airframe mass at *position*
    (m from the nose tip). Dry CG and the liftoff CG move with it."""
    out = copy.deepcopy(cfg)
    if mass <= 0.0:
        return out
    m0 = cfg.dry_mass
    out.dry_mass = m0 + mass
    out.dry_cg = (m0 * cfg.dry_cg + mass * position) / out.dry_mass
    total0 = cfg.dry_mass + cfg.motor_dry_mass + cfg.propellant_mass
    out.cg = (total0 * cfg.cg + mass * position) / (total0 + mass)
    return out


def with_airbrake(cfg: BatchSimConfig, area: float) -> BatchSimConfig:
    out = copy.deepcopy(cfg)
    out.airbrake_cd_area = max(0.0, area)
    return out


def _solve(cfg: BatchSimConfig, target: float, build: Callable,
           first_guess: float, upper_limit: float, what: str,
           progress: Optional[Callable[[int], None]] = None) -> ApogeeSolution:
    """Find the smallest x >= 0 with apogee(build(cfg, x)) == target, given
    that apogee falls as x grows. Bracket by doubling, then bisect."""
    runs = 0

    def fly(x):
        nonlocal runs
        runs += 1
        if progress:
            progress(runs)
        return _fly(build(cfg, x))

    base = fly(0.0)
    sol = ApogeeSolution(False, target, base.apogee, base.apogee, 0.0,
                         base.stability_at_rail_exit, _min_stab(base), _min_stab(base))
    if not base.success:
        sol.message = "The design does not fly: " + "; ".join(base.failure_reasons)
        sol.runs = runs
        return sol
    if base.apogee <= target * (1.0 + REL_TOL):
        sol.message = (f"Unreachable with {what}: the rocket reaches only "
                       f"{base.apogee:.0f} m with none, and {what} can only "
                       f"lower the apogee.")
        sol.runs = runs
        return sol

    lo, hi = 0.0, max(first_guess, 1e-6)
    res_hi = fly(hi)
    while res_hi.apogee > target and hi < upper_limit:
        lo, hi = hi, min(hi * 2.0, upper_limit)
        res_hi = fly(hi)
    if res_hi.apogee > target:
        sol.message = (f"Unreachable: even {hi:.3g} of {what} leaves the "
                       f"apogee at {res_hi.apogee:.0f} m.")
        sol.runs = runs
        return sol

    best, best_res = hi, res_hi
    for _ in range(40):
        if abs(best_res.apogee - target) <= REL_TOL * target:
            break
        mid = 0.5 * (lo + hi)
        res = fly(mid)
        if res.apogee > target:
            lo = mid
        else:
            hi = mid
        if abs(res.apogee - target) < abs(best_res.apogee - target):
            best, best_res = mid, res
        if hi - lo < 1e-9:
            break

    sol.reachable = True
    sol.value = best
    sol.apogee = best_res.apogee
    sol.stability_at_rail_exit = best_res.stability_at_rail_exit
    sol.min_stability = _min_stab(best_res)
    sol.warnings = list(best_res.warnings)
    sol.runs = runs
    return sol


def solve_ballast(cfg: BatchSimConfig, target: float, position: float,
                  min_stability: float = 1.0,
                  progress: Optional[Callable[[int], None]] = None) -> ApogeeSolution:
    """Ballast mass (kg) at *position* (m from the nose tip) that brings the
    apogee down to *target* (m above the pad).

    The solution reports the static margin with the ballast in. Ballast
    ahead of the CG raises it, ballast behind lowers it; a solution whose
    minimum margin falls under *min_stability* calibers carries a warning
    rather than being hidden.
    """
    upper = max(10.0, 5.0 * cfg.dry_mass)
    first = max(0.02 * cfg.dry_mass, 0.01)
    sol = _solve(cfg, target, lambda c, m: with_ballast(c, m, position),
                 first, upper, "ballast", progress)
    if sol.reachable and sol.min_stability is not None and sol.min_stability < min_stability:
        sol.warnings.append(
            f"Stability falls to {sol.min_stability:.2f} cal with this ballast "
            f"(limit {min_stability:.2f}). Move it forward.")
    if sol.reachable:
        sol.message = (f"{sol.value:.3f} kg at {position:.3f} m from the nose "
                       f"→ apogee {sol.apogee:.0f} m")
    return sol


def solve_airbrake(cfg: BatchSimConfig, target: float,
                   progress: Optional[Callable[[int], None]] = None) -> ApogeeSolution:
    """Airbrake drag area (Cd × A, m²), deployed from burnout to apogee,
    that brings the apogee down to *target*.

    A fixed deployment: real airbrakes are usually flown closed-loop, which
    needs less area for the same correction (Release B). This answers "how
    big must they be at most".
    """
    ref = math.pi * (cfg.diameter / 2.0) ** 2
    sol = _solve(cfg, target, with_airbrake, max(0.1 * ref, 1e-4),
                 50.0 * max(ref, 1e-4), "airbrake area", progress)
    if sol.reachable:
        sol.message = (f"{sol.value * 1e4:.1f} cm² of airbrake (Cd×A), out from "
                       f"burnout → apogee {sol.apogee:.0f} m")
    return sol


# ── Sensitivity ────────────────────────────────────────────────────────────

@dataclass
class Sensitivity:
    name: str
    unit: str
    step: float            # perturbation used (± this)
    per_unit: float        # d(apogee)/d(input), m per unit
    apogee_minus: float
    apogee_plus: float


def _impulse_scaled(cfg: BatchSimConfig, scale: float) -> BatchSimConfig:
    out = copy.deepcopy(cfg)
    out.motor_avg_thrust *= scale
    out.motor_max_thrust *= scale
    out.motor_total_impulse *= scale
    out.propellant_mass *= scale
    if out.custom_thrust_curve:
        out.custom_thrust_curve = [(t, f * scale) for t, f in out.custom_thrust_curve]
    return out


def _drag_scaled(cfg: BatchSimConfig, scale: float) -> BatchSimConfig:
    out = copy.deepcopy(cfg)
    if out.cd_nominal <= 0.0:
        out.cd_nominal = out.cd
    out.cd = out.cd_nominal * scale
    return out


def _with(cfg: BatchSimConfig, **kw) -> BatchSimConfig:
    out = copy.deepcopy(cfg)
    for k, v in kw.items():
        setattr(out, k, v)
    return out


def _pad_temperature(cfg: BatchSimConfig) -> float:
    from environment.atmosphere_model import isa_temperature
    return cfg.ground_temperature if cfg.ground_temperature > 0 \
        else isa_temperature(cfg.launch_altitude)


def apogee_sensitivity(cfg: BatchSimConfig,
                       progress: Optional[Callable[[int], None]] = None) -> List[Sensitivity]:
    """Central-difference d(apogee)/d(x) for the launch-day unknowns:
    liftoff mass, drag, motor impulse, wind speed, launch angle and pad
    temperature (the last only without an imported sounding)."""
    t_pad = _pad_temperature(cfg)
    cases = [
        ("Dry mass", "kg", 0.02 * cfg.dry_mass,
         lambda d: _with(cfg, dry_mass=cfg.dry_mass + d)),
        ("Drag (Cd)", "%", 5.0, lambda d: _drag_scaled(cfg, 1.0 + d / 100.0)),
        ("Motor impulse", "%", 2.0, lambda d: _impulse_scaled(cfg, 1.0 + d / 100.0)),
        ("Wind speed", "m/s", 1.0,
         lambda d: _with(cfg, wind_speed=max(0.0, cfg.wind_speed + d))),
        ("Launch angle", "deg", 1.0,
         lambda d: _with(cfg, launch_angle=cfg.launch_angle + d)),
    ]
    if not cfg.atmosphere_profile:
        cases.append(("Pad temperature", "K", 5.0,
                      lambda d: _with(cfg, ground_temperature=t_pad + d)))
    out = []
    n = 0
    for name, unit, step, build in cases:
        if step <= 0:
            continue
        lo = _fly(build(-step)).apogee
        hi = _fly(build(+step)).apogee
        n += 2
        if progress:
            progress(n)
        # Wind can't go below zero and a vertical launch can't tilt "more
        # than vertical" without flipping azimuth: use a one-sided slope
        # there, measured from the unperturbed flight.
        if name == "Wind speed" and cfg.wind_speed < step:
            base = _fly(cfg).apogee
            per = (hi - base) / step
        elif name == "Launch angle" and cfg.launch_angle + step > 90.0:
            base = _fly(cfg).apogee
            per = (base - lo) / step
        else:
            per = (hi - lo) / (2.0 * step)
        out.append(Sensitivity(name, unit, step, per, lo, hi))
    return out
