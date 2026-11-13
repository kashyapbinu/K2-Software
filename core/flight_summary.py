"""
K2 AeroSim — Flight Summary
=============================
One flight's headline numbers, built the same way from either flight engine.

The Results tab, the rules checker and the technical report read a flight
through this object and nothing else, so a number cannot mean one thing on
one screen and something else on another. Before it existed the Results tab
read maxima straight off RocketState, Monte Carlo read BatchSimResult, and the
two disagreed on what "minimum stability" and "max acceleration" were.

Definitions (both engines):

- ``max_acceleration``: peak *sensed* acceleration, |non-gravitational
  force| / mass — what an on-board accelerometer reads and what loads the
  airframe. Not d|v|/dt, which is short by about 1 g on the way up.
- ``max_q``: peak ½ρv² on ground-relative speed.
- ``rail_exit_velocity``: speed as the rocket clears ``launch_rod_length``,
  interpolated on v² between steps.
- ``stability_at_rail_exit`` / ``min_stability``: static margin in calibers,
  ``AeroModel.static_margin`` (Barrowman CP at the current Mach, zero
  incidence, live CG). The minimum is taken over powered and coasting flight
  from rail exit to apogee.
- ``drogue_descent_rate``: descent rate just before the main-deploy event,
  when a drogue canopy was open.
- ``landing_velocity``: descent rate at ground contact.

A value the flight never produced is ``None``, never zero: a rocket that did
not land has no landing velocity, not a landing velocity of 0 m/s.

No Qt imports — batch workers import this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

LANDED = "Landed"


def flight_warnings(*, landed: bool, apogee: float,
                    min_stability: Optional[float],
                    canopy_opened: bool,
                    landing_velocity: Optional[float]) -> List[str]:
    """Warnings for a flight that ran but should not be trusted as flown.

    Shared by both engines so the same flight carries the same warnings
    wherever it is shown.
    """
    out: List[str] = []
    if min_stability is not None and min_stability < 0.0:
        out.append(f"Unstable: stability margin fell to {min_stability:.2f} cal")
    if landed and apogee >= 1.0 and not canopy_opened:
        speed = f"{landing_velocity:.0f} m/s" if landing_velocity is not None else "speed"
        out.append(f"Ballistic impact at {speed} — no parachute was out")
    return out


def recovery_fields(canopies: list, peak_total_force: float) -> dict:
    """The summary's recovery fields from canopy reports
    (recovery.parachute_dynamics.Canopy.report), the same for both engines.

    drogue_deployed / main_deployed: an apogee- / altitude-triggered canopy
    opened. drogue_descent_rate: the first canopy's descent rate when a
    second one opened after it.
    """
    opened = sorted((c for c in canopies if c.get("deploy_time") is not None),
                    key=lambda c: c["deploy_time"])
    return {
        "canopies": [dict(c) for c in canopies],
        "drogue_deployed": any(c["event"] == "apogee" for c in opened),
        "main_deployed": any(c["event"] == "altitude" for c in opened),
        "drogue_descent_rate": (opened[0].get("descent_rate")
                                if len(opened) >= 2 else None),
        "recovery_shock_force": peak_total_force if opened else None,
    }


@dataclass
class FlightSummary:
    """Headline numbers for one flight. See the module docstring."""

    source: str                       # "interactive" | "batch"
    final_phase: str
    apogee: float
    max_velocity: float
    max_mach: float
    max_acceleration: float           # sensed (specific force), m/s²
    max_q: float                      # Pa
    flight_time: float
    apogee_time: Optional[float] = None
    rail_exit_velocity: Optional[float] = None
    rail_exit_time: Optional[float] = None
    stability_at_rail_exit: Optional[float] = None
    min_stability: Optional[float] = None
    drogue_descent_rate: Optional[float] = None
    landing_velocity: Optional[float] = None
    landing_x: Optional[float] = None
    landing_y: Optional[float] = None
    landing_drift: Optional[float] = None
    descent_time: Optional[float] = None
    recovery_shock_force: Optional[float] = None   # N, peak summed canopy drag
    drogue_deployed: bool = False     # an apogee-triggered canopy opened
    main_deployed: bool = False       # an altitude-triggered canopy opened
    # One entry per canopy on the vehicle (Canopy.report): trigger, when and
    # where it opened, how fast the vehicle was going, its peak opening
    # force, and the descent rate under it.
    canopies: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def landed(self) -> bool:
        return self.final_phase == LANDED

    def to_dict(self) -> dict:
        return asdict(self)

    # ── Builders ──────────────────────────────────────────────────

    @classmethod
    def from_batch(cls, result) -> "FlightSummary":
        """From a core.batch_simulation.BatchSimResult."""
        landed = result.final_phase == LANDED
        rail_t = getattr(result, "rail_exit_time", None)
        apogee_t = getattr(result, "apogee_time", None)
        # min_stability_margin reads 0.0 when it was never evaluated; it is
        # evaluated from rail exit on, so no rail exit means no value.
        min_stab = result.min_stability_margin if rail_t is not None else None
        return cls(
            source="batch",
            final_phase=result.final_phase,
            apogee=result.apogee,
            max_velocity=result.max_velocity,
            max_mach=result.max_mach,
            max_acceleration=result.max_acceleration,
            max_q=result.max_dynamic_pressure,
            flight_time=result.flight_time,
            apogee_time=apogee_t,
            rail_exit_velocity=result.rail_exit_velocity if rail_t is not None else None,
            rail_exit_time=rail_t,
            stability_at_rail_exit=getattr(result, "stability_at_rail_exit", None),
            min_stability=min_stab,
            landing_velocity=result.landing_velocity if landed else None,
            landing_x=result.landing_x if landed else None,
            landing_y=result.landing_y if landed else None,
            landing_drift=result.landing_distance if landed else None,
            descent_time=(result.flight_time - apogee_t
                          if landed and apogee_t is not None else None),
            warnings=list(result.warnings),
            **recovery_fields(result.canopies, result.recovery_shock_force),
        )
