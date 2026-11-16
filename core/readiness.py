"""
K2 AeroSim — Mission readiness
================================
Which analyses have been run on the rocket as it is drawn *now*.

Every analysis in the app reads the rocket state when it starts and keeps its
result on screen until it is run again. Nothing recorded which rocket a result
belonged to: change a fin after a flutter run and the Dynamics tab still shows
the old flutter speed, the Results tab the old apogee, and the technical
report prints both for a rocket that no longer exists.

This module records, for each analysis, a fingerprint of the inputs it read.
A result is

- ``NOT_RUN``  never produced in this session,
- ``CURRENT``  produced from the inputs the state holds now,
- ``STALE``    produced from inputs that have since changed (the status names
               which group changed), or from an upstream result that is
               itself no longer current.

Inputs are grouped (geometry, mass, structure, motor, recovery, launch,
weather) so a wind change does not mark a CFD run stale.

Results live for the session only, like the flight itself (FLIGHT_FIELDS are
not saved), so the tracker is cleared when a project is opened or reset.

No Qt imports.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NOT_RUN = "not_run"
CURRENT = "current"
STALE = "stale"

# RocketState fields each input group is made of. Only what the user or the
# Design tab sets: no flight field, and no value a flight rewrites as it goes
# (`propellant_mass` burns down, `cg` moves with it, `cp` and
# `stability_margin` follow the Mach number) — those would mark a flight
# stale the moment it finished.
INPUT_GROUPS: Dict[str, Tuple[str, ...]] = {
    "geometry": (
        "length", "diameter", "nose_length", "nose_type", "nose_diameter",
        "fin_height", "fin_root_chord", "fin_tip_chord", "fin_count",
        "fin_span", "fin_sweep_angle", "fin_thickness", "fin_position",
        "fin_cross_section", "fin_body_radius", "surface_finish",
        "extra_fin_sets", "transitions", "airbrake_cd_area",
    ),
    "mass": ("dry_mass", "dry_cg", "motor_position"),
    "structure": (
        "wall_thickness", "material_name", "yield_strength",
        "elastic_modulus", "material_density", "recovery_hardware_rating",
    ),
    "motor": (
        "motor_designation", "motor_avg_thrust", "motor_max_thrust",
        "motor_total_impulse", "motor_burn_time", "motor_dry_mass",
        "motor_length", "propellant_mass_initial", "custom_thrust_curve",
        "stages_config",
    ),
    "recovery": (
        "drogue_deploy_delay", "main_deploy_altitude", "drogue_cd_area",
        "main_cd_area", "recovery_inflation_time", "recovery_canopies",
    ),
    "launch": (
        "launch_altitude", "launch_latitude", "launch_longitude",
        "launch_azimuth", "launch_angle", "launch_rod_length",
    ),
    "weather": (
        "wind_speed", "wind_direction", "wind_gust_intensity", "wind_mode",
        "wind_layers", "ground_temperature", "atmosphere_profile",
    ),
}

GROUP_LABELS = {
    "geometry": "geometry", "mass": "mass", "structure": "material",
    "motor": "motor", "recovery": "recovery", "launch": "launch setup",
    "weather": "weather",
}

_FLIGHT_INPUTS = ("geometry", "mass", "motor", "recovery", "launch", "weather")


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    tab: str                       # main-window tab that produces it
    inputs: Tuple[str, ...] = ()   # input groups it reads
    needs: Tuple[str, ...] = ()    # stages whose result it is built from
    optional: bool = False         # not required for the report to be whole


# In the order a design is taken through them.
STAGES: Tuple[Stage, ...] = (
    Stage("flight", "Flight", "Simulation", _FLIGHT_INPUTS),
    Stage("monte_carlo", "Monte Carlo", "Monte Carlo", _FLIGHT_INPUTS),
    Stage("cfd", "CFD", "CFD", ("geometry",), optional=True),
    # Its load cases are the flight's (peak thrust, max-Q, opening shocks)
    # and the canopies'. It had neither link: a new flight or a bigger
    # parachute left the safety factor "current".
    Stage("structures", "Structures", "Structures",
          ("geometry", "mass", "structure", "motor", "recovery"), needs=("flight",)),
    Stage("flutter", "Flutter", "Dynamics", ("geometry", "structure")),
    Stage("rules", "Rules", "Results", (), needs=("flight",)),
    Stage("report", "Report", "Results", (),
          needs=("flight", "monte_carlo", "structures", "flutter", "rules")),
)
STAGE_BY_KEY: Dict[str, Stage] = {s.key: s for s in STAGES}


def _canonical(value):
    """A JSON-stable form: floats rounded past display precision so a value
    that went through a spin box and back is the same value."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return float(f"{value:.9g}")
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return repr(value)


def fingerprint(state) -> Dict[str, str]:
    """One short hash per input group for *state* (a RocketState)."""
    out = {}
    for group, names in INPUT_GROUPS.items():
        payload = [[n, _canonical(getattr(state, n, None))] for n in names]
        blob = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
        out[group] = hashlib.sha1(blob).hexdigest()[:16]
    return out


def design_problems(state) -> List[str]:
    """What the design still lacks before anything can be analysed."""
    problems = []
    if not (getattr(state, "length", 0) > 0 and getattr(state, "diameter", 0) > 0):
        problems.append("No airframe drawn")
    if not getattr(state, "dry_mass", 0) > 0:
        problems.append("No mass")
    has_motor = (getattr(state, "motor_total_impulse", 0) > 0
                 or bool(getattr(state, "custom_thrust_curve", None))
                 or bool(getattr(state, "stages_config", None)))
    if not has_motor:
        problems.append("No motor")
    return problems


@dataclass
class _Stamp:
    prints: Dict[str, str]
    seq: int
    headline: str = ""
    verdict: Optional[bool] = None   # True pass, False fail, None no verdict
    # Upstream results that existed when this one was made: stage key -> seq.
    built_on: Dict[str, int] = field(default_factory=dict)


@dataclass
class StageStatus:
    stage: Stage
    status: str                      # NOT_RUN | CURRENT | STALE
    headline: str = ""
    verdict: Optional[bool] = None
    changed: List[str] = field(default_factory=list)   # input-group labels
    upstream: List[str] = field(default_factory=list)  # stage titles re-run or stale
    running: bool = False

    @property
    def reason(self) -> str:
        """Why a stale result is stale, as one phrase."""
        parts = []
        if self.changed:
            parts.append(", ".join(self.changed) + " changed")
        if self.upstream:
            parts.append(", ".join(self.upstream) + " re-run, dropped or out of date")
        return "; ".join(parts)


class ReadinessTracker:
    """Stamps each analysis with the inputs it ran on."""

    def __init__(self):
        self._stamps: Dict[str, _Stamp] = {}
        self._pending: Dict[str, Dict[str, str]] = {}
        self._seq = 0

    def clear(self) -> None:
        self._stamps.clear()
        self._pending.clear()

    def forget(self, key: str) -> None:
        """Drop a result that no longer exists (the flight after a reset)."""
        self._require(key)
        self._stamps.pop(key, None)
        self._pending.pop(key, None)

    def begin(self, key: str, state) -> None:
        """An analysis has started: remember the inputs it is reading. A long
        run (CFD, Monte Carlo) is stamped with these, not with whatever the
        state holds when it finishes."""
        self._require(key)
        self._pending[key] = fingerprint(state)

    def abandon(self, key: str) -> None:
        """A started analysis failed or was cancelled; the previous result,
        if any, stands."""
        self._pending.pop(key, None)

    def complete(self, key: str, state, headline: str = "",
                 verdict: Optional[bool] = None) -> None:
        """An analysis has produced a result. Uses the inputs captured by
        :meth:`begin`, or the state as it is now when begin was not called."""
        stage = self._require(key)
        prints = self._pending.pop(key, None) or fingerprint(state)
        self._seq += 1
        built_on = {n: self._stamps[n].seq for n in stage.needs if n in self._stamps}
        self._stamps[key] = _Stamp(prints, self._seq, headline, verdict, built_on)

    def status(self, key: str, state, _now: Dict[str, str] = None) -> StageStatus:
        stage = self._require(key)
        stamp = self._stamps.get(key)
        running = key in self._pending
        if stamp is None:
            return StageStatus(stage, NOT_RUN, running=running)
        now = _now or fingerprint(state)
        changed = [GROUP_LABELS[g] for g in stage.inputs if stamp.prints.get(g) != now[g]]
        upstream = []
        for need in stage.needs:
            up = self._stamps.get(need)
            if up is None:
                # Never run: this result was built without it. Run and since
                # dropped (the flight after Reset): it was built on something
                # that no longer exists. Both used to read as the first.
                if need in stamp.built_on:
                    upstream.append(STAGE_BY_KEY[need].title)
                continue
            if up.seq > stamp.seq or self.status(need, state, now).status == STALE:
                upstream.append(STAGE_BY_KEY[need].title)
        return StageStatus(stage, STALE if (changed or upstream) else CURRENT,
                           stamp.headline, stamp.verdict, changed, upstream, running)

    def all(self, state) -> List[StageStatus]:
        now = fingerprint(state)
        return [self.status(s.key, state, now) for s in STAGES]

    def next_step(self, state) -> str:
        """The one thing to do next, as a sentence."""
        problems = design_problems(state)
        if problems:
            return "Design: " + ", ".join(p.lower() for p in problems) + "."
        statuses = self.all(state)
        by_key = {st.stage.key: st for st in statuses}
        for st in statuses:
            if st.status == STALE:
                # Stale because what it was built on is gone (the flight
                # after Reset): that comes first.
                for need in st.stage.needs:
                    up = by_key[need]
                    if up.status == NOT_RUN and up.stage.title in st.upstream:
                        return f"Run {up.stage.title} ({up.stage.tab} tab)."
                return f"Re-run {st.stage.title} ({st.stage.tab} tab): {st.reason}."
        for st in statuses:
            if st.status == NOT_RUN and not st.stage.optional:
                return f"Run {st.stage.title} ({st.stage.tab} tab)."
        failed = [st.stage.title for st in statuses if st.verdict is False]
        if failed:
            return "Everything is current. Failing: " + ", ".join(failed) + "."
        return "Everything is current for this design."

    @staticmethod
    def _require(key: str) -> Stage:
        try:
            return STAGE_BY_KEY[key]
        except KeyError:
            raise KeyError(f"unknown readiness stage {key!r}; "
                           f"known: {', '.join(STAGE_BY_KEY)}") from None
