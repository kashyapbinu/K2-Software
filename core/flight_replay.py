"""
K2 AeroSim — Flight replay
============================
One recorded flight as something that can be asked "what was happening at
time t": every recorded quantity at that instant, and the list of events with
their times.

The Mission Visualizer and the Results tab each kept their own copy of the
flight and their own idea of the current instant: the visualizer replayed
from buffers it filled during the flight and stepped five frames per timer
tick (so playback speed followed the integrator's step size, not the clock),
and its telemetry panel went on showing touchdown while the rocket on screen
was at max-Q. Both now read frames from this object, built from the flight
recorder (core.history_manager), and share one clock (ui.replay_clock).

World axes are the flight engine's: x, y horizontal, z up; pitch is the body
axis elevation above the horizon and yaw its heading about z, so the nose
points along (cos θ cos ψ, cos θ sin ψ, sin θ).

No Qt imports.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.history_manager import HISTORY_FIELDS

G0 = 9.80665


@dataclass(frozen=True)
class ReplayEvent:
    key: str        # "liftoff", "rail_exit", "max_q", "burnout", "apogee", "deploy", "landing"
    label: str
    time: float


def body_axis(pitch: float, yaw: float):
    """Unit vector along the rocket, tail to nose."""
    c = math.cos(pitch)
    return (c * math.cos(yaw), c * math.sin(yaw), math.sin(pitch))


class FlightReplay:
    """A recorded flight, addressable by time."""

    def __init__(self, series: Dict[str, np.ndarray], phases: List[str],
                 events: List[ReplayEvent]):
        self._s = series
        self._phases = phases
        self.times = series["time"]
        self.events = sorted(events, key=lambda e: e.time)

    # ── construction ──
    @classmethod
    def from_history(cls, history, summary=None) -> Optional["FlightReplay"]:
        """Build from a HistoryManager (and the flight's FlightSummary, which
        supplies the event times the recorder does not hold). None when the
        recorder holds fewer than two points."""
        if history is None or history.count < 2:
            return None
        series = {}
        for name in HISTORY_FIELDS:
            if name == "phase":
                continue
            series[name] = np.asarray(history.get_values(name), dtype=float)
        phases = [str(p) for p in history.get_values("phase")]
        t = series["time"]
        # The recorder appends in time order; a repeated time (two records in
        # one step) would make interpolation divide by zero.
        keep = np.concatenate(([True], np.diff(t) > 0))
        if not keep.all():
            series = {k: v[keep] for k, v in series.items()}
            phases = [p for p, k in zip(phases, keep) if k]
        return cls(series, phases, cls._find_events(series, phases, summary))

    @staticmethod
    def _find_events(s, phases, summary) -> List[ReplayEvent]:
        t = s["time"]
        ev: List[ReplayEvent] = []

        def first_phase(name):
            for i, p in enumerate(phases):
                if p == name:
                    return float(t[i])
            return None

        liftoff = first_phase("Boost")
        if liftoff is not None:
            ev.append(ReplayEvent("liftoff", "Liftoff", liftoff))
        rail = getattr(summary, "rail_exit_time", None)
        if rail is not None:
            ev.append(ReplayEvent("rail_exit", "Rail Exit", float(rail)))
        q = s["dynamic_pressure"]
        if q.size and np.nanmax(q) > 0:
            ev.append(ReplayEvent("max_q", "Max-Q", float(t[int(np.nanargmax(q))])))
        # Burnout: each time thrust falls to zero after having been on (one
        # per stage).
        on = s["thrust"] > 0
        ends = np.flatnonzero(on[:-1] & ~on[1:]) + 1
        for n, i in enumerate(ends, 1):
            label = "Burnout" if len(ends) == 1 else f"Burnout {n}"
            ev.append(ReplayEvent("burnout", label, float(t[i])))
        apogee = getattr(summary, "apogee_time", None)
        if apogee is None and s["altitude"].size:
            i = int(np.nanargmax(s["altitude"]))
            if 0 < i < len(t) - 1:
                apogee = float(t[i])
        if apogee is not None:
            ev.append(ReplayEvent("apogee", "Apogee", float(apogee)))
        for c in (getattr(summary, "canopies", None) or []):
            if c.get("deploy_time") is not None:
                ev.append(ReplayEvent("deploy", f"{c.get('name') or 'Canopy'} Deploy",
                                      float(c["deploy_time"])))
        if phases and phases[-1] == "Landed":
            ev.append(ReplayEvent("landing", "Landing", float(t[-1])))
        t0, t1 = float(t[0]), float(t[-1])
        return [e for e in ev if t0 <= e.time <= t1]

    # ── reading ──
    @property
    def count(self) -> int:
        return len(self.times)

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        return self.end - self.start

    def clamp(self, t: float) -> float:
        return min(self.end, max(self.start, float(t)))

    def index_at(self, t: float) -> int:
        """Index of the recorded point nearest to time *t*."""
        t = self.clamp(t)
        i = int(np.searchsorted(self.times, t))
        if i <= 0:
            return 0
        if i >= self.count:
            return self.count - 1
        return i if (self.times[i] - t) < (t - self.times[i - 1]) else i - 1

    def series(self, name: str) -> np.ndarray:
        return self._s[name]

    def frame(self, t: float) -> dict:
        """Every recorded quantity at time *t*, linearly interpolated between
        the two recorded points around it; ``phase`` is the earlier point's
        (a phase does not interpolate). Angles are interpolated the short way
        round."""
        t = self.clamp(t)
        hi = int(np.searchsorted(self.times, t, side="right"))
        hi = min(max(hi, 1), self.count - 1)
        lo = hi - 1
        span = self.times[hi] - self.times[lo]
        w = float((t - self.times[lo]) / span) if span > 0 else 0.0
        out = {"time": t, "phase": self._phases[lo if w < 1.0 else hi], "index": lo}
        for name, arr in self._s.items():
            if name == "time":
                continue
            a, b = float(arr[lo]), float(arr[hi])
            if name in ("pitch", "yaw", "roll"):
                d = (b - a + math.pi) % (2 * math.pi) - math.pi
                out[name] = a + w * d
            elif math.isnan(a) or math.isnan(b):
                out[name] = a if w < 0.5 else b     # a gap stays a gap
            else:
                out[name] = a + w * (b - a)
        return out

    def forces(self, t: float) -> dict:
        """The forces on the rocket at time *t* as world-frame vectors (N),
        plus the two velocities that explain them (m/s) and where along the
        body CG and CP sit (m from the nose tip).

        thrust   along the body axis
        drag     opposite the air-relative velocity
        lift     the aerodynamic normal force, as recorded
        weight   straight down
        velocity ground-relative; wind is the air's own motion
        """
        f = self.frame(t)
        axis = body_axis(f["pitch"], f["yaw"])
        vel = (f["vx"], f["vy"], f["vz"])
        wind = (f["wind_x"], f["wind_y"], 0.0)
        rel = tuple(v - w for v, w in zip(vel, wind))
        speed = math.sqrt(sum(c * c for c in rel))
        drag = (tuple(-f["drag"] * c / speed for c in rel) if speed > 1e-6
                else (0.0, 0.0, 0.0))
        return {
            "thrust": tuple(f["thrust"] * c for c in axis),
            "drag": drag,
            "lift": (f["lift_x"], f["lift_y"], f["lift_z"]),
            "weight": (0.0, 0.0, -f["mass"] * G0),
            "velocity": vel,
            "wind": wind,
            "axis": axis,
            "cg": f["cg"],
            "cp": f["cp"],
            "aoa": f["aoa"],
            "frame": f,
        }

    def event_near(self, t: float, tolerance: float) -> Optional[ReplayEvent]:
        """The event closest to *t*, if one lies within *tolerance* seconds."""
        best = min(self.events, key=lambda e: abs(e.time - t), default=None)
        return best if best is not None and abs(best.time - t) <= tolerance else None
