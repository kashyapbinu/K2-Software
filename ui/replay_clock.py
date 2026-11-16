"""
K2 AeroSim — Replay clock
===========================
The instant of the last flight that every view is looking at.

The Mission Visualizer and the Results tab each had their own slider and
their own "current frame", so scrubbing one left the other where it was, and
the visualizer's playback advanced a fixed number of recorded frames per
timer tick (fast through the coast, slow through the boost, where the
integrator takes short steps). One clock per simulation engine now holds the
flight (core.flight_replay.FlightReplay) and the time; it plays in real
time at a chosen speed, and every view follows ``time_changed``.

    clock = clock_for(sim_engine)     # the same object from any workspace
    clock.seek(12.5); clock.play()
"""
from __future__ import annotations

import logging
import weakref
from time import perf_counter

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.flight_replay import FlightReplay

logger = logging.getLogger("K2.ReplayClock")

SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

_clocks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def clock_for(sim_engine) -> "ReplayClock":
    """The replay clock of *sim_engine*, made on first use."""
    clock = _clocks.get(sim_engine)
    if clock is None:
        clock = _clocks[sim_engine] = ReplayClock(sim_engine)
    return clock


class ReplayClock(QObject):
    flight_changed = pyqtSignal()        # a flight was loaded or cleared
    time_changed = pyqtSignal(float)     # the instant being looked at (s)
    playing_changed = pyqtSignal(bool)

    TICK_MS = 33

    def __init__(self, sim_engine=None, parent=None):
        super().__init__(parent)
        self._sim = sim_engine
        self._replay = None
        self._time = 0.0
        self._speed = 1.0
        self._last_wall = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._on_tick)
        if sim_engine is not None:
            sim_engine.sim_started.connect(self.clear)
            # Open Project, Reset, or a start that was refused: the engine
            # dropped the flight without starting another.
            if hasattr(sim_engine, "flight_cleared"):
                sim_engine.flight_cleared.connect(self.clear)
            sim_engine.sim_finished.connect(self.load_from_engine)

    # ── the flight ──
    @property
    def replay(self):
        """The loaded FlightReplay, or None."""
        return self._replay

    def set_flight(self, replay) -> None:
        self.pause()
        self._replay = replay
        self._time = replay.end if replay is not None else 0.0
        self.flight_changed.emit()
        self.time_changed.emit(self._time)

    def clear(self) -> None:
        if self._replay is not None or self._timer.isActive():
            self.set_flight(None)

    def load_from_engine(self) -> None:
        """Take the flight the engine has just recorded."""
        if self._sim is None:
            return
        try:
            replay = FlightReplay.from_history(
                self._sim.history, getattr(self._sim, "summary", None))
        except Exception:
            logger.exception("Could not build the replay from the flight recorder")
            replay = None
        self.set_flight(replay)

    def ensure_loaded(self) -> bool:
        """Load the engine's flight if none is loaded yet (a view opened
        after the flight ended). True when a flight is available."""
        if self._replay is None and self._sim is not None \
                and not getattr(self._sim, "_running", False):
            self.load_from_engine()
        return self._replay is not None

    # ── the instant ──
    @property
    def time(self) -> float:
        return self._time

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def speed(self) -> float:
        return self._speed

    def set_speed(self, speed: float) -> None:
        self._speed = max(0.01, float(speed))

    def seek(self, t: float) -> None:
        if self._replay is None:
            return
        t = self._replay.clamp(t)
        if t != self._time:
            self._time = t
            self.time_changed.emit(t)

    def seek_index(self, index: int) -> None:
        if self._replay is not None and self._replay.count:
            index = max(0, min(int(index), self._replay.count - 1))
            self.seek(float(self._replay.times[index]))

    def play(self) -> None:
        if self._replay is None or self.playing:
            return
        if self._time >= self._replay.end:
            self.seek(self._replay.start)
        self._last_wall = perf_counter()
        self._timer.start()
        self.playing_changed.emit(True)

    def pause(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.playing_changed.emit(False)

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def restart(self) -> None:
        self.pause()
        if self._replay is not None:
            self.seek(self._replay.start)

    def advance(self, wall_seconds: float) -> None:
        """Move the flight on by *wall_seconds* of real time at the current
        speed; stops at the end of the flight."""
        if self._replay is None:
            return
        t = self._time + wall_seconds * self._speed
        if t >= self._replay.end:
            self.seek(self._replay.end)
            self.pause()
        else:
            self.seek(t)

    def _on_tick(self) -> None:
        now = perf_counter()
        # A stalled event loop (a long render) must not make the flight jump.
        self.advance(min(now - self._last_wall, 0.25))
        self._last_wall = now
