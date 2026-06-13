"""
Headless flight runner — drives the REAL simulation engine without Qt UI.
=========================================================================

The production engine (:class:`core.simulation_engine.SimulationEngine`) is a
``QObject`` whose physics loop is normally pumped by a ``QTimer`` inside the
PyQt event loop. For validation we must exercise *that* engine — not a
re-implementation — so this module:

1. spins up a headless ``QCoreApplication`` (no GUI),
2. builds a real ``RocketStateEngine`` + ``SimulationEngine``,
3. calls ``start()`` (which configures aero, thrust curve, integrator), then
4. pumps ``_step()`` directly in a loop until the flight LANDS or a step cap,
   bypassing the timer (which would need a running event loop to fire).

This replaces the ad-hoc ``scratch/test_sim_headless.py`` helper, which wrongly
re-derived the equations of motion instead of calling the engine.

Returns a :class:`FlightResult` with the summary scalars and the full recorded
history, so the same run feeds both the sim benchmarks and the structures
benchmarks (which need the flight load history).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.rocket_state import RocketState, RocketStateEngine
from core.simulation_engine import SimulationEngine

# A single shared QCoreApplication for the whole process. Creating QObjects /
# QTimers requires one to exist; a GUI QApplication is unnecessary headless.
_qapp = None


def _ensure_qapp():
    global _qapp
    if _qapp is not None:
        return _qapp
    from PyQt6.QtCore import QCoreApplication
    _qapp = QCoreApplication.instance() or QCoreApplication([])
    return _qapp


@dataclass
class FlightResult:
    """Summary of one headless flight plus the full recorded history."""
    apogee_m: float
    max_velocity_ms: float
    max_acceleration_ms2: float
    max_mach: float
    max_q_pa: float
    burnout_velocity_ms: float
    flight_time_s: float
    landed: bool
    history: object               # HistoryManager — full time series

    def series(self, field: str):
        """(times, values) for a recorded field, e.g. 'altitude', 'velocity'."""
        return self.history.get_series(field)


def run_flight(state: RocketState, max_sim_time: float = 1800.0,
               max_steps: int = 2_000_000) -> FlightResult:
    """Run one flight to landing (or caps) and return a :class:`FlightResult`.

    Args:
        state:        Fully populated RocketState (see cases.rocket_canonical).
        max_sim_time: Hard stop in *simulation* seconds (covers slow drogue
                      descents); guards against a non-terminating recovery.
        max_steps:    Hard stop on integrator steps (guards an adaptive-dt stall).
    """
    _ensure_qapp()

    state_engine = RocketStateEngine(state)
    sim = SimulationEngine(state_engine)

    sim.start()
    if not sim._running:
        # start() refuses with no motor selected, etc.
        raise RuntimeError(
            f"Simulation did not start (motor='{state.motor_designation}'). "
            "Check the motor fields on the RocketState."
        )

    steps = 0
    while sim._running and steps < max_steps:
        sim._step()
        steps += 1
        if state_engine.state.sim_time >= max_sim_time:
            sim.stop()
            break

    hist = sim.history
    s = state_engine.state

    # Burnout velocity = speed at the BOOST→COAST transition (motor_burn_time).
    burnout_v = _value_at_time(hist, "velocity", state.motor_burn_time)
    # Max-Q from the recorded dynamic-pressure series (engine tracks it live too).
    _, max_q, _ = hist.find_max("dynamic_pressure")
    flight_time = hist.get_last().get("time", 0.0) if hist.count else 0.0

    return FlightResult(
        apogee_m=s.max_altitude,
        max_velocity_ms=s.max_velocity,
        max_acceleration_ms2=s.max_acceleration,
        max_mach=s.max_mach,
        max_q_pa=max_q,
        burnout_velocity_ms=burnout_v,
        flight_time_s=flight_time,
        landed=(s.sim_phase == "Landed"),
        history=hist,
    )


def _value_at_time(hist, field: str, t_target: float) -> float:
    """Nearest-sample value of `field` at simulation time `t_target`."""
    if not hist.count:
        return 0.0
    times = hist.get_values("time")
    vals = hist.get_values(field)
    best_i, best_dt = 0, float("inf")
    for i, t in enumerate(times):
        d = abs(t - t_target)
        if d < best_dt:
            best_dt, best_i = d, i
    return vals[best_i]
