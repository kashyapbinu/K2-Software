"""
K2 Aerospace — Flight Phase System
====================================
Enum-based flight phases with automatic transition logic.

Phase sequence:
    PRELAUNCH → IGNITION → BOOST → COAST → APOGEE
    → DROGUE_DESCENT → MAIN_DESCENT → LANDED
"""

from enum import Enum
import logging

logger = logging.getLogger("K2.FlightPhases")


class FlightPhase(Enum):
    """All possible flight phases."""
    PRELAUNCH      = "Pre-Launch"
    IGNITION       = "Ignition"
    BOOST          = "Boost"
    COAST          = "Coast"
    APOGEE         = "Apogee"
    DROGUE_DESCENT = "Drogue Descent"
    MAIN_DESCENT   = "Main Descent"
    LANDED         = "Landed"
    TERMINATED     = "Terminated"
    TIMEOUT        = "Timeout"

    @property
    def is_ascending(self) -> bool:
        return self in (FlightPhase.IGNITION, FlightPhase.BOOST, FlightPhase.COAST)

    @property
    def is_descending(self) -> bool:
        return self in (FlightPhase.DROGUE_DESCENT, FlightPhase.MAIN_DESCENT)

    @property
    def is_active(self) -> bool:
        """True if simulation is in an active flying state."""
        return self in (
            FlightPhase.IGNITION, FlightPhase.BOOST, FlightPhase.COAST,
            FlightPhase.APOGEE, FlightPhase.DROGUE_DESCENT, FlightPhase.MAIN_DESCENT
        )

    @property
    def fc_state(self) -> str:
        """Corresponding flight computer state string."""
        return {
            FlightPhase.PRELAUNCH: "IDLE",
            FlightPhase.IGNITION: "ARMED",
            FlightPhase.BOOST: "BOOST",
            FlightPhase.COAST: "COAST",
            FlightPhase.APOGEE: "APOGEE",
            FlightPhase.DROGUE_DESCENT: "RECOVERY",
            FlightPhase.MAIN_DESCENT: "RECOVERY",
            FlightPhase.LANDED: "LANDED",
            FlightPhase.TERMINATED: "IDLE",
            FlightPhase.TIMEOUT: "IDLE",
        }.get(self, "IDLE")


# Phase transition colors for UI
PHASE_COLORS = {
    FlightPhase.PRELAUNCH:      "#484f58",
    FlightPhase.IGNITION:       "#f0883e",
    FlightPhase.BOOST:          "#f0883e",
    FlightPhase.COAST:          "#d29922",
    FlightPhase.APOGEE:         "#58a6ff",
    FlightPhase.DROGUE_DESCENT: "#7ee787",
    FlightPhase.MAIN_DESCENT:   "#7ee787",
    FlightPhase.LANDED:         "#3fb950",
    FlightPhase.TERMINATED:     "#f85149",
    FlightPhase.TIMEOUT:        "#f85149",
}


class PhaseManager:
    """
    Evaluates phase transitions based on current simulation state.
    Pure logic — no side effects, no UI coupling.
    """

    IGNITION_DELAY = 0.1  # seconds of ignition before boost

    def __init__(self):
        self._apogee_handled = False
        self._prev_velocity = 0.0
        self._apogee_time = None

    def reset(self):
        """Reset for a new simulation run."""
        self._apogee_handled = False
        self._prev_velocity = 0.0
        self._apogee_time = None

    def evaluate(self, phase: FlightPhase, t: float, altitude: float,
                 velocity: float, thrust: float,
                 main_deploy_alt: float = 300.0,
                 drogue_delay: float = 0.0) -> FlightPhase:
        """
        Determine the next flight phase based on current conditions.

        Args:
            phase: Current flight phase.
            t: Simulation time (s).
            altitude: Current altitude (m).
            velocity: Current velocity (m/s, positive = up).
            thrust: Current thrust (N).
            main_deploy_alt: Main parachute deployment altitude (m).
            drogue_delay: Seconds after apogee before drogue deployment
                (vehicle falls ballistic during the delay).

        Returns:
            New FlightPhase (may be same as input if no transition).
        """
        new_phase = phase

        if phase == FlightPhase.PRELAUNCH:
            if t > 0 and thrust > 0:
                new_phase = FlightPhase.IGNITION
                logger.info("Phase: PRELAUNCH → IGNITION")

        elif phase == FlightPhase.IGNITION:
            if t > self.IGNITION_DELAY:
                new_phase = FlightPhase.BOOST
                logger.info("Phase: IGNITION → BOOST")

        elif phase == FlightPhase.BOOST:
            if thrust <= 0:
                new_phase = FlightPhase.COAST
                logger.info(f"Phase: BOOST → COAST (burnout at t={t:.2f}s, alt={altitude:.1f}m)")

        elif phase == FlightPhase.COAST:
            if velocity <= 0 and self._prev_velocity > 0:
                new_phase = FlightPhase.APOGEE
                logger.info(f"Phase: COAST → APOGEE (alt={altitude:.1f}m)")

        elif phase == FlightPhase.APOGEE:
            if self._apogee_time is None:
                self._apogee_time = t
            if not self._apogee_handled and (t - self._apogee_time) >= drogue_delay:
                self._apogee_handled = True
                new_phase = FlightPhase.DROGUE_DESCENT
                logger.info(f"Phase: APOGEE → DROGUE_DESCENT "
                            f"(delay {drogue_delay:.1f}s elapsed)")

        elif phase == FlightPhase.DROGUE_DESCENT:
            if altitude <= main_deploy_alt:
                new_phase = FlightPhase.MAIN_DESCENT
                logger.info(f"Phase: DROGUE → MAIN_DESCENT (alt={altitude:.1f}m)")

        elif phase == FlightPhase.MAIN_DESCENT:
            if altitude <= 0 and t > 1.0:
                new_phase = FlightPhase.LANDED
                logger.info(f"Phase: MAIN_DESCENT → LANDED (t={t:.2f}s)")

        self._prev_velocity = velocity
        return new_phase
