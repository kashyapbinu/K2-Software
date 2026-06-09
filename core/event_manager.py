"""
K2 Aerospace — Simulation Event Manager
=========================================
Publish-subscribe event bus for simulation events.
Decouples simulation logic from UI and subsystem responses.

Usage:
    mgr = EventManager()
    mgr.subscribe(SimEvent.APOGEE, on_apogee_handler)
    mgr.fire(SimEvent.APOGEE, {"altitude": 1500.0, "time": 12.3})
"""

from enum import Enum
from typing import Callable
import logging

logger = logging.getLogger("K2.EventManager")


class SimEvent(Enum):
    """Simulation events that subsystems can subscribe to."""
    SIM_START      = "sim_start"
    SIM_END        = "sim_end"
    MOTOR_IGNITION = "motor_ignition"
    MOTOR_BURNOUT  = "motor_burnout"
    APOGEE         = "apogee"
    DROGUE_DEPLOY  = "drogue_deploy"
    MAIN_DEPLOY    = "main_deploy"
    LANDING        = "landing"
    PHASE_CHANGE   = "phase_change"
    MAX_Q          = "max_q"           # Maximum dynamic pressure


class EventManager:
    """
    Simple publish-subscribe event bus.
    Thread-safe is NOT required — all events fire on the main Qt thread.
    """

    def __init__(self):
        self._subscribers: dict[SimEvent, list[Callable]] = {}
        self._event_log: list[dict] = []

    def subscribe(self, event: SimEvent, callback: Callable):
        """
        Register a callback for a specific event.

        Args:
            event: The SimEvent to listen for.
            callback: Function(data: dict) to call when event fires.
        """
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)

    def unsubscribe(self, event: SimEvent, callback: Callable):
        """Remove a callback from an event."""
        if event in self._subscribers:
            try:
                self._subscribers[event].remove(callback)
            except ValueError:
                pass

    def fire(self, event: SimEvent, data: dict = None):
        """
        Fire an event, calling all registered subscribers.

        Args:
            event: The event to fire.
            data: Context dict passed to all subscribers.
        """
        data = data or {}
        data["event"] = event.value

        self._event_log.append(data.copy())
        logger.info(f"Event: {event.value} — {data}")

        for cb in self._subscribers.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Event handler error ({event.value}): {e}")

    def clear(self):
        """Clear all subscribers and event log."""
        self._subscribers.clear()
        self._event_log.clear()

    def clear_log(self):
        """Clear event log but keep subscribers."""
        self._event_log.clear()

    @property
    def event_log(self) -> list[dict]:
        """Return the list of all fired events."""
        return self._event_log.copy()

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())
