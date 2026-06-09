"""
K2 Aerospace — Recovery System
================================
Manages drogue and main parachute deployment.
Listens to simulation events and commands deployment.
"""

import logging
from core.event_manager import EventManager, SimEvent

logger = logging.getLogger("K2.Recovery")


class RecoverySystem:
    """
    Dual-deploy recovery system.
    
    Drogue: deploys at apogee (or after configurable delay)
    Main: deploys at configurable altitude
    """

    def __init__(self, event_manager: EventManager):
        self.event_mgr = event_manager
        self.drogue_deployed = False
        self.main_deployed = False

        # Configuration
        self.drogue_delay = 1.0       # seconds after apogee
        self.main_altitude = 300.0    # meters AGL

        # Subscribe to events
        self.event_mgr.subscribe(SimEvent.APOGEE, self._on_apogee)

    def reset(self):
        """Reset for new flight."""
        self.drogue_deployed = False
        self.main_deployed = False

    def configure(self, drogue_delay: float = None, main_altitude: float = None):
        if drogue_delay is not None:
            self.drogue_delay = drogue_delay
        if main_altitude is not None:
            self.main_altitude = main_altitude

    def _on_apogee(self, data: dict):
        """Handle apogee event — schedule drogue deployment."""
        if not self.drogue_deployed:
            self.drogue_deployed = True
            logger.info(f"Recovery: Drogue armed at apogee ({data.get('altitude', 0):.1f}m)")

    def check_main_deploy(self, altitude: float, time: float) -> bool:
        """Check if main chute should deploy."""
        if self.drogue_deployed and not self.main_deployed:
            if altitude <= self.main_altitude:
                self.main_deployed = True
                self.event_mgr.fire(SimEvent.MAIN_DEPLOY, {
                    "time": time, "altitude": altitude,
                })
                logger.info(f"Recovery: Main deployed at {altitude:.1f}m")
                return True
        return False
