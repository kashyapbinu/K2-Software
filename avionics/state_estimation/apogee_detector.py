"""
K2 Aerospace — Apogee Detector
================================
Multi-condition voting apogee detection for the flight computer.
"""
import logging

logger = logging.getLogger("K2.ApogeeDetector")


class ApogeeDetector:
    """
    Votes on apogee using three independent conditions:
      1. Filtered vertical velocity sign change (negative = descending).
      2. Altitude plateau (altitude change < threshold over window).
      3. Accelerometer load factor drop below 1G (free-fall indication).

    Apogee is confirmed when 2 out of 3 conditions are met.
    """

    def __init__(self, velocity_threshold: float = -0.5,
                 altitude_window: int = 10,
                 altitude_plateau_tol: float = 1.0):
        self.vel_threshold = velocity_threshold
        self.alt_window    = altitude_window
        self.alt_tol       = altitude_plateau_tol

        self._alt_history  = []
        self._apogee_fired = False
        self._votes        = 0

    def update(self, filtered_velocity: float, filtered_altitude: float,
               accel_z: float) -> bool:
        """
        Update detector with new filtered state.

        Args:
            filtered_velocity:  Kalman-filtered vertical velocity (m/s).
            filtered_altitude:  Kalman-filtered altitude (m).
            accel_z:            Vertical acceleration (m/s²) — negative = free-fall.

        Returns:
            True if apogee is detected for the first time this flight.
        """
        if self._apogee_fired:
            return False

        self._alt_history.append(filtered_altitude)
        if len(self._alt_history) > self.alt_window:
            self._alt_history.pop(0)

        votes = 0

        # Condition 1: velocity sign change
        if filtered_velocity < self.vel_threshold:
            votes += 1

        # Condition 2: altitude plateau
        if len(self._alt_history) >= self.alt_window:
            alt_change = max(self._alt_history) - min(self._alt_history)
            if alt_change < self.alt_tol:
                votes += 1

        # Condition 3: load factor < 1G (free-fall / coast)
        if accel_z < -8.0:   # < -8 m/s² (less than 1G upward)
            votes += 1

        if votes >= 2:
            self._apogee_fired = True
            logger.info(
                f"Apogee detected: vel={filtered_velocity:.1f}m/s "
                f"alt={filtered_altitude:.1f}m votes={votes}/3"
            )
            return True

        return False

    def reset(self):
        self._alt_history  = []
        self._apogee_fired = False
        self._votes        = 0
