"""
K2 Aerospace — Telemetry Bus
==============================
Aggregates sensor data from the flight computer
and provides a unified data interface for UI consumption.
"""

import logging
from avionics.flight_computer.flight_computer import FlightComputer

logger = logging.getLogger("K2.Telemetry")


class TelemetryBus:
    """
    Aggregates and formats telemetry data from the flight computer.
    Provides a clean interface for UI widgets.
    """

    def __init__(self, flight_computer: FlightComputer):
        self.fc = flight_computer
        self._telemetry_log: list[dict] = []

    def get_current(self) -> dict:
        """Get the latest telemetry snapshot."""
        r = self.fc.readings
        return {
            "fc_state": self.fc.state,
            "accel_x": r["accel_x"],
            "accel_y": r["accel_y"],
            "accel_z": r["accel_z"],
            "baro_altitude": r["baro_alt"],
            "baro_pressure": r["baro_pressure"],
            "gyro_x": r["gyro_x"],
            "gyro_y": r["gyro_y"],
            "gyro_z": r["gyro_z"],
            "gps_altitude": r["gps_alt"],
            "gps_velocity": r["gps_vel"],
        }

    def record(self, time: float):
        """Record current telemetry to the log."""
        entry = self.get_current()
        entry["time"] = time
        self._telemetry_log.append(entry)

    def clear(self):
        """Clear telemetry log."""
        self._telemetry_log.clear()

    @property
    def log(self) -> list[dict]:
        return self._telemetry_log
