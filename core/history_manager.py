"""
K2 Aerospace — Flight History Manager
=======================================
Records and retrieves complete simulation history.
Replaces the simple list[dict] approach in RocketStateEngine.

Supports:
    - High-frequency recording during simulation
    - Per-field time series extraction for plotting
    - Snapshot retrieval at any index
    - CSV export
"""

import csv
import logging
from typing import Optional

logger = logging.getLogger("K2.History")

# All fields recorded per timestep
HISTORY_FIELDS = [
    "time", "x", "y", "altitude", "vx", "vy", "vz", "velocity",
    "pitch", "yaw", "roll", "pitch_rate", "yaw_rate", "roll_rate", "acceleration",
    "mach", "thrust", "drag", "net_force",
    "mass", "cg", "cp", "stability_margin",
    "phase", "cd",
    "atm_temperature", "atm_pressure", "atm_density",
    "dynamic_pressure",
    "axial_stress", "hoop_stress", "von_mises_stress", "shear_stress", "thermal_stress",
    "safety_factor", "margin_of_safety", "wall_temperature",
    "flutter_margin", "propellant_mass",
]


class HistoryManager:
    """Full-fidelity flight data recorder."""

    def __init__(self):
        self._data: dict[str, list] = {f: [] for f in HISTORY_FIELDS}
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def fields(self) -> list[str]:
        return list(HISTORY_FIELDS)

    def clear(self):
        """Clear all recorded data."""
        self._data = {f: [] for f in HISTORY_FIELDS}
        self._count = 0

    def record(self, **kwargs):
        """
        Record one timestep. Pass field=value for each tracked field.
        Missing fields get a default of 0.0 (or "" for 'phase').
        """
        for field in HISTORY_FIELDS:
            default = "" if field == "phase" else 0.0
            self._data[field].append(kwargs.get(field, default))
        self._count += 1

    def get_series(self, field: str) -> tuple[list, list]:
        """
        Return (times, values) for a given field name.
        Useful for direct plotting.
        """
        if field not in self._data:
            logger.warning(f"Unknown history field: {field}")
            return [], []
        return self._data["time"][:], self._data[field][:]

    def get_values(self, field: str) -> list:
        """Return just the values list for a field."""
        return self._data.get(field, [])[:]

    def get_snapshot(self, index: int) -> dict:
        """Return a dict of all fields at the given timestep index."""
        if index < 0 or index >= self._count:
            return {}
        return {f: self._data[f][index] for f in HISTORY_FIELDS}

    def get_last(self) -> dict:
        """Return the most recent data point."""
        if self._count == 0:
            return {}
        return self.get_snapshot(self._count - 1)

    def find_max(self, field: str) -> tuple[float, float, int]:
        """
        Find the maximum value of a field.
        Returns (time_at_max, max_value, index).
        """
        values = self._data.get(field, [])
        if not values:
            return 0.0, 0.0, 0
        try:
            max_val = max(values)
            idx = values.index(max_val)
            return self._data["time"][idx], max_val, idx
        except (ValueError, TypeError):
            return 0.0, 0.0, 0

    def find_min(self, field: str) -> tuple[float, float, int]:
        """Find the minimum value of a field."""
        values = self._data.get(field, [])
        if not values:
            return 0.0, 0.0, 0
        try:
            min_val = min(values)
            idx = values.index(min_val)
            return self._data["time"][idx], min_val, idx
        except (ValueError, TypeError):
            return 0.0, 0.0, 0

    def to_legacy_list(self) -> list[dict]:
        """
        Convert to the old list[dict] format for backward compatibility
        with existing UI code.
        """
        result = []
        for i in range(self._count):
            result.append({f: self._data[f][i] for f in HISTORY_FIELDS})
        return result

    def export_csv(self, filepath: str):
        """Export all recorded data to a CSV file."""
        if self._count == 0:
            logger.warning("No data to export")
            return

        try:
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(HISTORY_FIELDS)
                for i in range(self._count):
                    writer.writerow([self._data[f][i] for f in HISTORY_FIELDS])
            logger.info(f"History exported: {filepath} ({self._count} points)")
        except Exception as e:
            logger.error(f"CSV export failed: {e}")

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count > 0
