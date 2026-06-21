"""
K2 AeroSim — Flight Log Importer
===================================
Parse a real flight log (altimeter / GPS CSV) so it can be overlaid on the
simulated trajectory for sim-vs-real validation.

Handles the common formats from hobby altimeters (Eggtimer, Featherweight,
Blue Raven, PerfectFlite, etc.): a header row naming the columns, then numeric
rows. Column names are matched fuzzily and units auto-detected (feet vs metres).
"""

from __future__ import annotations

import csv
import io
import logging

logger = logging.getLogger("K2.FlightLog")

FT_TO_M = 0.3048

# Fuzzy column-name fragments → canonical field
_COLUMN_HINTS = {
    "time": ("time", "seconds", "sec", "t(s)", "flighttime", "elapsed"),
    "altitude": ("altitude", "alt", "height", "agl", "baroalt", "h(", "apogee"),
    "velocity": ("velocity", "speed", "vel", "vertvel", "vz"),
    "acceleration": ("acceleration", "accel", "accelz", "axial", "az"),
}


def _classify(header: str):
    """Map a header cell to a canonical field name, or None."""
    h = header.strip().lower().replace(" ", "").replace("_", "")
    for field, hints in _COLUMN_HINTS.items():
        if any(hint in h for hint in hints):
            return field
    return None


def _is_feet(header: str) -> bool:
    h = header.lower()
    return ("ft" in h or "feet" in h) and "left" not in h


def parse_flight_log(text: str, source: str = "") -> dict:
    """Parse flight-log CSV text into aligned arrays.

    Args:
        text:   raw CSV file contents.
        source: filename (for reporting).

    Returns dict:
        time, altitude:      list[float] (required; metres, seconds)
        velocity, accel:     list[float] | None (optional channels)
        apogee, apogee_time: float
        source, n_points:    metadata

    Raises ValueError if no usable time+altitude columns are found.
    """
    # Sniff delimiter; fall back to comma.
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delim = dialect.delimiter
    except Exception:
        delim = ","

    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError("Flight log is empty")

    # Detect a header row: first row has at least one non-numeric cell.
    def _is_num(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False

    first = rows[0]
    has_header = any(not _is_num(c) for c in first)

    col_field = {}     # column index -> canonical field
    col_feet = {}      # column index -> bool
    if has_header:
        for i, name in enumerate(first):
            f = _classify(name)
            if f is not None and f not in col_field.values():
                col_field[i] = f
                col_feet[i] = _is_feet(name)
        data_rows = rows[1:]
    else:
        # No header: assume col0=time, col1=altitude (the universal minimum).
        col_field = {0: "time", 1: "altitude"}
        col_feet = {0: False, 1: False}
        data_rows = rows

    if "time" not in col_field.values() or "altitude" not in col_field.values():
        raise ValueError(
            "Could not find time and altitude columns in the flight log")

    out = {"time": [], "altitude": [], "velocity": [], "acceleration": []}
    field_cols = {v: k for k, v in col_field.items()}

    for r in data_rows:
        try:
            t = float(r[field_cols["time"]])
            alt = float(r[field_cols["altitude"]])
        except (ValueError, IndexError):
            continue   # skip non-numeric / short rows (footers, blanks)
        if col_feet.get(field_cols["altitude"]):
            alt *= FT_TO_M
        out["time"].append(t)
        out["altitude"].append(alt)
        for opt in ("velocity", "acceleration"):
            if opt in field_cols:
                try:
                    v = float(r[field_cols[opt]])
                    if col_feet.get(field_cols[opt]):
                        v *= FT_TO_M
                    out[opt].append(v)
                except (ValueError, IndexError):
                    out[opt].append(float("nan"))

    if len(out["time"]) < 2:
        raise ValueError("Flight log has fewer than 2 valid data rows")

    # Drop optional channels that came up empty.
    for opt in ("velocity", "acceleration"):
        if not out[opt]:
            out[opt] = None

    apogee = max(out["altitude"])
    apogee_time = out["time"][out["altitude"].index(apogee)]

    return {
        "time": out["time"],
        "altitude": out["altitude"],
        "velocity": out["velocity"],
        "acceleration": out["acceleration"],
        "apogee": apogee,
        "apogee_time": apogee_time,
        "source": source,
        "n_points": len(out["time"]),
    }


def parse_flight_log_file(path: str) -> dict:
    import os
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    return parse_flight_log(text, source=os.path.basename(path))


def compare_apogee(sim_apogee: float, log: dict) -> dict:
    """Sim-vs-measured apogee error metrics."""
    meas = log.get("apogee", 0.0)
    err = sim_apogee - meas
    pct = (err / meas * 100.0) if meas else 0.0
    return {"sim_apogee": sim_apogee, "measured_apogee": meas,
            "error_m": err, "error_pct": pct}
