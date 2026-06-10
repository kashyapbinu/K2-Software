"""
Tier-B sim reference: OpenRocket comparison.
============================================

Compares the K2 head-less flight of the canonical rocket against an OpenRocket
simulation of the *same* rocket.

Why a CSV, not a live OpenRocket run
------------------------------------
Two things block automating OpenRocket here, and a user-exported CSV sidesteps
both:

  1. OpenRocket is a Java app whose head-less CLI flags differ across releases
     (and Java is not always installed — it was absent on the dev machine).
  2. K2 has an ``.ork`` *importer* but no exporter, so there is no canonical
     ``.ork`` to feed OpenRocket automatically.

So the workflow is: build the canonical rocket in OpenRocket (see
``validation/cases/rocket_canonical.py`` for the exact geometry, mass and the
**L1090W** motor), run its simulation, *Export flight data as CSV*, then point
``OPENROCKET_CSV`` at the file. This benchmark parses that CSV as the reference.

Until ``OPENROCKET_CSV`` is set to an existing file, the benchmark skips.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from core.validation import ValidationLevel
from validation.harness import Benchmark, Comparison, curve_rmse


# ── OpenRocket CSV parsing ────────────────────────────────────────────────────

# OpenRocket export headers carry units, e.g. "Altitude (m)". Match by keyword.
_COL_PATTERNS = {
    "time":     re.compile(r"\btime\b", re.I),
    "altitude": re.compile(r"\baltitude\b", re.I),
    "velocity": re.compile(r"total velocity|vertical velocity", re.I),
}


def parse_openrocket_csv(path: Path) -> dict:
    """Parse an OpenRocket flight-data CSV → {time:[...], altitude:[...], velocity:[...]}.

    OpenRocket CSVs prefix the header row with '# '. Columns are matched by
    keyword so minor label/locale differences across versions still resolve.
    """
    rows = []
    header = None
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if header is None:
                if line.startswith("#"):
                    line = line.lstrip("#").strip()
                # The header is the first line that names a Time column.
                if _COL_PATTERNS["time"].search(line):
                    header = next(csv.reader([line]))
                continue
            if line.startswith("#"):
                continue
            rows.append(next(csv.reader([line])))

    if header is None:
        raise ValueError(f"No header with a Time column found in {path}")

    def find(key):
        for i, h in enumerate(header):
            if _COL_PATTERNS[key].search(h):
                return i
        return None

    idx = {k: find(k) for k in _COL_PATTERNS}
    if idx["time"] is None or idx["altitude"] is None:
        raise ValueError(f"CSV missing Time/Altitude columns: {header}")

    out = {k: [] for k in _COL_PATTERNS}
    for r in rows:
        try:
            t = float(r[idx["time"]])
            alt = float(r[idx["altitude"]])
        except (ValueError, IndexError):
            continue
        out["time"].append(t)
        out["altitude"].append(alt)
        if idx["velocity"] is not None:
            try:
                out["velocity"].append(float(r[idx["velocity"]]))
            except (ValueError, IndexError):
                out["velocity"].append(float("nan"))
    return out


# ── benchmark ─────────────────────────────────────────────────────────────────

def run_openrocket_benchmark() -> Benchmark:
    name = "Sim vs OpenRocket"
    ref = "OpenRocket flight CSV (canonical rocket)"

    csv_env = os.environ.get("OPENROCKET_CSV", "").strip()
    if not csv_env or not Path(csv_env).exists():
        bm = Benchmark(name=name, domain="sim", reference=ref)
        bm.skipped = True
        bm.skip_reason = (
            "set OPENROCKET_CSV to an OpenRocket flight-data CSV of the canonical "
            "rocket (build it per validation/cases/rocket_canonical.py, motor "
            "L1090W, and export flight data)")
        return bm

    from validation.cases.rocket_canonical import canonical_state
    from validation.sim.headless_runner import run_flight

    or_data = parse_openrocket_csv(Path(csv_env))
    or_apogee = max(or_data["altitude"]) if or_data["altitude"] else float("nan")
    or_vmax = (max(v for v in or_data["velocity"] if v == v)
               if or_data["velocity"] else float("nan"))

    res = run_flight(canonical_state())

    bm = Benchmark(name=name, domain="sim", reference=ref,
                   level=ValidationLevel.ESTIMATED)
    bm.add(Comparison.make("Apogee", res.apogee_m, or_apogee,
                           "OpenRocket", "m", tol_rel=0.05))
    if or_vmax == or_vmax:        # not NaN
        bm.add(Comparison.make("Max velocity", res.max_velocity_ms, or_vmax,
                               "OpenRocket", "m/s", tol_rel=0.05))

    # Altitude(t) curve RMSE, normalised by apogee for a relative tolerance.
    t_k2, alt_k2 = res.series("altitude")
    rmse = curve_rmse(t_k2, alt_k2, or_data["time"], or_data["altitude"])
    bm.add(Comparison.make("Altitude(t) RMSE", rmse, 0.0,
                           "OpenRocket", "m", tol_abs=0.05 * or_apogee))
    bm.curves["altitude"] = {"x": or_data["time"], "ref": or_data["altitude"],
                             "k2": [], "xlabel": "t (s)", "ylabel": "altitude (m)"}
    return bm
