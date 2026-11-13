"""
K2 AeroSim — Weather soundings
================================
Read a measured atmosphere (a radiosonde sounding or a forecast profile)
and turn it into what a flight uses:

- ``atmosphere_profile``: ``[[altitude_m ASL, pressure_Pa, temperature_K],
  ...]`` for environment.atmosphere_model.ProfileAtmosphere;
- ``wind_layers``: ``[(height_m above the pad, speed_m_s, from_deg), ...]``
  for environment.wind_model.MultiLevelWindModel.

Formats read:

1. University of Wyoming "Text: List" soundings (the page text, or just the
   table). Columns PRES (hPa), HGHT (m), TEMP (°C), DRCT (deg), SKNT (knot).
   Rows without a temperature are skipped.
2. CSV with a header row. Column names say the quantity and its unit:

   ====================  ==============================================
   quantity              accepted headers (case-insensitive)
   ====================  ==============================================
   altitude ASL          height_m, altitude_m, alt_m, hght_m,
                         height_ft, altitude_ft
   pressure              pressure_hpa, pres_hpa, pressure_mb,
                         pressure_pa, pressure_kpa
   temperature           temperature_c, temp_c, temperature_k, temp_k,
                         temperature_f
   wind speed            wind_speed_ms, wind_speed_m_s, speed_ms,
                         wind_speed_kt, wind_speed_knots, sknt,
                         wind_speed_kmh, wind_speed_mph
   wind direction (FROM) wind_dir_deg, wind_direction_deg, drct, dir_deg
   ====================  ==============================================

   Altitude, pressure and temperature are required; wind columns are
   optional. A header without a unit is refused rather than guessed.

Altitudes in a sounding are above sea level. Wind layers are above the pad,
so they are converted with the launch-site elevation, and levels below the
pad are dropped.

No Qt imports.
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

KNOT = 0.514444          # m/s
FOOT = 0.3048            # m


class SoundingError(ValueError):
    """The text is not a sounding this module can read; the message says why."""


@dataclass
class Level:
    altitude: float                      # m above sea level
    pressure: float                      # Pa
    temperature: float                   # K
    wind_speed: Optional[float] = None   # m/s
    wind_from: Optional[float] = None    # deg, bearing the wind blows FROM


@dataclass
class Sounding:
    levels: List[Level]
    source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def has_wind(self) -> bool:
        return any(l.wind_speed is not None for l in self.levels)

    def atmosphere_profile(self) -> List[list]:
        return [[l.altitude, l.pressure, l.temperature] for l in self.levels]

    def wind_layers(self, site_altitude: float) -> List[tuple]:
        """Wind levels above the pad as (height above pad, speed, from_deg).

        The pad itself gets the interpolated wind at its elevation, so the
        lowest layer is the surface wind rather than the first level above it.
        """
        winds = [(l.altitude, l.wind_speed, l.wind_from) for l in self.levels
                 if l.wind_speed is not None and l.wind_from is not None]
        if not winds:
            return []
        above = [(h - site_altitude, s, d) for h, s, d in winds if h >= site_altitude]
        below = [w for w in winds if w[0] < site_altitude]
        if below and above and above[0][0] > 0.0:
            (h0, s0, d0), (h1, s1, d1) = below[-1], (above[0][0] + site_altitude,
                                                     above[0][1], above[0][2])
            f = (site_altitude - h0) / (h1 - h0)
            delta = ((d1 - d0 + 180.0) % 360.0) - 180.0
            above.insert(0, (0.0, s0 + f * (s1 - s0), (d0 + f * delta) % 360.0))
        return [(round(h, 3), max(0.0, s), d % 360.0) for h, s, d in above]

    def surface(self, site_altitude: float) -> Tuple[float, float]:
        """(pressure_Pa, temperature_K) at the pad, from the profile."""
        from environment.atmosphere_model import ProfileAtmosphere
        atm = ProfileAtmosphere(self.atmosphere_profile())
        return atm.pressure(site_altitude), atm.temperature(site_altitude)


def _finalise(levels: List[Level], source: str) -> Sounding:
    levels = sorted(levels, key=lambda l: l.altitude)
    deduped: List[Level] = []
    for l in levels:
        if deduped and abs(l.altitude - deduped[-1].altitude) < 1e-6:
            continue                       # repeated height: keep the first
        deduped.append(l)
    if len(deduped) < 2:
        raise SoundingError("a sounding needs at least two levels with "
                            "altitude, pressure and temperature")
    for a, b in zip(deduped, deduped[1:]):
        if b.pressure >= a.pressure:
            raise SoundingError(
                f"pressure does not fall with altitude between {a.altitude:.0f} m "
                f"and {b.altitude:.0f} m — check the pressure and altitude units")
    from environment.atmosphere_model import pressure_at_altitude
    for l in deduped:
        if not (150.0 <= l.temperature <= 350.0):
            raise SoundingError(
                f"temperature {l.temperature:.1f} K at {l.altitude:.0f} m is not "
                f"plausible — check the temperature unit")
        # Weather moves pressure a few percent off the standard value at a
        # given height; a unit slip (hPa read as Pa, feet read as metres)
        # moves it by a large factor.
        ratio = l.pressure / pressure_at_altitude(max(0.0, l.altitude))
        if not (0.6 <= ratio <= 1.4):
            raise SoundingError(
                f"pressure {l.pressure:.0f} Pa at {l.altitude:.0f} m is "
                f"{ratio:.2g}× the standard value there — check the pressure "
                f"and altitude units")
    return Sounding(deduped, source)


# ── University of Wyoming "Text: List" ─────────────────────────────────────

def parse_uwyo(text: str, source: str = "") -> Sounding:
    lines = text.splitlines()
    header_i = next((i for i, ln in enumerate(lines)
                     if "PRES" in ln and "HGHT" in ln and "TEMP" in ln), None)
    if header_i is None:
        raise SoundingError("no PRES / HGHT / TEMP header: not a Wyoming text list")
    names = lines[header_i].split()
    col = {n: i for i, n in enumerate(names)}
    width = 7                                # the table's fixed column width

    def cell(row: str, name: str) -> Optional[float]:
        if name not in col:
            return None
        piece = row[col[name] * width:(col[name] + 1) * width].strip()
        try:
            return float(piece) if piece else None
        except ValueError:
            return None

    levels = []
    started = False
    for ln in lines[header_i + 1:]:
        if ln.strip().startswith("---"):
            if started:
                break
            continue
        stripped = ln.strip()
        if not stripped or stripped[0].isalpha():
            if started:
                break
            continue                         # the units line
        started = True
        p, h, t = cell(ln, "PRES"), cell(ln, "HGHT"), cell(ln, "TEMP")
        if p is None or h is None or t is None:
            continue
        spd, drc = cell(ln, "SKNT"), cell(ln, "DRCT")
        levels.append(Level(
            altitude=h, pressure=p * 100.0, temperature=t + 273.15,
            wind_speed=spd * KNOT if spd is not None else None,
            wind_from=drc if drc is not None else None))
    return _finalise(levels, source or "University of Wyoming sounding")


# ── CSV ────────────────────────────────────────────────────────────────────

_ALT = {"height_m": 1.0, "altitude_m": 1.0, "alt_m": 1.0, "hght_m": 1.0,
        "height_ft": FOOT, "altitude_ft": FOOT}
_PRES = {"pressure_hpa": 100.0, "pres_hpa": 100.0, "pressure_mb": 100.0,
         "pressure_pa": 1.0, "pressure_kpa": 1000.0}
_TEMP = {"temperature_c": "C", "temp_c": "C", "temperature_k": "K",
         "temp_k": "K", "temperature_f": "F"}
_SPD = {"wind_speed_ms": 1.0, "wind_speed_m_s": 1.0, "speed_ms": 1.0,
        "wind_speed_kt": KNOT, "wind_speed_knots": KNOT, "sknt": KNOT,
        "wind_speed_kmh": 1.0 / 3.6, "wind_speed_mph": 0.44704}
_DIR = {"wind_dir_deg", "wind_direction_deg", "drct", "dir_deg"}


def _norm(h: str) -> str:
    h = h.strip().lower()
    for a, b in (("(", "_"), (")", ""), ("[", "_"), ("]", ""), (" ", "_"),
                 ("/", "_"), ("°", "")):
        h = h.replace(a, b)
    while "__" in h:
        h = h.replace("__", "_")
    return h.strip("_")


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "C":
        return value + 273.15
    if unit == "F":
        return (value - 32.0) * 5.0 / 9.0 + 273.15
    return value


def parse_csv(text: str, source: str = "") -> Sounding:
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(c.strip() for c in r)
            and not r[0].lstrip().startswith("#")]
    if not rows:
        raise SoundingError("the file is empty")
    header = [_norm(h) for h in rows[0]]

    def find(table):
        for i, h in enumerate(header):
            if h in table:
                return i, (table[h] if isinstance(table, dict) else None)
        return None, None

    ia, alt_scale = find(_ALT)
    ip, p_scale = find(_PRES)
    it, t_unit = find(_TEMP)
    iw, w_scale = find(_SPD)
    idir, _ = find(_DIR)
    missing = [name for name, i in (("altitude", ia), ("pressure", ip),
                                    ("temperature", it)) if i is None]
    if missing:
        raise SoundingError(
            "no column for " + ", ".join(missing) + ". Headers must name the "
            "unit, e.g. height_m, pressure_hPa, temperature_C "
            f"(found: {', '.join(rows[0])})")

    def num(row, i):
        if i is None or i >= len(row):
            return None
        try:
            v = float(row[i])
        except ValueError:
            return None
        return v if math.isfinite(v) else None

    levels = []
    for row in rows[1:]:
        h, p, t = num(row, ia), num(row, ip), num(row, it)
        if h is None or p is None or t is None:
            continue
        spd, drc = num(row, iw), num(row, idir)
        levels.append(Level(
            altitude=h * alt_scale, pressure=p * p_scale,
            temperature=_to_kelvin(t, t_unit),
            wind_speed=spd * w_scale if spd is not None else None,
            wind_from=drc if (drc is not None and spd is not None) else None))
    return _finalise(levels, source or "CSV sounding")


def parse_sounding(text: str, source: str = "") -> Sounding:
    """Read either format. Wyoming text is recognised by its PRES/HGHT
    header; anything else is read as CSV."""
    if "PRES" in text and "HGHT" in text:
        return parse_uwyo(text, source)
    return parse_csv(text, source)


def read_sounding_file(path: str) -> Sounding:
    import os
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse_sounding(f.read(), source=os.path.basename(path))
