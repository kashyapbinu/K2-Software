"""
K2 AeroSim — Display units
============================
Everything inside K2 is SI. This module converts at the edge, for display
and for reading a typed value back, according to the "display/units"
setting: "SI" or "imperial" (what US competitions quote: ft, ft/s, lb, in,
lbf).

One table, one pair of functions. A widget asks for a quantity kind, never
for a conversion factor, so no screen can pick a different factor from
another.

    format_value(3048.0, "altitude")   -> "3,048 m"   or "10,000 ft"
    to_si(10000, "altitude")           -> 3048.0      (imperial)
"""
from __future__ import annotations

import math
from typing import Optional

FT = 0.3048
LB = 0.45359237
LBF = 4.4482216152605
IN = 0.0254

# kind -> {system: (unit label, SI value per display unit, decimals)}
KINDS = {
    "altitude":     {"SI": ("m", 1.0, 0),     "imperial": ("ft", FT, 0)},
    "distance":     {"SI": ("m", 1.0, 0),     "imperial": ("ft", FT, 0)},
    "length":       {"SI": ("m", 1.0, 3),     "imperial": ("in", IN, 2)},
    "velocity":     {"SI": ("m/s", 1.0, 1),   "imperial": ("ft/s", FT, 0)},
    "acceleration": {"SI": ("m/s²", 1.0, 1),  "imperial": ("ft/s²", FT, 0)},
    "mass":         {"SI": ("kg", 1.0, 3),    "imperial": ("lb", LB, 2)},
    "force":        {"SI": ("N", 1.0, 0),     "imperial": ("lbf", LBF, 1)},
    "pressure":     {"SI": ("kPa", 1000.0, 1), "imperial": ("psi", 6894.757293168, 2)},
}
SYSTEMS = ("SI", "imperial")


def current_system() -> str:
    try:
        from ui import settings
        system = settings.get("display/units") or "SI"
    except Exception:
        system = "SI"
    return system if system in SYSTEMS else "SI"


def unit_label(kind: str, system: Optional[str] = None) -> str:
    return KINDS[kind][system or current_system()][0]


def from_si(value: float, kind: str, system: Optional[str] = None) -> float:
    return value / KINDS[kind][system or current_system()][1]


def to_si(value: float, kind: str, system: Optional[str] = None) -> float:
    return value * KINDS[kind][system or current_system()][1]


def format_value(value, kind: str, system: Optional[str] = None,
                 decimals: Optional[int] = None) -> str:
    """SI *value* as display text with its unit; "—" for None / non-finite."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    system = system or current_system()
    label, factor, dec = KINDS[kind][system]
    dec = dec if decimals is None else decimals
    return f"{value / factor:,.{dec}f} {label}"


def parse_value(text: str, kind: str, system: Optional[str] = None) -> float:
    """Typed display text (unit optional, thousands separators allowed) back
    to SI. Raises ValueError for text that is not a number."""
    system = system or current_system()
    label = KINDS[kind][system][0]
    t = text.strip().replace(",", "")
    if t.endswith(label):
        t = t[: -len(label)].strip()
    return to_si(float(t), kind, system)
