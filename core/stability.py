"""
K2 AeroSim — Stability verdict
================================
One reading of a static margin, for every screen that puts a word or a colour
on it.

There were four. At 0.3 cal the Design tab said UNSTABLE, the Properties
panel beside it "MARGINAL–" and the assistant "marginal"; at 2.8 cal both
panels said OVERSTABLE and the assistant "stable". The Mission Visualizer
flagged "FAILURE: Instability" below 0.5 cal, while the flight's own warning
(core.flight_summary.flight_warnings) is raised for a margin below zero.

    unstable     margin < 0: the centre of pressure is ahead of the CG
    marginal     0 to 1 cal
    stable       1 to 2.5 cal
    overstable   above 2.5 cal: flies, but turns into the wind harder

The 1 and 2.5 cal marks are the usual rule of thumb, as the two panels had
them; a competition's own limits belong in the rules check (core.rules).

No Qt imports.
"""
from __future__ import annotations

import math
from typing import Optional

UNSTABLE = "unstable"
MARGINAL = "marginal"
STABLE = "stable"
OVERSTABLE = "overstable"

MARGINAL_BELOW = 1.0      # calibres
OVERSTABLE_ABOVE = 2.5    # calibres

# The word each read-out shows.
LABELS = {UNSTABLE: "UNSTABLE", MARGINAL: "MARGINAL",
          STABLE: "✓ STABLE", OVERSTABLE: "OVERSTABLE"}


def stability_verdict(margin_cal) -> Optional[str]:
    """UNSTABLE / MARGINAL / STABLE / OVERSTABLE for a static margin in
    calibres; None when there is no margin to judge (None, NaN)."""
    if margin_cal is None:
        return None
    try:
        margin = float(margin_cal)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(margin):
        return None
    if margin < 0.0:
        return UNSTABLE
    if margin < MARGINAL_BELOW:
        return MARGINAL
    if margin <= OVERSTABLE_ABOVE:
        return STABLE
    return OVERSTABLE
