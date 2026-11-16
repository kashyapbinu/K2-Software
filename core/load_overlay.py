"""
K2 AeroSim — Loads along the rocket
=====================================
A CFD or FE result reduced to one value per axial station, so the flight
replay can paint it on the rocket it is drawing.

The CFD and Structures tabs each show their field on their own mesh, in
their own tab, at one condition. Neither says where on the vehicle the load
sits while it is flying. A ``LoadProfile`` is that field as a function of
distance from the nose tip — the frame both solvers already use
(cfd.meshing: nose tip at x = 0; structures.meshing: z = 0 at the nose tip).

What is and is not shown:

- CFD pressure is the surface pressure coefficient of the *one* solved
  condition (its Mach and angle of attack), averaged round the body at each
  station. The replay multiplies it by the dynamic pressure of the instant
  shown, giving gauge pressure (p − p∞). Away from the solved Mach number
  that is an approximation, and the overlay says which Mach it was solved at.
- FE stress is the von Mises profile of the *one* static load case that was
  run. It does not change with the replay instant.

Nothing is drawn for an analysis that has not been run.

No Qt imports.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("K2.LoadOverlay")

CFD_PRESSURE = "cfd_pressure"
FE_STRESS = "fe_stress"


@dataclass
class LoadProfile:
    kind: str                 # CFD_PRESSURE | FE_STRESS
    x: np.ndarray             # station, m from the nose tip, ascending
    value: np.ndarray         # Cp (CFD) or von Mises stress in Pa (FE)
    source: str               # what was solved, for the caption
    scales_with_q: bool       # True: multiply by dynamic pressure to get Pa

    def at(self, stations, q: float = 1.0) -> np.ndarray:
        """Value at each station (m from the nose); held at the end values
        beyond the profile. For a CFD profile pass the dynamic pressure
        *q* (Pa) to get gauge pressure in Pa."""
        v = np.interp(np.asarray(stations, dtype=float), self.x, self.value)
        return v * q if self.scales_with_q else v

    def limits(self, q_max: float = 1.0):
        """Colour-scale limits over a whole flight whose peak dynamic
        pressure is *q_max*: symmetric about zero for pressure, zero to
        peak for stress."""
        if self.scales_with_q:
            # 98th percentile, not the maximum: one station at the base or
            # the nose tip would otherwise wash the rest of the body out.
            peak = float(np.percentile(np.abs(self.value), 98)) * max(q_max, 1e-9)
            return -peak, peak
        return 0.0, float(np.max(self.value))


def cfd_pressure_profile(vtk_path, mach: float = 0.0, bins: int = 80) -> Optional[LoadProfile]:
    """Pressure coefficient along the body from a CFD surface file (the
    solver's surface_flow.vtu). None when the file is missing or holds no
    pressure coefficient.

    Each station's value is the mean over the body surface there. Fin
    surfaces are left out (their points lie well outside the body radius at
    that station), so a fin's leading-edge suction does not read as a ring
    of low pressure round the tube.
    """
    if not vtk_path or not Path(vtk_path).is_file():
        return None
    try:
        import pyvista as pv
        mesh = pv.read(str(vtk_path))
    except Exception as exc:
        logger.warning("Could not read CFD surface %s: %s", vtk_path, exc)
        return None
    name = next((n for n in ("Pressure_Coefficient", "Cp") if n in mesh.array_names), None)
    if name is None or mesh.n_points < 10:
        return None
    return profile_from_points(mesh.points, np.asarray(mesh[name], dtype=float),
                               mach=mach, bins=bins)


def profile_from_points(points, cp, mach: float = 0.0, bins: int = 80) -> Optional[LoadProfile]:
    """The CFD profile from surface points (x along the body from the nose
    tip, y and z lateral) and the pressure coefficient at each."""
    points = np.asarray(points, dtype=float)
    cp = np.asarray(cp, dtype=float).reshape(-1)
    ok = np.isfinite(cp) & np.isfinite(points).all(axis=1)
    points, cp = points[ok], cp[ok]
    if len(cp) < 10:
        return None
    x = points[:, 0]
    r = np.hypot(points[:, 1], points[:, 2])
    edges = np.linspace(x.min(), x.max(), bins + 1)
    which = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    xs, vals = [], []
    for b in range(bins):
        sel = which == b
        if not sel.any():
            continue
        rb, cb = r[sel], cp[sel]
        if b < bins - 1:
            # The body is the inner surface at this station; anything well
            # outside it is a fin. (The last bin keeps the base face, whose
            # points run from the axis to the rim.)
            # 8 % covers the change of radius across one bin of a nose cone
            # or transition; a fin's own root strip inside that band is a
            # small share of the ring.
            body_r = np.percentile(rb, 10)
            keep = rb <= max(body_r * 1.08, body_r + 1e-4)
            if keep.any():
                cb = cb[keep]
        xs.append(0.5 * (edges[b] + edges[b + 1]) - x.min())
        vals.append(float(cb.mean()))
    if len(xs) < 2:
        return None
    src = f"CFD at Mach {mach:g}" if mach > 0 else "CFD"
    return LoadProfile(CFD_PRESSURE, np.asarray(xs), np.asarray(vals), src, True)


def fe_stress_profile(fem_result, condition: str = "") -> Optional[LoadProfile]:
    """Von Mises stress along the body from a static FE result
    (``element_stresses``: [(station from the nose, Pa), ...]). Where several
    elements share a station the highest is kept. None without a result."""
    rows = getattr(fem_result, "element_stresses", None)
    if not rows:
        return None
    peak = {}
    for x, s in rows:
        x, s = float(x), float(s)
        if np.isfinite(x) and np.isfinite(s):
            peak[x] = max(peak.get(x, 0.0), s)
    if len(peak) < 2:
        return None
    xs = np.asarray(sorted(peak))
    src = "FE static" + (f", {condition}" if condition else "")
    return LoadProfile(FE_STRESS, xs, np.asarray([peak[x] for x in xs]), src, False)
