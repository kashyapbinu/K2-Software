"""
K2 AeroSim — Physical drag decomposition
========================================
Splits the integrated drag of a converged SU2 solution into components that are
each *computed from the flow field*, rather than apportioned by fixed ratios.

Two independent calculations:

**Base drag** — near-field. Integrates the pressure force over the rearward-
facing part of the wall only:

    Cd_base = -∮_base (p - p∞) (n·d̂) dA / (q∞ S_ref)

Exact, given the surface solution. "Rearward-facing" means the outward normal
lies within ``base_angle_deg`` of the downstream *body* axis, which picks out a
flat aft face (and any rearward-facing step) but deliberately excludes a
boattail — boattail drag is forebody pressure drag, not base drag. The force is
then projected onto the *wind* axis d̂ = (cos α, 0, sin α), matching the frame
the total pressure drag is reported in.

**Wave drag** — the remaining (forebody) part of the integrated pressure drag:

    Cd_wave = Cd_pressure - Cd_base        (supersonic / transonic)

This is the conventional missile-aerodynamics decomposition, and every term in
it is an exact surface integral. Supersonically the forebody pressure drag *is*
the wave drag: an inviscid body generates forebody pressure drag only through
shock compression. Below the transonic threshold it is reported as zero, since
d'Alembert's paradox says an inviscid subsonic body has no forebody pressure
drag — whatever the solver produces there is numerical, not wave drag, and is
kept visible as ``cd_forebody_pressure`` rather than relabelled.

Validated against exact Taylor–Maccoll theory: a 9.462° half-angle cone at
M=2.0 has Cp_surface = 0.09553, i.e. Cd_wave = 0.09553 on base area. This
decomposition returns 0.09028 on a "fine" mesh — **5.5% low**.

Oswatitsch's entropy-production integral (:func:`wave_drag_from_volume`) is
also implemented and is the more fundamental method, but it does *not* survive
contact with this solver's tet-only meshes: integrated over all cells it should
reproduce the total drag (0.207 for the case above) and instead returns
0.30–0.36, because numerical entropy production on a coarse tet Euler mesh is
comparable to the physical shock entropy. It is therefore kept as a
mesh-quality **diagnostic only**, logged at DEBUG level, never reported.

Base and wave are *components of* the integrated pressure drag, not additions
to it. The total is always Cd = Cd_pressure + Cd_friction, and
Cd_pressure = Cd_base + Cd_forebody_pressure exactly.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger("K2.CFD.DragDecomp")

GAMMA = 1.4
R_GAS = 287.05

# Ducros Φ above this counts as "shock-like" (irrotational compression).
_DUCROS_THRESHOLD = 0.95
# Dilatation must be compressive and non-trivial: ∇·V < -_DIV_FRACTION·u∞/L.
# Filters out smooth isentropic compressions that the Ducros sensor alone,
# being a ratio, happily reports as ~1.
_DIV_FRACTION = 0.05


def base_drag_from_surface(
    surf_vtk: Path,
    p_inf: float,
    q_inf: float,
    ref_area: float,
    aoa_deg: float = 0.0,
    base_angle_deg: float = 20.0,
) -> Optional[dict]:
    """
    Integrate the pressure force over the rearward-facing (base) wall cells.

    Returns ``{"cd_base", "base_area", "base_cp_mean", "n_cells"}`` or None if
    the surface file/fields are unusable. ``cd_base`` is positive when the base
    is at sub-ambient pressure (the normal case — a suction pulling aft).

    Two different directions are in play and mixing them was a real bug:

    * *Selection* is by the body axis — a cell is "base" when its outward
      normal lies within ``base_angle_deg`` of +x. That is a property of the
      geometry and must not rotate with the flow.
    * *Projection* is onto the wind axis, exactly as
      ``_integrate_surface_forces`` does it, because the caller subtracts this
      from a wind-axis ``cd_pressure``. Projecting onto +x instead left
      ``cd_pressure - cd_base`` comparing two frames, so the forebody/wave
      split drifted with angle of attack (and could go negative).

    With both integrals in the wind axis over disjoint cell sets,
    ``cd_base + cd_forebody_pressure == cd_pressure`` holds exactly at any AoA.
    """
    try:
        import numpy as np
        import pyvista as pv

        surf_vtk = Path(surf_vtk)
        if not surf_vtk.is_file():
            return None
        mesh = pv.read(str(surf_vtk))
        try:
            surf = mesh.extract_surface(algorithm=None)
        except TypeError:
            surf = mesh.extract_surface()
        if "Pressure" not in surf.point_data:
            return None

        # Same normal settings as _integrate_surface_forces, which reproduces
        # SU2's own force coefficients to 5 decimals on this mesh family.
        surf = surf.compute_normals(
            cell_normals=True, point_normals=False, consistent_normals=False
        )
        cell = surf.point_data_to_cell_data()
        area = np.asarray(surf.compute_cell_sizes()["Area"], dtype=float)
        normals = np.asarray(surf.cell_data["Normals"], dtype=float)
        p = np.asarray(cell["Pressure"], dtype=float)

        # Rearward-facing: outward normal within base_angle_deg of +x.
        nx = normals[:, 0]
        mask = nx >= math.cos(math.radians(base_angle_deg))
        n_sel = int(mask.sum())
        if n_sel == 0:
            logger.info(
                "No rearward-facing wall cells within %.0f° of the axis — "
                "body has no flat base, Cd_base = 0.", base_angle_deg
            )
            return {"cd_base": 0.0, "base_area": 0.0,
                    "base_cp_mean": 0.0, "n_cells": 0}

        dp = p[mask] - p_inf
        a_sel = area[mask]
        nx_sel = nx[mask]
        n_sel_vec = normals[mask]
        # Pressure force on the base cells: dF = -(p - p∞)·n_outward·dA.
        # Sub-ambient base pressure (dp < 0) on a downstream-facing cell
        # (n_x > 0) gives a positive (drag) contribution.
        f_base = (-dp[:, None] * n_sel_vec * a_sel[:, None]).sum(axis=0)
        a = math.radians(aoa_deg)
        drag_dir = np.array([math.cos(a), 0.0, math.sin(a)])
        f_drag = float(f_base @ drag_dir)
        # Projected (frontal) base area, not the wetted area of the facets.
        base_area = float((nx_sel * a_sel).sum())

        return {
            "cd_base": f_drag / (q_inf * ref_area),
            "base_area": base_area,
            "base_cp_mean": float((dp / q_inf * a_sel).sum() / max(a_sel.sum(), 1e-30)),
            "n_cells": n_sel,
        }
    except Exception as e:
        logger.warning(f"Base drag integration failed: {e}")
        return None


def split_pressure_drag(
    cd_pressure: float,
    cd_base: float,
    mach: float,
    transonic_threshold: float = 0.8,
) -> dict:
    """
    Split the integrated pressure drag into base and forebody parts, and decide
    how much of the forebody part counts as wave drag.

    Exact by construction: ``cd_forebody_pressure = cd_pressure - cd_base``,
    both terms being surface integrals. The only judgement is the threshold
    above which forebody pressure drag is *called* wave drag — below it,
    inviscid theory says there should be none, so reporting a wave-drag number
    there would be dressing up numerical error.
    """
    forebody = cd_pressure - cd_base
    if mach >= transonic_threshold:
        return {
            "cd_forebody_pressure": forebody,
            "cd_wave": max(0.0, forebody),
            "method": f"forebody pressure integral (M={mach:g} ≥ "
                      f"{transonic_threshold:g}, treated as wave drag)",
        }
    return {
        "cd_forebody_pressure": forebody,
        "cd_wave": 0.0,
        "method": f"subsonic (M={mach:g}) — no wave drag; forebody pressure "
                  f"drag {forebody:.5f} reported separately",
    }


def wave_drag_from_volume(
    vol_vtk: Path,
    p_inf: float,
    rho_inf: float,
    T_inf: float,
    u_inf: float,
    q_inf: float,
    ref_area: float,
    ref_length: float,
    mach: float,
    body_bounds: Optional[tuple] = None,
) -> Optional[dict]:
    """
    Wave drag by Oswatitsch entropy production over shock-flagged cells.

    DIAGNOSTIC ONLY — do not report this as the wave drag. On the tet-only
    meshes this solver produces, spurious numerical entropy is comparable to
    the physical shock entropy: integrated over every cell this formula should
    return the total drag and instead overshoots it by 45–75%. Use
    :func:`split_pressure_drag` for the reported value. The ratio of this
    number to the surface-integral one is, however, a useful mesh-quality
    signal — the closer to 1, the less numerical dissipation.

    ``body_bounds`` is an optional (xmin, xmax, ymin, ymax, zmin, zmax) of the
    body; the integration region is grown from it so the far-field cells — huge,
    and carrying only discretisation noise — cannot contribute.

    Returns ``{"cd_wave", "n_shock_cells", "shock_volume", "method"}`` or None.
    Subsonic freestreams short-circuit to zero without touching the mesh.
    """
    try:
        import numpy as np
        import pyvista as pv

        vol_vtk = Path(vol_vtk)
        if not vol_vtk.is_file():
            return None

        # No freestream shocks below M≈0.7; local pockets can appear
        # transonically, so only hard-zero the clearly subsonic case.
        if mach < 0.7:
            return {"cd_wave": 0.0, "n_shock_cells": 0, "shock_volume": 0.0,
                    "method": "subsonic (no shocks)"}

        mesh = pv.read(str(vol_vtk))
        need = {"Pressure", "Density", "Velocity"}
        if not need.issubset(set(mesh.array_names)):
            logger.warning(
                f"Volume file lacks {need - set(mesh.array_names)} — "
                "wave drag unavailable."
            )
            return None

        # Restrict to a region around the body: the outer mesh is far too
        # coarse for a gradient integral to mean anything there.
        #
        # Selection is by cell centroid via extract_cells rather than
        # clip_box — clip_box's `invert` flag keeps the region OUTSIDE the
        # bounds, which silently handed this integral nothing but undisturbed
        # freestream and reported Cd_wave = 0 at M = 2.
        #
        # The radial limit follows the Mach cone: at M the disturbance reaches
        # r = x·tan(asin(1/M)) — 0.29 m by the tail of a 0.5 m body at M=2, so
        # a body-scaled box (8·r_body = 0.16 m) would cut the shock away.
        if body_bounds is not None:
            x0, x1 = body_bounds[0], body_bounds[1]
            L = max(x1 - x0, ref_length, 1e-9)
            r_body = max(
                abs(body_bounds[3] - body_bounds[2]),
                abs(body_bounds[5] - body_bounds[4]),
            ) * 0.5
            r_body = max(r_body, 0.05 * L)

            x_lo, x_hi = x0 - 2.0 * L, x1 + 5.0 * L
            if mach > 1.0:
                mu = math.asin(min(1.0 / mach, 1.0))      # Mach angle
                r_max = max(8.0 * r_body, (x_hi - x0) * math.tan(mu))
            else:
                r_max = max(8.0 * r_body, 2.0 * L)

            try:
                cc = np.asarray(mesh.cell_centers().points, dtype=float)
                sel = (
                    (cc[:, 0] >= x_lo) & (cc[:, 0] <= x_hi)
                    & (np.abs(cc[:, 1]) <= r_max) & (np.abs(cc[:, 2]) <= r_max)
                )
                n_sel = int(sel.sum())
                if n_sel:
                    mesh = mesh.extract_cells(np.flatnonzero(sel))
                    logger.info(
                        f"Shock integration region: x[{x_lo:.3f}, {x_hi:.3f}] "
                        f"r<{r_max:.3f} m → {n_sel:,} of {len(cc):,} cells"
                    )
                else:
                    logger.warning("Shock region selected no cells — using full domain.")
            except Exception as e:
                logger.warning(f"Shock-region selection failed ({e}) — using full domain.")
        if mesh.n_cells == 0:
            return {"cd_wave": 0.0, "n_shock_cells": 0, "shock_volume": 0.0,
                    "method": "empty integration region"}

        p = np.asarray(mesh["Pressure"], dtype=float)
        rho = np.asarray(mesh["Density"], dtype=float)
        vel = np.asarray(mesh["Velocity"], dtype=float)

        # Specific entropy referenced to freestream:
        #   s = R/(γ-1) · [ln(p/p∞) - γ·ln(ρ/ρ∞)]
        # The additive constant is irrelevant — only ∇s enters the formula.
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (R_GAS / (GAMMA - 1.0)) * (
                np.log(np.maximum(p / p_inf, 1e-30))
                - GAMMA * np.log(np.maximum(rho / rho_inf, 1e-30))
            )
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        mesh["_Entropy"] = s

        grad = mesh.compute_derivative(scalars="_Entropy", gradient=True)
        grad_s = np.asarray(grad["gradient"], dtype=float)

        div = mesh.compute_derivative(scalars="Velocity", divergence=True)
        div_v = np.asarray(div["divergence"], dtype=float).ravel()

        # Shock mask: irrotational (Ducros) AND compressive (dilatation).
        from cfd.shock_detection import ducros_shock_sensor
        try:
            ducros = np.asarray(
                ducros_shock_sensor(mesh, prefilter=False)["Ducros_Sensor"],
                dtype=float,
            )
        except Exception as e:
            logger.warning(f"Ducros sensor unavailable ({e}) — using dilatation only.")
            ducros = np.ones_like(div_v)

        div_floor = -_DIV_FRACTION * u_inf / max(ref_length, 1e-9)
        shock_pts = (ducros > _DUCROS_THRESHOLD) & (div_v < div_floor)

        # Oswatitsch integrand, per point: ρ (u · ∇s)
        integrand = rho * np.einsum("ij,ij->i", vel, grad_s)
        integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
        mesh["_OswIntegrand"] = np.where(shock_pts, integrand, 0.0)
        mesh["_ShockMask"] = shock_pts.astype(float)

        cellified = mesh.point_data_to_cell_data()
        vol = np.asarray(
            mesh.compute_cell_sizes(length=False, area=False, volume=True)["Volume"],
            dtype=float,
        )
        cell_integrand = np.asarray(cellified["_OswIntegrand"], dtype=float)
        cell_mask = np.asarray(cellified["_ShockMask"], dtype=float) > 0.5

        d_wave = float((cell_integrand * np.abs(vol)).sum()) * T_inf / max(u_inf, 1e-30)
        cd_wave = d_wave / (q_inf * ref_area)

        # Entropy cannot decrease through a shock, so a negative integral means
        # the mask caught numerical noise rather than a wave. Report zero.
        if cd_wave < 0:
            logger.info(
                f"Oswatitsch integral came out negative ({cd_wave:.5f}) — "
                "mask is picking up discretisation noise, reporting Cd_wave = 0."
            )
            cd_wave = 0.0

        return {
            "cd_wave": cd_wave,
            "n_shock_cells": int(cell_mask.sum()),
            "shock_volume": float(np.abs(vol[cell_mask]).sum()),
            "method": "Oswatitsch entropy production over Ducros-flagged cells",
        }
    except Exception as e:
        logger.warning(f"Wave drag integration failed: {e}")
        return None
