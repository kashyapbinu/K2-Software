"""
K2 AeroSim — CFD → FEM Pressure Mapping
==========================================
Transfers aerodynamic pressure distributions from CFD solutions onto
structural FEM meshes and verifies force / moment conservation.

Three mapping strategies are provided:

1. **Analytical** — evaluates a Cp(x) function at FEM element stations
   using linear shape-function interpolation (default, no dependencies).
2. **Nearest-neighbour** — snaps each FEM element centroid to the
   closest CFD surface point.
3. **IDW (Inverse-Distance Weighting)** — Shepard interpolation from
   the *k* nearest CFD points.

The module can also attempt to parse simple VTU (VTK Unstructured Grid)
XML files to extract point coordinates and associated pressure data
without requiring the VTK library.

References
----------
- Anderson, *Fundamentals of Aerodynamics*, 6th ed., §3.12 (Cp definition)
- CalculiX User Manual v2.21, §6.4.2 (*DLOAD keyword)
- Shepard, 1968, "A two-dimensional interpolation function for
  irregularly-spaced data", *Proc. 23rd ACM Natl. Conf.*
"""
from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple

logger = logging.getLogger("K2.PressureMap")


# ── Result Dataclass ─────────────────────────────────────────────────────────

@dataclass
class PressureMappingResult:
    """Stores the outcome of a CFD → FEM pressure transfer and the
    force / moment conservation check.

    Attributes
    ----------
    cfd_total_force : tuple
        Resultant force integrated over the CFD surface (Fx, Fy, Fz) [N].
    cfd_total_moment : tuple
        Resultant moment about the origin (Mx, My, Mz) [N·m].
    fem_total_force : tuple
        Resultant force after mapping onto FEM elements (Fx, Fy, Fz) [N].
    fem_total_moment : tuple
        Resultant moment after mapping (Mx, My, Mz) [N·m].
    force_error_pct : float
        ‖F_fem − F_cfd‖ / ‖F_cfd‖ × 100  [%].
    moment_error_pct : float
        ‖M_fem − M_cfd‖ / ‖M_cfd‖ × 100  [%].
    num_cfd_points : int
        Number of CFD surface points used.
    num_fem_elements : int
        Number of FEM elements that received loads.
    mapping_method : str
        ``'analytical'``, ``'nearest'``, or ``'IDW'``.
    accepted : bool
        ``True`` if both force and moment errors are < 2 %.
    element_pressures : list
        Per-element pressure assignments ``[(elem_id, pressure_Pa), ...]``.
    """
    cfd_total_force: tuple = (0.0, 0.0, 0.0)     # (Fx, Fy, Fz) N
    cfd_total_moment: tuple = (0.0, 0.0, 0.0)     # (Mx, My, Mz) N·m
    fem_total_force: tuple = (0.0, 0.0, 0.0)
    fem_total_moment: tuple = (0.0, 0.0, 0.0)
    force_error_pct: float = 0.0
    moment_error_pct: float = 0.0
    num_cfd_points: int = 0
    num_fem_elements: int = 0
    mapping_method: str = 'analytical'             # 'IDW'|'nearest'|'analytical'
    accepted: bool = False                          # True if error < 2%
    element_pressures: list = field(default_factory=list)


# ── Vector Helpers (pure-Python, no numpy) ───────────────────────────────────

def _vec_mag(v: tuple) -> float:
    """Euclidean magnitude of a 3-tuple."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vec_sub(a: tuple, b: tuple) -> tuple:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: tuple, b: tuple) -> tuple:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_scale(v: tuple, s: float) -> tuple:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_cross(a: tuple, b: tuple) -> tuple:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


# ── Simple VTU Parser ────────────────────────────────────────────────────────

_PRESSURE_ARRAY_NAMES = ('pressure', 'p', 'cp')


def _parse_vtu_via_pyvista(
    vtu_path: Path,
) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Read points + pressure through a full VTK reader (handles the
    appended-binary VTU files SU2 writes). Returns ([], []) when pyvista
    is unavailable or the file cannot be read.
    """
    try:
        import pyvista as pv
    except ImportError:
        return [], []
    try:
        mesh = pv.read(str(vtu_path))
    except Exception as exc:
        logger.debug("pyvista could not read %s: %s", vtu_path, exc)
        return [], []

    points = [tuple(map(float, p)) for p in mesh.points]

    pressures: List[float] = []
    for name in mesh.point_data.keys():
        if name.lower() in _PRESSURE_ARRAY_NAMES:
            pressures = [float(v) for v in mesh.point_data[name]]
            break
    else:
        # Pressure stored per-cell — interpolate to points.
        for name in mesh.cell_data.keys():
            if name.lower() in _PRESSURE_ARRAY_NAMES:
                interp = mesh.cell_data_to_point_data()
                pressures = [float(v) for v in interp.point_data[name]]
                break
    return points, pressures


def _parse_vtu_points_and_pressure(
    vtu_path: Path,
) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    """Extract point coordinates and pressure from a VTU file.

    Tries a full VTK reader first (SU2 writes ``format="appended"``
    binary VTU, which plain XML parsing cannot handle), then falls back
    to parsing the ``<Points>`` and ``<PointData>`` sections of an
    ASCII / raw-text ``.vtu`` file.

    Parameters
    ----------
    vtu_path : Path
        Path to the ``.vtu`` file.

    Returns
    -------
    points : list of (x, y, z)
    pressures : list of float
        Pressure at each point (Pa).  Empty if no pressure array found.
    """
    points, pressures = _parse_vtu_via_pyvista(vtu_path)
    if points and pressures:
        logger.debug("VTU parse (pyvista): %d points, %d pressure values",
                     len(points), len(pressures))
        return points, pressures

    tree = ET.parse(str(vtu_path))
    root = tree.getroot()

    points: list = []
    pressures: list = []

    # ── Points ──
    for pts_elem in root.iter('Points'):
        for da in pts_elem.iter('DataArray'):
            ncomp = int(da.get('NumberOfComponents', '3'))
            text = (da.text or '').strip()
            if not text:
                continue
            vals = [float(v) for v in text.split()]
            for i in range(0, len(vals), ncomp):
                x = vals[i]
                y = vals[i + 1] if ncomp > 1 else 0.0
                z = vals[i + 2] if ncomp > 2 else 0.0
                points.append((x, y, z))

    # ── Pressure in PointData ──
    for pd in root.iter('PointData'):
        for da in pd.iter('DataArray'):
            name = (da.get('Name') or '').lower()
            if name in _PRESSURE_ARRAY_NAMES:
                text = (da.text or '').strip()
                if text:
                    pressures = [float(v) for v in text.split()]
                break

    logger.debug("VTU parse: %d points, %d pressure values",
                 len(points), len(pressures))
    return points, pressures


# ── Analytical Mapping ───────────────────────────────────────────────────────

def map_pressures_analytical(
    cp_distribution: Callable[[float], float],
    fem_stations: List[Tuple[int, float, float, float]],
    q_inf: float,
    p_inf: float = 101325.0,
    reference_area_m2: float = 1.0,
    body_radius_m: float = 0.05,
) -> PressureMappingResult:
    """Map an analytical Cp(x) distribution onto FEM element stations.

    For each FEM station the local pressure is:

    .. math::
        p(x) = p_\\infty + C_p(x) \\cdot q_\\infty

    and the axially-projected element force is:

    .. math::
        F_{x,i} = -p(x_i) \\cdot A_i

    (negative because positive Cp acts inward / aft on the body).

    Parameters
    ----------
    cp_distribution : callable
        ``Cp(x)`` returning the pressure coefficient at axial station
        *x* (m).  Must accept a single float.
    fem_stations : list of (elem_id, x, area, outward_nx)
        Each entry provides the element ID, axial coordinate (m),
        tributary area (m²), and outward-normal x-component.
    q_inf : float
        Freestream dynamic pressure (Pa).
    p_inf : float
        Freestream static pressure (Pa).  Default 101 325 Pa (sea level).
    reference_area_m2 : float
        Reference area for integrated Cp (used for CFD force estimate).
    body_radius_m : float
        Body radius for moment-arm estimation.

    Returns
    -------
    PressureMappingResult
    """
    result = PressureMappingResult(mapping_method='analytical')
    element_pressures: list = []

    fem_fx, fem_fy, fem_fz = 0.0, 0.0, 0.0
    fem_mx, fem_my, fem_mz = 0.0, 0.0, 0.0

    for elem_id, x_m, area_m2, nx in fem_stations:
        cp = cp_distribution(x_m)
        p_local = p_inf + cp * q_inf                        # Pa
        element_pressures.append((elem_id, p_local))

        # Force contribution (pressure × area, projected along outward normal)
        f_x = -p_local * area_m2 * nx
        fem_fx += f_x

        # Moment about origin (simplified: arm = body_radius for lateral,
        # x for axial-induced bending)
        fem_mz += f_x * body_radius_m  # bending moment from pressure

    result.fem_total_force = (fem_fx, fem_fy, fem_fz)
    result.fem_total_moment = (fem_mx, fem_my, fem_mz)
    result.element_pressures = element_pressures
    result.num_fem_elements = len(fem_stations)

    # Estimate CFD total force by trapezoidal integration over stations
    if fem_stations:
        sorted_st = sorted(fem_stations, key=lambda s: s[1])
        cfd_fx = 0.0
        for i in range(len(sorted_st) - 1):
            _, x0, a0, nx0 = sorted_st[i]
            _, x1, a1, nx1 = sorted_st[i + 1]
            cp0 = cp_distribution(x0)
            cp1 = cp_distribution(x1)
            p0 = p_inf + cp0 * q_inf
            p1 = p_inf + cp1 * q_inf
            avg_p = 0.5 * (p0 + p1)
            avg_a = 0.5 * (a0 + a1)
            avg_nx = 0.5 * (nx0 + nx1)
            cfd_fx += -avg_p * avg_a * avg_nx
        result.cfd_total_force = (cfd_fx, 0.0, 0.0)
        result.num_cfd_points = len(sorted_st)

    logger.info("Analytical mapping: %d elements, F_fem=(%.1f, %.1f, %.1f) N",
                result.num_fem_elements, fem_fx, fem_fy, fem_fz)
    return result


# ── Nearest-Neighbour Mapping ────────────────────────────────────────────────

def map_pressures_nearest(
    cfd_points: List[Tuple[float, float, float]],
    cfd_pressures: List[float],
    fem_stations: List[Tuple[int, float, float, float]],
) -> PressureMappingResult:
    """Map CFD pressures to FEM elements via nearest-neighbour lookup.

    Parameters
    ----------
    cfd_points : list of (x, y, z)
        CFD surface point coordinates (m).
    cfd_pressures : list of float
        Pressure at each CFD point (Pa).
    fem_stations : list of (elem_id, x, area, outward_nx)
        FEM element stations.

    Returns
    -------
    PressureMappingResult
    """
    result = PressureMappingResult(mapping_method='nearest')
    element_pressures: list = []

    fem_fx = 0.0
    for elem_id, x_fem, area_m2, nx in fem_stations:
        # Find closest CFD point (Euclidean in x only for axisymmetric bodies)
        best_dist = float('inf')
        best_p = 0.0
        for (xc, yc, zc), pc in zip(cfd_points, cfd_pressures):
            d = abs(xc - x_fem)
            if d < best_dist:
                best_dist = d
                best_p = pc
        element_pressures.append((elem_id, best_p))
        fem_fx += -best_p * area_m2 * nx

    result.element_pressures = element_pressures
    result.num_fem_elements = len(fem_stations)
    result.num_cfd_points = len(cfd_points)
    result.fem_total_force = (fem_fx, 0.0, 0.0)

    # CFD resultant (simple sum)
    if cfd_points and cfd_pressures:
        # approximate — assumes uniform tributary area
        avg_area = 1.0
        if fem_stations:
            avg_area = sum(s[2] for s in fem_stations) / len(fem_stations)
        cfd_fx = sum(-p * avg_area for p in cfd_pressures) / max(len(cfd_pressures), 1) * len(fem_stations)
        result.cfd_total_force = (cfd_fx, 0.0, 0.0)

    logger.info("Nearest mapping: %d CFD pts → %d FEM elements",
                result.num_cfd_points, result.num_fem_elements)
    return result


# ── IDW (Inverse-Distance Weighting) Mapping ────────────────────────────────

def map_pressures_idw(
    cfd_points: List[Tuple[float, float, float]],
    cfd_pressures: List[float],
    fem_stations: List[Tuple[int, float, float, float]],
    power: float = 2.0,
    k_neighbours: int = 8,
) -> PressureMappingResult:
    """Map CFD pressures to FEM elements using Shepard IDW interpolation.

    .. math::
        p(\\mathbf{x}) = \\frac{\\sum_{i=1}^{k} w_i \\, p_i}
                              {\\sum_{i=1}^{k} w_i},
        \\qquad w_i = \\frac{1}{d_i^p}

    Parameters
    ----------
    cfd_points : list of (x, y, z)
    cfd_pressures : list of float
    fem_stations : list of (elem_id, x, area, outward_nx)
    power : float
        Distance weighting exponent (default 2 = inverse-square).
    k_neighbours : int
        Number of nearest neighbours to include (default 8).

    Returns
    -------
    PressureMappingResult
    """
    result = PressureMappingResult(mapping_method='IDW')
    element_pressures: list = []
    fem_fx = 0.0

    for elem_id, x_fem, area_m2, nx in fem_stations:
        # Compute distances to all CFD points
        dists = []
        for idx, (xc, yc, zc) in enumerate(cfd_points):
            d = math.sqrt((xc - x_fem) ** 2 + yc ** 2 + zc ** 2)
            dists.append((d, idx))
        dists.sort(key=lambda t: t[0])

        # Take k nearest
        neighbours = dists[:min(k_neighbours, len(dists))]

        w_sum = 0.0
        wp_sum = 0.0
        for d, idx in neighbours:
            if d < 1e-12:
                # Coincident point — use directly
                wp_sum = cfd_pressures[idx]
                w_sum = 1.0
                break
            w = 1.0 / (d ** power)
            w_sum += w
            wp_sum += w * cfd_pressures[idx]

        p_interp = wp_sum / max(w_sum, 1e-30)
        element_pressures.append((elem_id, p_interp))
        fem_fx += -p_interp * area_m2 * nx

    result.element_pressures = element_pressures
    result.num_fem_elements = len(fem_stations)
    result.num_cfd_points = len(cfd_points)
    result.fem_total_force = (fem_fx, 0.0, 0.0)

    logger.info("IDW mapping (k=%d, p=%.1f): %d CFD pts → %d FEM elements",
                k_neighbours, power, result.num_cfd_points,
                result.num_fem_elements)
    return result


# ── Force / Moment Conservation Verification ────────────────────────────────

def verify_mapping(result: PressureMappingResult,
                   tolerance_pct: float = 2.0) -> PressureMappingResult:
    """Check force and moment conservation between CFD and FEM resultants.

    Computes the relative error:

    .. math::
        \\varepsilon_F = \\frac{\\|\\mathbf{F}_{FEM} - \\mathbf{F}_{CFD}\\|}
                              {\\|\\mathbf{F}_{CFD}\\|} \\times 100

    and similarly for moments.  Sets ``result.accepted = True`` if both
    errors are below *tolerance_pct* (default 2 %).

    Parameters
    ----------
    result : PressureMappingResult
        Result from any ``map_pressures_*`` function.
    tolerance_pct : float
        Acceptance threshold (%).  Default 2 %.

    Returns
    -------
    PressureMappingResult
        Same object, with ``force_error_pct``, ``moment_error_pct``,
        and ``accepted`` updated in-place.
    """
    f_cfd_mag = _vec_mag(result.cfd_total_force)
    m_cfd_mag = _vec_mag(result.cfd_total_moment)

    f_diff = _vec_sub(result.fem_total_force, result.cfd_total_force)
    m_diff = _vec_sub(result.fem_total_moment, result.cfd_total_moment)

    result.force_error_pct = (_vec_mag(f_diff) / max(f_cfd_mag, 1e-30)) * 100.0
    result.moment_error_pct = (_vec_mag(m_diff) / max(m_cfd_mag, 1e-30)) * 100.0

    result.accepted = (result.force_error_pct < tolerance_pct and
                       result.moment_error_pct < tolerance_pct)

    status = "ACCEPTED" if result.accepted else "REJECTED"
    logger.info(
        "Mapping verification %s: ΔF=%.2f%%, ΔM=%.2f%% (tol=%.1f%%)",
        status, result.force_error_pct, result.moment_error_pct, tolerance_pct,
    )
    return result


# ── CalculiX DLOAD Card Generator ────────────────────────────────────────────

def generate_dload_cards(
    result: PressureMappingResult,
    output_path: Optional[Path] = None,
) -> str:
    """Generate CalculiX ``*DLOAD`` cards from mapped element pressures.

    Each element receives a ``P`` (pressure) distributed load:

    .. code-block:: text

        *DLOAD
        <elem_id>, P, <pressure_Pa>

    Parameters
    ----------
    result : PressureMappingResult
        Must contain ``element_pressures``.
    output_path : Path, optional
        If provided, writes the cards to this file.

    Returns
    -------
    str
        The *DLOAD card text.
    """
    lines = ["** K2 AeroSim — Mapped Aerodynamic Pressure Loads"]
    lines.append(f"** Method: {result.mapping_method}")
    lines.append(f"** CFD points: {result.num_cfd_points}  "
                 f"FEM elements: {result.num_fem_elements}")
    lines.append(f"** Force error: {result.force_error_pct:.2f}%  "
                 f"Moment error: {result.moment_error_pct:.2f}%")
    lines.append(f"** Accepted: {result.accepted}")
    lines.append("*DLOAD")

    for elem_id, pressure in result.element_pressures:
        # CalculiX format: element_id, load_type, magnitude
        lines.append(f"{elem_id}, P, {pressure:.6e}")

    card_text = "\n".join(lines) + "\n"

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card_text, encoding='utf-8')
        logger.info("DLOAD cards written to %s (%d elements)",
                     output_path, len(result.element_pressures))

    return card_text


# ── Convenience: Map from VTU File ───────────────────────────────────────────

def map_pressures_from_vtk(
    vtk_path: Path,
    fem_stations: List[Tuple[int, float, float, float]],
    method: str = 'IDW',
    **kwargs,
) -> PressureMappingResult:
    """Load a VTU file and map its pressure field onto FEM stations.

    Attempts to parse simple ASCII VTU XML.  Falls back to nearest
    neighbour if IDW fails.

    Parameters
    ----------
    vtk_path : Path
        Path to a ``.vtu`` file.
    fem_stations : list
        FEM element stations as ``(elem_id, x, area, outward_nx)``.
    method : str
        ``'IDW'`` (default), ``'nearest'``, or ``'analytical'``.
    **kwargs
        Forwarded to the chosen mapping function.

    Returns
    -------
    PressureMappingResult
    """
    vtk_path = Path(vtk_path)
    if not vtk_path.exists():
        logger.error("VTU file not found: %s", vtk_path)
        return PressureMappingResult()

    try:
        cfd_points, cfd_pressures = _parse_vtu_points_and_pressure(vtk_path)
    except Exception as exc:
        logger.error("Failed to parse VTU file %s: %s", vtk_path, exc)
        return PressureMappingResult()

    if not cfd_points or not cfd_pressures:
        logger.warning("VTU file contained no usable pressure data.")
        return PressureMappingResult()

    if method.upper() == 'IDW':
        result = map_pressures_idw(cfd_points, cfd_pressures, fem_stations,
                                   **kwargs)
    elif method.lower() == 'nearest':
        result = map_pressures_nearest(cfd_points, cfd_pressures, fem_stations)
    else:
        logger.warning("Method '%s' not supported for VTU input, using IDW.",
                        method)
        result = map_pressures_idw(cfd_points, cfd_pressures, fem_stations,
                                   **kwargs)

    return verify_mapping(result)
