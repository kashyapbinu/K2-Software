"""
K2 AeroSim — Shock Detection System
========================================
Detects shock waves in supersonic flow fields using multiple sensor
methods: pressure gradient, Ducros sensor, dilatation (divergence),
Mach gradient, and entropy gradient.  Provides shock angle estimation
and pressure ratio computation for aerospace-grade post-processing.

Sensors
-------
* Pressure gradient  — legacy threshold on |∇P|
* Ducros sensor      — Φ = (∇·V)² / ((∇·V)² + |ω|² + ε)
* Dilatation         — θ = ∇·V  (negative = compression)
* Mach gradient      — |∇M|
* Entropy gradient   — |∇(P / ρ^γ)|

All sensors support optional Gaussian pre-filtering via KDTree
averaging before gradient computation.
"""
from __future__ import annotations
import logging
import warnings
import numpy as np
from typing import Optional, Tuple

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:  # pragma: no cover – scipy optional
    KDTree = None

logger = logging.getLogger("K2.CFD.Shock")

# ---------------------------------------------------------------------------
#  Pre-filtering utility
# ---------------------------------------------------------------------------

def gaussian_prefilter(
    mesh,
    scalar_name: str,
    sigma: float = 1.5,
) -> object:
    """Smooth a point-data scalar field using KDTree-based Gaussian averaging.

    Parameters
    ----------
    mesh : pyvista.DataSet
        Volume mesh that contains the scalar to smooth.
    scalar_name : str
        Name of the point-data scalar array to smooth.
    sigma : float, optional
        Standard deviation (in mesh length units) for the Gaussian kernel.
        Default is 1.5.

    Returns
    -------
    pyvista.DataSet
        The *same* mesh object with the scalar field replaced by the
        smoothed version (in-place modification).

    Raises
    ------
    RuntimeError
        If scipy is not installed.
    """
    if KDTree is None:
        raise RuntimeError(
            "scipy is required for gaussian_prefilter but is not installed."
        )

    points = np.asarray(mesh.points, dtype=np.float64)
    values = np.asarray(mesh[scalar_name], dtype=np.float64)
    n_pts = len(points)
    if n_pts < 4:
        return mesh

    # Vectorised k-nearest-neighbour Gaussian kernel. The previous
    # implementation looped query_ball_point() over every point in Python,
    # which took minutes on ~1M-node volume meshes and froze the UI.
    k = min(16, n_pts - 1)
    tree = KDTree(points)
    dists, idxs = tree.query(points, k=k + 1)   # includes self at column 0
    weights = np.exp(-0.5 * (dists / max(sigma, 1e-12)) ** 2)
    smoothed = (weights * values[idxs]).sum(axis=1) / weights.sum(axis=1)

    mesh[scalar_name] = smoothed.astype(np.float32)
    logger.debug(
        "Gaussian pre-filter applied to '%s' (σ=%.2f, k=%d)",
        scalar_name, sigma, k,
    )
    return mesh


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _find_gradient_name(mesh, base: str) -> str:
    """Return the actual gradient array name that PyVista created.

    Only accept exact/prefixed 'gradient*' names. A substring match would
    wrongly pick up sensor OUTPUT arrays carried over from a previous run
    (e.g. 'Mach_Gradient', 'Entropy_Gradient' — 1-D scalars, not the fresh
    (N,3) gradient), crashing the axis-1 norm.
    """
    preferred = f"gradient_{base}"
    if preferred in mesh.array_names:
        return preferred
    if "gradient" in mesh.array_names:
        return "gradient"
    for name in mesh.array_names:
        if name.lower().startswith("gradient"):
            return name
    return "gradient"


def _extract_iso(
    mesh,
    scalar_name: str,
    percentile: float,
    volume_mesh,
) -> Optional[object]:
    """Threshold *scalar_name* at *percentile* and return an iso-surface."""
    values = np.asarray(mesh[scalar_name], dtype=np.float32)
    valid = values[values > 0]
    if len(valid) == 0:
        logger.info("No positive values in '%s' — no shocks.", scalar_name)
        return None

    threshold = float(np.percentile(valid, percentile))
    iso = mesh.contour(isosurfaces=[threshold], scalars=scalar_name)
    if iso.n_cells == 0:
        logger.info("No shock iso-surface produced for '%s'.", scalar_name)
        return None

    # Transfer useful scalars for visualisation
    for scalar in ("Mach", "Pressure", "Mach_Number"):
        if scalar in volume_mesh.array_names:
            sampled = iso.sample(volume_mesh)
            if scalar in sampled.array_names:
                iso[scalar] = sampled[scalar]
            break

    logger.info(
        "Shock surface detected via '%s': %d cells at threshold=%.4g",
        scalar_name, iso.n_cells, threshold,
    )
    return iso


# ---------------------------------------------------------------------------
#  Sensor functions
# ---------------------------------------------------------------------------

def ducros_shock_sensor(
    volume_mesh,
    prefilter: bool = True,
    sigma: float = 1.5,
) -> object:
    """Compute the Ducros shock sensor on a volume mesh.

    The Ducros sensor cleanly separates shocks from vortices:

        Φ = (∇·V)² / ((∇·V)² + |ω|² + ε)

    * Φ → 1 in shock regions (irrotational compression)
    * Φ → 0 in vortical regions

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing a 3-component ``'Velocity'`` vector field.
    prefilter : bool, optional
        If *True* (default), apply Gaussian pre-filtering to the
        velocity magnitude before derivative computation.
    sigma : float, optional
        Standard deviation for the Gaussian kernel (default 1.5).

    Returns
    -------
    pyvista.UnstructuredGrid
        The input mesh with a ``'Ducros_Sensor'`` point-data array added.
    """
    import pyvista as pv

    if "Velocity" not in volume_mesh.array_names:
        raise KeyError("Volume mesh must contain a 'Velocity' vector field.")

    # Smooth a TEMPORARY copy — never overwrite the cached 'Velocity' array
    # (the workspace reuses the same volume mesh for slice/streamline views).
    vel_field = "Velocity"
    if prefilter and KDTree is not None:
        vmag = np.linalg.norm(
            np.asarray(volume_mesh["Velocity"]), axis=1
        ).astype(np.float32)
        volume_mesh["_VelMag_tmp"] = vmag
        gaussian_prefilter(volume_mesh, "_VelMag_tmp", sigma=sigma)
        # Scale velocity components proportionally
        raw_mag = np.linalg.norm(
            np.asarray(volume_mesh["Velocity"]), axis=1, keepdims=True
        )
        raw_mag = np.where(raw_mag < 1e-30, 1.0, raw_mag)
        scale = (volume_mesh["_VelMag_tmp"] / raw_mag.ravel()).astype(np.float32)
        vel_smooth = np.asarray(volume_mesh["Velocity"]) * scale[:, None]
        volume_mesh["_Vel_tmp"] = vel_smooth.astype(np.float32)
        del volume_mesh.point_data["_VelMag_tmp"]
        vel_field = "_Vel_tmp"

    # Compute divergence (∇·V) and vorticity (ω)
    deriv = volume_mesh.compute_derivative(
        scalars=vel_field,
        divergence=True,
        vorticity=True,
    )
    if "_Vel_tmp" in volume_mesh.point_data:
        del volume_mesh.point_data["_Vel_tmp"]

    div_V = np.asarray(deriv["divergence"], dtype=np.float64)
    omega = np.asarray(deriv["vorticity"], dtype=np.float64)
    omega_mag = np.linalg.norm(omega, axis=1)

    epsilon = 1e-20
    phi = (div_V ** 2) / (div_V ** 2 + omega_mag ** 2 + epsilon)
    phi = phi.astype(np.float32)

    volume_mesh["Ducros_Sensor"] = phi
    logger.info(
        "Ducros sensor computed — Φ range [%.4f, %.4f]",
        float(phi.min()), float(phi.max()),
    )
    return volume_mesh


def dilatation_shock_sensor(
    volume_mesh,
    prefilter: bool = True,
    sigma: float = 1.5,
) -> object:
    """Compute the dilatation (divergence) shock sensor.

    θ = ∇·V

    Strong negative dilatation indicates compressive shocks. The
    returned array is clipped so that only negative values are
    retained (positive dilatation is set to zero).

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing a 3-component ``'Velocity'`` vector field.
    prefilter : bool, optional
        Apply Gaussian pre-filtering (default *True*).
    sigma : float, optional
        Standard deviation for the Gaussian kernel (default 1.5).

    Returns
    -------
    pyvista.UnstructuredGrid
        The input mesh with a ``'Dilatation'`` point-data array added.
        Values are ``|θ|`` for θ < 0, and 0 otherwise.
    """
    if "Velocity" not in volume_mesh.array_names:
        raise KeyError("Volume mesh must contain a 'Velocity' vector field.")

    # Smooth a TEMPORARY copy — never overwrite the cached 'Velocity' array.
    vel_field = "Velocity"
    if prefilter and KDTree is not None:
        vmag = np.linalg.norm(
            np.asarray(volume_mesh["Velocity"]), axis=1
        ).astype(np.float32)
        volume_mesh["_VelMag_tmp"] = vmag
        gaussian_prefilter(volume_mesh, "_VelMag_tmp", sigma=sigma)
        raw_mag = np.linalg.norm(
            np.asarray(volume_mesh["Velocity"]), axis=1, keepdims=True
        )
        raw_mag = np.where(raw_mag < 1e-30, 1.0, raw_mag)
        scale = (volume_mesh["_VelMag_tmp"] / raw_mag.ravel()).astype(np.float32)
        vel_smooth = np.asarray(volume_mesh["Velocity"]) * scale[:, None]
        volume_mesh["_Vel_tmp"] = vel_smooth.astype(np.float32)
        del volume_mesh.point_data["_VelMag_tmp"]
        vel_field = "_Vel_tmp"

    deriv = volume_mesh.compute_derivative(
        scalars=vel_field,
        divergence=True,
    )
    if "_Vel_tmp" in volume_mesh.point_data:
        del volume_mesh.point_data["_Vel_tmp"]

    div_V = np.asarray(deriv["divergence"], dtype=np.float64)

    # Keep only compressive (negative) dilatation; flip sign so the
    # scalar is positive where shocks exist.
    dilatation = np.where(div_V < 0, np.abs(div_V), 0.0).astype(np.float32)

    volume_mesh["Dilatation"] = dilatation
    logger.info(
        "Dilatation sensor computed — max |θ_neg| = %.4g",
        float(dilatation.max()),
    )
    return volume_mesh


def mach_gradient_shock_sensor(
    volume_mesh,
    prefilter: bool = True,
    sigma: float = 1.5,
) -> object:
    """Compute the Mach-gradient shock sensor.

    |∇M| — the magnitude of the gradient of the Mach number field.
    Large values indicate abrupt Mach transitions (shocks).

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing a ``'Mach'`` scalar field.
    prefilter : bool, optional
        Apply Gaussian pre-filtering to the Mach field before computing
        the gradient (default *True*).
    sigma : float, optional
        Standard deviation for the Gaussian kernel (default 1.5).

    Returns
    -------
    pyvista.UnstructuredGrid
        The input mesh with a ``'Mach_Gradient'`` point-data array added.
    """
    if "Mach" not in volume_mesh.array_names:
        raise KeyError("Volume mesh must contain a 'Mach' scalar field.")

    # Smooth a TEMPORARY copy — never overwrite the cached 'Mach' array
    # (the Mach slice view reads it after shock detection runs).
    mach_field = "Mach"
    if prefilter and KDTree is not None:
        volume_mesh["_Mach_tmp"] = np.asarray(
            volume_mesh["Mach"], dtype=np.float32
        ).copy()
        gaussian_prefilter(volume_mesh, "_Mach_tmp", sigma=sigma)
        mach_field = "_Mach_tmp"

    grad = volume_mesh.compute_derivative(scalars=mach_field)
    grad_name = _find_gradient_name(grad, mach_field)
    if "_Mach_tmp" in volume_mesh.point_data:
        del volume_mesh.point_data["_Mach_tmp"]
    grad_vec = np.asarray(grad[grad_name], dtype=np.float64)
    grad_mag = np.linalg.norm(grad_vec, axis=1).astype(np.float32)

    volume_mesh["Mach_Gradient"] = grad_mag
    logger.info(
        "Mach gradient sensor computed — max |∇M| = %.4g",
        float(grad_mag.max()),
    )
    return volume_mesh


def entropy_gradient_shock_sensor(
    volume_mesh,
    gamma: float = 1.4,
    prefilter: bool = True,
    sigma: float = 1.5,
) -> object:
    """Compute the entropy-gradient shock sensor.

    Entropy proxy:  s = P / ρ^γ

    Across a shock, entropy increases; therefore |∇s| spikes at
    the shock location.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing ``'Pressure'`` and ``'Density'`` scalar
        fields.
    gamma : float, optional
        Ratio of specific heats (default 1.4 for air).
    prefilter : bool, optional
        Apply Gaussian pre-filtering to the entropy field before
        computing the gradient (default *True*).
    sigma : float, optional
        Standard deviation for the Gaussian kernel (default 1.5).

    Returns
    -------
    pyvista.UnstructuredGrid
        The input mesh with an ``'Entropy_Gradient'`` point-data array
        added.
    """
    if "Pressure" not in volume_mesh.array_names:
        raise KeyError("Volume mesh must contain a 'Pressure' scalar field.")
    if "Density" not in volume_mesh.array_names:
        raise KeyError("Volume mesh must contain a 'Density' scalar field.")

    pressure = np.asarray(volume_mesh["Pressure"], dtype=np.float64)
    density = np.asarray(volume_mesh["Density"], dtype=np.float64)
    density = np.where(density < 1e-30, 1e-30, density)  # guard against zero

    entropy = (pressure / density ** gamma).astype(np.float32)
    volume_mesh["_Entropy_tmp"] = entropy

    if prefilter and KDTree is not None:
        gaussian_prefilter(volume_mesh, "_Entropy_tmp", sigma=sigma)

    grad = volume_mesh.compute_derivative(scalars="_Entropy_tmp")
    grad_name = _find_gradient_name(grad, "_Entropy_tmp")
    grad_vec = np.asarray(grad[grad_name], dtype=np.float64)
    grad_mag = np.linalg.norm(grad_vec, axis=1).astype(np.float32)

    volume_mesh["Entropy_Gradient"] = grad_mag
    # Clean up temporary array
    if "_Entropy_tmp" in volume_mesh.point_data:
        del volume_mesh.point_data["_Entropy_tmp"]

    logger.info(
        "Entropy gradient sensor computed — max |∇s| = %.4g",
        float(grad_mag.max()),
    )
    return volume_mesh


# ---------------------------------------------------------------------------
#  Legacy pressure-gradient detector (preserved, updated)
# ---------------------------------------------------------------------------

def detect_shock_surfaces(
    volume_mesh,
    pressure_field: str = "Pressure",
    gradient_percentile: float = 95.0,
    method: str = "pressure_gradient",
) -> Optional[object]:
    """Detect shock surfaces using pressure gradient magnitude.

    .. deprecated::
        This function is retained for backward compatibility.  New code
        should use :func:`detect_shocks` which supports multiple sensor
        methods and Gaussian pre-filtering.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing at least the *pressure_field* scalar.
    pressure_field : str, optional
        Name of the pressure scalar array (default ``'Pressure'``).
    gradient_percentile : float, optional
        Percentile of |∇P| used as the iso-surface threshold (default
        95.0).
    method : str, optional
        Sensor method to use.  Accepts any method supported by
        :func:`detect_shocks`.  Defaults to ``'pressure_gradient'``
        for backward compatibility.

    Returns
    -------
    pyvista.PolyData or None
        Iso-surface mesh of the shock region, or *None* if no shocks
        are detected.
    """
    warnings.warn(
        "detect_shock_surfaces() is deprecated — use detect_shocks() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # If caller explicitly chose a non-legacy method, delegate directly
    if method != "pressure_gradient":
        return detect_shocks(
            volume_mesh,
            method=method,
            percentile=gradient_percentile,
            prefilter=True,
        )

    import pyvista as pv

    if volume_mesh is None or pressure_field not in volume_mesh.array_names:
        logger.warning(f"'{pressure_field}' not found in volume mesh.")
        return None

    # Shocks only exist in transonic/supersonic flow
    if "Mach" in volume_mesh.array_names:
        max_mach = float(np.max(volume_mesh["Mach"]))
        if max_mach < 0.95:
            logger.info(f"Max Mach ({max_mach:.2f}) < 0.95. No physical shocks possible.")
            return None

    try:
        # Compute pressure gradient
        grad = volume_mesh.compute_derivative(scalars=pressure_field)
        grad_name = _find_gradient_name(grad, pressure_field)

        grad_vec = grad[grad_name]
        grad_mag = np.linalg.norm(grad_vec, axis=1).astype(np.float32)
        grad["PressureGradientMag"] = grad_mag

        # Threshold: use the given percentile of gradient magnitude
        valid = grad_mag[grad_mag > 0]
        if len(valid) == 0:
            logger.info("No pressure gradient detected — no shocks.")
            return None

        threshold = float(np.percentile(valid, gradient_percentile))

        # Extract iso-surface at threshold
        iso = grad.contour(isosurfaces=[threshold], scalars="PressureGradientMag")
        if iso.n_cells == 0:
            logger.info("No shock surfaces found at this threshold.")
            return None

        # Transfer Mach and Pressure to iso-surface for coloring
        for scalar in ["Mach", "Pressure", "Mach_Number"]:
            if scalar in volume_mesh.array_names:
                sampled = iso.sample(volume_mesh)
                if scalar in sampled.array_names:
                    iso[scalar] = sampled[scalar]
                break

        logger.info(f"Shock surface detected: {iso.n_cells} cells at |∇P| = {threshold:.0f}")
        return iso

    except Exception as e:
        logger.error(f"Shock detection failed: {e}")
        return None


# ---------------------------------------------------------------------------
#  Shock angle & pressure ratio (unchanged)
# ---------------------------------------------------------------------------

def estimate_shock_angle(
    shock_surface,
    freestream_direction: np.ndarray = None,
) -> float:
    """Estimate the shock angle relative to the freestream direction.

    Returns angle in degrees. The shock angle is the average angle between
    the shock surface normals and the freestream vector.
    """
    if shock_surface is None or shock_surface.n_cells == 0:
        return 0.0

    if freestream_direction is None:
        freestream_direction = np.array([1.0, 0.0, 0.0])  # +X = flow direction

    freestream_direction = freestream_direction / np.linalg.norm(freestream_direction)

    try:
        normals = shock_surface.compute_normals(cell_normals=True, point_normals=False)
        cell_normals = normals["Normals"]
        # Shock angle = 90° - angle between normal and freestream
        dots = np.abs(np.dot(cell_normals, freestream_direction))
        dots = np.clip(dots, 0, 1)
        angles = np.degrees(np.arccos(dots))
        # Shock wave angle is complement
        shock_angles = 90.0 - angles
        mean_angle = float(np.mean(shock_angles))
        logger.info(f"Estimated shock angle: {mean_angle:.1f}°")
        return mean_angle
    except Exception as e:
        logger.error(f"Shock angle estimation failed: {e}")
        return 0.0


def compute_pressure_ratio(
    volume_mesh,
    shock_surface,
    pressure_field: str = "Pressure",
    sample_distance: float = 0.005,
) -> Tuple[float, float, float]:
    """Estimate pressure ratio across a shock surface.

    Returns (P_upstream, P_downstream, ratio).
    """
    if shock_surface is None or volume_mesh is None:
        return 0.0, 0.0, 1.0

    if pressure_field not in volume_mesh.array_names:
        return 0.0, 0.0, 1.0

    try:
        import pyvista as pv

        normals = shock_surface.compute_normals(cell_normals=True)
        centers = np.asarray(shock_surface.cell_centers().points, dtype=np.float64)
        cell_normals = np.asarray(normals["Normals"], dtype=np.float64)

        # Sample a subset of shock cells
        n_sample = min(100, len(centers))
        if n_sample == 0:
            return 0.0, 0.0, 1.0
        indices = np.random.choice(len(centers), n_sample, replace=False)

        # Build upstream/downstream probe points a small step along each normal
        pts_up = centers[indices] - cell_normals[indices] * sample_distance
        pts_down = centers[indices] + cell_normals[indices] * sample_distance

        # Probe the volume field at both sets of points
        p_up = np.asarray(pv.PolyData(pts_up).sample(volume_mesh)[pressure_field],
                          dtype=np.float64)
        p_down = np.asarray(pv.PolyData(pts_down).sample(volume_mesh)[pressure_field],
                            dtype=np.float64)

        # Orient each pair so 'up' is the low-pressure (pre-shock) side
        lo = np.minimum(p_up, p_down)
        hi = np.maximum(p_up, p_down)
        valid = lo > 0
        if not np.any(valid):
            return 0.0, 0.0, 1.0

        p_upstream = float(np.mean(lo[valid]))
        p_downstream = float(np.mean(hi[valid]))
        ratio = p_downstream / p_upstream if p_upstream > 0 else 1.0
        logger.info(f"Shock pressure ratio p2/p1 = {ratio:.3f} "
                    f"(p1={p_upstream:.0f} Pa, p2={p_downstream:.0f} Pa)")
        return p_upstream, p_downstream, ratio
    except Exception as e:
        logger.error(f"Pressure ratio computation failed: {e}")
        return 0.0, 0.0, 1.0


# ---------------------------------------------------------------------------
#  Unified sensor interface
# ---------------------------------------------------------------------------

_SENSOR_METHODS = {
    "ducros":           ("Ducros_Sensor",    ducros_shock_sensor),
    "dilatation":       ("Dilatation",       dilatation_shock_sensor),
    "mach_gradient":    ("Mach_Gradient",    mach_gradient_shock_sensor),
    "entropy_gradient": ("Entropy_Gradient", entropy_gradient_shock_sensor),
}


def detect_shocks(
    volume_mesh,
    method: str = "ducros",
    percentile: float = 90.0,
    prefilter: bool = True,
    sigma: float = 1.5,
    gamma: float = 1.4,
) -> Optional[object]:
    """Unified shock detection interface.

    Dispatches to the appropriate sensor function, computes the scalar
    field, and returns an iso-surface mesh at the requested percentile
    threshold.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh with the required flow-field arrays for the chosen
        method.
    method : str, optional
        Sensor method.  One of ``'ducros'``, ``'dilatation'``,
        ``'mach_gradient'``, ``'entropy_gradient'``, or
        ``'pressure_gradient'`` (legacy).  Default is ``'ducros'``.
    percentile : float, optional
        Percentile of the sensor scalar used as the iso-surface
        threshold (default 90.0).
    prefilter : bool, optional
        Apply Gaussian pre-filtering before gradient computation
        (default *True*).
    sigma : float, optional
        Standard deviation for the Gaussian kernel (default 1.5).
    gamma : float, optional
        Ratio of specific heats, used only by the entropy-gradient
        method (default 1.4).

    Returns
    -------
    pyvista.PolyData or None
        Iso-surface mesh of the shock region, or *None* if no shocks
        are detected.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    if volume_mesh is None:
        logger.warning("detect_shocks() received None volume mesh.")
        return None

    # ---- Legacy pressure-gradient path ----
    if method == "pressure_gradient":
        return detect_shock_surfaces.__wrapped__(
            volume_mesh,
            pressure_field="Pressure",
            gradient_percentile=percentile,
        ) if hasattr(detect_shock_surfaces, "__wrapped__") else _pressure_gradient_path(
            volume_mesh, percentile
        )

    # ---- New sensor methods ----
    if method not in _SENSOR_METHODS:
        raise ValueError(
            f"Unknown shock detection method '{method}'. "
            f"Choose from: {', '.join(list(_SENSOR_METHODS) + ['pressure_gradient'])}"
        )

    scalar_name, sensor_fn = _SENSOR_METHODS[method]

    try:
        kwargs = {"prefilter": prefilter, "sigma": sigma}
        if method == "entropy_gradient":
            kwargs["gamma"] = gamma
        sensor_fn(volume_mesh, **kwargs)
    except Exception as e:
        logger.error("Sensor '%s' failed: %s", method, e)
        return None

    return _extract_iso(volume_mesh, scalar_name, percentile, volume_mesh)


def _pressure_gradient_path(
    volume_mesh,
    percentile: float,
) -> Optional[object]:
    """Internal helper — pressure-gradient iso-surface without deprecation warning."""
    pressure_field = "Pressure"
    if pressure_field not in volume_mesh.array_names:
        logger.warning("'%s' not found in volume mesh.", pressure_field)
        return None

    if "Mach" in volume_mesh.array_names:
        max_mach = float(np.max(volume_mesh["Mach"]))
        if max_mach < 0.95:
            logger.info(
                "Max Mach (%.2f) < 0.95. No physical shocks possible.", max_mach
            )
            return None

    try:
        grad = volume_mesh.compute_derivative(scalars=pressure_field)
        grad_name = _find_gradient_name(grad, pressure_field)
        grad_vec = grad[grad_name]
        grad_mag = np.linalg.norm(grad_vec, axis=1).astype(np.float32)
        grad["PressureGradientMag"] = grad_mag

        return _extract_iso(grad, "PressureGradientMag", percentile, volume_mesh)
    except Exception as e:
        logger.error("Pressure-gradient shock detection failed: %s", e)
        return None
