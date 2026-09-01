"""
K2 AeroSim — CFD Post-Processing
=====================================
Loads SU2 VTK outputs into PyVista for visualization inside the CFD Workspace.
Provides Cp distribution extraction, mesh statistics, force vector computation,
and utilities to inject CFD-derived coefficients into the simulation engine.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING

import numpy as np
import pyvista as pv

# ── Optional scipy imports with graceful fallback ────────────────────────────
try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

if TYPE_CHECKING:
    from cfd.solvers.base import CFDResult

logger = logging.getLogger("K2.CFD.PostProcess")


# ── Mesh-density-independent field statistics ────────────────────────────────

def field_percentiles(
    mesh: pv.DataSet,
    name: str,
    q,
    weighted: bool = True,
) -> np.ndarray:
    """Percentiles of a mesh scalar field, weighted by cell size.

    ``np.percentile`` treats every mesh point as one equally-important
    sample. CFD meshes are graded, so that is a lie: on a typical rocket
    surface mesh the cell areas span 3e5 : 1, with the smallest slivers
    clustered on the nose tip. The tip therefore contributes thousands of
    times more samples per unit of physical area than the body does, and
    the reported percentiles describe the tip rather than the model. On a
    measured M=0.8 case the point-based 95th percentile of Cp was 0.907
    while the area-weighted one was 0.057 — a 16x error, and the reason
    auto-scaled colour bars rendered the whole airframe flat.

    Weighting each cell by its own area (surfaces) or volume (3D grids)
    removes the bias: the returned values describe how much of the *model*
    sits below each level. Falls back to the plain unweighted percentile
    when cell sizes are unavailable or the field is not a scalar.

    Parameters
    ----------
    mesh : pv.DataSet
        Surface or volume mesh holding the field.
    name : str
        Scalar array to summarise. Vector arrays are rejected.
    q : float or sequence of float
        Percentile(s) in [0, 100].
    weighted : bool, optional
        Set False to force the plain ``np.percentile`` behaviour.

    Returns
    -------
    np.ndarray
        One value per requested percentile; ``nan`` if the field is absent
        or holds no finite values.
    """
    q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if mesh is None or name not in mesh.array_names:
        return np.full(q_arr.shape, np.nan)

    def _plain() -> np.ndarray:
        try:
            raw = np.asarray(mesh[name], dtype=np.float64)
        except Exception:
            return np.full(q_arr.shape, np.nan)
        if raw.ndim > 1:
            return np.full(q_arr.shape, np.nan)
        raw = raw[np.isfinite(raw)]
        if raw.size == 0:
            return np.full(q_arr.shape, np.nan)
        return np.asarray(np.percentile(raw, q_arr), dtype=np.float64)

    if not weighted:
        return _plain()

    try:
        # Align values with cells so each weight has exactly one value.
        cell_mesh = mesh
        if name in mesh.point_data:
            cell_mesh = mesh.point_data_to_cell_data()
        vals = np.asarray(cell_mesh.cell_data[name], dtype=np.float64)
        if vals.ndim > 1:
            return _plain()

        sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
        weights = np.abs(np.asarray(sized.cell_data["Volume"], dtype=np.float64))
        if not np.any(weights > 0):
            weights = np.abs(np.asarray(sized.cell_data["Area"], dtype=np.float64))
        if weights.shape != vals.shape or not np.any(weights > 0):
            return _plain()

        ok = np.isfinite(vals) & np.isfinite(weights) & (weights > 0)
        if not ok.any():
            return _plain()
        vals, weights = vals[ok], weights[ok]

        order = np.argsort(vals)
        vals, weights = vals[order], weights[order]
        # Cumulative area at the CENTRE of each cell's weight band, so a
        # single dominant cell cannot bias the result toward its own edge.
        cum = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
        return np.interp(q_arr / 100.0, cum, vals)
    except Exception as exc:
        logger.debug("field_percentiles fell back to unweighted for '%s': %s", name, exc)
        return _plain()


# ── VTK / Flow field loaders ─────────────────────────────────────────────────

def load_volume_flow(vtk_path: Path) -> Optional[pv.UnstructuredGrid]:
    """Load SU2 volume flow field (flow.vtu) for volumetric rendering."""
    vtk_path = Path(vtk_path)
    if not vtk_path.is_file():
        logger.warning(f"Volume VTK not found: {vtk_path}")
        return None
    try:
        mesh = pv.read(str(vtk_path))
        logger.info(f"Volume flow loaded: {mesh.n_cells} cells, arrays={mesh.array_names}")
        return mesh
    except Exception as e:
        logger.error(f"Failed to load volume VTK: {e}")
        return None


def load_surface_flow(vtk_path: Path) -> Optional[pv.PolyData]:
    """Load SU2 surface flow (surface_flow.vtu) for wall quantity rendering."""
    vtk_path = Path(vtk_path)
    if not vtk_path.is_file():
        logger.warning(f"Surface VTK not found: {vtk_path}")
        return None
    try:
        mesh = pv.read(str(vtk_path))
        logger.info(f"Surface flow loaded: {mesh.n_cells} cells, arrays={mesh.array_names}")
        return mesh
    except Exception as e:
        logger.error(f"Failed to load surface VTK: {e}")
        return None


# ── Cp distribution extraction ────────────────────────────────────────────────

def extract_cp_distribution(
    surface_mesh,
    axis: str = "x",
    n_stations: int = 100,
    freestream_pressure: float = None,
    dynamic_pressure: float = None,
    side: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Circumferentially-averaged Cp along the body axis, weighted by cell area.

    Three things were wrong with the previous version and each of them changed
    the curve rather than the noise on it:

    * It averaged POINTS (``np.mean(cp_vals[mask])``). Wall mesh density follows
      the geometry, not the flow: on a measured surface mesh the cell areas span
      3.3e4 : 1 and the edge ratio is 183 : 1, with the slivers packed onto the
      nose tip and the fin edges. One sample per point therefore describes the
      tip, not the airframe -- the same bias :func:`field_percentiles` exists to
      remove for the colour bars (point 95th percentile of Cp 0.936, area
      weighted 0.046). Cells are now weighted by their own area.
    * It averaged over EVERY point at a station, fins included, so at the fin
      station the "body Cp" was a blend of body skin and fin panel.
    * At incidence it averaged the windward and leeward sides together, which
      is where the loading is -- the two sides cancel and the curve reports a
      body carrying no normal force. ``side`` now selects one of them.

    Parameters
    ----------
    surface_mesh : PyVista surface mesh with Cp or Pressure arrays
    axis : str - rocket axis direction ("x" for CFD frame)
    n_stations : int - number of sampling stations along the body
    freestream_pressure : float - P_inf for computing Cp from Pressure
    dynamic_pressure : float - q_inf for computing Cp from Pressure
    side : "mean" | "windward" | "leeward"
        Which half of the circumference to average. AoA rotates the freestream
        in the x-z plane (see the SU2 config), so the split is by the sign of z:
        windward is z < 0, the side the flow runs into at positive alpha.

    Returns
    -------
    (x_normalized, cp_values) : normalized position [0,1] and Cp values.
    Empty arrays when there is no usable Cp field.
    """
    if surface_mesh is None:
        return np.array([]), np.array([])

    # Find Cp array
    cp_name = None
    for name in ["Pressure_Coefficient", "CpTotal", "Cp"]:
        if name in surface_mesh.array_names:
            cp_name = name
            break

    # Compute Cp from pressure if not available
    if cp_name is None and "Pressure" in surface_mesh.array_names:
        if freestream_pressure is not None and dynamic_pressure is not None and dynamic_pressure > 0:
            P = surface_mesh["Pressure"]
            surface_mesh["Cp_computed"] = (P - freestream_pressure) / dynamic_pressure
            cp_name = "Cp_computed"

    if cp_name is None:
        logger.warning("No Cp data available for distribution extraction.")
        return np.array([]), np.array([])

    ax_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 0)

    # Work on CELLS, so every sample carries an area. Falls back to the old
    # point average only if the mesh cannot produce cell data or areas.
    try:
        surf = surface_mesh.extract_surface() \
            if hasattr(surface_mesh, "extract_surface") else surface_mesh
        cell = surf.point_data_to_cell_data() \
            if cp_name in surf.point_data else surf
        cp_vals = np.asarray(cell.cell_data[cp_name], dtype=float)
        centers = np.asarray(surf.cell_centers().points, dtype=float)
        weights = np.abs(np.asarray(
            surf.compute_cell_sizes(length=False, area=True,
                                    volume=False)["Area"], dtype=float))
        if (weights.shape != cp_vals.shape or centers.shape[0] != cp_vals.shape[0]
                or not np.any(weights > 0)):
            raise ValueError("cell areas unusable")
    except Exception as exc:                                  # noqa: BLE001
        logger.debug(f"Cp distribution fell back to point sampling: {exc}")
        centers = np.asarray(surface_mesh.points, dtype=float)
        cp_vals = np.asarray(surface_mesh[cp_name], dtype=float).ravel()
        weights = np.ones_like(cp_vals)

    ok = np.isfinite(cp_vals) & np.isfinite(weights) & (weights > 0)
    centers, cp_vals, weights = centers[ok], cp_vals[ok], weights[ok]
    if cp_vals.size == 0:
        return np.array([]), np.array([])

    # Keep the body, drop the fins. The body is a surface of revolution about
    # the flow axis, so its cells sit at a radius that varies smoothly with x
    # while a fin panel reaches far outside it at one station. Anything beyond
    # the 75th percentile radius AT ITS OWN STATION is treated as a fin.
    lat = np.delete(np.arange(3), ax_idx)
    radius = np.hypot(centers[:, lat[0]], centers[:, lat[1]])

    if side in ("windward", "leeward"):
        # AoA tilts the freestream in x-z, so z separates the two sides.
        z = centers[:, 2]
        sel = z < 0 if side == "windward" else z > 0
        if sel.sum() >= 8:
            centers, cp_vals, weights, radius = (
                centers[sel], cp_vals[sel], weights[sel], radius[sel])
        else:
            logger.debug(f"Cp distribution: too few cells on the {side} side; "
                         f"reporting the full circumference instead.")

    x_coords = centers[:, ax_idx]
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    x_range = x_max - x_min
    if x_range < 1e-6:
        return np.array([]), np.array([])

    stations = np.linspace(x_min, x_max, n_stations + 1)
    idx = np.clip(np.searchsorted(stations, x_coords, side="right") - 1,
                  0, n_stations - 1)

    x_norm = ((stations[:-1] + stations[1:]) * 0.5 - x_min) / x_range
    cp_avg = np.zeros(n_stations)
    last = 0.0
    for i in range(n_stations):
        m = idx == i
        if not np.any(m):
            cp_avg[i] = last                       # carry forward across gaps
            continue
        r, w, c = radius[m], weights[m], cp_vals[m]
        body = r <= np.percentile(r, 75) * 1.05
        if body.sum() >= 3:
            w, c = w[body], c[body]
        cp_avg[i] = last = float(np.sum(w * c) / np.sum(w))

    return x_norm, cp_avg


# ── Gaussian smoothing utilities ─────────────────────────────────────────────

def gaussian_smooth_surface(
    mesh: pv.DataSet,
    name: str,
    sigma: float = 1.5,
    n_iter: int = 2,
) -> pv.DataSet:
    """
    Apply Gaussian-weighted smoothing to a surface scalar field.

    Uses scipy's cKDTree to find k-nearest neighbours and applies
    Gaussian-weighted averaging.  More physically accurate than the
    existing Laplacian approach because it preserves stagnation peaks
    while still removing solver-level noise.

    Parameters
    ----------
    mesh : pv.DataSet
        Surface mesh containing the scalar to smooth.
    name : str
        Name of the point-data scalar array to smooth.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (in mesh length
        units).  Default 1.5.
    n_iter : int, optional
        Number of smoothing iterations.  Default 2.

    Returns
    -------
    pv.DataSet
        The *same* mesh object with the scalar array replaced by its
        smoothed version.  The original values are stored under
        ``{name}_raw``.
    """
    if mesh is None or name not in mesh.array_names:
        logger.warning(f"gaussian_smooth_surface: '{name}' not found in mesh.")
        return mesh

    if not _HAS_SCIPY:
        logger.warning("scipy not available — skipping Gaussian smoothing.")
        return mesh

    try:
        pts = mesh.points
        n_pts = len(pts)
        if n_pts < 4:
            return mesh

        k = min(20, n_pts - 1)
        tree = cKDTree(pts)
        dists, idxs = tree.query(pts, k=k + 1)  # includes self at index 0

        field = np.asarray(mesh[name], dtype=np.float64).copy()

        # Preserve the raw field before smoothing
        mesh[f"{name}_raw"] = field.astype(np.float32)

        for _ in range(n_iter):
            smoothed = np.zeros_like(field)
            weight_sum = np.zeros(n_pts, dtype=np.float64)
            for j in range(k + 1):
                d = dists[:, j]
                w = np.exp(-0.5 * (d / max(sigma, 1e-12)) ** 2)
                smoothed += w * field[idxs[:, j]]
                weight_sum += w
            weight_sum = np.where(weight_sum < 1e-30, 1.0, weight_sum)
            field = smoothed / weight_sum

        mesh[name] = field.astype(np.float32)
        logger.info(
            f"Gaussian-smoothed '{name}' (σ={sigma}, {n_iter} iters, k={k})"
        )

    except Exception as e:
        logger.error(f"gaussian_smooth_surface failed: {e}")

    return mesh


def smooth_volume_field(
    mesh: pv.DataSet,
    scalar_name: str,
    sigma: float = 0.6,
    k: int = 12,
) -> pv.DataSet:
    """
    Apply Gaussian kernel smoothing on a volume mesh scalar field.

    Intended for pre-smoothing noisy fields (Q-criterion, Lambda-2)
    before iso-surface extraction so that the resulting contours are
    clean and free of solver-level artefacts.

    Parameters
    ----------
    mesh : pv.DataSet
        Volume mesh (UnstructuredGrid or PolyData) containing the scalar.
    scalar_name : str
        Name of the point-data scalar array to smooth.
    sigma : float, optional
        Kernel width as a MULTIPLE OF LOCAL MESH SPACING (default 0.6),
        not an absolute distance — see the note below.
    k : int, optional
        Number of nearest neighbours used for the kernel.  Default 12.

    Returns
    -------
    pv.DataSet
        The *same* mesh with the scalar replaced by the smoothed version.

    Notes
    -----
    ``sigma`` is scaled by each point's own mean neighbour distance. It used
    to be an absolute length, which made the "Gaussian" a lie on every real
    mesh: neighbours sit ~1e-3 m apart while sigma defaulted to 1.5 m, so
    exp(-0.5*(d/sigma)^2) evaluated to 1.0 for all k+1 neighbours and the
    filter degenerated into a flat 13-point box mean. On Q-criterion — a
    field with a 1e9 dynamic range concentrated in thin vortex sheets — that
    box mean cut the peak by 12x (8.9e8 -> 7.1e7), and since the iso-surface
    views threshold on a percentile of the *smoothed* array, the surfaces
    were extracted from a field that no longer had the structure in it.
    Scaling by local spacing also makes the result mesh-independent.
    """
    if mesh is None or scalar_name not in mesh.array_names:
        logger.warning(
            f"smooth_volume_field: '{scalar_name}' not in mesh arrays."
        )
        return mesh

    if not _HAS_SCIPY:
        logger.warning("scipy not available — skipping volume smoothing.")
        return mesh

    try:
        pts = mesh.points
        n_pts = len(pts)
        if n_pts < 4:
            return mesh

        k_actual = min(k, n_pts - 1)
        tree = cKDTree(pts)
        dists, idxs = tree.query(pts, k=k_actual + 1)

        field = np.asarray(mesh[scalar_name], dtype=np.float64).copy()

        # Per-point kernel width = sigma * local mean neighbour spacing.
        # Column 0 of `dists` is the point itself (d=0), so average 1..k.
        local_h = dists[:, 1:].mean(axis=1) if k_actual >= 1 else np.ones(n_pts)
        local_h = np.where(local_h < 1e-30, 1.0, local_h)
        sigma_local = max(sigma, 1e-3) * local_h        # (n_pts,)

        smoothed = np.zeros_like(field)
        weight_sum = np.zeros(n_pts, dtype=np.float64)
        for j in range(k_actual + 1):
            d = dists[:, j]
            w = np.exp(-0.5 * (d / sigma_local) ** 2)
            smoothed += w * field[idxs[:, j]]
            weight_sum += w
        weight_sum = np.where(weight_sum < 1e-30, 1.0, weight_sum)
        field = smoothed / weight_sum

        mesh[scalar_name] = field.astype(np.float32)
        logger.info(
            f"Volume-smoothed '{scalar_name}' (σ={sigma}×local spacing, k={k_actual})"
        )

    except Exception as e:
        logger.error(f"smooth_volume_field failed: {e}")

    return mesh


# ── Derived field computation ────────────────────────────────────────────────

def compute_derived_fields(
    mesh: pv.DataSet, 
    gamma: float = 1.4, 
    r_gas: float = 287.05, 
    p_inf: float = 101325.0, 
    q_inf: float = 0.0
) -> pv.DataSet:
    """
    Computes derived fields (Velocity, Mach, Cp, Vorticity, Q-Criterion, Lambda2,
    Total Pressure, Total Temperature, Entropy) if the SU2 VTK only contains
    primitive variables (Density, Momentum, Energy, Pressure).
    Returns the mesh with added arrays.
    """
    if mesh is None:
        return mesh
        
    try:
        arrays = mesh.array_names
        
        # 1. Velocity (Momentum / Density)
        if "Velocity" not in arrays and "Momentum" in arrays and "Density" in arrays:
            rho = mesh["Density"].flatten()
            mom = mesh["Momentum"]
            safe_rho = np.where(rho < 1e-12, 1e-12, rho)
            mesh["Velocity"] = mom / safe_rho[:, np.newaxis]
            mesh["Speed"] = np.linalg.norm(mesh["Velocity"], axis=1)
            arrays = mesh.array_names
            logger.info("Computed Velocity & Speed from Momentum/Density")

        # 2. Mach (Speed / a) where a = sqrt(gamma * P / rho)
        if "Mach" not in arrays and "Speed" in arrays and "Pressure" in arrays and "Density" in arrays:
            rho = mesh["Density"].flatten()
            p = mesh["Pressure"].flatten()
            safe_rho = np.where(rho < 1e-12, 1e-12, rho)
            safe_p   = np.where(p   < 1e-12, 1e-12, p)
            a_local  = np.sqrt(gamma * safe_p / safe_rho)
            mesh["Mach"] = mesh["Speed"] / a_local
            logger.info("Computed local Mach number")

        # ── Additional thermodynamic derived fields ──────────────────────
        arrays = mesh.array_names  # refresh after Mach computation

        # 2b. Total (stagnation) Pressure — compressible isentropic relation:
        #     P_total = P * (1 + (gamma-1)/2 * M^2) ^ (gamma/(gamma-1))
        #
        # This was P + 0.5*rho*V^2, the INCOMPRESSIBLE Bernoulli form, which
        # under-reads badly once the flow is fast: at M=0.8 it is off by ~4%
        # of P_inf, and the error grows without bound through the transonic
        # and supersonic range this solver is used in. It was also gated on
        # "Speed", an array only created when Velocity has to be derived from
        # Momentum — so on real SU2 output (which ships Velocity directly) the
        # branch never ran at all and P_total was silently absent.
        if "P_total" not in arrays and "Pressure" in arrays and "Mach" in arrays:
            try:
                p = mesh["Pressure"].flatten()
                M = mesh["Mach"].flatten()
                ratio = (1.0 + (gamma - 1.0) / 2.0 * M ** 2) ** (gamma / (gamma - 1.0))
                mesh["P_total"] = (p * ratio).astype(np.float32)
                logger.info("Computed Total Pressure (P_total, isentropic)")
            except Exception as e:
                logger.warning(f"Failed to compute P_total: {e}")

        # 2c. Total Temperature: T_total = T * (1 + (gamma-1)/2 * M^2)
        if "T_total" not in arrays and "Temperature" in arrays and "Mach" in arrays:
            try:
                T = mesh["Temperature"].flatten()
                M = mesh["Mach"].flatten()
                mesh["T_total"] = (T * (1.0 + (gamma - 1.0) / 2.0 * M ** 2)).astype(np.float32)
                logger.info("Computed Total Temperature (T_total)")
            except Exception as e:
                logger.warning(f"Failed to compute T_total: {e}")

        # 2d. Entropy function: Entropy = P / rho^gamma  (shock sensor support)
        if "Entropy" not in arrays and "Pressure" in arrays and "Density" in arrays:
            try:
                rho = mesh["Density"].flatten()
                p = mesh["Pressure"].flatten()
                safe_rho = np.where(rho < 1e-12, 1e-12, rho)
                mesh["Entropy"] = (p / safe_rho ** gamma).astype(np.float32)
                logger.info("Computed Entropy function (P / rho^gamma)")
            except Exception as e:
                logger.warning(f"Failed to compute Entropy: {e}")

        # 3. Cp
        arrays = mesh.array_names
        if "Pressure_Coefficient" not in arrays and "Cp" not in arrays and "Pressure" in arrays:
            if q_inf > 1e-6:
                mesh["Pressure_Coefficient"] = (mesh["Pressure"].flatten() - p_inf) / q_inf
                logger.info(f"Computed Pressure_Coefficient using q_inf={q_inf:.2f}")

        # 4. Vorticity, Q-Criterion, Lambda-2 — fill in whatever the solver did
        #    not write. Each field below is guarded individually: this block
        #    used to recompute (and overwrite) ALL THREE whenever any one was
        #    missing, so because SU2 never emits Lambda2, every load quietly
        #    replaced SU2's own Q_Criterion with the Python one — a different
        #    field by a factor of ~3 at the peak (2.5e9 -> 8.9e8). Solver
        #    output now wins wherever it exists.
        _need_vort = "Vorticity_Magnitude" not in arrays
        _need_q    = "Q_Criterion" not in arrays
        _need_l2   = "Lambda2" not in arrays
        if "Velocity" in arrays and (_need_vort or _need_q or _need_l2):
            try:
                # Build a working copy with float64 Velocity for compute_derivative
                # (PyVista 0.48 compute_derivative requires float64 on point_data)
                working = mesh.copy()
                vel_arr = working.point_data["Velocity"]
                working.point_data["Velocity"] = vel_arr.astype(np.float64)

                # ── Vorticity & Q-Criterion ──────────────────────────────────────
                derived_vq = working.compute_derivative(
                    scalars="Velocity",
                    vorticity=_need_vort,
                    qcriterion=_need_q,
                )
                if _need_vort and "vorticity" in derived_vq.array_names:
                    vort = derived_vq["vorticity"]
                    mesh["Vorticity"]           = vort.astype(np.float32)
                    mesh["Vorticity_Magnitude"] = np.linalg.norm(vort, axis=1).astype(np.float32)
                    logger.info("Computed Vorticity & Vorticity_Magnitude")

                if _need_q and "qcriterion" in derived_vq.array_names:
                    mesh["Q_Criterion"] = derived_vq["qcriterion"].astype(np.float32)
                    logger.info("Computed Q-Criterion")

                # ── Lambda-2 (from raw velocity-gradient tensor eigenvalues) ─────
                if _need_l2:
                    derived_grad = working.compute_derivative(scalars="Velocity", gradient=True)
                    grad_key = next(
                        (k for k in derived_grad.array_names if "gradient" in k.lower()), None
                    )
                    if grad_key is not None:
                        G = derived_grad[grad_key]   # (N, 9)
                        if G.ndim == 2 and G.shape[1] == 9:
                            # Fully vectorized Lambda-2 using NumPy batch einsum
                            # Reshape to (N, 3, 3) gradient tensors
                            Gm = G.reshape(-1, 3, 3)
                            S  = 0.5 * (Gm + Gm.transpose(0, 2, 1))   # symmetric part
                            Om = 0.5 * (Gm - Gm.transpose(0, 2, 1))   # antisymmetric part
                            # M = S²+Ω² via batch matrix multiply
                            M  = np.einsum("nij,njk->nik", S, S) + np.einsum("nij,njk->nik", Om, Om)
                            # Batch eigenvalues (sorted ascending per row)
                            eigs = np.linalg.eigvalsh(M)   # (N, 3) — eigvalsh returns sorted
                            l2   = eigs[:, 1].astype(np.float32)   # 2nd eigenvalue = Lambda-2
                            mesh["Lambda2"] = l2
                            logger.info(f"Computed Lambda-2 vectorized (range {l2.min():.2e} … {l2.max():.2e})")
                        else:
                            logger.warning(f"Gradient shape {G.shape} unexpected, skipping Lambda-2")
                    else:
                        logger.warning("No gradient array found, skipping Lambda-2")


            except Exception as e:
                logger.warning(f"Failed to compute derivative fields (Vorticity/Q/Lambda2): {e}")



    except Exception as e:
        logger.error(f"Error computing derived fields: {e}")
        
    return mesh


# ── Pre-smoothed field caching ────────────────────────────────────────────────

def precompute_smoothed_fields(mesh: pv.DataSet) -> pv.DataSet:
    """
    Pre-compute Gaussian-smoothed versions of Q_Criterion and Lambda2.

    Stores smoothed copies as ``Q_Criterion_Smooth`` and ``Lambda2_Smooth``
    so the workspace doesn't have to re-smooth on every view switch.

    Parameters
    ----------
    mesh : pv.DataSet
        Volume or surface mesh that already contains ``Q_Criterion``
        and/or ``Lambda2`` arrays (typically after
        :func:`compute_derived_fields`).

    Returns
    -------
    pv.DataSet
        The *same* mesh object with the cached smooth arrays added.
    """
    if mesh is None:
        return mesh

    for field_name in ("Q_Criterion", "Lambda2"):
        smooth_name = f"{field_name}_Smooth"
        if field_name in mesh.array_names and smooth_name not in mesh.array_names:
            # Stash the original before smoothing overwrites it
            original = np.asarray(mesh[field_name], dtype=np.float32).copy()
            # sigma is in units of local mesh spacing. 0.4 keeps the kernel
            # centre dominant: measured on a real Q-criterion field it holds
            # 47% of the peak and 75% of the 99.9th percentile, against 8%
            # and 54% at the old 1.5 — enough to kill isolated solver noise
            # without flattening the vortex sheets the iso-surface is for.
            smooth_volume_field(mesh, field_name, sigma=0.4, k=12)
            mesh[smooth_name] = np.asarray(mesh[field_name], dtype=np.float32).copy()
            # Restore the original un-smoothed field
            mesh[field_name] = original
            logger.info(f"Cached pre-smoothed field: {smooth_name}")

    return mesh


# ── Mesh statistics ──────────────────────────────────────────────────────────

def compute_mesh_statistics(
    volume_mesh=None,
    surface_mesh=None,
) -> Dict:
    """
    Compute mesh quality statistics for display in the UI.

    Returns a dict with:
        total_cells, total_nodes, mean_aspect_ratio, max_skewness,
        quality_rating ("Good" | "Fair" | "Poor"), yplus_range, etc.
    """
    stats = {
        "total_cells": 0,
        "total_nodes": 0,
        "mean_aspect_ratio": 0.0,
        "max_aspect_ratio": 0.0,
        # Retained for callers that still read them. Skew is NOT computed any
        # more: VTK returns a constant -1.0 for tetrahedra, so the value was a
        # sentinel rather than a measurement. Scaled Jacobian replaces it.
        "max_skewness": 0.0,
        "mean_skewness": 0.0,
        "mean_scaled_jacobian": 0.0,
        "min_scaled_jacobian": 0.0,
        "growth_p99": 0.0,
        "growth_max": 0.0,
        "quality_rating": "Unknown",
        "quality_color": "#8b949e",
        "yplus_min": 0.0,
        "yplus_max": 0.0,
        "yplus_mean": 0.0,
    }

    vm = volume_mesh
    sm = surface_mesh

    if vm is not None:
        stats["total_cells"] = vm.n_cells
        stats["total_nodes"] = vm.n_points

        # Cell quality analysis.
        #
        # Two bugs lived here, both silent because the handlers were bare
        # `except: pass`:
        #
        #  * ``compute_cell_quality`` was REMOVED in PyVista 0.48 (it is
        #    ``cell_quality`` now, and it names its output array after the
        #    measure rather than "CellQuality"). Every call raised
        #    AttributeError, so aspect ratio and skew came back 0.0 and the
        #    rating below fell through to "Unknown" -- on every mesh, for as
        #    long as 0.48 has been installed.
        #  * ``skew`` is not defined for tetrahedra. VTK returns exactly -1.0
        #    for every tet, so even when the call worked the number was a
        #    sentinel. Scaled Jacobian is the shape metric for tets, and unlike
        #    skew it also detects inversion (<= 0 means the cell is turned
        #    inside out).
        for _measure, _lo_key, _hi_key in (
            ("aspect_ratio", "mean_aspect_ratio", "max_aspect_ratio"),
            ("scaled_jacobian", "mean_scaled_jacobian", "min_scaled_jacobian"),
        ):
            arr = None
            for _call in ("cell_quality", "compute_cell_quality"):
                fn = getattr(vm, _call, None)
                if fn is None:
                    continue
                try:
                    q = (fn(_measure) if _call == "cell_quality"
                         else fn(quality_measure=_measure))
                    key = next((k for k in (_measure, "CellQuality")
                                if k in q.array_names), None)
                    if key is None:
                        continue
                    a = np.asarray(q[key], dtype=float)
                    a = a[np.isfinite(a)]
                    if a.size:
                        arr = a
                        break
                except Exception as exc:                      # noqa: BLE001
                    logger.debug(f"{_call}({_measure}) unavailable: {exc}")
            if arr is None:
                logger.warning(f"Cell quality measure '{_measure}' unavailable "
                               f"— the mesh quality rating will be incomplete.")
                continue
            stats[_lo_key] = float(np.mean(arr))
            # Aspect ratio is worst when large, scaled Jacobian when small.
            stats[_hi_key] = float(np.max(arr) if _measure == "aspect_ratio"
                                   else np.min(arr))

    elif sm is not None:
        stats["total_cells"] = sm.n_cells
        stats["total_nodes"] = sm.n_points

    # Y+ from surface mesh. Area-weighted percentiles, not np.mean over points:
    # wall cell areas span 3.3e4 : 1 with the slivers on the nose tip, so a
    # point-counted mean describes the tip rather than the airframe (measured:
    # 241 point-counted against an area-weighted median of 338 on the same
    # surface). Reading y+ LOW is the dangerous direction -- it is the number
    # that decides whether skin friction can be trusted at all.
    if sm is not None:
        from cfd.boundary_layer import extract_yplus
        yp = extract_yplus(sm)
        if yp is not None and len(yp) > 0:
            valid = yp[yp > 0]
            if len(valid) > 0:
                stats["yplus_min"] = float(np.min(valid))
                stats["yplus_max"] = float(np.max(valid))
                name = next((n for n in ("Y_Plus", "YPlus", "y_plus")
                             if n in sm.array_names), None)
                w = field_percentiles(sm, name, [50.0]) if name else None
                stats["yplus_mean"] = (float(w[0]) if w is not None
                                       and np.isfinite(w[0])
                                       else float(np.mean(valid)))

    # ── Cell-to-cell growth ratio ────────────────────────────────────────────
    # Aspect ratio and skew describe the shape of one cell. Neither can see the
    # thing that actually bounds the accuracy of a finite-volume scheme: how
    # much the cell size changes ACROSS a face, which is where every flux and
    # every gradient is evaluated. A mesh whose size collapses over one band
    # scores "Good" on both of the old metrics.
    if vm is not None:
        try:
            ratio = _neighbour_size_ratio(vm)
            if ratio is not None:
                stats["growth_p99"] = float(np.percentile(ratio, 99))
                stats["growth_max"] = float(ratio.max())
        except Exception:
            pass

    # ── Quality rating ───────────────────────────────────────────────────────
    # Three axes, because no one of them can condemn a mesh on its own:
    #   aspect ratio     the shape of a cell
    #   scaled Jacobian  whether that shape has degenerated or inverted
    #   growth p99       how much the size changes across a face, which is where
    #                    every flux and gradient is actually evaluated
    ar = stats["mean_aspect_ratio"]
    sj = stats.get("min_scaled_jacobian", 1.0)
    gp99 = stats.get("growth_p99", 0.0)
    if ar > 0:
        if sj <= 0.0:
            stats["quality_rating"] = "Invalid"
            stats["quality_color"] = "#f85149"
        elif ar < 2.0 and sj > 0.10 and (gp99 == 0.0 or gp99 <= 1.5):
            stats["quality_rating"] = "Good"
            stats["quality_color"] = "#7ee787"
        elif ar < 4.0 and sj > 0.02 and (gp99 == 0.0 or gp99 <= 2.5):
            stats["quality_rating"] = "Fair"
            stats["quality_color"] = "#d29922"
        else:
            stats["quality_rating"] = "Poor"
            stats["quality_color"] = "#f85149"

    return stats


def _neighbour_size_ratio(vm) -> Optional[np.ndarray]:
    """max(h_a/h_b, h_b/h_a) over every interior face of a tet grid, or None."""
    try:
        cells = vm.cells_dict if hasattr(vm, "cells_dict") else {}
        tets = cells.get(10)                 # VTK_TETRA
        if tets is None or len(tets) < 100:
            return None
        tets = np.asarray(tets, dtype=np.int64)
        p = np.asarray(vm.points, dtype=float)
        v6 = np.abs(np.einsum("ij,ij->i",
                              p[tets[:, 1]] - p[tets[:, 0]],
                              np.cross(p[tets[:, 2]] - p[tets[:, 0]],
                                       p[tets[:, 3]] - p[tets[:, 0]])))
        h = np.maximum(v6, 1e-30) ** (1.0 / 3.0)
        combos = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        faces = np.concatenate([np.sort(tets[:, c], axis=1) for c in combos])
        owner = np.tile(np.arange(len(tets)), 4)
        order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
        faces, owner = faces[order], owner[order]
        same = np.all(faces[1:] == faces[:-1], axis=1)
        a, b = owner[:-1][same], owner[1:][same]
        if a.size == 0:
            return None
        return np.maximum(h[a] / h[b], h[b] / h[a])
    except Exception:
        return None


# ── Surface sampling helpers ─────────────────────────────────────────────────
#
# Both glyph builders below need the same two things: an area weight per point,
# and a spatially-uniform subset of points to draw at. Both were previously done
# with a Python loop per cell / per voxel, which on a real wall mesh (100k-500k
# cells) froze the UI thread for seconds. These are the vectorized equivalents.

def _point_areas(surf) -> Optional[np.ndarray]:
    """
    Mean incident cell area for every point of a surface mesh.

    Vectorized over the flat cell-connectivity array. The obvious loop —
    ``for ci in range(n_cells): surf.get_cell(ci).point_ids`` — constructs a
    VTK cell wrapper per cell and cost ~2.4 s on a 50k-cell sphere; this is the
    same arithmetic in two bincounts. Handles mixed cell sizes (tris + quads),
    so it does not assume a triangulated wall.

    Returns None when the mesh carries no usable area (caller falls back to a
    density estimate).
    """
    try:
        sized = surf.compute_cell_sizes(length=False, area=True, volume=False)
        if "Area" not in sized.array_names:
            return None
        cell_areas = np.asarray(sized["Area"], dtype=float)

        ug = surf.cast_to_unstructured_grid()
        conn = np.asarray(ug.cell_connectivity, dtype=np.int64)
        sizes = np.diff(np.asarray(ug.offset, dtype=np.int64))
        if conn.size == 0 or sizes.size != len(cell_areas):
            return None

        # Each entry of conn belongs to the cell that owns that slot.
        cell_of_slot = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
        n_pts = surf.n_points
        area_sum = np.bincount(conn, weights=cell_areas[cell_of_slot], minlength=n_pts)
        count = np.bincount(conn, minlength=n_pts)
        return area_sum / np.where(count == 0, 1, count)
    except Exception as e:
        logger.debug(f"Vectorized point-area computation unavailable: {e}")
        return None


def _voxel_representatives(
    pts: np.ndarray,
    bounds,
    voxel_size: float,
    scalar: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    One representative point index per occupied voxel — uniform spatial coverage
    without clusters or gaps.

    ``scalar`` (typically pressure) selects WHICH point represents a voxel: the
    one with the median value, so a single solver outlier cannot become the
    glyph for a whole region. Without it the first point in the voxel is used.

    Two fixes over the previous implementation:

    * Voxel identity comes from ``np.unique(..., axis=0)`` on the integer grid
      index, not from ``ix*100003 + iy*1009 + iz``. That hash aliases as soon as
      a z index reaches 1009 — two genuinely different voxels then merged into
      one and lost a sample.
    * Grouping is a single lexsort instead of ``for vid in unique: where(id==vid)``,
      which was O(n_voxels x n_points).
    """
    origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=float)
    grid_idx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    # Collision-free voxel labels (0..n_voxels-1), unlike a multiplicative hash.
    _, voxel_id = np.unique(grid_idx, axis=0, return_inverse=True)
    voxel_id = voxel_id.ravel()

    key = voxel_id if scalar is None else (scalar, voxel_id)
    order = np.lexsort(key if isinstance(key, tuple) else (key,))
    _, start, counts = np.unique(
        voxel_id[order], return_index=True, return_counts=True
    )
    # Middle of each sorted run = the median-scalar point of that voxel.
    return order[start + counts // 2]


# ── Force vector computation ─────────────────────────────────────────────────

def compute_force_vectors(
    surface_mesh,
    freestream_pressure: float = 101325.0,
    dynamic_pressure: float = 1.0,
    n_samples: int = 300,
    smoothing_iterations: int = 2,
) -> Optional[pv.PolyData]:
    """
    Compute physically accurate pressure force vectors on the rocket surface
    for professional-grade glyph visualization.

    Uses:
      - Smooth point normals (not noisy cell normals)
      - Poisson-disk-like spatial binning for uniform coverage
      - Laplacian smoothing of the pressure field for clean gradients
      - Proper gauge pressure physics: F = (P - P_inf) * A * n_hat
      - Adaptive sqrt-scaling of magnitudes for balanced arrow lengths

    Returns
    -------
    pv.PolyData with arrays:
        ForceVector       (N, 3) — direction & magnitude of local force
        ForceMagnitude    (N,)   — ||ForceVector||
        GaugePressure     (N,)   — (P - P_inf) at each sample
        Cp                (N,)   — pressure coefficient
        NormalDirection   (N, 3) — unit outward normal (for orient)
    """
    if surface_mesh is None or "Pressure" not in surface_mesh.array_names:
        return None

    try:
        # ── 1.  Build a clean PolyData with smooth point normals ─────────
        if hasattr(surface_mesh, 'extract_surface'):
            sm = surface_mesh.extract_surface()
        else:
            sm = surface_mesh.copy()

        sm = sm.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
            flip_normals=False,
        )

        # Ensure Pressure is on points (interpolate from cells if needed)
        if "Pressure" not in sm.point_data:
            if "Pressure" in sm.cell_data:
                sm = sm.cell_data_to_point_data()

        pts      = sm.points                        # (N_pts, 3)
        normals  = sm.point_data["Normals"]         # (N_pts, 3) — smooth
        pressure = sm.point_data["Pressure"].copy() # (N_pts,)

        n_pts = len(pts)
        if n_pts == 0:
            return None

        # ── 2.  Laplacian smooth the pressure field ──────────────────────
        # Reduces noisy spikes from solver discretization while
        # preserving the large-scale physical pressure distribution.
        if smoothing_iterations > 0 and n_pts > 20:
            if _HAS_SCIPY:
                try:
                    tree = cKDTree(pts)
                    k = min(12, n_pts - 1)
                    _, idx_nn = tree.query(pts, k=k + 1)  # includes self
                    for _ in range(smoothing_iterations):
                        p_smooth = np.zeros_like(pressure)
                        for j in range(k + 1):
                            p_smooth += pressure[idx_nn[:, j]]
                        pressure = p_smooth / (k + 1)
                except Exception:
                    pass  # smoothing failure is non-fatal

        # ── 3.  Spatial binning — Poisson-disk-like uniform sampling ─────
        # Divide the bounding box into a 3D grid and pick one
        # representative point per occupied voxel.  This guarantees
        # uniform spatial coverage with no gaps or clusters.
        bounds = sm.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
        diag   = np.sqrt(
            (bounds[1]-bounds[0])**2 +
            (bounds[3]-bounds[2])**2 +
            (bounds[5]-bounds[4])**2
        )
        # Target voxel size so we get roughly n_samples occupied voxels
        voxel_size = diag / max(n_samples ** (1/3) * 2.5, 1.0)
        voxel_size = max(voxel_size, diag * 0.005)  # floor

        # One representative point per occupied voxel, chosen by median pressure
        # so a single solver outlier never becomes a voxel's glyph.
        selected = _voxel_representatives(pts, bounds, voxel_size, scalar=pressure)

        # Clamp to n_samples if too many voxels
        if len(selected) > n_samples * 1.5:
            # Prioritize high |gauge_p| regions — sort by |P-P_inf|
            gp = np.abs(pressure[selected] - freestream_pressure)
            # Keep all above-median gauge pressure, subsample the rest
            med_gp = np.median(gp)
            high = selected[gp >= med_gp]
            low  = selected[gp < med_gp]
            n_keep_low = max(n_samples - len(high), len(low) // 4)
            if len(low) > n_keep_low:
                low = low[np.linspace(0, len(low)-1, n_keep_low, dtype=int)]
            selected = np.concatenate([high, low])

        # ── 4.  Compute force vectors at selected points ─────────────────
        sel_pts     = pts[selected]
        sel_normals = normals[selected]
        sel_press   = pressure[selected]
        sel_gauge   = sel_press - freestream_pressure

        # Re-normalize normals (they should be unit but ensure it)
        nrm_len = np.linalg.norm(sel_normals, axis=1, keepdims=True)
        nrm_len = np.where(nrm_len < 1e-12, 1.0, nrm_len)
        sel_normals = sel_normals / nrm_len

        # Cell areas averaged to points (mean incident face area).
        pt_area = _point_areas(sm)
        if pt_area is not None:
            sel_areas = pt_area[selected]
        else:
            # Fallback: estimate from point density
            avg_area = diag**2 / max(n_pts, 1) * 0.1
            sel_areas = np.full(len(selected), avg_area)

        # Force the fluid exerts ON THE BODY:
        #
        #     dF = -(P - P_inf) * n_outward * dA
        #
        # The minus sign is the whole physics: pressure PUSHES on a surface, so
        # above-ambient pressure acts along the INWARD normal. Without it every
        # arrow was drawn reversed — the stagnation point appeared to blow the
        # nose forward, and summing the glyphs gave thrust instead of drag
        # (net -3.74 N on a case the verified integrator scores at Cd=+0.80).
        # This matches _integrate_surface_forces in cfd/solvers/su2_solver.py,
        # which carries the same minus and reproduces SU2's own coefficients.
        force_mag  = sel_gauge * sel_areas
        force_vecs = -sel_normals * force_mag[:, np.newaxis]

        # ── 5.  Adaptive magnitude scaling ───────────────────────────────
        # Use sqrt-scaling to compress dynamic range:
        # strong vectors remain visible, weak ones aren't invisible
        abs_mag = np.abs(force_mag)
        mag_max = float(abs_mag.max()) if len(abs_mag) > 0 else 1.0
        if mag_max > 0:
            # sqrt-scaled magnitude for glyph sizing (same inward convention)
            sign     = np.sign(force_mag)
            sqrt_mag = sign * np.sqrt(abs_mag / mag_max) * mag_max
            scaled_vecs = -sel_normals * sqrt_mag[:, np.newaxis]
        else:
            scaled_vecs = force_vecs

        # ── 6.  Cp computation ───────────────────────────────────────────
        q_inf = max(dynamic_pressure, 1e-6)
        cp = sel_gauge / q_inf

        # ── 7.  Build output PolyData ────────────────────────────────────
        result = pv.PolyData(sel_pts.astype(np.float32))
        result["ForceVector"]     = scaled_vecs.astype(np.float32)
        result["ForceVectorTrue"] = force_vecs.astype(np.float32)
        result["ForceMagnitude"]  = np.linalg.norm(force_vecs, axis=1).astype(np.float32)
        result["GaugePressure"]   = sel_gauge.astype(np.float32)
        result["Cp"]              = cp.astype(np.float32)
        result["NormalDirection"]  = sel_normals.astype(np.float32)

        logger.info(
            f"Force vectors: {len(selected)} samples from {n_pts} surface pts  "
            f"|ΔP| range [{sel_gauge.min():.0f}, {sel_gauge.max():.0f}] Pa  "
            f"Cp range [{cp.min():.3f}, {cp.max():.3f}]"
        )
        return result

    except Exception as e:
        logger.error(f"Force vector computation failed: {e}")
        import traceback; traceback.print_exc()
        return None


# ── Pressure / Shear force decomposition ─────────────────────────────────────

def compute_pressure_shear_vectors(
    surface_mesh: pv.DataSet,
    freestream_pressure: float = 101325.0,
    dynamic_pressure: float = 1.0,
    n_samples: int = 300,
) -> Tuple[Optional[pv.PolyData], Optional[pv.PolyData]]:
    """
    Decompose surface forces into pressure and shear (friction) components.

    Returns TWO PolyData objects:
      1. **Pressure forces**: F = (P − P_inf) · A · n̂
      2. **Shear forces**:   F = τ_wall · A · t̂

    Each PolyData carries:
      - ``ForceVector``     (N, 3) — force direction and magnitude
      - ``ForceMagnitude``  (N,)   — ‖ForceVector‖
      - ``GaugePressure`` / ``WallShearStress`` — type-specific scalar

    Parameters
    ----------
    surface_mesh : pv.DataSet
        Surface mesh with ``Pressure`` (required) and optionally
        ``Skin_Friction_Coefficient`` or ``Wall_Shear_Stress`` arrays.
    freestream_pressure : float
        Free-stream static pressure P_inf [Pa].
    dynamic_pressure : float
        Free-stream dynamic pressure q_inf [Pa].
    n_samples : int
        Target number of spatially uniform sample points.

    Returns
    -------
    (pressure_poly, shear_poly) : Tuple[Optional[pv.PolyData], Optional[pv.PolyData]]
        Either may be ``None`` if the required data arrays are missing.
    """
    pressure_poly: Optional[pv.PolyData] = None
    shear_poly: Optional[pv.PolyData] = None

    if surface_mesh is None:
        return None, None

    try:
        # ── Prepare surface with smooth normals ──────────────────────────
        if hasattr(surface_mesh, "extract_surface"):
            sm = surface_mesh.extract_surface()
        else:
            sm = surface_mesh.copy()

        sm = sm.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
            flip_normals=False,
        )

        # Ensure point data
        if "Pressure" in sm.cell_data and "Pressure" not in sm.point_data:
            sm = sm.cell_data_to_point_data()

        pts = sm.points
        n_pts = len(pts)
        if n_pts == 0:
            return None, None

        normals = sm.point_data.get("Normals")
        if normals is None:
            return None, None

        # ── Spatial subsampling (same Poisson-disk strategy) ─────────────
        bounds = sm.bounds
        diag = np.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )
        voxel_size = diag / max(n_samples ** (1 / 3) * 2.5, 1.0)
        voxel_size = max(voxel_size, diag * 0.005)

        selected = _voxel_representatives(pts, bounds, voxel_size)
        if len(selected) > n_samples * 1.5:
            step = max(1, len(selected) // n_samples)
            selected = selected[::step]

        sel_pts = pts[selected]
        sel_normals = normals[selected]

        # Normalize normals
        nrm_len = np.linalg.norm(sel_normals, axis=1, keepdims=True)
        nrm_len = np.where(nrm_len < 1e-12, 1.0, nrm_len)
        sel_normals = sel_normals / nrm_len

        # ── Estimate cell areas at sample points ─────────────────────────
        pt_area = _point_areas(sm)
        if pt_area is not None:
            sel_areas = pt_area[selected]
        else:
            avg_area = diag ** 2 / max(n_pts, 1) * 0.1
            sel_areas = np.full(len(selected), avg_area)

        # ── 1. Pressure forces ───────────────────────────────────────────
        if "Pressure" in sm.point_data:
            pressure = sm.point_data["Pressure"][selected]
            gauge_p = pressure - freestream_pressure
            p_force_mag = gauge_p * sel_areas
            # dF = -(P - P_inf)*n_outward*dA — pressure pushes INWARD. Same sign
            # error as compute_force_vectors carried; see the note there.
            p_force_vecs = -sel_normals * p_force_mag[:, np.newaxis]

            pressure_poly = pv.PolyData(sel_pts.astype(np.float32))
            pressure_poly["ForceVector"] = p_force_vecs.astype(np.float32)
            pressure_poly["ForceMagnitude"] = np.linalg.norm(
                p_force_vecs, axis=1
            ).astype(np.float32)
            pressure_poly["GaugePressure"] = gauge_p.astype(np.float32)
            logger.info(
                f"Pressure force decomposition: {len(selected)} samples, "
                f"|ΔP| range [{gauge_p.min():.0f}, {gauge_p.max():.0f}] Pa"
            )

        # ── 2. Shear / friction forces ───────────────────────────────────
        # Look for wall shear stress data in various SU2/CFD naming conventions
        tau_name = None
        for candidate in [
            "Wall_Shear_Stress",
            "Skin_Friction_Coefficient",
            "WallShearStress",
            "Cf",
        ]:
            if candidate in sm.point_data:
                tau_name = candidate
                break

        if tau_name is not None:
            tau_data = sm.point_data[tau_name][selected]

            # If the array is a coefficient (Cf), convert to stress: τ = Cf * q_inf
            if "coefficient" in tau_name.lower() or tau_name in ("Cf",):
                tau_wall = tau_data * dynamic_pressure
            else:
                tau_wall = tau_data

            # Shear direction: tangent = velocity_direction projected onto surface
            # Approximate tangent from the velocity field if available
            if "Velocity" in sm.point_data:
                vel = sm.point_data["Velocity"][selected]
                # Remove normal component → tangent direction
                v_dot_n = np.sum(vel * sel_normals, axis=1, keepdims=True)
                tangent = vel - v_dot_n * sel_normals
                t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
                t_len = np.where(t_len < 1e-12, 1.0, t_len)
                t_hat = tangent / t_len
            else:
                # Fallback: use streamwise unit vector (assume x-axis)
                t_hat = np.zeros_like(sel_normals)
                t_hat[:, 0] = 1.0

            # Handle scalar vs vector tau_wall
            if tau_wall.ndim == 1:
                s_force_mag = tau_wall * sel_areas
                s_force_vecs = t_hat * s_force_mag[:, np.newaxis]
            else:
                s_force_vecs = tau_wall * sel_areas[:, np.newaxis]

            shear_poly = pv.PolyData(sel_pts.astype(np.float32))
            shear_poly["ForceVector"] = s_force_vecs.astype(np.float32)
            shear_poly["ForceMagnitude"] = np.linalg.norm(
                s_force_vecs, axis=1
            ).astype(np.float32)
            if tau_wall.ndim == 1:
                shear_poly["WallShearStress"] = tau_wall.astype(np.float32)
            else:
                shear_poly["WallShearStress"] = np.linalg.norm(
                    tau_wall, axis=1
                ).astype(np.float32)
            logger.info(
                f"Shear force decomposition: {len(selected)} samples from '{tau_name}'"
            )

    except Exception as e:
        logger.error(f"Pressure/Shear decomposition failed: {e}")
        import traceback
        traceback.print_exc()

    return pressure_poly, shear_poly


# ── Export utilities for CFD-FEM coupling ─────────────────────────────────────

def export_pressure_field(surface_mesh, output_path: Path) -> Path:
    """Export surface pressure field as CSV for FEM import."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if surface_mesh is None or "Pressure" not in surface_mesh.array_names:
        raise ValueError("No pressure data in surface mesh.")

    pts = surface_mesh.points
    P = surface_mesh["Pressure"]

    with open(output_path, "w") as f:
        f.write("x,y,z,Pressure_Pa\n")
        for i in range(len(pts)):
            f.write(f"{pts[i,0]:.6f},{pts[i,1]:.6f},{pts[i,2]:.6f},{P[i]:.2f}\n")

    logger.info(f"Pressure field exported: {output_path} ({len(pts)} points)")
    return output_path


def export_thermal_loads(surface_mesh, output_path: Path) -> Path:
    """Export surface temperature / heat flux for thermal FEM coupling."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if surface_mesh is None:
        raise ValueError("No surface mesh available.")

    pts = surface_mesh.points
    T = surface_mesh.get("Temperature", np.zeros(len(pts)))

    with open(output_path, "w") as f:
        f.write("x,y,z,Temperature_K\n")
        for i in range(len(pts)):
            f.write(f"{pts[i,0]:.6f},{pts[i,1]:.6f},{pts[i,2]:.6f},{T[i]:.2f}\n")

    logger.info(f"Thermal loads exported: {output_path}")
    return output_path


def export_aero_forces(result: "CFDResult", output_path: Path) -> Path:
    """Export aerodynamic force summary as JSON for downstream modules."""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "cd_total": result.cd,
        "cd_pressure": result.cd_pressure,
        "cd_friction": result.cd_friction,
        # Components OF cd_pressure (cd_base + cd_forebody_pressure == cd_pressure),
        # not additional terms — see cfd/drag_decomposition.py.
        "cd_base": result.cd_base,
        "cd_forebody_pressure": result.cd_forebody_pressure,
        "cd_wave": result.cd_wave,
        "base_area_m2": result.base_area_m2,
        "drag_decomposition_method": result.drag_decomposition_method,
        "cl": result.cl,
        "cm": result.cm,
        "force_axial_N": result.force_axial,
        "force_normal_N": result.force_normal,
        "cp_location_m": result.cp_location_m,
        "mach": result.mach,
        "reynolds": result.reynolds,
        "dynamic_pressure_Pa": result.dynamic_pressure,
        "converged": result.converged,
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Aero forces exported: {output_path}")
    return output_path


# ── Legacy utilities ──────────────────────────────────────────────────────────

def compute_pressure_coefficient(mesh: pv.DataSet, freestream_pressure: float,
                                  dynamic_pressure: float) -> pv.DataSet:
    """Add Cp = (P - P_inf) / q_inf as a scalar array to the mesh."""
    if "Pressure" not in mesh.array_names:
        logger.warning("No 'Pressure' array found in mesh.")
        return mesh
    P = mesh["Pressure"]
    Cp = (P - freestream_pressure) / dynamic_pressure
    mesh["Cp"] = Cp
    return mesh


def extract_streamlines(
    volume: pv.UnstructuredGrid,
    n_seeds: int = 50,
    source_radius: float = 0.05,
    surface_mesh: Optional[pv.DataSet] = None,
    seed_density: float = 1.0,
    wake_seeds: bool = True,
    adaptive_seeding: bool = True,
    integrator_type: int = 45,
) -> Optional[pv.PolyData]:
    """
    Extract velocity streamlines from the volume flow field.

    Enhanced version with RK4-5 adaptive integration, wake seed points
    behind the rocket base for recirculation visualisation, and adaptive
    seeding that places more seeds near high-Q-criterion regions.

    Parameters
    ----------
    volume : pv.UnstructuredGrid
        Volume mesh with a ``Velocity`` point-data array.
    n_seeds : int
        Base number of seed points on the upstream sphere.
    source_radius : float
        Radius of the default seed sphere (fraction of domain diagonal).
    surface_mesh : pv.DataSet, optional
        Rocket surface mesh — used to determine wake seed placement.
    seed_density : float
        Multiplier for seed count (>1 = denser seeding).
    wake_seeds : bool
        If *True*, add extra seed points behind the rocket base for
        wake / recirculation visualisation.
    adaptive_seeding : bool
        If *True* and ``Q_Criterion`` is present in *volume*, place
        additional seeds in high-Q regions (vortex cores).
    integrator_type : int
        PyVista integrator type.  Default 45 → RK4-5 adaptive.

    Returns
    -------
    pv.PolyData or None
        Streamline geometry with a ``Speed`` array suitable for tube
        radius scaling.
    """
    if volume is None:
        return None
    if "Velocity" not in volume.array_names:
        logger.warning("No 'Velocity' array for streamlines.")
        return None

    try:
        effective_seeds = max(int(n_seeds * seed_density), 4)

        # ── Primary seed source: sphere at domain centre ─────────────────
        seed_sources: List[pv.PolyData] = []
        seed_sphere = pv.Sphere(
            radius=source_radius, center=volume.center,
            theta_resolution=max(int(np.sqrt(effective_seeds)), 6),
            phi_resolution=max(int(np.sqrt(effective_seeds)), 6),
        )
        seed_sources.append(seed_sphere)

        # ── Wake seeds behind the rocket base ────────────────────────────
        if wake_seeds:
            try:
                if surface_mesh is not None:
                    surf_pts = surface_mesh.points
                    x_max = float(surf_pts[:, 0].max())
                    y_c = float(surf_pts[:, 1].mean())
                    z_c = float(surf_pts[:, 2].mean())
                    # Estimate base radius from surface extent at x_max
                    base_mask = surf_pts[:, 0] > (x_max - 0.05 * (x_max - surf_pts[:, 0].min()))
                    if np.any(base_mask):
                        r_base = float(
                            np.max(
                                np.sqrt(
                                    (surf_pts[base_mask, 1] - y_c) ** 2
                                    + (surf_pts[base_mask, 2] - z_c) ** 2
                                )
                            )
                        )
                    else:
                        r_base = source_radius
                else:
                    bounds = volume.bounds
                    x_max = bounds[1]
                    y_c = (bounds[2] + bounds[3]) / 2.0
                    z_c = (bounds[4] + bounds[5]) / 2.0
                    r_base = source_radius

                # Place a disc of seed points just downstream of the base
                n_wake = max(int(effective_seeds * 0.3), 6)
                theta = np.linspace(0, 2 * np.pi, n_wake, endpoint=False)
                radii = np.linspace(0.1 * r_base, 0.9 * r_base, 3)
                wake_pts_list: List[np.ndarray] = []
                for r in radii:
                    for t in theta:
                        wake_pts_list.append([
                            x_max + r_base * 0.3,  # just behind base
                            y_c + r * np.cos(t),
                            z_c + r * np.sin(t),
                        ])
                if wake_pts_list:
                    wake_cloud = pv.PolyData(np.array(wake_pts_list, dtype=np.float32))
                    seed_sources.append(wake_cloud)
                    logger.info(f"Added {len(wake_pts_list)} wake seed points behind base")
            except Exception as e:
                logger.warning(f"Wake seeding failed (non-fatal): {e}")

        # ── Adaptive seeds near high-Q regions ───────────────────────────
        if adaptive_seeding and "Q_Criterion" in volume.array_names:
            try:
                q_vals = volume["Q_Criterion"]
                q_thresh = np.percentile(q_vals[q_vals > 0], 90) if np.any(q_vals > 0) else 0
                if q_thresh > 0:
                    high_q_mask = q_vals > q_thresh
                    high_q_pts = volume.points[high_q_mask]
                    # Sub-sample to keep count manageable
                    n_adaptive = max(int(effective_seeds * 0.4), 6)
                    if len(high_q_pts) > n_adaptive:
                        idx = np.linspace(0, len(high_q_pts) - 1, n_adaptive, dtype=int)
                        high_q_pts = high_q_pts[idx]
                    if len(high_q_pts) > 0:
                        adaptive_cloud = pv.PolyData(high_q_pts.astype(np.float32))
                        seed_sources.append(adaptive_cloud)
                        logger.info(
                            f"Added {len(high_q_pts)} adaptive seeds near high-Q regions "
                            f"(threshold={q_thresh:.2e})"
                        )
            except Exception as e:
                logger.warning(f"Adaptive seeding failed (non-fatal): {e}")

        # ── Merge all seed sources ───────────────────────────────────────
        if len(seed_sources) == 1:
            combined_seeds = seed_sources[0]
        else:
            combined_seeds = seed_sources[0]
            for extra in seed_sources[1:]:
                combined_seeds = combined_seeds.merge(extra)

        # ── Integrate streamlines (RK4-5 adaptive) ──────────────────────
        stream = volume.streamlines_from_source(
            combined_seeds,
            vectors="Velocity",
            max_time=10.0,
            initial_step_length=0.01,
            integration_direction="both",
            integrator_type=integrator_type,
        )

        # ── Attach Speed array for tube radius scaling ───────────────────
        if stream is not None and "Velocity" in stream.array_names:
            vel = stream["Velocity"]
            stream["Speed"] = np.linalg.norm(vel, axis=1).astype(np.float32)
        elif stream is not None:
            stream["Speed"] = np.ones(stream.n_points, dtype=np.float32)

        logger.info(
            f"Streamlines extracted: {stream.n_points if stream else 0} points, "
            f"{len(seed_sources)} seed sources"
        )
        return stream

    except Exception as e:
        logger.error(f"Streamline extraction failed: {e}")
        return None


def extract_mach_iso(volume: pv.UnstructuredGrid, mach_value: float = 1.0) -> Optional[pv.PolyData]:
    """Extract the sonic surface (Mach = 1 iso-surface) from the flow field."""
    if volume is None:
        return None
    if "Mach" not in volume.array_names:
        logger.warning("No 'Mach' array found in volume flow.")
        return None
    try:
        iso = volume.contour([mach_value], scalars="Mach")
        return iso
    except Exception as e:
        logger.error(f"Mach iso-surface extraction failed: {e}")
        return None


# ── Integration with K2 simulation engine ─────────────────────────────────────

def inject_cfd_results_into_engine(result: "CFDResult", engine) -> None:
    """
    Push CFD-derived aerodynamic coefficients into the K2 RocketStateEngine,
    replacing the theoretical Barrowman approximations with high-fidelity values.
    Also injects dynamic pressure, forces, Mach, and Reynolds for use by the
    Structures workspace and other downstream consumers.
    """
    if not result.converged:
        logger.warning(
            f"CFD did not converge — not injecting results into engine. "
            f"{getattr(result, 'convergence_note', '')}"
        )
        return
    # Injecting a drag coefficient that is missing its friction component makes
    # the flight simulation over-predict apogee, and nothing downstream of the
    # engine can tell that happened. Say so at the moment the number crosses
    # over, not only in the CFD log the user has already scrolled past.
    if not getattr(result, "wall_resolved", True):
        logger.warning(
            f"INJECTING A LOWER-BOUND DRAG: {result.wall_warning} "
            f"Cd={result.cd:.5f} carries Cd_friction={result.cd_friction:.5f}. "
            f"The flight simulation will under-predict drag and therefore "
            f"over-predict apogee. Re-run with 'Euler + flat-plate friction' or "
            f"a prism boundary layer for a trustworthy total drag."
        )
    try:
        engine.update(
            cfd_cd=result.cd,
            cfd_cl=result.cl,
            cfd_cm=result.cm,
            cfd_cp_location=result.cp_location_m,
            cfd_converged=True,
            cfd_dynamic_pressure=result.dynamic_pressure,
            cfd_force_axial=result.force_axial,
            cfd_force_normal=result.force_normal,
            cfd_mach=result.mach,
            cfd_reynolds=result.reynolds,
            cfd_surface_vtk=str(result.surface_vtk) if result.surface_vtk else "",
        )
        logger.info(
            f"CFD results injected → Cd={result.cd:.4f}, Cl={result.cl:.4f}, "
            f"Cm={result.cm:.4f}, q={result.dynamic_pressure:.0f} Pa, "
            f"F_axial={result.force_axial:.2f} N, Mach={result.mach:.3f}"
        )
    except Exception as e:
        logger.error(f"Failed to inject CFD results into engine: {e}")
