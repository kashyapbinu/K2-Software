"""
K2 Aerospace — CFD Post-Processing
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract circumferentially-averaged Cp distribution along the rocket body.

    Parameters
    ----------
    surface_mesh : PyVista surface mesh with Cp or Pressure arrays
    axis : str - rocket axis direction ("x" for CFD frame)
    n_stations : int - number of sampling stations along the body
    freestream_pressure : float - P_inf for computing Cp from Pressure
    dynamic_pressure : float - q_inf for computing Cp from Pressure

    Returns
    -------
    (x_normalized, cp_values) : normalized position [0,1] and Cp values
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

    pts = surface_mesh.points
    cp_vals = surface_mesh[cp_name]

    # Determine axis index
    ax_idx = {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 0)

    x_coords = pts[:, ax_idx]
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    x_range = x_max - x_min
    if x_range < 1e-6:
        return np.array([]), np.array([])

    # Bin and average Cp at each station
    stations = np.linspace(x_min, x_max, n_stations + 1)
    x_norm = np.zeros(n_stations)
    cp_avg = np.zeros(n_stations)

    for i in range(n_stations):
        mask = (x_coords >= stations[i]) & (x_coords < stations[i + 1])
        if np.any(mask):
            cp_avg[i] = float(np.mean(cp_vals[mask]))
        elif i > 0:
            cp_avg[i] = cp_avg[i - 1]  # carry forward
        x_norm[i] = (stations[i] + stations[i + 1]) / 2.0
        x_norm[i] = (x_norm[i] - x_min) / x_range  # normalize to [0, 1]

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
    sigma: float = 1.5,
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
        Standard deviation of the Gaussian kernel.  Default 1.5.
    k : int, optional
        Number of nearest neighbours used for the kernel.  Default 12.

    Returns
    -------
    pv.DataSet
        The *same* mesh with the scalar replaced by the smoothed version.
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

        smoothed = np.zeros_like(field)
        weight_sum = np.zeros(n_pts, dtype=np.float64)
        for j in range(k_actual + 1):
            d = dists[:, j]
            w = np.exp(-0.5 * (d / max(sigma, 1e-12)) ** 2)
            smoothed += w * field[idxs[:, j]]
            weight_sum += w
        weight_sum = np.where(weight_sum < 1e-30, 1.0, weight_sum)
        field = smoothed / weight_sum

        mesh[scalar_name] = field.astype(np.float32)
        logger.info(
            f"Volume-smoothed '{scalar_name}' (σ={sigma}, k={k_actual})"
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

        # 2b. Total Pressure: P_total = P + 0.5 * rho * V^2
        if "P_total" not in arrays and "Pressure" in arrays and "Density" in arrays and "Speed" in arrays:
            try:
                rho = mesh["Density"].flatten()
                p = mesh["Pressure"].flatten()
                speed = mesh["Speed"].flatten()
                safe_rho = np.where(rho < 1e-12, 1e-12, rho)
                mesh["P_total"] = (p + 0.5 * safe_rho * speed ** 2).astype(np.float32)
                logger.info("Computed Total Pressure (P_total)")
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

        # 4. Vorticity, Q-Criterion, Lambda-2 — always recompute if Velocity is present
        if "Velocity" in arrays and not all(k in arrays for k in ["Vorticity_Magnitude", "Q_Criterion", "Lambda2"]):
            try:
                # Build a working copy with float64 Velocity for compute_derivative
                # (PyVista 0.48 compute_derivative requires float64 on point_data)
                working = mesh.copy()
                vel_arr = working.point_data["Velocity"]
                working.point_data["Velocity"] = vel_arr.astype(np.float64)

                # ── Vorticity & Q-Criterion ──────────────────────────────────────
                derived_vq = working.compute_derivative(
                    scalars="Velocity", vorticity=True, qcriterion=True
                )
                if "vorticity" in derived_vq.array_names:
                    vort = derived_vq["vorticity"]
                    mesh["Vorticity"]           = vort.astype(np.float32)
                    mesh["Vorticity_Magnitude"] = np.linalg.norm(vort, axis=1).astype(np.float32)
                    logger.info("Computed Vorticity & Vorticity_Magnitude")

                if "qcriterion" in derived_vq.array_names:
                    mesh["Q_Criterion"] = derived_vq["qcriterion"].astype(np.float32)
                    logger.info("Computed Q-Criterion")

                # ── Lambda-2 (from raw velocity-gradient tensor eigenvalues) ─────
                if "Lambda2" not in arrays:
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
            smooth_volume_field(mesh, field_name, sigma=1.5, k=12)
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
        "max_skewness": 0.0,
        "mean_skewness": 0.0,
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

        # Cell quality analysis
        try:
            qual = vm.compute_cell_quality(quality_measure="aspect_ratio")
            ar = qual["CellQuality"]
            stats["mean_aspect_ratio"] = float(np.mean(ar))
            stats["max_aspect_ratio"] = float(np.max(ar))
        except Exception:
            pass

        try:
            qual = vm.compute_cell_quality(quality_measure="skew")
            sk = qual["CellQuality"]
            stats["mean_skewness"] = float(np.mean(sk))
            stats["max_skewness"] = float(np.max(sk))
        except Exception:
            pass

    elif sm is not None:
        stats["total_cells"] = sm.n_cells
        stats["total_nodes"] = sm.n_points

    # Y+ from surface mesh
    if sm is not None:
        from cfd.boundary_layer import extract_yplus
        yp = extract_yplus(sm)
        if yp is not None and len(yp) > 0:
            valid = yp[yp > 0]
            if len(valid) > 0:
                stats["yplus_min"] = float(np.min(valid))
                stats["yplus_max"] = float(np.max(valid))
                stats["yplus_mean"] = float(np.mean(valid))

    # Quality rating
    ar = stats["mean_aspect_ratio"]
    sk = stats["max_skewness"]
    if ar > 0:
        if ar < 5.0 and sk < 0.7:
            stats["quality_rating"] = "Good"
            stats["quality_color"] = "#7ee787"
        elif ar < 15.0 and sk < 0.85:
            stats["quality_rating"] = "Fair"
            stats["quality_color"] = "#d29922"
        else:
            stats["quality_rating"] = "Poor"
            stats["quality_color"] = "#f85149"

    return stats


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

        # Assign each point to a voxel
        grid_idx = np.floor((pts - [bounds[0], bounds[2], bounds[4]]) / voxel_size).astype(np.int32)
        # Hash to unique voxel ID
        voxel_id = grid_idx[:, 0] * 100003 + grid_idx[:, 1] * 1009 + grid_idx[:, 2]

        # Pick the point closest to voxel center in each bin
        unique_voxels = np.unique(voxel_id)
        selected = np.empty(len(unique_voxels), dtype=np.int64)
        for i, vid in enumerate(unique_voxels):
            mask = np.where(voxel_id == vid)[0]
            if len(mask) == 1:
                selected[i] = mask[0]
            else:
                # Pick point with median pressure (avoids outliers)
                p_local = pressure[mask]
                median_idx = np.argmin(np.abs(p_local - np.median(p_local)))
                selected[i] = mask[median_idx]

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

        # Cell areas: estimate via Voronoi area (avg face area per point)
        # Use compute_cell_sizes on the full mesh, then average to points
        try:
            sized = sm.compute_cell_sizes()
            if "Area" in sized.array_names:
                cell_areas = sized["Area"]
                # Average cell area to points via point-cell connectivity
                pt_area = np.zeros(n_pts)
                pt_count = np.zeros(n_pts)
                for ci in range(sm.n_cells):
                    cell_pt_ids = sm.get_cell(ci).point_ids
                    for pid in cell_pt_ids:
                        pt_area[pid] += cell_areas[ci]
                        pt_count[pid] += 1
                pt_count = np.where(pt_count == 0, 1, pt_count)
                pt_area = pt_area / pt_count  # average area per face
                sel_areas = pt_area[selected]
            else:
                raise ValueError("No Area")
        except Exception:
            # Fallback: estimate from point density
            avg_area = diag**2 / max(n_pts, 1) * 0.1
            sel_areas = np.full(len(selected), avg_area)

        # Force = (P - P_inf) * A * n_hat  (points outward for positive gauge)
        force_mag  = sel_gauge * sel_areas
        force_vecs = sel_normals * force_mag[:, np.newaxis]

        # ── 5.  Adaptive magnitude scaling ───────────────────────────────
        # Use sqrt-scaling to compress dynamic range:
        # strong vectors remain visible, weak ones aren't invisible
        abs_mag = np.abs(force_mag)
        mag_max = float(abs_mag.max()) if len(abs_mag) > 0 else 1.0
        if mag_max > 0:
            # sqrt-scaled magnitude for glyph sizing
            sign     = np.sign(force_mag)
            sqrt_mag = sign * np.sqrt(abs_mag / mag_max) * mag_max
            scaled_vecs = sel_normals * sqrt_mag[:, np.newaxis]
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

        grid_idx = np.floor(
            (pts - [bounds[0], bounds[2], bounds[4]]) / voxel_size
        ).astype(np.int32)
        voxel_id = (
            grid_idx[:, 0] * 100003
            + grid_idx[:, 1] * 1009
            + grid_idx[:, 2]
        )
        unique_voxels = np.unique(voxel_id)
        selected = np.empty(len(unique_voxels), dtype=np.int64)
        for i, vid in enumerate(unique_voxels):
            mask = np.where(voxel_id == vid)[0]
            selected[i] = mask[0]
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
        try:
            sized = sm.compute_cell_sizes()
            if "Area" in sized.array_names:
                cell_areas = sized["Area"]
                pt_area = np.zeros(n_pts)
                pt_count = np.zeros(n_pts)
                for ci in range(sm.n_cells):
                    cell_pt_ids = sm.get_cell(ci).point_ids
                    for pid in cell_pt_ids:
                        pt_area[pid] += cell_areas[ci]
                        pt_count[pid] += 1
                pt_count = np.where(pt_count == 0, 1, pt_count)
                pt_area = pt_area / pt_count
                sel_areas = pt_area[selected]
            else:
                raise ValueError("No Area")
        except Exception:
            avg_area = diag ** 2 / max(n_pts, 1) * 0.1
            sel_areas = np.full(len(selected), avg_area)

        # ── 1. Pressure forces ───────────────────────────────────────────
        if "Pressure" in sm.point_data:
            pressure = sm.point_data["Pressure"][selected]
            gauge_p = pressure - freestream_pressure
            p_force_mag = gauge_p * sel_areas
            p_force_vecs = sel_normals * p_force_mag[:, np.newaxis]

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
        "cd_base": result.cd_base,
        "cd_wave": result.cd_wave,
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
        logger.warning("CFD did not converge — not injecting results into engine.")
        return
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
