"""
K2 Aerospace — Boundary Layer Analysis
=========================================
Extracts boundary layer diagnostics from CFD surface and volume meshes
for visualization: Y+, wall shear stress, separation detection, and
laminar/turbulent regime classification.

Extended capabilities include skin-friction streamline computation,
prism layer quality diagnostics, wall-normal profile extraction, and
boundary layer thickness estimation.
"""
from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

try:
    from scipy.spatial import cKDTree
    from scipy.interpolate import interp1d
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

logger = logging.getLogger("K2.CFD.BoundaryLayer")


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class PrismLayerData:
    """Prism layer quality diagnostics."""
    first_cell_height_mean: float = 0.0
    first_cell_height_min: float = 0.0
    growth_ratio_mean: float = 0.0
    yplus_target_achieved: bool = False
    diagnostic_text: str = ""


@dataclass
class BoundaryLayerData:
    """Structured boundary layer diagnostic data."""
    yplus_range: Tuple[float, float] = (0.0, 0.0)
    yplus_mean: float = 0.0
    separation_fraction: float = 0.0       # fraction of wall with separated flow
    n_separation_cells: int = 0
    regime: str = "unknown"                 # "laminar" | "transitional" | "turbulent"
    wall_shear_max: float = 0.0
    wall_shear_mean: float = 0.0
    bl_thickness_mean: float = 0.0
    bl_thickness_min: float = 0.0
    bl_thickness_max: float = 0.0
    first_cell_height: float = 0.0
    growth_ratio: float = 0.0
    prism_quality_text: str = ""


# ---------------------------------------------------------------------------
#  Backward-compatible separation result wrapper
# ---------------------------------------------------------------------------

class _SeparationResult(dict):
    """Dict subclass that also behaves as a NumPy-array-like for the mask.

    Provides backward compatibility: legacy code that treats the return
    value of ``detect_separation`` as a plain ``np.ndarray`` will get the
    boolean *mask* array via ``__array__``, indexing, ``len()``, etc.,
    while new code can access the dict keys ``'mask'``,
    ``'separation_lines'`` and ``'reattachment_lines'``.
    """

    # numpy protocol ----------------------------------------------------------
    def __array__(self, dtype=None):
        arr = self["mask"]
        return np.asarray(arr, dtype=dtype) if dtype else np.asarray(arr)

    def __len__(self):
        return len(self["mask"])

    def __bool__(self):
        # Avoid ambiguity — same behavior as np.ndarray
        raise ValueError(
            "The truth value of a _SeparationResult is ambiguous. "
            "Use result['mask'] or np.any(result['mask'])."
        )

    def __getitem__(self, key):
        # Support integer / slice / boolean indexing on the mask
        if isinstance(key, str):
            return super().__getitem__(key)
        return self["mask"][key]

    def __iter__(self):
        return iter(self["mask"])

    @property
    def dtype(self):
        return self["mask"].dtype

    @property
    def shape(self):
        return self["mask"].shape

    @property
    def ndim(self):
        return self["mask"].ndim

    def sum(self, *args, **kwargs):
        return self["mask"].sum(*args, **kwargs)


# ---------------------------------------------------------------------------
#  Core extraction helpers
# ---------------------------------------------------------------------------

def extract_yplus(surface_mesh) -> Optional[np.ndarray]:
    """Extract Y+ distribution from the surface mesh."""
    if surface_mesh is None:
        return None
    for name in ["Y_Plus", "y_plus", "YPlus", "yplus", "Y+"]:
        if name in surface_mesh.array_names:
            return np.asarray(surface_mesh[name], dtype=np.float32)
    logger.warning("Y+ field not found in surface mesh.")
    return None


def extract_wall_shear(surface_mesh, q_inf: Optional[float] = None) -> Optional[np.ndarray]:
    """Extract wall shear stress magnitude from the surface mesh.

    SU2 surface output stores the dimensionless *skin-friction
    coefficient* (Cf ~ 1e-3), not the stress. Pass the freestream
    dynamic pressure ``q_inf`` (Pa) to dimensionalize coefficient
    fields into actual wall shear stress  τ_w = Cf · q_inf.  Fields
    already stored as stress (e.g. OpenFOAM ``wallShearStress``) are
    returned unscaled.
    """
    if surface_mesh is None:
        return None

    def _dimensionalize(mag: np.ndarray, field_name: str) -> np.ndarray:
        is_coeff = ("skin_friction" in field_name.lower()
                    or field_name.startswith("SF"))
        if is_coeff and q_inf is not None and q_inf > 0:
            return (mag * q_inf).astype(np.float32)
        return mag

    # Try vector wall shear first
    for name in ["Wall_Shear_Stress", "Skin_Friction_Coefficient",
                 "wallShearStress", "skin_friction"]:
        if name in surface_mesh.array_names:
            data = surface_mesh[name]
            if data.ndim > 1:
                mag = np.linalg.norm(data, axis=1).astype(np.float32)
            else:
                mag = np.asarray(data, dtype=np.float32)
            return _dimensionalize(mag, name)

    # Try skin friction components
    for prefix in ["Skin_Friction_Coefficient", "SF"]:
        x_name = f"{prefix}_X" if f"{prefix}_X" in surface_mesh.array_names else None
        y_name = f"{prefix}_Y" if f"{prefix}_Y" in surface_mesh.array_names else None
        z_name = f"{prefix}_Z" if f"{prefix}_Z" in surface_mesh.array_names else None
        if x_name:
            cfx = surface_mesh[x_name]
            cfy = surface_mesh[y_name] if y_name else np.zeros_like(cfx)
            cfz = surface_mesh[z_name] if z_name else np.zeros_like(cfx)
            mag = np.sqrt(cfx**2 + cfy**2 + cfz**2).astype(np.float32)
            return _dimensionalize(mag, prefix)

    logger.warning("Wall shear stress field not found in surface mesh.")
    return None


# ---------------------------------------------------------------------------
#  Skin-friction vector extraction helper
# ---------------------------------------------------------------------------

def _extract_wall_shear_vector(surface_mesh) -> Optional[np.ndarray]:
    """Return the wall shear stress as an (N, 3) vector array.

    Tries common vector field names first, then constructs the vector
    from X/Y/Z component arrays, and finally falls back to projecting
    a scalar field onto the surface tangent.
    """
    if surface_mesh is None:
        return None

    # --- 1. Direct vector field -------------------------------------------
    for name in ["Skin_Friction_Coefficient", "wallShearStress",
                 "Wall_Shear_Stress", "skin_friction"]:
        if name in surface_mesh.array_names:
            data = np.asarray(surface_mesh[name], dtype=np.float32)
            if data.ndim == 2 and data.shape[1] == 3:
                return data

    # --- 2. Component arrays (X, Y, Z) -----------------------------------
    for prefix in ["Skin_Friction_Coefficient", "SF", "wallShearStress",
                   "Wall_Shear_Stress"]:
        x_key = f"{prefix}_X"
        if x_key in surface_mesh.array_names:
            cfx = np.asarray(surface_mesh[x_key], dtype=np.float32)
            y_key = f"{prefix}_Y"
            z_key = f"{prefix}_Z"
            cfy = np.asarray(surface_mesh[y_key], dtype=np.float32) \
                if y_key in surface_mesh.array_names else np.zeros_like(cfx)
            cfz = np.asarray(surface_mesh[z_key], dtype=np.float32) \
                if z_key in surface_mesh.array_names else np.zeros_like(cfx)
            return np.column_stack([cfx, cfy, cfz])

    # --- 3. Scalar → project onto surface tangent -------------------------
    scalar_shear = extract_wall_shear(surface_mesh)
    if scalar_shear is not None:
        try:
            normals_mesh = surface_mesh.compute_normals(
                point_normals=True, cell_normals=False, consistent_normals=True,
            )
            n = np.asarray(normals_mesh["Normals"], dtype=np.float32)
            # Approximate tangent as a vector in the surface plane.
            # Choose a reference vector not aligned with the normal.
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            # Per-point cross to get tangent
            tangent = np.cross(n, ref)
            tiny = np.linalg.norm(tangent, axis=1, keepdims=True) < 1e-8
            if np.any(tiny):
                ref2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                tangent[tiny.ravel()] = np.cross(n[tiny.ravel()], ref2)
            norms = np.linalg.norm(tangent, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            tangent /= norms
            return (tangent * scalar_shear[:, np.newaxis]).astype(np.float32)
        except Exception as exc:
            logger.debug(f"Tangent projection fallback failed: {exc}")

    return None


# ---------------------------------------------------------------------------
#  Separation detection (enhanced)
# ---------------------------------------------------------------------------

def detect_separation(surface_mesh) -> Optional[_SeparationResult]:
    """Detect separated flow regions on the surface.

    Returns a :class:`_SeparationResult` (dict subclass) with keys:

    * ``'mask'``  — boolean array, ``True`` = separated.
    * ``'separation_lines'``  — PolyData of zero-crossing contour lines
      where *Cf_x* crosses from positive to negative.
    * ``'reattachment_lines'`` — PolyData of zero-crossing contour lines
      where *Cf_x* crosses from negative to positive.

    For backward compatibility the returned object also behaves as a
    NumPy array (delegates to the *mask*).
    """
    import pyvista as pv

    if surface_mesh is None:
        return None

    cf_x: Optional[np.ndarray] = None

    # Look for skin friction vector (x-component indicates streamwise direction)
    for name in ["Skin_Friction_Coefficient_X", "SF_X", "wallShearStress"]:
        if name in surface_mesh.array_names:
            data = surface_mesh[name]
            if data.ndim > 1:
                # Use X component (freestream direction)
                cf_x = np.asarray(data[:, 0], dtype=np.float32)
            else:
                cf_x = np.asarray(data, dtype=np.float32)
            break

    # Fallback: use wall shear magnitude threshold  (no vector → no lines)
    if cf_x is None:
        shear = extract_wall_shear(surface_mesh)
        if shear is not None:
            positive = shear[shear > 0]
            if len(positive) == 0:
                return None   # all-zero shear — nothing meaningful to threshold
            threshold = float(np.percentile(positive, 5)) * 0.1
            separated = shear < threshold
            result = _SeparationResult(
                mask=separated,
                separation_lines=pv.PolyData(),
                reattachment_lines=pv.PolyData(),
            )
            return result
        return None

    # ---- Boolean mask ----
    separated = cf_x <= 0
    n_sep = int(np.sum(separated))
    frac = n_sep / max(len(cf_x), 1)
    logger.info(f"Separation: {n_sep} cells ({frac*100:.1f}%) of surface")

    # ---- Zero-crossing contour lines ----
    separation_lines = pv.PolyData()
    reattachment_lines = pv.PolyData()

    try:
        # Attach Cf_x to the mesh so contour() can operate on it
        work = surface_mesh.copy()
        work["Cf_x"] = cf_x
        work.set_active_scalars("Cf_x")

        contour = work.contour(isosurfaces=[0.0], scalars="Cf_x")

        if contour is not None and contour.n_points > 0:
            # Classify each contour segment as separation or reattachment.
            # Sample Cf_x gradient direction at contour points to decide.
            sampled = contour.sample(work)
            if "Cf_x" in sampled.array_names:
                # Points right *on* the contour have Cf_x ≈ 0; use the
                # gradient sign from nearby points (sampled value will be
                # very close to 0, so we rely on the gradient).
                # Approximate: shift contour points slightly in +x and
                # re-sample; positive → flow reattaches, negative → separates.
                try:
                    grad_mesh = work.compute_derivative(scalars="Cf_x")
                    grad_name = None
                    for arr_name in grad_mesh.array_names:
                        if "gradient" in arr_name.lower():
                            grad_name = arr_name
                            break
                    if grad_name is not None:
                        grad_at_contour = contour.sample(grad_mesh)
                        if grad_name in grad_at_contour.array_names:
                            grad_vals = np.asarray(
                                grad_at_contour[grad_name], dtype=np.float32,
                            )
                            # x-component of gradient of Cf_x
                            dCfx_dx = grad_vals[:, 0] if grad_vals.ndim > 1 else grad_vals
                            # Separation: dCfx/dx < 0  (Cf_x going neg)
                            # Reattachment: dCfx/dx > 0 (Cf_x going pos)
                            sep_mask = dCfx_dx < 0
                            reat_mask = ~sep_mask

                            if np.any(sep_mask):
                                separation_lines = contour.extract_points(
                                    sep_mask, adjacent_cells=True,
                                )
                            if np.any(reat_mask):
                                reattachment_lines = contour.extract_points(
                                    reat_mask, adjacent_cells=True,
                                )
                except Exception as grad_exc:
                    logger.debug(
                        f"Gradient-based sep/reattachment split failed: {grad_exc}"
                    )
                    # Fall back: whole contour labelled as separation_lines
                    separation_lines = contour

            else:
                separation_lines = contour
    except Exception as exc:
        logger.warning(f"Zero-crossing contour extraction failed: {exc}")

    result = _SeparationResult(
        mask=separated,
        separation_lines=separation_lines,
        reattachment_lines=reattachment_lines,
    )
    return result


def classify_bl_regime(surface_mesh) -> str:
    """Classify the boundary layer regime based on Y+ and skin friction.

    Returns: "laminar" | "transitional" | "turbulent" | "unknown"
    """
    yplus = extract_yplus(surface_mesh)
    if yplus is None or len(yplus) == 0:
        return "unknown"

    yp_mean = float(np.mean(yplus[yplus > 0])) if np.any(yplus > 0) else 0.0

    # Rough classification based on Y+ levels
    if yp_mean < 1.0:
        return "turbulent"   # resolved viscous sublayer (typical RANS)
    elif yp_mean < 30.0:
        return "transitional"
    else:
        return "turbulent"   # wall function region


# ---------------------------------------------------------------------------
#  Skin-friction streamlines
# ---------------------------------------------------------------------------

def compute_skin_friction_streamlines(
    surface_mesh,
    n_seeds: int = 50,
    max_length: Optional[float] = None,
):
    """Compute skin-friction streamlines on the wall surface.

    Parameters
    ----------
    surface_mesh : pyvista.PolyData
        Surface mesh with wall shear vector data.
    n_seeds : int
        Number of seed points for streamline integration.
    max_length : float or None
        Maximum streamline length. Defaults to 20 % of the surface
        bounding-box diagonal.

    Returns
    -------
    pyvista.PolyData
        Streamline polydata.  Falls back to arrow glyphs if streamline
        integration fails.
    """
    import pyvista as pv

    if surface_mesh is None or surface_mesh.n_points == 0:
        logger.warning("No surface mesh provided for skin-friction streamlines.")
        return pv.PolyData()

    shear_vec = _extract_wall_shear_vector(surface_mesh)
    if shear_vec is None:
        logger.warning("Cannot resolve wall shear vector — aborting streamlines.")
        return pv.PolyData()

    # Attach shear vector to mesh
    work = surface_mesh.copy()
    work["ShearVec"] = shear_vec.astype(np.float32)
    work.set_active_vectors("ShearVec")

    # Default max_length: 20% of bounding-box diagonal
    if max_length is None:
        diag = np.linalg.norm(
            np.asarray(work.bounds[1::2]) - np.asarray(work.bounds[::2])
        )
        max_length = float(diag * 0.2)
        if max_length <= 0:
            max_length = 1.0

    # ---- Seed points (Poisson-disk-like uniform sampling) ----------------
    seed_points = _poisson_disk_sample(work, n_seeds)
    source = pv.PolyData(seed_points)

    # ---- Streamline integration ------------------------------------------
    try:
        streamlines = work.streamlines_from_source(
            source,
            vectors="ShearVec",
            max_time=max_length,
            integration_direction="both",
        )
        if streamlines is not None and streamlines.n_points > 0:
            logger.info(
                f"Skin-friction streamlines: {streamlines.n_lines} lines "
                f"from {n_seeds} seeds."
            )
            return streamlines
    except Exception as exc:
        logger.warning(f"Streamline integration failed: {exc}")

    # ---- Fallback: arrow glyphs -----------------------------------------
    logger.info("Falling back to arrow glyphs for skin-friction visualisation.")
    try:
        sampled = work.extract_points(
            np.linspace(0, work.n_points - 1, min(n_seeds, work.n_points), dtype=int)
        )
        if sampled.n_points == 0:
            return pv.PolyData()
        sampled["ShearVec"] = shear_vec[
            np.linspace(0, work.n_points - 1, min(n_seeds, work.n_points), dtype=int)
        ].astype(np.float32)
        sampled.set_active_vectors("ShearVec")
        arrows = sampled.glyph(orient="ShearVec", scale=False, factor=max_length * 0.05)
        return arrows
    except Exception as exc2:
        logger.error(f"Arrow glyph fallback also failed: {exc2}")
        return pv.PolyData()


def _poisson_disk_sample(mesh, n_points: int) -> np.ndarray:
    """Approximate Poisson-disk sampling of surface mesh points.

    Picks an initial random point, then iteratively selects the point
    farthest from all already-selected points (greedy farthest-point
    sampling).  This yields a well-distributed seed set without
    requiring a full Poisson-disk algorithm.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Source mesh.
    n_points : int
        Number of samples to return.

    Returns
    -------
    np.ndarray
        (n_points, 3) array of selected coordinates.
    """
    pts = np.asarray(mesh.points, dtype=np.float32)
    n_total = len(pts)
    if n_total <= n_points:
        return pts.copy()

    selected_idx: List[int] = []
    rng = np.random.default_rng(42)
    first = int(rng.integers(0, n_total))
    selected_idx.append(first)

    # Distance from every point to the nearest selected point
    min_dist = np.full(n_total, np.inf, dtype=np.float32)

    for _ in range(n_points - 1):
        last = selected_idx[-1]
        d = np.linalg.norm(pts - pts[last], axis=1)
        min_dist = np.minimum(min_dist, d)
        # Pick the point with the largest minimum distance
        next_idx = int(np.argmax(min_dist))
        selected_idx.append(next_idx)

    return pts[selected_idx]


# ---------------------------------------------------------------------------
#  Prism layer quality diagnostics
# ---------------------------------------------------------------------------

def compute_prism_layer_quality(
    volume_mesh,
    surface_mesh,
) -> PrismLayerData:
    """Estimate prism layer quality from volume and surface meshes.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh containing the prism (inflation) layer.
    surface_mesh : pyvista.PolyData
        Wall surface mesh.

    Returns
    -------
    PrismLayerData
        Dataclass with first-cell height, growth ratio, and diagnostic text.
    """
    result = PrismLayerData()

    if volume_mesh is None or surface_mesh is None:
        result.diagnostic_text = "Missing volume or surface mesh."
        logger.warning(result.diagnostic_text)
        return result

    if not _HAS_SCIPY:
        result.diagnostic_text = "scipy unavailable — cannot compute prism quality."
        logger.warning(result.diagnostic_text)
        return result

    try:
        surf_pts = np.asarray(surface_mesh.points, dtype=np.float64)
        vol_centroids = np.asarray(
            volume_mesh.cell_centers().points, dtype=np.float64,
        )

        # Build KD-tree of volume cell centroids
        tree = cKDTree(vol_centroids)

        # For each surface point find the closest volume-cell centroid
        dists, _ = tree.query(surf_pts, k=1)
        first_heights = dists.astype(np.float32)

        result.first_cell_height_mean = float(np.mean(first_heights))
        result.first_cell_height_min = float(np.min(first_heights))

        # --- Growth ratio estimation -------------------------------------
        # For a subsample of surface points, find the nearest *two*
        # volume-cell centroids and estimate the growth ratio.
        n_sample = min(500, len(surf_pts))
        sample_idx = np.linspace(0, len(surf_pts) - 1, n_sample, dtype=int)
        dists_k2, _ = tree.query(surf_pts[sample_idx], k=2)
        h1 = dists_k2[:, 0]
        h2 = dists_k2[:, 1]
        valid = h1 > 1e-12
        if np.any(valid):
            ratios = h2[valid] / h1[valid]
            result.growth_ratio_mean = float(np.mean(ratios))
        else:
            result.growth_ratio_mean = 0.0

        # --- Y+ target check ----------------------------------------------
        yplus = extract_yplus(surface_mesh)
        if yplus is not None and len(yplus) > 0:
            yp_valid = yplus[yplus > 0]
            if len(yp_valid) > 0:
                result.yplus_target_achieved = bool(float(np.mean(yp_valid)) < 1.0)

        result.diagnostic_text = (
            f"First-cell height: mean={result.first_cell_height_mean:.3e}, "
            f"min={result.first_cell_height_min:.3e}. "
            f"Growth ratio (mean): {result.growth_ratio_mean:.2f}. "
            f"Y+ target (<1) achieved: {result.yplus_target_achieved}."
        )
        logger.info(f"Prism quality — {result.diagnostic_text}")

    except Exception as exc:
        result.diagnostic_text = f"Prism layer analysis failed: {exc}"
        logger.error(result.diagnostic_text)

    return result


# ---------------------------------------------------------------------------
#  Wall-normal profile extraction
# ---------------------------------------------------------------------------

def extract_wall_normal_profile(
    volume_mesh,
    surface_mesh,
    point_idx: int,
    n_samples: int = 50,
) -> Dict[str, np.ndarray]:
    """Extract velocity / temperature / TKE along the wall-normal direction.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh.
    surface_mesh : pyvista.PolyData
        Wall surface mesh.
    point_idx : int
        Index of the surface point to probe from.
    n_samples : int
        Number of sample points along the normal ray.

    Returns
    -------
    dict
        ``{'distance': ndarray, 'velocity': ndarray, 'temperature': ndarray,
        'tke': ndarray}`` — arrays of length *n_samples*.  Fields not
        found in the volume mesh are filled with NaN.
    """
    import pyvista as pv

    empty = {
        "distance": np.zeros(0, dtype=np.float32),
        "velocity": np.zeros(0, dtype=np.float32),
        "temperature": np.zeros(0, dtype=np.float32),
        "tke": np.zeros(0, dtype=np.float32),
    }

    if volume_mesh is None or surface_mesh is None:
        logger.warning("Missing meshes for wall-normal profile extraction.")
        return empty

    if point_idx < 0 or point_idx >= surface_mesh.n_points:
        logger.warning(f"point_idx {point_idx} out of range.")
        return empty

    try:
        # Compute outward normals
        normals_mesh = surface_mesh.compute_normals(
            point_normals=True, cell_normals=False, consistent_normals=True,
        )
        normals = np.asarray(normals_mesh["Normals"], dtype=np.float64)
        origin = np.asarray(surface_mesh.points[point_idx], dtype=np.float64)
        direction = normals[point_idx]
        direction /= (np.linalg.norm(direction) + 1e-30)

        # Determine ray length from bounding box
        diag = np.linalg.norm(
            np.asarray(volume_mesh.bounds[1::2]) - np.asarray(volume_mesh.bounds[::2])
        )
        ray_length = diag * 0.1  # probe up to 10% of domain size

        # Build sample points along the ray
        t_vals = np.linspace(0, ray_length, n_samples, dtype=np.float64)
        probe_pts = origin[np.newaxis, :] + np.outer(t_vals, direction)
        probe_poly = pv.PolyData(probe_pts)

        sampled = probe_poly.sample(volume_mesh)

        result: Dict[str, np.ndarray] = {
            "distance": t_vals.astype(np.float32),
        }

        # Velocity magnitude
        vel_found = False
        for vname in ["Velocity", "velocity", "U", "Velocity_Magnitude"]:
            if vname in sampled.array_names:
                v = np.asarray(sampled[vname], dtype=np.float32)
                if v.ndim > 1:
                    v = np.linalg.norm(v, axis=1).astype(np.float32)
                result["velocity"] = v
                vel_found = True
                break
        if not vel_found:
            result["velocity"] = np.full(n_samples, np.nan, dtype=np.float32)

        # Temperature
        temp_found = False
        for tname in ["Temperature", "temperature", "T", "Static_Temperature"]:
            if tname in sampled.array_names:
                result["temperature"] = np.asarray(
                    sampled[tname], dtype=np.float32,
                )
                temp_found = True
                break
        if not temp_found:
            result["temperature"] = np.full(n_samples, np.nan, dtype=np.float32)

        # TKE
        tke_found = False
        for kname in ["TKE", "k", "Turbulent_Kinetic_Energy", "tke"]:
            if kname in sampled.array_names:
                result["tke"] = np.asarray(sampled[kname], dtype=np.float32)
                tke_found = True
                break
        if not tke_found:
            result["tke"] = np.full(n_samples, np.nan, dtype=np.float32)

        logger.debug(
            f"Wall-normal profile at point {point_idx}: "
            f"{n_samples} samples, ray length={ray_length:.4f}"
        )
        return result

    except Exception as exc:
        logger.error(f"Wall-normal profile extraction failed: {exc}")
        return empty


# ---------------------------------------------------------------------------
#  Boundary layer thickness estimation
# ---------------------------------------------------------------------------

def estimate_boundary_layer_thickness(
    volume_mesh,
    surface_mesh,
    n_samples: int = 20,
) -> Dict[str, float]:
    """Estimate δ₉₉ boundary layer thickness at sampled surface locations.

    Samples *n_samples* wall-normal profiles and finds the distance at
    which the velocity reaches 99 % of the edge (freestream) velocity.

    Parameters
    ----------
    volume_mesh : pyvista.UnstructuredGrid
        Volume mesh.
    surface_mesh : pyvista.PolyData
        Wall surface mesh.
    n_samples : int
        Number of surface sample locations.

    Returns
    -------
    dict
        ``{'mean': float, 'min': float, 'max': float}`` of δ₉₉.
    """
    fallback = {"mean": 0.0, "min": 0.0, "max": 0.0}

    if volume_mesh is None or surface_mesh is None:
        logger.warning("Missing meshes for BL thickness estimation.")
        return fallback

    n_surf = surface_mesh.n_points
    if n_surf == 0:
        return fallback

    sample_indices = np.linspace(0, n_surf - 1, min(n_samples, n_surf), dtype=int)
    thicknesses: List[float] = []

    for idx in sample_indices:
        profile = extract_wall_normal_profile(
            volume_mesh, surface_mesh, int(idx), n_samples=60,
        )
        vel = profile.get("velocity")
        dist = profile.get("distance")
        if vel is None or dist is None or len(vel) == 0:
            continue
        if np.all(np.isnan(vel)):
            continue

        # Edge velocity = maximum velocity in the profile
        u_edge = float(np.nanmax(vel))
        if u_edge <= 0:
            continue

        # δ₉₉: first distance where u >= 0.99 * u_edge
        target = 0.99 * u_edge
        above = np.where(vel >= target)[0]
        if len(above) > 0:
            first_above = above[0]
            if first_above > 0 and _HAS_SCIPY:
                # Linear interpolation for better accuracy
                try:
                    f_interp = interp1d(
                        vel[first_above - 1:first_above + 1],
                        dist[first_above - 1:first_above + 1],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    delta99 = float(f_interp(target))
                except Exception:
                    delta99 = float(dist[first_above])
            else:
                delta99 = float(dist[first_above])
            thicknesses.append(delta99)

    if len(thicknesses) == 0:
        logger.info("Could not estimate BL thickness at any sample point.")
        return fallback

    bl_arr = np.array(thicknesses, dtype=np.float32)
    result = {
        "mean": float(np.mean(bl_arr)),
        "min": float(np.min(bl_arr)),
        "max": float(np.max(bl_arr)),
    }
    logger.info(
        f"BL thickness δ₉₉: mean={result['mean']:.5f}, "
        f"min={result['min']:.5f}, max={result['max']:.5f}"
    )
    return result


# ---------------------------------------------------------------------------
#  Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_boundary_layer(
    surface_mesh,
    volume_mesh=None,
    q_inf: Optional[float] = None,
) -> BoundaryLayerData:
    """Full boundary layer analysis returning structured diagnostics.

    Parameters
    ----------
    surface_mesh : pyvista.PolyData
        Wall surface mesh.
    volume_mesh : pyvista.UnstructuredGrid, optional
        Volume mesh.  When provided, prism layer quality and BL
        thickness diagnostics are also computed.
    q_inf : float, optional
        Freestream dynamic pressure (Pa).  When given, skin-friction
        coefficient fields are dimensionalized so the wall shear stats
        are in Pa instead of dimensionless Cf.

    Returns
    -------
    BoundaryLayerData
    """
    data = BoundaryLayerData()

    yplus = extract_yplus(surface_mesh)
    if yplus is not None and len(yplus) > 0:
        valid = yplus[yplus > 0]
        if len(valid) > 0:
            data.yplus_range = (float(np.min(valid)), float(np.max(valid)))
            data.yplus_mean = float(np.mean(valid))

    shear = extract_wall_shear(surface_mesh, q_inf=q_inf)
    if shear is not None and len(shear) > 0:
        data.wall_shear_max = float(np.max(shear))
        data.wall_shear_mean = float(np.mean(shear))

    sep = detect_separation(surface_mesh)
    if sep is not None:
        data.n_separation_cells = int(np.sum(sep))
        data.separation_fraction = data.n_separation_cells / max(len(sep), 1)

    data.regime = classify_bl_regime(surface_mesh)

    # --- Volume-mesh-dependent diagnostics --------------------------------
    if volume_mesh is not None:
        # Prism layer quality
        prism = compute_prism_layer_quality(volume_mesh, surface_mesh)
        data.first_cell_height = prism.first_cell_height_mean
        data.growth_ratio = prism.growth_ratio_mean
        data.prism_quality_text = prism.diagnostic_text

        # Boundary layer thickness
        bl = estimate_boundary_layer_thickness(volume_mesh, surface_mesh)
        data.bl_thickness_mean = bl["mean"]
        data.bl_thickness_min = bl["min"]
        data.bl_thickness_max = bl["max"]

    logger.info(
        f"BL Analysis: Y+=[{data.yplus_range[0]:.1f}, {data.yplus_range[1]:.1f}] "
        f"mean={data.yplus_mean:.1f}, regime={data.regime}, "
        f"sep={data.separation_fraction*100:.1f}%"
    )
    return data
