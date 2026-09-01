"""
K2 AeroSim — External CAD geometry for CFD
==========================================
Lets the CFD workstation run on an arbitrary imported body (STEP / IGES /
BREP / STL / OBJ / PLY) instead of the parametric rocket built by
``cfd.meshing._build_mesh``.

Two import routes, both ending in a watertight fluid volume that the shared
meshing/solving pipeline can use:

  * B-Rep formats (.step/.stp/.iges/.igs/.brep) — imported natively by the
    Gmsh OCC kernel (``occ.importShapes``). No CadQuery needed; Gmsh links
    OpenCASCADE directly. Exact surfaces are preserved, so curvature-based
    sizing and boolean subtraction from the wind-tunnel box both work.

  * Discrete formats (.stl/.obj/.ply) — merged as a triangulated shell and
    re-topologised (``classifySurfaces`` + ``createGeometry``). These live in
    the ``geo`` kernel, which cannot do booleans, so the fluid domain is built
    as a box volume *with the body shell as an inner hole* instead.

Reference values (area/length) for the force coefficients cannot be guessed
from a rocket template here, so they are measured off the tessellation:

    ref_length = bounding-box extent along the flow axis
    ref_area   = true projected frontal area (raster of the triangles onto the
                 plane normal to the flow) — exact for concave bodies too,
                 unlike the 0.5·Σ|n·x̂|·A formula which double-counts.

Coordinate convention: the body is aligned so the chosen flow axis becomes +X
and its upstream extreme sits at x = 0 — identical to the rocket frame used by
``cfd/meshing.py`` (nose at x=0, flow along +X), so every downstream consumer
(moment origin, CP recovery, post-processing) keeps working unchanged.
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("K2.CFD.ExternalCAD")

# Formats the Gmsh OCC kernel reads natively as B-Rep solids.
BREP_SUFFIXES = {".step", ".stp", ".iges", ".igs", ".brep", ".brp"}
# Discrete surface formats — imported as a triangulated shell.
MESH_SUFFIXES = {".stl", ".obj", ".ply"}
SUPPORTED_SUFFIXES = BREP_SUFFIXES | MESH_SUFFIXES

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# ── Model units ──────────────────────────────────────────────────────────────
# Everything downstream (ISA conditions, Reynolds, reference area, force
# coefficients) is SI. A STEP authored in millimetres arrives as bare numbers,
# so a 0.816 m engine reads as 816 m and every coefficient is off by 10^3-10^6
# with nothing to flag it. UNIT_SCALES maps the declared unit to metres.
UNIT_SCALES = {
    "m":  1.0,
    "cm": 0.01,
    "mm": 0.001,
    "in": 0.0254,
    "ft": 0.3048,
}
# Above this bounding-box length (metres) an "auto" model is assumed to be in
# millimetres. Nothing anyone runs external aero on is 100 m across, and a mm
# model of a sane object lands 1000x above its true size — the two populations
# do not overlap in practice.
_AUTO_MM_THRESHOLD_M = 100.0

# classifySurfaces parameters for the discrete (STL) route.
_CLASSIFY_ANGLE_DEG = 40.0
_CLASSIFY_CURVE_ANGLE_DEG = 180.0


# ── Info record ──────────────────────────────────────────────────────────────

@dataclass
class CADInfo:
    """Measured properties of an imported CAD body, in the aligned CFD frame."""
    source: str = ""                 # original file path
    suffix: str = ""                 # lowercased extension
    is_brep: bool = False            # True → exact OCC import, False → discrete shell
    preview_stl: str = ""            # tessellated + aligned STL (also used for ref values)

    flow_axis: str = "x"             # axis of the ORIGINAL file mapped to +X
    flow_axis_auto: bool = False     # True when picked by longest-extent heuristic

    units: str = "m"                 # unit the source file was interpreted in
    unit_scale: float = 1.0          # factor applied to reach metres
    units_auto: bool = False         # True when the unit was inferred, not given

    wrapped: bool = False            # True → geometry is a distance-field wrap
    wrap_cell: float = 0.0           # wrap grid spacing [m]
    wrap_offset: float = 0.0         # wrap surface offset [m]

    length: float = 0.0              # bbox extent along flow axis [m]
    cross_width: float = 0.0         # bbox extent across flow, axis 1 [m]
    cross_height: float = 0.0        # bbox extent across flow, axis 2 [m]
    cross_radius: float = 0.0        # max distance from the flow axis [m]

    frontal_area: float = 0.0        # true projected area normal to flow [m²]
    wetted_area: float = 0.0         # total surface area [m²]
    volume: float = 0.0              # enclosed volume [m³] (0 if not watertight)

    n_triangles: int = 0
    n_solids: int = 1
    watertight: bool = True
    open_edges: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        wt = "watertight" if self.watertight else f"OPEN ({self.open_edges} free edges)"
        wr = (f"WRAPPED (cell {self.wrap_cell*1000:.2f} mm, offset "
              f"{self.wrap_offset*1000:.2f} mm), " if self.wrapped else "")
        un = ("" if self.unit_scale == 1.0 else
              f"units {self.units}{'(auto)' if self.units_auto else ''} "
              f"x{self.unit_scale:g}, ")
        return (
            f"{Path(self.source).name} — {wr}{un}L={self.length:.4f} m along "
            f"{self.flow_axis.upper()}{'(auto)' if self.flow_axis_auto else ''}, "
            f"cross {self.cross_width:.4f}×{self.cross_height:.4f} m, "
            f"frontal A={self.frontal_area:.6f} m², wetted={self.wetted_area:.4f} m², "
            f"{self.n_triangles:,} tris, {self.n_solids} solid(s), {wt}"
        )


class CADImportError(RuntimeError):
    """Raised when a CAD file cannot be loaded or is unusable for CFD."""


# ── Gmsh session helper ──────────────────────────────────────────────────────

@contextmanager
def gmsh_session(name: str = "K2_CAD", verbosity: int = 2):
    """
    Short-lived Gmsh session. Clears any stale session first (Gmsh is a global
    singleton — an abandoned session from a previous failed run makes
    ``initialize`` silently reuse the old model).
    """
    try:
        import gmsh
    except ImportError as e:
        raise CADImportError(
            "Gmsh is required for CAD import. Install it with: pip install gmsh"
        ) from e

    try:
        if gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("General.Verbosity", verbosity)
    gmsh.model.add(name)
    try:
        yield gmsh
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass


# ── Axis alignment ───────────────────────────────────────────────────────────

def resolve_flow_axis(bounds: tuple, requested: str = "auto") -> tuple[str, bool]:
    """
    Pick the model axis that should become the freestream (+X) direction.

    ``bounds`` is a PyVista-style 6-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
    "auto" takes the longest bounding-box extent — right for slender bodies
    (rockets, missiles, fuselages), wrong for a wing meshed span-wise, hence
    the explicit override.
    """
    req = (requested or "auto").strip().lower()
    if req in _AXIS_INDEX:
        return req, False
    extents = [
        abs(bounds[1] - bounds[0]),
        abs(bounds[3] - bounds[2]),
        abs(bounds[5] - bounds[4]),
    ]
    axis = "xyz"[int(max(range(3), key=lambda i: extents[i]))]
    logger.info(
        f"Flow axis auto-detected: {axis.upper()} "
        f"(extents X={extents[0]:.4f} Y={extents[1]:.4f} Z={extents[2]:.4f} m)"
    )
    return axis, True


def resolve_units(raw_length: float, requested: str = "auto") -> tuple[str, float, bool]:
    """
    Decide the unit the source file should be read in.

    ``raw_length`` is the flow-axis bounding-box extent in the file's own bare
    numbers. Returns ``(unit_name, scale_to_metres, was_auto)``.

    "auto" only ever chooses between metres and millimetres, and only on a
    margin wide enough to be unambiguous — a model measuring more than
    ``_AUTO_MM_THRESHOLD_M`` is millimetres, because nothing anyone runs
    external aero on is 100 m across. Anything else keeps metres. Explicit
    choices are honoured without argument.
    """
    req = (requested or "auto").strip().lower()
    if req in UNIT_SCALES:
        return req, UNIT_SCALES[req], False
    if raw_length > _AUTO_MM_THRESHOLD_M:
        logger.info(
            f"Model measures {raw_length:.1f} in file units along the flow axis "
            f"— assuming millimetres ({raw_length/1000:.4f} m). Override in the "
            f"CAD panel if that is wrong."
        )
        return "mm", UNIT_SCALES["mm"], True
    return "m", 1.0, True


def _align_polydata(mesh, axis: str):
    """Rotate a PyVista mesh so ``axis`` points along +X, then shift min-X to 0."""
    if axis == "y":
        mesh = mesh.rotate_z(-90.0, inplace=False)
    elif axis == "z":
        mesh = mesh.rotate_y(90.0, inplace=False)
    b = mesh.bounds
    mesh = mesh.translate(
        (-b[0], -0.5 * (b[2] + b[3]), -0.5 * (b[4] + b[5])), inplace=False
    )
    return mesh


def _align_occ_shapes(gmsh, dim_tags, axis: str):
    """
    Rotate/translate OCC shapes so ``axis`` → +X and the upstream extreme sits
    at x=0, centred on the X axis. Mirrors ``_align_polydata`` exactly so the
    exact-BRep mesh and the tessellated reference values share one frame.
    """
    occ = gmsh.model.occ
    if axis == "y":
        occ.rotate(dim_tags, 0, 0, 0, 0, 0, 1, -math.pi / 2)
    elif axis == "z":
        occ.rotate(dim_tags, 0, 0, 0, 0, 1, 0, math.pi / 2)
    occ.synchronize()

    xmin, ymin, zmin, xmax, ymax, zmax = _occ_bbox(gmsh, dim_tags)
    occ.translate(dim_tags, -xmin, -0.5 * (ymin + ymax), -0.5 * (zmin + zmax))
    occ.synchronize()


def _occ_bbox(gmsh, dim_tags) -> tuple:
    """Union bounding box of a list of OCC dimTags."""
    xs0, ys0, zs0, xs1, ys1, zs1 = [], [], [], [], [], []
    for dim, tag in dim_tags:
        b = gmsh.model.occ.getBoundingBox(dim, tag)
        xs0.append(b[0]); ys0.append(b[1]); zs0.append(b[2])
        xs1.append(b[3]); ys1.append(b[4]); zs1.append(b[5])
    return (min(xs0), min(ys0), min(zs0), max(xs1), max(ys1), max(zs1))


# ── Tessellation (BRep → STL) ────────────────────────────────────────────────

def tessellate_to_stl(
    cad_path: Path,
    out_stl: Path,
    deflection_frac: float = 0.002,
) -> Path:
    """
    Write a triangulated STL of ``cad_path``.

    B-Rep files are meshed by Gmsh at a size derived from the model diagonal
    (``deflection_frac`` of it) — fine enough for a faithful preview and for
    measuring the frontal area, far coarser than the CFD surface mesh.
    Discrete inputs are simply read and re-written by PyVista.
    """
    import pyvista as pv

    cad_path = Path(cad_path)
    out_stl = Path(out_stl)
    out_stl.parent.mkdir(parents=True, exist_ok=True)
    suffix = cad_path.suffix.lower()

    if suffix in MESH_SUFFIXES:
        mesh = pv.read(str(cad_path))
        mesh = mesh.extract_surface().triangulate().clean()
        mesh.save(str(out_stl))
        return out_stl

    if suffix not in BREP_SUFFIXES:
        raise CADImportError(f"Unsupported CAD format: {suffix or '(no extension)'}")

    with gmsh_session("K2_CAD_tessellate") as gmsh:
        try:
            gmsh.model.occ.importShapes(str(cad_path))
        except Exception as e:
            raise CADImportError(f"Gmsh could not read {cad_path.name}: {e}") from e
        gmsh.model.occ.synchronize()

        vols = gmsh.model.getEntities(3)
        surfs = gmsh.model.getEntities(2)
        if not surfs:
            raise CADImportError(
                f"{cad_path.name} contains no surfaces — nothing to mesh."
            )

        entities = vols if vols else surfs
        b = _occ_bbox(gmsh, entities)
        diag = math.dist(b[0:3], b[3:6])
        lc = max(diag * deflection_frac, 1e-9)

        gmsh.option.setNumber("Mesh.MeshSizeMin", lc * 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", lc * 4.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(out_stl))

    if not out_stl.is_file():
        raise CADImportError(f"Tessellation produced no output for {cad_path.name}")
    return out_stl


# ── Measurements ─────────────────────────────────────────────────────────────

def projected_frontal_area(mesh, axis_index: int = 0, grid: int = 384) -> float:
    """
    True projected area of a triangulated surface onto the plane normal to
    ``axis_index``, by rasterising every triangle onto a ``grid``×``grid``
    occupancy mask.

    The cheap analytic alternative, ``0.5·Σ|n·â|·A``, is only correct for
    convex bodies: on anything with a step, cavity or overlapping fin it counts
    the hidden rear-facing area as well and over-reports. Rasterising takes the
    union instead, so concave shapes are handled correctly. Grid resolution
    sets the accuracy (384² ⇒ well under 1% for typical airframes).
    """
    import numpy as np

    tri = mesh.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    pts = np.asarray(tri.points, dtype=float)
    if len(faces) == 0:
        return 0.0

    u_i, v_i = [i for i in range(3) if i != axis_index]
    u = pts[:, u_i]
    v = pts[:, v_i]

    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    du = (umax - umin) / grid
    dv = (vmax - vmin) / grid
    if du <= 0 or dv <= 0:
        return 0.0

    # Triangle vertices in grid (cell-index) space.
    au = (u[faces] - umin) / du
    av = (v[faces] - vmin) / dv

    mask = np.zeros((grid, grid), dtype=bool)

    lo_u = np.clip(np.floor(au.min(axis=1)).astype(int), 0, grid - 1)
    hi_u = np.clip(np.ceil(au.max(axis=1)).astype(int), 0, grid - 1)
    lo_v = np.clip(np.floor(av.min(axis=1)).astype(int), 0, grid - 1)
    hi_v = np.clip(np.ceil(av.max(axis=1)).astype(int), 0, grid - 1)

    # Fast path: a triangle whose bbox falls inside a single cell can only mark
    # that cell, so the point-in-triangle test is pointless. On a dense import
    # a large share of triangles are sub-cell (400k tris vs a 384² grid).
    single = (lo_u == hi_u) & (lo_v == hi_v)
    if single.any():
        mask[lo_u[single], lo_v[single]] = True

    # Everything else is batched by bounding-box footprint (nu × nv cells).
    # Triangles sharing a footprint have identically-shaped sample grids, so a
    # whole group is tested in one broadcast instead of one Python iteration
    # per triangle — for a dense mesh the footprints are overwhelmingly 2×2 and
    # 2×1, so the loop below runs a handful of times rather than ~400k.
    rest = np.flatnonzero(~single)
    if rest.size:
        nu_all = (hi_u - lo_u + 1)[rest]
        nv_all = (hi_v - lo_v + 1)[rest]
        # Guard peak memory: a group is processed in chunks of at most this
        # many sample points, so one huge triangle cannot allocate the world.
        max_samples = 4_000_000

        for nu, nv in set(zip(nu_all.tolist(), nv_all.tolist())):
            grp = rest[(nu_all == nu) & (nv_all == nv)]
            step = max(1, max_samples // (nu * nv))
            for s in range(0, grp.size, step):
                k = grp[s:s + step]
                # Cell-centre sample coordinates: (K, nu, 1) and (K, 1, nv).
                UU = (lo_u[k][:, None] + np.arange(nu) + 0.5)[:, :, None]
                VV = (lo_v[k][:, None] + np.arange(nv) + 0.5)[:, None, :]

                x1, x2, x3 = au[k, 0, None, None], au[k, 1, None, None], au[k, 2, None, None]
                y1, y2, y3 = av[k, 0, None, None], av[k, 1, None, None], av[k, 2, None, None]
                det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                # Degenerate (edge-on) triangles project to zero area; give them
                # a harmless denominator and mask them out afterwards.
                ok = np.abs(det) >= 1e-12
                den = np.where(ok, det, 1.0)
                l1 = ((y2 - y3) * (UU - x3) + (x3 - x2) * (VV - y3)) / den
                l2 = ((y3 - y1) * (UU - x3) + (x1 - x3) * (VV - y3)) / den
                l3 = 1.0 - l1 - l2
                inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9) & ok

                if inside.any():
                    IU = np.broadcast_to(UU.astype(int), inside.shape)
                    IV = np.broadcast_to(VV.astype(int), inside.shape)
                    mask[IU[inside], IV[inside]] = True

    return float(mask.sum()) * du * dv


# Wrap grid spacing as a fraction of the body's flow-axis length.
WRAP_RESOLUTIONS = {
    "coarse": 1.0 / 60.0,
    "medium": 1.0 / 100.0,
    "fine":   1.0 / 160.0,
}


def wrap_surface(mesh, cell: float, offset: float, target_triangles: int = 250_000):
    """
    Rebuild a triangulated body as a single clean outer shell by contouring its
    signed-distance field — the standard "wrap" used to make dirty CAD meshable.

    Why this exists: real assemblies fail meshing for reasons no setting fixes.
    A self-intersecting face kills the exact B-Rep route; parts that touch or
    interpenetrate leave the tessellation's shells non-disjoint, so the fluid
    volume is undefined and no mesher can fill it. A wrap never consults that
    topology — it samples distance on a grid and contours it, so overlaps,
    self-intersections and internal clutter simply do not survive.

    What it costs, and it is not free:
      * the surface is offset OUTWARD by ``offset``;
      * anything smaller than ``cell`` is rounded away;
      * internal passages are sealed — this is an OUTER MOLD LINE only.
    Right for external aero over a nacelle or airframe. Wrong if you need flow
    through the body.

    ``mesh`` must already be in metres and aligned to the CFD frame.
    """
    import numpy as np
    import pyvista as pv

    surf = mesh.extract_surface().triangulate().clean()
    b = surf.bounds
    pad = offset * 3.0 + cell * 2.0
    dims = [
        int(math.ceil((b[2 * i + 1] - b[2 * i] + 2 * pad) / cell)) + 1
        for i in range(3)
    ]
    n_pts = dims[0] * dims[1] * dims[2]
    if n_pts > 40_000_000:
        raise CADImportError(
            f"Wrap grid would need {n_pts/1e6:.0f}M points at {cell*1000:.2f} mm "
            f"spacing. Use a coarser wrap resolution."
        )
    logger.info(
        f"Wrapping: grid {dims[0]}x{dims[1]}x{dims[2]} = {n_pts/1e6:.2f}M points "
        f"@ {cell*1000:.2f} mm, offset {offset*1000:.2f} mm"
    )

    grid = pv.ImageData(
        dimensions=dims, spacing=(cell, cell, cell),
        origin=(b[0] - pad, b[2] - pad, b[4] - pad),
    )
    grid = grid.compute_implicit_distance(surf)
    wrap = grid.contour([offset], scalars="implicit_distance")
    wrap = wrap.extract_surface().triangulate().clean()
    if wrap.n_cells == 0:
        raise CADImportError(
            "Wrap produced no surface — the offset may exceed the body size."
        )

    # Keep the outer shell only; the contour can also close around cavities.
    wrap = wrap.connectivity("largest").extract_surface().triangulate().clean()

    if wrap.n_cells > target_triangles:
        wrap = wrap.decimate(
            1.0 - target_triangles / wrap.n_cells
        ).triangulate().clean()

    open_edges = _count_open_edges(wrap)
    logger.info(
        f"Wrap: {wrap.n_cells:,} triangles, {open_edges} open edge(s), "
        f"volume {wrap.volume:.6g} m3, area {wrap.area:.6g} m2"
    )
    if open_edges:
        raise CADImportError(
            f"Wrap came out with {open_edges} open edges — it should be closed "
            f"by construction. Try a coarser wrap resolution."
        )
    return wrap


def _count_open_edges(mesh) -> int:
    try:
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        return int(edges.n_cells)
    except Exception:
        return 0


def _count_solids(mesh) -> int:
    try:
        return max(1, int(mesh.split_bodies().n_blocks))
    except Exception:
        return 1


def analyze_cad(
    cad_path: Path,
    flow_axis: str = "auto",
    work_dir: Optional[Path] = None,
    max_preview_triangles: int = 400_000,
    units: str = "auto",
    wrap: bool = False,
    wrap_resolution: str = "medium",
) -> CADInfo:
    """
    Import ``cad_path``, align it to the CFD frame and measure everything the
    solver needs that a rocket template would otherwise have supplied.

    ``units`` names the unit the file's bare numbers are in ("auto", "m", "cm",
    "mm", "in", "ft"). The tessellation is scaled to METRES before anything is
    measured, so every number on the returned :class:`CADInfo` — and the aligned
    STL written to disk — is SI. That matters well beyond cosmetics: reference
    area, Reynolds number and every force coefficient are computed from these.

    Returns a :class:`CADInfo`. The aligned tessellation is written next to the
    source (or into ``work_dir``) as ``<stem>_cfd_aligned.stl`` and is what the
    3D preview and — for discrete inputs — the mesher consume.
    """
    import pyvista as pv

    cad_path = Path(cad_path)
    if not cad_path.is_file():
        raise CADImportError(f"CAD file not found: {cad_path}")
    suffix = cad_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise CADImportError(
            f"Unsupported CAD format '{suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_SUFFIXES))
        )

    out_dir = Path(work_dir) if work_dir else cad_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_stl = out_dir / f"{cad_path.stem}_cfd_raw.stl"
    aligned_stl = out_dir / f"{cad_path.stem}_cfd_aligned.stl"

    tessellate_to_stl(cad_path, raw_stl)
    mesh = pv.read(str(raw_stl)).extract_surface().triangulate().clean()
    if mesh.n_points == 0:
        raise CADImportError(f"{cad_path.name} tessellated to an empty mesh.")

    axis, auto = resolve_flow_axis(mesh.bounds, flow_axis)

    # Scale to metres BEFORE measuring, so length/areas/volume and the aligned
    # STL on disk are all SI. Auto-detection uses the flow-axis extent, so
    # resolve the axis first.
    _ax_i = _AXIS_INDEX[axis]
    _raw_len = abs(mesh.bounds[2 * _ax_i + 1] - mesh.bounds[2 * _ax_i])
    unit_name, unit_scale, units_auto = resolve_units(_raw_len, units)
    if unit_scale != 1.0:
        mesh = mesh.scale(unit_scale, inplace=False)
        logger.info(
            f"Scaled model by {unit_scale:g} ({unit_name} -> m): "
            f"{_raw_len:.4g} {unit_name} = {_raw_len * unit_scale:.4f} m"
        )

    mesh = _align_polydata(mesh, axis)

    # Wrap AFTER scaling and alignment: the grid spacing is a real length, and
    # the result must land in the CFD frame. Everything below — frontal area,
    # wetted area, the preview, and the mesher's input — then describes the
    # wrap, not the original, because the wrap IS what gets solved.
    wrapped = False
    wrap_cell = wrap_offset = 0.0
    if wrap:
        _L = abs(mesh.bounds[1] - mesh.bounds[0])
        frac = WRAP_RESOLUTIONS.get(
            (wrap_resolution or "medium").strip().lower(),
            WRAP_RESOLUTIONS["medium"],
        )
        wrap_cell = max(_L * frac, 1e-6)
        wrap_offset = 0.5 * wrap_cell
        mesh = wrap_surface(mesh, wrap_cell, wrap_offset,
                            target_triangles=max_preview_triangles)
        # Re-align: the wrap is offset outward, so its upstream extreme moved.
        mesh = _align_polydata(mesh, "x")
        wrapped = True

    # Keep the preview responsive on very dense imports; measurements below are
    # taken on the decimated copy too, which stays well within tolerance.
    if mesh.n_cells > max_preview_triangles:
        target = 1.0 - (max_preview_triangles / mesh.n_cells)
        try:
            mesh = mesh.decimate(target).triangulate().clean()
            logger.info(f"Preview decimated to {mesh.n_cells:,} triangles")
        except Exception as e:
            logger.warning(f"Decimation skipped ({e})")

    mesh.save(str(aligned_stl))
    try:
        raw_stl.unlink()
    except OSError:
        pass

    b = mesh.bounds
    length = abs(b[1] - b[0])
    cross_w = abs(b[3] - b[2])
    cross_h = abs(b[5] - b[4])
    cross_r = 0.5 * math.hypot(cross_w, cross_h)

    open_edges = _count_open_edges(mesh)
    watertight = open_edges == 0
    try:
        vol = float(mesh.volume) if watertight else 0.0
    except Exception:
        vol = 0.0

    info = CADInfo(
        source=str(cad_path),
        suffix=suffix,
        # A wrap is a triangulation, so the exact-B-Rep route no longer applies
        # even for a STEP source — the solid in the file is not what we solve.
        is_brep=(suffix in BREP_SUFFIXES) and not wrapped,
        preview_stl=str(aligned_stl),
        wrapped=wrapped,
        wrap_cell=wrap_cell,
        wrap_offset=wrap_offset,
        flow_axis=axis,
        flow_axis_auto=auto,
        units=unit_name,
        unit_scale=unit_scale,
        units_auto=units_auto,
        length=length,
        cross_width=cross_w,
        cross_height=cross_h,
        cross_radius=cross_r,
        frontal_area=projected_frontal_area(mesh, axis_index=0),
        wetted_area=float(mesh.area),
        volume=vol,
        n_triangles=int(mesh.n_cells),
        n_solids=_count_solids(mesh),
        watertight=watertight,
        open_edges=open_edges,
    )

    logger.info(f"CAD analysed: {info.summary()}")
    # Unit sanity. STEP/IGES carry a unit declaration that gmsh applies, but
    # plenty of exports are authored in millimetres and land here as bare
    # numbers — a 0.816 m jet engine then reads as 816 m. Nothing downstream can
    # tell, so Reynolds comes out 1000x high and the reference area 1e6x high,
    # and the coefficients are silently meaningless. Flag it loudly rather than
    # rescaling behind the user's back.
    if length > _AUTO_MM_THRESHOLD_M:
        logger.warning(
            f"{cad_path.name} still measures {length:.1f} m along the flow axis "
            f"after applying units='{unit_name}'. Pick the right unit in the CAD "
            f"panel, or Reynolds number, reference area and every force "
            f"coefficient will be wrong."
        )
    elif length < 1e-3:
        logger.warning(
            f"{cad_path.name} measures only {length:.3g} m along the flow axis — "
            f"check the model units."
        )
    if not watertight:
        logger.warning(
            f"{cad_path.name} is not watertight ({open_edges} free edges). "
            "Meshing may fail or leak; repair the surface before running CFD."
        )
    if info.frontal_area <= 0:
        logger.warning("Projected frontal area measured as zero — check the flow axis.")
    return info


def cad_info_from_dict(d: Optional[dict]) -> Optional[CADInfo]:
    """Rebuild a :class:`CADInfo` from the plain dict carried on CFDConfig."""
    if not d:
        return None
    known = {f for f in CADInfo.__dataclass_fields__}
    return CADInfo(**{k: v for k, v in d.items() if k in known})


# ── Gmsh import into an active meshing session ───────────────────────────────

def import_brep_into_occ(gmsh, cad_path: Path, flow_axis: str,
                         unit_scale: float = 1.0) -> list:
    """
    Import an exact B-Rep file into the *current* Gmsh OCC model, align it to
    the CFD frame and return its volume dimTags.

    ``unit_scale`` must be the SAME factor :func:`analyze_cad` applied to the
    tessellation. The exact and discrete routes have to land in one frame:
    the reference area and length come off the tessellation, and if the solid
    meshed here were still at file scale the coefficients would be normalised by
    a body a thousand times the size of the one actually solved.

    Surface-only (sheet) B-Reps are sewn into a solid so the boolean cut has
    something to subtract.
    """
    occ = gmsh.model.occ
    try:
        occ.importShapes(str(cad_path))
    except Exception as e:
        raise CADImportError(f"Gmsh could not read {Path(cad_path).name}: {e}") from e
    occ.synchronize()

    vols = gmsh.model.getEntities(3)
    if not vols:
        surfs = gmsh.model.getEntities(2)
        if not surfs:
            raise CADImportError(f"{Path(cad_path).name} contains no geometry.")
        logger.info(
            f"No solids in {Path(cad_path).name} — sewing {len(surfs)} surface(s) "
            "into a solid."
        )
        try:
            occ.healShapes(surfs, sewFaces=True, makeSolids=True)
            occ.synchronize()
            vols = gmsh.model.getEntities(3)
        except Exception as e:
            raise CADImportError(
                f"Could not sew {Path(cad_path).name} into a closed solid: {e}. "
                "Export it as a solid body (not a surface shell) and retry."
            ) from e
        if not vols:
            raise CADImportError(
                f"{Path(cad_path).name} is an open surface shell — CFD needs a "
                "closed solid. Cap the open ends in your CAD tool and re-export."
            )

    # Multi-solid assemblies must become ONE obstacle before the domain cut.
    #
    # Left separate, parts that touch or interpenetrate (the normal state of a
    # real assembly — a shaft inside a housing, a flange seated on a case) give
    # the boolean cut non-manifold junctions, and the resulting surface mesh has
    # facets from different parts crossing each other. gmsh's 3D pass then dies
    # with "PLC Error: A segment and a facet intersect". Fusing resolves the
    # overlaps into shared faces first; removeAllDuplicates merges the coincident
    # vertices/edges that STEP exports are full of.
    if len(vols) > 1:
        try:
            fused, _ = occ.fuse([vols[0]], vols[1:],
                                removeObject=True, removeTool=True)
            occ.synchronize()
            if fused:
                logger.info(f"Fused {len(vols)} solids into {len(fused)} body(ies).")
                vols = fused
        except Exception as e:
            logger.warning(
                f"Could not fuse the {len(vols)} imported solids ({e}) — meshing "
                f"them separately; overlapping parts may fail the 3D pass."
            )
    try:
        occ.removeAllDuplicates()
        occ.synchronize()
        vols = [v for v in gmsh.model.getEntities(3)]
    except Exception as e:
        logger.debug(f"removeAllDuplicates skipped: {e}")

    if unit_scale != 1.0:
        occ.dilate(vols, 0, 0, 0, unit_scale, unit_scale, unit_scale)
        occ.synchronize()
        logger.info(f"Scaled imported solids by {unit_scale:g} to metres.")

    _align_occ_shapes(gmsh, vols, flow_axis)
    logger.info(f"Imported {len(vols)} solid(s) from {Path(cad_path).name} (exact B-Rep)")
    return vols


# Above this triangle count, never attempt a reparametrisation. It is useless
# on a dense organic surface (there are no analytic patches to recover) and it
# is not merely slow — gmsh can spin forever, printing
#   "Partitioning face N with 3 triangles that all have the same partition"
#   "Tolerance too large - aborting partitioning"
# in an unbounded loop. That never raises, so an exception handler cannot save
# you; the only defence is not to call it.
# Whether a tessellation can be reparametrised is decided by SLIVER QUALITY,
# not by triangle count. Measured on the same ONERA M6 wing, classifySurfaces
# with forReparametrization=True followed by createGeometry:
#
#     tris      AR p95   AR max    time
#    19,844       2.4       21      0.6 s
#    79,048       2.4       41      2.8 s
#   219,748       2.4       68      9.1 s
#    33,796      43.0      609      >14 MINUTES of CPU, producing nothing (killed)
#
# 220k clean triangles take nine seconds; 34k sliver-ridden ones hang. The old
# gate here was a flat 25,000-triangle ceiling, which had it backwards: it
# rejected clean meshes that would have taken a few seconds and admitted
# degenerate ones that hang. A tessellator that clusters points along one
# parametric direction while leaving the other coarse (an airfoil table with
# 1e-5-chord nose spacing against a 20 mm span step) makes knife triangles, and
# the patch fitting chokes on those.
#
# So the gate is now aspect ratio, with a count ceiling kept only as a runtime
# backstop -- set at the largest case actually measured, not extrapolated.
#
# Failing this gate is not fatal: the triangulation is used directly as the wall
# mesh, which means no refinement setting can change the wall from then on.
# That cost is REPORTED, not silent -- see _surface_mesh_resolution in
# cfd/meshing.py. The productive fix for a rejected import is to re-tessellate
# the source isotropically, not to export it coarser.
_MAX_REPARAM_TRIANGLES = 250_000
_MAX_REPARAM_ASPECT_P95 = 10.0


def _triangle_aspect_p95(stl_path) -> "float | None":
    """95th-percentile triangle aspect ratio of a tessellation, or None.

    The predictor for whether ``classifySurfaces(forReparametrization=True)``
    completes — see the table above ``_MAX_REPARAM_TRIANGLES``. p95 rather than
    max, because a handful of bad triangles is survivable and the max is set by
    whichever single facet is worst.
    """
    try:
        import numpy as np
        import pyvista as pv
        m = pv.read(str(stl_path)).extract_surface().triangulate()
        f = m.faces.reshape(-1, 4)[:, 1:]
        p = np.asarray(m.points)
        e = np.stack([
            np.linalg.norm(p[f[:, 1]] - p[f[:, 0]], axis=1),
            np.linalg.norm(p[f[:, 2]] - p[f[:, 1]], axis=1),
            np.linalg.norm(p[f[:, 0]] - p[f[:, 2]], axis=1),
        ], axis=1)
        ar = e.max(axis=1) / np.maximum(e.min(axis=1), 1e-12)
        ar = ar[np.isfinite(ar)]
        return float(np.percentile(ar, 95)) if ar.size else None
    except Exception as e:                                    # noqa: BLE001
        logger.debug(f"Aspect-ratio probe failed on {stl_path}: {e}")
        return None


def import_stl_into_geo(gmsh, stl_path: Path,
                        allow_reparametrization: bool = True) -> tuple[list, bool]:
    """
    Merge an aligned STL into the current model as surface shells (``geo``
    kernel) and return ``(surface_loop_tags, reparametrised)`` — one loop per
    connected shell, since an assembly tessellates to several disjoint bodies.

    Two strategies, best first:

    1. **Re-topologised** — ``classifySurfaces(forReparametrization=True)`` plus
       ``createGeometry()`` fits analytic patches over the triangles. The wall
       can then be REMESHED to whatever the size field asks for, which is what
       you want when the STL's own resolution is not the resolution you want to
       solve on.

    2. **Discrete** — classify without a parametrisation and keep the triangles
       as the wall mesh.

    Strategy 1 is what this function used to do unconditionally, and it throws
    ``Wrong topology of boundary mesh for parametrization`` on anything organic
    or densely faceted — including surfaces that are perfectly watertight and
    manifold. That failure had nothing to do with the input being broken, and it
    made the discrete route useless as a fallback for dirty CAD, which is
    exactly when it is needed. So a parametrisation failure now drops to
    strategy 2 instead of aborting the run.

    The cost of strategy 2 is that the wall keeps the STL's own triangle sizes —
    the size field still controls the volume, but cannot refine or coarsen the
    surface. Tessellate finer upstream if you need a finer wall.

    The STL must already be aligned to the CFD frame — use the ``preview_stl``
    produced by :func:`analyze_cad`.
    """
    stl_path = Path(stl_path)
    gmsh.merge(str(stl_path))
    gmsh.model.mesh.removeDuplicateNodes()

    angle = math.radians(_CLASSIFY_ANGLE_DEG)
    curve_angle = math.radians(_CLASSIFY_CURVE_ANGLE_DEG)

    # Decide up front whether a parametrisation is even worth attempting: the
    # failure mode is a hang, not an exception, so this has to be a pre-check.
    try:
        _n_tris = sum(len(t) for t in gmsh.model.mesh.getElements(2)[1])
    except Exception:
        _n_tris = 0
    if allow_reparametrization and _n_tris > _MAX_REPARAM_TRIANGLES:
        logger.info(
            f"{stl_path.name} has {_n_tris:,} triangles — skipping "
            f"reparametrisation (above the {_MAX_REPARAM_TRIANGLES:,} runtime "
            f"backstop) and using the triangulation directly as the wall mesh."
        )
        allow_reparametrization = False
    if allow_reparametrization:
        _ar = _triangle_aspect_p95(stl_path)
        if _ar is not None and _ar > _MAX_REPARAM_ASPECT_P95:
            logger.warning(
                f"{stl_path.name} has sliver triangles (aspect ratio p95 "
                f"{_ar:.1f}, limit {_MAX_REPARAM_ASPECT_P95:g}) — skipping "
                f"reparametrisation, which would hang on them, and using the "
                f"triangulation directly as the wall mesh. THE WALL CANNOT BE "
                f"REFINED from here: re-export the geometry with a more "
                f"isotropic tessellation, or supply a STEP/IGES file."
            )
            allow_reparametrization = False

    reparametrised = True
    if not allow_reparametrization:
        reparametrised = False
        try:
            gmsh.model.mesh.classifySurfaces(angle, True, False, curve_angle)
        except Exception as e:
            raise CADImportError(
                f"Could not classify {stl_path.name}: {e}. The triangulation is "
                f"probably non-manifold — repair it, or supply a STEP file."
            ) from e
    else:
        try:
            gmsh.model.mesh.classifySurfaces(angle, True, True, curve_angle)
            gmsh.model.mesh.createGeometry()
        except Exception as e:
            logger.warning(
                f"Could not build a parametrisation for {stl_path.name} ({e}). "
                f"Falling back to the raw triangulation as the wall mesh — the "
                f"size field will control the volume but not the surface."
            )
            reparametrised = False
            # Start clean: the failed attempt leaves partial entities behind.
            try:
                gmsh.clear()
            except Exception:
                pass
            gmsh.merge(str(stl_path))
            gmsh.model.mesh.removeDuplicateNodes()
            try:
                gmsh.model.mesh.classifySurfaces(angle, True, False, curve_angle)
            except Exception as e2:
                raise CADImportError(
                    f"Could not rebuild geometry from {stl_path.name}: {e2}. The "
                    f"triangulation is probably non-manifold — repair it, or "
                    f"supply a STEP file instead."
                ) from e2

    surfs = [s[1] for s in gmsh.model.getEntities(2)]
    if not surfs:
        raise CADImportError(f"No surfaces recovered from {stl_path.name}.")

    # ONE SURFACE LOOP PER CONNECTED SHELL.
    #
    # A surface loop must be a single closed shell. An assembly tessellates to
    # several disjoint bodies, and dumping every patch into one loop makes a
    # loop that closes nothing — gmsh accepts it and then quietly meshes no
    # volume at all ("No tetrahedra in region 1"). Group by shared boundary
    # curves (union-find) so each body gets its own loop; the caller passes them
    # all to addVolume as holes in the domain.
    parent = {s: s for s in surfs}

    def _find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    curve_owner: dict = {}
    for s in surfs:
        for dim, ctag in gmsh.model.getBoundary(
            [(2, s)], combined=False, oriented=False, recursive=False
        ):
            if dim != 1:
                continue
            ctag = abs(ctag)
            if ctag in curve_owner:
                _union(curve_owner[ctag], s)
            else:
                curve_owner[ctag] = s

    shells: dict = {}
    for s in surfs:
        shells.setdefault(_find(s), []).append(s)

    loops = [gmsh.model.geo.addSurfaceLoop(patches) for patches in shells.values()]
    gmsh.model.geo.synchronize()
    logger.info(
        f"Imported {stl_path.name} as {len(loops)} discrete shell(s): "
        f"{len(surfs)} patch(es) total "
        f"(sizes {[len(v) for v in shells.values()]}, "
        f"{'reparametrised' if reparametrised else 'raw triangulation'})"
    )
    return loops, reparametrised
