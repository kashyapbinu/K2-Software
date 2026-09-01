"""
K2 AeroSim — SU2 Solver Backend
====================================
Implements CFDSolver using the Stanford SU2 open-source CFD suite.
SU2 binaries (SU2_CFD, SU2_DEF) must be present in the K2 bin/ folder.

Physics:
  - Euler/RANS compressible Navier-Stokes
  - SST k-omega turbulence model
  - Density-based implicit solver (accurate for transonic/supersonic)
"""
from __future__ import annotations

import csv
import logging
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from cfd.solvers.base import CFDSolver, CFDConfig, CFDResult, isa_conditions

logger = logging.getLogger("K2.CFD.SU2")

# Suppress the console window when launching the (console) solver from the
# windowed frozen app — otherwise a terminal flashes on every Run.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW

# ── Locate the bundled SU2 binary ────────────────────────────────────────────
from core.paths import bin_dir
_ROOT = Path(__file__).resolve().parents[2]   # K2 Software root
_BIN_DIR = bin_dir()   # platform-aware: bin/mac-<arch> on macOS, bin/ on Windows

def _find_su2() -> Optional[Path]:
    """Find SU2_CFD: bundled bin/ first, then system PATH."""
    candidates = [
        _BIN_DIR / "SU2_CFD.exe",
        _BIN_DIR / "SU2_CFD",
        _BIN_DIR / "su2_cfd",
    ]
    for c in candidates:
        if c.is_file():
            return c
    # Fall back to PATH
    import shutil
    found = shutil.which("SU2_CFD") or shutil.which("su2_cfd")
    return Path(found) if found else None


def _find_mpi() -> Optional[Path]:
    """Find an MPI launcher (mpiexec/mpirun): bundled bin/ first, then PATH.

    Used to run SU2_CFD across multiple ranks. Returns None when no launcher is
    available, in which case the solver runs serial. Note: a launcher being
    present does NOT guarantee SU2 was built with MPI — if SU2 is a serial build,
    running it under mpiexec spawns N independent single-rank copies that clobber
    each other's output. Only raise n_cores when the bundled SU2 is an MPI build.
    """
    import shutil
    for name in ("mpiexec.exe", "mpiexec", "mpirun.exe", "mpirun"):
        c = _BIN_DIR / name
        if c.is_file():
            return c
    for name in ("mpiexec", "mpirun"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


# Cached result of the MPI-build probe (see SU2Solver.run):
#   None  = not yet probed
#   True  = SU2 confirmed an MPI build (parallel runs are valid)
#   False = SU2 is a serial build (mpiexec spawns N clobbering copies) → use serial
# Module-level so the probe runs once per session, not once per sweep point.
_MPI_BUILD_OK: Optional[bool] = None


# ── SU2 Configuration Template ───────────────────────────────────────────────
# Minimum fall in the density residual, in decades, for a run to count as
# converged on the residual alone. Relative because the residual is
# dimensional -- see CFDResult.residual_drop_decades. Three decades is the low
# end of accepted steady-RANS practice and is corroborated in parse_results by
# a stationarity test on the drag coefficient itself, which is the quantity the
# polars actually report.
_MIN_RESIDUAL_DECADES = 3.0

_TURB_MODEL_MAP = {
    "Euler":   {"solver": "EULER",          "turb": None,  "turb_line": ""},
    "Laminar": {"solver": "NAVIER_STOKES",  "turb": None,  "turb_line": ""},
    "SA":      {"solver": "RANS",           "turb": "SA",  "turb_line": "KIND_TURB_MODEL= SA"},
    "SST":     {"solver": "RANS",           "turb": "SST", "turb_line": "KIND_TURB_MODEL= SST\nSST_OPTIONS= VORTICITY"},
    # NOTE: no "KE" entry — SU2 has no k-epsilon model (SA/SST only).
    # Unknown keys fall back to SST in generate_case().
}

_SU2_CONFIG_TEMPLATE = """\
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% K2 AeroSim - SU2 CFD Configuration                   %%
%% Auto-generated - do not edit by hand                   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ── Solver ───────────────────────────────────────────────
SOLVER= {solver_type}
{turb_model_lines}
MATH_PROBLEM= DIRECT
RESTART_SOL= {restart_sol}

% ── Free-stream (ISA dimensional) ────────────────────────
MACH_NUMBER= {mach}
AOA= {aoa}
SIDESLIP_ANGLE= {sideslip}
FREESTREAM_PRESSURE=    {pressure}
FREESTREAM_TEMPERATURE= {temperature}
REYNOLDS_NUMBER=  {reynolds}
REYNOLDS_LENGTH=  {ref_length}

% Free-stream turbulence (Ti=0.1% — low external turbulence, typical for rockets)
FREESTREAM_TURBULENCEINTENSITY= 0.001
FREESTREAM_TURB2LAMVISCRATIO= 10.0

% ── Reference values ─────────────────────────────────────
% Moment origin = nose tip (x=0 in the mesh frame; see cfd/meshing.py).
% Pitch moment for an AoA run is CMy (SU2 rotates the freestream about Y),
% so CP-from-nose = -(CMy/CN)*L.
REF_ORIGIN_MOMENT_X= {moment_x}
REF_ORIGIN_MOMENT_Y= 0.0
REF_ORIGIN_MOMENT_Z= 0.0
REF_LENGTH=  {ref_length}
REF_AREA=    {ref_area}
REF_DIMENSIONALIZATION= DIMENSIONAL

% ── Boundary conditions ───────────────────────────────────
{wall_bc}
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( rocket_wall )
MARKER_MONITORING= ( rocket_wall )

% ── Numerical schemes ─────────────────────────────────────
% GREEN_GAUSS. This was switched to WEIGHTED_LEAST_SQUARES on the textbook
% argument -- Green-Gauss assumes a regular cell and is not first-order
% consistent on irregular tets, which is all this mesher produces -- and the
% argument did not survive measurement. A/B on both benchmarks, everything else
% held fixed:
%
%                        WLS       GREEN_GAUSS
%   ONERA M6  CL        0.2618      0.2624
%   M6 c_n y/b=0.65      3.3%        2.2%
%   M6 c_n y/b=0.80      1.5%        0.4%
%   cone Cp vs exact     8.89%       4.84%
%
% Equal or better everywhere, and decisively better on the shock case. Least
% squares is the more accurate reconstruction on a smooth field but the less
% monotone one across a discontinuity, and both of these cases are
% shock-dominated. Do not switch it back without re-running both.
NUM_METHOD_GRAD= GREEN_GAUSS
CONV_NUM_METHOD_FLOW= ROE
% Roe entropy fix, as a fraction of (|u| + c) below which an eigenvalue is
% floored. This was 0.2 -- two HUNDRED times SU2's own default of 0.001 -- which
% means every acoustic and entropy wave in the domain was given at least 20% of
% (|u| + c) worth of numerical dissipation whether it needed it or not.
%
% 0.01 keeps an order of magnitude over SU2's default, so a strong bow shock is
% still protected from the carbuncle instability, without dissipating the whole
% domain. Measured on the M=2 cone it is NEUTRAL, not an improvement: 8.89% at
% 0.01 against 9.06% at 0.2, which is noise. It is kept because 0.2 has no
% evidence behind it and adds dissipation everywhere, not because it was shown
% to buy accuracy. Re-run the cone if you change it.
ENTROPY_FIX_COEFF= 0.01
% 2nd-order MUSCL reconstruction — required to resolve transonic/supersonic
% shocks (and thus wave drag). 1st-order smears shocks and kills drag rise.
% Venkatakrishnan-Wang limiter keeps it monotone near discontinuities.
MUSCL_FLOW= YES
SLOPE_LIMITER_FLOW= VENKATAKRISHNAN_WANG
VENKAT_LIMITER_COEFF= 0.15

{turb_numerics}

TIME_DISCRE_FLOW= EULER_IMPLICIT
{turb_time_discre}
CFL_NUMBER= 1.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.2, 1.5, 0.5, 40.0 )

LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 20

% ── Convergence ───────────────────────────────────────────
ITER= {max_iter}
% CONV_FIELD is what selects the criteria, and it was never set — so SU2 fell
% back to its default of RMS_DENSITY alone and the CONV_CAUCHY_* pair below
% sat inert for every run ever made. Listing DRAG activates them.
%
% SU2 requires ALL listed fields to converge, which is the behaviour we want:
% the density residual alone is a poor stopping rule here. Residuals are
% DIMENSIONAL (see REF_DIMENSIONALIZATION), so their starting magnitude
% depends on the flow scale rather than on the solution quality — a measured
% M=0.8 case began at rms[Rho]=-3.04 and rms[RhoE]=+2.44, which makes a fixed
% -6 floor mean "3 decades" for one equation and "8 decades" for another.
% Requiring the drag coefficient to also go stationary pins convergence to the
% quantity the polars actually report.
CONV_FIELD= (RMS_DENSITY, DRAG)
CONV_RESIDUAL_MINVAL= -{conv_order}
CONV_STARTITER= {conv_startiter}
CONV_CAUCHY_ELEMS= 100
CONV_CAUCHY_EPS= 1E-6

% ── I/O ───────────────────────────────────────────────────
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SOLUTION_FILENAME= solution_flow.dat
% Force the restart output name so it matches what the sweep marching copies
% into the next point's solution_flow.dat. Without this SU2 defaults to
% restart.dat, the warm-start lookup misses, and every point cold-starts.
RESTART_FILENAME= restart_flow.dat
TABULAR_FORMAT= CSV

% NOTE: DRAG_PRESSURE / DRAG_VISCOUS are not valid SU2 history fields (they
% never appear in history.csv) — the pressure/friction split is integrated
% from surface_flow.vtu in parse_results() instead.
HISTORY_OUTPUT= ITER, RMS_DENSITY, RMS_ENERGY, {turb_hist_fields} LIFT, DRAG, MOMENT_Y, FORCE_X, FORCE_Y, FORCE_Z
CONV_FILENAME= history

VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ= 250

% Volume fields: solution + derived.
% VORTICITY and LAMBDA2 were requested here for a long time but this SU2
% build emits neither — verified absent from flow.vtu on every run. They are
% recomputed from the velocity gradient in cfd/post_processing.py
% (compute_derived_fields) instead, so keeping them in the list only made the
% config claim more than it delivered. Q_CRITERION *is* written and is kept.
VOLUME_OUTPUT= COORDINATES, SOLUTION, PRIMITIVE, PRESSURE_COEFFICIENT, MACH, Q_CRITERION, Y_PLUS
% Surface fields: wall quantities for post-processing
% SURFACE_OUTPUT is not supported in all SU2 versions; surface VTK inherits from VOLUME_OUTPUT
% SURFACE_OUTPUT= COORDINATES, SOLUTION, PRESSURE_COEFFICIENT, SKIN_FRICTION, Y_PLUS
"""



def _wall_spacing_from_su2_mesh(mesh_path: Path) -> Optional[dict]:
    """Measure how far the first cell centroid sits off ``rocket_wall``.

    Returns ``{"normal": m, "tangential": m, "source": "prism"|"tet"}``, or None
    if the file cannot be parsed — this is diagnostics, so it must never take a
    run down with it.

    The distinction matters because y+ is a wall-NORMAL quantity and the two
    mesh types put the first cell in completely different places:

    * With a prism stack (VTK type 13), the wall-normal spacing is the first
      layer's own thickness, measured here as the distance between the wall
      triangle and the matching triangle on the far face of its prism. It is
      typically thousands of times smaller than the triangle's own edges.
    * On a tet-only mesh there is no wall-normal direction to measure, and the
      first cell centroid is roughly half a facet off the wall, so the surface
      triangle's median edge length is the only available proxy.

    Reading the triangle edge in both cases — as this did while prisms were
    impossible — reports a tet-mesh y+ for a prism mesh, i.e. it would claim the
    wall is unresolved on exactly the meshes that finally resolve it.
    """
    try:
        import numpy as np
        with open(mesh_path, "r") as fh:
            lines = fh.read().split("\n")

        points = None
        tris: list[tuple[int, int, int]] = []
        prisms: list[tuple[int, ...]] = []
        i = 0
        while i < len(lines):
            head = lines[i].strip()
            if head.startswith("NPOIN="):
                n = int(head.split("=")[1].split()[0])
                points = np.empty((n, 3), dtype=float)
                for j in range(n):
                    points[j] = [float(v) for v in lines[i + 1 + j].split()[:3]]
                i += n
            elif head.startswith("NELEM="):
                n = int(head.split("=")[1].split()[0])
                for j in range(n):
                    parts = lines[i + 1 + j].split()
                    if parts and parts[0] == "13":      # VTK wedge / prism
                        prisms.append(tuple(int(v) for v in parts[1:7]))
                i += n
            elif head.startswith("MARKER_TAG=") and "rocket_wall" in head:
                n = int(lines[i + 1].split("=")[1])
                for j in range(n):
                    parts = lines[i + 2 + j].split()
                    if parts and parts[0] == "5":       # VTK triangle
                        tris.append((int(parts[1]), int(parts[2]), int(parts[3])))
                i += n + 1
            i += 1

        if points is None or not tris:
            return None
        tri = points[np.asarray(tris)]
        edges = np.concatenate([
            np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
            np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
            np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
        ])
        tangential = float(np.median(edges))

        # First-layer prisms are the ones whose bottom face is a wall triangle.
        # Match on the node SET, because the prism's bottom face may be wound
        # opposite to the marker triangle.
        if prisms:
            wall_faces = {frozenset(t) for t in tris}
            first: list[float] = []
            for pr in prisms:
                bot, top = pr[:3], pr[3:]
                if frozenset(bot) in wall_faces:
                    first.append(float(np.linalg.norm(
                        points[list(top)].mean(axis=0)
                        - points[list(bot)].mean(axis=0)
                    )))
                elif frozenset(top) in wall_faces:
                    first.append(float(np.linalg.norm(
                        points[list(bot)].mean(axis=0)
                        - points[list(top)].mean(axis=0)
                    )))
            if first:
                return {
                    "normal": float(np.median(first)),
                    "tangential": tangential,
                    "source": "prism",
                }
            logger.warning(
                "Mesh contains prisms but none sit on rocket_wall — falling back "
                "to the tet-mesh y+ estimate, which will read far too high."
            )

        return {"normal": tangential, "tangential": tangential, "source": "tet"}
    except Exception as exc:
        logger.debug(f"Wall-spacing probe failed on {mesh_path}: {exc}")
        return None


def _integrate_surface_forces(
    surf_vtk: Path,
    p_inf: float,
    q_inf: float,
    ref_area: float,
    aoa_deg: float,
) -> Optional[dict]:
    """Integrate wall pressure and skin friction from SU2's surface VTK.

    Returns the real pressure/friction drag split (wind-axis) plus the mean
    wall y+, or None if the file/fields are unavailable. SU2 has no valid
    history fields for the split, so this integration is the only honest
    source — verified to reproduce SU2's own CFx/CFz/CMy to 4 decimals.
    Skin_Friction_Coefficient is stored dimensionless (τ = Cf·q∞).
    """
    try:
        import numpy as np
        import pyvista as pv
        if not Path(surf_vtk).is_file():
            return None
        mesh = pv.read(str(surf_vtk))
        try:
            surf = mesh.extract_surface(algorithm=None)
        except TypeError:
            surf = mesh.extract_surface()
        if "Pressure" not in surf.point_data:
            return None
        surf = surf.compute_normals(
            cell_normals=True, point_normals=False, consistent_normals=False
        )
        cell = surf.point_data_to_cell_data()
        area = np.asarray(surf.compute_cell_sizes()["Area"], dtype=float)
        normals = np.asarray(surf.cell_data["Normals"], dtype=float)

        # Wind-axis drag direction (AoA rotates the freestream in x-z).
        a = math.radians(aoa_deg)
        drag_dir = np.array([math.cos(a), 0.0, math.sin(a)])

        # Pressure force on the body: dF = -(p - p_inf) * n_outward * dA
        p = np.asarray(cell["Pressure"], dtype=float)
        f_pressure = (-(p - p_inf)[:, None] * normals * area[:, None]).sum(axis=0)

        out = {
            "cd_pressure": float(f_pressure @ drag_dir) / (q_inf * ref_area),
            "cd_friction": 0.0,
            "yplus_mean": 0.0,
        }

        if "Skin_Friction_Coefficient" in cell.array_names:
            cf = np.asarray(cell["Skin_Friction_Coefficient"], dtype=float)
            if cf.ndim == 2 and cf.shape[1] >= 3:
                f_friction = (cf * q_inf * area[:, None]).sum(axis=0)
                out["cd_friction"] = float(f_friction @ drag_dir) / (q_inf * ref_area)
        if "Y_Plus" in cell.array_names:
            # Area-weighted, not a plain mean. Wall cell areas span 3.3e4 : 1 on
            # a measured mesh with the slivers on the nose tip, so an unweighted
            # mean describes the tip: measured 241 unweighted against an
            # area-weighted median of 338 on the same surface. y+ is used to
            # decide whether the boundary layer is resolved, so reading it low
            # is the dangerous direction.
            yp = np.asarray(cell["Y_Plus"], dtype=float)
            good = np.isfinite(yp) & (yp > 0) & np.isfinite(area) & (area > 0)
            if good.any():
                out["yplus_mean"] = float(
                    np.sum(yp[good] * area[good]) / np.sum(area[good]))
        return out
    except Exception as e:
        logger.warning(f"Surface force integration failed: {e}")
        return None


class SU2Solver(CFDSolver):
    """SU2 CFD solver backend for K2 AeroSim."""

    def __init__(self, config: CFDConfig):
        super().__init__(config)
        self._mesh_path: Optional[Path] = None
        self._config_path: Optional[Path] = None
        self._proc = None            # live SU2_CFD subprocess (set in _execute, for stop())
        # Warm start: when True, generate_case() writes RESTART_SOL= YES so SU2
        # initialises from solution_flow.dat (a converged neighbour) instead of
        # freestream. Same converged answer, far fewer iterations. Used by the
        # sweep to march from one flow point to the next. Default False keeps the
        # single-solve path cold-starting exactly as before.
        self.warm_start: bool = False
        # OpenMP threads for this solve. None = auto (cores-1). The parallel
        # sweep sets this so several points can run at once, each on a slice of
        # the cores instead of all points fighting over every core.
        self.omp_threads: Optional[int] = None
        self._su2_exe = _find_su2()
        if self._su2_exe:
            logger.info(f"SU2 found at: {self._su2_exe}")
        else:
            logger.warning("SU2_CFD binary not found. Meshing will work but solver won't run.")

    # ── Mesh generation (delegates to cfd.meshing) ───────────────────────────

    def generate_mesh(self) -> Path:
        """Generate SU2 mesh via Gmsh. Returns path to .su2 mesh file."""
        from cfd.meshing import build_wind_tunnel_mesh
        cfg = self.config
        out_mesh = cfg.work_dir / "rocket_mesh.su2"
        build_wind_tunnel_mesh(
            stl_path=cfg.geometry_stl,
            output_path=out_mesh,
            refinement=cfg.mesh_refinement,
            domain_length_scale=cfg.domain_length_scale,
            domain_radius_scale=cfg.domain_radius_scale,
            bl_prisms=cfg.bl_prisms,
            bl_layers=cfg.boundary_layer_layers,
            bl_growth=cfg.boundary_layer_growth,
            bl_first_height=cfg.bl_first_height,
            bl_tessellation=cfg.bl_tessellation,
            bl_max_apex_radius=cfg.bl_max_apex_radius,
            geometry_dict=cfg.geometry_dict,   # exact dims if available
            custom_wall_size=cfg.custom_wall_size,
            target_element_count=cfg.target_element_count,
            external_cad=cfg.external_cad,     # arbitrary imported body
            flow_axis=cfg.flow_axis,
            cad_info=cfg.cad_info,
            cad_units=cfg.cad_units,
            cad_wrap=cfg.cad_wrap,
            cad_wrap_resolution=cfg.cad_wrap_resolution,
            cad_curvature_elements=cfg.cad_curvature_elements,
        )
        self._mesh_path = out_mesh
        logger.info(f"Mesh written to {out_mesh}")
        return out_mesh


    # ── Configuration file generation ────────────────────────────────────────

    def _reference_values(self) -> tuple[float, float]:
        """Return ``(ref_area_m2, ref_length_m)`` for the force coefficients.

        Extracted from generate_case() so anything that needs the reference
        length before the solve — mesh sizing, the pre-run y+ prediction —
        reads the same value the coefficients are normalised by, rather than
        recomputing its own and drifting by the ratio of the two.
        """
        cfg = self.config
        ref_length = 1.0
        ref_area = 0.1

        # Priority 0: external CAD — measured projected frontal area and
        # flow-wise bbox extent. An arbitrary body has no "body diameter" to
        # infer a reference from, so it is measured off the tessellation.
        if cfg.external_cad and cfg.cad_info:
            _ci = cfg.cad_info
            _fa = float(_ci.get("frontal_area", 0.0) or 0.0)
            if _fa > 0:
                ref_area = _fa
            else:
                logger.warning("CAD frontal area is zero — keeping the default "
                               "reference area; coefficients will be off.")
            ref_length = float(_ci.get("length", 1.0) or 1.0)
            logger.info(f"Reference values from imported CAD: "
                        f"A={ref_area:.6f} m² (projected frontal)  "
                        f"L={ref_length:.4f} m (flow-axis extent)")
        # Priority 1: Use exact geometry dict parameters (avoids fin span bug)
        elif cfg.geometry_dict and "max_diameter" in cfg.geometry_dict:
            max_d = cfg.geometry_dict["max_diameter"]
            ref_area = math.pi * (max_d / 2.0) ** 2
            ref_length = cfg.geometry_dict.get("length", 1.0)
            logger.info(f"Using exact max diameter ({max_d*1000:.1f} mm) for reference area.")
        # Priority 2: Fallback to STL bounding box (may include fins)
        elif cfg.geometry_stl and cfg.geometry_stl.is_file():
            try:
                import pyvista as pv
                m = pv.read(str(cfg.geometry_stl))
                bounds = m.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
                # K2 STL: Z-axis is the rocket longitudinal axis
                # Use Z-span as ref_length (not max of all dims, which
                # would include fin span and overestimate)
                z_span = abs(bounds[5] - bounds[4])
                x_span = abs(bounds[1] - bounds[0])
                y_span = abs(bounds[3] - bounds[2])
                ref_length = max(z_span, x_span, y_span)  # longest = rocket axis
                # Cross-section radius: use the SMALLER of X/Y spans
                # (fins make the larger span unreliable for body diameter)
                cross_spans = sorted([x_span, y_span, z_span])
                body_diam = cross_spans[0]  # smallest span = body diameter
                ref_area = math.pi * (body_diam / 2) ** 2
                logger.info(f"STL bounding box: X={x_span:.4f} Y={y_span:.4f} "
                            f"Z={z_span:.4f} → ref_L={ref_length:.4f} m, "
                            f"ref_A={ref_area:.6f} m² (body_d={body_diam:.4f} m)")
            except Exception as e:
                logger.warning(f"Could not read STL bounds: {e}. Using defaults.")

        # Explicit user overrides win over every automatic source above. Applied
        # last (and independently) so setting only one of the two keeps the
        # measured value for the other.
        if cfg.ref_area_override and cfg.ref_area_override > 0:
            ref_area = float(cfg.ref_area_override)
            logger.info(f"Reference area overridden by user: {ref_area:.6f} m²")
        if cfg.ref_length_override and cfg.ref_length_override > 0:
            ref_length = float(cfg.ref_length_override)
            logger.info(f"Reference length overridden by user: {ref_length:.4f} m")

        return ref_area, ref_length

    def generate_case(self) -> Path:
        """Write the SU2 .cfg file with correct ISA conditions."""
        cfg = self.config
        P, T, rho = isa_conditions(cfg.altitude_m)
        a = math.sqrt(1.4 * 287.05 * T)          # speed of sound
        V_inf = cfg.mach * a                       # freestream velocity
        mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)  # Sutherland

        # Reference values
        ref_area, ref_length = self._reference_values()

        Re = rho * V_inf * ref_length / mu
        q_inf = 0.5 * rho * V_inf ** 2
        conv_order = int(-math.log10(cfg.convergence_tolerance))

        # Warm-start (sweep marching): init from a converged neighbour solution.
        # Lower CONV_STARTITER so the fast-converging restart isn't forced to
        # grind out the full 50-iter monitoring floor — the residual/Cauchy
        # criteria (CONV_RESIDUAL_MINVAL unchanged) still gate the stop, so the
        # converged answer is identical; only the iteration count drops.
        restart_sol = "YES" if self.warm_start else "NO"
        conv_startiter = 10 if self.warm_start else 50

        # Moment origin at the nose tip. Mesh frame (cfd/meshing.py): nose tip
        # at x=0, nozzle at x=total_L, flow along +X — so the nose is at x=0,
        # NOT at x=ref_length. Verified against SU2 v8.5: with origin at x=0
        # the pitch moment CMy gives CP-from-nose = -(CMy/CN)*L at a plausible
        # station (0.63L for a finned test rocket); CMz is ~15x smaller noise.
        moment_x = 0.0   # nose tip location in CFD x-axis

        # Store flow metadata for results
        self._flow_meta = {
            "reynolds": Re, "dynamic_pressure": q_inf,
            "ref_length": ref_length, "ref_area": ref_area,
            "v_inf": V_inf, "rho": rho, "T": T, "P": P, "mu": mu,
            "moment_x": moment_x,
        }

        # Turbulence model configuration
        turb_key = cfg.turbulence_model if cfg.turbulence_model in _TURB_MODEL_MAP else "SST"
        turb_cfg = _TURB_MODEL_MAP[turb_key]
        is_viscous = turb_cfg["solver"] != "EULER"
        is_rans = turb_cfg["solver"] == "RANS"

        # Wall BC: Euler uses slip wall, viscous uses no-slip heatflux.
        #
        # NO wall functions. This is a measured decision, not caution — do not
        # add MARKER_WALL_FUNCTIONS here without re-running the numbers below.
        #
        # STANDARD_WALL_FUNCTION diverges (T_Wall < 0 -> NaN) from a freestream
        # cold start. From a CONVERGED restart it does run to completion, which
        # looks like the fix and is not: SU2's wall-coefficient Newton solve
        # failed on 5554 of 6318 wall points (88%) on a measured M=0.8 case. On
        # those points SU2 pins y+ at exactly 30.0 -- the value is a fallback,
        # not a solution -- and skin friction collapses to a median 3.1e-8,
        # i.e. no wall shear at all. On the 12% where it did converge Cf came
        # out at 1.46e-3 against a flat-plate 1.9e-3, so the model is right
        # where it works and silently absent where it does not. Total drag
        # rose 0.079 -> 0.137 and the wall-temperature violation fell from
        # 19.1% to 4.2%, both of which read like improvements and are partly
        # just the missing viscous heating on 88% of the surface.
        #
        # Tuning it makes it worse, not better: WALLMODEL_MAXITER= 1000 with
        # WALLMODEL_RELFAC= 0.1 and WALLMODEL_MINYPLUS= 2.0 took the failure
        # rate to 6252/6318 (99.0%) over 300 restart iterations, drifting up
        # rather than down as the run advanced. It also took the surviving
        # points with it: the 66 still solving freely ended at a median Cf of
        # 1.4e-6, against 1.46e-3 for the untuned run's free points. More
        # Newton iterations at lower relaxation do not recover a wall function
        # whose first cell is outside the band it can invert.
        #
        # Root cause is the mesh, not the model. A wall function is only valid
        # for 30 < y+ < 300; the first cell here sits near y+ 3500 (see
        # predict_wall_yplus below), far outside the band the Newton solve can
        # invert. Both a low-Re model and a wall function fail on the same
        # cause -- no prism layers.
        #
        # That blocker was re-measured on Gmsh 4.15.2 and is narrower than it
        # was written up as. Extrusion is not what fails; leaving a valid mesh
        # behind is. The prisms come out attached and generate(3) completes, but
        # the tets fill the boundary-layer region as well, so the wall ends up
        # with cells on both sides and SU2 cannot converge on it at all.
        # Rebuilding the volume against the stack is where Gmsh actually stops
        # ("non-manifold quad boundaries not supported yet"). See the
        # cfd/meshing.py module docstring. Until that is solved, wall shear from
        # RANS is not trustworthy on this mesh and total drag should come from
        # the hybrid Euler + analytic-friction mode instead.
        if is_viscous:
            wall_bc = "MARKER_HEATFLUX= ( rocket_wall, 0.0 )"
        else:
            wall_bc = "MARKER_EULER= ( rocket_wall )"

        # Turbulence numerics (only for RANS)
        turb_numerics = "CONV_NUM_METHOD_TURB= SCALAR_UPWIND\nMUSCL_TURB= NO" if is_rans else ""
        turb_time_discre = "TIME_DISCRE_TURB= EULER_IMPLICIT" if is_rans else ""
        turb_hist_fields = "RMS_TKE, " if is_rans else ""

        # ── Predict wall resolution before spending minutes on the solve ─────
        # A tet-only mesh cannot put a cell close enough to the wall for a
        # low-Re turbulence model, but nothing downstream says so: the Y+ view
        # renders a smooth field and the polars report a confident Cd whether
        # or not the boundary layer was resolved. Say it up front instead.
        # Verdict recorded on the solver, not just logged: parse_results puts it
        # on the CFDResult so the UI, the PDF export and the CSV all state it
        # without having to re-derive it. An inviscid run has no skin friction
        # by construction and is not "unresolved" -- it is answered by the
        # analytic build-up instead, so it starts clean.
        self._wall_verdict = (True, "")
        if is_viscous and self._mesh_path and Path(self._mesh_path).is_file():
            spacing = _wall_spacing_from_su2_mesh(Path(self._mesh_path))
            if spacing:
                from cfd.boundary_layer import predict_wall_yplus
                yp = predict_wall_yplus(
                    wall_spacing=spacing["normal"], velocity=V_inf, density=rho,
                    viscosity=mu, ref_length=ref_length,
                )
                where = ("the first prism layer" if spacing["source"] == "prism"
                         else "the wall facet size (tet-only mesh)")
                msg = (
                    f"Predicted wall y+ ≈ {yp['y_plus']:.1f} ({yp['regime']}) "
                    f"from a {spacing['normal']*1e6:.1f} µm wall-normal spacing, "
                    f"measured off {where}. "
                    f"y+ = 1 would need {yp['spacing_for_yplus_1']*1e6:.2f} µm."
                )
                if yp["y_plus"] > 30:
                    _w = (
                        f"Boundary layer UNRESOLVED (predicted wall y+ "
                        f"{yp['y_plus']:.0f}, needs y+ < 5 with no wall model "
                        f"available). Skin friction and wall heat transfer from "
                        f"this run are not physical — a measured SST solution "
                        f"on this mesh family returned Cf 3.2e-6 against a "
                        f"flat-plate 1.9e-3, i.e. ~600x low, so total drag is "
                        f"missing essentially all of its friction component. "
                        f"Pressure drag, Cp, lift and CP are still usable. For "
                        f"total drag use Euler + flat-plate friction."
                    )
                    self._wall_verdict = (False, _w)
                    logger.warning(msg + " " + _w)
                elif 5 <= yp["y_plus"] <= 30:
                    _w = (
                        f"Wall in the BUFFER LAYER (predicted y+ "
                        f"{yp['y_plus']:.1f}), where neither wall resolution "
                        f"nor a wall function is valid. Skin friction from this "
                        f"run is not trustworthy."
                    )
                    self._wall_verdict = (False, _w)
                    logger.warning(msg + " " + _w)
                else:
                    logger.info(msg)
            else:
                self._wall_verdict = (
                    False,
                    "Wall spacing could not be measured from the mesh, so the "
                    "boundary-layer resolution is unknown. Treat skin friction "
                    "as unverified."
                )
                logger.warning(self._wall_verdict[1])

        config_text = _SU2_CONFIG_TEMPLATE.format(
            solver_type=turb_cfg["solver"],
            turb_model_lines=turb_cfg["turb_line"],
            mach=cfg.mach,
            aoa=cfg.angle_of_attack_deg,
            sideslip=cfg.sideslip_angle_deg,
            pressure=round(P, 2),
            temperature=round(T, 4),
            reynolds=round(Re, 2),
            ref_length=round(ref_length, 5),
            ref_area=round(ref_area, 6),
            moment_x=round(moment_x, 5),
            max_iter=cfg.max_iterations,
            conv_order=conv_order,
            restart_sol=restart_sol,
            conv_startiter=conv_startiter,
            mesh_file=self._mesh_path.name if self._mesh_path else "rocket_mesh.su2",
            wall_bc=wall_bc,
            turb_numerics=turb_numerics,
            turb_time_discre=turb_time_discre,
            turb_hist_fields=turb_hist_fields,
        )

        config_path = cfg.work_dir / "su2_config.cfg"
        # Strip non-ASCII chars (box-drawing etc.) before writing — SU2 is ASCII-only
        ascii_text = config_text.encode("ascii", errors="ignore").decode("ascii")
        config_path.write_text(ascii_text, encoding="ascii")
        self._config_path = config_path
        logger.info(f"SU2 config written: Mach={cfg.mach}, Alt={cfg.altitude_m}m, "
                     f"AoA={cfg.angle_of_attack_deg}°, Model={turb_key}")
        return config_path

    # ── Run solver ───────────────────────────────────────────────────────────

    def run(self):
        """Run SU2_CFD. Generator yielding (iteration, rms_density) tuples."""
        if not self._su2_exe:
            raise RuntimeError(
                "SU2_CFD executable not found.\n"
                "Please place SU2_CFD.exe inside the K2 bin/ folder."
            )
        if not self._config_path:
            raise RuntimeError("Call generate_case() before run().")

        logger.info(f"Launching SU2: {self._su2_exe}")

        # Write a log file next to the config for full diagnostics
        log_path = self.config.work_dir / "su2_run.log"

        # Resolve MPI rank count: explicit n_cores>0, else auto = all cores.
        import os
        global _MPI_BUILD_OK
        # Auto leaves one core free so the Qt UI stays responsive during solves.
        n_cores = (self.config.n_cores if self.config.n_cores > 0
                   else max(1, (os.cpu_count() or 1) - 1))
        serial_cmd = [str(self._su2_exe), str(self._config_path.name)]
        mpi = _find_mpi() if n_cores > 1 else None

        # Attempt parallel unless a prior probe proved SU2 is a serial build.
        attempt_mpi = (mpi is not None and n_cores > 1 and _MPI_BUILD_OK is not False)

        if attempt_mpi:
            cmd = [str(mpi), "-n", str(n_cores), *serial_cmd]
            logger.info(f"Launching SU2 on {n_cores} ranks via {mpi.name}")
            self._emit_log(f"Running SU2 on {n_cores} cores via {mpi.name}…")
            yield from self._execute(cmd, log_path)
            # ── MPI-build probe ────────────────────────────────────────────
            # A real MPI build prints the SU2 banner only from rank 0 → once.
            # A serial build launched under mpiexec spawns N independent copies,
            # each printing the banner and clobbering the others' output → banner
            # appears N times and results are corrupt. Detect, cache, re-run serial.
            if self._exec_banner_count > 1:
                _MPI_BUILD_OK = False
                logger.error(
                    f"MPI probe FAILED: SU2 banner seen {self._exec_banner_count}x "
                    f"under mpiexec → serial build, parallel output invalid. "
                    f"Re-running serial."
                )
                self._emit_log(
                    f"MPI probe failed: SU2 is a serial build (launched "
                    f"{self._exec_banner_count} copies). Multi-core unavailable — "
                    f"re-running on 1 core. Results valid."
                )
                self._emit_progress(-1, 0.0)
                yield from self._execute(serial_cmd, log_path)   # valid overwrite
            elif _MPI_BUILD_OK is None:
                _MPI_BUILD_OK = True
                logger.info("MPI probe OK: SU2 is an MPI build — parallel runs valid.")
                self._emit_log(f"✓ MPI build confirmed — running on {n_cores} cores.")
        else:
            # No MPI launch — but the bundled SU2 is an OpenMP build, so a single
            # process still threads across cores. Report the real OMP thread count
            # instead of implying serial execution.
            omp_n = (self.omp_threads if (self.omp_threads and self.omp_threads > 0)
                     else (self.config.n_cores if self.config.n_cores > 0
                           else max(1, (os.cpu_count() or 1) - 1)))
            if mpi is None and n_cores > 1:
                logger.info(f"No MPI launcher — single rank, {omp_n} OpenMP threads.")
                self._emit_log(f"Running SU2: 1 rank × {omp_n} OpenMP threads "
                               f"(no MPI launcher in bin/PATH).")
            elif _MPI_BUILD_OK is False:
                self._emit_log(f"Running SU2: 1 rank × {omp_n} OpenMP threads "
                               f"(SU2 is not an MPI build).")
            else:
                self._emit_log(f"Running SU2: 1 rank × {omp_n} OpenMP threads.")
            yield from self._execute(serial_cmd, log_path)

        logger.info("SU2 run complete.")

    def _execute(self, cmd: list, log_path: Path):
        """Run one SU2_CFD invocation, streaming (iteration, rms_density).

        Sets ``self._proc`` (so SweepThread.stop() can kill the live process)
        and ``self._exec_banner_count`` (SU2 start-up banners seen — used by the
        MPI-build probe in run()). Raises on a non-zero exit code.
        """
        # OMP_NUM_THREADS fallback: an OpenMP-built SU2 (win64-omp) threads a
        # single process across cores when this is set. Default it only if the
        # user/system hasn't, so an explicit value always wins. Under mpiexec
        # the ranks already provide parallelism, so pin threads to 1 there to
        # avoid ranks×threads oversubscription; on the serial/OMP launch path
        # give it cores-1 (leaving one for the UI), matching run()'s auto rule.
        env = os.environ.copy()
        is_mpi_launch = (
            Path(cmd[0]).name.lower() != Path(str(self._su2_exe)).name.lower()
        )
        # Explicit per-solve override (set by the parallel sweep) always wins,
        # even over an inherited OMP_NUM_THREADS, so concurrent points each get
        # their assigned thread slice.
        if self.omp_threads is not None and self.omp_threads > 0 and not is_mpi_launch:
            env["OMP_NUM_THREADS"] = str(self.omp_threads)
        elif "OMP_NUM_THREADS" not in env:
            if is_mpi_launch:
                env["OMP_NUM_THREADS"] = "1"
            else:
                n = (self.config.n_cores if self.config.n_cores > 0
                     else max(1, (os.cpu_count() or 1) - 1))
                env["OMP_NUM_THREADS"] = str(n)

        proc = subprocess.Popen(
            cmd,
            cwd=str(self.config.work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=_NO_WINDOW,
        )
        self._proc = proc

        # SU2 v8.5 screen output format:
        # |  Inner_Iter|    rms[Rho]|   rms[RhoU]|   rms[RhoV]|   rms[RhoE]|
        # |           0|   -3.962618|   -4.348516|   ...
        rms_pattern = re.compile(
            r"\|\s*(\d+)\s*\|\s*([\-\d.eE+]+)\s*\|"
        )

        # Keywords that indicate important SU2 status lines worth showing in UI
        _SHOW = {"error", "warning", "reading", "writing", "setting",
                 "direct", "problem", "memory", "mesh", "marker",
                 "initializ", "failed", "cannot", "unknown"}

        self._exec_banner_count = 0
        with open(log_path, "w", encoding="utf-8") as log_f:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                log_f.write(raw_line)

                ll = line.lower()
                # Count SU2 start-up banners. Header line is stable across
                # v7/v8: "This is SU2, the open-source CFD code."
                if "su2" in ll and "open-source" in ll:
                    self._exec_banner_count += 1

                # Show important diagnostic lines in the UI log
                if any(kw in ll for kw in _SHOW):
                    self._emit_progress(-1, 0.0)   # signal a log line
                    logger.info(f"[SU2] {line}")

                m = rms_pattern.match(line)
                if m:
                    try:
                        it = int(m.group(1))
                        rms = float(m.group(2))
                        self._emit_progress(it, rms)
                        yield it, rms
                        # Also log every 10 iterations so the console shows it's alive
                        if it % 10 == 0:
                            logger.info(f"[SU2] {line}")
                    except ValueError:
                        pass

        proc.wait()
        if proc.returncode != 0:
            # Tail the log file to get the actual error
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace")
                last_lines = "\n".join(tail.splitlines()[-30:])
                logger.error(f"SU2 log tail:\n{last_lines}")
            except Exception:
                pass
            raise RuntimeError(
                f"SU2_CFD failed (exit code {proc.returncode}).\n"
                f"See {log_path} for full output."
            )

    # ── Parse results ─────────────────────────────────────────────────────────

    def parse_results(self) -> CFDResult:
        """Parse SU2 history.csv and return a CFDResult."""
        result = CFDResult()
        history_file = self.config.work_dir / "history.csv"
        vol_vtk = self.config.work_dir / "flow.vtu"
        surf_vtk = self.config.work_dir / "surface_flow.vtu"

        if not history_file.is_file():
            logger.error("history.csv not found — solver may not have run.")
            return result

        try:
            rows = []
            with open(history_file, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                # Strip surrounding whitespace from all field names
                reader.fieldnames = [k.strip().strip('"') for k in reader.fieldnames]
                for row in reader:
                    # Create a clean version of each row with stripped keys
                    clean = {k.strip().strip('"'): v.strip() for k, v in row.items()}
                    rows.append(clean)

            if not rows:
                logger.warning("history.csv is empty.")
                return result

            last = rows[-1]

            def _get(key):
                """Try case-insensitive match for SU2 column names."""
                key_lower = key.lower()
                for k, v in last.items():
                    if k.lower().strip() == key_lower:
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            return 0.0
                return 0.0

            # Tail window for coefficient averaging. On this tet-only mesh the
            # density residual limit-cycles (unsteady base flow under steady
            # RANS) while the force coefficients plateau — so the last row can
            # land anywhere in the cycle. Averaging the tail gives the
            # cycle-mean force; the window spread doubles as a force-convergence
            # check independent of the (never-satisfied) residual floor.
            # Never let the window reach back into the startup transient: on a
            # fast-converging case (small clean body, warm start, imported CAD)
            # the whole run can be ~50 iterations, and the first few carry
            # order-of-magnitude garbage — a sphere that settles at CD=0.244
            # peaks at CD=7.1 around iteration 4, and a flat 50-row mean
            # reported 0.877. Capping at half the run leaves long solves
            # (thousands of iterations) on the original 50-row window.
            tail_n = max(1, min(50, len(rows) // 2))

            def _get_tail(key):
                """Mean of a column over the last tail_n rows (case-insensitive)."""
                key_lower = key.lower()
                col = None
                for k in rows[-1].keys():
                    if k.lower().strip() == key_lower:
                        col = k
                        break
                if col is None:
                    return 0.0, 0.0, 0.0
                vals = []
                for row in rows[-tail_n:]:
                    try:
                        vals.append(float(row[col]))
                    except (ValueError, TypeError, KeyError):
                        pass
                if not vals:
                    return 0.0, 0.0, 0.0
                mean = sum(vals) / len(vals)
                return mean, min(vals), max(vals)

            # Core coefficients. Pitch moment is CMy: SU2 applies AoA as a
            # rotation about the Y axis (freestream tilts in the x-z plane),
            # so the AoA-induced normal force is along Z and its moment is
            # about Y. CMz is the (near-zero) yaw component — do not use it.
            cd_mean, cd_min, cd_max = _get_tail("CD")
            # Signed, not abs(). A negative total drag is not a sign error to be
            # tidied away -- it means the surface solution or the wall normals
            # are wrong, and abs() turned that into a plausible-looking number.
            # cd_pressure was de-abs'd for this reason already; cd was missed.
            result.cd = cd_mean
            if cd_mean < 0:
                logger.error(
                    f"Integrated total drag is NEGATIVE (Cd={cd_mean:.5f}). "
                    f"The solution is non-physical -- check convergence and the "
                    f"wall normals. Reported as-is rather than as its magnitude."
                )
            result.cl = _get_tail("CL")[0]
            result.cm = _get_tail("CMy")[0]
            result.iterations = len(rows)
            # Forces stationary ⇔ CD spread over the tail window is small
            # relative to its mean. Used below to grant convergence when the
            # residual limit-cycles above the floor.
            forces_stationary = (
                len(rows) > tail_n
                and abs(cd_mean) > 1e-6
                and (cd_max - cd_min) / abs(cd_mean) < 0.005
            )

            # Drag decomposition — integrate the real pressure/friction split
            # from the surface VTK (SU2 writes no valid history fields for it;
            # the old DRAG_PRESSURE/DRAG_VISCOUS lookups always returned 0 and
            # the split silently fell back to a fixed 55/30/15 guess).
            _meta = getattr(self, '_flow_meta', {})
            split = _integrate_surface_forces(
                surf_vtk,
                p_inf=_meta.get("P", 101325.0),
                q_inf=_meta.get("dynamic_pressure", 1.0),
                ref_area=_meta.get("ref_area", 0.1),
                aoa_deg=self.config.angle_of_attack_deg,
            )
            if split is not None:
                # Signed, not abs(): cd_base is computed by the same integral
                # over a subset of the same cells, so folding the sign here
                # would break cd_base + cd_forebody_pressure == cd_pressure.
                # A negative pressure drag is physically wrong anyway — it means
                # the surface solution or the wall normals are bad, and hiding
                # that behind abs() turned it into a plausible-looking number.
                result.cd_pressure = split["cd_pressure"]
                result.cd_friction = abs(split["cd_friction"])
                result.yplus_mean = split["yplus_mean"]
                if result.cd_pressure < 0:
                    logger.warning(
                        f"Integrated pressure drag is negative "
                        f"(Cd_p={result.cd_pressure:.5f}) — check the wall "
                        f"normals and that the solution converged. The base/"
                        f"wave split below inherits the problem."
                    )
            else:
                result.cd_pressure = 0.0
                result.cd_friction = 0.0
                logger.warning(
                    "Surface force integration unavailable — the pressure/"
                    "friction split is reported as zero rather than guessed."
                )

            # ── Base and wave drag: both solved, neither apportioned ──────────
            # These are COMPONENTS OF cd_pressure, not additions to it: the
            # total is always cd = cd_pressure + cd_friction. The previous code
            # took (cd - cd_pressure - cd_friction), which is ~0 for a converged
            # solve, and split that residue 70/30 wave/base — so both numbers
            # were an artefact of integration error, not physics.
            from cfd.drag_decomposition import (
                base_drag_from_surface, split_pressure_drag,
            )
            _p_inf = _meta.get("P", 101325.0)
            _q_inf = _meta.get("dynamic_pressure", 1.0)
            _A_ref = _meta.get("ref_area", 0.1)
            _L_ref = _meta.get("ref_length", 1.0)

            base = base_drag_from_surface(
                surf_vtk, p_inf=_p_inf, q_inf=_q_inf, ref_area=_A_ref,
                # Same wind axis _integrate_surface_forces used above, so the
                # forebody remainder below is a like-for-like subtraction.
                aoa_deg=self.config.angle_of_attack_deg,
            )
            if base is not None:
                result.cd_base = base["cd_base"]
                result.base_area_m2 = base["base_area"]
                logger.info(
                    f"Base drag (integrated over {base['n_cells']} rearward-facing "
                    f"cells, {base['base_area']:.6f} m² projected): "
                    f"Cd_base={result.cd_base:.5f}  mean Cp={base['base_cp_mean']:.4f}"
                )
            else:
                result.cd_base = 0.0

            fore = split_pressure_drag(
                result.cd_pressure, result.cd_base, self.config.mach
            )
            result.cd_forebody_pressure = fore["cd_forebody_pressure"]
            result.cd_wave = fore["cd_wave"]
            result.drag_decomposition_method = fore["method"]
            logger.info(
                f"Pressure split: Cd_p={result.cd_pressure:.5f} = "
                f"base {result.cd_base:.5f} + forebody "
                f"{result.cd_forebody_pressure:.5f}  →  "
                f"Cd_wave={result.cd_wave:.5f} [{fore['method']}]"
            )

            # Oswatitsch entropy-production integral — mesh-quality diagnostic
            # only (see cfd/drag_decomposition.py); never reported, and skipped
            # unless someone has turned DEBUG on, since it re-reads the volume
            # mesh and differentiates it.
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    from cfd.drag_decomposition import wave_drag_from_volume
                    import pyvista as _pv
                    _bb = _pv.read(str(surf_vtk)).bounds if surf_vtk.is_file() else None
                    _osw = wave_drag_from_volume(
                        vol_vtk, p_inf=_p_inf, rho_inf=_meta.get("rho", 1.225),
                        T_inf=_meta.get("T", 288.15),
                        u_inf=_meta.get("v_inf", 1.0),
                        q_inf=_q_inf, ref_area=_A_ref, ref_length=_L_ref,
                        mach=self.config.mach, body_bounds=_bb,
                    )
                    if _osw:
                        logger.debug(
                            f"[diagnostic] Oswatitsch Cd_wave={_osw['cd_wave']:.5f} "
                            f"over {_osw['n_shock_cells']} shock cells vs "
                            f"surface-integral {result.cd_wave:.5f} — ratio "
                            f"{_osw['cd_wave']/max(result.cd_wave,1e-9):.2f} "
                            f"(≫1 means heavy numerical entropy)"
                        )
                except Exception as e:
                    logger.debug(f"Oswatitsch diagnostic unavailable: {e}")

            # Force components — use coefficients * q * A for reliability
            # SU2 FORCE_X/Z columns are non-dimensional; convert to Newtons
            meta_local = getattr(self, '_flow_meta', {})
            _q = meta_local.get("dynamic_pressure", 1.0)
            _A = meta_local.get("ref_area", 0.1)
            # Dimensional forces from coefficients (robust vs. history column naming)
            result.force_axial  = result.cd * _q * _A   # drag force [N]
            result.force_normal = result.cl * _q * _A   # lift/normal force [N]

            # Flow conditions from stored metadata
            meta = getattr(self, '_flow_meta', {})
            result.v_inf = meta.get("v_inf", self.config.mach * 340.0)
            result.mach = self.config.mach
            result.altitude_m = self.config.altitude_m
            result.angle_of_attack_deg = self.config.angle_of_attack_deg
            result.reynolds = meta.get("reynolds", 0.0)
            result.dynamic_pressure = meta.get("dynamic_pressure", 0.0)
            result.ref_length = meta.get("ref_length", 1.0)
            result.reference_area_m2 = meta.get("ref_area", 0.1)

            # Turbulence model info
            turb_key = self.config.turbulence_model if self.config.turbulence_model in _TURB_MODEL_MAP else "SST"
            result.turbulence_model = turb_key
            result.solver_name = "SU2"

            # ── Hybrid Euler + flat-plate friction ───────────────────────────
            # Applied HERE, not in the sweep, because it belongs to a solved
            # point rather than to the act of sweeping. It used to live only in
            # cfd/sweep.py, which meant the mode the UI calls "recommended" --
            # and which exists precisely because wall-unresolved RANS on this
            # mesh reports essentially no skin friction -- was unreachable from
            # the single-run button. A single run therefore defaulted to SST and
            # returned a drag missing its entire friction component.
            #
            # Pressure/wave drag, lift, moments and CP keep their integrated
            # (inviscid) values untouched; only Cd gains the build-up.
            if getattr(self.config, "euler_analytic_friction", False):
                from cfd.sweep import analytic_friction_cd, analytic_friction_cd_cad
                if self.config.external_cad and self.config.cad_info:
                    cd_f = analytic_friction_cd_cad(
                        self.config.cad_info, result.reynolds, result.mach,
                        result.reference_area_m2)
                else:
                    cd_f = analytic_friction_cd(
                        self.config.geometry_dict, result.reynolds, result.mach,
                        result.reference_area_m2)
                if cd_f is not None:
                    result.cd += cd_f
                    result.cd_friction = cd_f
                    result.force_axial = (result.cd * result.dynamic_pressure
                                          * result.reference_area_m2)
                    result.solver_name = "SU2 Euler + flat-plate friction"
                    logger.info(f"Analytic friction added: Cd_f={cd_f:.5f} "
                                f"-> Cd={result.cd:.5f}")
                else:
                    logger.warning(
                        "Euler+friction mode: geometry or Reynolds number "
                        "unavailable — Cd is inviscid-only for this point and "
                        "is missing its friction component entirely."
                    )

            # ── Base drag on an inviscid solve ───────────────────────────────
            # Kept as a warning rather than a correction, because there is no
            # honest correction to make. Inviscid flow has no separated base
            # recirculation: the pressure recovers over the aft face instead of
            # sitting below ambient, so the integrated cd_base is a lower bound
            # and the real base drag (30-50% of subsonic rocket Cd) is partly
            # missing. The flat-plate build-up above supplies friction, not this.
            if turb_key == "Euler" and result.cd_base < 0.02:
                logger.warning(
                    f"Cd_base={result.cd_base:.5f} comes from an INVISCID solve, "
                    f"which has no base recirculation — the aft face recovers "
                    f"pressure instead of sitting below ambient. Treat it as a "
                    f"lower bound; on a blunt-based rocket the real base drag is "
                    f"a large fraction of the total."
                )

            # Wall-resolution verdict from generate_case (see _wall_verdict).
            result.wall_resolved, result.wall_warning = getattr(
                self, "_wall_verdict", (True, ""))
            if not result.wall_resolved:
                logger.warning(
                    f"Cd_friction={result.cd_friction:.6f} comes from a wall "
                    f"the mesh does not resolve — see the wall warning on this "
                    f"result. Total Cd={result.cd:.5f} is therefore a LOWER "
                    f"BOUND."
                )

            # CP location — recovered per point from the integrated surface forces.
            # SU2's LIFT/MOMENT_Y are the integrated pressure+shear loads, so the CP
            # derived from them IS a pressure-integration CP (not a fitted curve).
            # Mesh frame: nose tip at x=0, nozzle at x=total_L (cfd/meshing.py).
            # Cm (= CMy) is taken about REF_ORIGIN_MOMENT_X = 0 (the nose tip):
            #     My = -(x_cp - 0) * N   ⇒   Cm = -CN * x_cp / ref_length
            #  => x_cp_from_nose = -(Cm / CN) * ref_length
            # Verified against SU2 v8.5 (finned 1 m test rocket, AoA 4°):
            # CMy=-0.54 → x_cp=0.626 m from nose; CMz was 15x smaller (noise).
            aoa_rad = math.radians(self.config.angle_of_attack_deg)
            # Normal force coefficient from wind-frame CL/CD (exact, matters >5° AoA)
            _CN = result.cl * math.cos(aoa_rad) + result.cd * math.sin(aoa_rad)
            # True rocket length for clamping — prefer geometry_dict (exact)
            # over ref_length (which may include fin span from STL bbox)
            _true_len = result.ref_length
            if self.config.geometry_dict and "length" in self.config.geometry_dict:
                _true_len = self.config.geometry_dict["length"]
            elif self.config.external_cad and self.config.cad_info:
                # Imported body: the CP must lie within the body's own flow-wise
                # extent, which is the measured bbox length — not ref_length,
                # which the user may have overridden to a chord or span.
                _true_len = float(self.config.cad_info.get("length", _true_len))
            # Threshold scales with the swept normal force so a single near-zero-AoA
            # point is excluded (CN→0 makes Cm/CN indeterminate) but every genuinely
            # loaded point is kept and computed independently.
            # Body-frame normal force (what the airframe actually bends under) —
            # more correct than the wind-frame CL set above, especially >5° AoA.
            result.force_normal = abs(_CN) * _q * _A
            # Two conditions, not one. The CN threshold alone was 0.003, which
            # a symmetric body at EXACTLY zero angle of attack can exceed on
            # residual asymmetry alone: measured CN = -0.0065 at alpha = 0,
            # which produced "CP raw x_cp = -0.819 m from nose outside body,
            # clamped" -- a clamp firing on a quantity that has no value to
            # clamp. CP is the ratio of two numbers that both go to zero with
            # incidence, so it is undefined at alpha = 0 no matter how well the
            # forces converged, and the honest answer is to say so.
            _has_incidence = (abs(self.config.angle_of_attack_deg) > 0.05
                              or abs(self.config.sideslip_angle_deg) > 0.05)
            if abs(_CN) > 0.003 and _has_incidence:
                _xcp_nose = -(result.cm / _CN) * result.ref_length
                # Clamp to the physical body range [0, true_length]; warn (don't
                # silently saturate) if the raw value lands outside so a flat-line
                # artefact is visible in the log rather than hidden.
                cp_nose = max(0.0, min(_xcp_nose, _true_len))
                if not (0.0 <= _xcp_nose <= _true_len):
                    logger.warning(f"CP raw x_cp={_xcp_nose:.4f} m from nose outside "
                                   f"body [0,{_true_len:.3f}] — clamped "
                                   f"(Cm={result.cm:.5f}, CN={_CN:.5f})")
                result.cp_from_nose_m = cp_nose
                result.cp_location_m = _true_len - cp_nose   # from nozzle/tail

                # ── Static-stability moment about the CG ─────────────────────
                # Transfer the integrated nose-tip moment to the CG (both
                # stations measured from the nose):
                #     Cm_cg = CN * (x_cg - x_cp) / ref_length
                # CP aft of CG (x_cp > x_cg) ⇒ Cm_cg < 0 at positive AoA
                # ⇒ restoring ⇒ statically stable. This is exactly
                # M = (CG - CP) × F reduced to coefficient form.
                _cg_nose = self.config.cg_from_nose_m
                if _cg_nose is not None:
                    result.x_cg_m = _cg_nose   # from nose (CFD x-axis)
                    result.cm_cg = _CN * (_cg_nose - cp_nose) / result.ref_length
                logger.info(f"CP = {result.cp_from_nose_m:.4f} m from nose "
                            f"({result.cp_location_m:.4f} from nozzle) "
                            f"(Cm_nose={result.cm:.5f}, Cm_cg={result.cm_cg:.5f}, "
                            f"CN={_CN:.5f}, ref_L={result.ref_length:.3f})")
            else:
                # AoA≈0: CP indeterminate (no net normal force). Leave as sentinel 0;
                # the sweep interpolates this point from its loaded neighbours so the
                # CP-vs-AoA curve stays smooth instead of dropping to zero.
                result.cp_location_m = 0.0
                result.cm_cg = 0.0   # zero normal force ⇒ zero stability moment

            # Converged if the residual hit the configured floor, OR SU2
            # stopped early (a convergence criterion — residual or Cauchy —
            # fired before the iteration cap), OR the force coefficients are
            # stationary over the tail window. The last case covers residual
            # limit cycles (unsteady base flow under steady RANS) where the
            # density residual orbits above the floor forever while CD/CL/CMy
            # are flat to <0.5% — the forces, which are what the polar reports,
            # ARE converged there.
            try:
                last_rho = _get("rms[Rho]")
                conv_floor = math.log10(self.config.convergence_tolerance)  # e.g. -6
                stopped_early = len(rows) < self.config.max_iterations
                result.final_residual = last_rho
                # Decades fallen from the FIRST iteration. The absolute floor
                # cannot mean anything on its own here: residuals are
                # dimensional, so where a case starts is set by the flow scale.
                # A run beginning at rms[Rho] = -3 clears a -6 floor after three
                # decades; one beginning at +2 needs eight for the same flag.
                try:
                    first_rho = float(rows[0].get("rms[Rho]", "nan"))
                except (TypeError, ValueError):
                    first_rho = float("nan")
                drop = (first_rho - last_rho
                        if math.isfinite(first_rho) and math.isfinite(last_rho)
                        else 0.0)
                result.residual_drop_decades = drop

                # ── Convergence, with the failure cases actually excluded ─────
                # "Stopped before the iteration cap" used to be sufficient on its
                # own, which made every abnormal termination look converged —
                # including a solve that went non-physical and exited 0. That
                # flag gates inject_cfd_results_into_engine, so a diverged run
                # could push garbage coefficients into the sim engine.
                #
                # Now a run must clear three hurdles: the reported forces have to
                # be finite numbers, the residual must not have blown up from its
                # own best value, and one genuine convergence signal has to be
                # present.
                _vals = [result.cd, result.cl, result.cm, last_rho]
                finite = all(isinstance(v, float) and math.isfinite(v)
                             for v in _vals)

                # Divergence: the density residual climbing far above its best
                # is the signature of a solution running away, regardless of
                # where it happens to stop.
                _rhos = []
                for _r in rows:
                    try:
                        _v = float(_r.get("rms[Rho]", "nan"))
                        if math.isfinite(_v):
                            _rhos.append(_v)
                    except (TypeError, ValueError):
                        pass
                best_rho = min(_rhos) if _rhos else last_rho
                diverging = bool(_rhos) and (last_rho > best_rho + 2.0)  # 100x

                hit_floor = finite and last_rho <= conv_floor
                # A genuine convergence signal is now REQUIRED, and "SU2 stopped
                # before the iteration cap" is no longer one on its own. It used
                # to be, which meant a case whose dimensional residual started
                # just above the floor cleared it in three decades and was
                # flagged converged with the forces still moving. Either the
                # residual has fallen _MIN_RESIDUAL_DECADES from where it began,
                # or the reported forces are stationary. The early stop is kept
                # only as corroboration.
                deep_enough = finite and drop >= _MIN_RESIDUAL_DECADES
                result.converged = bool(
                    finite and not diverging
                    and (hit_floor or deep_enough or forces_stationary)
                )

                if not result.converged:
                    if not finite:
                        result.convergence_note = (
                            f"Non-finite results (Cd={result.cd}, Cl={result.cl}, "
                            f"Cm={result.cm}, rms[Rho]={last_rho}) — the solve "
                            f"went non-physical.")
                    elif diverging:
                        result.convergence_note = (
                            f"Residual diverged: rms[Rho] ended at {last_rho:.2f} "
                            f"against a best of {best_rho:.2f} "
                            f"({last_rho - best_rho:.1f} decades worse).")
                    else:
                        result.convergence_note = (
                            f"rms[Rho] fell only {drop:.1f} decades (need "
                            f"{_MIN_RESIDUAL_DECADES:g}), the absolute floor "
                            f"{conv_floor:.0f} was not reached, and the drag "
                            f"coefficient is still moving "
                            f"{(cd_max - cd_min) / max(abs(cd_mean), 1e-9) * 100:.2f}% "
                            f"over the last {tail_n} iterations.")
                    logger.warning("Not converged: " + result.convergence_note)
                else:
                    _why = []
                    if hit_floor:
                        _why.append(f"rms[Rho]={last_rho:.2f} at or below the "
                                    f"{conv_floor:.0f} floor")
                    if deep_enough:
                        _why.append(f"residual fell {drop:.1f} decades")
                    if forces_stationary:
                        _why.append(f"Cd stationary to "
                                    f"{(cd_max - cd_min) / max(abs(cd_mean), 1e-9) * 100:.2f}% "
                                    f"over {tail_n} iterations")
                    if stopped_early:
                        _why.append(f"SU2 stopped at iteration {len(rows)} of "
                                    f"{self.config.max_iterations}")
                    result.convergence_note = "; ".join(_why)
                    logger.info("Converged: " + result.convergence_note)
            except Exception as e:
                logger.warning(f"Convergence check failed ({e}) — "
                               f"reporting not converged.")
                result.converged = False

            # Build residual history for the convergence plot
            for row in rows:
                try:
                    it = int(row.get("Inner_Iter", row.get("Time_Iter", 0)))
                    rms = float(row.get("rms[Rho]", 0.0))
                    result.residual_history.append((it, rms))
                except (ValueError, TypeError, KeyError):
                    pass

        except Exception as e:
            logger.error(f"Error parsing history.csv: {e}")

        result.volume_vtk = vol_vtk if vol_vtk.is_file() else None
        result.surface_vtk = surf_vtk if surf_vtk.is_file() else None

        logger.info(
            f"CFD Results → Cd={result.cd:.4f} (P:{result.cd_pressure:.4f} F:{result.cd_friction:.4f}), "
            f"Cl={result.cl:.4f}, Cm={result.cm:.4f}, Re={result.reynolds:.2e}, "
            f"converged={result.converged}"
        )
        return result

