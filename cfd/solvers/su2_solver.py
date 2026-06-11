"""
K2 Aerospace — SU2 Solver Backend
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

# ── Locate the bundled SU2 binary ────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]   # K2 Software root
_BIN_DIR = _ROOT / "bin"

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
%% K2 Aerospace - SU2 CFD Configuration                   %%
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
NUM_METHOD_GRAD= GREEN_GAUSS
CONV_NUM_METHOD_FLOW= ROE
ENTROPY_FIX_COEFF= 0.2
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

HISTORY_OUTPUT= ITER, RMS_DENSITY, RMS_ENERGY, {turb_hist_fields} LIFT, DRAG, DRAG_PRESSURE, DRAG_VISCOUS, MOMENT_Y, FORCE_X, FORCE_Y, FORCE_Z
CONV_FILENAME= history

VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
OUTPUT_FILES= (RESTART, PARAVIEW, SURFACE_PARAVIEW)
OUTPUT_WRT_FREQ= 250

% Volume fields: solution + derived
VOLUME_OUTPUT= COORDINATES, SOLUTION, PRIMITIVE, PRESSURE_COEFFICIENT, MACH, VORTICITY, Q_CRITERION, LAMBDA2, Y_PLUS
% Surface fields: wall quantities for post-processing
% SURFACE_OUTPUT is not supported in all SU2 versions; surface VTK inherits from VOLUME_OUTPUT
% SURFACE_OUTPUT= COORDINATES, SOLUTION, PRESSURE_COEFFICIENT, SKIN_FRICTION, Y_PLUS
"""



class SU2Solver(CFDSolver):
    """SU2 CFD solver backend for K2 Aerospace."""

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
            bl_layers=cfg.boundary_layer_layers,
            bl_growth=cfg.boundary_layer_growth,
            geometry_dict=cfg.geometry_dict,   # exact dims if available
            custom_wall_size=cfg.custom_wall_size,
            target_element_count=cfg.target_element_count,
        )
        self._mesh_path = out_mesh
        logger.info(f"Mesh written to {out_mesh}")
        return out_mesh


    # ── Configuration file generation ────────────────────────────────────────

    def generate_case(self) -> Path:
        """Write the SU2 .cfg file with correct ISA conditions."""
        cfg = self.config
        P, T, rho = isa_conditions(cfg.altitude_m)
        a = math.sqrt(1.4 * 287.05 * T)          # speed of sound
        V_inf = cfg.mach * a                       # freestream velocity
        mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)  # Sutherland

        # Reference values
        ref_length = 1.0
        ref_area = 0.1
        
        # Priority 1: Use exact geometry dict parameters (avoids fin span bug)
        if cfg.geometry_dict and "max_diameter" in cfg.geometry_dict:
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

        # Wall BC: Euler uses slip wall, viscous uses no-slip heatflux
        if is_viscous:
            wall_bc = "MARKER_HEATFLUX= ( rocket_wall, 0.0 )"
        else:
            wall_bc = "MARKER_EULER= ( rocket_wall )"

        # Turbulence numerics (only for RANS)
        turb_numerics = "CONV_NUM_METHOD_TURB= SCALAR_UPWIND\nMUSCL_TURB= NO" if is_rans else ""
        turb_time_discre = "TIME_DISCRE_TURB= EULER_IMPLICIT" if is_rans else ""
        turb_hist_fields = "RMS_TKE, " if is_rans else ""

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

            # Core coefficients. Pitch moment is CMy: SU2 applies AoA as a
            # rotation about the Y axis (freestream tilts in the x-z plane),
            # so the AoA-induced normal force is along Z and its moment is
            # about Y. CMz is the (near-zero) yaw component — do not use it.
            result.cd = abs(_get("CD"))
            result.cl = _get("CL")
            result.cm = _get("CMy")
            result.iterations = len(rows)

            # Drag decomposition
            result.cd_pressure = abs(_get("CD_Pressure") or _get("DRAG_PRESSURE"))
            result.cd_friction = abs(_get("CD_Viscous") or _get("DRAG_VISCOUS"))
            # Estimate base drag and wave drag from total
            if result.cd > 0 and result.cd_pressure + result.cd_friction > 0:
                accounted = result.cd_pressure + result.cd_friction
                remainder = max(0.0, result.cd - accounted)
                if self.config.mach > 0.8:
                    result.cd_wave = remainder * 0.7
                    result.cd_base = remainder * 0.3
                else:
                    result.cd_base = remainder
            elif result.cd > 0:
                # SU2 didn't output decomposition — estimate
                result.cd_pressure = result.cd * 0.55
                result.cd_friction = result.cd * 0.30
                result.cd_base = result.cd * 0.15

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
            result.reynolds = meta.get("reynolds", 0.0)
            result.dynamic_pressure = meta.get("dynamic_pressure", 0.0)
            result.ref_length = meta.get("ref_length", 1.0)
            result.reference_area_m2 = meta.get("ref_area", 0.1)

            # Turbulence model info
            turb_key = self.config.turbulence_model if self.config.turbulence_model in _TURB_MODEL_MAP else "SST"
            result.turbulence_model = turb_key
            result.solver_name = "SU2"

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
            # Threshold scales with the swept normal force so a single near-zero-AoA
            # point is excluded (CN→0 makes Cm/CN indeterminate) but every genuinely
            # loaded point is kept and computed independently.
            # Body-frame normal force (what the airframe actually bends under) —
            # more correct than the wind-frame CL set above, especially >5° AoA.
            result.force_normal = abs(_CN) * _q * _A
            if abs(_CN) > 0.003:  # meaningful normal force present
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
            # fired before the iteration cap). Exhausting max_iterations
            # without hitting the floor is NOT convergence.
            try:
                last_rho = _get("rms[Rho]")
                conv_floor = math.log10(self.config.convergence_tolerance)  # e.g. -6
                stopped_early = len(rows) < self.config.max_iterations
                result.converged = (last_rho <= conv_floor) or stopped_early
                result.final_residual = last_rho
            except Exception:
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

