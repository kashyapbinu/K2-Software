"""
K2 AeroSim — CFD Workspace
Clean, scrollable layout matching other K2 workspaces.
"""
from __future__ import annotations
import logging, math, threading
from pathlib import Path

from core.paths import user_data_dir

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox,
    QFileDialog, QProgressBar, QTextEdit, QFormLayout,
    QRadioButton, QButtonGroup, QScrollArea, QFrame, QTabWidget,
    QCheckBox, QSlider, QToolButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from ui.icons import icon
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from cfd.post_processing import field_percentiles

from ui import theme

logger = logging.getLogger("K2.CFD.Workspace")

# ── Shared style helpers ──────────────────────────────────────────────────────
def _make_val_label(text="—"):
    lbl = QLabel(text)
    lbl.setProperty("value", True)
    return lbl


def _utf8_env():
    """Environment for a mesh subprocess whose stdout the parent reads as UTF-8.

    On Windows the child otherwise picks the locale codec (cp1252) for its
    pipe, which breaks the mesh log both ways: characters outside cp1252
    ('→') raise UnicodeEncodeError inside logging, and characters that do
    encode ('±') produce bytes the UTF-8 parent cannot decode. Set at the
    interpreter level so it also covers the frozen build, where the script's
    own sys.stdout.reconfigure() may run too late.
    """
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8:replace"
    return env


# ── Worker thread — runs mesh gen (subprocess) + SU2 solver ──────────────────
# Gmsh calls signal.signal() during initialize(), which ONLY works in the
# main thread of the main interpreter. So we run mesh generation in a
# separate Python subprocess via multiprocessing, keeping the UI responsive.
class SolverThread(QThread):
    progress = pyqtSignal(int, float)
    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(object)
    errored  = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.solver = None
        self._mesh_proc = None  # track subprocess for cleanup on stop

    def run(self):
        import logging, traceback
        logger = logging.getLogger("K2.CFD.SolverThread")
        try:
            from cfd.solvers.su2_solver import SU2Solver

            # Phase 1: Mesh generation — MUST run in subprocess (Gmsh + signal)
            self.log_msg.emit("Phase 1/3: Generating computational mesh (Gmsh)…")
            self._run_mesh_subprocess()
            self.log_msg.emit("Mesh generation complete.")

            # Phase 2 & 3: Solver config + run (no Gmsh, safe in thread)
            self.solver = SU2Solver(self.config)
            # Point solver to the already-generated mesh
            self.solver._mesh_path = self.config.work_dir / "rocket_mesh.su2"

            self.log_msg.emit("Phase 2/3: Writing SU2 solver configuration…")
            self.solver.generate_case()
            self.log_msg.emit("Configuration ready.")

            self.log_msg.emit("Phase 3/3: Running SU2 RANS solver…")
            self.solver.set_log_callback(self.log_msg.emit)
            for it, rms in self.solver.run():
                self.progress.emit(it, rms)

            self.log_msg.emit("Parsing results…")
            result = self.solver.parse_results()
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            logger.critical(f"SolverThread crashed:\n{tb}")
            self.errored.emit(str(e))

    def _run_mesh_subprocess(self):
        """Run Gmsh meshing in a separate Python process to avoid signal issues."""
        import subprocess, sys, json
        from pathlib import Path

        cfg = self.config
        cfg.work_dir.mkdir(parents=True, exist_ok=True)

        # Resolve K2 Software root from this file's location
        # cfd_workspace.py → ui/workspaces/ → ui/ → K2 Software/
        k2_root = str(Path(__file__).resolve().parents[2])

        # Serialize parameters to JSON for the subprocess
        mesh_params = {
            "stl_path":             str(cfg.geometry_stl),
            "output_path":          str(cfg.work_dir / "rocket_mesh.su2"),
            "refinement":           cfg.mesh_refinement,
            "domain_length_scale":  cfg.domain_length_scale,
            "domain_radius_scale":  cfg.domain_radius_scale,
            "bl_prisms":            cfg.bl_prisms,
            "bl_layers":            cfg.boundary_layer_layers,
            "bl_growth":            cfg.boundary_layer_growth,
            "geometry_dict":        cfg.geometry_dict,
            "custom_wall_size":     cfg.custom_wall_size,
            "target_element_count": cfg.target_element_count,
            # External-CAD mode. Without these three the subprocess silently
            # falls back to the parametric rocket and meshes the wrong body.
            "external_cad":         (str(cfg.external_cad)
                                     if cfg.external_cad else None),
            "flow_axis":            cfg.flow_axis,
            "cad_info":             cfg.cad_info,
            "cad_units":            cfg.cad_units,
            "cad_wrap":             cfg.cad_wrap,
            "cad_wrap_resolution":  cfg.cad_wrap_resolution,
            # Was missing, so a curvature-element setting reached generate_mesh()
            # on the in-process path and was silently dropped on the subprocess
            # path the UI actually uses.
            "cad_curvature_elements": cfg.cad_curvature_elements,
        }

        params_file = cfg.work_dir / "_mesh_params.json"

        # Build a small standalone script (avoids quoting issues with -c)
        script_file = cfg.work_dir / "_run_mesh.py"
        script_file.write_text(
            "import sys, json, logging\n"
            # The parent decodes this pipe as UTF-8, but the child would
            # otherwise encode it with the Windows locale codec (cp1252).
            # Mesh logs contain '→' and '±': the former is not in
            # cp1252 at all, so logging raised UnicodeEncodeError, dropped the
            # message and dumped a handler traceback into the CFD console on
            # every mesh build; the latter encoded to a byte that is not valid
            # UTF-8, so the parent replaced it with a mojibake marker.
            # Reconfigure before basicConfig so the handler binds the UTF-8
            # stream, not the one it replaced.
            "try:\n"
            "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
            "except Exception:\n"
            "    pass\n"
            "logging.basicConfig(level=logging.INFO, "
            "format='%(name)s: %(message)s', stream=sys.stdout)\n"
            f"sys.path.insert(0, {repr(k2_root)})\n"
            "from pathlib import Path\n"
            f"params = json.loads(open({repr(str(params_file))}, "
            "encoding='utf-8').read())\n"
            "from cfd.meshing import build_wind_tunnel_mesh\n"
            "build_wind_tunnel_mesh(\n"
            "    stl_path=Path(params['stl_path']),\n"
            "    output_path=Path(params['output_path']),\n"
            "    refinement=params['refinement'],\n"
            "    domain_length_scale=params['domain_length_scale'],\n"
            "    domain_radius_scale=params['domain_radius_scale'],\n"
            "    bl_prisms=params.get('bl_prisms', False),\n"
            "    bl_layers=params['bl_layers'],\n"
            "    bl_growth=params['bl_growth'],\n"
            "    geometry_dict=params['geometry_dict'],\n"
            "    custom_wall_size=params.get('custom_wall_size'),\n"
            "    target_element_count=params.get('target_element_count'),\n"
            "    external_cad=(Path(params['external_cad'])\n"
            "                  if params.get('external_cad') else None),\n"
            "    flow_axis=params.get('flow_axis', 'auto'),\n"
            "    cad_info=params.get('cad_info'),\n"
            "    cad_units=params.get('cad_units', 'auto'),\n"
            "    cad_wrap=params.get('cad_wrap', False),\n"
            "    cad_wrap_resolution=params.get('cad_wrap_resolution', 'medium'),\n"
            "    cad_curvature_elements=params.get('cad_curvature_elements'),\n"
            ")\n"
            "print('MESH_OK')\n",
            encoding="utf-8",
        )

        params_file.write_text(
            json.dumps(mesh_params, default=str), encoding="utf-8"
        )

        # Run in subprocess. In a frozen build sys.executable is K2.exe, so it
        # must be told to run the script (else it re-opens the GUI); in a source
        # run sys.executable is python and runs the script directly.
        cmd = ([sys.executable, "--run-script", str(script_file)]
               if getattr(sys, "frozen", False)
               else [sys.executable, str(script_file)])
        self._mesh_proc = subprocess.Popen(
            cmd,
            cwd=k2_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_utf8_env(),
            creationflags=(0x08000000 if sys.platform == "win32" else 0),
        )
        proc = self._mesh_proc

        # Stream output to the console log
        mesh_ok = False
        for line in proc.stdout:
            line = line.strip()
            if line:
                if "MESH_OK" in line:
                    mesh_ok = True
                else:
                    self.log_msg.emit(f"[Gmsh] {line}")

        proc.wait()

        if proc.returncode != 0 or not mesh_ok:
            raise RuntimeError(
                f"Mesh generation subprocess failed (exit code {proc.returncode}). "
                f"Check console log for Gmsh errors."
            )

        # Clean up temp files
        for f in [params_file, script_file]:
            try:
                f.unlink()
            except Exception:
                pass


# ── Post-processing thread — loads VTK + runs compute_derived_fields ──────────
class PostProcessThread(QThread):
    done    = pyqtSignal(object, object)  # (volume_mesh, surface_mesh)
    log_msg = pyqtSignal(str)
    errored = pyqtSignal(str)

    def __init__(self, work_dir: Path, p_inf: float, q_inf: float):
        super().__init__()
        self.work_dir = work_dir
        self.p_inf    = p_inf
        self.q_inf    = q_inf

    def run(self):
        import logging, traceback, gc
        import pyvista as pv
        from cfd.post_processing import compute_derived_fields
        logger = logging.getLogger("K2.CFD.PostProcess")

        vol_mesh  = None
        surf_mesh = None

        # Load volume VTK
        for name in ["flow.vtu", "flow.vtk"]:
            p = self.work_dir / name
            if p.is_file():
                try:
                    vol_mesh = pv.read(str(p))
                    self.log_msg.emit(
                        f"Volume mesh loaded: {vol_mesh.n_points:,} pts, "
                        f"arrays: {vol_mesh.array_names[:6]}"
                    )
                    self.log_msg.emit("Computing derived fields (Vorticity, Lambda-2, Cp)…")
                    vol_mesh = compute_derived_fields(
                        vol_mesh, p_inf=self.p_inf, q_inf=self.q_inf
                    )
                    derived = [a for a in vol_mesh.array_names
                               if a in ["Vorticity_Magnitude", "Q_Criterion", "Lambda2",
                                        "Mach", "Pressure_Coefficient"]]
                    self.log_msg.emit(f"Derived fields done: {derived}")
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.critical(f"Volume VTK processing crashed:\n{tb}")
                    self.log_msg.emit(f"Volume VTK error: {e}")
                finally:
                    gc.collect()
                break

        # Load surface VTK
        for name in ["surface_flow.vtu", "surface_flow.vtk"]:
            p = self.work_dir / name
            if p.is_file():
                try:
                    surf_mesh = pv.read(str(p))
                    surf_mesh = compute_derived_fields(
                        surf_mesh, p_inf=self.p_inf, q_inf=self.q_inf
                    )
                    self.log_msg.emit(f"Surface mesh loaded: {surf_mesh.n_points:,} pts")
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.critical(f"Surface VTK processing crashed:\n{tb}")
                    self.log_msg.emit(f"Surface VTK error: {e}")
                break

        self.done.emit(vol_mesh, surf_mesh)


# ── Sweep thread — one mesh, N solves over AoA or Mach ────────────────────────
class SweepThread(QThread):
    point    = pyqtSignal(object)            # SweepPoint (one solved condition)
    progress = pyqtSignal(int, int, int, float)  # (pt_idx, n_pts, iter, rms)
    log_msg  = pyqtSignal(str)
    finished = pyqtSignal(object)            # SweepData (all points)
    errored  = pyqtSignal(str)

    def __init__(self, base_config, var: str, values: list):
        super().__init__()
        self.config = base_config
        self.var    = var
        self.values = values
        self.solver = None          # current point's SU2 solver (sequential path)
        self._solvers = []          # all live solvers (parallel path)
        self._mesh_proc = None
        self._stop = False

    def stop(self):
        self._stop = True
        # Kill every live SU2 subprocess (sequential + all concurrent points).
        solvers = list(self._solvers)
        if self.solver is not None:
            solvers.append(self.solver)
        procs = [self._mesh_proc] + [getattr(s, "_proc", None) for s in solvers]
        for proc in procs:
            try:
                if proc and proc.poll() is None:
                    proc.kill()
            except Exception:
                pass

    def run(self):
        import logging, traceback
        logger = logging.getLogger("K2.CFD.SweepThread")
        try:
            from cfd.sweep import SweepData

            n = len(self.values)
            self.log_msg.emit(
                f"Sweep: {self.var} over {n} points "
                f"[{self.values[0]:g} … {self.values[-1]:g}] — building shared mesh…"
            )

            # Phase 1: one mesh for the whole sweep (Gmsh in subprocess).
            self._run_mesh_subprocess()
            mesh_path = self.config.work_dir / "rocket_mesh.su2"
            self.log_msg.emit("Shared mesh ready. Starting solves…")

            data = SweepData(var=self.var)
            # AoA points are independent (cold-started) → run them concurrently.
            # Mach points chain through warm-start → must stay sequential.
            if self.var == "aoa" and n > 1:
                self._solve_parallel(mesh_path, data, n)
            else:
                self._solve_sequential(mesh_path, data, n)

            self.finished.emit(data)
        except Exception as e:
            tb = traceback.format_exc()
            logger.critical(f"SweepThread crashed:\n{tb}")
            self.errored.emit(str(e))

    def _make_point_solver(self, mesh_path, val, omp_threads=None):
        """Build (cfg, solver) for one sweep point on the shared mesh."""
        import copy
        from pathlib import Path
        from cfd.solvers.su2_solver import SU2Solver
        from cfd.sweep import SWEEP_VARS, stage_mesh
        cfg = copy.deepcopy(self.config)
        setattr(cfg, SWEEP_VARS[self.var], val)
        tag = (f"{self.var}_{val:+.3f}"
               .replace("+", "p").replace("-", "m").replace(".", "_"))
        cfg.work_dir = Path(self.config.work_dir) / "sweep" / tag
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        solver = SU2Solver(cfg)
        solver._mesh_path = stage_mesh(mesh_path, cfg.work_dir)
        if omp_threads:
            solver.omp_threads = omp_threads
        return cfg, solver

    def _solve_sequential(self, mesh_path, data, n):
        """Solve points one at a time. Mach sweeps warm-start point→point."""
        import shutil
        from cfd.sweep import SweepPoint
        prev_restart = None   # converged restart_flow.dat from the last point
        for i, val in enumerate(self.values):
            if self._stop:
                self.log_msg.emit("Sweep stopped by user.")
                break
            self.log_msg.emit(f"[{i+1}/{n}] Solving {self.var} = {val:g}…")

            cfg, self.solver = self._make_point_solver(mesh_path, val)

            # Warm start: march from the previous point's converged solution.
            # Only for non-AoA (Mach) sweeps — AoA must stay independent (req #7).
            allow_warm = self.var != "aoa"
            if allow_warm and prev_restart is not None and prev_restart.is_file():
                try:
                    shutil.copy2(prev_restart, cfg.work_dir / "solution_flow.dat")
                    self.solver.warm_start = True
                    self.log_msg.emit("    warm-starting from previous point")
                except Exception as e:
                    self.log_msg.emit(f"    warm-start skipped ({e}); cold start")

            self.solver.generate_case()
            idx = i
            self.solver.set_progress_callback(
                lambda it, rms, _i=idx: self.progress.emit(_i, n, it, rms)
            )
            self.solver.set_log_callback(lambda msg: self.log_msg.emit(f"    {msg}"))
            for _it, _rms in self.solver.run():
                if self._stop:
                    break
            if self._stop:
                self.log_msg.emit("Sweep stopped by user.")
                break

            result = self.solver.parse_results()
            restart_file = cfg.work_dir / "restart_flow.dat"
            if restart_file.is_file():
                prev_restart = restart_file
            pt = SweepPoint(self.var, float(val), result)
            data.points.append(pt)
            self.point.emit(pt)
            self.log_msg.emit(
                f"[{i+1}/{n}] {self.var}={val:g} → "
                f"Cd={result.cd:.4f} Cl={result.cl:.4f} Cm={result.cm:.4f} "
                f"{'✓' if result.converged else '(not converged)'}"
            )

    def _solve_parallel(self, mesh_path, data, n):
        """Solve independent points concurrently as separate SU2 processes.

        Each point cold-starts (no cross-point coupling), so they run as W
        simultaneous processes, each pinned to a slice of the cores via
        OMP_NUM_THREADS. Concurrent independent processes scale far better than
        OpenMP within one solve (which measured only ~2× on 11 threads).
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from cfd.sweep import SweepPoint

        cores = os.cpu_count() or 1
        workers = min(n, max(1, cores - 1))         # leave one core for the UI
        omp_each = max(1, (cores - 1) // workers)   # thread slice per point
        self.log_msg.emit(
            f"Parallel sweep: up to {workers} points at once × {omp_each} OpenMP "
            f"threads each ({n} points across {cores} cores)."
        )

        def solve(i, val):
            if self._stop:
                return None
            cfg, solver = self._make_point_solver(mesh_path, val, omp_threads=omp_each)
            self._solvers.append(solver)   # tracked so stop() can kill it
            solver.generate_case()
            solver.set_progress_callback(
                lambda it, rms, _i=i: self.progress.emit(_i, n, it, rms)
            )
            solver.set_log_callback(
                lambda msg, _i=i: self.log_msg.emit(f"    [{_i+1}] {msg}")
            )
            for _it, _rms in solver.run():
                if self._stop:
                    break
            return SweepPoint(self.var, float(val), solver.parse_results())

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(solve, i, val): (i, val)
                    for i, val in enumerate(self.values)}
            for fut in as_completed(futs):
                i, val = futs[fut]
                if self._stop:
                    continue   # drain remaining futures without acting
                try:
                    pt = fut.result()
                except Exception as e:
                    self.log_msg.emit(f"[{i+1}/{n}] {self.var}={val:g} FAILED: {e}")
                    continue
                if pt is None:
                    continue
                data.points.append(pt)
                self.point.emit(pt)
                done += 1
                r = pt.result
                self.log_msg.emit(
                    f"[{done}/{n}] {self.var}={val:g} → "
                    f"Cd={r.cd:.4f} Cl={r.cl:.4f} Cm={r.cm:.4f} "
                    f"{'✓' if r.converged else '(not converged)'}"
                )
        if self._stop:
            self.log_msg.emit("Sweep stopped by user.")

    def _run_mesh_subprocess(self):
        """Generate the shared sweep mesh in a Gmsh subprocess (signal-safe)."""
        import subprocess, sys, json
        from pathlib import Path

        cfg = self.config
        cfg.work_dir = Path(cfg.work_dir)
        cfg.work_dir.mkdir(parents=True, exist_ok=True)
        k2_root = str(Path(__file__).resolve().parents[2])

        mesh_params = {
            "stl_path":             str(cfg.geometry_stl),
            "output_path":          str(cfg.work_dir / "rocket_mesh.su2"),
            "refinement":           cfg.mesh_refinement,
            "domain_length_scale":  cfg.domain_length_scale,
            "domain_radius_scale":  cfg.domain_radius_scale,
            "bl_prisms":            cfg.bl_prisms,
            "bl_layers":            cfg.boundary_layer_layers,
            "bl_growth":            cfg.boundary_layer_growth,
            "geometry_dict":        cfg.geometry_dict,
            "custom_wall_size":     cfg.custom_wall_size,
            "target_element_count": cfg.target_element_count,
            # External-CAD mode. Without these three the subprocess silently
            # falls back to the parametric rocket and meshes the wrong body.
            "external_cad":         (str(cfg.external_cad)
                                     if cfg.external_cad else None),
            "flow_axis":            cfg.flow_axis,
            "cad_info":             cfg.cad_info,
            "cad_units":            cfg.cad_units,
            "cad_wrap":             cfg.cad_wrap,
            "cad_wrap_resolution":  cfg.cad_wrap_resolution,
            # Was missing, so a curvature-element setting reached generate_mesh()
            # on the in-process path and was silently dropped on the subprocess
            # path the UI actually uses.
            "cad_curvature_elements": cfg.cad_curvature_elements,
        }
        params_file = cfg.work_dir / "_mesh_params.json"

        script_file = cfg.work_dir / "_run_mesh.py"
        script_file.write_text(
            "import sys, json, logging\n"
            # The parent decodes this pipe as UTF-8, but the child would
            # otherwise encode it with the Windows locale codec (cp1252).
            # Mesh logs contain '→' and '±': the former is not in
            # cp1252 at all, so logging raised UnicodeEncodeError, dropped the
            # message and dumped a handler traceback into the CFD console on
            # every mesh build; the latter encoded to a byte that is not valid
            # UTF-8, so the parent replaced it with a mojibake marker.
            # Reconfigure before basicConfig so the handler binds the UTF-8
            # stream, not the one it replaced.
            "try:\n"
            "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
            "except Exception:\n"
            "    pass\n"
            "logging.basicConfig(level=logging.INFO, "
            "format='%(name)s: %(message)s', stream=sys.stdout)\n"
            f"sys.path.insert(0, {repr(k2_root)})\n"
            "from pathlib import Path\n"
            f"params = json.loads(open({repr(str(params_file))}, "
            "encoding='utf-8').read())\n"
            "from cfd.meshing import build_wind_tunnel_mesh\n"
            "build_wind_tunnel_mesh(\n"
            "    stl_path=Path(params['stl_path']),\n"
            "    output_path=Path(params['output_path']),\n"
            "    refinement=params['refinement'],\n"
            "    domain_length_scale=params['domain_length_scale'],\n"
            "    domain_radius_scale=params['domain_radius_scale'],\n"
            "    bl_prisms=params.get('bl_prisms', False),\n"
            "    bl_layers=params['bl_layers'],\n"
            "    bl_growth=params['bl_growth'],\n"
            "    geometry_dict=params['geometry_dict'],\n"
            "    custom_wall_size=params.get('custom_wall_size'),\n"
            "    target_element_count=params.get('target_element_count'),\n"
            "    external_cad=(Path(params['external_cad'])\n"
            "                  if params.get('external_cad') else None),\n"
            "    flow_axis=params.get('flow_axis', 'auto'),\n"
            "    cad_info=params.get('cad_info'),\n"
            "    cad_units=params.get('cad_units', 'auto'),\n"
            "    cad_wrap=params.get('cad_wrap', False),\n"
            "    cad_wrap_resolution=params.get('cad_wrap_resolution', 'medium'),\n"
            "    cad_curvature_elements=params.get('cad_curvature_elements'),\n"
            ")\n"
            "print('MESH_OK')\n",
            encoding="utf-8",
        )

        params_file.write_text(
            json.dumps(mesh_params, default=str), encoding="utf-8"
        )

        # Frozen build: tell K2.exe to run the script (else it re-opens the GUI).
        cmd = ([sys.executable, "--run-script", str(script_file)]
               if getattr(sys, "frozen", False)
               else [sys.executable, str(script_file)])
        self._mesh_proc = subprocess.Popen(
            cmd,
            cwd=k2_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=_utf8_env(),
            creationflags=(0x08000000 if sys.platform == "win32" else 0),
        )
        proc = self._mesh_proc
        mesh_ok = False
        # Persist the full Gmsh output so a failure is diagnosable after the
        # fact (the live log_msg stream is transient UI text). On exit-1 we
        # tail this into the error instead of losing the real cause.
        mesh_log = cfg.work_dir / "mesh_gen.log"
        with open(mesh_log, "w", encoding="utf-8", errors="replace") as log_f:
            for line in proc.stdout:
                log_f.write(line)
                line = line.strip()
                if line:
                    if "MESH_OK" in line:
                        mesh_ok = True
                    else:
                        self.log_msg.emit(f"[Gmsh] {line}")
        proc.wait()
        if proc.returncode != 0 or not mesh_ok:
            try:
                tail = "\n".join(
                    mesh_log.read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-25:]
                )
            except Exception:
                tail = "(mesh log unavailable)"
            raise RuntimeError(
                f"Mesh generation subprocess failed (exit code {proc.returncode}).\n"
                f"See {mesh_log} for full output. Last lines:\n{tail}"
            )
        for f in (params_file, script_file):
            try:
                f.unlink()
            except Exception:
                pass


# ── Main workspace ────────────────────────────────────────────────────────────
class CFDWorkspace(QWidget):
    def __init__(self, engine, assembly_provider=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.assembly_provider = assembly_provider
        self._result        = None
        self._solver_thread = None
        self._pp_thread     = None    # post-processing background thread
        self._volume_mesh   = None
        self._surface_mesh  = None
        self._current_stl: Path | None = None
        # External CAD mode: source file + its measurements (see _load_cad).
        self._cad_path: Path | None = None
        self._cad_info: dict | None = None
        self._res_iters: list = []
        self._res_vals:  list = []
        self._v_inf: float = 263.0   # freestream speed (m/s), updated from result
        self._mach:  float = 0.8
        # Sweep (polar) state
        self._sweep_thread = None
        self._sweep_data   = None    # cfd.sweep.SweepData once a sweep runs
        self._setup_ui()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([320, 860, 320])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ── LEFT: scrollable setup panel ─────────────────────────────────────────
    def _build_left(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(340)
        scroll.setMinimumWidth(260)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(12)

        # Title
        title = QLabel("CFD Analysis")
        title.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 2px 0 8px 0;")
        lay.addWidget(title)

        # ── Geometry source ──
        geo_grp = QGroupBox("Geometry Source")
        gl = QVBoxLayout(geo_grp)
        gl.setSpacing(6)
        self._rb_assembly = QRadioButton("Use current rocket design")
        self._rb_cad      = QRadioButton("Import external CAD file")
        self._rb_assembly.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._rb_assembly)
        bg.addButton(self._rb_cad)
        self._rb_assembly.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
        self._rb_cad.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
        gl.addWidget(self._rb_assembly)
        gl.addWidget(self._rb_cad)

        self._cad_lbl = QLabel("No file selected")
        self._cad_lbl.setStyleSheet(f"color:{theme.LINE_STRONG}; font-size:11px; padding-left:4px;")
        self._cad_lbl.setWordWrap(True)
        gl.addWidget(self._cad_lbl)

        self._btn_browse = QPushButton(icon("browse"), "Browse CAD File…")
        self._btn_browse.setEnabled(False)
        self._btn_browse.clicked.connect(self._browse_cad)
        self._rb_cad.toggled.connect(self._btn_browse.setEnabled)
        gl.addWidget(self._btn_browse)
        lay.addWidget(geo_grp)

        # ── Imported CAD setup (visible only in CAD mode) ──
        # An arbitrary body carries none of the assumptions the rocket path
        # relies on: which axis faces the flow, and what area the force
        # coefficients should be normalised by. Both are measured
        # automatically and both can be overridden here.
        self._cad_grp = QGroupBox("Imported CAD")
        cgl = QVBoxLayout(self._cad_grp)
        cgl.setSpacing(6)

        cad_form = QFormLayout()
        cad_form.setSpacing(8)
        cad_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._cb_flow_axis = QComboBox()
        self._cb_flow_axis.addItems([
            "Auto (longest extent)", "X axis", "Y axis", "Z axis",
        ])
        self._cb_flow_axis.setToolTip(
            "Which model axis points into the freestream.\n"
            "Auto picks the longest bounding-box extent — correct for slender "
            "bodies, wrong for a wing modelled span-wise. Set it explicitly if "
            "the preview shows the body sideways to the flow."
        )
        self._cb_flow_axis.currentIndexChanged.connect(self._on_cad_option_changed)

        self._cb_cad_units = QComboBox()
        self._cb_cad_units.addItems([
            "Auto-detect", "Millimetres (mm)", "Centimetres (cm)",
            "Metres (m)", "Inches (in)", "Feet (ft)",
        ])
        self._cb_cad_units.setToolTip(
            "Unit the CAD file's coordinates are in. The model is scaled to "
            "metres on import.\n"
            "Everything downstream is SI — get this wrong and the reference "
            "area, Reynolds number and every force coefficient are wrong with "
            "it (a millimetre model read as metres is 1000x too big).\n"
            "Auto assumes millimetres for anything over 100 m across."
        )
        self._cb_cad_units.currentIndexChanged.connect(self._on_cad_option_changed)

        self._chk_wrap = QCheckBox("Wrap geometry (repair dirty CAD)")
        self._chk_wrap.setToolTip(
            "Rebuild the body as a single clean shell by contouring its "
            "distance field.\n"
            "Use when meshing fails: it is the only route that survives a "
            "self-intersecting face or an assembly whose parts interpenetrate, "
            "because it never reads the original topology.\n\n"
            "ALTERS THE GEOMETRY — the surface is offset outward, detail below "
            "the grid spacing is rounded away, and internal passages are sealed. "
            "Outer mold line only: right for external aero, wrong if you need "
            "flow through the body."
        )
        self._chk_wrap.toggled.connect(self._on_cad_option_changed)

        self._cb_wrap_res = QComboBox()
        self._cb_wrap_res.addItems(["Coarse", "Medium", "Fine"])
        self._cb_wrap_res.setCurrentIndex(1)
        self._cb_wrap_res.setToolTip(
            "Wrap grid spacing as a fraction of body length: coarse L/60, "
            "medium L/100, fine L/160.\n"
            "Finer keeps more detail but costs memory as the cube of the "
            "resolution."
        )
        self._cb_wrap_res.setEnabled(False)
        self._chk_wrap.toggled.connect(self._cb_wrap_res.setEnabled)
        self._cb_wrap_res.currentIndexChanged.connect(self._on_cad_option_changed)

        self._sp_ref_area = QDoubleSpinBox()
        self._sp_ref_area.setRange(0.0, 1e6)
        self._sp_ref_area.setDecimals(6)
        self._sp_ref_area.setSuffix(" m²")
        self._sp_ref_area.setSpecialValueText("Auto (frontal)")
        self._sp_ref_area.setValue(0.0)
        self._sp_ref_area.setToolTip(
            "Reference area for Cd/Cl. 0 = the measured projected frontal "
            "area. Set it to publish coefficients about another reference "
            "(e.g. wing planform area)."
        )

        self._sp_ref_len = QDoubleSpinBox()
        self._sp_ref_len.setRange(0.0, 1e6)
        self._sp_ref_len.setDecimals(5)
        self._sp_ref_len.setSuffix(" m")
        self._sp_ref_len.setSpecialValueText("Auto (bbox)")
        self._sp_ref_len.setValue(0.0)
        self._sp_ref_len.setToolTip(
            "Reference length for Cm and Reynolds. 0 = the bounding-box extent "
            "along the flow axis. Set it to a mean chord for wing-type bodies."
        )

        cad_form.addRow("Units:", self._cb_cad_units)
        cad_form.addRow("Repair:", self._chk_wrap)
        cad_form.addRow("Wrap detail:", self._cb_wrap_res)
        cad_form.addRow("Flow axis:", self._cb_flow_axis)
        cad_form.addRow("Ref. area:", self._sp_ref_area)
        cad_form.addRow("Ref. length:", self._sp_ref_len)
        cgl.addLayout(cad_form)

        self._cad_info_lbl = QLabel("Import a file to see its measurements.")
        self._cad_info_lbl.setWordWrap(True)
        self._cad_info_lbl.setStyleSheet(
            f"color:{theme.TEXT_DIM}; font-size:10px; padding:4px 2px 0 2px;"
        )
        cgl.addWidget(self._cad_info_lbl)

        lay.addWidget(self._cad_grp)
        self._cad_grp.setVisible(False)
        self._rb_cad.toggled.connect(self._cad_grp.setVisible)

        # ── Flow conditions ──
        flow_grp = QGroupBox("Flow Conditions")
        fl = QFormLayout(flow_grp)
        fl.setSpacing(8)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._sp_mach = QDoubleSpinBox()
        self._sp_mach.setRange(0.01, 10.0); self._sp_mach.setValue(0.8)
        self._sp_mach.setSingleStep(0.05); self._sp_mach.setDecimals(2)

        self._sp_alt = QDoubleSpinBox()
        self._sp_alt.setRange(0, 50000); self._sp_alt.setValue(3000)
        self._sp_alt.setSuffix(" m"); self._sp_alt.setSingleStep(500)

        self._sp_aoa = QDoubleSpinBox()
        self._sp_aoa.setRange(-30, 30); self._sp_aoa.setValue(0)
        self._sp_aoa.setSuffix(" °")

        self._cb_turb = QComboBox()
        # SU2 supports SA and SST only \u2014 no k-epsilon model exists in SU2.
        self._cb_turb.addItems(["Euler", "Laminar", "Spalart-Allmaras", "k-\u03c9 SST"])
        self._cb_turb.setCurrentIndex(3)  # SST default

        fl.addRow("Mach:", self._sp_mach)
        fl.addRow("Altitude:", self._sp_alt)
        fl.addRow("Angle of Attack:", self._sp_aoa)
        fl.addRow("Turbulence Model:", self._cb_turb)
        lay.addWidget(flow_grp)

        # ── Atmosphere & flow readout ──
        atm_grp = QGroupBox("Atmosphere & Flow")
        al = QFormLayout(atm_grp)
        al.setSpacing(6)
        al.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_P   = _make_val_label(); al.addRow("Pressure:",       self._lbl_P)
        self._lbl_T   = _make_val_label(); al.addRow("Temperature:",    self._lbl_T)
        self._lbl_rho = _make_val_label(); al.addRow("Density:",        self._lbl_rho)
        self._lbl_a   = _make_val_label(); al.addRow("Speed of Sound:", self._lbl_a)
        self._lbl_Re  = _make_val_label(); al.addRow("Reynolds:",       self._lbl_Re)
        self._lbl_q   = _make_val_label(); al.addRow("Dyn. Pressure:",  self._lbl_q)
        lay.addWidget(atm_grp)

        self._sp_mach.valueChanged.connect(self._update_isa)
        self._sp_alt.valueChanged.connect(self._update_isa)
        self._update_isa()

        # ── Fidelity ──
        # These two live OUTSIDE the sweep panel deliberately. The hybrid mode
        # was previously a sweep-only control, so the single-run button — the
        # primary path — always used whatever the turbulence combo said, default
        # k-omega SST. On a tet-only mesh that is wall-unresolved RANS, and a
        # measured SST solution on this mesh family returned a skin friction of
        # Cf = 3.2e-6 against a flat-plate 1.9e-3, i.e. ~600x low. The setting
        # that exists to avoid exactly that could not be reached from the run
        # the users actually press.
        fid_grp = QGroupBox("Fidelity")
        fil = QVBoxLayout(fid_grp)
        fil.setSpacing(6)

        self._chk_euler_fric = QCheckBox("Euler + flat-plate friction (recommended)")
        self._chk_euler_fric.setChecked(True)
        self._chk_euler_fric.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
        self._chk_euler_fric.setToolTip(
            "Solve inviscid (Euler) and add an analytic skin-friction build-up\n"
            "(Schlichting flat plate + form factors) to Cd.\n\n"
            "Applies to single runs AND sweeps. Recommended whenever the mesh\n"
            "has no prism boundary layer: RANS cannot resolve the wall there,\n"
            "so its skin friction is not merely approximate, it is absent.\n"
            "Uncheck to use the turbulence model selected above."
        )
        fil.addWidget(self._chk_euler_fric)

        self._chk_bl_prisms = QCheckBox("Prism boundary layer (unsupported)")
        self._chk_bl_prisms.setChecked(False)
        self._chk_bl_prisms.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
        self._chk_bl_prisms.setToolTip(
            "Extrude real prism layers off the wall instead of meshing the\n"
            "boundary layer with tetrahedra.\n\n"
            "UNSUPPORTED - leave this off unless you are developing it.\n\n"
            "On the canonical rocket the mesh BUILDS and is structurally\n"
            "sound (5.3M cells, 141k prisms, wall manifold, no inverted\n"
            "cells) and SU2 then will not solve it: the density residual\n"
            "fell ZERO decades in 98 iterations and drag read 0.898 against\n"
            "0.343 from the supported path.\n\n"
            "Not a settings problem. gmsh's extrusion has no corner\n"
            "treatment, so fronts collide at a sharp nose tip or a fin root.\n"
            "It also blunts the nose apex to let the stack wrap it, which\n"
            "changes the geometry being solved.\n\n"
            "For trustworthy drag use 'Euler + flat-plate friction' above.\n"
            "If this fails it refuses in ~30 s with the reason - no retry,\n"
            "no silent fallback."
        )
        fil.addWidget(self._chk_bl_prisms)
        lay.addWidget(fid_grp)

        # ── Analysis mode: single point vs sweep (polar) ──
        mode_grp = QGroupBox("Analysis Mode")
        mol = QVBoxLayout(mode_grp)
        mol.setSpacing(6)
        self._rb_single = QRadioButton("Single point")
        self._rb_sweep  = QRadioButton("Sweep (polar)")
        self._rb_single.setChecked(True)
        mode_bg = QButtonGroup(self)
        mode_bg.addButton(self._rb_single)
        mode_bg.addButton(self._rb_sweep)
        for rb in (self._rb_single, self._rb_sweep):
            rb.setStyleSheet(f"color:{theme.TEXT}; font-size:12px;")
            mol.addWidget(rb)

        # Sweep parameter sub-panel (hidden until Sweep selected)
        self._sweep_widget = QWidget()
        swl = QFormLayout(self._sweep_widget)
        swl.setContentsMargins(0, 4, 0, 0)
        swl.setSpacing(8)
        swl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._cb_sweep_var = QComboBox()
        self._cb_sweep_var.addItems(["Angle of Attack", "Mach"])
        self._cb_sweep_var.currentIndexChanged.connect(self._on_sweep_var_changed)

        self._sp_sw_start = QDoubleSpinBox()
        self._sp_sw_start.setRange(-30, 30); self._sp_sw_start.setDecimals(2)
        self._sp_sw_start.setValue(-4.0)
        self._sp_sw_stop = QDoubleSpinBox()
        self._sp_sw_stop.setRange(-30, 30); self._sp_sw_stop.setDecimals(2)
        self._sp_sw_stop.setValue(12.0)
        self._sp_sw_step = QDoubleSpinBox()
        self._sp_sw_step.setRange(0.05, 10.0); self._sp_sw_step.setDecimals(2)
        self._sp_sw_step.setValue(2.0)

        swl.addRow("Variable:", self._cb_sweep_var)
        swl.addRow("Start:", self._sp_sw_start)
        swl.addRow("Stop:",  self._sp_sw_stop)
        swl.addRow("Step:",  self._sp_sw_step)

        # (The Euler + flat-plate friction control now lives in the Fidelity
        # group above, where it applies to single runs as well as sweeps.)

        self._lbl_sweep_info = QLabel("9 points")
        self._lbl_sweep_info.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px; padding:2px 0;")
        self._lbl_sweep_info.setWordWrap(True)
        swl.addRow("", self._lbl_sweep_info)

        for sp in (self._sp_sw_start, self._sp_sw_stop, self._sp_sw_step):
            sp.valueChanged.connect(self._update_sweep_info)

        self._sweep_widget.setVisible(False)
        mol.addWidget(self._sweep_widget)
        self._rb_sweep.toggled.connect(self._sweep_widget.setVisible)
        lay.addWidget(mode_grp)

        # ── Streamline options ──
        self._stream_grp = QGroupBox("Streamline Options")
        stl = QFormLayout(self._stream_grp)
        stl.setSpacing(6)
        self._sp_seed_density = QSpinBox()
        self._sp_seed_density.setRange(8, 64); self._sp_seed_density.setValue(24)
        stl.addRow("Seed Density:", self._sp_seed_density)
        self._cb_stream_color = QComboBox()
        self._cb_stream_color.addItems(["Velocity", "Pressure", "Mach"])
        stl.addRow("Color By:", self._cb_stream_color)
        self._cb_stream_type = QComboBox()
        self._cb_stream_type.addItems(["Tubes", "Lines", "Ribbons"])
        stl.addRow("Type:", self._cb_stream_type)
        self._stream_grp.setVisible(False)  # hidden until CFD results ready
        lay.addWidget(self._stream_grp)

        # ── Display Options (engineering-grade controls) ──
        self._disp_grp = QGroupBox("Display Options")
        dl = QFormLayout(self._disp_grp)
        dl.setSpacing(6)
        dl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Scalar range editor
        rng_lay = QHBoxLayout()
        self._sp_smin = QDoubleSpinBox()
        self._sp_smin.setRange(-1e9, 1e9); self._sp_smin.setDecimals(4)
        self._sp_smin.setSpecialValueText("auto")
        self._sp_smin.setValue(-1e9)  # default to "auto"
        self._sp_smin.setStyleSheet("max-width:90px;")
        self._sp_smax = QDoubleSpinBox()
        self._sp_smax.setRange(-1e9, 1e9); self._sp_smax.setDecimals(4)
        self._sp_smax.setSpecialValueText("auto")
        self._sp_smax.setValue(-1e9)  # default to "auto"
        self._sp_smax.setStyleSheet("max-width:90px;")
        self._sp_smin.editingFinished.connect(self._schedule_refresh)
        self._sp_smax.editingFinished.connect(self._schedule_refresh)
        rng_lay.addWidget(self._sp_smin)
        rng_lay.addWidget(QLabel("\u2013"))
        rng_lay.addWidget(self._sp_smax)
        dl.addRow("Scalar Range:", rng_lay)

        # Log scale toggle
        self._chk_log_scale = QCheckBox("Logarithmic")
        self._chk_log_scale.setStyleSheet(f"color:{theme.TEXT};")
        self._chk_log_scale.stateChanged.connect(lambda _: self._schedule_refresh())
        dl.addRow("Scale:", self._chk_log_scale)

        # Contour opacity slider
        self._sl_opacity = QSlider(Qt.Orientation.Horizontal)
        self._sl_opacity.setRange(10, 100); self._sl_opacity.setValue(100)
        self._sl_opacity.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{theme.RAISED};height:6px;border-radius:3px;}}"
            f"QSlider::handle:horizontal{{background:{theme.ACCENT};width:14px;margin:-4px 0;border-radius:7px;}}"
        )
        self._lbl_opacity = QLabel("1.00")
        self._lbl_opacity.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;min-width:30px;")
        self._sl_opacity.valueChanged.connect(lambda v: (self._lbl_opacity.setText(f"{v/100:.2f}"), self._schedule_refresh()))
        op_lay = QHBoxLayout()
        op_lay.addWidget(self._sl_opacity); op_lay.addWidget(self._lbl_opacity)
        dl.addRow("Opacity:", op_lay)

        # Slice position slider
        self._sl_slice = QSlider(Qt.Orientation.Horizontal)
        self._sl_slice.setRange(-100, 100); self._sl_slice.setValue(0)
        self._sl_slice.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{theme.RAISED};height:6px;border-radius:3px;}}"
            f"QSlider::handle:horizontal{{background:{theme.OK};width:14px;margin:-4px 0;border-radius:7px;}}"
        )
        self._lbl_slice = QLabel("0.00")
        self._lbl_slice.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;min-width:30px;")
        self._sl_slice.valueChanged.connect(lambda v: (self._lbl_slice.setText(f"{v/100:.2f}"), self._schedule_refresh()))
        sl_lay = QHBoxLayout()
        sl_lay.addWidget(self._sl_slice); sl_lay.addWidget(self._lbl_slice)
        dl.addRow("Slice Y Offset:", sl_lay)

        # Iso threshold slider (log-scale, for Q/Lambda2)
        self._sl_iso = QSlider(Qt.Orientation.Horizontal)
        self._sl_iso.setRange(1, 100); self._sl_iso.setValue(85)
        self._sl_iso.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{theme.RAISED};height:6px;border-radius:3px;}}"
            "QSlider::handle:horizontal{background:#d2a8ff;width:14px;margin:-4px 0;border-radius:7px;}"
        )
        self._lbl_iso = QLabel("85%")
        self._lbl_iso.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;min-width:30px;")
        self._sl_iso.valueChanged.connect(lambda v: (self._lbl_iso.setText(f"{v}%"), self._schedule_refresh()))
        iso_lay = QHBoxLayout()
        iso_lay.addWidget(self._sl_iso); iso_lay.addWidget(self._lbl_iso)
        dl.addRow("Iso Threshold:", iso_lay)

        # Colormap selector
        self._cb_cmap = QComboBox()
        self._cb_cmap.addItems(["auto", "turbo", "plasma", "inferno", "viridis",
                                "coolwarm", "RdBu_r", "jet", "hot", "cividis"])
        self._cb_cmap.currentIndexChanged.connect(lambda _: self._schedule_refresh())
        dl.addRow("Colormap:", self._cb_cmap)

        # Screenshot button
        self._btn_screenshot = QPushButton(icon("screenshot"), "Screenshot (4K)")
        self._btn_screenshot.clicked.connect(self._screenshot_hq)
        dl.addRow("", self._btn_screenshot)

        self._disp_grp.setVisible(False)  # hidden until CFD results ready
        lay.addWidget(self._disp_grp)

        # ── Mesh settings ──
        mesh_grp = QGroupBox("Mesh Settings")
        ml = QFormLayout(mesh_grp)
        ml.setSpacing(8)
        ml.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._cb_ref = QComboBox()
        self._cb_ref.addItems([
            "Coarse (fast)", "Medium", "Fine (accurate)",
            "Very Fine", "Ultra Fine", "Custom…"
        ])
        self._cb_ref.setCurrentIndex(1)
        self._cb_ref.currentIndexChanged.connect(self._on_ref_changed)

        # No "BL Inflation" or "target y+" control: the mesher is tet-only. A
        # setting here would be read, threaded through the config and the
        # meshing subprocess, and then ignored — implying an inflation layer the
        # mesh does not have. Prism extrusion runs on the OCC boolean-cut domain
        # but leaves the prisms overlapping the tets rather than bounded by
        # them, which SU2 cannot converge on (cfd/meshing.py step 7). CFDConfig
        # still carries bl_layers/bl_growth at their defaults for call
        # compatibility.

        self._sp_iter = QSpinBox()
        self._sp_iter.setRange(100, 50000); self._sp_iter.setValue(5000)
        self._sp_iter.setSingleStep(500)

        import os as _os
        self._sp_cores = QSpinBox()
        self._sp_cores.setRange(0, max(1, _os.cpu_count() or 1))
        self._sp_cores.setValue(0)   # 0 = auto (all cores but one)
        self._sp_cores.setSpecialValueText("Auto")   # shown when value == 0
        self._sp_cores.setSuffix(" cores")
        self._sp_cores.setToolTip(
            "Cores SU2 may use. Auto uses all but one, so the UI stays responsive.\n"
            "The bundled SU2 is an OpenMP build, so this is a thread count on a "
            "single process. If an MPI-built SU2 and mpiexec are both found it is "
            "used as an MPI rank count instead."
        )

        ml.addRow("Refinement:", self._cb_ref)
        ml.addRow("Max Iterations:", self._sp_iter)
        ml.addRow("CPU Cores:", self._sp_cores)

        # ── Custom mesh controls (visible when "Custom…" selected) ──
        self._custom_mesh_widget = QWidget()
        cml = QFormLayout(self._custom_mesh_widget)
        cml.setSpacing(6)
        cml.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        cml.setContentsMargins(0, 4, 0, 0)

        # Target element count
        self._sp_target_count = QSpinBox()
        self._sp_target_count.setRange(10000, 5000000)
        self._sp_target_count.setValue(200000)
        self._sp_target_count.setSingleStep(50000)
        self._sp_target_count.setSuffix(" elements")
        self._sp_target_count.valueChanged.connect(self._on_target_count_changed)
        cml.addRow("Target Count:", self._sp_target_count)

        # Wall element size (mm)
        self._sp_wall_size = QDoubleSpinBox()
        self._sp_wall_size.setRange(0.1, 50.0)
        self._sp_wall_size.setValue(5.0)
        self._sp_wall_size.setDecimals(2)
        self._sp_wall_size.setSuffix(" mm")
        self._sp_wall_size.setSingleStep(0.5)
        self._sp_wall_size.valueChanged.connect(self._on_wall_size_changed)
        cml.addRow("Wall Element:", self._sp_wall_size)

        # Power slider (log-mapped)
        self._sl_power = QSlider(Qt.Orientation.Horizontal)
        self._sl_power.setRange(0, 100)
        self._sl_power.setValue(40)
        self._sl_power.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{theme.RAISED};height:6px;border-radius:3px;}}"
            f"QSlider::handle:horizontal{{background:{theme.ACCENT};width:14px;margin:-4px 0;border-radius:7px;}}"
        )
        self._sl_power.valueChanged.connect(self._on_power_slider_changed)
        self._lbl_power = QLabel("40%")
        self._lbl_power.setStyleSheet(f"color:{theme.TEXT_DIM};font-size:11px;min-width:30px;")
        pw_lay = QHBoxLayout()
        pw_lay.addWidget(self._sl_power)
        pw_lay.addWidget(self._lbl_power)
        cml.addRow("Power:", pw_lay)

        # Estimate label
        self._lbl_estimate = QLabel("≈ 200K elements")
        self._lbl_estimate.setStyleSheet(
            f"color:{theme.ACCENT}; font-size:11px; font-weight:600; padding:2px 0;"
        )
        cml.addRow("", self._lbl_estimate)

        # Warning label (shown for high element counts)
        self._lbl_mesh_warn = QLabel("")
        self._lbl_mesh_warn.setStyleSheet(
            f"color:{theme.WARN}; font-size:11px; font-weight:600; padding:2px 0;"
        )
        self._lbl_mesh_warn.setWordWrap(True)
        self._lbl_mesh_warn.setVisible(False)
        cml.addRow("", self._lbl_mesh_warn)

        self._custom_mesh_widget.setVisible(False)
        ml.addRow(self._custom_mesh_widget)

        # Warning for preset ultra-fine
        self._lbl_preset_warn = QLabel("")
        self._lbl_preset_warn.setStyleSheet(
            f"color:{theme.WARN}; font-size:11px; font-weight:600; padding:2px 0;"
        )
        self._lbl_preset_warn.setWordWrap(True)
        self._lbl_preset_warn.setVisible(False)
        ml.addRow(self._lbl_preset_warn)

        lay.addWidget(mesh_grp)

        # Track internal sync to avoid feedback loops
        self._mesh_sync_guard = False

        # ── Action buttons ──
        self._btn_export = QPushButton(icon("export"), "Export Geometry to STL")
        self._btn_export.clicked.connect(self._export_geometry)

        self._btn_run = QPushButton(icon("run", color="#fff"), "Run CFD Analysis")
        self._btn_run.setProperty("primary", True)
        self._btn_run.clicked.connect(self._run_cfd)

        self._btn_stop = QPushButton(icon("stop"), "Stop Solver")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_cfd)

        lay.addWidget(self._btn_export)
        lay.addWidget(self._btn_run)
        lay.addWidget(self._btn_stop)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setFixedHeight(20)
        self._progress.setTextVisible(True)
        self._progress.setFormat(" %p% (%v / %m iterations)")
        self._progress.setStyleSheet(
            f"QProgressBar{{background:{theme.RAISED}; border-radius:4px; border:1px solid {theme.LINE}; color:{theme.TEXT}; font-weight:bold; text-align:center;}}"
            f"QProgressBar::chunk{{background:{theme.ACCENT_DEEP}; border-radius:3px;}}"
        )
        lay.addWidget(self._progress)

        # ── Export results (PDF + CSV) ──
        exp_group = QGroupBox("Export Results")
        eg = QHBoxLayout(); eg.setSpacing(6)
        self._btn_export_pdf = QPushButton(icon("report"), "PDF")
        self._btn_export_pdf.setEnabled(False)
        self._btn_export_pdf.clicked.connect(self._export_pdf)
        eg.addWidget(self._btn_export_pdf)
        self._btn_export_csv = QPushButton(icon("export"), "CSV")
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.clicked.connect(self._export_results_csv)
        eg.addWidget(self._btn_export_csv)
        exp_group.setLayout(eg)
        lay.addWidget(exp_group)

        lay.addStretch()

        scroll.setWidget(inner)
        return scroll

    # ── CENTER: tabbed viewer (3D field + polar curves) ───────────────────────
    def _build_center(self):
        self._center_tabs = QTabWidget()
        self._center_tabs.setStyleSheet(
            f"QTabWidget::pane{{border:none;background:{theme.BG};}}"
            f"QTabBar::tab{{background:{theme.PANEL};color:{theme.TEXT_DIM};padding:6px 16px;"
            f"font-size:12px;font-weight:600;border:1px solid {theme.RAISED};border-bottom:none;}}"
            f"QTabBar::tab:selected{{background:{theme.BG};color:{theme.ACCENT};}}"
            "QTabBar::scroller{width:30px;}"
            f"QTabBar QToolButton{{background:{theme.RAISED};border:1px solid {theme.LINE};"
            f"border-radius:4px;margin:2px 1px;width:22px;color:{theme.TEXT};}}"
            f"QTabBar QToolButton:hover{{background:{theme.ACCENT_DEEP};border-color:{theme.ACCENT_DEEP};}}"
            f"QTabBar QToolButton:disabled{{background:{theme.PANEL};border-color:{theme.RAISED};}}"
        )
        self._center_tabs.addTab(self._build_field_view(), "3D Field")
        self._center_tabs.addTab(self._build_polar_view(), "Polars")
        # Matplotlib canvases drawn while their tab is hidden cache a blank
        # buffer (zero size) and won't repaint on show — force a redraw when
        # the Polars tab becomes visible.
        self._center_tabs.currentChanged.connect(self._on_center_tab_changed)
        return self._center_tabs

    def _on_center_tab_changed(self, idx):
        if idx == 1 and self._sweep_data and self._sweep_data.points:
            self._refresh_polar()

    def showEvent(self, event):
        # Returning to the CFD workspace tab: repaint the polar canvas, which
        # may have cached a blank buffer if it was drawn while not visible.
        super().showEvent(event)
        if (self._center_tabs.currentIndex() == 1
                and self._sweep_data and self._sweep_data.points):
            self._refresh_polar()

    # ── CENTER tab 1: 3D viewer ───────────────────────────────────────────────
    def _build_field_view(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        bar = QWidget()
        bar.setStyleSheet(f"background:{theme.PANEL}; border-bottom:1px solid {theme.RAISED};")
        bar.setFixedHeight(44)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 8, 0)

        lbl = QLabel("CFD Visualization")
        lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:700; font-size:13px;")
        bl.addWidget(lbl)
        bl.addStretch()

        bl.addWidget(QLabel("View:"))
        self._vis_combo = QComboBox()
        self._vis_combo.addItems([
            "Geometry Preview",           # 0
            "Pressure \u2014 Surface Cp",      # 1
            "Pressure \u2014 Volume Slice",    # 2
            "Temperature \u2014 Volume Slice", # 3
            "Velocity \u2014 Magnitude Slice", # 4
            "Velocity \u2014 Streamlines",     # 5
            "Mach \u2014 Volume Slice",        # 6
            "Density \u2014 Volume Slice",     # 7
            "Vorticity Magnitude",        # 8
            "Q-Criterion Iso-Surface",    # 9
            "Cp \u2014 Volume Slice",          # 10
            "Compression Region",         # 11
            "Boundary Layer \u2014 Y+",        # 12
            "Wall Shear Stress",          # 13
            "Force Vectors",              # 14
            "Lambda-2 Criterion",         # 15
            "Temperature \u2014 Surface Contour",  # 16
        ])

        self._vis_combo.setEnabled(False)
        self._vis_combo.currentIndexChanged.connect(self._refresh_vis)
        bl.addWidget(self._vis_combo)

        self._chk_interactive_slice = QCheckBox("Interactive Slice")
        self._chk_interactive_slice.setStyleSheet(f"color:{theme.TEXT}; font-weight:600;")
        self._chk_interactive_slice.setEnabled(False)
        self._chk_interactive_slice.stateChanged.connect(self._toggle_interactive_slice)
        bl.addWidget(self._chk_interactive_slice)

        self._chk_contour_lines = QCheckBox("Contour Lines")
        self._chk_contour_lines.setStyleSheet(f"color:{theme.TEXT}; font-weight:600;")
        self._chk_contour_lines.setChecked(False)
        self._chk_contour_lines.stateChanged.connect(self._refresh_vis)
        bl.addWidget(self._chk_contour_lines)

        self._chk_mesh_edges = QCheckBox("Mesh Edges")
        self._chk_mesh_edges.setStyleSheet(f"color:{theme.TEXT}; font-weight:600;")
        self._chk_mesh_edges.setChecked(False)
        self._chk_mesh_edges.stateChanged.connect(self._refresh_vis)
        bl.addWidget(self._chk_mesh_edges)

        # Compression sensor selector (visible only on compression region view)
        self._cb_shock_sensor = QComboBox()
        self._cb_shock_sensor.addItems([
            "Pressure Gradient", "Ducros Sensor", "Dilatation",
            "Mach Gradient", "Entropy Gradient"
        ])
        self._cb_shock_sensor.setFixedWidth(140)
        self._cb_shock_sensor.setVisible(False)
        self._cb_shock_sensor.currentIndexChanged.connect(self._refresh_vis)
        bl.addWidget(self._cb_shock_sensor)

        # Probe mode toggle
        self._btn_probe = QPushButton("Probe")
        self._btn_probe.setCheckable(True)
        self._btn_probe.setFixedHeight(28)
        self._btn_probe.toggled.connect(self._toggle_probe_mode)
        bl.addWidget(self._btn_probe)

        btn_cam = QPushButton(icon("reset_view"), "Reset Camera")
        btn_cam.setFixedHeight(28)
        btn_cam.clicked.connect(lambda: self._plotter.reset_camera())
        bl.addWidget(btn_cam)
        lay.addWidget(bar)

        # PyVista viewer
        frame = QFrame()
        frame.setStyleSheet(f"background:{theme.BG};")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(0, 0, 0, 0)
        self._plotter = QtInteractor(frame, auto_update=False)
        self._plotter.set_background(theme.BG)
        try:
            self._plotter.enable_ssao(radius=0.25, bias=0.002, kernel_size=512, blur=True)
            self._plotter.enable_anti_aliasing('msaa')
        except Exception as e:
            logger.warning(f"Failed to enable SSAO: {e}")
        try:
            self._plotter.enable_depth_peeling(number_of_peeling_layers=12)
        except Exception:
            pass
        # Professional multi-light setup
        try:
            self._plotter.enable_lightkit()
        except Exception:
            pass
        fl.addWidget(self._plotter)
        lay.addWidget(frame, 1)
        # Status bar
        self._status_lbl = QLabel("Load a rocket or import a CAD file to begin.")
        self._status_lbl.setStyleSheet(
            f"color:{theme.TEXT_DIM}; padding:5px 12px; font-size:11px;"
            f"background:{theme.PANEL}; border-top:1px solid {theme.RAISED};"
        )
        self._status_lbl.setFixedHeight(28)
        lay.addWidget(self._status_lbl)
        return w

    # ── CENTER tab 2: polar curves ────────────────────────────────────────────
    def _build_polar_view(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Top bar
        bar = QWidget()
        bar.setStyleSheet(f"background:{theme.PANEL}; border-bottom:1px solid {theme.RAISED};")
        bar.setFixedHeight(44)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 0, 8, 0)
        lbl = QLabel("Aerodynamic Polars")
        lbl.setStyleSheet(f"color:{theme.ACCENT}; font-weight:700; font-size:13px;")
        bl.addWidget(lbl)
        bl.addStretch()
        bl.addWidget(QLabel("Curve:"))
        self._cb_polar = QComboBox()
        self._cb_polar.currentIndexChanged.connect(self._refresh_polar)
        bl.addWidget(self._cb_polar)
        self._btn_polar_export = QPushButton("Export CSV")
        self._btn_polar_export.setFixedHeight(28)
        self._btn_polar_export.setEnabled(False)
        self._btn_polar_export.clicked.connect(self._export_polar_csv)
        bl.addWidget(self._btn_polar_export)
        lay.addWidget(bar)

        # Plot canvas
        try:
            from ui.widgets.plot_widget import PlotWidget
            self._polar_plot = PlotWidget(title="", xlabel="", ylabel="")
        except Exception:
            self._polar_plot = QLabel("Plot unavailable")
        lay.addWidget(self._polar_plot, 1)

        # Derived-metrics strip
        self._lbl_polar_metrics = QLabel(
            "Run a sweep to populate Cl/Cd/Cm vs AoA or Cd vs Mach curves."
        )
        self._lbl_polar_metrics.setWordWrap(True)
        self._lbl_polar_metrics.setStyleSheet(
            f"color:{theme.TEXT_DIM}; padding:8px 12px; font-size:12px;"
            f"background:{theme.PANEL}; border-top:1px solid {theme.RAISED};"
        )
        lay.addWidget(self._lbl_polar_metrics)
        return w

    # ── RIGHT: results panel ──────────────────────────────────────────────────
    def _build_right(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        scroll.setMinimumWidth(280)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(12, 14, 12, 14)
        lay.setSpacing(10)

        title = QLabel("Results")
        title.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600; letter-spacing: 1px; padding: 2px 0 8px 0;")
        lay.addWidget(title)

        # ── Aerodynamic Coefficients ──
        coef_grp = QGroupBox("Aerodynamic Coefficients")
        cf = QFormLayout(coef_grp)
        cf.setSpacing(6)
        cf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        # Cd = Pressure + Friction (both integrated off the wall solution).
        # Base and Wave are a further split OF the pressure term, so they are
        # indented under it and must not be added to the two above — the old
        # flat list invited exactly that misreading.
        self._lbl_cd   = _make_val_label(); cf.addRow("Total Cd:",      self._lbl_cd)
        self._lbl_cdp  = _make_val_label(); cf.addRow("  Pressure:",    self._lbl_cdp)
        self._lbl_cdf  = _make_val_label(); cf.addRow("  Friction:",    self._lbl_cdf)
        self._lbl_cdb  = _make_val_label(); cf.addRow("    ↳ Base:",    self._lbl_cdb)
        self._lbl_cdw  = _make_val_label(); cf.addRow("    ↳ Wave:",    self._lbl_cdw)
        for _lbl, _tip in (
            (self._lbl_cdp, "Pressure drag, integrated over the wall.\n"
                            "Cd = Pressure + Friction."),
            (self._lbl_cdf, "Skin-friction drag, integrated over the wall.\n"
                            "Under-resolved on this tet-only mesh — check the\n"
                            "Y+ Range below; above 30 it reads low."),
            (self._lbl_cdb, "Base drag: pressure integrated over the rearward-\n"
                            "facing surfaces only. A COMPONENT of Pressure above,\n"
                            "not an extra term."),
            (self._lbl_cdw, "Wave drag: the forebody part of the pressure drag\n"
                            "(Pressure - Base). Reported only at M >= 0.8;\n"
                            "an inviscid subsonic body has no wave drag."),
        ):
            _lbl.setToolTip(_tip)
        self._lbl_cl   = _make_val_label(); cf.addRow("Lift Cl:",       self._lbl_cl)
        self._lbl_cm   = _make_val_label(); cf.addRow("Moment Cm:",     self._lbl_cm)
        self._lbl_conv = _make_val_label(); cf.addRow("Converged:",     self._lbl_conv)
        # Five decimal places on a coefficient is a claim about precision. The
        # panel used to make it next to a green "Converged: Yes" and nothing
        # else, while the log carried the reason the friction component was
        # missing. These two rows put the caveat where the number is.
        self._lbl_conv_why = _make_val_label()
        self._lbl_conv_why.setWordWrap(True)
        self._lbl_conv_why.setStyleSheet(
            f"color:{theme.TEXT_DIM}; font-size:10px;")
        cf.addRow("", self._lbl_conv_why)
        self._lbl_trust = _make_val_label()
        self._lbl_trust.setWordWrap(True)
        self._lbl_trust.setVisible(False)
        cf.addRow("", self._lbl_trust)
        lay.addWidget(coef_grp)

        # ── Forces & CP ──
        force_grp = QGroupBox("Forces & Center of Pressure")
        ff = QFormLayout(force_grp)
        ff.setSpacing(6)
        ff.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_fa   = _make_val_label(); ff.addRow("Axial Force:",   self._lbl_fa)
        self._lbl_fn   = _make_val_label(); ff.addRow("Normal Force:",  self._lbl_fn)
        self._lbl_cp   = _make_val_label(); ff.addRow("CP Location:",   self._lbl_cp)
        lay.addWidget(force_grp)

        # ── Solver & Flow Info ──
        solver_grp = QGroupBox("Solver & Flow")
        sf = QFormLayout(solver_grp)
        sf.setSpacing(6)
        sf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_solver = _make_val_label(); sf.addRow("Solver:",    self._lbl_solver)
        self._lbl_turb_r = _make_val_label(); sf.addRow("Model:",     self._lbl_turb_r)
        self._lbl_re_r   = _make_val_label(); sf.addRow("Reynolds:",  self._lbl_re_r)
        self._lbl_q_r    = _make_val_label(); sf.addRow("Dyn. Press:", self._lbl_q_r)
        lay.addWidget(solver_grp)

        # ── Mesh Statistics ──
        mesh_grp = QGroupBox("Mesh Quality")
        mf = QFormLayout(mesh_grp)
        mf.setSpacing(6)
        mf.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self._lbl_cells  = _make_val_label(); mf.addRow("Cells:",       self._lbl_cells)
        self._lbl_nodes  = _make_val_label(); mf.addRow("Nodes:",       self._lbl_nodes)
        self._lbl_mq     = _make_val_label(); mf.addRow("Quality:",     self._lbl_mq)
        self._lbl_ar     = _make_val_label(); mf.addRow("Aspect Ratio:", self._lbl_ar)
        self._lbl_growth = _make_val_label(); mf.addRow("Growth ratio:", self._lbl_growth)
        self._lbl_growth.setToolTip(
            "How much the cell size changes between face-adjacent cells\n"
            "(99th percentile, and the worst pair).\n\n"
            "This is what bounds a finite-volume scheme's accuracy: every\n"
            "flux and every gradient is evaluated across a face. Aspect ratio\n"
            "and skew describe one cell's shape and cannot see it.\n"
            "Target <= 1.2; above 2 expect elevated numerical entropy."
        )
        self._lbl_yp_r   = _make_val_label(); mf.addRow("Y+ Range:",    self._lbl_yp_r)
        self._lbl_yp_r.setToolTip(
            "Wall y+ over the surface. Area-weighted median, plus the range.\n\n"
            "y+ < 5 is needed for a turbulence model to integrate to the wall,\n"
            "and there is no wall function behind it. Above ~30 the reported\n"
            "skin friction is not physical."
        )
        lay.addWidget(mesh_grp)

        # Inject button
        self._btn_inject = QPushButton(icon("inject"), "Inject into Flight Simulation")
        self._btn_inject.setProperty("success", True)
        self._btn_inject.setEnabled(False)
        self._btn_inject.clicked.connect(self._inject_results)
        lay.addWidget(self._btn_inject)

        # Cp distribution plot
        cp_grp = QGroupBox("Cp Distribution")
        cpl = QVBoxLayout(cp_grp)
        cpl.setContentsMargins(4, 8, 4, 4)
        try:
            from ui.widgets.cp_plot_widget import CpPlotWidget
            self._cp_plot = CpPlotWidget()
            self._cp_plot.setMinimumHeight(180)
        except Exception:
            self._cp_plot = QLabel("Cp plot unavailable")
        cpl.addWidget(self._cp_plot)
        lay.addWidget(cp_grp)

        # Convergence residuals plot
        res_grp = QGroupBox("Convergence Residuals")
        rl = QVBoxLayout(res_grp)
        rl.setContentsMargins(4, 8, 4, 4)
        try:
            from ui.widgets.plot_widget import PlotWidget
            self._res_plot = PlotWidget(
                title="", xlabel="Iteration", ylabel="log(RMS Density)"
            )
            self._res_plot.setMinimumHeight(180)
        except Exception:
            self._res_plot = QLabel("Plot unavailable")
        rl.addWidget(self._res_plot)
        lay.addWidget(res_grp)

        # Solver log
        log_grp = QGroupBox("Solver Log")
        ll = QVBoxLayout(log_grp)
        ll.setContentsMargins(4, 8, 4, 4)
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMinimumHeight(120)
        self._log_box.setMaximumHeight(220)
        self._log_box.setStyleSheet(
            f"QTextEdit{{background:{theme.BG}; color:{theme.TEXT_DIM}; font-family:monospace;"
            f"font-size:11px; border:1px solid {theme.RAISED}; border-radius:4px; padding:4px;}}"
        )
        ll.addWidget(self._log_box)
        lay.addWidget(log_grp)

        # Export (PDF/CSV moved to the left panel; VTK + FEM mapping stay here)
        self._btn_export_vtk = QPushButton(icon("export"), "Export VTK Results")
        self._btn_export_vtk.setEnabled(False)
        self._btn_export_vtk.clicked.connect(self._export_vtk)
        lay.addWidget(self._btn_export_vtk)
        
        self._btn_export_struct = QPushButton(icon("map_fem"), "Map Pressure to FEM")
        self._btn_export_struct.setEnabled(False)
        self._btn_export_struct.clicked.connect(self._export_to_structures)
        lay.addWidget(self._btn_export_struct)
        
        lay.addStretch()

        scroll.setWidget(inner)
        return scroll

    # ── Logic ─────────────────────────────────────────────────────────────────
    def _update_isa(self):
        from cfd.solvers.base import isa_conditions
        try:
            P, T, rho = isa_conditions(self._sp_alt.value())
            a = math.sqrt(1.4 * 287.05 * T)
            V = self._sp_mach.value() * a
            mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)
            Re = rho * V * 1.0 / mu  # assume L=1m, updated after geometry
            q = 0.5 * rho * V ** 2
            self._lbl_P.setText(f"{P/1000:.2f} kPa")
            self._lbl_T.setText(f"{T:.1f} K  ({T-273.15:.1f} \u00b0C)")
            self._lbl_rho.setText(f"{rho:.4f} kg/m\u00b3")
            self._lbl_a.setText(f"{a:.1f} m/s")
            self._lbl_Re.setText(f"{Re:.2e}")
            self._lbl_q.setText(f"{q/1000:.2f} kPa")
        except Exception:
            pass

    # ── Custom mesh control handlers ──────────────────────────────────────────
    def _on_ref_changed(self, idx):
        """Show/hide custom mesh controls based on refinement selection."""
        is_custom = (idx == 5)
        self._custom_mesh_widget.setVisible(is_custom)

        # Preset warnings for expensive levels
        if idx == 3:  # Very Fine
            self._lbl_preset_warn.setText("Very Fine mesh — may take several minutes")
            self._lbl_preset_warn.setVisible(True)
        elif idx == 4:  # Ultra Fine
            self._lbl_preset_warn.setText("Ultra Fine mesh — expect 10+ minutes and high memory usage")
            self._lbl_preset_warn.setVisible(True)
        else:
            self._lbl_preset_warn.setVisible(False)

    def _on_power_slider_changed(self, value):
        """Map slider 0–100 to element count ~10K–2M (logarithmic) and sync wall size."""
        if self._mesh_sync_guard:
            return
        self._mesh_sync_guard = True
        self._lbl_power.setText(f"{value}%")
        # Log scale: 10K at 0, ~2M at 100 (cap at 5M)
        import math
        count = int(10000 * (200 ** (value / 100.0)))
        count = max(10000, min(count, 5000000))
        self._sp_target_count.setValue(count)

        # Estimate wall size from count
        s = self.engine.state
        body_r = s.diameter / 2.0 if s.diameter > 0 else 0.05
        length = s.length if s.length > 0 else 1.0
        domain_volume = (length * 10.0) * (2 * body_r * 20.0) ** 2
        lc_avg = (6.0 * domain_volume / max(count, 1000)) ** (1.0 / 3.0)
        lc_wall = max(lc_avg * 0.15, body_r * 0.005)
        lc_wall = min(lc_wall, body_r * 0.5)
        self._sp_wall_size.setValue(lc_wall * 1000.0)  # m to mm

        self._update_mesh_warnings(count)
        self._mesh_sync_guard = False

    def _on_target_count_changed(self, count):
        """Update slider and wall size from target element count."""
        if self._mesh_sync_guard:
            return
        self._mesh_sync_guard = True
        # Inverse log: slider = 100 * log(count/10000) / log(200)
        import math
        ratio = max(count / 10000.0, 1.0)
        slider_val = int(100 * math.log(ratio) / math.log(200))
        slider_val = max(0, min(slider_val, 100))
        self._sl_power.setValue(slider_val)
        self._lbl_power.setText(f"{slider_val}%")
        self._update_mesh_warnings(count)

        # Estimate wall size from count
        s = self.engine.state
        body_r = s.diameter / 2.0 if s.diameter > 0 else 0.05
        length = s.length if s.length > 0 else 1.0
        domain_volume = (length * 10.0) * (2 * body_r * 20.0) ** 2
        lc_avg = (6.0 * domain_volume / max(count, 1000)) ** (1.0 / 3.0)
        lc_wall = max(lc_avg * 0.15, body_r * 0.005)
        lc_wall = min(lc_wall, body_r * 0.5)
        self._sp_wall_size.setValue(lc_wall * 1000.0)  # m to mm

        # Update estimate label
        if count >= 1000000:
            self._lbl_estimate.setText(f"≈ {count/1e6:.1f}M elements")
        else:
            self._lbl_estimate.setText(f"≈ {count/1000:.0f}K elements")
        self._mesh_sync_guard = False

    def _on_wall_size_changed(self, size_mm):
        """Update target count and slider from wall element size."""
        if self._mesh_sync_guard:
            return
        self._mesh_sync_guard = True
        lc_wall = size_mm / 1000.0  # mm to m
        s = self.engine.state
        body_r = s.diameter / 2.0 if s.diameter > 0 else 0.05
        length = s.length if s.length > 0 else 1.0
        domain_volume = (length * 10.0) * (2 * body_r * 20.0) ** 2

        # Inverse size estimation: lc_wall = lc_avg * 0.15 => lc_avg = lc_wall / 0.15
        lc_avg = lc_wall / 0.15
        if lc_avg > 0:
            count = int(6.0 * domain_volume / (lc_avg ** 3))
            count = max(10000, min(count, 5000000))
        else:
            count = 10000

        self._sp_target_count.setValue(count)

        # Update slider
        import math
        ratio = max(count / 10000.0, 1.0)
        slider_val = int(100 * math.log(ratio) / math.log(200))
        slider_val = max(0, min(slider_val, 100))
        self._sl_power.setValue(slider_val)
        self._lbl_power.setText(f"{slider_val}%")
        self._update_mesh_warnings(count)

        # Update estimate label
        if count >= 1000000:
            self._lbl_estimate.setText(f"≈ {count/1e6:.1f}M elements")
        else:
            self._lbl_estimate.setText(f"≈ {count/1000:.0f}K elements")
        self._mesh_sync_guard = False

    def _update_mesh_warnings(self, count):
        """Show warnings for very large meshes."""
        if count >= 1000000:
            self._lbl_mesh_warn.setText(
                "Very large mesh (>1M elements) — may require 8+ GB RAM "
                "and take 15+ minutes to generate"
            )
            self._lbl_mesh_warn.setVisible(True)
        elif count >= 500000:
            self._lbl_mesh_warn.setText(
                "Fine mesh — may take 5–10 minutes to generate"
            )
            self._lbl_mesh_warn.setVisible(True)
        else:
            self._lbl_mesh_warn.setVisible(False)

    def _get_turb_key(self):
        """Map UI turbulence combo index to config key."""
        return ["Euler", "Laminar", "SA", "SST"][self._cb_turb.currentIndex()]

    def _flow_axis_key(self) -> str:
        """Map the flow-axis combo to the CFDConfig/meshing key."""
        return ["auto", "x", "y", "z"][self._cb_flow_axis.currentIndex()]

    def _wrap_resolution_key(self) -> str:
        """Map the wrap-detail combo to the analyze_cad key."""
        return ["coarse", "medium", "fine"][self._cb_wrap_res.currentIndex()]

    def _cad_units_key(self) -> str:
        """Map the units combo to the CFDConfig/analyze_cad key."""
        return ["auto", "mm", "cm", "m", "in", "ft"][
            self._cb_cad_units.currentIndex()
        ]

    def _browse_cad(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CAD File", "",
            "CAD Files (*.step *.stp *.iges *.igs *.brep *.stl *.obj *.ply);;"
            "STEP / IGES / BREP (*.step *.stp *.iges *.igs *.brep);;"
            "Mesh formats (*.stl *.obj *.ply);;All Files (*)"
        )
        if path:
            self._load_cad(Path(path))

    def _on_cad_option_changed(self):
        """Re-import the loaded body after a units or flow-axis change.

        Both feed analyze_cad, and both change every measurement it returns,
        so the file has to be re-read rather than the numbers patched.
        """
        if self._cad_path is not None and self._rb_cad.isChecked():
            self._load_cad(self._cad_path)

    def _load_cad(self, path: Path):
        """
        Import, align and measure a CAD file, then show it in the 3D preview.

        STEP/IGES are tessellated through Gmsh's OCC kernel (no CadQuery
        needed); the result is aligned so the chosen flow axis points along +X,
        matching the frame the mesher and the moment origin use.
        """
        from PyQt6.QtWidgets import QApplication
        from cfd.external_geometry import analyze_cad

        self._cad_path = path
        self._log(f"Importing {path.name} …")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            info = analyze_cad(
                path,
                flow_axis=self._flow_axis_key(),
                work_dir=user_data_dir("cfd_run"),
                units=self._cad_units_key(),
                wrap=self._chk_wrap.isChecked(),
                wrap_resolution=self._wrap_resolution_key(),
            )
        except Exception as e:
            self._cad_info = None
            self._current_stl = None
            self._cad_lbl.setText(f"{path.name} (failed)")
            self._cad_info_lbl.setText(f"Import failed: {e}")
            self._cad_info_lbl.setStyleSheet(
                f"color:{theme.ERR}; font-size:10px; padding:4px 2px 0 2px;"
            )
            self._log(f"CAD import failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._cad_info = info.as_dict()
        self._cad_lbl.setText(path.name)
        # The aligned tessellation doubles as the geometry STL: it is what the
        # preview shows, what the discrete mesher consumes, and the fallback
        # the solver reads bounds from.
        self._current_stl = Path(info.preview_stl)

        axis_note = " (auto)" if info.flow_axis_auto else ""
        wrap_note = (
            f"<span style='color:#d9a441'>Wrapped — outer mold line only "
            f"(cell {info.wrap_cell*1000:.1f} mm, offset "
            f"{info.wrap_offset*1000:.1f} mm)</span><br>"
            if info.wrapped else ""
        )
        unit_note = (
            f"Units <b>{info.units}</b>{' (auto)' if info.units_auto else ''}"
            + (f" — scaled x{info.unit_scale:g} to metres" if info.unit_scale != 1.0
               else " — no scaling")
            + "<br>"
        )
        warn = "" if info.watertight else (
            f"<br><span style='color:#d9a441'>⚠ not watertight — "
            f"{info.open_edges} free edges; meshing may fail</span>"
        )
        self._cad_info_lbl.setText(
            wrap_note + unit_note +
            f"Flow axis <b>{info.flow_axis.upper()}</b>{axis_note} → +X<br>"
            f"Length {info.length:.4f} m · cross "
            f"{info.cross_width:.4f} × {info.cross_height:.4f} m<br>"
            f"Frontal area {info.frontal_area:.6f} m² · wetted "
            f"{info.wetted_area:.4f} m²<br>"
            f"{info.n_triangles:,} triangles · {info.n_solids} solid(s) · "
            f"{'exact B-Rep' if info.is_brep else 'discrete shell'}{warn}"
        )
        self._cad_info_lbl.setStyleSheet(
            f"color:{theme.TEXT_DIM}; font-size:10px; padding:4px 2px 0 2px;"
        )
        self._sp_ref_area.setSpecialValueText(f"Auto ({info.frontal_area:.6f} m²)")
        self._sp_ref_len.setSpecialValueText(f"Auto ({info.length:.5f} m)")
        self._log(f"CAD ready — {info.summary()}")
        self._preview(self._current_stl)

    def _export_geometry(self):
        from cfd.geometry_exporter import export_assembly_to_stl
        if self._rb_cad.isChecked():
            if self._cad_info:
                self._log(f"Using imported CAD: {Path(self._cad_info['source']).name}")
            else:
                self._log("No CAD file loaded — use Browse CAD File first.")
            return
        assembly = self.assembly_provider() if self.assembly_provider else None
        if assembly is None:
            self._log("No rocket assembly available.")
            return
        work = user_data_dir("cfd_run"); work.mkdir(exist_ok=True)
        stl = work / "rocket.stl"
        try:
            export_assembly_to_stl(assembly, stl)
            self._current_stl = stl
            self._log(f"Geometry exported: {stl}")
            self._preview(stl)
        except Exception as e:
            self._log(f"Export error: {e}")

    def _preview(self, path: Path):
        try:
            self._plotter.clear()
            mesh = pv.read(str(path))
            self._plotter.add_mesh(mesh, color="#b0b8c8", opacity=0.9,
                                   show_edges=True, edge_color=theme.LINE, line_width=0.5)
            self._plotter.add_axes()
            self._plotter.reset_camera()
            self._status_lbl.setText(
                f"Geometry: {path.name}  |  {mesh.n_cells:,} triangles"
            )
        except Exception as e:
            self._log(f"Preview error: {e}")

    def _stl_in_cfd_frame(self):
        """
        The current geometry STL expressed in the CFD frame
        (X = flow direction, upstream tip at x=0).

        Rocket exports are written in the K2 body frame (Z = length axis, nose
        at Z_max) and have to be rotated. The aligned tessellation produced for
        an imported CAD body is *already* in the CFD frame, so rotating it
        again would drop the overlay in the wrong place.

        Returns None when no usable STL is loaded.
        """
        import numpy as np
        if not self._current_stl or not self._current_stl.is_file():
            return None
        raw = pv.read(str(self._current_stl))
        if self._rb_cad.isChecked() and self._cad_info:
            return raw
        pts = raw.points.copy()
        z_max = pts[:, 2].max()          # nose tip Z in K2 space
        out = raw.copy()
        out.points = np.column_stack([
            z_max - pts[:, 2],   # X_cfd = distance from nose
            pts[:, 0],           # Y_cfd = K2 X (radial)
            pts[:, 1],           # Z_cfd = K2 Y (radial)
        ])
        return out

    def _set_params_locked(self, locked: bool):
        """Disable/enable all simulation parameter controls.
        Called when the solver starts (locked=True) and when it
        finishes, errors, or is stopped (locked=False)."""
        enabled = not locked
        # Flow conditions
        self._sp_mach.setEnabled(enabled)
        self._sp_alt.setEnabled(enabled)
        self._sp_aoa.setEnabled(enabled)
        self._cb_turb.setEnabled(enabled)
        # Mesh settings
        self._cb_ref.setEnabled(enabled)
        self._sp_iter.setEnabled(enabled)
        self._sp_cores.setEnabled(enabled)
        # Custom mesh controls
        self._sp_target_count.setEnabled(enabled)
        self._sp_wall_size.setEnabled(enabled)
        self._sl_power.setEnabled(enabled)
        # Geometry source
        self._rb_assembly.setEnabled(enabled)
        self._rb_cad.setEnabled(enabled)
        self._btn_browse.setEnabled(enabled and self._rb_cad.isChecked())
        self._btn_export.setEnabled(enabled)
        # Imported-CAD controls (changing the axis re-runs the import)
        self._cb_cad_units.setEnabled(enabled)
        self._chk_wrap.setEnabled(enabled)
        self._cb_wrap_res.setEnabled(enabled and self._chk_wrap.isChecked())
        self._cb_flow_axis.setEnabled(enabled)
        self._sp_ref_area.setEnabled(enabled)
        self._sp_ref_len.setEnabled(enabled)
        # Analysis mode + sweep controls
        self._rb_single.setEnabled(enabled)
        self._rb_sweep.setEnabled(enabled)
        self._cb_sweep_var.setEnabled(enabled)
        self._sp_sw_start.setEnabled(enabled)
        self._sp_sw_stop.setEnabled(enabled)
        self._sp_sw_step.setEnabled(enabled)

    def _run_cfd(self):
        from cfd.solvers.base import CFDConfig
        import shutil

        if self._rb_assembly.isChecked():
            self._export_geometry()
        elif not self._cad_info:
            self._log("No CAD file loaded — use Browse CAD File first.")
            return
        if not self._current_stl or not self._current_stl.is_file():
            self._log("No geometry ready — export the rocket first.")
            return

        # ── Disk space check ──────────────────────────────────────────────
        try:
            usage = shutil.disk_usage(str(user_data_dir("cfd_run").resolve().drive or "C:\\"))
            free_mb = usage.free / (1024 * 1024)
            if free_mb < 500:
                self._log(
                    f"INSUFFICIENT DISK SPACE: only {free_mb:.0f} MB free. "
                    f"CFD needs at least 500 MB. Free up space and try again."
                )
                return
        except Exception:
            pass  # non-critical — don't block if check fails

        ref_map = {
            0: "coarse", 1: "medium", 2: "fine",
            3: "very_fine", 4: "ultra_fine", 5: "custom",
        }
        ref_idx = self._cb_ref.currentIndex()

        # Resolve custom mesh overrides
        custom_wall = None
        target_count = None
        if ref_idx == 5:  # Custom
            custom_wall = self._sp_wall_size.value() / 1000.0   # mm → m
            target_count = self._sp_target_count.value()

        # Extract EXACT geometry from the live assembly (not STL estimation)
        geo_dict = None
        if self._rb_assembly.isChecked() and self.assembly_provider:
            assembly = self.assembly_provider()
            if assembly is not None:
                try:
                    from cfd.geometry_exporter import extract_cfd_geometry
                    geo_dict = extract_cfd_geometry(assembly)
                    self._log(
                        f"Geometry from design: L={geo_dict['length']:.3f} m  "
                        f"body_r={geo_dict['body_radius']:.4f} m  "
                        f"{geo_dict['fin_count']} fins  "
                        f"Cr={geo_dict['fin_root']:.3f} m  h={geo_dict['fin_height']:.3f} m"
                    )
                except Exception as e:
                    self._log(f"Assembly geometry extraction failed ({e}) - using STL estimate")


        # CG from nose tip — needed so the sweep can report dCm/dα about the CG
        # (the true static-stability metric). Best-effort; None ⇒ no verdict.
        # Only meaningful for the rocket design — the loaded assembly's CG has
        # nothing to do with an imported CAD body, so don't transfer moments to it.
        cg_from_nose = None
        if self._rb_assembly.isChecked():
            try:
                _asm = self.assembly_provider() if self.assembly_provider else None
                if _asm is None and self.engine and hasattr(self.engine, "assembly"):
                    _asm = self.engine.assembly
                if _asm is not None and hasattr(_asm, "compute_cg"):
                    cg_from_nose = float(_asm.compute_cg())
                    self._log(f"CG = {cg_from_nose:.3f} m from nose (for stability moment transfer)")
            except Exception as e:
                self._log(f"CG lookup failed ({e}) — stability verdict will be unavailable")

        # ── External CAD mode ──
        external_cad = None
        cad_info = None
        ref_area_ov = None
        ref_len_ov = None
        if self._rb_cad.isChecked() and self._cad_info:
            external_cad = Path(self._cad_info["source"])
            cad_info = self._cad_info
            ref_area_ov = self._sp_ref_area.value() or None
            ref_len_ov = self._sp_ref_len.value() or None
            self._log(
                f"CAD mode: {external_cad.name} | flow axis "
                f"{self._cad_info['flow_axis'].upper()} → +X | "
                f"ref A={ref_area_ov if ref_area_ov else self._cad_info['frontal_area']:.6f} m² "
                f"({'user' if ref_area_ov else 'auto frontal'}) | "
                f"ref L={ref_len_ov if ref_len_ov else self._cad_info['length']:.5f} m "
                f"({'user' if ref_len_ov else 'auto bbox'})"
            )
            if not self._cad_info.get("watertight", True):
                self._log(
                    f"WARNING: geometry is not watertight "
                    f"({self._cad_info.get('open_edges', 0)} free edges) — "
                    f"meshing may fail or the fluid volume may leak."
                )

        cfg = CFDConfig(
            mach=self._sp_mach.value(),
            altitude_m=self._sp_alt.value(),
            angle_of_attack_deg=self._sp_aoa.value(),
            mesh_refinement=ref_map.get(ref_idx, "medium"),
            bl_prisms=self._chk_bl_prisms.isChecked(),
            max_iterations=self._sp_iter.value(),
            n_cores=self._sp_cores.value(),
            turbulence_model=self._get_turb_key(),
            work_dir=user_data_dir("cfd_run"),
            geometry_stl=self._current_stl,
            geometry_dict=geo_dict,
            custom_wall_size=custom_wall,
            target_element_count=target_count,
            cg_from_nose_m=cg_from_nose,
            external_cad=external_cad,
            flow_axis=self._flow_axis_key(),
            cad_info=cad_info,
            cad_units=self._cad_units_key(),
            cad_wrap=self._chk_wrap.isChecked(),
            cad_wrap_resolution=self._wrap_resolution_key(),
            ref_area_override=ref_area_ov,
            ref_length_override=ref_len_ov,
        )

        # Hybrid Euler + analytic friction. Applied to BOTH run modes — the
        # solver applies it in parse_results, so there is one code path.
        if self._chk_euler_fric.isChecked():
            cfg.turbulence_model = "Euler"
            cfg.euler_analytic_friction = True
            if cfg.external_cad and cfg.cad_info:
                self._log(
                    "Euler+friction mode: friction built up from the CAD wetted "
                    "area with a body-of-revolution form factor — an engineering "
                    "estimate, coarser than the rocket component breakdown."
                )
            elif cfg.geometry_dict is None:
                self._log(
                    "Euler+friction mode: no exact geometry available (STL "
                    "source) — the friction build-up will be skipped and Cd will "
                    "be inviscid-only, i.e. missing its friction component."
                )
        elif cfg.turbulence_model != "Euler" and not cfg.bl_prisms:
            self._log(
                "Running RANS on a tet-only mesh (no prism boundary layer). The "
                "wall is unresolved, so skin friction and wall heat transfer "
                "from this run are not physical and total Cd is a lower bound. "
                "Enable 'Euler + flat-plate friction' or 'Prism boundary layer'."
            )

        # ── Sweep mode branches off here (uses cfg as the base condition) ──
        if self._rb_sweep.isChecked():
            self._start_sweep(cfg)
            return

        # Nothing from the last solve may outlive this click: if the new run
        # fails, stale coefficients must not sit there looking like its output.
        self._clear_results()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_params_locked(True)
        max_it = self._sp_iter.value()
        self._progress.setRange(0, max_it)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._log_box.clear()
        self._res_iters.clear(); self._res_vals.clear()
        self._log(f"CFD start | Mach {cfg.mach} | Alt {cfg.altitude_m} m | AoA {cfg.angle_of_attack_deg}°")

        # All heavy work (mesh gen + config gen + solver) runs in background thread
        self._solver_thread = SolverThread(cfg)
        self._solver_thread.progress.connect(self._on_progress)
        self._solver_thread.finished.connect(self._on_finished)
        self._solver_thread.errored.connect(self._on_error)
        self._solver_thread.log_msg.connect(self._log)
        self._solver_thread.start()
        self._log("Background solver thread started — UI remains responsive.")

    # ── Sweep mode ────────────────────────────────────────────────────────────
    def _sweep_var_key(self) -> str:
        return "mach" if self._cb_sweep_var.currentIndex() == 1 else "aoa"

    def _on_sweep_var_changed(self, idx):
        """Retune start/stop/step ranges & defaults for the chosen variable."""
        is_mach = (idx == 1)
        if is_mach:
            for sp in (self._sp_sw_start, self._sp_sw_stop):
                sp.setRange(0.05, 10.0); sp.setSuffix("")
            self._sp_sw_step.setRange(0.01, 2.0)
            self._sp_sw_start.setValue(0.40)
            self._sp_sw_stop.setValue(1.40)
            self._sp_sw_step.setValue(0.10)
        else:
            for sp in (self._sp_sw_start, self._sp_sw_stop):
                sp.setRange(-30, 30); sp.setSuffix(" °")
            self._sp_sw_step.setRange(0.05, 10.0)
            self._sp_sw_start.setValue(-4.0)
            self._sp_sw_stop.setValue(12.0)
            self._sp_sw_step.setValue(2.0)
        self._update_sweep_info()

    def _update_sweep_info(self):
        from cfd.sweep import build_value_list
        vals = build_value_list(
            self._sp_sw_start.value(), self._sp_sw_stop.value(), self._sp_sw_step.value()
        )
        unit = "Mach" if self._sweep_var_key() == "mach" else "°"
        self._lbl_sweep_info.setText(
            f"{len(vals)} points × ~minutes each. "
            f"Other conditions fixed at Mach {self._sp_mach.value():g}, "
            f"AoA {self._sp_aoa.value():g}{'' if self._sweep_var_key()=='aoa' else '°'}, "
            f"Alt {self._sp_alt.value():g} m."
        )
        return vals

    def _start_sweep(self, base_cfg):
        var = self._sweep_var_key()
        vals = self._update_sweep_info()
        if len(vals) < 2:
            self._log("Sweep needs at least 2 points — widen the range or shrink the step.")
            return
        # Sweep-specific solver caps: enough iterations per point that the
        # per-point Cm (and thus dCm/dα stability) is trustworthy, while
        # staying tractable across many points.
        #
        # The cap used to be a hard-coded 800 that silently overrode whatever
        # the user had set, so raising Max Iterations did nothing to a sweep
        # and there was no way to give a stubborn point more room. It is now a
        # floor: sweeps still get at least 800 even if the spinbox is lower,
        # but a user asking for more gets it.
        #
        # The tolerance override is gone. It was 1e-8 against 1e-6 for single
        # runs, and CONV_RESIDUAL_MINVAL is an ABSOLUTE floor on dimensional
        # residuals — a measured case starts near rms[Rho]=-3, so -8 demands 5
        # decades where -6 demands 3, with no principled reason for a sweep
        # point to be held to a different standard than the same case run on
        # its own. Both paths now use the config default, and the DRAG Cauchy
        # criterion added to the SU2 template does the real stopping work.
        self._sweep_max_iter = max(self._sp_iter.value(), 800)
        base_cfg.max_iterations = self._sweep_max_iter
        # The hybrid mode is already applied to base_cfg by _run_cfd, which
        # does it for both run modes; recorded here only for the polar labels.
        self._sweep_euler_fric = self._chk_euler_fric.isChecked()
        if self._sweep_euler_fric:
            self._log(
                "Polar fidelity: Euler (inviscid) + flat-plate friction build-up. "
                "Uncheck it in the Fidelity group to use the selected "
                "turbulence model instead."
            )
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_params_locked(True)
        self._log_box.clear()
        self._res_iters.clear(); self._res_vals.clear()
        self._sweep_data = None
        self._progress.setRange(0, 0)   # busy/indeterminate until first iter
        self._progress.setVisible(True)
        self._center_tabs.setCurrentIndex(1)  # show Polars tab
        if hasattr(self._polar_plot, "clear"):
            self._polar_plot.clear()
        self._log(
            f"Sweep start | var={var} | {len(vals)} pts "
            f"[{vals[0]:g} … {vals[-1]:g}] | base Mach {base_cfg.mach:g} "
            f"AoA {base_cfg.angle_of_attack_deg:g}° Alt {base_cfg.altitude_m:g} m "
            f"| {base_cfg.max_iterations} iters/pt, tol {base_cfg.convergence_tolerance:g}"
        )

        self._sweep_thread = SweepThread(base_cfg, var, vals)
        self._sweep_thread.progress.connect(self._on_sweep_progress)
        self._sweep_thread.point.connect(self._on_sweep_point)
        self._sweep_thread.finished.connect(self._on_sweep_finished)
        self._sweep_thread.errored.connect(self._on_sweep_error)
        self._sweep_thread.log_msg.connect(self._log)
        self._sweep_thread.start()

    def _on_sweep_progress(self, pt_idx, n_pts, it, rms):
        if it < 0:
            return
        max_it = getattr(self, "_sweep_max_iter", 800)
        # Overall progress = completed points + fraction of current point.
        frac = min(it / max(max_it, 1), 1.0)
        self._progress.setRange(0, n_pts * 100)
        self._progress.setValue(int((pt_idx + frac) * 100))
        self._progress.setFormat(f" Point {pt_idx+1}/{n_pts} — iter {it}")

    def _on_sweep_point(self, pt):
        """A single sweep condition finished — accumulate & live-redraw."""
        from cfd.sweep import SweepData
        if self._sweep_data is None:
            self._sweep_data = SweepData(var=pt.var)
            self._populate_polar_combo(pt.var)
        self._sweep_data.points.append(pt)
        self._refresh_polar()

    def _populate_polar_combo(self, var: str):
        self._cb_polar.blockSignals(True)
        self._cb_polar.clear()
        if var == "aoa":
            self._cb_polar.addItems([
                "Cl vs AoA", "Cd vs AoA", "Cm vs AoA",
                "Drag polar (Cd vs Cl)", "CP vs AoA",
            ])
        else:
            self._cb_polar.addItems([
                "Cd vs Mach (drag rise)", "Cl vs Mach", "Cm vs Mach",
                "CP vs Mach", "Wave drag vs Mach",
            ])
        self._cb_polar.blockSignals(False)

    def _refresh_polar(self):
        d = self._sweep_data
        if d is None or len(d.points) == 0 or not hasattr(self._polar_plot, "update_plot"):
            return
        choice = self._cb_polar.currentText()
        x = d.x()
        if d.var == "aoa":
            xlabel = "Angle of Attack (°)"
            # Prefer the CG moment (true stability curve); fall back to the
            # nose-tip moment when CG was unavailable so the plot isn't flat-zero.
            cm_cg_series = d.cm_cg()
            if any(abs(v) > 1e-9 for v in cm_cg_series):
                cm_y, cm_label = cm_cg_series, "Cm about CG"
            else:
                cm_y, cm_label = d.cm(), "Cm (nose tip — CG unavailable)"
            series = {
                "Cl vs AoA":   (x, d.cl(), "Cl",  theme.ACCENT),
                "Cd vs AoA":   (x, d.cd(), "Cd",  theme.ACCENT),
                "Cm vs AoA":   (x, cm_y, cm_label,  "#a371f7"),
                "CP vs AoA":   (x, d.cp_smooth(), "CP from nozzle (m)", theme.OK),
            }
            if choice == "Drag polar (Cd vs Cl)":
                self._polar_plot.update_plot(d.cl(), d.cd(), "Drag Polar", "Cl", "Cd", theme.ACCENT)
                self._update_polar_metrics()
                return
        else:
            xlabel = "Mach"
            series = {
                "Cd vs Mach (drag rise)": (x, d.cd(), "Cd",  theme.ACCENT),
                "Cl vs Mach":   (x, d.cl(), "Cl",  theme.ACCENT),
                "Cm vs Mach":   (x, d.cm(), "Cm",  "#a371f7"),
                "CP vs Mach":   (x, d.cp(), "CP from nozzle (m)", theme.OK),
                "Wave drag vs Mach": (x, d.cd_wave(), "Cd_wave", theme.ERR),
            }
        sx, sy, ylabel, color = series.get(choice, (x, d.cd(), "Cd", theme.ACCENT))
        self._polar_plot.update_plot(sx, sy, choice, xlabel, ylabel, color)
        self._update_polar_metrics()

    def _update_polar_metrics(self):
        from cfd.sweep import compute_sweep_metrics
        d = self._sweep_data
        if d is None or len(d.points) < 2:
            return
        m = compute_sweep_metrics(d)
        if d.var == "aoa":
            verdict = m.get("stability_verdict")
            cm_cg_slope = m.get("cm_cg_alpha_per_rad")
            if verdict is not None and cm_cg_slope is not None:
                badge = {"Stable": "✓ Stable", "Marginal": "≈ Marginal",
                         "Unstable": "✗ Unstable"}.get(verdict, verdict)
                stab_txt = f"dCm/dα|CG = {cm_cg_slope:+.3f} /rad  →  {badge}"
            else:
                stab_txt = "dCm/dα|CG = — (CG unavailable)"
            txt = (
                f"dCl/dα = {m.get('cl_alpha_per_rad', 0):.3f} /rad "
                f"({m.get('cl_alpha_per_deg', 0):.4f} /°)   |   "
                f"{stab_txt}   |   "
                f"Cd₀ = {m.get('cd0', 0):.4f}"
            )
            if "k_induced" in m:
                txt += f"   |   Cd≈Cd₀+k·Cl², k = {m['k_induced']:.4f}"
            if "cp_travel_m" in m:
                txt += f"   |   CP travel = {m['cp_travel_m']*1000:.1f} mm"
        else:
            mdd = m.get("drag_divergence_mach")
            txt = (
                f"Drag-divergence Mach = {mdd:.3f}" if mdd is not None
                else "Drag-divergence Mach = none in range"
            )
            txt += (
                f"   |   Cd bucket {m.get('cd_min', 0):.4f} @ M {m.get('mach_at_cd_min', 0):.3f}"
                f"   |   Peak Cd {m.get('cd_peak', 0):.4f} @ M {m.get('mach_at_cd_peak', 0):.3f}"
                f"  (×{m.get('transonic_drag_rise_ratio', 0):.2f})"
            )
            if "wave_drag_peak" in m:
                txt += (f"   |   Wave drag peak = {m['wave_drag_peak']:.4f} "
                        f"@ M {m.get('mach_at_wave_peak', 0):.3f}")
            if "cp_shift_m" in m:
                txt += f"   |   CP shift = {m['cp_shift_m']*1000:.1f} mm"

        if getattr(self, "_sweep_euler_fric", False):
            cdf = [p.result.cd_friction for p in d.points if p.result.cd_friction > 0]
            cdf_txt = f", Cd_f ≈ {sum(cdf)/len(cdf):.4f}" if cdf else ""
            txt += (f"\nMode: Euler + flat-plate friction build-up"
                    f"{cdf_txt} (lift / CP / wave drag inviscid)")

        # ── Fidelity caveats — the numbers above are only as good as the solve ──
        n_unconv = sum(1 for p in d.points if not p.result.converged)
        if n_unconv:
            txt += (f"\n⚠ {n_unconv}/{len(d.points)} points unconverged "
                    f"(forces not stationary) — treat affected values as approximate.")
        yp = [p.result.yplus_mean for p in d.points if p.result.yplus_mean > 0]
        if yp and (sum(yp) / len(yp)) > 30.0:
            txt += (f"\n⚠ Wall under-resolved (mean y+ ≈ {sum(yp)/len(yp):.0f}, "
                    f"no wall functions): Cd is inflated and CP biased forward "
                    f"— spurious viscous body lift can read as \"Unstable\". "
                    f"Trends vs AoA/Mach are usable; absolute Cd₀ and the "
                    f"stability verdict are not. Cross-check CP against "
                    f"Barrowman, or re-run with the \"Euler + flat-plate "
                    f"friction\" sweep option for a cleaner pressure CP.")
        self._lbl_polar_metrics.setText(txt)

    def _on_sweep_finished(self, data):
        self._sweep_data = data
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_params_locked(False)
        self._progress.setVisible(False)
        self._progress.setFormat(" %p% (%v / %m iterations)")
        n_ok = sum(1 for p in data.points if p.result.converged)
        self._log(f"Sweep complete — {len(data.points)} points ({n_ok} converged).")
        if data.points:
            self._btn_polar_export.setEnabled(True)
            self._refresh_polar()

    def _on_sweep_error(self, msg):
        for line in msg.splitlines():
            if line.strip():
                self._log(f"ERROR: {line}")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._set_params_locked(False)

    def _export_polar_csv(self):
        d = self._sweep_data
        if d is None or not d.points:
            self._log("No sweep data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Polar CSV", "cfd_polar.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            import csv
            pts = sorted(d.points, key=lambda q: q.value)
            # Per-point, solved from the surface integral — no longer
            # reconstructed across the sweep. Aligned to sorted order.
            wave = d.cd_wave()
            base = d.cd_base()
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([d.var, "Cd", "Cl", "Cm", "Cd_pressure", "Cd_friction",
                            "Cd_base", "Cd_wave", "CP_m", "Reynolds", "converged"])
                for i, p in enumerate(pts):
                    r = p.result
                    w.writerow([p.value, r.cd, r.cl, r.cm, r.cd_pressure,
                                r.cd_friction,
                                base[i] if i < len(base) else 0.0,
                                wave[i] if i < len(wave) else 0.0,
                                r.cp_location_m, r.reynolds, r.converged])
            self._log(f"Polar exported: {path}")
        except Exception as e:
            self._log(f"CSV export failed: {e}")

    def _stop_cfd(self):
        # Sweep thread takes priority if it's the one running.
        if self._sweep_thread and self._sweep_thread.isRunning():
            self._sweep_thread.stop()
            self._sweep_thread.wait(2000)
            self._log("Sweep stopped by user.")
            self._btn_run.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self._progress.setVisible(False)
            self._set_params_locked(False)
            # Keep whatever points completed so far for inspection.
            if self._sweep_data and self._sweep_data.points:
                self._btn_polar_export.setEnabled(True)
                self._refresh_polar()
            return
        if self._solver_thread and self._solver_thread.isRunning():
            # Kill mesh subprocess if running
            if (hasattr(self._solver_thread, '_mesh_proc')
                    and self._solver_thread._mesh_proc
                    and self._solver_thread._mesh_proc.poll() is None):
                try:
                    self._solver_thread._mesh_proc.kill()
                except Exception:
                    pass
            # Kill SU2 solver subprocess if running
            if (self._solver_thread.solver
                    and hasattr(self._solver_thread.solver, '_proc')
                    and self._solver_thread.solver._proc
                    and self._solver_thread.solver._proc.poll() is None):
                try:
                    self._solver_thread.solver._proc.kill()
                except Exception:
                    pass
            self._solver_thread.terminate()
            self._solver_thread.wait(2000)  # wait up to 2s for cleanup
            self._log("Solver stopped by user.")
            self._clear_results("Partial run discarded — no results to show.")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._set_params_locked(False)

    def _on_progress(self, it: int, rms: float):
        if it < 0:
            return   # diagnostic-only emit, no residual data
            
        # Update progress bar
        max_it = self._sp_iter.value()
        self._progress.setRange(0, max_it)
        self._progress.setValue(it)
        
        self._res_iters.append(it)
        self._res_vals.append(rms)
        if hasattr(self._res_plot, "update_plot") and len(self._res_iters) > 1:
            self._res_plot.update_plot(
                self._res_iters, self._res_vals,
                "Convergence", "Iteration", "RMS Density", theme.ACCENT
            )

    def _on_finished(self, result):
        self._result = result
        # Core coefficients
        self._lbl_cd.setText(f"{result.cd:.5f}")
        self._lbl_cl.setText(f"{result.cl:.5f}")
        self._lbl_cm.setText(f"{result.cm:.5f}")

        # Drag decomposition
        self._lbl_cdp.setText(f"{result.cd_pressure:.5f}  ({result.cd_pressure/max(result.cd,1e-9)*100:.0f}%)")
        self._lbl_cdf.setText(f"{result.cd_friction:.5f}  ({result.cd_friction/max(result.cd,1e-9)*100:.0f}%)")
        # Base/wave are percentages OF the pressure drag they subdivide, not of
        # the total — showing them against Cd would imply they were separate terms.
        # Magnitude in the denominator: cd_pressure is no longer forced positive,
        # and a bad solution that integrates negative would otherwise print a
        # percentage in the billions rather than reading as obviously wrong.
        _cdp = max(abs(result.cd_pressure), 1e-9)
        _share = "% of pressure" if result.cd_pressure > 0 else "% of |pressure|"
        self._lbl_cdb.setText(
            f"{result.cd_base:.5f}  ({result.cd_base/_cdp*100:.0f}{_share})"
        )
        if result.cd_wave > 0:
            self._lbl_cdw.setText(
                f"{result.cd_wave:.5f}  ({result.cd_wave/_cdp*100:.0f}{_share})"
            )
        else:
            self._lbl_cdw.setText("—  (subsonic)" if result.mach < 0.8 else "—")

        # Forces & CP
        # Fallback: Compute dimensional forces if parser missed them (e.g., older SU2 versions)
        f_axial  = result.force_axial
        f_normal = result.force_normal
        ref_area = getattr(result, "reference_area_m2", 0.0) or getattr(result, "ref_area", 0.0)
        if f_axial == 0.0 and result.cd != 0 and ref_area > 0:
            f_axial = result.cd * result.dynamic_pressure * ref_area
        if f_normal == 0.0 and result.cl != 0 and ref_area > 0:
            f_normal = result.cl * result.dynamic_pressure * ref_area
            
        self._lbl_fa.setText(f"{f_axial:.2f} N")
        self._lbl_fn.setText(f"{f_normal:.2f} N")
        if result.cp_location_m > 0.01:
            # Get the TRUE rocket length from multiple sources
            rocket_len = result.ref_length  # fallback (may be inflated by fin span)
            len_source = "ref_length"
            # Priority 0: imported CAD — its own measured flow-wise extent. The
            # assembly length and the STL Z-span below both describe the rocket
            # design, which has nothing to do with the imported body.
            if self._rb_cad.isChecked() and self._cad_info:
                rocket_len = float(self._cad_info.get("length", rocket_len))
                len_source = "CAD bbox"
            # Priority 1: Assembly total length (most reliable)
            elif self.engine and hasattr(self.engine, 'assembly'):
                try:
                    rocket_len = self.engine.assembly.total_length()
                    len_source = "assembly"
                except Exception:
                    pass
            # Priority 2: STL Z-span (Z is rocket axis in K2)
            if (len_source not in ("assembly", "CAD bbox")
                    and self._current_stl and self._current_stl.is_file()):
                try:
                    stl_m = pv.read(str(self._current_stl))
                    b = stl_m.bounds
                    rocket_len = abs(b[5] - b[4])  # Z-span = axial length
                    len_source = "STL Z-span"
                except Exception:
                    pass
            # Clamp CP to rocket length
            cp_val = min(result.cp_location_m, rocket_len)
            if result.cp_location_m > rocket_len * 1.01:
                self._lbl_cp.setText(f"{cp_val:.4f} m from nozzle")
                self._log(f"CP={result.cp_location_m:.4f} m exceeded rocket length "
                          f"{rocket_len:.3f} m ({len_source}) — clamped to {cp_val:.4f} m")
            else:
                self._lbl_cp.setText(f"{cp_val:.4f} m from nozzle")
        else:
            self._lbl_cp.setText("— (undefined at AoA=0°)")

        # Solver info
        self._lbl_solver.setText(result.solver_name or "SU2")
        turb_display = {"SA": "Spalart-Allmaras", "SST": "k-\u03c9 SST",
                        "Euler": "Euler", "Laminar": "Laminar"}.get(result.turbulence_model, result.turbulence_model)
        self._lbl_turb_r.setText(turb_display)
        self._lbl_re_r.setText(f"{result.reynolds:.2e}")
        self._lbl_q_r.setText(f"{result.dynamic_pressure/1000:.2f} kPa")

        # Convergence
        self._lbl_conv.setText("Yes" if result.converged else "No (check log)")
        self._lbl_conv.setStyleSheet(
            theme.value_qss(theme.OK if result.converged else theme.ERR)
        )
        _note = getattr(result, "convergence_note", "") or ""
        _drop = getattr(result, "residual_drop_decades", 0.0)
        if _drop:
            _note = f"rms[Rho] fell {_drop:.1f} decades. " + _note
        self._lbl_conv_why.setText(_note)

        # Trustworthiness of the wall-dependent quantities. The solver decides
        # this (it is the only thing that knows the mesh spacing and the flow),
        # and the panel simply states it next to the coefficients rather than
        # leaving it in a log.
        _wall_ok = getattr(result, "wall_resolved", True)
        _wall_msg = getattr(result, "wall_warning", "") or ""
        self._lbl_trust.setVisible(not _wall_ok and bool(_wall_msg))
        if not _wall_ok and _wall_msg:
            self._lbl_trust.setText("⚠ " + _wall_msg)
            self._lbl_trust.setStyleSheet(
                f"color:{theme.WARN}; font-size:10px; font-weight:600;")
            self._log("WARNING: " + _wall_msg)
        self._btn_inject.setEnabled(result.converged)
        self._btn_export_vtk.setEnabled(bool(result.volume_vtk or result.surface_vtk))
        self._btn_export_struct.setEnabled(bool(result.surface_vtk))
        self._btn_export_pdf.setEnabled(True)
        self._btn_export_csv.setEnabled(True)
        self._vis_combo.setEnabled(True)
        self._vis_combo.setCurrentIndex(2)  # auto-show Pressure volume slice
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._set_params_locked(False)
        self._progress.setRange(0, 100)
        self._progress.setValue(95)
        self._log(f"Done — Cd={result.cd:.4f}  Cl={result.cl:.4f}  Cm={result.cm:.4f}  Re={result.reynolds:.2e}")
        self._log("Loading flow field & computing derived fields in background…")

        # Store freestream state for dimensional viz
        self._v_inf   = getattr(result, "v_inf", self._sp_mach.value() * 328.0)
        self._mach    = getattr(result, "mach",  self._sp_mach.value())
        self._result  = result   # store for use in _on_postprocess_done

        # Launch post-processing in a background thread — avoids blocking the UI
        # Derive static pressure from dynamic_pressure + mach (ISA: p = q * 2 / (gamma * M^2))
        _mach_sq = max(result.mach, 0.01) ** 2
        P_inf = result.dynamic_pressure * 2.0 / (1.4 * _mach_sq) if _mach_sq > 0 else 101325.0
        q_inf = result.dynamic_pressure
        # Load the VTKs this result actually wrote, not a fixed directory.
        # Sweep points write into work_dir/sweep/<tag>/, so the base folder can
        # still hold flow.vtu from an older single run — which would then be
        # loaded and displayed as if it belonged to this solve.
        _vtk_dir = user_data_dir("cfd_run")
        for _p in (result.volume_vtk, result.surface_vtk):
            if _p:
                _vtk_dir = Path(_p).parent
                break
        self._pp_thread = PostProcessThread(_vtk_dir, P_inf, q_inf)
        self._pp_thread.log_msg.connect(self._log)
        self._pp_thread.done.connect(self._on_postprocess_done)
        self._pp_thread.start()

    def _on_postprocess_done(self, vol_mesh, surf_mesh):
        """Called from PostProcessThread when VTK loading & derived fields are complete."""
        self._volume_mesh  = vol_mesh
        self._surface_mesh = surf_mesh
        self._progress.setVisible(False)

        if vol_mesh is not None:
            derived = [a for a in vol_mesh.array_names
                       if a in ["Vorticity_Magnitude", "Q_Criterion", "Lambda2",
                                "Mach", "Pressure_Coefficient", "Velocity"]]
            self._log(f"Post-processing complete. Fields: {derived}")
        else:
            self._log("Warning: Volume mesh could not be loaded.")

        # Mesh statistics — measured, not asserted.
        #
        # This panel used to hardcode setText("Good") in green and show "—" for
        # the aspect ratio, so it reported a quality it had never computed, on
        # every mesh. compute_mesh_statistics was never called from here at all.
        try:
            from cfd.post_processing import compute_mesh_statistics
            if vol_mesh is not None:
                self._lbl_cells.setText(f"{vol_mesh.n_cells:,}")
                self._lbl_nodes.setText(f"{vol_mesh.n_points:,}")
            stats = compute_mesh_statistics(vol_mesh, surf_mesh)
            rating = stats.get("quality_rating", "Unknown")
            self._lbl_mq.setText(rating)
            self._lbl_mq.setStyleSheet(
                theme.value_qss({"Good": theme.OK, "Fair": theme.WARN}
                                .get(rating, theme.ERR)
                                if rating != "Unknown" else theme.TEXT_DIM))
            _ar, _mar = (stats.get("mean_aspect_ratio", 0.0),
                         stats.get("max_aspect_ratio", 0.0))
            self._lbl_ar.setText(f"{_ar:.2f} mean, {_mar:.1f} max"
                                 if _ar > 0 else "—")
            _g99, _gmx = stats.get("growth_p99", 0.0), stats.get("growth_max", 0.0)
            if _g99 > 0:
                self._lbl_growth.setText(f"{_g99:.2f} p99, {_gmx:.1f} max")
                self._lbl_growth.setStyleSheet(theme.value_qss(
                    theme.OK if _g99 <= 1.5 else
                    theme.WARN if _g99 <= 2.5 else theme.ERR))
            else:
                self._lbl_growth.setText("—")
            _ypm = stats.get("yplus_mean", 0.0)
            if _ypm > 0:
                self._lbl_yp_r.setText(
                    f"{_ypm:.0f} med  ({stats.get('yplus_min', 0):.1f} – "
                    f"{stats.get('yplus_max', 0):.0f})")
                self._lbl_yp_r.setStyleSheet(theme.value_qss(
                    theme.OK if _ypm < 5 else
                    theme.WARN if _ypm < 30 else theme.ERR))
            else:
                self._lbl_yp_r.setText("— (inviscid)")
        except Exception as e:
            self._log(f"Mesh stats error: {e}")

        result = self._result   # use stored result — not in scope from signal

        # Cp distribution plot
        try:
            from cfd.post_processing import extract_cp_distribution
            if self._surface_mesh is not None and result is not None:
                # CFDResult carries no static pressure — derive P_inf from
                # q and Mach (p = 2q / (gamma*M^2)) so the fallback Cp uses
                # the actual altitude, not sea-level 101325 Pa.
                _q = max(getattr(result, "dynamic_pressure", 1.0), 1.0)
                _msq = max(getattr(result, "mach", 0.01), 0.01) ** 2
                _p_inf = _q * 2.0 / (1.4 * _msq)
                x_n, cp_v = extract_cp_distribution(
                    self._surface_mesh,
                    freestream_pressure=_p_inf,
                    dynamic_pressure=_q,
                )
                # At incidence the circumferential mean is the wrong curve to
                # show on its own: the windward and leeward sides carry equal
                # and opposite loading, so averaging them reports a body with
                # no normal force on it. Plot both.
                _extra = []
                if abs(getattr(result, "angle_of_attack_deg", 0.0)) > 0.5:
                    for _side, _col in (("windward", theme.ERR),
                                        ("leeward", theme.OK)):
                        _x, _c = extract_cp_distribution(
                            self._surface_mesh, side=_side,
                            freestream_pressure=_p_inf, dynamic_pressure=_q)
                        if len(_x) == len(x_n):
                            _extra.append((_side.capitalize(), _c, _col))
                if len(x_n) > 0 and hasattr(self._cp_plot, "update_cp"):
                    try:
                        self._cp_plot.update_cp(x_n, cp_v, extra=_extra)
                    except TypeError:
                        self._cp_plot.update_cp(x_n, cp_v)
        except Exception as e:
            self._log(f"Cp plot error: {e}")

        # Convergence history plot
        try:
            if result is not None and result.residual_history and hasattr(self._res_plot, "update_plot"):
                iters = [r[0] for r in result.residual_history]
                vals  = [r[1] for r in result.residual_history]
                self._res_plot.update_plot(iters, vals, "Convergence", "Iteration", "log(RMS ρ)", theme.ACCENT)
        except Exception as e:
            self._log(f"Convergence plot error: {e}")

        self._refresh_vis()

    def _clear_results(self, reason: str = ""):
        """Drop every artefact of the previous solve.

        Without this a failed or stopped run left the last successful result on
        screen — coefficients, convergence badge, loaded flow field, and live
        Inject/Export buttons — now implicitly attributed to the NEW settings.
        The user sees plausible numbers for a run that never produced any.
        """
        self._result = None
        self._volume_mesh = None
        self._surface_mesh = None
        for lbl in ("_lbl_cd", "_lbl_cdp", "_lbl_cdf", "_lbl_cdb", "_lbl_cdw",
                    "_lbl_cl", "_lbl_cm", "_lbl_conv", "_lbl_fa", "_lbl_fn",
                    "_lbl_cp", "_lbl_solver", "_lbl_turb_r", "_lbl_re_r",
                    "_lbl_q_r"):
            w = getattr(self, lbl, None)
            if w is not None:
                w.setText("—")
        if getattr(self, "_lbl_conv", None) is not None:
            self._lbl_conv.setProperty("value", True)
        for btn in ("_btn_inject", "_btn_export_vtk", "_btn_export_struct",
                    "_btn_export_pdf", "_btn_export_csv"):
            w = getattr(self, btn, None)
            if w is not None:
                w.setEnabled(False)
        # Blank the 3D view too — a stale flow field is the most convincing
        # wrong answer of the lot.
        try:
            self._plotter.clear()
            self._last_vis_idx = -1
        except Exception:
            pass
        if reason:
            self._log(reason)

    def _on_error(self, msg: str):
        # Show each line of the error separately so long messages are readable
        for line in msg.splitlines():
            if line.strip():
                self._log(f"ERROR: {line}")
        self._clear_results(
            "Previous results cleared — this run produced none."
        )
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._progress.setVisible(False)
        self._set_params_locked(False)

    def _rocket_region(self, vm):
        """
        Return the rocket bounding box for camera focus.
        We no longer clip the volume mesh before slicing — clip_box() cuts cells
        at its boundary, which creates rectangular rendering artifacts when those
        cut-boundary faces intersect the 2D slice plane.
        The VTK slice filter is fast enough on the full domain.
        """
        # Use surface mesh for the tight rocket bounding box (camera / seed placement)
        ref = None
        if self._surface_mesh is not None:
            ref = self._surface_mesh
        elif self._current_stl and self._current_stl.is_file():
            ref = pv.read(str(self._current_stl))

        if ref is not None:
            rb = ref.bounds   # (xmin,xmax,ymin,ymax,zmin,zmax)
        else:
            rb = vm.bounds

        # Return the FULL volume mesh — no clip_box
        return vm, rb

    def _filter_interior_cells(self, slc):
        """
        Remove mesh points that are inside the rocket solid body.
        SU2 volume output includes cells inside the Boolean-cut rocket geometry
        with stagnant (near-zero momentum) bogus values. These render as a
        dark rectangular artifact on 2D slices. We detect them by:
          speed < 0.5% of V_inf  AND  r < body_radius  AND  x WITHIN the body

        Both thresholds used to be wrong in a way that deleted real flow.
        The speed cut was a hard-coded 20 m/s, which is not "stagnant" — it is
        ordinary separated flow at any realistic freestream, and below V_inf=20
        it condemned the entire near-field. The axial window ran to
        ``rb[1] + 0.1``, i.e. 100 mm PAST the tail, so the base recirculation
        got cut out along with the dead cells: 358 slice points on a measured
        M=0.8 case, covering exactly the region base drag comes from. Trapped
        cells inside the boolean-cut solid are genuinely at machine-zero
        momentum, so a fraction-of-freestream threshold separates them from
        real slow flow, and the window now stops at the tail.
        """
        sm = self._surface_mesh
        if sm is None or slc is None or slc.n_points == 0:
            return slc
        rb = sm.bounds
        rR = max(abs(rb[3] - rb[2]), abs(rb[5] - rb[4])) / 2
        pts = slc.points
        r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2)

        # Compute speed from Momentum/Density (SU2 conservative variables)
        if "Momentum" in slc.array_names and "Density" in slc.array_names:
            mom = slc["Momentum"]
            rho = slc["Density"].flatten()
            speed = np.linalg.norm(mom, axis=1) / np.maximum(rho, 1e-12)
        elif "Velocity" in slc.array_names:
            speed = np.linalg.norm(slc["Velocity"], axis=1)
        else:
            return slc  # can't filter without velocity info

        v_ref = float(getattr(self, "_v_inf", 0.0) or 0.0)
        if v_ref <= 0.0:
            v_ref = float(np.percentile(speed, 99)) if speed.size else 0.0
        stagnant = max(v_ref * 0.005, 1e-6)

        interior = (
            (speed < stagnant) &
            (r < rR * 1.05) &
            (pts[:, 0] > rb[0] - 0.01) &
            (pts[:, 0] < rb[1])          # stop at the tail — keep the base wake
        )
        if interior.sum() == 0:
            return slc
        return slc.extract_points(~interior)

    @staticmethod
    def _smooth_surface_scalars(mesh, name, n_iter=3) -> str:
        """Laplacian-smooth a point-data scalar. Returns the array name to render.

        Each round-trip (point→cell→point) replaces every vertex value with the
        average of its incident-cell averages — a weighted Laplacian smooth that
        removes cell-level noise while preserving the field shape. 2-5 iterations
        is typical for CFD surface data.

        The smoothed values go into a SEPARATE ``<name>_smoothed`` array, and the
        result is cached. Previously this wrote back over the source array on the
        cached surface mesh, so every re-render smoothed an already-smoothed
        field: switching views back and forth quietly eroded the solution. A
        sharp stagnation peak lost 5% after five view switches and kept going,
        taking the "stagnation Cp" readout in the status bar down with it. The
        solver's own arrays are now left exactly as SU2 wrote them.
        """
        if name not in mesh.point_data:
            return name
        out = f"{name}_smoothed"
        if out in mesh.point_data:
            return out          # already computed for this mesh — reuse, no drift

        n_pts_orig = mesh.n_points
        try:
            mesh.point_data[out] = mesh.point_data[name].copy()
        except Exception:
            return name
        for _ in range(n_iter):
            try:
                tmp = mesh.point_data_to_cell_data()
                tmp = tmp.cell_data_to_point_data()
                if out in tmp.point_data and len(tmp.point_data[out]) == n_pts_orig:
                    mesh.point_data[out] = tmp.point_data[out]
                else:
                    break  # topology changed — stop smoothing to avoid corruption
            except Exception:
                break
        return out

    def _add_freestream_indicator(self, rb, rL, rR):
        """Add freestream direction arrow and flow condition text to the 3D viewport.

        Draws a V∞ arrow upstream of the rocket showing the flow direction
        (accounting for AoA) and annotates Mach, AoA, and velocity.
        """
        if rb is None:
            return
        import numpy as np
        # Annotate the solved condition, not the current spin-box state — the
        # boxes are unlocked the moment a run ends, so they drift from the flow
        # field being drawn.
        if self._result:
            aoa = self._result.angle_of_attack_deg
            mach = self._result.mach
            v_inf = self._result.v_inf
        else:
            aoa = self._sp_aoa.value()
            mach = self._sp_mach.value()
            v_inf = 0.0

        # Arrow geometry: placed upstream-left and above the rocket
        cx = (rb[0] + rb[1]) / 2
        cy = (rb[2] + rb[3]) / 2
        cz = (rb[4] + rb[5]) / 2
        arrow_len = rL * 0.22

        # Start upstream of the nose
        start_x = rb[0] - rL * 0.5
        start_z = cz + rR * 3.5

        aoa_rad = math.radians(aoa)
        dx = math.cos(aoa_rad)
        dz = math.sin(aoa_rad)

        try:
            arrow = pv.Arrow(
                start=(start_x, cy, start_z),
                direction=(dx, 0, dz),
                scale=arrow_len,
                shaft_radius=0.018,
                tip_radius=0.055,
                tip_length=0.22,
                shaft_resolution=24,
                tip_resolution=24,
            )
            self._plotter.add_mesh(
                arrow, color=theme.ACCENT, smooth_shading=True,
                specular=0.5, specular_power=30,
                ambient=0.3,
                name="freestream_arrow"
            )
        except Exception:
            pass

        # Flow condition text annotation
        parts = []
        if v_inf > 0:
            parts.append(f"V\u221e = {v_inf:.0f} m/s")
        parts.append(f"M\u221e = {mach:.2f}")
        parts.append(f"AoA = {aoa:.1f}\u00b0")
        text = "   \u2502   ".join(parts)

        try:
            self._plotter.add_text(
                text,
                position="upper_left",
                font_size=9,
                color=theme.ACCENT,
                shadow=True,
                name="freestream_text",
                font="courier",
            )
        except Exception:
            pass

    def _add_contour_lines(self, mesh, scalar_name, n_lines=12, weighted=True):
        """Overlay iso-contour lines on a surface scalar field.

        Extracts iso-lines at evenly spaced scalar values and renders
        them as thin semi-transparent black lines on top of the colored
        surface, mimicking professional CFD post-processing overlays.

        Levels are spread over the 2-98% band rather than over raw min/max.
        Spanning min/max put nearly every line inside the narrow value range
        held by the nose-tip slivers: on a measured M=0.8 case Cp ran
        [-0.49, 1.53] while 96% of the airframe area sat within [-0.31,
        0.23], so ten of twelve iso-lines drew on the tip and the body — the
        part anyone is looking at — got two.

        ``weighted`` selects area-weighted percentiles (right for surfaces)
        or point-counted ones (right for slices); see :meth:`_auto_clim` for
        why the two differ.
        """
        if mesh is None or scalar_name not in mesh.array_names:
            return
        vals = mesh[scalar_name].flatten()
        # Safety: skip if scalar array size doesn't match mesh topology
        if len(vals) != mesh.n_points and len(vals) != mesh.n_cells:
            return
        valid = vals[np.isfinite(vals)]
        if len(valid) < 10:
            return
        v_min, v_max = field_percentiles(mesh, scalar_name, [2.0, 98.0],
                                         weighted=weighted)
        if not np.isfinite(v_min) or not np.isfinite(v_max):
            v_min, v_max = float(valid.min()), float(valid.max())
        v_min, v_max = float(v_min), float(v_max)
        if v_max - v_min < 1e-10:
            return
        levels = np.linspace(v_min, v_max, n_lines + 2)[1:-1]
        try:
            contours = mesh.contour(isosurfaces=levels.tolist(), scalars=scalar_name)
            if contours.n_points > 0:
                self._plotter.add_mesh(
                    contours, color="#000000", line_width=1.0,
                    opacity=0.45, name="contour_lines",
                    render_lines_as_tubes=False,
                )
        except Exception:
            pass

    # ── Debounced refresh ─────────────────────────────────────────────────────
    def _schedule_refresh(self):
        """Schedule a debounced refresh — batches rapid slider changes
        into one expensive re-render (300 ms delay)."""
        if not hasattr(self, '_refresh_timer'):
            self._refresh_timer = QTimer()
            self._refresh_timer.setSingleShot(True)
            self._refresh_timer.setInterval(300)
            self._refresh_timer.timeout.connect(self._refresh_vis)
        self._refresh_timer.start()  # restart the 300ms countdown

    def _refresh_vis(self):
        idx = self._vis_combo.currentIndex()
        self._plotter.clear()
        # Screen-space ambient occlusion is a depth-buffer pass: it darkens
        # concave areas (fin roots, the nose/body shoulder) after the scalar
        # colours are mapped, so on a field view it silently reports geometry
        # as a lower field value. Keep it for the geometry preview, where it
        # is the point, and drop it wherever the colour has to be readable
        # against a bar.
        try:
            if idx == 0:
                self._plotter.enable_ssao(radius=0.25, bias=0.002,
                                          kernel_size=512, blur=True)
            else:
                self._plotter.disable_ssao()
        except Exception:
            pass
        # Reset scalar range to auto when switching to a DIFFERENT view
        # (don't reset when the same view re-renders from slider changes)
        if not hasattr(self, '_last_vis_idx'):
            self._last_vis_idx = -1
        if idx != self._last_vis_idx:
            self._sp_smin.setValue(self._sp_smin.minimum())
            self._sp_smax.setValue(self._sp_smax.minimum())
            self._last_vis_idx = idx
        # Show compression sensor combo only on compression region view
        self._cb_shock_sensor.setVisible(idx == 11)
        # Turn off probe mode when switching views to prevent click conflicts
        if self._btn_probe.isChecked():
            self._btn_probe.setChecked(False)
        # Show/hide post-processing panels based on view
        has_results = (self._volume_mesh is not None or self._surface_mesh is not None)
        self._disp_grp.setVisible(has_results and idx > 0)
        self._stream_grp.setVisible(has_results and idx == 5)  # streamlines only
        
        if idx in [2, 3, 4, 6, 7, 10]:
            self._chk_interactive_slice.setEnabled(True)
            if self._chk_interactive_slice.isChecked():
                self._toggle_interactive_slice(2)
                return
        else:
            self._chk_interactive_slice.setEnabled(False)
            self._chk_interactive_slice.setChecked(False)
        try:
            if idx == 0:  # Geometry Preview
                if self._current_stl:
                    self._preview(self._current_stl)
                return

            if self._volume_mesh is None and self._surface_mesh is None:
                self._status_lbl.setText("Run CFD first to see flow field results.")
                return

            vm = self._volume_mesh    # full 3D UnstructuredGrid
            sm = self._surface_mesh   # rocket surface

            # ── Clip to rocket near-field ─────────────────────────────────────
            import numpy as np
            if vm is not None:
                vm_local, rb = self._rocket_region(vm)
                # Mid-plane slice origin — Y offset from slider
                y_off = self._get_slice_offset() * max(abs(rb[3] - rb[2]), abs(rb[5] - rb[4])) / 2
                slice_origin = [
                    (rb[0] + rb[1]) / 2,
                    y_off,
                    (rb[4] + rb[5]) / 2,
                ]
            else:
                vm_local = None
                rb = sm.bounds if sm else None
                slice_origin = None

            rL = max(rb[1] - rb[0], 0.01) if rb else 1.0
            rR = max(abs(rb[3] - rb[2]), abs(rb[5] - rb[4])) / 2 if rb else 0.1

            # Helper: resolve scalar name with fallback list
            def _scalar(mesh, *candidates):
                names = mesh.array_names if mesh is not None else []
                for c in candidates:
                    if c in names:
                        return c
                return None

            # -- idx 1: Surface Cp ------------------------------------------------
            if idx == 1 and sm:
                cp_name = _scalar(sm, "Pressure_Coefficient", "Cp", "CpTotal")
                if cp_name is None and "Pressure" in sm.array_names:
                    p_data = sm["Pressure"].flatten()
                    # Resolve P_inf and q_inf robustly from result or ISA spinners
                    q_inf = 1.0
                    P_inf = 101325.0
                    if self._result and getattr(self._result, "dynamic_pressure", 0.0) > 0.0:
                        q_inf = self._result.dynamic_pressure
                        _mach_sq = max(self._mach, 0.01) ** 2
                        P_inf = q_inf * 2.0 / (1.4 * _mach_sq)
                    else:
                        from cfd.solvers.base import isa_conditions
                        try:
                            alt = self._sp_alt.value()
                            mach = self._sp_mach.value()
                            P_isa, T_isa, rho_isa = isa_conditions(alt)
                            a_isa = math.sqrt(1.4 * 287.05 * T_isa)
                            V_isa = mach * a_isa
                            q_inf = 0.5 * rho_isa * V_isa ** 2
                            P_inf = P_isa
                        except Exception:
                            pass
                    sm["Cp_surface"] = (p_data - P_inf) / max(q_inf, 1.0)
                    cp_name = "Cp_surface"
                if cp_name:
                    # Ensure Cp is on point data for smooth rendering
                    # cell_data_to_point_data averages cell values to vertices
                    if cp_name in sm.cell_data and cp_name not in sm.point_data:
                        sm = sm.cell_data_to_point_data()

                    cp_vals = sm[cp_name].copy().flatten()

                    # Replace NaN/inf with 0.0 (neutral Cp) — SU2 surface VTK
                    # may have NaN at boundary junctions or mesh seams
                    nan_mask = ~np.isfinite(cp_vals)
                    n_nan = int(nan_mask.sum())
                    if n_nan > 0:
                        cp_vals[nan_mask] = 0.0
                        sm[cp_name] = cp_vals
                        self._log(f"Fixed {n_nan} NaN/inf Cp values → 0.0")

                    # Light Laplacian smooth (2 iter) to clean up cell artifacts
                    cp_name = self._smooth_surface_scalars(sm, cp_name, n_iter=2)
                    cp_vals = sm[cp_name].flatten()

                    # Dynamic Cp range — symmetric around 0 for diverging colormap.
                    # Bounds are area-weighted percentiles, NOT raw min/max: the
                    # nose tip carries cells ~3e5x smaller than the body, and a
                    # couple of those slivers overshoot the isentropic stagnation
                    # limit (Cp 1.53 measured against a physical ceiling of 1.17
                    # at M=0.8). Scaling to them set clim=+/-1.53 while 99% of the
                    # airframe AREA lives in [-0.35, 0.41] — a quarter of the bar,
                    # which rendered the whole body flat white.
                    cp_min = float(cp_vals.min())
                    cp_max = float(cp_vals.max())
                    clim = self._auto_clim(sm, cp_name, 0.5, 99.5,
                                           symmetric=True, floor=0.05)
                    user_opacity = self._get_user_opacity()
                    user_cmap = self._get_user_cmap("RdBu_r")

                    self._log(
                        f"Cp range: [{cp_min:.3f}, {cp_max:.3f}], "
                        f"clim=[{clim[0]:.3f}, {clim[1]:.3f}], stagnation Cp_max={cp_max:.3f}"
                    )

                    sbar = {
                        "title": "Surface Cp",
                        "vertical": False,
                        "title_font_size": 11,
                        "label_font_size": 10,
                        "color": theme.TEXT,
                        "position_x": 0.25,
                        "position_y": 0.02,
                        "width": 0.5,
                        "height": 0.06,
                        "fmt": "%.3f",
                    }
                    self._add_mesh_pbr(
                        sm, scalars=cp_name, cmap=user_cmap,
                        clim=clim, opacity=user_opacity,
                        show_scalar_bar=True, scalar_bar_args=sbar,
                    )
                    # Contour line overlays for professional CFD look
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(sm, cp_name, n_lines=12)
                    self._status_lbl.setText(
                        f"Surface Cp  [{cp_min:.3f} to {cp_max:.3f}]  "
                        f"(stagnation \u2248 {cp_max:.2f})"
                    )
                else:
                    self._status_lbl.setText("Cp not available on surface mesh.")

            # -- idx 2: Pressure volume slice -------------------------------------
            elif idx == 2 and vm_local:
                slc = vm_local.slice(normal="y", origin=slice_origin)
                slc = self._filter_interior_cells(slc)
                if "Pressure" in slc.array_names:
                    clim = self._auto_clim(slc, "Pressure", 1.0, 99.0, weighted=False)
                    user_cmap = self._get_user_cmap("plasma")
                    self._plotter.add_mesh(
                        slc, scalars="Pressure", cmap=user_cmap,
                        clim=clim,
                        smooth_shading=True,
                        interpolate_before_map=True,
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Pressure (Pa)", "color": theme.TEXT,
                                         "fmt": "%.0f", "position_x": 0.25, "width": 0.5}
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(slc, "Pressure", n_lines=12, weighted=False)
                    self._status_lbl.setText(f"Pressure mid-plane slice  [{clim[0]:.0f} – {clim[1]:.0f} Pa]")

            # -- idx 3: Temperature -----------------------------------------------
            elif idx == 3 and vm_local:
                slc = vm_local.slice(normal="y", origin=slice_origin)
                slc = self._filter_interior_cells(slc)
                if "Temperature" in slc.array_names:
                    clim = self._auto_clim(slc, "Temperature", 1.0, 99.0, weighted=False)
                    user_cmap = self._get_user_cmap("inferno")
                    self._plotter.add_mesh(
                        slc, scalars="Temperature", cmap=user_cmap,
                        clim=clim,
                        smooth_shading=True,
                        interpolate_before_map=True,
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Temperature (K)", "color": theme.TEXT,
                                         "fmt": "%.1f", "position_x": 0.25, "width": 0.5}
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(slc, "Temperature", n_lines=12, weighted=False)
                    self._status_lbl.setText(f"Temperature mid-plane slice  [{clim[0]:.1f} – {clim[1]:.1f} K]")

            # -- idx 4: Velocity magnitude -----------------------------------------
            elif idx == 4 and vm_local:
                slc = vm_local.slice(normal="y", origin=slice_origin)
                slc = self._filter_interior_cells(slc)
                vel_name = _scalar(slc, "Velocity", "V")
                if vel_name:
                    vel = slc[vel_name]
                    slc["Velocity_Magnitude"] = np.linalg.norm(vel, axis=1).astype(np.float32)
                    scalar = "Velocity_Magnitude"
                else:
                    scalar = _scalar(slc, "Velocity_Magnitude", "U")
                # Scale to the field. The fixed 1.4*V_inf ceiling wasted a
                # quarter of the bar on a measured case (max speed 278 m/s
                # against a clim top of 368), and would clip instead on any
                # case with a strong expansion.
                clim = self._auto_clim(slc, scalar or "Pressure", 0.0, 99.5, weighted=False)
                user_cmap = self._get_user_cmap("viridis")
                self._plotter.add_mesh(
                    slc, scalars=scalar or "Pressure", cmap=user_cmap,
                    clim=clim,
                    smooth_shading=True,
                    interpolate_before_map=True,
                    opacity=self._get_user_opacity(),
                    show_scalar_bar=True,
                    scalar_bar_args={"title": f"Speed (m/s)   V\u221e={self._v_inf:.0f}",
                                     "color": theme.TEXT, "fmt": "%.1f",
                                     "position_x": 0.25, "width": 0.5}
                )
                if self._chk_contour_lines.isChecked():
                    self._add_contour_lines(slc, scalar or "Pressure", n_lines=12, weighted=False)
                self._status_lbl.setText(f"Velocity magnitude  V\u221e = {self._v_inf:.1f} m/s")

            # ── idx 5: Streamlines ────────────────────────────────────────
            elif idx == 5 and vm_local:
                vel_name = _scalar(vm_local, "Velocity", "V")
                if vel_name:
                    color_mode = self._cb_stream_color.currentText().lower()
                    if color_mode == "pressure":
                        scalar = _scalar(vm_local, "Pressure", "P")
                        cmap = "plasma"
                        title = "Pressure (Pa)"
                    elif color_mode == "mach":
                        scalar = _scalar(vm_local, "Mach", "Mach_Number")
                        cmap = "coolwarm"
                        title = "Mach"
                    else: # velocity
                        if "Speed" not in vm_local.array_names:
                            vm_local["Speed"] = np.linalg.norm(vm_local[vel_name], axis=1).astype(np.float32)
                        scalar = "Speed"
                        cmap = "viridis"
                        title = "Speed (m/s)"

                    # Seed plane upstream (avoids exact stagnation point singularity on X-axis)
                    n_seeds = self._sp_seed_density.value()
                    res = max(2, int(np.sqrt(n_seeds)))
                    
                    # Ensure it is placed safely inside the wind tunnel inlet boundary
                    seed_x = rb[0] + rL * 0.05
                    
                    seed = pv.Plane(
                        center=(seed_x, 0.001, 0.001), # slightly off-center to miss stagnation singularity
                        direction=(1, 0, 0),
                        i_size=rR * 6.0,
                        j_size=rR * 6.0,
                        i_resolution=res,
                        j_resolution=res,
                    )
                    try:
                        lines = vm_local.streamlines_from_source(
                            seed, vectors=vel_name,
                            integration_direction="forward",
                            max_length=rL * 8.0,
                            initial_step_length=rL * 0.005,
                            terminal_speed=0.01, # higher floor to avoid boundary layer traps
                        )
                        if lines.n_points == 0:
                            raise ValueError("No streamlines generated (flow might be trapped or outside domain)")
                            
                        stream_type = self._cb_stream_type.currentText().lower()
                        if stream_type == "tubes":
                            geom = lines.tube(radius=rR * 0.02)
                        elif stream_type == "ribbons":
                            geom = lines.ribbon(width=rR * 0.05)
                        else:
                            geom = lines
                            
                        self._plotter.add_mesh(
                            geom,
                            scalars=scalar if scalar in geom.array_names else None,
                            cmap=cmap,
                            show_scalar_bar=True,
                            scalar_bar_args={"title": title}
                        )
                        self._status_lbl.setText(f"Aerodynamic streamlines ({lines.n_points:,} pts)")
                    except Exception as se:
                        self._status_lbl.setText(f"Streamline error: {se}")
                else:
                    self._status_lbl.setText("Velocity array not found in volume mesh.")

            elif idx == 6 and vm_local:
                mach_name = _scalar(vm_local, "Mach", "Mach_Number")
                slc = vm_local.slice(normal="y", origin=slice_origin)
                slc = self._filter_interior_cells(slc)
                # Scale to the solved field, not to a fixed guess. The old
                # clim of [0, max(1.5*M_inf, 1.5)] spent 43% of the bar on
                # Mach numbers that do not occur: an M_inf=0.8 case peaks at
                # 0.85, so the whole transonic field was squeezed into the
                # bottom half of a colormap whose midpoint is not the sonic
                # line either. Held to at least [0, 1] so the sonic point
                # keeps a fixed, readable position whenever the flow reaches
                # it, and so subsonic cases are not over-stretched.
                if mach_name:
                    m_hi = float(field_percentiles(slc, mach_name, 99.5, weighted=False)[0])
                    if not np.isfinite(m_hi):
                        m_hi = max(self._mach * 1.2, 1.0)
                    clim = self._get_user_clim(0.0, max(m_hi, 1.0))
                else:
                    clim = self._auto_clim(slc, "Pressure", 1.0, 99.0, weighted=False)
                user_cmap = self._get_user_cmap("coolwarm")
                self._plotter.add_mesh(
                    slc, scalars=mach_name or "Pressure", cmap=user_cmap,
                    clim=clim,
                    smooth_shading=True,
                    interpolate_before_map=True,
                    opacity=self._get_user_opacity(),
                    show_scalar_bar=True,
                    scalar_bar_args={"title": f"Mach  M\u221e={self._mach:.2f}",
                                     "color": theme.TEXT, "fmt": "%.2f",
                                     "position_x": 0.25, "width": 0.5}
                )
                if self._chk_contour_lines.isChecked():
                    self._add_contour_lines(slc, mach_name or "Pressure", n_lines=12, weighted=False)
                self._status_lbl.setText(f"Mach number  M\u221e = {self._mach:.2f}")

            elif idx == 7 and vm_local:
                slc = vm_local.slice(normal="y", origin=slice_origin)
                slc = self._filter_interior_cells(slc)
                if "Density" in slc.array_names:
                    clim = self._auto_clim(slc, "Density", 1.0, 99.0, weighted=False)
                    user_cmap = self._get_user_cmap("cividis")
                    self._plotter.add_mesh(
                        slc, scalars="Density", cmap=user_cmap,
                        clim=clim,
                        smooth_shading=True,
                        interpolate_before_map=True,
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Density (kg/m\u00b3)", "color": theme.TEXT,
                                         "fmt": "%.4f", "position_x": 0.25, "width": 0.5}
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(slc, "Density", n_lines=10, weighted=False)
                self._status_lbl.setText("Density \u2014 mid-plane slice")

            # ── idx 8: Vorticity magnitude ────────────────────────────────
            elif idx == 8 and vm_local:
                # Prefer pre-computed Vorticity_Magnitude from compute_derived_fields
                vort_scalar = _scalar(vm_local, "Vorticity_Magnitude")
                if vort_scalar is None:
                    # Try computing from Vorticity vector
                    vort_vec_name = _scalar(vm_local, "Vorticity", "vorticity")
                    if vort_vec_name is not None:
                        vort_data = vm_local[vort_vec_name]
                        if vort_data.ndim > 1:
                            vm_local["Vorticity_Magnitude"] = np.linalg.norm(vort_data, axis=1).astype(np.float32)
                            vort_scalar = "Vorticity_Magnitude"
                        else:
                            vort_scalar = vort_vec_name  # already scalar

                if vort_scalar:
                    slc = vm_local.slice(normal="y", origin=slice_origin)
                    slc = self._filter_interior_cells(slc)
                    v_vals = slc[vort_scalar] if vort_scalar in slc.array_names else vm_local[vort_scalar]
                    v95 = float(field_percentiles(slc, vort_scalar, 95.0, weighted=False)[0])
                    if not np.isfinite(v95):
                        v95 = float(np.percentile(np.abs(v_vals), 95)) if len(v_vals) > 0 else 1.0
                    clim = self._get_user_clim(0, max(v95, 1.0))
                    user_cmap = self._get_user_cmap("hot")
                    self._plotter.add_mesh(
                        slc, scalars=vort_scalar, cmap=user_cmap,
                        clim=clim,
                        smooth_shading=True,
                        interpolate_before_map=True,
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Vorticity Magnitude (1/s)",
                                         "color": theme.TEXT, "fmt": "%.1f",
                                         "position_x": 0.25, "width": 0.5}
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(slc, vort_scalar, n_lines=10, weighted=False)
                    self._status_lbl.setText(f"Vorticity magnitude \u2014 mid-plane slice (max={v95:.1f} 1/s)")
                else:
                    self._status_lbl.setText("Vorticity not available. Re-run CFD to recompute.")

            # ── idx 9: Q-criterion iso-surface ─────────────────────────────
            elif idx == 9 and vm_local:
                q_name = _scalar(vm_local, "Q_Criterion_Smooth", "Q_Criterion", "Q_criterion", "QCriterion")
                if q_name:
                    try:
                        q_vals = vm_local[q_name]
                        # Use slider-controlled percentile for iso threshold
                        iso_pct = self._get_iso_percentile()
                        q_level = float(np.percentile(q_vals[q_vals > 0], iso_pct)) if (q_vals > 0).any() else 0.1
                        iso = vm_local.contour(isosurfaces=[q_level], scalars=q_name)
                        if iso.n_cells > 0:
                            # Smooth the iso-surface normals for cleaner rendering
                            try:
                                iso = iso.compute_normals(auto_orient_normals=True)
                            except Exception:
                                pass
                            # Color by velocity magnitude for context
                            vel_name = _scalar(iso, "Velocity", "V")
                            user_opacity = self._get_user_opacity()
                            user_cmap = self._get_user_cmap("plasma")
                            if vel_name:
                                iso["Speed"] = np.linalg.norm(iso[vel_name], axis=1).astype(np.float32)
                                v_max = self._v_inf * 1.4
                                self._plotter.add_mesh(
                                    iso, scalars="Speed", cmap=user_cmap,
                                    clim=[0, v_max], opacity=user_opacity * 0.8,
                                    smooth_shading=True,
                                    specular=0.4, specular_power=30,
                                    ambient=0.15,
                                    show_scalar_bar=True,
                                    scalar_bar_args={"title": "Speed (m/s)",
                                                     "color": theme.TEXT}
                                )
                            else:
                                self._plotter.add_mesh(
                                    iso, color=theme.ACCENT_HOVER, opacity=user_opacity * 0.8,
                                    smooth_shading=True,
                                    specular=0.4, specular_power=30,
                                    ambient=0.15,
                                )
                            self._status_lbl.setText(
                                f"Q-criterion iso-surface  (Q={q_level:.2e}, P{iso_pct:.0f}%)  \u2014 coherent vortices"
                            )
                        else:
                            self._status_lbl.setText("No coherent vortex structures found at this Q level")
                    except Exception as e:
                        self._status_lbl.setText(f"Q-criterion error: {e}")
                else:
                    self._status_lbl.setText("Q_Criterion not in output (check VOLUME_OUTPUT config)")

            # -- idx 10: Cp volume slice ------------------------------------------
            elif idx == 10 and vm_local:
                cp_name = _scalar(vm_local, "Pressure_Coefficient", "Cp", "CpTotal")
                if cp_name:
                    slc = vm_local.slice(normal="y", origin=slice_origin)
                    slc = self._filter_interior_cells(slc)
                    clim = self._auto_clim(slc, cp_name, 1.0, 99.0,
                                           symmetric=True, floor=0.05,
                                           weighted=False)
                    half = clim[1]
                    user_cmap = self._get_user_cmap("RdBu_r")
                    self._plotter.add_mesh(
                        slc, scalars=cp_name, cmap=user_cmap,
                        clim=clim,
                        smooth_shading=True,
                        interpolate_before_map=True,
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True,
                        scalar_bar_args={"title": f"Cp  [{clim[0]:.2f} to {clim[1]:.2f}]",
                                         "color": theme.TEXT, "fmt": "%.3f",
                                         "position_x": 0.25, "width": 0.5}
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(slc, cp_name, n_lines=12, weighted=False)
                    self._status_lbl.setText(
                        f"Cp mid-plane slice (auto-scaled +/-{half:.3f})  |  blue=suction, red=stagnation"
                    )
                else:
                    self._status_lbl.setText("Cp not in output -- check VOLUME_OUTPUT= ..., PRESSURE_COEFFICIENT")

            # ── idx 11: Compression Region (unified sensor interface) ─────
            elif idx == 11 and vm_local:
                try:
                    sensor_idx = self._cb_shock_sensor.currentIndex()
                    sensor_map = {
                        0: 'pressure_gradient',
                        1: 'ducros',
                        2: 'dilatation',
                        3: 'mach_gradient',
                        4: 'entropy_gradient',
                    }
                    method = sensor_map.get(sensor_idx, 'pressure_gradient')
                    sensor_label = self._cb_shock_sensor.currentText()

                    # Slice-based visualization
                    p_name = _scalar(vm_local, "Pressure", "P")
                    if p_name:
                        slc = vm_local.slice(normal="y", origin=slice_origin)
                        slc = self._filter_interior_cells(slc)

                        if method == 'pressure_gradient' and slc.n_points > 0 and p_name in slc.array_names:
                            # Legacy pressure gradient on slice
                            grad = slc.compute_derivative(scalars=p_name)
                            grad_key = None
                            for gn in grad.array_names:
                                if "gradient" in gn.lower():
                                    grad_key = gn; break
                            if grad_key is not None:
                                grad_vec = grad[grad_key]
                                grad_mag = np.linalg.norm(grad_vec, axis=1).astype(np.float32)
                                grad["PressureGradient"] = grad_mag
                                g95 = float(field_percentiles(grad, "PressureGradient", 95.0, weighted=False)[0])
                                if not np.isfinite(g95) or g95 <= 0:
                                    g95 = float(np.percentile(grad_mag[grad_mag > 0], 95)) if (grad_mag > 0).any() else 1.0
                                user_cmap = self._get_user_cmap("hot")
                                self._plotter.add_mesh(
                                    grad, scalars="PressureGradient", cmap=user_cmap,
                                    clim=[0, g95],
                                    smooth_shading=True,
                                    interpolate_before_map=True,
                                    opacity=self._get_user_opacity(),
                                    show_scalar_bar=True,
                                    scalar_bar_args={"title": "|∇P| (Pa/m)",
                                                     "color": theme.TEXT, "fmt": "%.0f"}
                                )
                        else:
                            # Use new physics-based sensor on slice
                            try:
                                from cfd.shock_detection import detect_shocks
                                iso = detect_shocks(vm_local, method=method,
                                                    percentile=self._get_iso_percentile())
                                if iso is not None and iso.n_cells > 0:
                                    user_cmap = self._get_user_cmap("hot")
                                    self._plotter.add_mesh(
                                        iso, color=theme.ERR,
                                        opacity=self._get_user_opacity() * 0.7,
                                        smooth_shading=True,
                                        specular=0.3, specular_power=20,
                                        ambient=0.15,
                                        label=f"{sensor_label} compression surface"
                                    )
                                    # Also show on slice for context
                                    if p_name in slc.array_names:
                                        self._plotter.add_mesh(
                                            slc, scalars=p_name, cmap="plasma",
                                            smooth_shading=True,
                                            interpolate_before_map=True,
                                            opacity=0.4,
                                            show_scalar_bar=True,
                                            scalar_bar_args={"title": "Pressure (Pa)",
                                                             "color": theme.TEXT}
                                        )
                                else:
                                    self._status_lbl.setText(f"{sensor_label}: no compression structures detected")
                            except ImportError:
                                self._status_lbl.setText("Compression sensor module not available")
                            except Exception as se:
                                self._status_lbl.setText(f"{sensor_label} error: {se}")

                        # Also try 3D iso-surface overlay for supersonic
                        mach_name = _scalar(vm_local, "Mach", "Mach_Number")
                        max_mach = float(vm_local[mach_name].max()) if mach_name else 0.0
                        if max_mach >= 0.95 and method == 'pressure_gradient':
                            try:
                                from cfd.shock_detection import detect_shock_surfaces
                                iso = detect_shock_surfaces(vm_local, p_name)
                                if iso is not None and iso.n_cells > 0:
                                    self._plotter.add_mesh(
                                        iso, color=theme.ERR, opacity=0.5,
                                        smooth_shading=True, specular=0.3,
                                        label="Compression surface"
                                    )
                            except Exception:
                                pass
                        if max_mach >= 0.95:
                            try:
                                self._plotter.add_legend()
                            except Exception:
                                pass

                        self._status_lbl.setText(
                            f"{sensor_label} — compression/expansion regions  "
                            f"(M_max={max_mach:.2f})"
                        )
                    else:
                        self._status_lbl.setText("Pressure not found for compression region view")
                except Exception as e:
                    self._status_lbl.setText(f"Compression region error: {e}")
                    import traceback; traceback.print_exc()

            # ── idx 12: Boundary Layer Y+ ───────────────────────────
            elif idx == 12 and sm:
                from cfd.boundary_layer import extract_yplus
                yp = extract_yplus(sm)
                if yp is not None:
                    sm["YPlus_Vis"] = yp
                    # Smooth Y+ for cleaner visualization
                    yp_name = self._smooth_surface_scalars(sm, "YPlus_Vis", n_iter=2)
                    yp_vals = sm[yp_name]
                    # Area-weighted: the tip slivers carry their own y+ and
                    # vastly outnumber the body cells per unit of wetted area.
                    yp_p95 = float(field_percentiles(sm, yp_name, 95.0)[0])
                    if not np.isfinite(yp_p95):
                        yp_p95 = float(np.percentile(yp_vals[yp_vals > 0], 95)) if (yp_vals > 0).any() else 300.0
                    yp_hi = min(300.0, max(yp_p95, 1.0))
                    clim = self._get_user_clim(0, yp_hi)
                    sbar = {
                        "title": "Y+",
                        "vertical": False,
                        "title_font_size": 11,
                        "label_font_size": 10,
                        "color": theme.TEXT,
                        "position_x": 0.25,
                        "position_y": 0.02,
                        "width": 0.5,
                        "height": 0.06,
                        "fmt": "%.1f",
                    }
                    self._add_mesh_pbr(
                        sm, scalars=yp_name, cmap=self._get_user_cmap("turbo"),
                        clim=clim, opacity=self._get_user_opacity(),
                        show_scalar_bar=True, scalar_bar_args=sbar,
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(sm, yp_name, n_lines=10)
                    self._status_lbl.setText("Boundary layer Y+ (SST needs Y+ ~ 1)")
                else:
                    self._status_lbl.setText("Y+ not in output")

            # ── idx 13: Wall Shear Stress + Skin-Friction Lines ────────
            elif idx == 13 and sm:
                from cfd.boundary_layer import extract_wall_shear, detect_separation
                # SU2 stores the dimensionless skin-friction coefficient —
                # dimensionalize with q_inf so the scalar bar really is Pa.
                _q_inf = getattr(self._result, "dynamic_pressure", None) if self._result else None
                shear = extract_wall_shear(sm, q_inf=_q_inf)
                if shear is not None:
                    sm["WallShear"] = shear
                    ws_name = self._smooth_surface_scalars(sm, "WallShear", n_iter=2)
                    sbar = {
                        "title": "Wall Shear (Pa)",
                        "vertical": False,
                        "title_font_size": 11,
                        "label_font_size": 10,
                        "color": theme.TEXT,
                        "position_x": 0.25,
                        "position_y": 0.02,
                        "width": 0.5,
                        "height": 0.06,
                        "fmt": "%.3f",
                    }
                    # Percentile clim, like every other field view: raw wall
                    # shear has a long tail and one outlier cell was setting the
                    # whole scale. This also makes the clim slider work here.
                    _ws = np.asarray(sm[ws_name]).flatten()
                    _ws = _ws[np.isfinite(_ws)]
                    _hi = float(field_percentiles(sm, ws_name, 99.0)[0])
                    if not np.isfinite(_hi):
                        _hi = float(np.percentile(_ws, 99)) if _ws.size else 1.0
                    self._add_mesh_pbr(
                        sm, scalars=ws_name, cmap=self._get_user_cmap("turbo"),
                        clim=self._get_user_clim(0.0, max(_hi, 1e-9)),
                        opacity=self._get_user_opacity(),
                        show_scalar_bar=True, scalar_bar_args=sbar,
                    )
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(sm, ws_name, n_lines=10)

                    # Skin-friction streamlines overlay
                    try:
                        from cfd.boundary_layer import compute_skin_friction_streamlines
                        sf_lines = compute_skin_friction_streamlines(sm, n_seeds=40)
                        if sf_lines is not None and sf_lines.n_points > 0:
                            self._plotter.add_mesh(
                                sf_lines, color="#f0f6fc", line_width=1.5,
                                opacity=0.6, label="Skin-friction lines"
                            )
                    except Exception:
                        pass

                    # Separation detection with lines
                    sep = detect_separation(sm)
                    if sep is not None:
                        sep_lines = getattr(sep, 'get', lambda k, d=None: d)('separation_lines', None) if hasattr(sep, 'get') else None
                        if sep_lines is not None and hasattr(sep_lines, 'n_points') and sep_lines.n_points > 0:
                            self._plotter.add_mesh(
                                sep_lines, color=theme.ACCENT, line_width=3.0,
                                opacity=0.9, label="Separation line"
                            )
                        elif hasattr(sep, '__array__') or isinstance(sep, np.ndarray):
                            sep_arr = np.asarray(sep)
                            if sep_arr.any():
                                sep_mesh = sm.extract_points(sep_arr)
                                if sep_mesh.n_points > 0:
                                    self._plotter.add_mesh(
                                        sep_mesh, color=theme.ACCENT, point_size=3,
                                        render_points_as_spheres=True, label="Separation"
                                    )
                        reattach_lines = getattr(sep, 'get', lambda k, d=None: d)('reattachment_lines', None) if hasattr(sep, 'get') else None
                        if reattach_lines is not None and hasattr(reattach_lines, 'n_points') and reattach_lines.n_points > 0:
                            self._plotter.add_mesh(
                                reattach_lines, color=theme.OK, line_width=3.0,
                                opacity=0.9, label="Reattachment line"
                            )
                        self._plotter.add_legend()

                    self._status_lbl.setText("Wall shear stress + skin-friction lines")
                else:
                    self._status_lbl.setText("Wall shear not in output")

            # ── idx 14: Force Vectors (Pressure + Shear) ─────────────
            elif idx == 14 and sm:
                try:
                    from cfd.post_processing import compute_force_vectors
                    # Resolve q_inf/P_inf the same way the Cp view does. This
                    # read dynamic_pressure straight off the result with only
                    # an `if self._result` guard, so a result carrying q_inf=0
                    # (never solved, or a stale/partial load) produced P_inf=0
                    # and then a divide-by-zero building Cp_surface below —
                    # the whole view collapsed into the except with a bare
                    # "Force vector error" and no arrows. The `_mach_sq > 0`
                    # ternary could never fire as a fallback either, since
                    # _mach_sq is floored at 1e-4.
                    _mach_sq = max(self._mach, 0.01) ** 2
                    q_inf = float(getattr(self._result, "dynamic_pressure", 0.0) or 0.0) if self._result else 0.0
                    if q_inf > 0.0:
                        P_inf = q_inf * 2.0 / (1.4 * _mach_sq)
                    else:
                        from cfd.solvers.base import isa_conditions
                        P_isa, T_isa, rho_isa = isa_conditions(self._sp_alt.value())
                        V_isa = self._sp_mach.value() * math.sqrt(1.4 * 287.05 * T_isa)
                        q_inf = max(0.5 * rho_isa * V_isa ** 2, 1.0)
                        P_inf = P_isa

                    fv_data = compute_force_vectors(
                        sm,
                        freestream_pressure=P_inf,
                        dynamic_pressure=q_inf,
                        n_samples=1500,
                        smoothing_iterations=2
                    )

                    if fv_data is not None and fv_data.n_points > 0:
                        cp_max = 1.0   # arrow clim fallback when Cp array missing
                        # PBR base surface colored by Cp
                        if "Cp" in fv_data.array_names:
                            cp_vals = fv_data["Cp"]
                            cp_max = max(abs(float(np.percentile(cp_vals, 2))),
                                         abs(float(np.percentile(cp_vals, 98))), 0.1)
                            sm_copy = sm.copy()
                            sm_p = sm_copy.point_data["Pressure"] if "Pressure" in sm_copy.point_data else sm_copy.cell_data_to_point_data().point_data["Pressure"]
                            sm_copy["Cp_surface"] = (sm_p - P_inf) / q_inf
                            self._add_mesh_pbr(
                                sm_copy, scalars="Cp_surface",
                                cmap=self._get_user_cmap("RdBu_r"),
                                clim=[-cp_max, cp_max],
                                opacity=0.35, show_scalar_bar=False,
                            )
                        else:
                            self._plotter.add_mesh(sm, color="#c8d0dc", opacity=0.4, smooth_shading=True)

                        # Generate high-quality arrows.
                        # Uniform arrow size: direction shows where the force
                        # acts, Cp colouring carries the magnitude. Raw force
                        # magnitudes scale with local cell area, so length-
                        # scaled glyphs measure the mesh as much as the flow.
                        # (compute_force_vectors still returns a sqrt-scaled
                        # ForceVector and a true-magnitude ForceVectorTrue for
                        # callers that do want length encoding; this view
                        # deliberately uses neither, and no longer claims
                        # "adaptive scaling" in the status line.)
                        arrow_len = rL * 0.08

                        # Stand the arrows OFF the wall.
                        #
                        # pv.Arrow puts its TAIL at the glyph point and grows
                        # along the vector. Pressure arrows point inward, so
                        # drawn as-is every compression arrow is buried inside
                        # the body — visible only as a smear through the 0.35-
                        # opacity surface, which made the stagnation region look
                        # empty while suction arrows stuck out cleanly.
                        #
                        # Shifting the inward ones out by one arrow length puts
                        # their HEAD on the wall and their tail in free space:
                        # compression reads as pushing into the surface, suction
                        # as pulling off it, and neither is occluded.
                        _p = np.asarray(fv_data.points)
                        _v = np.asarray(fv_data["ForceVector"])
                        _n = np.asarray(fv_data["NormalDirection"])
                        _inward = np.einsum("ij,ij->i", _v, _n) < 0.0
                        offset = np.where(_inward[:, None], _n * arrow_len, 0.0)
                        fv_draw = fv_data.copy()
                        fv_draw.points = (_p + offset).astype(np.float32)

                        arrows = fv_draw.glyph(
                            orient="ForceVector",
                            scale=False,
                            factor=arrow_len,
                            geom=pv.Arrow(shaft_resolution=20, tip_resolution=40, shaft_radius=0.03, tip_radius=0.1)
                        )
                        
                        # Color arrows by gauge pressure/Cp (red=pushing in, blue=pulling out)
                        self._plotter.add_mesh(
                            arrows, scalars="Cp", cmap="RdBu_r", clim=[-cp_max, cp_max],
                            show_scalar_bar=True,
                            scalar_bar_args={"title": "Force (Cp)"},
                            smooth_shading=True,
                            ambient=0.2, diffuse=0.8, specular=0.3
                        )
                        
                        self._status_lbl.setText(
                            f"Aerodynamic force vectors ({fv_data.n_points} samples)  "
                            f"uniform length, coloured by Cp  |  q∞={q_inf:.0f} Pa"
                        )
                    else:
                        self._status_lbl.setText("Could not compute force vectors (missing Pressure data)")
                except Exception as e:
                    self._status_lbl.setText(f"Force vector error: {e}")
                    import traceback; traceback.print_exc()

            # ── idx 15: Lambda-2 criterion (enhanced) ─────────────────
            elif idx == 15 and vm_local:
                lambda_name = _scalar(vm_local, "Lambda2_Smooth", "Lambda2", "LAMBDA2", "lambda2")
                if lambda_name:
                    try:
                        l_vals = vm_local[lambda_name]
                        neg_mask = l_vals < 0
                        if neg_mask.any():
                            # Slider-controlled percentile threshold
                            iso_pct = 100.0 - self._get_iso_percentile()
                            l_level = float(np.percentile(l_vals[neg_mask], max(iso_pct, 1)))
                            iso = vm_local.contour(isosurfaces=[l_level], scalars=lambda_name)
                            if iso.n_cells > 0:
                                # Smooth normals for clean rendering
                                try:
                                    iso = iso.compute_normals(auto_orient_normals=True)
                                except Exception:
                                    pass
                                v_name = _scalar(iso, "Vorticity_Magnitude", "Speed", "Mach")
                                user_opacity = self._get_user_opacity()
                                user_cmap = self._get_user_cmap("plasma")
                                if v_name and v_name in iso.array_names:
                                    self._plotter.add_mesh(
                                        iso, scalars=v_name, cmap=user_cmap,
                                        opacity=user_opacity * 0.8,
                                        smooth_shading=True,
                                        specular=0.4, specular_power=30,
                                        ambient=0.15,
                                        show_scalar_bar=True,
                                        scalar_bar_args={"title": f"{v_name}"}
                                    )
                                else:
                                    self._plotter.add_mesh(
                                        iso, color="#d2a8ff",
                                        opacity=user_opacity * 0.8,
                                        smooth_shading=True,
                                        specular=0.4, specular_power=30,
                                        ambient=0.15,
                                    )
                                self._status_lbl.setText(
                                    f"Lambda-2 iso-surface  (λ2={l_level:.2e}, P{iso_pct:.0f}%)  — coherent vortex cores"
                                )
                            else:
                                self._status_lbl.setText("No coherent vortex structures found at this Lambda-2 level")
                        else:
                            self._status_lbl.setText("No negative Lambda-2 regions (no strong vortex cores detected)")
                    except Exception as e:
                        self._status_lbl.setText(f"Lambda-2 error: {e}")
                else:
                    self._status_lbl.setText("Lambda-2 not yet computed. Re-run CFD to regenerate results.")

            # -- idx 16: Temperature — Surface Contour (Aerodynamic Wall Heating) ----
            elif idx == 16 and sm:
                t_name = _scalar(sm, "Temperature")
                t_is_modelled = t_name is None
                if t_name is None:
                    # Adiabatic-wall recovery temperature, for solvers that do
                    # not write a wall Temperature field.
                    #
                    #   T_r = T_local * (1 + r * (gamma-1)/2 * M_local^2)
                    #
                    # with the local Mach recovered isentropically from the
                    # wall pressure and the freestream stagnation pressure.
                    # Everything here is either measured (P, rho) or a
                    # textbook constant, and the result is bounded above by
                    # the freestream stagnation temperature because r < 1.
                    #
                    # What this replaces was not a physical model: it took a
                    # min/max-normalised Cp, raised it to an arbitrary 0.45
                    # power for "peak sharpening", and multiplied by an
                    # invented axial decay (1 - 0.15*x^0.8) presented as
                    # boundary-layer cooling. Because the Cp normalisation was
                    # against the data's own extremes, the coldest point on any
                    # body was pinned to zero heating no matter what the flow
                    # was doing, and the whole field rescaled itself whenever
                    # the Cp range moved. It also took T_inf as the median of
                    # the SURFACE static temperature, which is not a freestream
                    # value on a body that is mostly boundary layer.
                    if "Pressure" in sm.array_names and "Density" in sm.array_names:
                        p_arr = sm["Pressure"].flatten().astype(np.float64)
                        rho_arr = sm["Density"].flatten().astype(np.float64)
                        safe_rho = np.where(rho_arr < 1e-12, 1e-12, rho_arr)
                        T_static = p_arr / (safe_rho * 287.05)

                        gamma = 1.4
                        r_turb = 0.89   # turbulent recovery factor \u2248 Pr^(1/3)
                        M_inf = self._mach if self._result else self._sp_mach.value()
                        M_inf = max(float(M_inf), 1e-3)

                        # Freestream static/stagnation pressure
                        q_inf = float(getattr(self._result, "dynamic_pressure", 0.0) or 0.0) if self._result else 0.0
                        if q_inf > 0.0:
                            P_inf = q_inf * 2.0 / (gamma * M_inf ** 2)
                        else:
                            from cfd.solvers.base import isa_conditions
                            P_inf, _T_isa, _rho_isa = isa_conditions(self._sp_alt.value())
                        P_0 = P_inf * (1.0 + (gamma - 1.0) / 2.0 * M_inf ** 2) ** (gamma / (gamma - 1.0))

                        # Local Mach from the isentropic pressure ratio.
                        # Clamped at 0: wall pressure above P_0 is a solver
                        # overshoot, not a real compression, and would give a
                        # negative M^2.
                        ratio = np.maximum(P_0 / np.maximum(p_arr, 1.0), 1.0)
                        M_local_sq = np.maximum(
                            2.0 / (gamma - 1.0) * (ratio ** ((gamma - 1.0) / gamma) - 1.0),
                            0.0,
                        )

                        T_wall = T_static * (1.0 + r_turb * (gamma - 1.0) / 2.0 * M_local_sq)

                        sm["Temperature"] = T_wall.astype(np.float32)
                        t_name = "Temperature"
                        self._log(
                            f"Recovery wall temperature modelled from wall pressure "
                            f"(M\u221e={M_inf:.2f}, P\u221e={P_inf:.0f} Pa, r={r_turb}) \u2014 "
                            f"range [{T_wall.min():.1f}, {T_wall.max():.1f}] K"
                        )

                if t_name:
                    # Cell \u2192 point conversion for smooth interpolation
                    if t_name in sm.cell_data and t_name not in sm.point_data:
                        sm = sm.cell_data_to_point_data()

                    t_vals = sm[t_name].copy().flatten()

                    # Fix NaN/inf
                    nan_mask = ~np.isfinite(t_vals)
                    n_nan = int(nan_mask.sum())
                    if n_nan > 0:
                        t_median = float(np.nanmedian(t_vals[~nan_mask])) if (~nan_mask).any() else 288.0
                        t_vals[nan_mask] = t_median
                        sm[t_name] = t_vals
                        self._log(f"\u26a0 Fixed {n_nan} NaN/inf Temperature values")

                    # Multi-pass smoothing: Laplacian (3 iter) + Gaussian if available
                    t_name = self._smooth_surface_scalars(sm, t_name, n_iter=3)
                    try:
                        # gaussian_smooth_surface also writes back in place, so
                        # only let it touch the derived copy, never the raw field.
                        from cfd.post_processing import gaussian_smooth_surface
                        if t_name.endswith("_smoothed"):
                            gaussian_smooth_surface(sm, t_name, sigma=1.5, n_iter=1)
                    except (ImportError, Exception):
                        pass
                    t_vals = sm[t_name].flatten()

                    # Full range for diagnostics
                    t_lo = float(t_vals.min())
                    t_hi = float(t_vals.max())
                    # Area-weighted percentile clim (see _auto_clim)
                    clim = self._auto_clim(sm, t_name, 1.0, 99.0, floor=10.0)

                    self._log(
                        f"Wall Temperature: [{t_lo:.1f}, {t_hi:.1f}] K  "
                        f"(clim: {clim[0]:.1f}\u2013{clim[1]:.1f})"
                    )

                    sbar = {
                        "title": "Wall Temperature (K)",
                        "vertical": False,
                        "title_font_size": 11,
                        "label_font_size": 10,
                        "color": theme.TEXT,
                        "position_x": 0.25,
                        "position_y": 0.02,
                        "width": 0.5,
                        "height": 0.06,
                        "fmt": "%.1f",
                    }
                    user_cmap = self._get_user_cmap("inferno")
                    self._add_mesh_pbr(
                        sm, scalars=t_name, cmap=user_cmap,
                        clim=clim, opacity=self._get_user_opacity(),
                        show_scalar_bar=True, scalar_bar_args=sbar,
                    )
                    # Contour line overlays
                    if self._chk_contour_lines.isChecked():
                        self._add_contour_lines(sm, t_name, n_lines=12)
                    # Say which one it is. This claimed "recovery model" even
                    # when the solver had supplied a real wall Temperature,
                    # which is the usual case on SU2 output.
                    _src = "recovery model from wall pressure" if t_is_modelled else "solver wall field"
                    self._status_lbl.setText(
                        f"Aerodynamic Wall Temperature  [{t_lo:.1f} K to {t_hi:.1f} K]  "
                        f"(M\u221e={self._mach:.2f}, {_src})"
                    )
                else:
                    self._status_lbl.setText("Temperature not available on surface mesh.")

            # -- Overlay: STL transformed into CFD frame (K2: Z-forward -> CFD: X-forward) --
            # Skip for surface-mapped views (Cp, Y+, WallShear, ForceVectors)
            # where the CFD surface mesh IS the visualization — the opaque STL
            # overlay would hide the scalar contour.
            _surface_viz_indices = {1, 12, 13, 14, 16}
            if idx > 0 and idx not in _surface_viz_indices:
                stl_surf = None
                if self._current_stl and self._current_stl.is_file():
                    try:
                        stl_surf = self._stl_in_cfd_frame()
                    except Exception:
                        stl_surf = sm   # fallback to SU2 surface mesh
                else:
                    stl_surf = sm
                if stl_surf is not None:
                    self._plotter.add_mesh(
                        stl_surf, color="#c8d0dc", opacity=0.82,
                        show_scalar_bar=False, lighting=True, smooth_shading=True
                    )

            self._plotter.add_axes()

            # ── Freestream direction indicator (V\u221e arrow + Mach/AoA) ──────────
            if idx > 0 and rb is not None:
                self._add_freestream_indicator(rb, rL, rR)

            # ── Camera: focus on the rocket ───────────────────────────────────
            if rb is not None:
                cx = (rb[0] + rb[1]) / 2
                cy = (rb[2] + rb[3]) / 2
                cz = (rb[4] + rb[5]) / 2
                self._plotter.camera_position = [
                    (cx, cy - rL * 2.2, cz + rL * 0.1),
                    (cx, cy, cz),
                    (0, 0, 1),
                ]
                self._plotter.reset_camera(bounds=[
                    rb[0] - rL * 0.5,  rb[1] + rL * 0.7,
                    rb[2] - rR * 3.5,  rb[3] + rR * 3.5,
                    rb[4] - rR * 3.5,  rb[5] + rR * 3.5,
                ])
            else:
                self._plotter.reset_camera()

        except Exception as e:
            self._log(f"Visualization error: {e}")
            import traceback; traceback.print_exc()




    def _inject_results(self):
        if self._result:
            from cfd.post_processing import inject_cfd_results_into_engine
            inject_cfd_results_into_engine(self._result, self.engine)
            self._log("CFD results injected into simulation engine.")

    def _export_results_csv(self):
        """CSV of the single-run CFD results: boundary conditions, force
        coefficients + drag split, and the convergence residual history."""
        if not self._result:
            return
        import csv
        from pathlib import Path
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "cfd_results.csv", "CSV Files (*.csv)")
        if not path:
            return
        r = self._result
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["quantity", "value"])
                # From the RESULT, never the spin boxes: _on_finished unlocks
                # them, so a spinner touched after the run would be exported
                # against the old coefficients.
                w.writerow(["mach", r.mach])
                w.writerow(["altitude_m", r.altitude_m])
                w.writerow(["aoa_deg", r.angle_of_attack_deg])
                w.writerow(["reynolds", r.reynolds])
                w.writerow(["dynamic_pressure_Pa", r.dynamic_pressure])
                w.writerow(["reference_area_m2", r.reference_area_m2])
                w.writerow(["reference_length_m", r.ref_length])
                w.writerow(["cd", r.cd])
                w.writerow(["cd_pressure", r.cd_pressure])
                w.writerow(["cd_friction", r.cd_friction])
                w.writerow(["cd_base", r.cd_base])
                w.writerow(["cd_forebody_pressure", r.cd_forebody_pressure])
                w.writerow(["cd_wave", r.cd_wave])
                w.writerow(["cl", r.cl])
                w.writerow(["cm", r.cm])
                w.writerow(["cm_cg", r.cm_cg])
                w.writerow(["cp_from_nose_m", r.cp_from_nose_m])
                w.writerow(["cp_location_m", r.cp_location_m])
                w.writerow(["force_axial_N", r.force_axial])
                w.writerow(["force_normal_N", r.force_normal])
                w.writerow(["turbulence_model", r.turbulence_model])
                w.writerow(["converged", r.converged])
                # A number leaving the app has to carry what it is worth with
                # it. A CSV of coefficients with no provenance is exactly how a
                # lower-bound drag ends up in someone else's spreadsheet as a
                # drag.
                w.writerow(["convergence_note",
                            getattr(r, "convergence_note", "")])
                w.writerow(["residual_drop_decades",
                            f"{getattr(r, 'residual_drop_decades', 0.0):.2f}"])
                w.writerow(["wall_resolved", getattr(r, "wall_resolved", True)])
                w.writerow(["wall_warning", getattr(r, "wall_warning", "")])
                w.writerow([])
                w.writerow(["iteration", "residual"])
                for it, res in (getattr(r, "residual_history", None) or []):
                    w.writerow([it, res])
            self._log(f"Results CSV saved: {Path(path).name}")
        except Exception as e:
            self._log(f"CSV export failed: {e}")

    def _export_pdf(self):
        """CFD PDF: boundary conditions, force coefficients, convergence, Cp,
        polar (if a sweep ran) and a 3D contour screenshot."""
        if not self._result:
            return
        from pathlib import Path
        from ui.pdf_report import save_report
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report", "cfd_report.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        r = self._result
        kv = [
            # From the result, not the spin boxes — see _export_results_csv.
            ("Mach", f"{r.mach:.2f}"),
            ("Altitude", f"{r.altitude_m:.0f} m"),
            ("Angle of attack", f"{r.angle_of_attack_deg:.1f} °"),
            ("Total Cd", f"{r.cd:.5f}"),
            ("  Pressure Cd", f"{r.cd_pressure:.5f}"),
            ("  Friction Cd", f"{r.cd_friction:.5f}"),
            ("Cl", f"{r.cl:.5f}"),
            ("Cm", f"{r.cm:.5f}"),
            ("Converged", "yes" if r.converged else "no"),
        ]
        # Capture the matplotlib plot widgets that have data.
        figs = []
        for w in ("_res_plot", "_cp_plot", "_polar_plot"):
            wid = getattr(self, w, None)
            fig = getattr(wid, "figure", None)
            if fig is not None and len(fig.axes) and fig.axes[0].has_data():
                figs.append(fig)
        # Capture every contour view: cycle the visualization combo through all
        # field modes, render each, and screenshot it. Restore the original view
        # afterward.
        images = []
        plotter = getattr(self, "_plotter", None)
        combo = getattr(self, "_vis_combo", None)
        if plotter is not None and combo is not None and combo.isEnabled():
            from PyQt6.QtWidgets import QApplication
            orig = combo.currentIndex()
            self._log("Rendering contours for the report…")
            for i in range(1, combo.count()):     # skip 0 = Geometry Preview
                name = combo.itemText(i).replace("—", "-")
                try:
                    combo.blockSignals(True); combo.setCurrentIndex(i); combo.blockSignals(False)
                    self._refresh_vis()
                    plotter.render()
                    QApplication.processEvents()
                    img = plotter.screenshot(return_img=True)
                    if img is not None:
                        images.append((name, img))
                except Exception as e:
                    logger.debug("contour %s skipped: %s", name, e)
            # Restore the view the user was looking at.
            try:
                combo.blockSignals(True); combo.setCurrentIndex(orig); combo.blockSignals(False)
                self._refresh_vis()
            except Exception:
                pass
        elif plotter is not None:
            try:
                images.append(("3D view", plotter.screenshot(return_img=True)))
            except Exception:
                pass
        ok = save_report(path, "CFD Report",
                         f"Mach {self._sp_mach.value():.2f} · {self._sp_alt.value():.0f} m · "
                         f"AoA {self._sp_aoa.value():.1f}°", kv, figures=figs, images=images)
        if ok:
            self._log(f"PDF report saved: {Path(path).name}")
        else:
            self._log("PDF report failed — see log.")

    def _export_vtk(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder and self._result:
            import shutil
            for f in [self._result.volume_vtk, self._result.surface_vtk]:
                if f and Path(f).is_file():
                    shutil.copy(f, Path(folder) / Path(f).name)
            self._log(f"VTK exported to {folder}")

    def _export_to_structures(self):
        """Map CFD pressure, wall shear, and temperature to the Structures workspace.
        
        Injects force coefficients + flow conditions into the engine state,
        and extracts summary wall shear / temperature data from the surface mesh.
        """
        if not self._result:
            self._log("No CFD results to export.")
            return

        # Step 1: Inject aero coefficients into engine (same as _inject_results)
        #
        # inject_cfd_results_into_engine refuses a non-converged result, and it
        # is right to. The old else-branch here announced "injecting raw values"
        # and then injected nothing at all — so the log claimed a hand-off that
        # never happened, and the surface-VTK steps below carried on regardless.
        if self._result.converged:
            from cfd.post_processing import inject_cfd_results_into_engine
            inject_cfd_results_into_engine(self._result, self.engine)
            self._log(f"Force coefficients injected: Cd={self._result.cd:.4f}, "
                      f"Cl={self._result.cl:.4f}, Cm={self._result.cm:.4f}")
        else:
            self._log("CFD did not converge — coefficients NOT injected into the "
                      "engine. The surface field below is still exported, but "
                      "treat any structural result from it as unvalidated.")

        # Step 2: Extract wall shear and temperature from surface mesh
        sm = self._surface_mesh
        if sm is not None:
            import numpy as np
            info_lines = []

            # Pressure map
            if "Pressure" in sm.array_names:
                P = sm["Pressure"]
                info_lines.append(f"  Pressure: min={P.min():.0f} Pa, max={P.max():.0f} Pa")

            # Wall shear stress — SU2 stores the dimensionless coefficient;
            # report it dimensionalized (τ = Cf·q_inf) so structures gets Pa.
            q_inf = float(getattr(self._result, "dynamic_pressure", 0.0) or 0.0)
            for shear_name in ["Skin_Friction_Coefficient", "Wall_Shear", "Cf"]:
                if shear_name in sm.array_names:
                    cf = sm[shear_name]
                    if cf.ndim > 1:
                        cf_mag = np.linalg.norm(cf, axis=1)
                    else:
                        cf_mag = np.abs(cf)
                    if shear_name in ("Skin_Friction_Coefficient", "Cf") and q_inf > 0:
                        tau = cf_mag * q_inf
                        info_lines.append(
                            f"  Wall shear: min={tau.min():.2f} Pa, "
                            f"max={tau.max():.2f} Pa (Cf max={cf_mag.max():.5f}, "
                            f"q_inf={q_inf:.0f} Pa)")
                    else:
                        info_lines.append(f"  Wall shear ({shear_name}): "
                                          f"min={cf_mag.min():.6f}, max={cf_mag.max():.6f}")
                    break

            # Temperature
            for temp_name in ["Temperature", "T"]:
                if temp_name in sm.array_names:
                    T = sm[temp_name]
                    info_lines.append(f"  Temperature: min={T.min():.1f} K, max={T.max():.1f} K")
                    break

            if info_lines:
                self._log("Surface field summary:")
                for line in info_lines:
                    self._log(line)
            else:
                self._log("No pressure/shear/temperature arrays found in surface mesh.")

            self._log(f"  Surface VTK: {self._result.surface_vtk}")
        else:
            self._log("Surface mesh not loaded — run post-processing first.")

        # Step 3: Confirm readiness
        if self._result.surface_vtk and self._result.surface_vtk.is_file():
            self._log("Surface VTK ready for FEM pressure mapping.")
            self._log("→ Switch to Structures workspace, check 'Map Pressure from CFD', and run analysis.")
        else:
            self._log("No surface VTK file found for direct pressure field mapping.")

    def _toggle_interactive_slice(self, state):
        import numpy as np
        if self._plotter is None or self._volume_mesh is None:
            return
            
        is_checked = (state == 2)
        idx = self._vis_combo.currentIndex()
        
        # Helper: resolve scalar name with fallback list
        def _scalar(mesh, *candidates):
            names = mesh.array_names if mesh is not None else []
            for c in candidates:
                if c in names:
                    return c
            return None
        
        # We only support interactive slicing on certain volume fields
        scalar = None
        cmap = "viridis"
        clim = None
        title = ""
        
        if idx == 2:   # Pressure Volume Slice
            scalar = _scalar(self._volume_mesh, "Pressure", "P")
            cmap = "plasma"
            title = "Pressure (Pa)"
        elif idx == 3: # Temperature
            scalar = _scalar(self._volume_mesh, "Temperature", "T")
            cmap = "inferno"
            title = "Temperature (K)"
        elif idx == 4: # Velocity
            vel_name = _scalar(self._volume_mesh, "Velocity", "V")
            if vel_name:
                if "Speed" not in self._volume_mesh.array_names:
                    self._volume_mesh["Speed"] = np.linalg.norm(self._volume_mesh[vel_name], axis=1)
                scalar = "Speed"
                cmap = "viridis"
                v_max = self._v_inf * 1.4
                clim = [0, v_max]
                title = "Speed (m/s)"
        elif idx == 6: # Mach
            scalar = _scalar(self._volume_mesh, "Mach", "Mach_Number")
            cmap = "coolwarm"
            mach_max = max(self._mach * 1.5, 1.5)
            clim = [0, mach_max]
            title = "Mach"
        elif idx == 7: # Density
            scalar = _scalar(self._volume_mesh, "Density")
            cmap = "cividis"
            title = "Density (kg/m\u00b3)"
        elif idx == 10: # Cp
            scalar = _scalar(self._volume_mesh, "Pressure_Coefficient", "Cp", "CpTotal")
            cmap = "RdBu_r"
            clim = [-1.5, 1.0]
            title = "Cp"
            
        if is_checked and scalar:
            self._plotter.clear()
            self._plotter.add_mesh_clip_plane(
                self._volume_mesh,
                normal='y',
                scalars=scalar,
                cmap=cmap,
                clim=clim,
                show_scalar_bar=True,
                scalar_bar_args={"title": title}
            )
            # Re-add STL outline
            if self._current_stl and self._current_stl.is_file():
                try:
                    stl_surf = self._stl_in_cfd_frame()
                    if stl_surf is not None:
                        self._plotter.add_mesh(stl_surf, color="#c8d0dc",
                                               opacity=0.3, style="wireframe")
                except Exception:
                    pass
            self._plotter.add_axes()
        else:
            # Turn off and revert to normal view
            self._plotter.clear_plane_widgets()
            self._refresh_vis()

    def reset_workspace(self):
        """Blank CFD results + plots/3D view (called on New Project)."""
        self._clear_results()          # also drops the cached VTK meshes
        from ui.workspace_reset import clear_visuals
        clear_visuals(self)
        for name, w in list(vars(self).items()):
            if name.startswith("_lbl_") and hasattr(w, "setText"):
                try:
                    w.setText("—")
                except Exception:
                    pass
        for b in ("_btn_export_pdf", "_btn_export_csv", "_btn_export_vtk",
                  "_btn_export_struct", "_btn_inject"):
            w = getattr(self, b, None)
            if w is not None:
                try:
                    w.setEnabled(False)
                except Exception:
                    pass
        try:
            if hasattr(self, "_log_box"):
                self._log_box.clear()
        except Exception:
            pass

    def _log(self, msg: str):
        self._log_box.append(msg)
        logger.info(msg)

    # ── Display control helpers ──────────────────────────────────────────────

    def _get_user_clim(self, auto_lo: float, auto_hi: float):
        """Return [lo, hi] from the scalar range spin boxes, or auto values.
        When auto-scaling, writes the computed values into the spinboxes
        so the user can see the actual range and tweak it."""
        lo = self._sp_smin.value()
        hi = self._sp_smax.value()
        _MIN = self._sp_smin.minimum()  # -1e9
        # Use auto if either is at the minimum (special "auto" value)
        use_auto = (lo <= _MIN + 1.0 or hi <= _MIN + 1.0 or lo >= hi)
        if use_auto:
            # Populate spinboxes with the auto-computed range so
            # the user sees real numbers and can adjust from there
            self._sp_smin.blockSignals(True)
            self._sp_smax.blockSignals(True)
            self._sp_smin.setValue(round(auto_lo, 4))
            self._sp_smax.setValue(round(auto_hi, 4))
            self._sp_smin.blockSignals(False)
            self._sp_smax.blockSignals(False)
            return [auto_lo, auto_hi]
        return [lo, hi]

    def _auto_clim(self, mesh, name, lo_pct=1.0, hi_pct=99.0,
                   symmetric=False, floor=0.0, weighted=True):
        """Colour range for *name* on *mesh*, honouring the user's spin boxes.

        ``symmetric`` mirrors the range about zero, for diverging colormaps.

        ``weighted`` picks how the percentiles are counted, and the right
        answer differs by mesh — pass it deliberately:

        * **Surfaces (True).** Wall mesh density follows CAD geometry, not
          the flow. Cell areas span ~3e5 : 1 with the slivers packed onto the
          nose tip, so counting one sample per point describes the tip rather
          than the airframe: the point-based 95th percentile of Cp on a
          measured M=0.8 case was 0.907 against an area-weighted 0.057.
          Weighting by area is what makes the colour bar describe the model.

        * **Volume slices (False).** Here the grading IS the physics — the
          mesher deliberately packs cells into the boundary layer and wake.
          Weighting by area hands almost all of it to the big undisturbed
          farfield cells and collapses the range onto freestream: the
          measured 1-99% pressure band went to [68193, 72571] Pa on a field
          that really spans [55k, 118k] near the body. Point counting
          inherits the mesher's judgement and is the better estimator.
        """
        lo, hi = field_percentiles(mesh, name, [lo_pct, hi_pct], weighted=weighted)
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, max(floor, 1.0)
        lo, hi = float(lo), float(hi)
        if symmetric:
            half = max(abs(lo), abs(hi), floor)
            lo, hi = -half, half
        elif hi - lo < 1e-12:
            hi = lo + max(floor, 1e-6)
        return self._get_user_clim(lo, hi)

    def _get_user_opacity(self) -> float:
        """Return opacity from the slider (0.1 – 1.0)."""
        return self._sl_opacity.value() / 100.0

    def _get_slice_offset(self) -> float:
        """Return slice Y offset from slider, normalized [-1, 1]."""
        return self._sl_slice.value() / 100.0

    def _get_user_cmap(self, default: str) -> str:
        """Return user-selected colormap or default."""
        sel = self._cb_cmap.currentText()
        return default if sel == "auto" else sel

    def _get_iso_percentile(self) -> float:
        """Return iso threshold percentile from slider (1–100)."""
        return float(self._sl_iso.value())

    def _get_show_edges(self) -> bool:
        """Return whether mesh edges should be shown."""
        return self._chk_mesh_edges.isChecked()

    # ── Probe mode ───────────────────────────────────────────────────────────

    def _toggle_probe_mode(self, checked: bool):
        """Enable/disable point probe picking in the 3D viewport."""
        if checked:
            try:
                self._plotter.enable_point_picking(
                    callback=self._on_probe_pick,
                    show_message=False,
                    show_point=True,
                    point_size=12,
                    color="#fbc02d",
                    use_picker=True,
                    tolerance=0.025,
                )
                self._status_lbl.setText("Probe mode ON — click on the visualization to query values")
            except Exception as e:
                self._status_lbl.setText(f"Probe init error: {e}")
        else:
            try:
                self._plotter.disable_picking()
            except Exception:
                pass
            self._status_lbl.setText("Probe mode OFF")

    def _on_probe_pick(self, point):
        """Handle probe pick — display scalar values at the clicked point."""
        import numpy as np
        if point is None:
            return
        x, y, z = point
        # Try to find scalar values from the currently displayed mesh
        parts = [f"x={x:.4f}  y={y:.4f}  z={z:.4f}"]
        # Sample from volume or surface mesh
        probe_mesh = self._volume_mesh or self._surface_mesh
        if probe_mesh is not None:
            try:
                probe_pt = pv.PolyData([x, y, z])
                sampled = probe_pt.sample(probe_mesh)
                for name in sampled.array_names:
                    val = sampled[name]
                    if val is not None and len(val) > 0:
                        v = val[0]
                        if hasattr(v, '__len__') and len(v) > 1:
                            parts.append(f"{name}=[{', '.join(f'{c:.3f}' for c in v)}]")
                        else:
                            parts.append(f"{name}={float(v):.4f}")
            except Exception:
                pass
        self._status_lbl.setText("Probe: " + "  │  ".join(parts[:6]))

    # ── Screenshot ───────────────────────────────────────────────────────────

    def _screenshot_hq(self):
        """Render a 4K screenshot with maximum quality settings."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "cfd_screenshot.png",
            "PNG Images (*.png);;All Files (*)"
        )
        if not path:
            return
        try:
            self._plotter.screenshot(
                filename=path,
                transparent_background=False,
                window_size=(3840, 2160),
            )
            self._log(f"Screenshot saved: {path} (3840×2160)")
            self._status_lbl.setText(f"Screenshot saved: {path}")
        except Exception as e:
            self._log(f"Screenshot error: {e}")

    # ── PBR surface rendering helper ─────────────────────────────────────────

    def _add_mesh_pbr(self, mesh, scalars=None, cmap="turbo", clim=None,
                      opacity=1.0, show_scalar_bar=True, scalar_bar_args=None,
                      name=None, lit=None):
        """Add a surface mesh, shaded or flat depending on what it is showing.

        ``lit`` defaults to *False* whenever ``scalars`` is set. A scalar
        field view is a measurement: the pixel colour has to be the colour
        the bar says it is, and PBR does not allow that. Metallic/roughness
        shading plus ``enable_lightkit`` and SSAO multiply the mapped colour
        by a lighting term that varies over the body, so the same Cp read
        one colour on the lit top of the airframe and another underneath,
        and near-white values (most of a transonic body) came out grey
        metal. Untextured geometry — the STL preview — still gets the nice
        shading by passing ``lit=True``.
        """
        show_edges = self._get_show_edges()
        edge_kw = dict(show_edges=show_edges, edge_color=theme.PANEL, line_width=0.5) if show_edges else dict(show_edges=False)

        # Safety: validate scalar array size matches mesh topology
        if scalars is not None and scalars in mesh.array_names:
            arr = mesh[scalars]
            n = len(arr.flatten()) if arr.ndim <= 1 else len(arr)
            if n != mesh.n_points and n != mesh.n_cells:
                # Scalar array corrupted — render without scalars to avoid crash
                scalars = None
        if lit is None:
            lit = scalars is None

        common = dict(
            scalars=scalars, cmap=cmap, clim=clim,
            nan_color=theme.PANEL,
            interpolate_before_map=True,
            opacity=opacity,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args=scalar_bar_args or {},
            name=name,
            **edge_kw,
        )
        if not lit:
            # Flat mapping — rendered colour == colorbar colour.
            self._plotter.add_mesh(mesh, lighting=False, smooth_shading=True, **common)
            return
        try:
            self._plotter.add_mesh(
                mesh, smooth_shading=True,
                pbr=True, metallic=0.08, roughness=0.35,
                **common,
            )
        except Exception:
            # Fallback to Phong rendering
            self._plotter.add_mesh(
                mesh, smooth_shading=True,
                specular=0.4, specular_power=30,
                ambient=0.2, diffuse=0.7,
                **common,
            )
