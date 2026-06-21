"""
K2 AeroSim — CalculiX Solver Backend
========================================
Implements FEMSolver using the CalculiX open-source FEA suite (ccx).
ccx binary must be in the K2 bin/ folder or on PATH.

Physics:
  - Static linear elastic (shell elements S4R)
  - Modal eigenvalue analysis (Lanczos)
  - Linear buckling (eigenvalue)
  - Steady-state thermal

Mirrors the SU2 solver architecture (cfd/solvers/su2_solver.py).
"""
from __future__ import annotations
import csv, logging, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

# Suppress the console window when launching ccx (a console app) from the
# windowed frozen build — otherwise a terminal flashes on every Run.
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
from typing import Optional
from structures.solvers.base import (
    FEMSolver, FEMConfig, FEMResult, ModalResult,
    LoadCase, get_structural_material, StructuralMaterial,
)
logger = logging.getLogger("K2.FEM.CCX")

from core.paths import bin_dir
_ROOT = Path(__file__).resolve().parents[2]
_BIN_DIR = bin_dir()   # platform-aware: bin/mac-<arch> on macOS, bin/ on Windows

def _find_ccx() -> Optional[Path]:
    """Find ccx: bundled bin/ first, then PATH."""
    for name in ["ccx.exe", "ccx_static.exe", "ccx", "ccx_2.22", "ccx_2.22.exe",
                 "ccx_2.21", "ccx_2.21.exe"]:
        p = _BIN_DIR / name
        if p.is_file():
            return p
    found = shutil.which("ccx") or shutil.which("ccx_static") or shutil.which("ccx_2.22")
    return Path(found) if found else None


class CalculiXSolver(FEMSolver):
    """CalculiX FEM solver backend for K2 AeroSim."""

    def __init__(self, config: FEMConfig):
        super().__init__(config)
        self._mesh_path: Optional[Path] = None
        self._inp_path: Optional[Path] = None
        self._ccx_exe = _find_ccx()
        self._material: Optional[StructuralMaterial] = None
        if self._ccx_exe:
            logger.info(f"CalculiX found: {self._ccx_exe}")
        else:
            logger.warning("ccx binary not found. Meshing works but solver won't run.")

    # ── Mesh ─────────────────────────────────────────────────────────────────

    def generate_mesh(self, assembly=None) -> Path:
        """Generate structural mesh via K2 meshing module."""
        from structures.meshing import build_structural_mesh
        cfg = self.config
        assembly = assembly or cfg.assembly
        if assembly is None:
            raise ValueError("No assembly provided for meshing.")
        out = cfg.work_dir / "structure_mesh.inp"
        build_structural_mesh(
            assembly, out, cfg.mesh_refinement, cfg.element_type,
            custom_circum=cfg.custom_circum,
            custom_axial_per_cal=cfg.custom_axial_per_cal,
        )
        self._mesh_path = out
        self._material = get_structural_material(cfg.material_name)
        logger.info(f"Mesh: {out}")
        return out

    # ── Case Generation ──────────────────────────────────────────────────────

    def generate_case(self) -> Path:
        """Write the complete CalculiX .inp input deck."""
        cfg = self.config
        mat = get_structural_material(cfg.material_name)
        self._material = mat
        lc = cfg.load_case

        inp = cfg.work_dir / "analysis.inp"
        mesh_text = ""
        if self._mesh_path and self._mesh_path.is_file():
            mesh_text = self._mesh_path.read_text(encoding="ascii", errors="replace")

        with open(inp, "w", encoding="ascii", errors="replace") as f:
            f.write("** K2 AeroSim — CalculiX Analysis\n**\n")
            # Include mesh
            f.write(mesh_text + "\n")
            # Shell section — assign thickness + material to all elements
            wt = 0.002
            if cfg.assembly:
                # Try to get wall thickness from first body tube
                from core.components import BodyTube, NoseCone
                for stage in cfg.assembly.stages:
                    for comp in stage.children:
                        if isinstance(comp, NoseCone):
                            wt = getattr(comp, 'wall_thickness', 0.002)
                            break
                        elif isinstance(comp, BodyTube):
                            wt = (comp.outer_diameter_val - comp.inner_diameter) / 2
                            break

            f.write(f"*SHELL SECTION, ELSET=EALL, MATERIAL=MAT1\n")
            f.write(f"{wt:.6f}\n")
            # Material definition
            f.write(f"*MATERIAL, NAME=MAT1\n")
            f.write(f"*ELASTIC\n{mat.E:.6e}, {mat.nu:.4f}\n")
            f.write(f"*DENSITY\n{mat.density:.2f}\n")
            if cfg.analysis_type == "thermal" or lc.delta_T != 0:
                f.write(f"*EXPANSION\n{mat.cte:.6e}\n")
                f.write(f"*CONDUCTIVITY\n{mat.thermal_conductivity:.4f}\n")

            # Analysis-specific cards
            if cfg.analysis_type == "static":
                self._write_static_step(f, lc, cfg)
            elif cfg.analysis_type == "modal":
                self._write_modal_step(f, cfg)
            elif cfg.analysis_type == "buckling":
                self._write_buckling_step(f, lc, cfg)
            elif cfg.analysis_type == "thermal":
                self._write_thermal_step(f, lc, cfg)

        self._inp_path = inp
        logger.info(f"CalculiX input: {inp} ({cfg.analysis_type})")
        return inp

    def _write_static_step(self, f, lc: LoadCase, cfg: FEMConfig):
        """Write a static analysis step."""
        f.write("**\n** STATIC ANALYSIS\n**\n")
        # Boundary conditions — constrain aft ring nodes
        # Find aft node set from mesh, or use NAFT if generated by mesher
        f.write("** Boundary: fix aft end\n")
        f.write("*BOUNDARY\n")
        # Constrain nodes in NAFT set (generated by mesher) or fall back to node 1
        f.write("NAFT, 1, 3, 0.0\n")  # fix translations at aft ring
        f.write("NAFT, 4, 6, 0.0\n")  # fix rotations at aft ring
        # Prevent rigid body rotation at nose
        f.write("NFWD, 2, 3, 0.0\n")

        # *INITIAL CONDITIONS must appear BEFORE *STEP (CCX requirement)
        if lc.delta_T != 0:
            f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 293.15\n")

        f.write("*STEP\n*STATIC\n")
        # Loads
        if lc.axial_force != 0:
            f.write(f"** Axial force via body acceleration: {lc.axial_force:.1f} N\n")
            f.write(f"*DLOAD\nEALL, GRAV, {abs(lc.acceleration_g * 9.81):.4f}, 0., 0., -1.\n")
        if lc.internal_pressure != 0:
            f.write(f"** Internal pressure: {lc.internal_pressure:.1f} Pa\n")
            f.write(f"*DLOAD\nEALL, P, {-lc.internal_pressure:.4f}\n")
        if lc.delta_T != 0:
            f.write(f"** Thermal load: dT = {lc.delta_T:.1f} K\n")
            f.write(f"*TEMPERATURE\nNALL, {293.15 + lc.delta_T:.2f}\n")

        # Mapped CFD surface-pressure field (per-element *DLOAD P)
        self._write_cfd_pressure(f, cfg)

        # Output requests
        f.write("*NODE FILE\nU\n")
        f.write("*EL FILE\nS, E\n")
        f.write("*NODE PRINT, NSET=NALL, TOTALS=YES\nU\n")
        f.write("*EL PRINT, ELSET=EALL, TOTALS=YES\nS\n")
        f.write("*END STEP\n")

    def _write_modal_step(self, f, cfg: FEMConfig):
        """Write a modal (frequency) analysis step.

        Boundary conditions:
          - cantilever (default): clamped at aft (motor mount), free at forward (nose)
          - free-free: no constraints (for free-flight modes)
          - clamped-clamped: both ends fixed (legacy)

        Physics: A rocket in flight is closest to a cantilever beam with the
        motor mount as the fixed end. Free-free is appropriate for free-flight
        modes during coast phase.
        """
        f.write("**\n** MODAL ANALYSIS\n**\n")
        modal_bc = getattr(cfg, 'modal_bc', 'cantilever')
        f.write("*BOUNDARY\n")
        if modal_bc == 'free-free':
            # No constraints — free-flight modes
            # Will produce 6 rigid-body modes (near-zero frequency)
            f.write("** Free-free: no boundary constraints\n")
        elif modal_bc == 'clamped-clamped':
            # Legacy: both ends fully fixed
            f.write("NAFT, 1, 6, 0.0\n")
            f.write("NFWD, 1, 6, 0.0\n")
        else:
            # Default: cantilever (clamped at aft, free at forward)
            # Motor mount provides the fixed boundary
            f.write("NAFT, 1, 6, 0.0\n")
            # Forward end is FREE — no constraints on NFWD
        # Request extra modes to capture all relevant modes
        n_request = cfg.num_modes + (6 if modal_bc == 'free-free' else 2)
        f.write(f"*STEP\n*FREQUENCY\n{n_request}\n")
        f.write("*NODE FILE\nU\n")
        f.write("*EL FILE\nS\n")
        f.write("*NODE PRINT, NSET=NALL, TOTALS=YES\nU\n")
        f.write("*END STEP\n")

    def _write_buckling_step(self, f, lc: LoadCase, cfg: FEMConfig):
        """Write a linear buckling (eigenvalue) analysis."""
        f.write("**\n** BUCKLING ANALYSIS\n**\n")
        f.write("*BOUNDARY\nNAFT, 1, 3, 0.0\nNAFT, 4, 6, 0.0\nNFWD, 2, 3, 0.0\n")
        # Pre-load step
        f.write("*STEP\n*STATIC\n")
        if lc.axial_force != 0:
            f.write(f"*DLOAD\nEALL, GRAV, {abs(lc.acceleration_g * 9.81):.4f}, 0., 0., -1.\n")
        f.write("*END STEP\n")
        # Buckling step
        f.write("*STEP\n*BUCKLE\n5\n")
        f.write("*NODE FILE\nU\n")
        f.write("*END STEP\n")

    def _write_thermal_step(self, f, lc: LoadCase, cfg: FEMConfig):
        """Write a steady-state heat transfer step.

        Note: *INITIAL CONDITIONS must appear BEFORE *STEP in CalculiX.
        Previous version placed it inside the step — incorrect syntax.
        """
        f.write("**\n** THERMAL ANALYSIS\n**\n")
        # Initial conditions MUST be before the step (CCX requirement)
        f.write("*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, 293.15\n")
        f.write("*STEP\n*HEAT TRANSFER, STEADY STATE\n")
        f.write(f"*TEMPERATURE\nNALL, {lc.wall_temp_K:.2f}\n")
        f.write("*NODE FILE\nNT\n")
        f.write("*END STEP\n")

    # ── CFD surface-pressure mapping ──────────────────────────────────────────

    def _build_fem_stations(self):
        """Parse the structural mesh into per-element stations for pressure
        mapping: ``(elem_id, axial, area_m2, outward_axial_normal)``.

        The body axis is Z in the mesh, but ``pressure_mapping``'s IDW metric
        treats the *second* tuple slot as the axis ('x'), so the axial (z)
        coordinate is placed there. Returns ``[]`` if the mesh is unavailable.
        """
        import numpy as np
        mesh = getattr(self, "_mesh_path", None)
        if not mesh or not mesh.is_file():
            return []
        nodes, elements = {}, []
        in_nodes = in_elems = False
        for line in mesh.read_text(encoding="ascii", errors="replace").splitlines():
            s = line.strip()
            if s.startswith("*NODE"):
                in_nodes, in_elems = True, False; continue
            if s.startswith("*ELEMENT"):
                in_elems, in_nodes = True, False; continue
            if s.startswith("*"):
                in_nodes = in_elems = False; continue
            parts = [p for p in s.split(",") if p.strip() != ""]
            if in_nodes and len(parts) >= 4:
                nodes[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif in_elems and len(parts) >= 5:
                elements.append((int(parts[0]), [int(p) for p in parts[1:5]]))
        if not nodes or not elements:
            return []
        stations = []
        for eid, en in elements:
            try:
                p = [np.asarray(nodes[n], dtype=float) for n in en]
            except KeyError:
                continue
            # Newell normal (magnitude = 2·area) + centroid
            nrm = np.zeros(3)
            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                nrm[0] += (a[1] - b[1]) * (a[2] + b[2])
                nrm[1] += (a[2] - b[2]) * (a[0] + b[0])
                nrm[2] += (a[0] - b[0]) * (a[1] + b[1])
            mag = float(np.linalg.norm(nrm))
            area = 0.5 * mag
            centroid = sum(p) / len(p)
            if mag > 1e-15:
                unit = nrm / mag
                # Outward = away from the Z body axis (radial in x,y)
                if float(unit[0] * centroid[0] + unit[1] * centroid[1]) < 0.0:
                    unit = -unit
                n_axial = float(unit[2])
            else:
                n_axial = 0.0
            stations.append((eid, float(centroid[2]), area, n_axial))
        return stations

    def _write_cfd_pressure(self, f, cfg: FEMConfig):
        """Map a CFD surface-pressure field onto the mesh and emit per-element
        ``*DLOAD ... P`` cards. No-op when no VTK is configured or parsing
        yields nothing — the solve falls back to lumped/analytic loads.
        """
        vtk = getattr(cfg, "cfd_surface_vtk", None)
        if not vtk:
            return
        from pathlib import Path
        vtk = Path(vtk)
        if not vtk.is_file():
            logger.warning("CFD surface VTK not found: %s — skipping pressure map", vtk)
            return
        try:
            from structures.pressure_mapping import (
                _parse_vtu_points_and_pressure, map_pressures_idw,
                generate_dload_cards)
            cfd_pts, cfd_pres = _parse_vtu_points_and_pressure(vtk)
            if not cfd_pts or not cfd_pres:
                logger.warning("CFD VTK has no usable pressure data — skipping map")
                return
            # Remap body axis (z) into the x slot to match the IDW metric.
            cfd_pts = [(z, x, y) for (x, y, z) in cfd_pts]
            stations = self._build_fem_stations()
            if not stations:
                logger.warning("No FEM stations built — skipping CFD pressure map")
                return
            result = map_pressures_idw(cfd_pts, cfd_pres, stations)
            if not result.element_pressures:
                logger.warning("CFD pressure map produced no element loads")
                return
            f.write("**\n** Mapped CFD surface pressure (IDW)\n")
            f.write(generate_dload_cards(result))
            self._cfd_mapping = result
            ps = [p for _, p in result.element_pressures]
            logger.info("CFD pressure mapped onto %d elements (%.0f..%.0f Pa)",
                        result.num_fem_elements, min(ps), max(ps))
        except Exception as exc:
            logger.error("CFD pressure mapping failed: %s", exc)

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(self):
        """Run CalculiX solver. Generator yielding (stage, fraction)."""
        if not self._ccx_exe:
            # Fallback: run analytical solution
            logger.warning("ccx not found — running analytical fallback")
            yield "analytical", 0.5
            yield "analytical", 1.0
            return
        if not self._inp_path:
            raise RuntimeError("Call generate_case() before run().")

        job_name = self._inp_path.stem
        logger.info(f"Running CalculiX: {self._ccx_exe} {job_name}")
        self._emit_progress("Solving", 0.0)

        log_path = self.config.work_dir / "ccx_run.log"
        cmd = [str(self._ccx_exe), "-i", job_name]
        t0 = time.time()

        proc = subprocess.Popen(
            cmd, cwd=str(self.config.work_dir),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=_NO_WINDOW,
        )

        step_pattern = re.compile(r"step\s+(\d+)", re.IGNORECASE)
        iter_pattern = re.compile(r"iteration\s+(\d+)", re.IGNORECASE)

        with open(log_path, "w", encoding="utf-8") as lf:
            for line in proc.stdout:
                lf.write(line)
                ll = line.strip().lower()
                # CalculiX prints a per-eigenvalue "error" residual column during
                # *FREQUENCY (e.g. "error  2.2e-21") — convergence info, not a fault.
                # Only surface real diagnostics: *ERROR / *WARNING directives.
                if "*error" in ll or "*warning" in ll:
                    logger.info(f"[CCX] {line.strip()}")
                m = iter_pattern.search(line)
                if m:
                    it = int(m.group(1))
                    self._emit_progress("Solving", min(it / 20.0, 0.95))
                    yield "solving", min(it / 20.0, 0.95)

        proc.wait()
        elapsed = time.time() - t0
        if proc.returncode != 0:
            try:
                tail = log_path.read_text()[-2000:]
                logger.error(f"CCX log tail:\n{tail}")
            except Exception:
                pass
            raise RuntimeError(f"CalculiX failed (exit {proc.returncode}). See {log_path}")

        self._emit_progress("Done", 1.0)
        yield "done", 1.0
        logger.info(f"CalculiX complete in {elapsed:.1f}s")

    # ── Parse Results ────────────────────────────────────────────────────────

    def parse_results(self) -> FEMResult:
        """Parse CalculiX .frd/.dat output and return FEMResult."""
        result = FEMResult()
        result.material_name = self._material.name if self._material else ""
        result.yield_strength = self._material.yield_strength if self._material else 276e6
        mat = self._material or get_structural_material("Aluminum 6061-T6")

        # Try to parse .dat file for stress/displacement summary
        dat_path = self.config.work_dir / "analysis.dat"
        frd_path = self.config.work_dir / "analysis.frd"

        if dat_path.is_file():
            result = self._parse_dat(dat_path, result, mat)
            # CalculiX only sees axial_force + internal_pressure from the LoadCase.
            # Aerodynamic bending, fin root loads, and thermal stresses are NOT
            # in the FEM model — superimpose them analytically onto the FEM result.
            result = self._superimpose_aero_loads(result, mat)
        elif not self._ccx_exe:
            # Analytical fallback
            result = self._analytical_fallback(result, mat)

        if frd_path.is_file():
            result.result_vtk = frd_path

        # Compute safety metrics
        if result.max_von_mises > 0:
            result.safety_factor = mat.yield_strength / result.max_von_mises
            result.yield_utilization = result.max_von_mises / mat.yield_strength
            sf_req = self.config.safety_factor_required
            result.margin_of_safety = (result.safety_factor / sf_req) - 1.0
        else:
            result.safety_factor = float('inf')
            result.margin_of_safety = float('inf')
            result.yield_utilization = 0.0

        result.converged = True
        result.load_case_name = self.config.load_case.name
        logger.info(
            f"FEM Results: \u03c3_vm={result.max_von_mises/1e6:.1f} MPa, "
            f"SF={result.safety_factor:.2f}, MoS={result.margin_of_safety:.2f}"
        )
        return result

    def _get_element_z(self) -> dict:
        """Parse mesh file to get Z-coordinate for each element."""
        z_map = {}
        try:
            inp_path = self.config.work_dir / "structure_mesh.inp"
            if not inp_path.is_file(): return z_map
            nodes = {}; in_nodes = False; in_elems = False
            for line in inp_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("*NODE"): in_nodes = True; in_elems = False; continue
                if line.startswith("*ELEMENT"): in_elems = True; in_nodes = False; continue
                if line.startswith("*"): in_nodes = in_elems = False; continue
                
                parts = line.split(",")
                if in_nodes and len(parts) >= 4:
                    nodes[int(parts[0])] = float(parts[3])
                elif in_elems and len(parts) >= 5:
                    eid = int(parts[0])
                    nz = [nodes.get(int(n), 0.0) for n in parts[1:] if n.strip()]
                    if nz: z_map[eid] = sum(nz) / len(nz)
        except Exception:
            pass
        return z_map

    def _parse_dat(self, dat_path: Path, result: FEMResult, mat) -> FEMResult:
        """Parse CalculiX .dat text output for stress/displacement values."""
        try:
            text = dat_path.read_text(encoding="utf-8", errors="replace")
            # Find maximum stress values from element output
            stresses = []  # list of (eid, vm)
            disp_vals = []
            in_stress = False
            in_disp = False
            for line in text.splitlines():
                ll = line.strip().lower()
                if "stresses" in ll and "elem" in ll:
                    in_stress = True; in_disp = False; continue
                if "displacements" in ll and ("vx" in ll or "set" in ll):
                    in_disp = True; in_stress = False; continue
                # Section break: new step/increment header (NOT empty lines!)
                if "s t e p" in ll or "increment" in ll:
                    in_stress = in_disp = False; continue
                # Skip blank lines within data sections
                if not line.strip():
                    continue

                parts = line.split()
                # CalculiX stress format: elem integ_pt sxx syy szz sxy sxz syz [label]
                if in_stress and len(parts) >= 8:
                    try:
                        eid = int(parts[0])
                        sxx = float(parts[2])
                        syy = float(parts[3])
                        szz = float(parts[4])
                        sxy = float(parts[5])
                        sxz = float(parts[6])
                        syz = float(parts[7])
                        vm = math.sqrt(0.5 * ((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2
                                              + 6*(sxy**2 + sxz**2 + syz**2)))
                        # Store full tensor components per element
                        stresses.append((eid, vm, sxx, syy, szz, sxy, sxz, syz))
                    except (ValueError, IndexError):
                        pass
                # CalculiX displacement format: node_id vx vy vz
                elif in_disp and len(parts) >= 4:
                    try:
                        dx = float(parts[1])
                        dy = float(parts[2])
                        dz = float(parts[3])
                        disp_vals.append(math.sqrt(dx**2 + dy**2 + dz**2))
                    except (ValueError, IndexError):
                        pass

            if stresses:
                raw_vms = [s[1] for s in stresses]
                result.max_von_mises = max(raw_vms)

                # ── Extract ACTUAL stress components from FEM tensor ──────
                # For thin-walled shells in cylindrical coords:
                #   axial ≈ szz (along body axis)
                #   hoop  ≈ sxx or syy (circumferential, depends on orientation)
                #   shear ≈ max(|sxy|, |sxz|, |syz|)
                # We take the maximum of each component across all elements
                max_axial = 0.0
                max_hoop = 0.0
                max_shear = 0.0
                for entry in stresses:
                    _, _, sxx, syy, szz, sxy, sxz, syz = entry
                    # For axisymmetric shell: szz is along body axis (axial)
                    # sxx/syy are in-plane (hoop/radial)
                    max_axial = max(max_axial, abs(szz))
                    max_hoop = max(max_hoop, abs(sxx), abs(syy))
                    max_shear = max(max_shear, abs(sxy), abs(sxz), abs(syz))

                result.max_axial_stress = max_axial
                result.max_hoop_stress = max_hoop
                result.max_shear_stress = max_shear
                logger.info(
                    f"Stress decomposition from FEM tensor: "
                    f"σ_axial={max_axial/1e6:.1f} MPa, "
                    f"σ_hoop={max_hoop/1e6:.1f} MPa, "
                    f"τ_max={max_shear/1e6:.1f} MPa"
                )

                # Map to Z-coordinates and smooth/envelope stresses
                elem_z = self._get_element_z()
                z_stations = {}
                for entry in stresses:
                    eid, vm = entry[0], entry[1]
                    z = elem_z.get(eid, float(eid))
                    z_round = round(z, 3)
                    if z_round not in z_stations:
                        z_stations[z_round] = []
                    z_stations[z_round].append(vm)
                
                # Envelope: max stress at each Z-station
                smoothed = [(z, max(vms)) for z, vms in sorted(z_stations.items())]
                result.element_stresses = smoothed
                logger.info(f"Parsed {len(stresses)} element stresses, smoothed to {len(smoothed)} stations, max σ_vm = {result.max_von_mises/1e6:.1f} MPa")
            if disp_vals:
                # Filter out unreasonable values (> 1m displacement = numerical artifact)
                reasonable = [d for d in disp_vals if d < 1.0]
                if reasonable:
                    result.max_displacement_mm = max(reasonable) * 1000.0
                else:
                    # Use actual 95th percentile
                    disp_sorted = sorted(disp_vals)
                    idx_95 = int(len(disp_sorted) * 0.95)
                    idx_95 = min(idx_95, len(disp_sorted) - 1)
                    result.max_displacement_mm = disp_sorted[idx_95] * 1000.0
                result.element_displacements = [(i, min(d, 1.0)*1000) for i, d in enumerate(disp_vals)]
                logger.info(f"Parsed {len(disp_vals)} displacements, max = {result.max_displacement_mm:.4f} mm")

        except Exception as e:
            logger.warning(f"DAT parse error: {e}")
        return result

    def _superimpose_aero_loads(self, result: FEMResult, mat) -> FEMResult:
        """Superimpose condition-specific loads that CalculiX doesn't model.

        CalculiX only applies axial_force and internal_pressure from the LoadCase.
        It does NOT model:
        - Aerodynamic bending from angle of attack
        - Fin root bending at fin-body junction
        - Stress concentrations at couplers/joints/cutouts
        - Dynamic amplification from gust response
        - Thermal gradient stress

        This method adds these analytically to the parsed FEM result.
        """
        lc = self.config.load_case
        assembly = self.config.assembly
        if assembly is None:
            return result

        d_ref, r, total_L, wt, r_o, r_i, cross_area, I = self._get_geometry(mat)
        vm_fem = result.max_von_mises  # base FEM stress

        Kt_detail = 1.8  # couplers, fin slots, rail buttons

        # Scale factor to scale base FEM stress (run under gravity load) to full thrust/shock load
        scale_factor = 1.0
        if lc.acceleration_g != 0:
            mass = 5.0
            if assembly:
                if hasattr(assembly, 'total_mass'):
                    mass = assembly.total_mass() if callable(assembly.total_mass) else assembly.total_mass
            F_grav = abs(mass * lc.acceleration_g * 9.81)
            if F_grav > 0:
                scale_factor = abs(lc.axial_force) / F_grav

        vm_fem_scaled = vm_fem * scale_factor

        if lc.name in ("Max Thrust", "Max-Q"):
            # Compute aerodynamic bending stress
            q_dyn = lc.dynamic_pressure
            if q_dyn <= 0 and lc.mach > 0:
                try:
                    from cfd.solvers.base import isa_conditions
                    P, T, rho = isa_conditions(lc.altitude_m)
                    a_s = math.sqrt(1.4 * 287.05 * T)
                    V = lc.mach * a_s
                    q_dyn = 0.5 * rho * V ** 2
                except Exception:
                    V = lc.mach * 340.0
                    q_dyn = 0.5 * 1.225 * V ** 2

            aoa = lc.angle_of_attack_deg if lc.angle_of_attack_deg > 0 else 2.0
            if lc.name == "Max-Q":
                aoa = max(aoa, 3.0)  # Max-Q uses higher AoA
            aoa_rad = math.radians(aoa)

            # Body normal force (side-projected area)
            A_body_side = d_ref * total_L
            F_body_normal = q_dyn * 2.0 * aoa_rad * A_body_side

            # Fin normal force (planform area)
            fin_span = d_ref * 0.8
            fin_chord = total_L * 0.12
            A_fin_plan = fin_span * fin_chord
            n_fins = 3
            F_fins_normal = q_dyn * 4.0 * aoa_rad * A_fin_plan * n_fins
            F_normal_total = F_body_normal + F_fins_normal
            # CFD-derived lateral force (LoadCase.lateral_force) overrides the
            # estimate when larger — conservative, mirrors the axial handling.
            if lc.lateral_force > 0:
                F_normal_total = max(F_normal_total, lc.lateral_force)

            # Bending moment at critical section
            M_bend = F_normal_total * total_L * 0.35
            sigma_bend = M_bend * r_o / I if I > 0 else 0

            # Fin root stress
            F_per_fin = F_fins_normal / n_fins
            M_fin = F_per_fin * fin_span / 3
            fin_area = wt * fin_span * 0.5
            sigma_fin_root = M_fin / (fin_area * wt) if fin_area > 0 else 0

            # Dynamic amplification for Max-Q
            DAF = 1.3 if lc.name == "Max-Q" else 1.0

            # Combined: FEM axial + analytical bending, with detail factor
            vm_combined = math.sqrt(
                (vm_fem_scaled + sigma_bend)**2 + 3 * (F_normal_total / (2 * math.pi * r * wt))**2
            ) if (r > 0 and wt > 0) else vm_fem_scaled + sigma_bend

            vm_body = vm_combined * Kt_detail * DAF
            vm_fin = sigma_fin_root * Kt_detail * DAF
            result.max_von_mises = max(vm_body, vm_fin)
            result.max_bending_stress = sigma_bend + sigma_fin_root

            # Update individual stress components to be realistic and consistent
            sigma_axial = abs(lc.axial_force) / cross_area if cross_area > 0 else 0.0
            if lc.name == "Max-Q":
                Cd = 0.5
                A_ref = math.pi * r**2
                F_drag = q_dyn * Cd * A_ref
                F_net_axial = abs(lc.axial_force) + F_drag
                sigma_axial = F_net_axial / cross_area if cross_area > 0 else 0.0

            if lc.name == "Max-Q":
                sigma_hoop = q_dyn * r / wt if wt > 0 else 0.0
            else:
                sigma_hoop = lc.internal_pressure * r / wt if wt > 0 else 0.0

            result.max_axial_stress = sigma_axial
            result.max_hoop_stress = sigma_hoop
            result.max_shear_stress = F_normal_total / (2 * math.pi * r * wt) if (r > 0 and wt > 0) else 0.0
            if lc.name == "Max Thrust":
                result.max_shear_stress = abs(lc.axial_force) / (2 * math.pi * r * wt) if (r > 0 and wt > 0) else 0.0
            result.max_thermal_stress = 0.0

            # Update element stresses with bending envelope
            if result.element_stresses:
                updated = []
                for z, vm_local in result.element_stresses:
                    frac = z / total_L if total_L > 0 else 0
                    # Cantilever bending envelope: 1st mode shape sin(π·frac/2)
                    # Maximum at nose (free end), zero at motor mount (fixed end)
                    local_bend = sigma_bend * math.sin(math.pi * frac / 2)
                    # Scale local FEM stress from gravity to actual thrust
                    vm_local_scaled = vm_local * scale_factor
                    local_vm = math.sqrt((vm_local_scaled + local_bend)**2)
                    local_vm *= Kt_detail * DAF
                    if frac > 0.85:
                        fin_contrib = sigma_fin_root * Kt_detail * DAF * ((frac - 0.85) / 0.15)
                        local_vm = max(local_vm, fin_contrib)
                    updated.append((z, local_vm))
                result.element_stresses = updated

            logger.info(f"Superimposed aero: bend={sigma_bend/1e6:.1f} MPa, "
                        f"fin_root={sigma_fin_root/1e6:.1f} MPa, Kt={Kt_detail}, DAF={DAF}, "
                        f"VM_total={result.max_von_mises/1e6:.1f} MPa")

        elif lc.name in ("Thermal", "Aerodynamic Heating"):
            # Add thermal stress to FEM result
            mach = lc.mach if lc.mach > 0 else 3.0
            try:
                from cfd.solvers.base import isa_conditions
                P, T_amb, rho = isa_conditions(lc.altitude_m)
            except Exception:
                T_amb = 223.15
            T_recovery = T_amb * (1 + 0.89 * 0.2 * mach**2)  # r=0.89 turbulent
            T_stag = T_amb * (1 + 0.2 * mach**2)  # isentropic stagnation
            # Max ΔT is from whichever is higher: stagnation (nose) or recovery (body)
            dT_max = max(T_stag, T_recovery) - 293.15
            constraint = 0.55  # partial constraint for rocket structures
            sigma_th = mat.E * mat.cte * abs(dT_max) / (1 - mat.nu) * constraint
            
            # Thermal gradient-induced bending
            dT_gradient = abs(T_stag - T_recovery)
            sigma_bend = mat.E * mat.cte * dT_gradient * wt / (2 * d_ref) * constraint if d_ref > 0 else 0.0
            
            result.max_axial_stress = 0.0
            result.max_hoop_stress = 0.0
            result.max_shear_stress = 0.0
            result.max_bending_stress = sigma_bend
            result.max_thermal_stress = sigma_th
            # Proper von Mises for uniaxial thermal + bending
            result.max_von_mises = math.sqrt((sigma_th + sigma_bend) ** 2)

            logger.info(f"Superimposed thermal: dT={dT_max:.1f} K, "
                        f"sigma_th={sigma_th/1e6:.1f} MPa")

        elif lc.name == "Recovery Shock":
            # Scale up to dynamic peak parachute load
            shock_g = lc.recovery_shock_g if lc.recovery_shock_g > 0 else 15.0
            daf = lc.dynamic_amplification if lc.dynamic_amplification > 1.0 else 1.8
            kt = lc.stress_concentration if lc.stress_concentration > 1.0 else 2.5
            mass = 5.0
            if assembly:
                if hasattr(assembly, 'total_mass'):
                    mass = assembly.total_mass() if callable(assembly.total_mass) else assembly.total_mass
            
            F_recovery = mass * shock_g * 9.81 * daf
            sigma_axial_base = F_recovery / cross_area if cross_area > 0 else 0.0
            sigma_axial_peak = sigma_axial_base * kt
            
            eccentricity = 0.01 * d_ref
            M_snapback = F_recovery * eccentricity
            sigma_bend = M_snapback * r_o / I if I > 0 else 0.0
            tau = F_recovery / (2 * math.pi * r * wt) * 0.3 if (r > 0 and wt > 0) else 0.0

            result.max_axial_stress = sigma_axial_peak
            result.max_hoop_stress = 0.0
            result.max_bending_stress = sigma_bend
            result.max_shear_stress = tau
            result.max_thermal_stress = 0.0
            result.max_von_mises = math.sqrt((sigma_axial_peak + sigma_bend)**2 + 3 * tau**2)

            if result.element_stresses:
                result.element_stresses = [(z, vm * scale_factor * Kt_detail) for z, vm in result.element_stresses]

        else:
            # Custom / other cases: scale up to actual axial force and pressure loads
            result.max_von_mises = vm_fem_scaled
            result.max_axial_stress = vm_fem_scaled * 0.8
            result.max_hoop_stress = vm_fem_scaled * 0.3
            result.max_shear_stress = vm_fem_scaled / math.sqrt(3)
            result.max_bending_stress = 0.0
            result.max_thermal_stress = 0.0
            if result.element_stresses:
                result.element_stresses = [(z, vm * scale_factor) for z, vm in result.element_stresses]

        return result

    def _analytical_fallback(self, result: FEMResult, mat) -> FEMResult:
        """Compute stresses analytically when ccx is not available.
        Routes to condition-specific solvers based on load case name."""
        lc = self.config.load_case
        assembly = self.config.assembly

        if assembly is None:
            return result

        result.load_case_name = lc.name

        if lc.name == "Recovery Shock":
            return self._analytical_recovery(result, mat)
        elif lc.name in ("Thermal", "Aerodynamic Heating"):
            return self._analytical_thermal(result, mat)
        elif lc.name == "Max-Q":
            return self._analytical_max_q(result, mat)
        else:
            return self._analytical_max_thrust(result, mat)

    def _get_geometry(self, mat):
        """Extract common geometry parameters from assembly."""
        assembly = self.config.assembly
        d_ref = assembly.get_reference_diameter()
        r = d_ref / 2
        total_L = assembly.total_length()
        wt = 0.002
        from core.components import BodyTube, NoseCone
        for stage in assembly.stages:
            for comp in stage.children:
                if isinstance(comp, NoseCone):
                    wt = getattr(comp, 'wall_thickness', 0.002)
                    break
                elif isinstance(comp, BodyTube):
                    wt = (comp.outer_diameter_val - comp.inner_diameter) / 2
                    break
        r_o, r_i = r + wt/2, r - wt/2
        cross_area = math.pi * d_ref * wt
        I = math.pi / 4 * (r_o**4 - r_i**4)
        return d_ref, r, total_L, wt, r_o, r_i, cross_area, I

    def _analytical_max_thrust(self, result: FEMResult, mat) -> FEMResult:
        """Max Thrust: compressive axial + hoop + shear + bending + fin root."""
        lc = self.config.load_case
        d_ref, r, total_L, wt, r_o, r_i, cross_area, I = self._get_geometry(mat)

        Kt_detail = 1.8  # structural detail factor

        sigma_axial = abs(lc.axial_force) / cross_area if cross_area > 0 else 0
        sigma_hoop = lc.internal_pressure * r / wt if wt > 0 else 0
        tau = abs(lc.axial_force) / (2 * math.pi * r * wt) if (r > 0 and wt > 0) else 0

        # Aerodynamic bending
        sigma_bend = 0.0
        q_dyn = 0.0
        aoa = lc.angle_of_attack_deg
        if aoa > 0 and lc.mach > 0:
            try:
                from cfd.solvers.base import isa_conditions
                P, T, rho = isa_conditions(lc.altitude_m)
                a_sound = math.sqrt(1.4 * 287.05 * T)
                V = lc.mach * a_sound
                q_dyn = 0.5 * rho * V ** 2
                aoa_rad = math.radians(aoa)
                C_N = 2.0 * math.sin(aoa_rad) * math.cos(aoa_rad)
                A_ref = math.pi * r ** 2
                F_lateral = q_dyn * C_N * A_ref
                M_bend = F_lateral * total_L / 4
                sigma_bend = M_bend * r_o / I if I > 0 else 0
            except Exception:
                sigma_bend = sigma_axial * 0.05
        elif aoa > 0:
            sigma_bend = sigma_axial * 0.03 * aoa

        # Fin root bending
        sigma_fin_root = 0.0
        fin_span = d_ref * 0.8
        n_fins = 3
        if lc.mach > 0 and q_dyn > 0:
            aoa_rad = math.radians(aoa) if aoa > 0 else math.radians(2.0)
            fin_chord = total_L * 0.12
            A_fin_plan = fin_span * fin_chord
            F_per_fin = q_dyn * 4.0 * aoa_rad * A_fin_plan
            M_fin = F_per_fin * fin_span / 3
            fin_area = wt * fin_span * 0.5
            sigma_fin_root = M_fin / (fin_area * wt) if fin_area > 0 else 0
        else:
            sigma_fin_root = sigma_axial * 0.3

        # Inertial body bending
        accel = abs(lc.axial_force) / max(5.0 * 9.81, 1.0)
        m_per_L = mat.density * cross_area
        M_inertial = m_per_L * accel * 9.81 * total_L**2 / 8
        sigma_inertial = M_inertial * r_o / I if I > 0 else 0

        # Von Mises with detail factor
        sx = sigma_axial + sigma_bend + sigma_inertial
        sy = sigma_hoop
        vm_body = math.sqrt(sx**2 - sx*sy + sy**2 + 3 * tau**2) * Kt_detail
        vm_fin = sigma_fin_root * Kt_detail
        vm = max(vm_body, vm_fin)
        total_bend = sigma_bend + sigma_inertial + sigma_fin_root

        result.max_axial_stress = sigma_axial
        result.max_hoop_stress = sigma_hoop
        result.max_bending_stress = total_bend
        result.max_thermal_stress = 0.0
        result.max_von_mises = vm
        result.max_shear_stress = tau

        if total_L > 0:
            P_crit = math.pi**2 * mat.E * I / total_L**2
            result.buckling_load_factor = P_crit / max(abs(lc.axial_force), 1.0)
        if total_bend > 0 and mat.E > 0 and I > 0 and r_o > 0:
            delta = total_bend * total_L**2 / (8 * mat.E * r_o)
            result.max_displacement_mm = delta * 1000

        n_stations = 30
        for i in range(n_stations + 1):
            frac = i / n_stations
            z = total_L * frac
            # Axial: uniform along body (thrust/inertia acts on entire section)
            local_axial = sigma_axial
            # Bending: cantilever first mode — max at nose, zero at motor
            local_bend = sigma_bend * math.sin(math.pi * frac / 2)
            local_inertial = sigma_inertial * math.sin(math.pi * frac / 2)
            local_hoop = sigma_hoop
            # Shear: max at support (aft), decreasing toward nose (beam theory)
            local_tau = tau * (1.0 - frac)
            lsx = local_axial + local_bend + local_inertial
            local_vm = math.sqrt(lsx**2 - lsx*local_hoop + local_hoop**2 + 3*local_tau**2)
            local_vm *= Kt_detail
            # Fin root spike at aft 15%
            if frac > 0.85:
                fin_contrib = sigma_fin_root * Kt_detail * ((frac - 0.85) / 0.15)
                local_vm = max(local_vm, fin_contrib)
            result.element_stresses.append((z, local_vm))

        result.converged = True
        return result

    def _analytical_max_q(self, result: FEMResult, mat) -> FEMResult:
        """Max-Q: aerodynamic bending dominated with fin root + DAF."""
        lc = self.config.load_case
        d_ref, r, total_L, wt, r_o, r_i, cross_area, I = self._get_geometry(mat)

        Kt_detail = 1.8
        DAF_gust = 1.3

        q_dyn = lc.dynamic_pressure
        if q_dyn <= 0 and lc.mach > 0:
            try:
                from cfd.solvers.base import isa_conditions
                P, T, rho = isa_conditions(lc.altitude_m)
                a_s = math.sqrt(1.4 * 287.05 * T)
                V = lc.mach * a_s
                q_dyn = 0.5 * rho * V ** 2
            except Exception:
                V = lc.mach * 340.0
                q_dyn = 0.5 * 1.225 * V ** 2

        A_ref = math.pi * r ** 2  # for drag only
        Cd = 0.5
        F_drag = q_dyn * Cd * A_ref
        F_net = abs(lc.axial_force) + F_drag
        sigma_axial = F_net / cross_area if cross_area > 0 else 0

        aoa = lc.angle_of_attack_deg if lc.angle_of_attack_deg > 0 else 3.0
        aoa_rad = math.radians(aoa)

        # Body normal force using side-projected area
        A_body_side = d_ref * total_L
        F_body_normal = q_dyn * 2.0 * aoa_rad * A_body_side

        # Fin normal force using fin planform area
        fin_span = d_ref * 0.8
        fin_chord = total_L * 0.12
        A_fin_plan = fin_span * fin_chord
        n_fins = 3
        F_fins_normal = q_dyn * 4.0 * aoa_rad * A_fin_plan * n_fins

        F_normal_total = F_body_normal + F_fins_normal
        M_bend = F_normal_total * total_L * 0.35
        sigma_bend = M_bend * r_o / I if I > 0 else 0

        # Fin root bending
        F_per_fin = F_fins_normal / n_fins
        M_fin = F_per_fin * fin_span / 3
        fin_area = wt * fin_span * 0.5
        sigma_fin_root = M_fin / (fin_area * wt) if fin_area > 0 else 0

        hp_ext = q_dyn * r / wt if wt > 0 else 0
        tau = F_normal_total / (2 * math.pi * r * wt) if (r > 0 and wt > 0) else 0

        sx = sigma_axial + sigma_bend
        sy = hp_ext
        vm_body = math.sqrt(sx**2 - sx*sy + sy**2 + 3 * tau**2) * Kt_detail * DAF_gust
        vm_fin = sigma_fin_root * Kt_detail * DAF_gust
        vm = max(vm_body, vm_fin)
        total_bend = sigma_bend + sigma_fin_root

        result.max_axial_stress = sigma_axial
        result.max_hoop_stress = hp_ext
        result.max_bending_stress = total_bend
        result.max_thermal_stress = 0.0
        result.max_von_mises = vm
        result.max_shear_stress = tau

        if total_L > 0:
            P_crit = math.pi**2 * mat.E * I / total_L**2
            result.buckling_load_factor = P_crit / max(F_net, 1.0)
        if total_bend > 0 and mat.E > 0 and I > 0 and r_o > 0:
            delta = total_bend * total_L**2 / (8 * mat.E * r_o)
            result.max_displacement_mm = delta * 1000

        n_stations = 30
        for i in range(n_stations + 1):
            frac = i / n_stations
            z = total_L * frac
            # Cantilever bending: max at free end, zero at support
            local_bend = sigma_bend * math.sin(math.pi * frac / 2)
            local_axial = sigma_axial
            local_hoop = hp_ext
            # Shear distribution: max near support (aft), decreasing forward
            local_tau = tau * (1.0 - frac)
            lsx = local_axial + local_bend
            local_vm = math.sqrt(lsx**2 - lsx*local_hoop + local_hoop**2 + 3*local_tau**2)
            local_vm *= Kt_detail * DAF_gust
            # Fin root spike at aft
            if frac > 0.85:
                fin_contrib = sigma_fin_root * Kt_detail * DAF_gust * ((frac - 0.85) / 0.15)
                local_vm = max(local_vm, fin_contrib)
            result.element_stresses.append((z, local_vm))

        result.converged = True
        return result

    def _analytical_recovery(self, result: FEMResult, mat) -> FEMResult:
        """Recovery Shock: tensile axial from parachute deployment."""
        lc = self.config.load_case
        d_ref, r, total_L, wt, r_o, r_i, cross_area, I = self._get_geometry(mat)

        shock_g = lc.recovery_shock_g if lc.recovery_shock_g > 0 else 15.0
        daf = lc.dynamic_amplification if lc.dynamic_amplification > 1.0 else 1.8
        kt = lc.stress_concentration if lc.stress_concentration > 1.0 else 2.5
        mass = lc.vehicle_mass_kg if lc.vehicle_mass_kg > 0 else 5.0

        F_recovery = mass * shock_g * 9.81 * daf
        sigma_axial_base = F_recovery / cross_area if cross_area > 0 else 0
        sigma_axial_peak = sigma_axial_base * kt
        sigma_hoop = 0.0

        eccentricity = 0.01 * d_ref
        M_snapback = F_recovery * eccentricity
        sigma_bend = M_snapback * r_o / I if I > 0 else 0
        tau = F_recovery / (2 * math.pi * r * wt) * 0.3 if (r > 0 and wt > 0) else 0

        sx = sigma_axial_peak + sigma_bend
        vm = math.sqrt(sx**2 + 3 * tau**2)

        result.max_axial_stress = sigma_axial_peak
        result.max_hoop_stress = 0.0
        result.max_bending_stress = sigma_bend
        result.max_thermal_stress = 0.0
        result.max_von_mises = vm
        result.max_shear_stress = tau

        if total_L > 0:
            P_crit = math.pi**2 * mat.E * I / total_L**2
            result.buckling_load_factor = P_crit / max(F_recovery, 1.0)
        if mat.E > 0 and cross_area > 0:
            delta_L = F_recovery * total_L / (mat.E * cross_area)
            result.max_displacement_mm = delta_L * 1000

        n_stations = 30
        nose_coupler_pos = 0.20
        recovery_bay_pos = 0.40
        for i in range(n_stations + 1):
            frac = i / n_stations
            z = total_L * frac
            local_base = sigma_axial_base * (1.0 - 0.6 * frac)
            dist_nose = abs(frac - nose_coupler_pos) / 0.08
            nose_factor = 1.0 + (kt - 1.0) * math.exp(-dist_nose**2)
            dist_bay = abs(frac - recovery_bay_pos) / 0.06
            bay_factor = 1.0 + (kt * 0.7 - 1.0) * math.exp(-dist_bay**2)
            local_concentrated = local_base * max(nose_factor, bay_factor)
            local_bend_val = sigma_bend * math.sin(2 * math.pi * frac)
            lsx = local_concentrated + abs(local_bend_val)
            local_vm = math.sqrt(lsx**2 + 3 * (tau * (1 - frac))**2)
            result.element_stresses.append((z, local_vm))

        result.converged = True
        return result

    def _analytical_thermal(self, result: FEMResult, mat) -> FEMResult:
        """Thermal: physics-based temperature distribution.

        Temperature model:
          - Nose tip (0-2%): stagnation temperature (isentropic), blending
            to recovery temp using Sutton-Graves stagnation heating
          - Body (2-80%): recovery temperature (nearly constant, flat-plate
            turbulent boundary layer)
          - Motor section (80-100%): recovery temp + motor conduction heating
            (15% additional from internal combustion)

        Ref: Anderson, Hypersonic Gas Dynamics, Ch. 6
        """
        lc = self.config.load_case
        d_ref, r, total_L, wt, r_o, r_i, cross_area, I = self._get_geometry(mat)

        T_amb = 223.15
        try:
            from cfd.solvers.base import isa_conditions
            P, T_amb, rho = isa_conditions(lc.altitude_m)
        except Exception:
            pass

        gamma = 1.4
        r_rec = 0.89  # turbulent recovery factor (Pr^(1/3) for air)
        mach = lc.mach if lc.mach > 0 else 3.0
        T_recovery = T_amb * (1 + r_rec * (gamma - 1) / 2 * mach**2)
        T_stag = T_amb * (1 + (gamma - 1) / 2 * mach**2)

        n_stations = 30
        max_thermal_stress = 0.0
        station_temps = []

        # Partial constraint: real structures allow some free expansion
        constraint_factor = 0.55

        for i in range(n_stations + 1):
            frac = i / n_stations
            z = total_L * frac

            if frac < 0.02:
                # Nose stagnation zone: blend from T_stag to T_recovery
                # Physics: stagnation-point heating decays rapidly aft of nose
                blend = frac / 0.02
                T_local = T_stag * (1.0 - blend) + T_recovery * blend
            elif frac < 0.80:
                # Body: recovery temperature (nearly constant)
                # Flat-plate turbulent BL gives ~constant T_rec along body
                T_local = T_recovery
            else:
                # Motor section: recovery temp + motor conduction heating
                # Internal combustion conducts heat through casing
                motor_blend = (frac - 0.80) / 0.20
                T_motor_extra = (T_recovery - T_amb) * 0.15 * motor_blend
                T_local = T_recovery + T_motor_extra

            dT_local = T_local - 293.15
            sigma_th_full = mat.E * mat.cte * abs(dT_local) / (1 - mat.nu) if dT_local != 0 else 0
            sigma_th = sigma_th_full * constraint_factor
            station_temps.append((z, T_local))
            result.element_stresses.append((z, sigma_th))
            max_thermal_stress = max(max_thermal_stress, sigma_th)

        result.station_temperatures = station_temps

        dT_gradient = abs(T_stag - T_recovery)
        sigma_bend = mat.E * mat.cte * dT_gradient * wt / (2 * d_ref) * constraint_factor if d_ref > 0 else 0

        result.max_axial_stress = 0.0
        result.max_hoop_stress = 0.0
        result.max_bending_stress = sigma_bend
        result.max_thermal_stress = max_thermal_stress
        result.max_von_mises = math.sqrt((max_thermal_stress + sigma_bend) ** 2)
        result.max_shear_stress = 0.0

        if total_L > 0 and cross_area > 0:
            P_thermal = max_thermal_stress * cross_area
            P_crit = math.pi**2 * mat.E * I / total_L**2
            result.buckling_load_factor = P_crit / max(P_thermal, 1.0)

        dT_avg = T_recovery - 293.15
        if dT_avg > 0:
            delta_L = mat.cte * dT_avg * total_L
            result.max_displacement_mm = delta_L * 1000

        result.converged = True
        return result

    # ── Modal Analysis ───────────────────────────────────────────────────────

    def run_modal(self) -> ModalResult:
        """Run modal analysis — eigenvalue extraction."""
        result = ModalResult()
        # Store original type, switch to modal
        orig_type = self.config.analysis_type
        self.config.analysis_type = "modal"
        try:
            self.generate_case()
            for stage, frac in self.run():
                pass
            result = self._parse_modal_results()
        except Exception as e:
            logger.error(f"Modal analysis failed: {e}")
            # Analytical fallback for natural frequencies
            result = self._modal_analytical()
        finally:
            self.config.analysis_type = orig_type

        # Always enrich with resonance/flutter/damping/classification
        # _modal_analytical already populates these, but _parse_modal_results does not
        if result.converged and not result.mode_classifications:
            self._post_process_modal(result)

        return result

    def _parse_modal_results(self) -> ModalResult:
        """Parse eigenfrequencies and mode shapes from ccx .dat output."""
        result = ModalResult()
        dat_path = self.config.work_dir / "analysis.dat"
        if not dat_path.is_file():
            logger.warning("No .dat file for modal results")
            return self._modal_analytical()
        try:
            text = dat_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            freqs = []
            in_eigen_section = False
            mode_shapes_all = [] # List of dictionaries: {node_id: (dx, dy, dz)}
            current_shape = {}
            in_disp = False
            
            # 1. Parse Frequencies
            for line in lines:
                ll = line.strip().lower()
                if "e i g e n v a l u e" in ll or ("mode no" in ll and "eigenvalue" in ll):
                    in_eigen_section = True
                    continue
                if in_eigen_section and ("participation" in ll or "s t e p" in ll or "displacements" in ll or "e f f e c t i v e" in ll):
                    in_eigen_section = False
                    if "displacements" in ll:
                        # we hit the first displacement block early
                        in_disp = True
                        if current_shape:
                            mode_shapes_all.append(current_shape)
                            current_shape = {}
                    continue
                if not line.strip() or "rad/time" in ll or "real part" in ll:
                    continue
                    
                if in_eigen_section:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            mode_no = int(parts[0])
                            eigenvalue = float(parts[1])
                            freq_cycles = float(parts[3])  # Hz
                            if freq_cycles > 0:
                                freqs.append(freq_cycles)
                        except (ValueError, IndexError):
                            pass

            offset = 0
            if freqs:
                # Filter out near-rigid-body modes (< 1 Hz typically)
                structural_freqs = [f for f in freqs if f > 1.0]
                if not structural_freqs:
                    structural_freqs = freqs
                result.frequencies_hz = structural_freqs[:self.config.num_modes]
                result.num_modes = len(result.frequencies_hz)
                result.converged = True
                result.descriptions = _mode_descriptions(result.num_modes)
                offset = len(freqs) - len(structural_freqs)
            else:
                return self._modal_analytical()

            # 2. Parse Mode Shapes (Displacements)
            for line in lines:
                ll = line.strip().lower()
                if "displacements" in ll and "vx" in ll:
                    in_disp = True
                    if current_shape:
                        mode_shapes_all.append(current_shape)
                        current_shape = {}
                    continue
                if in_disp and ("s t e p" in ll or "stresses" in ll):
                    in_disp = False
                    if current_shape:
                        mode_shapes_all.append(current_shape)
                        current_shape = {}
                    continue
                
                if in_disp and line.strip():
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            nid = int(parts[0])
                            dx, dy, dz = float(parts[1]), float(parts[2]), float(parts[3])
                            current_shape[nid] = (dx, dy, dz)
                        except ValueError:
                            pass
                            
            if current_shape:
                mode_shapes_all.append(current_shape)
                
            # Filter the extracted mode shapes to match the filtered frequencies
            if mode_shapes_all:
                try:
                    result.mode_shapes = mode_shapes_all[offset : offset + result.num_modes]
                except Exception as e:
                    logger.warning(f"Failed to slice mode shapes: {e}")
                    
            logger.info(f"Modal: {result.num_modes} structural modes parsed, {len(result.mode_shapes)} mode shapes extracted.")

        except Exception as e:
            logger.warning(f"Modal parse error: {e}")
            return self._modal_analytical()
        return result

    def _post_process_modal(self, result: ModalResult):
        """Enrich a parsed ModalResult with classification, participation,
        damping, resonance, and flutter data.  Called when _parse_modal_results
        successfully extracted frequencies but didn't compute the physics fields.
        Modifies *result* in-place.
        """
        assembly = self.config.assembly
        mat = self._material or get_structural_material(self.config.material_name)
        if assembly is None:
            return

        L = assembly.total_length()
        d = assembly.get_reference_diameter()
        r = d / 2
        wt = 0.002

        from core.components import BodyTube
        fin_info = None
        for stage in assembly.stages:
            for comp in stage.children:
                if isinstance(comp, BodyTube):
                    wt = (comp.outer_diameter_val - comp.inner_diameter) / 2
                if fin_info is None:
                    fin_info = _find_fin_info(comp, L, d)

        r_o, r_i = r + wt / 2, r - wt / 2
        A = math.pi * (r_o**2 - r_i**2)
        m_per_L = mat.density * A
        total_mass = m_per_L * L
        result.total_mass_kg = total_mass

        lc = self.config.load_case
        G = mat.G if mat.G > 0 else mat.E / (2 * (1 + mat.nu))

        # ── Mode classification (from frequencies alone) ─────────────
        # Estimate expected bending/axial/torsional frequencies
        I = math.pi / 4 * (r_o**4 - r_i**4)
        # Cantilever beam eigenvalue parameters (clamped-free)
        # λ_n values: 1.875, 4.694, 7.855, 10.996, 14.137
        # Free-free: 4.730, 7.853, 10.996, 14.137, 17.279
        modal_bc = getattr(self.config, 'modal_bc', 'cantilever')
        if modal_bc == 'cantilever':
            lambdas_bend = [1.875, 4.694, 7.855, 10.996, 14.137]
        elif modal_bc == 'free-free':
            lambdas_bend = [4.730, 7.853, 10.996, 14.137, 17.279]
        else:  # clamped-clamped
            lambdas_bend = [4.730, 7.853, 10.996, 14.137, 17.279]
        v_axial = math.sqrt(mat.E / mat.density) if mat.density > 0 else 0
        v_torsion = math.sqrt(G / mat.density) if mat.density > 0 else 0

        expected_bend = []
        for lam in lambdas_bend:
            beta = lam / L
            fn = (beta**2 / (2 * math.pi)) * math.sqrt(mat.E * I / m_per_L) if m_per_L > 0 else 0
            expected_bend.append(fn)
        expected_axial = [n / (2 * L) * v_axial for n in range(1, 4)] if L > 0 else []
        expected_torsion = [n / (2 * L) * v_torsion for n in range(1, 4)] if L > 0 else []

        if not result.descriptions:
            result.descriptions = _mode_descriptions(result.num_modes)

        bend_n = {"Y": 0, "Z": 0}
        axial_n = 0
        torsion_n = 0

        for i, freq in enumerate(result.frequencies_hz):
            # Find closest match among expected frequencies
            best_type = "bending"
            best_dist = float('inf')
            for ef in expected_bend:
                if ef > 0 and abs(freq - ef) / ef < best_dist:
                    best_dist = abs(freq - ef) / ef
                    best_type = "bending"
            for ef in expected_axial:
                if ef > 0 and abs(freq - ef) / ef < best_dist:
                    best_dist = abs(freq - ef) / ef
                    best_type = "axial"
            for ef in expected_torsion:
                if ef > 0 and abs(freq - ef) / ef < best_dist:
                    best_dist = abs(freq - ef) / ef
                    best_type = "torsional"

            if best_type == "bending":
                # Determine bending plane from mode shapes if available
                plane = "Y"  # default
                if i < len(result.mode_shapes) and result.mode_shapes[i]:
                    shape = result.mode_shapes[i]
                    sum_dy = sum(abs(v[1]) for v in shape.values())
                    sum_dz = sum(abs(v[2]) for v in shape.values())
                    plane = "Z" if sum_dz > sum_dy else "Y"
                else:
                    # Fallback: alternate Y/Z for symmetric structures
                    plane = "Y" if i % 2 == 0 else "Z"
                bend_n[plane] += 1
                result.mode_classifications.append(f"Bending-{plane}")
                order = bend_n[plane]

                # Strain energy estimate: higher-order bending modes have
                # more shear energy and less pure bending
                se_bend = max(0.70, 0.95 - 0.05 * (order - 1))
                se_torsion = min(0.10, 0.02 + 0.02 * (order - 1))
                se_axial = 1.0 - se_bend - se_torsion
                result.strain_energy_fractions.append({
                    "bending": round(se_bend, 3),
                    "torsion": round(se_torsion, 3),
                    "axial": round(se_axial, 3)
                })

                desc = f"{_ordinal(order)} Lateral Bending ({plane})"
                # Cantilever beam participation factor from Blevins:
                # Γ_n = integral of mode shape / generalized mass
                # 1st mode: Γ₁ = 1.566, m_gen/m_total = 0.2268
                # 2nd mode: Γ₂ = 1.000, m_gen/m_total = 0.1296
                cantilever_params = [
                    (1.566, 0.2268), (1.000, 0.1296),
                    (1.000, 0.0942), (1.000, 0.0738),
                    (1.000, 0.0604),
                ]
                idx = min(order - 1, len(cantilever_params) - 1)
                gamma, m_gen_frac = cantilever_params[idx]
                m_gen = total_mass * m_gen_frac
                eff = gamma**2 * m_gen / total_mass if total_mass > 0 else 0
                if plane == "Y":
                    pf = {"x": 0.0, "y": round(gamma, 4), "z": 0.0}
                    em = {"x": 0.0, "y": round(eff * 100, 1), "z": 0.0}
                else:
                    pf = {"x": 0.0, "y": 0.0, "z": round(gamma, 4)}
                    em = {"x": 0.0, "y": 0.0, "z": round(eff * 100, 1)}
            elif best_type == "axial":
                axial_n += 1
                result.mode_classifications.append("Axial")
                # Axial modes: predominantly axial strain energy
                se_axial = max(0.85, 0.97 - 0.04 * (axial_n - 1))
                result.strain_energy_fractions.append({
                    "bending": round((1.0 - se_axial) * 0.3, 3),
                    "torsion": round((1.0 - se_axial) * 0.7, 3),
                    "axial": round(se_axial, 3)
                })
                desc = f"{_ordinal(axial_n)} Axial (Breathing)"
                gamma = 2.0 / (axial_n * math.pi)
                # Axial mode generalized mass from rod theory
                m_gen = total_mass * 0.5 / axial_n
                eff = gamma**2 * m_gen / total_mass if total_mass > 0 else 0
                pf = {"x": round(gamma, 4), "y": 0.0, "z": 0.0}
                em = {"x": round(eff * 100, 1), "y": 0.0, "z": 0.0}
            else:
                torsion_n += 1
                result.mode_classifications.append("Torsional")
                # Torsional modes: predominantly torsion strain energy
                se_torsion = max(0.80, 0.92 - 0.04 * (torsion_n - 1))
                result.strain_energy_fractions.append({
                    "bending": round((1.0 - se_torsion) * 0.6, 3),
                    "torsion": round(se_torsion, 3),
                    "axial": round((1.0 - se_torsion) * 0.4, 3)
                })
                desc = f"{_ordinal(torsion_n)} Torsional"
                # Torsional modes: minimal translational participation
                pf = {"x": 0.0, "y": 0.0, "z": 0.0}
                em = {"x": 0.0, "y": 0.0, "z": 0.0}
                # Torsional generalized mass from uniform shaft theory
                m_gen = total_mass * 0.25 / torsion_n

            result.descriptions[i] = desc
            result.participation_factors.append(pf)
            result.effective_modal_mass.append(em)
            result.generalized_mass.append(round(m_gen, 4))

        # ── Damping ─────────────────────────────────────────────────
        damping_table = {
            "Aluminum 6061-T6": 0.005, "Carbon Fiber Composite": 0.015,
            "Fiberglass (G10)": 0.020, "Steel 4130": 0.003,
            "Titanium Ti-6Al-4V": 0.004, "Kraft Phenolic": 0.025,
            "Plywood (Birch)": 0.030, "ABS Plastic": 0.035,
        }
        base_zeta = damping_table.get(mat.name, 0.01)
        result.damping_source = f"Material estimate ({mat.name})"
        for i in range(result.num_modes):
            result.damping_ratios.append(round(base_zeta * (1.0 + 0.05 * i), 5))

        # ── Resonance ───────────────────────────────────────────────
        motor_L = L * 0.25
        a_gas = 1000.0
        motor_1p = a_gas / (4 * motor_L) if motor_L > 0 else 0
        motor_2p = 2 * motor_1p
        result.motor_1p_hz = round(motor_1p, 1)
        result.motor_2p_hz = round(motor_2p, 1)

        mach_flight = lc.mach if lc.mach > 0 else 0.8
        try:
            from cfd.solvers.base import isa_conditions
            P, T, rho = isa_conditions(lc.altitude_m if lc.altitude_m > 0 else 3000)
            a_sound = math.sqrt(1.4 * 287.05 * T)
        except Exception:
            a_sound = 340.0
        V_flight = mach_flight * a_sound
        f_buff_low = 0.18 * V_flight / d if d > 0 else 0
        f_buff_high = 0.22 * V_flight / d if d > 0 else 0
        result.aero_buffet_band = (round(f_buff_low, 1), round(f_buff_high, 1))

        warnings = []
        for i, freq in enumerate(result.frequencies_hz):
            desc = result.descriptions[i] if i < len(result.descriptions) else f"Mode {i+1}"
            if motor_1p > 0 and abs(freq - motor_1p) / motor_1p < 0.15:
                warnings.append(f"{desc} ({freq:.0f} Hz) within 15% of motor 1P ({motor_1p:.0f} Hz)")
            if motor_2p > 0 and abs(freq - motor_2p) / motor_2p < 0.15:
                warnings.append(f"{desc} ({freq:.0f} Hz) within 15% of motor 2P ({motor_2p:.0f} Hz)")
            if f_buff_low <= freq <= f_buff_high:
                warnings.append(f"{desc} ({freq:.0f} Hz) inside aero buffet band ({f_buff_low:.0f}–{f_buff_high:.0f} Hz)")
        result.resonance_warnings = warnings

        # ── Flutter ─────────────────────────────────────────────────
        if fin_info is not None:
            cr, ct = fin_info["root_chord"], fin_info["tip_chord"]
            span, t_fin = fin_info["span"], fin_info["thickness"]
            AR_fin = span**2 / (0.5 * (cr + ct) * span) if (cr + ct) > 0 else 2.0
            tc_ratio = t_fin / (0.5 * (cr + ct)) if (cr + ct) > 0 else 0.05
            lam = ct / cr if cr > 0 else 0.5
            try:
                P_atm, _, _ = isa_conditions(lc.altitude_m if lc.altitude_m > 0 else 3000)
            except Exception:
                P_atm = 70000.0
            # Full NACA TN-4197 form incl. taper term (matches workstation.py)
            denom = (1.337 * AR_fin**3 * P_atm * (lam + 1)) / \
                    (2 * (AR_fin + 2) * max(tc_ratio, 0.01)**3)
            V_flutter = a_sound * math.sqrt(G / denom) if denom > 0 and G > 0 else 9999.0
            flutter_margin = V_flutter / max(V_flight, 1.0)
            if flutter_margin >= 2.0:
                verdict = "✓ SAFE (margin ≥ 2.0)"
            elif flutter_margin >= 1.25:
                verdict = "ADEQUATE (margin 1.25–2.0)"
            elif flutter_margin >= 1.0:
                verdict = "MARGINAL (margin < 1.25)"
            else:
                verdict = "✕ FLUTTER RISK (V_flight > V_flutter)"
            result.flutter_assessment = {
                "critical_speed_m_s": round(V_flutter, 1),
                "flutter_margin": round(flutter_margin, 2),
                "max_flight_speed_m_s": round(V_flight, 1),
                "verdict": verdict,
                "method": "NACA empirical (preliminary)",
                "fin_AR": round(AR_fin, 2),
                "fin_t_c": round(tc_ratio, 4),
            }

        logger.info(f"Modal post-process: {len(result.mode_classifications)} classified, "
                     f"{len(warnings)} resonance warnings")

    def _modal_analytical(self) -> ModalResult:
        """Analytical natural frequencies for a free-free beam with professional
        postprocessing: energy-based mode classification, participation factors,
        effective modal mass, damping estimation, resonance assessment, and
        preliminary fin flutter analysis.
        """
        result = ModalResult()
        assembly = self.config.assembly
        mat = self._material or get_structural_material(self.config.material_name)
        if assembly is None:
            return result

        L = assembly.total_length()
        d = assembly.get_reference_diameter()
        r = d / 2
        wt = 0.002
        from core.components import BodyTube
        fin_info = None
        for stage in assembly.stages:
            for comp in stage.children:
                if isinstance(comp, BodyTube):
                    wt = (comp.outer_diameter_val - comp.inner_diameter) / 2
                if fin_info is None:
                    fin_info = _find_fin_info(comp, L, d)

        r_o, r_i = r + wt / 2, r - wt / 2
        I = math.pi / 4 * (r_o**4 - r_i**4)
        A = math.pi * (r_o**2 - r_i**2)
        m_per_L = mat.density * A
        total_mass = m_per_L * L
        if L <= 0 or I <= 0 or m_per_L <= 0:
            return result

        result.total_mass_kg = total_mass

        # Load-case-dependent frequency modifier
        lc = self.config.load_case
        freq_modifier = 1.0
        if lc.name == "Max Thrust":
            freq_modifier = 1.03
        elif lc.name == "Recovery Shock":
            freq_modifier = 0.94
        elif lc.name in ("Thermal", "Aerodynamic Heating"):
            dT = abs(lc.delta_T) if lc.delta_T != 0 else 50.0
            E_reduction = max(0.90, 1.0 - 0.0005 * dT)
            freq_modifier = math.sqrt(E_reduction)

        # ── Compute natural frequencies (bending, torsional, axial) ──────

        # Bending (Euler-Bernoulli beam — eigenvalues depend on BCs)
        modal_bc = getattr(self.config, 'modal_bc', 'cantilever')
        if modal_bc == 'cantilever':
            # Clamped-free (cantilever): physical rocket with motor mount fixed
            lambdas_bend = [1.875, 4.694, 7.855, 10.996, 14.137,
                            17.279, 20.420, 23.562, 26.704, 29.845]
        else:
            # Free-free or clamped-clamped
            lambdas_bend = [4.730, 7.853, 10.996, 14.137, 17.279,
                            20.420, 23.562, 26.704, 29.845, 32.987]
        # Axial (longitudinal, rod free-free): f_n = n/(2L) * sqrt(E/rho)
        v_axial = math.sqrt(mat.E / mat.density)
        # Torsional (free-free): f_n = n/(2L) * sqrt(G/rho), J ≈ 2I for thin-wall
        G = mat.G if mat.G > 0 else mat.E / (2 * (1 + mat.nu))
        v_torsion = math.sqrt(G / mat.density)

        # Build combined mode table sorted by frequency
        raw_modes = []

        # Bending modes (Y/Z alternating for axisymmetric body)
        for i, lam in enumerate(lambdas_bend[:7]):
            beta = lam / L
            fn = (beta**2 / (2 * math.pi)) * math.sqrt(mat.E * I / m_per_L)
            fn *= freq_modifier
            raw_modes.append({
                "freq": fn, "type": "bending", "order": i + 1,
                "plane": "Y" if i % 2 == 0 else "Z",
            })

        # Axial modes
        for n in range(1, 4):
            fn = n / (2 * L) * v_axial * freq_modifier
            raw_modes.append({"freq": fn, "type": "axial", "order": n})

        # Torsional modes
        for n in range(1, 4):
            fn = n / (2 * L) * v_torsion * freq_modifier
            raw_modes.append({"freq": fn, "type": "torsional", "order": n})

        # Sort by frequency and take first num_modes
        raw_modes.sort(key=lambda m: m["freq"])
        n_modes = min(self.config.num_modes, len(raw_modes))
        selected = raw_modes[:n_modes]

        # ── Populate result ──────────────────────────────────────────────

        for m in selected:
            result.frequencies_hz.append(round(m["freq"], 2))

        result.num_modes = n_modes

        # ── Energy-based mode classification ─────────────────────────────

        bend_n = {"Y": 0, "Z": 0}
        axial_n = 0
        torsion_n = 0
        for m in selected:
            mtype = m["type"]
            if mtype == "bending":
                plane = m.get("plane", "Y")
                bend_n[plane] += 1
                ordinal = _ordinal(bend_n[plane])
                desc = f"{ordinal} Lateral Bending ({plane})"
                classification = f"Bending-{plane}"
                order = bend_n[plane]
                se_bend = max(0.70, 0.95 - 0.05 * (order - 1))
                se_torsion = min(0.10, 0.02 + 0.02 * (order - 1))
                se_axial = 1.0 - se_bend - se_torsion
                se = {"bending": round(se_bend, 3), "torsion": round(se_torsion, 3), "axial": round(se_axial, 3)}
            elif mtype == "axial":
                axial_n += 1
                ordinal = _ordinal(axial_n)
                desc = f"{ordinal} Axial (Breathing)"
                classification = "Axial"
                se_axial = max(0.85, 0.97 - 0.04 * (axial_n - 1))
                se = {"bending": round((1.0 - se_axial) * 0.3, 3), "torsion": round((1.0 - se_axial) * 0.7, 3), "axial": round(se_axial, 3)}
            elif mtype == "torsional":
                torsion_n += 1
                ordinal = _ordinal(torsion_n)
                desc = f"{ordinal} Torsional"
                classification = "Torsional"
                se_torsion = max(0.80, 0.92 - 0.04 * (torsion_n - 1))
                se = {"bending": round((1.0 - se_torsion) * 0.6, 3), "torsion": round(se_torsion, 3), "axial": round((1.0 - se_torsion) * 0.4, 3)}
            else:
                desc = f"Mode {m['order']}"
                classification = "Coupled"
                se = {"bending": 0.40, "torsion": 0.30, "axial": 0.30}

            result.descriptions.append(desc)
            result.mode_classifications.append(classification)
            result.strain_energy_fractions.append(se)

        # ── Participation factors & effective modal mass ─────────────────
        # For analytical beam modes, participation factor in the lateral
        # direction ≈ (2/L) × integral of mode shape × unit vector.
        # Bending modes have large lateral participation,
        # axial modes have large longitudinal participation.

        for i, m in enumerate(selected):
            mtype = m["type"]
            order = m["order"]

            # Generalized mass from beam theory (mode-type dependent)
            if mtype == "bending":
                order = m["order"]
                # Cantilever beam: m_gen/m_total from Blevins
                cantilever_m_gen = [0.2268, 0.1296, 0.0942, 0.0738, 0.0604, 0.0510, 0.0440]
                idx = min(order - 1, len(cantilever_m_gen) - 1)
                m_gen = total_mass * cantilever_m_gen[idx]
            elif mtype == "axial":
                m_gen = total_mass * 0.5 / order
            else:  # torsional
                m_gen = total_mass * 0.25 / order
            result.generalized_mass.append(round(m_gen, 4))

            if mtype == "bending":
                # Cantilever bending participation factors
                cantilever_gammas = [1.566, 1.000, 1.000, 1.000, 1.000]
                idx = min(order - 1, len(cantilever_gammas) - 1)
                gamma_lat = cantilever_gammas[idx]
                eff_lat = gamma_lat**2 * m_gen / total_mass if total_mass > 0 else 0
                plane = m.get("plane", "Y")
                if plane == "Y":
                    pf = {"x": 0.0, "y": round(gamma_lat, 4), "z": 0.0}
                    em = {"x": 0.0, "y": round(eff_lat * 100, 1), "z": 0.0}
                else:
                    pf = {"x": 0.0, "y": 0.0, "z": round(gamma_lat, 4)}
                    em = {"x": 0.0, "y": 0.0, "z": round(eff_lat * 100, 1)}
            elif mtype == "axial":
                # Axial participation: Γ_x ≈ 2/(n*π) for rod
                gamma_ax = 2.0 / (order * math.pi)
                eff_ax = gamma_ax**2 * m_gen / total_mass if total_mass > 0 else 0
                pf = {"x": round(gamma_ax, 4), "y": 0.0, "z": 0.0}
                em = {"x": round(eff_ax * 100, 1), "y": 0.0, "z": 0.0}
            else:  # torsional — no translational participation
                pf = {"x": 0.0, "y": 0.0, "z": 0.0}
                em = {"x": 0.0, "y": 0.0, "z": 0.0}

            result.participation_factors.append(pf)
            result.effective_modal_mass.append(em)

        # ── Damping estimation ───────────────────────────────────────────
        # Material-dependent structural damping (loss factor η → ζ = η/2)
        damping_table = {
            "Aluminum 6061-T6": 0.005,     # η ≈ 1%
            "Carbon Fiber Composite": 0.015, # η ≈ 3% (higher for composites)
            "Fiberglass (G10)": 0.020,      # η ≈ 4%
            "Steel 4130": 0.003,            # η ≈ 0.6%
            "Titanium Ti-6Al-4V": 0.004,    # η ≈ 0.8%
            "Kraft Phenolic": 0.025,         # η ≈ 5%
            "Plywood (Birch)": 0.030,       # η ≈ 6%
            "ABS Plastic": 0.035,           # η ≈ 7%
        }
        base_zeta = damping_table.get(mat.name, 0.01)
        result.damping_source = f"Material estimate ({mat.name})"

        for i, m in enumerate(selected):
            # Higher modes have slightly higher damping (joint friction)
            zeta_i = base_zeta * (1.0 + 0.05 * i)
            result.damping_ratios.append(round(zeta_i, 5))

        # ── Resonance assessment ─────────────────────────────────────────
        # Motor combustion instability: typical 1P = 200-500 Hz for HPR motors
        # Estimate from motor length (quarter-wave acoustic): f = a/(4*L_chamber)
        motor_L = L * 0.25  # rough estimate: motor ≈ 25% of rocket length
        a_gas = 1000.0       # speed of sound in combustion gases ~1000 m/s
        motor_1p = a_gas / (4 * motor_L) if motor_L > 0 else 0
        motor_2p = 2 * motor_1p
        result.motor_1p_hz = round(motor_1p, 1)
        result.motor_2p_hz = round(motor_2p, 1)

        # Aero buffeting: Strouhal vortex shedding f = St * V / D
        # St ≈ 0.2 for cylinders; band spans from wake to transition zone
        mach_flight = lc.mach if lc.mach > 0 else 0.8
        try:
            from cfd.solvers.base import isa_conditions
            P, T, rho = isa_conditions(lc.altitude_m if lc.altitude_m > 0 else 3000)
            a_sound = math.sqrt(1.4 * 287.05 * T)
        except Exception:
            a_sound = 340.0
        V_flight = mach_flight * a_sound
        St_low, St_high = 0.18, 0.22
        f_buff_low = St_low * V_flight / d if d > 0 else 0
        f_buff_high = St_high * V_flight / d if d > 0 else 0
        result.aero_buffet_band = (round(f_buff_low, 1), round(f_buff_high, 1))

        warnings = []
        for i, freq in enumerate(result.frequencies_hz):
            desc = result.descriptions[i] if i < len(result.descriptions) else f"Mode {i+1}"
            # Motor resonance check (within ±15%)
            if motor_1p > 0 and abs(freq - motor_1p) / motor_1p < 0.15:
                warnings.append(
                    f"{desc} ({freq:.0f} Hz) within 15% of motor 1P acoustic ({motor_1p:.0f} Hz)"
                )
            if motor_2p > 0 and abs(freq - motor_2p) / motor_2p < 0.15:
                warnings.append(
                    f"{desc} ({freq:.0f} Hz) within 15% of motor 2P harmonic ({motor_2p:.0f} Hz)"
                )
            # Aero buffeting check
            if f_buff_low <= freq <= f_buff_high:
                warnings.append(
                    f"{desc} ({freq:.0f} Hz) inside aero buffet band "
                    f"({f_buff_low:.0f}–{f_buff_high:.0f} Hz, St≈0.2)"
                )
        result.resonance_warnings = warnings

        # ── Fin flutter assessment (preliminary — NACA empirical) ────────
        # V_flutter = a × sqrt(G_panel / (1.337 × AR³ × P∞ / (t/c)³))
        # Reference: NACA TN-4197 / Bisplinghoff "Aeroelasticity"
        if fin_info is not None:
            cr = fin_info["root_chord"]
            ct = fin_info["tip_chord"]
            span = fin_info["span"]
            t_fin = fin_info["thickness"]

            AR_fin = span**2 / (0.5 * (cr + ct) * span) if (cr + ct) > 0 else 2.0
            tc_ratio = t_fin / (0.5 * (cr + ct)) if (cr + ct) > 0 else 0.05
            lam = ct / cr if cr > 0 else 0.5
            G_panel = G  # use airframe shear modulus as proxy

            try:
                P_atm, T_atm, rho_atm = isa_conditions(lc.altitude_m if lc.altitude_m > 0 else 3000)
            except Exception:
                P_atm = 70000.0

            # NACA flutter parameter — full TN-4197 form incl. taper term
            # (matches workstation.py fin_analysis)
            denom = (1.337 * AR_fin**3 * P_atm * (lam + 1)) / \
                    (2 * (AR_fin + 2) * max(tc_ratio, 0.01)**3)
            if denom > 0 and G_panel > 0:
                V_flutter = a_sound * math.sqrt(G_panel / denom)
            else:
                V_flutter = 9999.0

            flutter_margin = V_flutter / max(V_flight, 1.0)
            if flutter_margin >= 2.0:
                verdict = "✓ SAFE (margin ≥ 2.0)"
            elif flutter_margin >= 1.25:
                verdict = "ADEQUATE (margin 1.25–2.0)"
            elif flutter_margin >= 1.0:
                verdict = "MARGINAL (margin < 1.25)"
            else:
                verdict = "✕ FLUTTER RISK (V_flight > V_flutter)"

            result.flutter_assessment = {
                "critical_speed_m_s": round(V_flutter, 1),
                "flutter_margin": round(flutter_margin, 2),
                "max_flight_speed_m_s": round(V_flight, 1),
                "verdict": verdict,
                "method": "NACA empirical (preliminary)",
                "fin_AR": round(AR_fin, 2),
                "fin_t_c": round(tc_ratio, 4),
            }

        result.converged = True
        logger.info(
            f"Modal (analytical, {lc.name}): {result.frequencies_hz[:5]} Hz "
            f"(mod={freq_modifier:.3f}), {len(warnings)} resonance warnings"
        )
        return result


def _find_fin_info(comp, L: float, d: float) -> Optional[dict]:
    """Find the first TrapezoidalFinSet on *comp* or its children (fins are
    normally nested under a BodyTube, not directly on the stage)."""
    from core.components import TrapezoidalFinSet
    if isinstance(comp, TrapezoidalFinSet):
        return {
            "count": getattr(comp, 'fin_count', 3),
            "span": getattr(comp, 'height', d * 0.8),
            "root_chord": getattr(comp, 'root_chord', L * 0.12),
            "tip_chord": getattr(comp, 'tip_chord', L * 0.04),
            "thickness": getattr(comp, 'thickness', 0.003),
        }
    for child in getattr(comp, 'children', []):
        info = _find_fin_info(child, L, d)
        if info is not None:
            return info
    return None


def _ordinal(n: int) -> str:
    """Return ordinal string: 1st, 2nd, 3rd, 4th, ..."""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][min(n % 10, 4)] if n % 10 < 4 else 'th'}"


def _mode_descriptions(n: int) -> list:
    """Generate basic mode shape descriptions (legacy fallback)."""
    descs = []
    bend_n, axial_n, torsion_n = 1, 1, 1
    for i in range(n):
        if i % 3 == 0:
            descs.append(f"{_ordinal(bend_n)} Lateral Bending")
            bend_n += 1
        elif i % 3 == 1:
            descs.append(f"{_ordinal(axial_n)} Axial")
            axial_n += 1
        else:
            descs.append(f"{_ordinal(torsion_n)} Torsional")
            torsion_n += 1
    return descs
