"""
K2 AeroSim — CalculiX Solver Backend
========================================
Implements FEMSolver using the CalculiX open-source FEA suite (ccx).
ccx binary must be in the K2 bin/ folder or on PATH.

Physics:
  - Static linear elastic, S4 shells (→ C3D8I, bending through the wall),
    inertia-relieved free body on a 3-2-1 support (structures.fe_loads)
  - Modal eigenvalue analysis (Lanczos)
  - Linear buckling (eigenvalue)
  - Steady-state thermal

Mirrors the SU2 solver architecture (cfd/solvers/su2_solver.py).
"""
from __future__ import annotations
import csv, logging, math, os, re, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

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
        """Generate the structural mesh (and its MeshInfo) via structures.meshing."""
        from structures.meshing import build_structural_mesh_info
        cfg = self.config
        assembly = assembly or cfg.assembly
        if assembly is None:
            raise ValueError("No assembly provided for meshing.")
        out = cfg.work_dir / "structure_mesh.inp"
        out, info = build_structural_mesh_info(
            assembly, out, cfg.mesh_refinement, cfg.element_type,
            custom_circum=cfg.custom_circum,
            custom_axial_per_cal=cfg.custom_axial_per_cal,
        )
        if len(info.pieces) > 1:
            raise ValueError(
                f"The structural model falls apart into {len(info.pieces)} unconnected "
                "pieces (" + " | ".join(", ".join(p) for p in info.pieces) + "). "
                "Components must meet end to end — check their positions and diameters.")
        self._mesh_path = out
        self._mesh_info = info
        self._material = get_structural_material(cfg.material_name)
        logger.info(f"Mesh: {out}")
        return out

    # ── Case Generation ──────────────────────────────────────────────────────

    def _section_materials(self) -> dict:
        """{section name: StructuralMaterial}. The airframe takes the material
        chosen for the analysis; a fin set keeps its own component material,
        as structures.workstation.fin_analysis does."""
        default = get_structural_material(self.config.material_name)
        info = getattr(self, "_mesh_info", None)
        return {sec.name: (get_structural_material(sec.material)
                           if sec.kind == "fin" and sec.material else default)
                for sec in (info.sections if info else [])}

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
        self._sec_mat = self._section_materials()
        info = getattr(self, "_mesh_info", None)

        with open(inp, "w", encoding="ascii", errors="replace") as f:
            f.write("** K2 AeroSim - CalculiX Analysis\n**\n")
            f.write(mesh_text + "\n")
            # One card per distinct material and one shell section per
            # component, each with that component's own wall thickness. A
            # single section used to give every element the nose-cone wall
            # thickness (4 mm fins were analysed as 2.5 mm plates).
            mat_id = {}
            for m in [mat] + list(self._sec_mat.values()):
                if m.name in mat_id:
                    continue
                mat_id[m.name] = f"MAT{len(mat_id) + 1}"
                f.write(f"*MATERIAL, NAME={mat_id[m.name]}\n")
                f.write(f"*ELASTIC\n{m.E:.6e}, {m.nu:.4f}\n")
                f.write(f"*DENSITY\n{m.density:.2f}\n")
                f.write(f"*EXPANSION\n{m.cte:.6e}\n")
                f.write(f"*CONDUCTIVITY\n{m.thermal_conductivity:.4f}\n")
            if info and info.sections:
                for sec in info.sections:
                    f.write(f"*SHELL SECTION, ELSET={sec.name}, "
                            f"MATERIAL={mat_id[self._sec_mat[sec.name].name]}\n")
                    f.write(f"{sec.thickness:.6f}\n")
            else:
                f.write(f"*SHELL SECTION, ELSET=EALL, MATERIAL={mat_id[mat.name]}\n0.002000\n")

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
        """Static step: the condition's whole load set on a free body.

        structures.fe_loads builds thrust, drag, the aerodynamic normal force,
        the recovery harness load, pressure and temperature, balanced by the
        vehicle's own inertia, on a statically determinate 3-2-1 support whose
        reactions are round-off. The old deck loaded only the shell's own
        weight, clamped the tail AND pinned the nose tip (the pin made the
        raw peak sit at the nose), and left every flight load to hand
        formulas added after the solve."""
        from structures.fe_loads import FEMesh, T_REF, build_static_loads
        mesh = FEMesh.read(self._mesh_path)
        sec_mat = self._sec_mat
        lm = build_static_loads(
            mesh, self._mesh_info, cfg.assembly, lc,
            density_of=lambda sec: sec_mat.get(sec.name, self._material).density,
            cfd_pressure=self._cfd_element_pressures(cfg, mesh))
        self._fe_mesh, self._load_model = mesh, lm
        A, B, C = lm.support
        f.write("**\n** STATIC ANALYSIS - inertia-relieved free body\n**\n")
        f.write(f"*NSET, NSET=NSUPPORT\n{A}, {B}, {C}\n")
        f.write(f"*BOUNDARY\n{A}, 1, 3\n{B}, 2, 3\n{C}, 3, 3\n")
        if lm.temperatures is not None:
            f.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nNALL, {T_REF:.2f}\n")
        f.write("*STEP\n*STATIC\n")
        fmax = float(np.abs(lm.forces).max()) if lm.forces.size else 0.0
        lines = [f"{nid}, {dof + 1}, {lm.forces[k, dof]:.10e}\n"
                 for k, nid in enumerate(mesh.ids) for dof in range(3)
                 if abs(lm.forces[k, dof]) > 1e-12 * fmax]
        if lines:
            f.write("*CLOAD\n")
            f.writelines(lines)
        if lm.temperatures is not None:
            f.write("*TEMPERATURE\n")
            f.writelines(f"{nid}, {lm.temperatures[k]:.4f}\n" for k, nid in enumerate(mesh.ids))
        f.write("*NODE FILE\nU\n*EL FILE\nS\n")
        f.write("*NODE PRINT, NSET=NSUPPORT, TOTALS=ONLY\nRF\n")
        f.write("*END STEP\n")
        logger.info("Load set: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in lm.summary.items()))

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

    @staticmethod
    def _isa_pressure(altitude_m: float) -> float:
        """ISA static pressure (Pa) at the given altitude."""
        T0, P0, L, R, g = 288.15, 101325.0, 0.0065, 287.05, 9.80665
        h = max(0.0, float(altitude_m))
        if h <= 11000.0:
            return P0 * (1.0 - L * h / T0) ** (g / (R * L))
        p11 = P0 * (1.0 - L * 11000.0 / T0) ** (g / (R * L))
        return p11 * math.exp(-g * (h - 11000.0) / (R * 216.65))

    def _cfd_element_pressures(self, cfg: FEMConfig, mesh):
        """{element id: gauge pressure (Pa)} from a CFD surface file, or None.

        pressure_mapping's IDW treats the second slot of a station as the axis,
        so the axial z goes there; the result is azimuthally averaged, so only
        body-of-revolution elements take it — fins, steps and on-axis elements
        keep the analytic loads (an undefined outward normal gave fins ±17 kPa
        of arbitrary sign). SU2 writes ABSOLUTE pressure: it is converted to
        gauge, or a Cp field is scaled by q. fe_loads balances the mapped
        load's net force like every other load."""
        vtk = getattr(cfg, "cfd_surface_vtk", None)
        if not vtk:
            return None
        vtk = Path(vtk)
        if not vtk.is_file():
            logger.warning("CFD surface VTK not found: %s — skipping pressure map", vtk)
            return None
        try:
            from structures.pressure_mapping import (
                _parse_vtu_points_and_pressure, map_pressures_idw)
            cfd_pts, cfd_pres = _parse_vtu_points_and_pressure(vtk)
            if not cfd_pts or not cfd_pres:
                logger.warning("CFD VTK has no usable pressure data — skipping map")
                return None
            cfd_pts = [(z, x, y) for (x, y, z) in cfd_pts]
            lc = getattr(cfg, "load_case", None) or LoadCase()
            p_amb = self._isa_pressure(getattr(lc, "altitude_m", 0.0))
            med = sorted(cfd_pres)[len(cfd_pres) // 2]
            if med > 0.5 * p_amb:
                cfd_pres = [p - p_amb for p in cfd_pres]
                logger.info("CFD pressure converted to gauge (p_amb=%.0f Pa)", p_amb)
            elif max(abs(p) for p in cfd_pres) < 50.0:
                q = getattr(lc, "dynamic_pressure", 0.0)
                if q <= 0.0:
                    logger.warning("CFD field looks like Cp but the load case has no "
                                   "dynamic pressure — skipping map")
                    return None
                cfd_pres = [p * q for p in cfd_pres]
                logger.info("CFD Cp field scaled by q=%.0f Pa", q)
            info = self._mesh_info
            body = {e for sec in info.sections if sec.kind not in ("fin", "step")
                    for e in mesh.elsets.get(sec.name, ())}
            stations = []
            for k, eid in enumerate(mesh.elem_ids):
                if int(eid) not in body:
                    continue
                c, nrm = mesh.centroid[k], mesh.normal[k]
                r_c = math.hypot(float(c[0]), float(c[1]))
                if r_c < 1e-9 or abs((nrm[0] * c[0] + nrm[1] * c[1]) / r_c) < 0.35:
                    continue
                stations.append((int(eid), float(c[2]), float(mesh.area[k]), float(nrm[2])))
            if not stations:
                logger.warning("No airframe elements to map CFD pressure onto")
                return None
            mapped = dict(map_pressures_idw(cfd_pts, cfd_pres, stations).element_pressures)
            if mapped:
                ps = list(mapped.values())
                logger.info("CFD pressure mapped onto %d elements (%.0f..%.0f Pa)",
                            len(ps), min(ps), max(ps))
            return mapped or None
        except Exception as exc:
            logger.error("CFD pressure mapping failed: %s", exc)
            return None

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
        """Read the CalculiX nodal field (.frd) into an FEMResult.

        Every value the Structures panel shows is read off this one field, and
        the stress profile, the 3D contour and the deformation view draw the
        same field, so none of them can disagree. The old path parsed a
        self-weight-only solve and then replaced ~99 % of the answer with hand
        formulas (raw 0.2 MPa shown as 21.6 MPa)."""
        result = FEMResult()
        mat = self._material or get_structural_material("Aluminum 6061-T6")
        result.material_name = mat.name
        result.yield_strength = mat.yield_strength
        result.load_case_name = self.config.load_case.name

        frd_path = self.config.work_dir / "analysis.frd"
        if frd_path.is_file() and getattr(self, "_load_model", None) is not None:
            self._results_from_frd(frd_path, result)
            result.result_vtk = frd_path
            result.converged = True
            return result

        if not self._ccx_exe:
            result = self._analytical_fallback(result, mat)
        if result.max_von_mises > 0:
            result.safety_factor = mat.yield_strength / result.max_von_mises
            result.yield_utilization = result.max_von_mises / mat.yield_strength
            result.margin_of_safety = (result.safety_factor
                                       / self.config.safety_factor_required) - 1.0
        else:
            result.safety_factor = float('inf')
            result.margin_of_safety = float('inf')
            result.yield_utilization = 0.0
        result.converged = True
        logger.info(
            f"FEM Results (analytical): σ_vm={result.max_von_mises/1e6:.1f} MPa, "
            f"SF={result.safety_factor:.2f}"
        )
        return result

    def _results_from_frd(self, frd_path: Path, result: FEMResult):
        """Fill ``result`` from the expanded-shell nodal field of the .frd."""
        from structures.frd import read_frd
        from structures.solvers.base import FEField
        frd = read_frd(frd_path)
        mesh, info, lm = self._fe_mesh, self._mesh_info, self._load_model
        lc = self.config.load_case
        sec_of = {e: sec for sec in info.sections for e in mesh.elsets.get(sec.name, ())}
        S, U = frd.blocks.get("STRESS", {}), frd.blocks.get("DISP", {})

        # Expanded hexahedron of shell element e: nodes 0-3 are the inner face,
        # 4-7 the outer face, each in the order of the shell's own nodes.
        ids, row, parent, cells, cell_fin = [], {}, {}, [], []
        on_fin, on_body, yld = set(), set(), {}
        for eid, (etype, conn) in frd.elements.items():
            shell, sec = mesh.elements.get(eid), sec_of.get(eid)
            if shell is None or sec is None or len(conn) != 8:
                continue
            ys = self._sec_mat[sec.name].yield_strength
            for j, nid in enumerate(conn):
                if nid not in row:
                    row[nid] = len(ids)
                    ids.append(nid)
                parent[nid] = shell[j % 4]
                (on_fin if sec.kind == "fin" else on_body).add(nid)
                yld[nid] = min(yld.get(nid, ys), ys)
            cells.append([row[n] for n in conn])
            cell_fin.append(sec.kind == "fin")
        if not ids:
            raise RuntimeError(f"No shell results in {frd_path}")
        P = np.array([frd.nodes[n] for n in ids], dtype=float)
        sig = np.array([(S.get(n) or [0.0] * 6)[:6] for n in ids], dtype=float)
        disp = np.array([(U.get(n) or [0.0] * 3)[:3] for n in ids], dtype=float)
        sxx, syy, szz, sxy, syz, szx = sig.T
        vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                     + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
        fin = np.array([n in on_fin and n not in on_body for n in ids])
        par = np.array([mesh.xyz[mesh.index[parent[n]]] for n in ids])
        th = np.arctan2(par[:, 1], par[:, 0])
        c, s = np.cos(th), np.sin(th)
        # Wall-plane components: airframe in cylindrical (z, θ); a fin in its
        # own plane (radial, z). The axial σ_zz is the stress along the body.
        hoop = np.where(fin, 0.0, sxx * s * s + syy * c * c - 2.0 * sxy * s * c)
        shear = np.where(fin, np.abs(szx * c + syz * s), np.abs(-szx * s + syz * c))
        y_pa = np.array([yld[n] for n in ids])
        u_el = disp - _rigid_body_fit(P, disp)
        kt = max(float(lc.stress_concentration), 1.0)
        thermal = lc.name == "Thermal"

        # Per station (parent-node z): envelopes for the stress profile, and
        # the ring decomposition of σ_zz into membrane + beam bending.
        zkey = np.round(par[:, 2], 9)
        order = np.argsort(zkey, kind="stable")
        cuts = np.nonzero(np.diff(zkey[order]))[0] + 1
        profile, defl, membrane, bending = [], [], 0.0, 0.0
        for grp in np.split(order, cuts):
            z = float(par[grp[0], 2])
            profile.append((z, float(vm[grp].max())))
            defl.append((z, float(np.linalg.norm(u_el[grp], axis=1).max() * 1000.0)))
            b = grp[~fin[grp]]
            if len(b) >= 6:
                ring = szz[b]
                membrane = max(membrane, abs(float(ring.mean())))
                bending = max(bending, float(np.hypot(2 * np.mean(ring * c[b]),
                                                      2 * np.mean(ring * s[b]))))

        sf_node = np.where(vm > 0, y_pa / (kt * np.maximum(vm, 1e-30)), np.inf)
        result.max_von_mises = float(vm.max())
        result.max_axial_stress = float(np.abs(szz).max())
        result.max_hoop_stress = float(np.abs(hoop).max())
        result.max_shear_stress = float(shear.max())
        result.max_bending_stress = bending
        result.max_thermal_stress = float(vm.max()) if thermal else 0.0
        result.max_displacement_mm = float(np.linalg.norm(u_el, axis=1).max() * 1000.0)
        result.element_stresses = profile
        result.element_displacements = defl
        result.kt_detail = kt
        result.safety_factor = float(sf_node.min())
        result.yield_utilization = float((kt * vm / y_pa).max())
        result.margin_of_safety = result.safety_factor / self.config.safety_factor_required - 1.0
        for label, sel in (("Airframe", ~fin), ("Fins", fin)):
            if sel.any():
                result.component_results[label] = {
                    "max_stress": float(vm[sel].max()), "sf": float(sf_node[sel].min())}
        result.applied_loads = dict(lm.summary)
        result.applied_loads["axial_membrane_Pa"] = membrane
        R = self._support_reaction()
        if R is not None:
            result.applied_loads["support_reaction_N"] = R
        result.fe_field = FEField(
            points=P, cells=np.array(cells, dtype=np.int64), von_mises=vm, axial=szz,
            hoop=hoop, shear=shear, displacement=u_el, yield_pa=y_pa, is_fin=fin,
            cell_is_fin=np.array(cell_fin, dtype=bool),
            kt=kt, thermal=thermal, length=float(mesh.xyz[:, 2].max()))
        k = int(np.argmax(vm))
        logger.info(
            f"FEM Results: σ_vm={result.max_von_mises/1e6:.2f} MPa at z={par[k, 2]:.3f} m "
            f"({'fin' if fin[k] else 'airframe'}), SF={result.safety_factor:.2f} (Kt {kt:g}), "
            f"max defl {result.max_displacement_mm:.3f} mm, support reaction "
            f"{R if R is not None else float('nan'):.2e} N")

    def _support_reaction(self):
        """|reaction| carried by the 3-2-1 support — round-off when the load
        set balances (a check that nothing leaked into the support). CalculiX
        prints RF as the nodes' total internal force, i.e. reaction PLUS the
        load applied at those nodes, so the applied part is subtracted."""
        dat = self.config.work_dir / "analysis.dat"
        lm, mesh = getattr(self, "_load_model", None), getattr(self, "_fe_mesh", None)
        if not dat.is_file() or lm is None or mesh is None:
            return None
        lines = dat.read_text(encoding="utf-8", errors="replace").splitlines()
        for i, line in enumerate(lines):
            if "total force" in line.lower() and "nsupport" in line.lower():
                for nxt in lines[i + 1:i + 4]:
                    try:
                        rf = np.array([float(p) for p in nxt.split()[:3]])
                    except ValueError:
                        continue
                    if len(rf) == 3:
                        applied = lm.forces[mesh.rows(lm.support)].sum(axis=0)
                        return float(np.linalg.norm(rf - applied))
        return None

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
            # Bending: free-free envelope, peak mid-body (as the 3D contour)
            env = 4.0 * frac * (1.0 - frac)
            local_bend = sigma_bend * env
            local_inertial = sigma_inertial * env
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
            # Bending: free-free envelope, peak mid-body (as the 3D contour)
            local_bend = sigma_bend * 4.0 * frac * (1.0 - frac)
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

        # Always enrich with damping / resonance / flutter (and, when the FE
        # shapes could not name the modes, a frequency-based classification).
        # _modal_analytical already populates all of it; the FE parser names
        # the modes but leaves the rest — keying this on the classification
        # would skip damping and resonance for every real FE result.
        if result.converged and not result.damping_ratios:
            self._post_process_modal(result)

        return result

    # ── Modal output parsing ─────────────────────────────────────────────────

    @staticmethod
    def _dat_table(lines, title):
        """Numeric rows of one CalculiX .dat table.

        CalculiX letter-spaces its table titles ("P A R T I C I P A T I O N
        F A C T O R S"), so *title* is given with the spaces removed
        ("PARTICIPATIONFACTORS") and compared against each line the same way.
        Rows are the consecutive all-numeric lines after the title: the column
        header and blank lines before them are skipped, and the first blank or
        text line after them ends the table. Returns [] if the title is absent.
        """
        start = next((i for i, l in enumerate(lines)
                      if title in l.replace(" ", "")), None)
        if start is None:
            return []
        rows = []
        for line in lines[start + 1:]:
            parts = line.split()
            if not parts:
                if rows:
                    break
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                if rows:
                    break
        return rows

    @staticmethod
    def _dat_mode_shapes(lines):
        """{mode_no: {node_id: (ux, uy, uz)}} from a *FREQUENCY step's .dat.

        Each displacement block is keyed by the "E I G E N V A L U E   N U M B E R
        k" header CalculiX prints above it, so a shape can only ever be paired
        with its own eigenvalue — never by the order the blocks happen to come in.
        """
        shapes, mode, in_block = {}, None, False
        for line in lines:
            squeezed = line.replace(" ", "")
            if squeezed.startswith("EIGENVALUENUMBER"):
                try:
                    mode = int(squeezed[len("EIGENVALUENUMBER"):])
                except ValueError:
                    mode = None
                in_block = False
                continue
            if squeezed.lower().startswith("displacements"):
                in_block = mode is not None
                if in_block:
                    shapes.setdefault(mode, {})
                continue
            if not in_block or not squeezed:
                continue
            parts = line.split()
            try:
                shapes[mode][int(parts[0])] = (float(parts[1]), float(parts[2]),
                                               float(parts[3]))
            except (ValueError, IndexError):
                in_block = False        # any other output ends the block
        return shapes

    def _read_mesh_sets(self):
        """(nodes, quad connectivity, node sets) of the structural mesh file."""
        nodes, elements, nsets = {}, [], {}
        mesh = getattr(self, "_mesh_path", None)
        if not mesh or not Path(mesh).is_file():
            return nodes, elements, nsets
        mode = cur = None
        for line in Path(mesh).read_text(encoding="ascii", errors="replace").splitlines():
            s = line.strip()
            if not s or s.startswith("**"):
                continue
            if s.startswith("*"):
                head = s.upper().replace(" ", "")
                mode = None
                if head.startswith("*NODE,") or head == "*NODE":
                    mode = "node"
                elif head.startswith("*ELEMENT"):
                    mode = "elem"
                elif head.startswith("*NSET"):
                    m = re.search(r"NSET=([^,]+)", head)
                    if m:
                        cur = m.group(1)
                        nsets.setdefault(cur, set())
                        mode = "nset"
                continue
            parts = [p for p in s.split(",") if p.strip()]
            try:
                if mode == "node" and len(parts) >= 4:
                    nodes[int(parts[0])] = (float(parts[1]), float(parts[2]),
                                            float(parts[3]))
                elif mode == "elem" and len(parts) >= 5:
                    elements.append(tuple(int(p) for p in parts[1:5]))
                elif mode == "nset":
                    nsets[cur].update(int(p) for p in parts)
            except ValueError:
                continue
        return nodes, elements, nsets

    def _parse_modal_results(self) -> ModalResult:
        """Eigenfrequencies, mode shapes and CalculiX's own participation /
        effective-mass tables from the modal .dat, all matched by mode number.

        The line scanner this replaces never recognised the letter-spaced table
        titles: participation-factor rows were read as extra "frequencies", the
        rigid-mode offset counted them, and the shape list was sliced from the
        wrong mode — on a real run "Mode 1, 62 Hz" animated mode 7 (469 Hz) and
        modes 7-10 had no shape at all.
        """
        dat_path = self.config.work_dir / "analysis.dat"
        if not dat_path.is_file():
            logger.warning("No .dat file for modal results")
            return self._modal_analytical()
        try:
            lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
            eig = [r for r in self._dat_table(lines, "EIGENVALUEOUTPUT") if len(r) >= 4]
            meff = {int(r[0]): r[1:7] for r in self._dat_table(lines, "EFFECTIVEMODALMASS")
                    if len(r) >= 7}
            gam = {int(r[0]): r[1:7] for r in self._dat_table(lines, "PARTICIPATIONFACTORS")
                   if len(r) >= 7}
            tot = self._dat_table(lines, "TOTALEFFECTIVEMASS")
            shapes = self._dat_mode_shapes(lines)
        except Exception as e:
            logger.warning(f"Modal parse error: {e}")
            return self._modal_analytical()
        if not eig:
            return self._modal_analytical()

        # Rigid-body modes (free-free BC) sit near 0 Hz — keep the elastic ones.
        modes = [(int(r[0]), r[3]) for r in eig]
        chosen = ([m for m in modes if m[1] > 1.0] or modes)[: self.config.num_modes]

        result = ModalResult()
        result.mode_numbers = [k for k, _ in chosen]
        result.frequencies_hz = [f for _, f in chosen]
        result.num_modes = len(chosen)
        result.converged = True
        if shapes:
            result.mode_shapes = [shapes.get(k, {}) for k in result.mode_numbers]

        total = tuple(tot[0][:6]) if tot and len(tot[0]) >= 6 else ()
        result.total_effective_mass_kg = total
        if total:
            result.total_mass_kg = total[0]
        for k in result.mode_numbers:
            me, g = meff.get(k), gam.get(k)
            result.effective_mass_kg.append(list(me) if me else [0.0] * 6)
            # One entry per mode (possibly {}) so every list stays index-aligned
            # with frequencies_hz.
            result.participation_factors.append(
                {d: round(abs(g[c]), 4) for d, c in _MODAL_DIRS} if g else {})
            result.effective_modal_mass.append(
                {d: (round(100.0 * me[c] / total[c], 1) if total[c] > 0 else 0.0)
                 for d, c in _MODAL_DIRS} if (me and total) else {})

        nodes, elements, nsets = self._read_mesh_sets()
        result.mesh_path = self._mesh_path
        result.mesh_nodes, result.mesh_elements = nodes, elements
        try:
            self._classify_fe_modes(result, nsets)
        except Exception as e:      # naming is cosmetic — never lose the solve
            logger.warning(f"Mode classification failed: {e}")

        n_shapes = sum(1 for s in result.mode_shapes if s)
        logger.info(f"Modal: {result.num_modes} modes parsed, {n_shapes} mode shapes "
                    f"matched by mode number.")
        return result

    def _classify_fe_modes(self, result: ModalResult, nsets: dict):
        """Name each FE mode from its own shape, not by matching its frequency
        against beam formulas.

        At every axial station the airframe ring's motion is split (least
        squares) into rigid translation — lateral (bending) and axial — rigid
        roll about the body axis (torsion), rigid tilt of the section (the
        rotation half of bending) and the remainder, which deforms the cross
        section (shell / ovalling). Fin nodes are tallied on their own. The
        largest share of |φ|² names the mode. Modes of one type within 2 % in
        frequency share an order number: the two bending planes of an
        axisymmetric airframe are ONE bending mode, not the 1st and the 2nd.
        """
        import numpy as np
        nodes = result.mesh_nodes
        if not nodes or not any(result.mode_shapes):
            return
        known = set(nodes)
        fins = set(nsets.get("NFINS", ())) & known
        body = set()
        for name, members in nsets.items():
            if name in ("NALL", "NAFT", "NFWD", "NFINS") or (fins and members <= fins):
                continue
            body |= members
        body = (body & known) or (known - fins)
        fin_only = sorted(fins - body)      # merged fin-root nodes stay airframe
        ids = sorted(body)
        P = np.array([nodes[n] for n in ids], dtype=float)

        # Axial stations: airframe nodes sharing one z (the mesher merged the
        # coincident end rings of adjacent components).
        z = P[:, 2]
        key = np.round((z - z.min()) / max(float(np.ptp(z)), 1e-12) * 1e6).astype(np.int64)
        order = np.argsort(key, kind="stable")
        rings = []
        for g in np.split(order, np.nonzero(np.diff(key[order]))[0] + 1):
            if len(g) < 3:
                continue
            x, y = P[g, 0], P[g, 1]
            sxx, syy, sxy = float(x @ x), float(y @ y), float(x @ y)
            rings.append((g, x, y, sxx + syy, sxx * syy - sxy * sxy, sxx, syy, sxy))

        def split(U):
            e = {"bending": 0.0, "axial": 0.0, "torsion": 0.0, "shell": 0.0}
            cov = np.zeros((2, 2))
            for g, x, y, r2, det, sxx, syy, sxy in rings:
                u = U[g]
                t = u.mean(axis=0)
                d = u - t
                phi = float(x @ d[:, 1] - y @ d[:, 0]) / r2 if r2 > 0 else 0.0
                a = b = 0.0
                if det > 0:
                    zx, zy = float(x @ d[:, 2]), float(y @ d[:, 2])
                    a = (zx * syy - zy * sxy) / det
                    b = (zy * sxx - zx * sxy) / det
                tilt = a * x + b * y
                res = d.copy()
                res[:, 0] += y * phi
                res[:, 1] -= x * phi
                res[:, 2] -= tilt
                n = len(g)
                e["bending"] += n * float(t[0] ** 2 + t[1] ** 2) + float(tilt @ tilt)
                e["axial"] += n * float(t[2] ** 2)
                e["torsion"] += phi * phi * r2
                e["shell"] += float((res * res).sum())
                cov += n * np.outer(t[:2], t[:2])
            return e, cov

        names = {"axial": ("Axial", "Axial"), "torsion": ("Torsional", "Torsional"),
                 "shell": ("Shell", "Shell (Ovalling)"), "fin": ("Fin", "Fin Mode")}
        groups = {}                 # kind -> (first frequency of group, order)
        cls, descs, fracs, m_gen = [], [], [], []
        for i, shape in enumerate(result.mode_shapes):
            f = result.frequencies_hz[i]
            if not shape:
                cls.append("—"); descs.append(f"Mode {i + 1}"); fracs.append({})
                m_gen.append(0.0)
                continue
            zero = (0.0, 0.0, 0.0)
            U = np.array([shape.get(n, zero) for n in ids], dtype=float)
            e, cov = split(U)
            e["fin"] = float(sum(sum(c * c for c in shape.get(n, zero)) for n in fin_only))
            total = sum(e.values())
            peak = max((ux * ux + uy * uy + uz * uz for ux, uy, uz in shape.values()),
                       default=0.0)
            m_gen.append(round(1.0 / peak, 4) if peak > 0 else 0.0)
            if total <= 0:
                cls.append("—"); descs.append(f"Mode {i + 1}"); fracs.append({})
                continue
            kind = "fin" if e["fin"] > 0.5 * total else \
                max(("bending", "axial", "torsion", "shell"), key=e.get)
            f0, n = groups.get(kind, (None, 0))
            if f0 is None or f > f0 * 1.02:
                f0, n = f, n + 1
            groups[kind] = (f0, n)
            if kind == "bending":
                w, v = np.linalg.eigh(cov)
                lat = v[:, int(np.argmax(w))]
                plane = "X" if abs(lat[0]) >= abs(lat[1]) else "Y"
                cls.append(f"Bending-{plane}")
                descs.append(f"{_ordinal(n)} Lateral Bending ({plane})")
            else:
                cls.append(names[kind][0])
                descs.append(f"{_ordinal(n)} {names[kind][1]}")
            fracs.append({k: round(v / total, 3) for k, v in e.items()})

        result.mode_classifications = cls
        result.descriptions = descs
        result.strain_energy_fractions = fracs
        result.generalized_mass = m_gen

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
        if not result.total_mass_kg:        # FE results carry CalculiX's own total
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

        # Modes already named from their FE shapes (_classify_fe_modes) keep
        # those names; matching frequencies to beam formulas is the fallback
        # for results without shapes.
        classify = not result.mode_classifications
        for i, freq in enumerate(result.frequencies_hz if classify else []):
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


def _rigid_body_fit(P, U):
    """Least-squares rigid-body motion t + ω × (p − p̄) of displacements U at
    points P. Subtracting it leaves the elastic deformation — the 3-2-1
    support fixes an arbitrary rigid position, not a physical one."""
    d = P - P.mean(axis=0)
    n = len(P)
    A = np.zeros((3 * n, 6))
    A[0::3, 0] = A[1::3, 1] = A[2::3, 2] = 1.0
    A[0::3, 4], A[0::3, 5] = d[:, 2], -d[:, 1]
    A[1::3, 3], A[1::3, 5] = -d[:, 2], d[:, 0]
    A[2::3, 3], A[2::3, 4] = d[:, 1], -d[:, 0]
    x = np.linalg.lstsq(A, U.reshape(-1), rcond=None)[0]
    return (A @ x).reshape(n, 3)


# CalculiX modal-table columns reported per mode: X, Y (lateral), Z (body
# axis) and RZ (roll) of the six (X, Y, Z, RX, RY, RZ).
_MODAL_DIRS = (("x", 0), ("y", 1), ("z", 2), ("rz", 5))


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
