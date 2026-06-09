"""
K2 Aerospace — FEM Interface (Orchestrator)
=============================================
High-level API that coordinates meshing, solving, and post-processing.
"""
from __future__ import annotations
import logging
from pathlib import Path
from structures.solvers.base import FEMConfig, FEMResult, ModalResult, ThermalResult, LoadCase
from structures.solvers.ccx_solver import CalculiXSolver
from structures.thermal_analysis import analyze_thermal

logger = logging.getLogger("K2.FEM")


class FEMInterface:
    """Orchestrates structural analysis for K2 Aerospace."""

    def __init__(self, work_dir: Path = None):
        self.work_dir = work_dir or Path("fem_run")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _get_work_dir(self, analysis_type: str) -> Path:
        """Return a per-analysis-type subdirectory to avoid file-locking
        conflicts when multiple CalculiX runs execute concurrently."""
        d = self.work_dir / analysis_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def analyze(self, assembly, load_case: LoadCase = None,
                material_name: str = "Aluminum 6061-T6",
                refinement: str = "medium",
                analysis_type: str = "static",
                cfd_vtk_path: Path = None,
                custom_circum: int | None = None,
                custom_axial_per_cal: int | None = None) -> FEMResult:
        """Run full FEM structural analysis."""
        if load_case is None:
            load_case = LoadCase()
        work = self._get_work_dir("static")
        cfg = FEMConfig(
            material_name=material_name,
            load_case=load_case,
            analysis_type=analysis_type,
            mesh_refinement=refinement,
            work_dir=work,
            assembly=assembly,
            cfd_surface_vtk=cfd_vtk_path,
            custom_circum=custom_circum,
            custom_axial_per_cal=custom_axial_per_cal,
        )
        solver = CalculiXSolver(cfg)
        solver.generate_mesh(assembly)
        solver.generate_case()
        for stage, frac in solver.run():
            pass
        return solver.parse_results()

    def modal_analysis(self, assembly, material_name: str = "Aluminum 6061-T6",
                       num_modes: int = 10, refinement: str = "medium",
                       custom_circum: int | None = None,
                       custom_axial_per_cal: int | None = None) -> ModalResult:
        """Run modal (eigenvalue) analysis for natural frequencies."""
        work = self._get_work_dir("modal")
        cfg = FEMConfig(
            material_name=material_name,
            analysis_type="modal",
            num_modes=num_modes,
            mesh_refinement=refinement,
            work_dir=work,
            assembly=assembly,
            custom_circum=custom_circum,
            custom_axial_per_cal=custom_axial_per_cal,
        )
        solver = CalculiXSolver(cfg)
        solver.generate_mesh(assembly)
        return solver.run_modal()

    def thermal_analysis(self, assembly, mach: float, altitude_m: float,
                         material_name: str = "Aluminum 6061-T6") -> ThermalResult:
        """Run thermal analysis (aerodynamic heating + thermal stress)."""
        return analyze_thermal(assembly, mach, altitude_m, material_name)

    def quick_check(self, assembly, force: float, material_name: str = "Aluminum 6061-T6") -> FEMResult:
        """Quick analytical structural check without full FEM."""
        lc = LoadCase.max_thrust(force, accel_g=5.0)
        cfg = FEMConfig(
            material_name=material_name,
            load_case=lc,
            work_dir=self.work_dir,
            assembly=assembly,
        )
        solver = CalculiXSolver(cfg)
        from structures.solvers.base import get_structural_material
        solver._material = get_structural_material(material_name)
        result = solver._analytical_fallback(FEMResult(), solver._material)
        mat = solver._material
        if result.max_von_mises > 0:
            result.safety_factor = mat.yield_strength / result.max_von_mises
            result.yield_utilization = result.max_von_mises / mat.yield_strength
            # Use config SF requirement for consistency with parse_results()
            result.margin_of_safety = (result.safety_factor / cfg.safety_factor_required) - 1.0
        result.material_name = mat.name
        result.yield_strength = mat.yield_strength
        return result

