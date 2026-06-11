"""
K2 Aerospace — FEM Solver Abstract Base
========================================
Defines the FEMSolver interface and data classes.
All concrete solver implementations (CalculiX, etc.)
must subclass FEMSolver and implement these methods.

Mirrors the CFD solver architecture (cfd/solvers/base.py).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("K2.FEM.Base")


# ── Material Data Class ─────────────────────────────────────────────────────

@dataclass
class StructuralMaterial:
    """Complete material properties for structural analysis."""
    name: str = "Aluminum 6061-T6"
    E: float = 68.9e9           # Young's modulus (Pa)
    nu: float = 0.33            # Poisson's ratio
    G: float = 0.0              # Shear modulus (Pa) — auto-calc from E, nu if 0
    density: float = 2700.0     # kg/m³
    yield_strength: float = 276e6       # Tensile yield (Pa)
    ultimate_strength: float = 310e6    # Ultimate tensile (Pa)
    cte: float = 23.6e-6        # Coefficient of thermal expansion (1/K)
    thermal_conductivity: float = 167.0 # W/(m·K)
    specific_heat: float = 896.0        # J/(kg·K)
    max_service_temp: float = 423.0     # K (150°C for Al 6061)
    emissivity: float = 0.8              # surface emissivity for radiation
    fatigue_endurance_factor: float = 0.4  # S_e / σ_ult ratio

    def __post_init__(self):
        if self.G <= 0:
            self.G = self.E / (2 * (1 + self.nu))


# ── Material Library ─────────────────────────────────────────────────────────

STRUCTURAL_MATERIALS = {
    "Aluminum 6061-T6": StructuralMaterial(
        name="Aluminum 6061-T6", E=68.9e9, nu=0.33, density=2700,
        yield_strength=276e6, ultimate_strength=310e6, cte=23.6e-6,
        thermal_conductivity=167.0, specific_heat=896.0, max_service_temp=423.0,
        emissivity=0.09, fatigue_endurance_factor=0.4,
    ),
    "Carbon Fiber Composite": StructuralMaterial(
        name="Carbon Fiber Composite", E=70e9, nu=0.30, density=1600,
        yield_strength=600e6, ultimate_strength=700e6, cte=2.0e-6,
        thermal_conductivity=5.0, specific_heat=800.0, max_service_temp=473.0,
        emissivity=0.85, fatigue_endurance_factor=0.35,
    ),
    "Fiberglass (G10)": StructuralMaterial(
        name="Fiberglass (G10)", E=18.6e9, nu=0.28, density=1800,
        yield_strength=310e6, ultimate_strength=380e6, cte=14.0e-6,
        thermal_conductivity=0.29, specific_heat=1050.0, max_service_temp=413.0,
        emissivity=0.90, fatigue_endurance_factor=0.35,
    ),
    "Steel 4130": StructuralMaterial(
        name="Steel 4130", E=200e9, nu=0.29, density=7850,
        yield_strength=460e6, ultimate_strength=560e6, cte=11.2e-6,
        thermal_conductivity=42.7, specific_heat=477.0, max_service_temp=700.0,
        emissivity=0.40, fatigue_endurance_factor=0.5,
    ),
    "Titanium Ti-6Al-4V": StructuralMaterial(
        name="Titanium Ti-6Al-4V", E=114e9, nu=0.34, density=4430,
        yield_strength=880e6, ultimate_strength=950e6, cte=8.6e-6,
        thermal_conductivity=6.7, specific_heat=526.0, max_service_temp=600.0,
        emissivity=0.50, fatigue_endurance_factor=0.4,
    ),
    "Kraft Phenolic": StructuralMaterial(
        name="Kraft Phenolic", E=8.0e9, nu=0.35, density=960,
        yield_strength=60e6, ultimate_strength=80e6, cte=30.0e-6,
        thermal_conductivity=0.2, specific_heat=1400.0, max_service_temp=400.0,
        emissivity=0.90, fatigue_endurance_factor=0.25,
    ),
    "Plywood (Birch)": StructuralMaterial(
        name="Plywood (Birch)", E=12.0e9, nu=0.30, density=630,
        yield_strength=40e6, ultimate_strength=60e6, cte=5.0e-6,
        thermal_conductivity=0.17, specific_heat=1200.0, max_service_temp=373.0,
        emissivity=0.90, fatigue_endurance_factor=0.20,
    ),
    "ABS Plastic": StructuralMaterial(
        name="ABS Plastic", E=2.3e9, nu=0.35, density=1050,
        yield_strength=40e6, ultimate_strength=45e6, cte=73.8e-6,
        thermal_conductivity=0.17, specific_heat=1400.0, max_service_temp=358.0,
        emissivity=0.92, fatigue_endurance_factor=0.25,
    ),
}


# Component-library material names (core.components.MATERIALS) → nearest
# structural analogue. Without these, unknown names silently fell back to
# aluminum — e.g. a fiberglass fin got aluminum's shear modulus, over-
# predicting flutter speed ~2×.
_MATERIAL_ALIASES = {
    "Fiberglass": "Fiberglass (G10)",
    "Carbon Fiber": "Carbon Fiber Composite",
    "Aluminum 6061": "Aluminum 6061-T6",
    "Balsa Wood": "Plywood (Birch)",
    "Cardboard": "Kraft Phenolic",
    "Polycarbonate": "ABS Plastic",
}


def get_structural_material(name: str) -> StructuralMaterial:
    """Look up a material (component-library aliases supported);
    falls back to Aluminum 6061-T6."""
    if name in STRUCTURAL_MATERIALS:
        return STRUCTURAL_MATERIALS[name]
    alias = _MATERIAL_ALIASES.get(name)
    if alias is not None:
        return STRUCTURAL_MATERIALS[alias]
    if name:
        logger.warning("Unknown structural material '%s' — using Aluminum 6061-T6", name)
    return STRUCTURAL_MATERIALS["Aluminum 6061-T6"]


# ── Load Case ────────────────────────────────────────────────────────────────

@dataclass
class LoadCase:
    """Defines a structural load case for FEM analysis."""
    name: str = "Max Thrust"
    # Axial
    axial_force: float = 0.0        # N (positive = compression from thrust)
    # Lateral (distributed aerodynamic)
    lateral_force: float = 0.0      # N (from angle-of-attack)
    lateral_distribution: str = "uniform"   # "uniform" | "triangular" | "from_aero"
    # Internal pressure
    internal_pressure: float = 0.0  # Pa (motor chamber or tank)
    # Inertia
    acceleration_g: float = 0.0     # G-load (axial)
    # Thermal
    delta_T: float = 0.0            # Temperature rise above reference (K)
    wall_temp_K: float = 293.15     # Wall temperature (K)
    # Flight condition
    mach: float = 0.0
    altitude_m: float = 0.0
    dynamic_pressure: float = 0.0   # Pa
    angle_of_attack_deg: float = 0.0
    # Recovery-specific
    recovery_shock_g: float = 0.0       # G-load for recovery shock
    dynamic_amplification: float = 1.0  # DAF for transient loads
    stress_concentration: float = 1.0   # Kt at attachment points
    vehicle_mass_kg: float = 5.0        # For recovery force calculation
    # Tensile flag (recovery loads are tensile, not compressive)
    is_tensile: bool = False

    @classmethod
    def max_thrust(cls, thrust: float, accel_g: float = 5.0,
                   internal_pressure: float = 0.0,
                   angle_of_attack_deg: float = 2.0,
                   mach: float = 0.0, altitude_m: float = 0.0) -> "LoadCase":
        """Max thrust load case: compressive axial + hoop + bending from AoA."""
        q_dyn = 0.0
        if mach > 0 and altitude_m >= 0:
            try:
                from cfd.solvers.base import isa_conditions
                import math as _m
                P, T, rho = isa_conditions(altitude_m)
                a = _m.sqrt(1.4 * 287.05 * T)
                V = mach * a
                q_dyn = 0.5 * rho * V ** 2
            except Exception:
                pass
        return cls(
            name="Max Thrust", axial_force=thrust,
            acceleration_g=accel_g, internal_pressure=internal_pressure,
            angle_of_attack_deg=angle_of_attack_deg,
            mach=mach, altitude_m=altitude_m,
            dynamic_pressure=q_dyn, is_tensile=False,
        )

    @classmethod
    def max_q(cls, thrust: float, q_dyn: float, mach: float,
              alt: float, aoa: float = 2.0) -> "LoadCase":
        return cls(name="Max-Q", axial_force=thrust, dynamic_pressure=q_dyn,
                   mach=mach, altitude_m=alt, angle_of_attack_deg=aoa,
                   acceleration_g=3.0)

    @classmethod
    def recovery(cls, vehicle_mass_kg: float = 5.0,
                 shock_g: float = 15.0, daf: float = 1.8,
                 kt: float = 2.5) -> "LoadCase":
        """Recovery shock: tensile axial from parachute deployment."""
        snap_force = vehicle_mass_kg * shock_g * 9.81
        return cls(
            name="Recovery Shock",
            axial_force=snap_force,
            acceleration_g=shock_g,
            recovery_shock_g=shock_g,
            dynamic_amplification=daf,
            stress_concentration=kt,
            vehicle_mass_kg=vehicle_mass_kg,
            is_tensile=True,
            internal_pressure=0.0,  # no motor chamber pressure
        )

    @classmethod
    def thermal(cls, mach: float, alt: float) -> "LoadCase":
        """Aerodynamic heating: thermal expansion stress with non-uniform ΔT."""
        from cfd.solvers.base import isa_conditions
        import math
        P, T, rho = isa_conditions(alt)
        gamma = 1.4
        r_recovery = 0.89  # turbulent recovery factor
        T_recovery = T * (1 + r_recovery * (gamma - 1) / 2 * mach ** 2)
        T_stag = T * (1 + (gamma - 1) / 2 * mach ** 2)
        return cls(
            name="Thermal", mach=mach, altitude_m=alt,
            delta_T=T_recovery - 293.15, wall_temp_K=T_recovery,
            axial_force=0.0, internal_pressure=0.0,
            is_tensile=False,
        )


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class FEMConfig:
    """All inputs needed to run a structural FEM analysis."""
    # Material
    material_name: str = "Aluminum 6061-T6"

    # Load case
    load_case: LoadCase = field(default_factory=LoadCase)

    # Analysis type
    analysis_type: str = "static"   # "static" | "modal" | "buckling" | "thermal"
    num_modes: int = 10             # for modal analysis
    safety_factor_required: float = 2.0

    # Mesh
    mesh_refinement: str = "medium"  # "coarse" | "medium" | "fine"
    element_type: str = "shell"      # "shell" | "beam" | "solid"

    # Advanced mesh control (override presets)
    custom_circum: int | None = None             # circumferential divisions override
    custom_axial_per_cal: int | None = None      # axial divisions per caliber override

    # Paths
    work_dir: Path = field(default_factory=lambda: Path("fem_run"))

    # Assembly reference (populated at runtime)
    assembly: Optional[object] = None

    # CFD load mapping
    cfd_surface_vtk: Optional[Path] = None

    # Modal boundary condition for effective-length factor
    modal_bc: str = 'cantilever'  # 'cantilever'|'pinned-pinned'|'fixed-fixed'|'fixed-pinned'


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class FEMResult:
    """Results returned after a structural FEM analysis."""
    # Stress (Pa)
    max_von_mises: float = 0.0
    max_axial_stress: float = 0.0
    max_hoop_stress: float = 0.0
    max_shear_stress: float = 0.0
    max_bending_stress: float = 0.0
    max_thermal_stress: float = 0.0

    # Displacement
    max_displacement_mm: float = 0.0
    max_rotation_deg: float = 0.0

    # Buckling
    buckling_load_factor: float = 0.0   # λ_crit (>1 = safe)
    buckling_mode: int = 0

    # Safety
    safety_factor: float = 0.0
    margin_of_safety: float = 0.0       # MoS = SF/SF_req - 1
    yield_utilization: float = 0.0      # σ_max / σ_yield

    # Material info
    material_name: str = ""
    yield_strength: float = 0.0

    # Load case identification
    load_case_name: str = ""

    # Convergence
    converged: bool = False
    solver_time_s: float = 0.0

    # Per-element data for visualization
    element_stresses: list = field(default_factory=list)   # [(x, σ_vm), ...]
    element_displacements: list = field(default_factory=list)
    station_temperatures: list = field(default_factory=list)  # [(x, T_K), ...]

    # VTK output for PyVista rendering
    result_vtk: Optional[Path] = None

    # Comprehensive safety assessment (populated post-solve)
    safety_assessment: Optional['SafetyAssessment'] = None

    # Component-level breakdown
    component_results: dict = field(default_factory=dict)
    # {"Nose Cone": {"max_stress": ..., "sf": ...}, "Body Tube": {...}, ...}


@dataclass
class ModalResult:
    """Results from modal (eigenvalue) analysis.
    
    Professional-grade output matching Nastran/Abaqus conventions:
    - Natural frequencies and mode shapes
    - Energy-based mode classification (bending, torsional, axial, coupled)
    - Modal participation factors and effective modal mass
    - Structural damping estimates (material-dependent)
    - Resonance assessment against motor and aero forcing
    - Preliminary fin flutter analysis
    """
    num_modes: int = 0
    frequencies_hz: list = field(default_factory=list)     # Natural frequencies
    mode_shapes: list = field(default_factory=list)        # [{node_id: (dx,dy,dz)}, ...]
    descriptions: list = field(default_factory=list)       # ["1st Lateral Bending", ...]

    # ── Energy-based mode classification ──
    mode_classifications: list = field(default_factory=list)   # ["Bending-Y", "Torsional", "Axial", "Coupled B-T"]
    strain_energy_fractions: list = field(default_factory=list)  # [{"bending": 0.85, "torsion": 0.10, "axial": 0.05}, ...]

    # ── Participation factors & effective modal mass ──
    participation_factors: list = field(default_factory=list)   # [{"x": Γx, "y": Γy, "z": Γz}, ...]
    effective_modal_mass: list = field(default_factory=list)    # [{"x": %, "y": %, "z": %}, ...] as fraction of total
    generalized_mass: list = field(default_factory=list)       # [m_gen_1, m_gen_2, ...] kg
    total_mass_kg: float = 0.0                                  # for normalization

    # ── Damping ──
    damping_ratios: list = field(default_factory=list)     # [ζ₁, ζ₂, ...] (fraction of critical)
    damping_source: str = ""                                # "Material estimate" / "Rayleigh" / etc.

    # ── Resonance assessment ──
    resonance_warnings: list = field(default_factory=list)     # ["Mode 3 near motor 1P (250 Hz)", ...]
    motor_1p_hz: float = 0.0                                    # Motor combustion 1P frequency
    motor_2p_hz: float = 0.0                                    # Motor combustion 2P
    aero_buffet_band: tuple = (0.0, 0.0)                        # (low_Hz, high_Hz) Strouhal buffeting

    # ── Flutter assessment (preliminary) ──
    flutter_assessment: dict = field(default_factory=lambda: {
        "critical_speed_m_s": 0.0,
        "flutter_margin": 0.0,
        "max_flight_speed_m_s": 0.0,
        "verdict": "—",
        "method": "NACA empirical (preliminary)",
    })

    # ── Solver metadata ──
    effective_mass: list = field(default_factory=list)   # Modal effective mass fractions (legacy)
    converged: bool = False
    solver_time_s: float = 0.0
    result_vtk: Optional[Path] = None


@dataclass
class ThermalResult:
    """Results from thermal analysis."""
    # Temperatures
    max_wall_temp_K: float = 0.0
    min_wall_temp_K: float = 0.0
    stagnation_temp_K: float = 0.0
    recovery_temp_K: float = 0.0

    # Thermal stress
    max_thermal_stress: float = 0.0
    thermal_safety_factor: float = 0.0

    # Material limits
    exceeds_service_temp: bool = False
    service_temp_limit_K: float = 0.0

    # Heat flux
    max_heat_flux_W_m2: float = 0.0
    total_heat_input_W: float = 0.0

    # Per-station data
    station_temps: list = field(default_factory=list)    # [(x, T_wall), ...]
    station_stresses: list = field(default_factory=list)  # [(x, σ_thermal), ...]

    converged: bool = False
    solver_time_s: float = 0.0


# ── Abstract Base Solver ──────────────────────────────────────────────────────

class FEMSolver(ABC):
    """Abstract interface that all K2 FEM solver backends must implement.

    Usage pattern (mirrors CFDSolver):
        solver = CalculiXSolver(config)
        solver.generate_mesh(assembly)
        solver.generate_case()
        for progress in solver.run():
            update_ui(progress)
        result = solver.parse_results()
    """

    def __init__(self, config: FEMConfig):
        self.config = config
        self.config.work_dir = Path(config.work_dir)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, fn: Callable[[str, float], None]):
        """Register a callback(stage_name, fraction) for live UI updates."""
        self._progress_callback = fn

    def _emit_progress(self, stage: str, fraction: float):
        if self._progress_callback:
            self._progress_callback(stage, fraction)

    @abstractmethod
    def generate_mesh(self, assembly=None) -> Path:
        """Generate the structural mesh. Returns path to mesh file."""
        ...

    @abstractmethod
    def generate_case(self) -> Path:
        """Write the solver input file. Returns path to input deck."""
        ...

    @abstractmethod
    def run(self):
        """Run the solver. Generator yielding (stage, progress_frac) tuples."""
        ...

    @abstractmethod
    def parse_results(self) -> FEMResult:
        """Parse solver output and return an FEMResult."""
        ...

    @abstractmethod
    def run_modal(self) -> ModalResult:
        """Run modal (eigenvalue) analysis."""
        ...

    @property
    def solver_name(self) -> str:
        return self.__class__.__name__
