"""
K2 Aerospace — CFD Solver Abstract Base
========================================
Defines the CFDSolver interface and data classes.
All concrete solver implementations (SU2, OpenFOAM, etc.)
must subclass CFDSolver and implement these methods.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("K2.CFD.Base")


# ── Atmosphere helper (ISA Standard Atmosphere) ─────────────────────────────

def isa_conditions(altitude_m: float) -> tuple[float, float, float]:
    """International Standard Atmosphere.
    Returns (pressure_Pa, temperature_K, density_kg_m3) at a given altitude.
    Standard layered model, valid 0 to 47 km (clamped above); the UI allows
    altitudes up to 50 km.
    """
    import math
    R, g = 287.05, 9.80665
    # (base altitude [m], base temperature [K], base pressure [Pa],
    #  lapse rate [K/m] — 0 means isothermal layer)
    layers = [
        (0.0,     288.15, 101325.0,  -0.0065),   # troposphere
        (11000.0, 216.65, 22632.06,   0.0),      # tropopause (isothermal)
        (20000.0, 216.65, 5474.889,   0.0010),   # lower stratosphere
        (32000.0, 228.65, 868.0187,   0.0028),   # upper stratosphere
        (47000.0, 270.65, 110.9063,   0.0),      # stratopause cap
    ]
    h = max(0.0, min(altitude_m, 47000.0))
    hb, Tb, Pb, L = next(
        layer for layer in reversed(layers) if h >= layer[0]
    )
    if L == 0.0:
        T = Tb
        P = Pb * math.exp(-g * (h - hb) / (R * Tb))
    else:
        T = Tb + L * (h - hb)
        P = Pb * (T / Tb) ** (-g / (L * R))
    rho = P / (R * T)
    return P, T, rho


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class CFDConfig:
    """All inputs needed to run a CFD simulation."""
    # Flow conditions
    mach: float = 0.8
    altitude_m: float = 3000.0
    angle_of_attack_deg: float = 0.0
    sideslip_angle_deg: float = 0.0

    # Mass properties (for static-stability moment transfer).
    # Distance of the center of gravity from the nose tip [m], K2 body frame.
    # When set, parse_results() transfers the nose-tip pitching moment to the CG
    # so the sweep can report dCm/dα about the CG (the true static-stability metric).
    cg_from_nose_m: Optional[float] = None

    # Domain sizing (auto-scaled if 0)
    domain_length_scale: float = 10.0   # multiple of rocket length
    domain_radius_scale: float = 20.0   # multiple of rocket max radius (>=15 for external aero)

    # Mesh quality
    mesh_refinement: str = "medium"     # "coarse" | "medium" | "fine"
    boundary_layer_layers: int = 15     # prism layers near wall
    boundary_layer_growth: float = 1.2  # growth rate

    # Advanced mesh control (override presets)
    custom_wall_size: float | None = None       # element size near wall (m), overrides refinement preset
    target_element_count: int | None = None     # target total elements, auto-computes sizes

    # Solver
    max_iterations: int = 5000
    convergence_tolerance: float = 1e-6
    turbulence_model: str = "SST"       # "Euler" | "Laminar" | "SA" | "SST" | "KE"
    n_cores: int = 0                    # MPI ranks for SU2_CFD. 0 = auto (all cores).
                                        # Requires an MPI-built SU2 + mpiexec on PATH/bin;
                                        # falls back to serial if neither is present.

    # Paths (populated at runtime)
    work_dir: Path = field(default_factory=lambda: Path("cfd_run"))
    geometry_stl: Optional[Path] = None   # filled by geometry exporter
    geometry_dict: Optional[dict] = None  # exact dims from extract_cfd_geometry()


@dataclass
class CFDResult:
    """Results returned after a successful CFD run."""
    cd: float = 0.0               # Drag coefficient (total)
    cl: float = 0.0               # Lift coefficient
    cm: float = 0.0               # Pitching moment coefficient (about nose tip)
    cm_cg: float = 0.0            # Pitching moment coefficient about the CG (static-stability moment)
    cp_location_m: float = 0.0    # CP location from nozzle/tail (CFD x-axis, m)
    cp_from_nose_m: float = 0.0   # CP location from nose tip (m)
    x_cg_m: float = 0.0           # CG location in CFD x-axis (from nozzle, m)
    reference_area_m2: float = 0.0
    v_inf: float = 0.0            # Freestream velocity (m/s) for dimensional display
    mach: float = 0.0             # Mach number for dimensional display

    # Drag decomposition
    cd_pressure: float = 0.0     # Pressure drag coefficient
    cd_friction: float = 0.0     # Skin friction drag coefficient
    cd_base: float = 0.0         # Base drag estimate
    cd_wave: float = 0.0         # Wave drag (supersonic only)

    # Force components (dimensional, Newtons)
    force_axial: float = 0.0     # Axial force (drag direction)
    force_normal: float = 0.0    # Normal force (lift direction)

    # Flow conditions (stored for display)
    reynolds: float = 0.0        # Reynolds number
    dynamic_pressure: float = 0.0  # q∞ (Pa)
    ref_length: float = 0.0      # Characteristic length (m)
    turbulence_model: str = ""   # Active turbulence model name
    solver_name: str = "SU2"     # Solver backend name

    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 1.0

    # VTK output paths (for visualization)
    volume_vtk: Optional[Path] = None
    surface_vtk: Optional[Path] = None

    # Full residual history for live plotting [(iter, rms_density), ...]
    residual_history: list = field(default_factory=list)


# ── Abstract Base Solver ──────────────────────────────────────────────────────

class CFDSolver(ABC):
    """Abstract interface that all K2 CFD solver backends must implement.

    Usage pattern:
        solver = SU2Solver(config)
        solver.generate_mesh()
        solver.generate_case()
        for progress in solver.run():   # generator yielding (iter, residual)
            update_ui(progress)
        result = solver.parse_results()
    """

    def __init__(self, config: CFDConfig):
        self.config = config
        self.config.work_dir = Path(config.work_dir)
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback: Optional[Callable[[int, float], None]] = None
        self._log_callback: Optional[Callable[[str], None]] = None

    def set_progress_callback(self, fn: Callable[[int, float], None]):
        """Register a callback(iteration, residual) for live UI updates."""
        self._progress_callback = fn

    def _emit_progress(self, iteration: int, residual: float):
        if self._progress_callback:
            self._progress_callback(iteration, residual)

    def set_log_callback(self, fn: Callable[[str], None]):
        """Register a callback(message) for surfacing solver log lines in the UI."""
        self._log_callback = fn

    def _emit_log(self, message: str):
        if self._log_callback:
            self._log_callback(message)

    @abstractmethod
    def generate_mesh(self) -> Path:
        """Generate the computational mesh. Returns path to mesh file."""
        ...

    @abstractmethod
    def generate_case(self) -> Path:
        """Write the solver configuration file. Returns path to config."""
        ...

    @abstractmethod
    def run(self):
        """Run the solver. Should be a generator yielding (iter, residual) tuples."""
        ...

    @abstractmethod
    def parse_results(self) -> CFDResult:
        """Parse solver output files and return a CFDResult."""
        ...

    @property
    def solver_name(self) -> str:
        return self.__class__.__name__
