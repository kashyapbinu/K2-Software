"""
K2 AeroSim — CFD Solver Abstract Base
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

from core.paths import user_data_dir

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

    # ── Prism boundary layer ─────────────────────────────────────────────────
    # Off by default, but REACHABLE — which it was not. cfd/meshing.py has had
    # a working prism path since 2026-08-19 (measured on the finned test rocket:
    # y+ median 0.41 with 100% of wall points below 1, against ~3500 on the
    # tet-only mesh) and nothing outside that module could switch it on: there
    # was no config field, generate_mesh() never passed the argument, and the UI
    # had no control. Every mesh the product shipped was tet-only, so every
    # viscous run reported a skin friction that was not physical.
    #
    # Default stays False until the AGARD-B, ONERA M6 and cone benchmarks have
    # been re-run on prism meshes; the failure modes are loud (the mesher raises
    # rather than falling back) so an opt-in is safe.
    bl_prisms: bool = False
    boundary_layer_layers: int = 15     # prism layers near wall
    boundary_layer_growth: float = 1.2  # growth rate
    # First-layer height (m). None = the mesher picks one from a y+ target.
    bl_first_height: Optional[float] = None
    # STL carrier resolution for the reparametrised extrusion source, and a
    # ceiling on how far the nose apex may be blunted to let the stack wrap it.
    # Both None = mesher defaults. See cfd/meshing.py for why these exist.
    bl_tessellation: Optional[float] = None
    bl_max_apex_radius: Optional[float] = None

    # Advanced mesh control (override presets)
    custom_wall_size: float | None = None       # element size near wall (m), overrides refinement preset
    target_element_count: int | None = None     # target total elements, auto-computes sizes

    # Solver
    max_iterations: int = 5000
    convergence_tolerance: float = 1e-6
    # "Euler" | "Laminar" | "SA" | "SST". SU2 has no k-epsilon model, so there is
    # no "KE" option — an unrecognised value silently falls back to SST.
    turbulence_model: str = "SST"
    # Hybrid polar mode: solve inviscid (turbulence_model="Euler") and add a
    # flat-plate skin-friction build-up to Cd per sweep point. Avoids the
    # spurious viscous body lift + inflated pressure drag of wall-unresolved
    # RANS on the tet-only mesh (y+ >> 1, no prism layers, no wall functions).
    euler_analytic_friction: bool = False
    # Parallelism for SU2_CFD. 0 = auto (cores - 1, leaving one for the UI).
    # The BUNDLED SU2 is an OpenMP build with no MPI, so in practice this sets
    # OMP_NUM_THREADS on a single process — it is a thread count, not a rank
    # count. If an MPI-built SU2 and an mpiexec/mpirun are both found it is used
    # as a rank count instead; run() probes for that and falls back to the
    # single-process OpenMP path when the probe fails.
    n_cores: int = 0

    # Paths (populated at runtime). Per-user writable in a frozen build; resolves
    # to repo/cfd_run in a source run (unchanged dev behavior).
    work_dir: Path = field(default_factory=lambda: user_data_dir("cfd_run"))
    geometry_stl: Optional[Path] = None   # filled by geometry exporter
    geometry_dict: Optional[dict] = None  # exact dims from extract_cfd_geometry()

    # ── External CAD mode ────────────────────────────────────────────────────
    # When external_cad is set the parametric rocket is bypassed: the imported
    # body itself is subtracted from the wind tunnel (cfd.meshing.
    # build_external_cad_mesh) and geometry_dict is ignored.
    external_cad: Optional[Path] = None   # .step/.stp/.iges/.brep/.stl/.obj/.ply
    flow_axis: str = "auto"               # model axis mapped to +X: auto|x|y|z
    cad_info: Optional[dict] = None       # CADInfo.as_dict() from analyze_cad()
    # Unit the CAD file's bare numbers are in: auto|m|cm|mm|in|ft. The model
    # is scaled to metres at import; without this a millimetre STEP is solved
    # as a 1000x body and every coefficient is meaningless.
    cad_units: str = "auto"
    # Repair dirty CAD by rebuilding it as a distance-field wrap. ALTERS the
    # geometry: offsets the surface outward, rounds features below the grid
    # spacing and seals internal passages — an outer mold line only. Off by
    # default; the only route that works when a solid self-intersects or an
    # assembly's parts interpenetrate.
    cad_wrap: bool = False
    cad_wrap_resolution: str = "medium"   # coarse | medium | fine
    # Elements gmsh spends per 2*pi of surface curvature, i.e. how hard leading
    # edges, trailing edges and nose tips are refined relative to flat panels.
    # None keeps the per-route default (20 exact B-Rep, 12 re-topologised STL).
    # A curved leading edge of radius r gets elements of about 2*pi*r/N, so a
    # thin wing needs a high N to resolve its LE without refining everywhere.
    cad_curvature_elements: Optional[int] = None
    # Force-coefficient references. An arbitrary body has no "body diameter" to
    # infer them from, so they default to the measured frontal area and
    # flow-wise bbox extent; set these to publish coefficients about a
    # different reference (wing area, mean chord, …). None = auto.
    ref_area_override: Optional[float] = None      # [m²]
    ref_length_override: Optional[float] = None    # [m]


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
    # The conditions this result was actually solved at. Carried on the result
    # so exports and annotations cannot drift: the UI unlocks its spin boxes as
    # soon as a run finishes, and anything reading them afterwards records
    # whatever the user has since typed against the old coefficients.
    altitude_m: float = 0.0
    angle_of_attack_deg: float = 0.0

    # ── Drag decomposition ───────────────────────────────────────────────────
    # The total is cd = cd_pressure + cd_friction (both integrated off the wall
    # solution in the wind axis). cd_base and cd_wave are COMPONENTS OF
    # cd_pressure, not additional terms, and must not be summed with the two
    # above. Both come from the same wall integral (cfd/drag_decomposition.py):
    #   cd_base = the integral restricted to rearward-facing cells
    #   cd_forebody_pressure = cd_pressure - cd_base   (exact, at any AoA)
    #   cd_wave = cd_forebody_pressure at M >= 0.8, else 0
    # Oswatitsch entropy production is implemented too but is a mesh-quality
    # DIAGNOSTIC only — logged at DEBUG, never stored in any field here.
    cd_pressure: float = 0.0     # Pressure drag coefficient (integrated)
    cd_friction: float = 0.0     # Skin friction drag coefficient (integrated)
    cd_base: float = 0.0         # Base drag — pressure integral over the aft face
    cd_forebody_pressure: float = 0.0   # cd_pressure - cd_base (exact)
    cd_wave: float = 0.0         # Wave drag = forebody pressure drag at M >= 0.8, else 0
    base_area_m2: float = 0.0    # Projected area the base integral covered
    drag_decomposition_method: str = ""   # provenance of cd_wave

    # Force components (dimensional, Newtons)
    force_axial: float = 0.0     # Axial force (drag direction)
    force_normal: float = 0.0    # Normal force (lift direction)

    # Flow conditions (stored for display)
    reynolds: float = 0.0        # Reynolds number
    dynamic_pressure: float = 0.0  # q∞ (Pa)
    yplus_mean: float = 0.0      # Mean wall y+ (viscous runs; 0 = unknown/inviscid)
    ref_length: float = 0.0      # Characteristic length (m)
    turbulence_model: str = ""   # Active turbulence model name
    solver_name: str = "SU2"     # Solver backend name

    # Convergence
    converged: bool = False
    iterations: int = 0
    final_residual: float = 1.0
    # How far the density residual actually fell, in decades, from the first
    # iteration to the last. This is the criterion that means something:
    # REF_DIMENSIONALIZATION is DIMENSIONAL, so the residual's ABSOLUTE value
    # depends on the flow scale rather than the solution quality -- a measured
    # M=0.8 case started at rms[Rho]=-3.04 and rms[RhoE]=+2.44, which makes a
    # fixed -6 floor mean "3 decades" for one equation and "8 decades" for
    # another. Reported so a run's convergence can be judged rather than
    # assumed.
    residual_drop_decades: float = 0.0
    # Why the run was accepted (or not) -- shown in the UI beside the flag, so
    # "Converged: Yes" is never the whole story.
    convergence_note: str = ""

    # ── Trustworthiness of the wall-dependent quantities ─────────────────────
    # A tet-only mesh puts the first cell far outside the range any turbulence
    # model can integrate to the wall, and there is no wall model behind it
    # (see cfd/solvers/su2_solver.py). Skin friction is then not small, it is
    # absent -- measured Cf median 3.2e-6 against a flat-plate 1.9e-3, i.e.
    # ~600x low, on a shipped SST solution. The result carries that verdict so
    # the UI and the exports do not have to re-derive it, and cannot forget to.
    wall_resolved: bool = True          # False => cd_friction is not physical
    wall_warning: str = ""              # human-readable reason, "" when fine

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
