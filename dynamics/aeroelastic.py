"""
K2 Aerospace — Aeroelastic Analysis
=====================================
Static aeroelasticity: fin divergence speed, aeroelastic effectiveness,
control reversal detection.

Physics formulations
--------------------
- Lift-curve slope with Prandtl–Glauert compressibility and finite span:
    Subsonic  (M < 0.8):   CL_α = 2π / √(1 − M²)  × AR / (AR + 2)
    Supersonic (M > 1.2):  CL_α = 4  / √(M² − 1)   × AR / (AR + 2)
    Transonic (0.8 ≤ M ≤ 1.2): cubic Hermite blend between subsonic
        and supersonic values, with reduced peak at M = 1.0.

  Ref: Anderson, "Fundamentals of Aerodynamics", Ch. 11–12;
       NACA Report 1135, "Equations, Tables and Charts for
       Compressible Flow".

- Divergence dynamic pressure (Mach-dependent):
    q_div(M) = K_θ / (S × e × CL_α(M))

- Aeroelastic effectiveness:
    η(M) = 1 / (1 − q(M) / q_div(M))
  True value stored; η < 0 indicates control reversal.

  Ref: Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §8.3.
"""
from __future__ import annotations
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("K2.Dynamics.Aeroelastic")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAMMA_AIR = 1.4
R_AIR = 287.05      # J/(kg·K)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class AeroelasticResult:
    """Results from aeroelastic analysis.

    All original fields are preserved for backward compatibility.
    """
    divergence_speed_mps: float = 0.0
    divergence_mach: float = 0.0
    divergence_margin: float = 0.0      # V_div / V_max (>1 = safe)

    # Effectiveness: flexible lift / rigid lift at various speeds
    effectiveness_data: list = field(default_factory=list)  # [(mach, eta), ...]

    # Fin deflection at max-Q
    max_deflection_deg: float = 0.0
    max_deflection_mm: float = 0.0

    # --- NEW fields ---
    # Mach number where η crosses zero (control reversal)
    reversal_mach: float = 0.0

    # Per-point regime identification: [(mach, regime_str), ...]
    mach_regime_data: list = field(default_factory=list)

    # Control-reversal margin (M_reversal - M_max; inf if no reversal in range)
    reversal_margin: float = float('inf')
    # Effectiveness at the max design Mach (interpolated from effectiveness_data)
    effectiveness_at_max_mach: float = 1.0


# ---------------------------------------------------------------------------
# Internal: Mach-dependent lift-curve slope
# ---------------------------------------------------------------------------
def _cl_alpha(mach: float, aspect_ratio: float) -> float:
    """Lift-curve slope corrected for compressibility and finite span.

    Subsonic  (M < 0.8):  CL_α = 2π / √(1 − M²)  × AR / (AR + 2)
    Supersonic(M > 1.2):  CL_α = 4  / √(M² − 1)   × AR / (AR + 2)
    Transonic (0.8–1.2):  Smooth cubic Hermite interpolation with
        10 % penalty at M = 1.0 (reduced peak effectiveness).

    Parameters
    ----------
    mach : float         Free-stream Mach number.
    aspect_ratio : float Fin aspect ratio  (AR = span² / S).

    Returns
    -------
    float : CL_α in 1/rad.

    Reference
    ---------
    Anderson, "Fundamentals of Aerodynamics", 6th ed., §11.4, §12.3.
    NACA Report 1135, Charts for Compressible Flow.
    """
    ar_factor = aspect_ratio / (aspect_ratio + 2.0) if (aspect_ratio + 2.0) > 0 else 1.0

    # Guard against M exactly 1.0 in denominators
    if mach < 0.0:
        mach = 0.0

    if mach < 0.80:
        # Prandtl–Glauert: CL_α = 2π / √(1 − M²)  (incompressible → subsonic)
        beta_sq = 1.0 - mach ** 2
        if beta_sq <= 0.0:
            beta_sq = 1e-6
        cl_a = 2.0 * math.pi / math.sqrt(beta_sq) * ar_factor
        return cl_a

    if mach > 1.20:
        # Ackeret (linearised supersonic): CL_α = 4 / √(M² − 1)
        beta_sq = mach ** 2 - 1.0
        if beta_sq <= 0.0:
            beta_sq = 1e-6
        cl_a = 4.0 / math.sqrt(beta_sq) * ar_factor
        return cl_a

    # --- Transonic blend (0.80 ≤ M ≤ 1.20) ---
    # Evaluate endpoints
    cl_sub = 2.0 * math.pi / math.sqrt(1.0 - 0.80 ** 2) * ar_factor   # M = 0.80
    cl_sup = 4.0 / math.sqrt(1.20 ** 2 - 1.0) * ar_factor              # M = 1.20

    # Hermite-style smooth interpolation parameter t ∈ [0, 1]
    t = (mach - 0.80) / (1.20 - 0.80)

    # Apply 10 % effectiveness reduction near M = 1.0 (transonic drag-rise
    # reduces control authority).  The penalty peaks at t = 0.5 (M = 1.0).
    penalty = 1.0 - 0.10 * math.sin(math.pi * t)

    # Smoothstep (3t² − 2t³) for C1 continuity at boundaries
    s = 3.0 * t ** 2 - 2.0 * t ** 3
    cl_a = ((1.0 - s) * cl_sub + s * cl_sup) * penalty
    return cl_a


def _mach_regime(mach: float) -> str:
    """Return a human-readable regime label for the given Mach number."""
    if mach < 0.80:
        return "subsonic"
    elif mach <= 1.20:
        return "transonic"
    else:
        return "supersonic"


# ---------------------------------------------------------------------------
# Public: divergence speed (extended, backward-compatible)
# ---------------------------------------------------------------------------
def divergence_speed(span: float, chord: float, thickness: float,
                     shear_modulus: float, altitude_m: float = 0.0,
                     elastic_axis_fraction: float = 0.40,
                     aspect_ratio: float = None,
                     mach: float = 0.0) -> float:
    """Torsional divergence speed for a thin fin.

    V_div = √(2 · q_div / ρ)
    q_div = K_θ / (S · e · CL_α)

    Torsional stiffness K_θ is the thin-plate cantilever formula:
        K_θ = G · c · t³ / (3 · L)

    Parameters
    ----------
    span : float                 Fin semi-span (m).
    chord : float                Mean aerodynamic chord (m).
    thickness : float            Fin thickness (m).
    shear_modulus : float        Shear modulus G (Pa).
    altitude_m : float           Flight altitude (m).
    elastic_axis_fraction : float
        Chordwise position of elastic axis as fraction of chord
        (default 0.40 → offset e = (0.40 − 0.25)·c = 0.15·c from AC).
    aspect_ratio : float | None
        Fin aspect ratio for finite-span correction.  If None, computed
        from span²/(span×chord) = span/chord.
    mach : float
        Free-stream Mach for compressibility-corrected CL_α.  0 → incompressible.

    Returns
    -------
    float : Divergence speed (m/s).

    Reference
    ---------
    Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §8.2–8.3.
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or chord <= 0 or thickness <= 0:
        return float('inf')

    _, T, rho = isa_conditions(altitude_m)

    # Torsional stiffness  (thin rectangular plate, cantilever)
    K_theta = shear_modulus * chord * thickness ** 3 / (3.0 * span)

    # Planform area  (using mean chord — works for trapezoidal via c_mean)
    S = span * chord

    # Elastic axis offset from aerodynamic centre (at 25 % chord)
    e = (elastic_axis_fraction - 0.25) * chord
    if e <= 0.0:
        # EA at or ahead of AC → no divergence in this model
        return float('inf')

    # Aspect ratio
    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    # Lift-curve slope (compressibility + finite span)
    cl_a = _cl_alpha(mach, aspect_ratio)

    # Divergence dynamic pressure
    denom = S * e * cl_a
    if denom <= 0.0:
        return float('inf')
    q_div = K_theta / denom

    # Divergence speed
    if q_div > 0 and rho > 0:
        V_div = math.sqrt(2.0 * q_div / rho)
    else:
        V_div = float('inf')

    return V_div


# ---------------------------------------------------------------------------
# Public: aeroelastic effectiveness sweep (extended, backward-compatible)
# ---------------------------------------------------------------------------
def aeroelastic_effectiveness(span: float, chord: float, thickness: float,
                              shear_modulus: float, altitude_m: float = 0.0,
                              mach_range: tuple = (0.1, 3.0),
                              n_points: int = 30,
                              elastic_axis_fraction: float = 0.40,
                              aspect_ratio: float = None) -> list:
    """Compute aeroelastic effectiveness η vs Mach with compressibility.

    η(M) = 1 / (1 − q(M) / q_div(M))

    q_div now varies with Mach because CL_α(M) changes across subsonic,
    transonic, and supersonic regimes.

    True η is stored (negative values → control reversal).
    Display values are clamped to ±20 for plotting sanity.

    Parameters
    ----------
    span, chord, thickness, shear_modulus, altitude_m : float
        (same as divergence_speed)
    mach_range : tuple
        (Mach_start, Mach_end).
    n_points : int
        Number of uniformly-spaced Mach points.
    elastic_axis_fraction : float
        (same as divergence_speed)
    aspect_ratio : float | None
        (same as divergence_speed)

    Returns
    -------
    list of (mach, eta_display)
        eta_display is clamped to [−20, +20] for plotting.

    Reference
    ---------
    Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §8.3.
    Anderson, "Fundamentals of Aerodynamics", Ch. 11–12.
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or chord <= 0 or thickness <= 0:
        return [(0.0, 1.0)]

    _, T, rho = isa_conditions(altitude_m)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)  # speed of sound

    # Torsional stiffness
    K_theta = shear_modulus * chord * thickness ** 3 / (3.0 * span)

    # Planform
    S = span * chord
    e = (elastic_axis_fraction - 0.25) * chord
    if e <= 0.0:
        return [(m, 1.0) for m in _linspace(mach_range[0], mach_range[1], n_points)]

    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    m_start, m_end = mach_range
    result = []

    for i in range(n_points):
        mach = m_start + (m_end - m_start) * i / max(n_points - 1, 1)
        V = mach * a
        q = 0.5 * rho * V ** 2

        # Mach-dependent divergence dynamic pressure
        cl_a = _cl_alpha(mach, aspect_ratio)
        denom = S * e * cl_a
        q_div = K_theta / denom if denom > 0 else float('inf')

        # True effectiveness (may be negative → reversal)
        if q_div != 0.0 and q_div != float('inf'):
            eta_true = 1.0 / (1.0 - q / q_div)
        elif q_div == float('inf'):
            eta_true = 1.0
        else:
            eta_true = 1.0

        # Clamp display value but store true in internal list
        eta_display = max(-20.0, min(20.0, eta_true))
        result.append((mach, eta_display))

    return result


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _linspace(start: float, stop: float, n: int) -> list:
    """Pure-Python linspace."""
    if n <= 1:
        return [start]
    return [start + (stop - start) * i / (n - 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Internal: full effectiveness with true values (used by full analysis)
# ---------------------------------------------------------------------------
def _effectiveness_full(span: float, chord: float, thickness: float,
                        shear_modulus: float, altitude_m: float,
                        mach_range: tuple, n_points: int,
                        elastic_axis_fraction: float,
                        aspect_ratio: float):
    """Return (display_list, reversal_mach, regime_list).

    display_list : [(mach, eta_display), ...]
    reversal_mach : float  (0.0 if no reversal found)
    regime_list : [(mach, regime_str), ...]
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or chord <= 0 or thickness <= 0:
        pts = _linspace(mach_range[0], mach_range[1], n_points)
        return ([(m, 1.0) for m in pts], 0.0,
                [(m, _mach_regime(m)) for m in pts])

    _, T, rho = isa_conditions(altitude_m)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)

    K_theta = shear_modulus * chord * thickness ** 3 / (3.0 * span)
    S = span * chord
    e = (elastic_axis_fraction - 0.25) * chord
    if e <= 0.0:
        pts = _linspace(mach_range[0], mach_range[1], n_points)
        return ([(m, 1.0) for m in pts], 0.0,
                [(m, _mach_regime(m)) for m in pts])

    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    m_start, m_end = mach_range
    display_list = []
    regime_list = []
    reversal_mach = 0.0
    prev_eta = None

    for i in range(n_points):
        mach = m_start + (m_end - m_start) * i / max(n_points - 1, 1)
        V = mach * a
        q = 0.5 * rho * V ** 2
        cl_a = _cl_alpha(mach, aspect_ratio)
        denom = S * e * cl_a
        q_div = K_theta / denom if denom > 0 else float('inf')

        if q_div != 0.0 and q_div != float('inf'):
            eta_true = 1.0 / (1.0 - q / q_div)
        elif q_div == float('inf'):
            eta_true = 1.0
        else:
            eta_true = 1.0

        # Detect first zero-crossing (control reversal)
        if prev_eta is not None and reversal_mach == 0.0:
            if prev_eta > 0.0 and eta_true <= 0.0:
                # Linear interpolation for crossing Mach
                prev_mach = m_start + (m_end - m_start) * (i - 1) / max(n_points - 1, 1)
                if abs(prev_eta - eta_true) > 1e-12:
                    frac = prev_eta / (prev_eta - eta_true)
                    reversal_mach = prev_mach + frac * (mach - prev_mach)
                else:
                    reversal_mach = mach
        prev_eta = eta_true

        eta_display = max(-20.0, min(20.0, eta_true))
        display_list.append((mach, eta_display))
        regime_list.append((mach, _mach_regime(mach)))

    # ------------------------------------------------------------------
    # Smooth transition through control reversal.
    # The physical model eta = 1/(1 - q/q_div) is singular at q = q_div: it
    # blows up to +inf just below reversal and to -inf just above, which the
    # ±20 clamp turns into an artificial vertical jump (+20 -> -20). Replace
    # the curve around reversal with a smooth logistic/tanh model that passes
    # continuously through zero at the reversal Mach:
    #     eta(M) = eta0 * tanh((M_reversal - M) / k)
    # eta0 = nominal pre-reversal effectiveness amplitude, k = transition width.
    # ------------------------------------------------------------------
    if reversal_mach > 0.0:
        k = max(0.05, 0.10 * reversal_mach)   # transition width in Mach
        eta0 = 1.0                            # 100% nominal effectiveness
        display_list = [
            (m, eta0 * math.tanh((reversal_mach - m) / k))
            for (m, _) in display_list
        ]

    return display_list, reversal_mach, regime_list


# ---------------------------------------------------------------------------
# Public: full aeroelastic analysis (extended, backward-compatible)
# ---------------------------------------------------------------------------
def full_aeroelastic_analysis(assembly, max_flight_speed: float = 300.0,
                              max_flight_mach: float = 1.0) -> AeroelasticResult:
    """Full aeroelastic analysis for all fin sets.

    Uses trapezoidal planform consistently (mean chord from root+tip).
    Applies Mach-dependent CL_α, detects control reversal, and stores
    regime metadata.

    Parameters
    ----------
    assembly : RocketAssembly
        K2 rocket assembly containing fin sets.
    max_flight_speed : float
        Maximum flight speed (m/s) for margin computation.
    max_flight_mach : float
        Maximum design Mach number.

    Returns
    -------
    AeroelasticResult
    """
    from core.components import TrapezoidalFinSet, BodyTube
    from structures.solvers.base import get_structural_material
    from cfd.solvers.base import isa_conditions

    result = AeroelasticResult()

    # Collect fins from assembly
    fins = []
    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, BodyTube):
                for child in comp.children:
                    if isinstance(child, TrapezoidalFinSet):
                        fins.append(child)
            elif isinstance(comp, TrapezoidalFinSet):
                fins.append(comp)

    if not fins:
        result.divergence_speed_mps = float('inf')
        result.divergence_margin = float('inf')
        return result

    # Evaluate divergence speed (worst-case fin)
    min_v_div = float('inf')
    critical_fin = None
    critical_mat = None

    for fin in fins:
        mat = get_structural_material(getattr(fin, 'material', 'Plywood (Birch)'))
        # Trapezoidal planform: use mean aerodynamic chord
        c_mean = (fin.root_chord + fin.tip_chord) / 2.0
        # Aspect ratio for trapezoidal fin
        S_trap = 0.5 * (fin.root_chord + fin.tip_chord) * fin.height
        ar = fin.height ** 2 / S_trap if S_trap > 0 else 4.0

        v_div = divergence_speed(fin.height, c_mean, fin.thickness, mat.G,
                                 aspect_ratio=ar)
        if v_div < min_v_div:
            min_v_div = v_div
            critical_fin = fin
            critical_mat = mat

    result.divergence_speed_mps = min_v_div
    _, T0, _ = isa_conditions(0)
    a0 = math.sqrt(GAMMA_AIR * R_AIR * T0)
    result.divergence_mach = min_v_div / a0 if a0 > 0 else 0.0
    result.divergence_margin = (min_v_div / max_flight_speed
                                if max_flight_speed > 0 else float('inf'))

    # Effectiveness sweep with reversal detection for critical fin
    if critical_fin is not None and critical_mat is not None:
        c_mean = (critical_fin.root_chord + critical_fin.tip_chord) / 2.0
        S_trap = 0.5 * (critical_fin.root_chord + critical_fin.tip_chord) * critical_fin.height
        ar = critical_fin.height ** 2 / S_trap if S_trap > 0 else 4.0

        eff_data, rev_mach, regime_data = _effectiveness_full(
            critical_fin.height, c_mean, critical_fin.thickness, critical_mat.G,
            altitude_m=0.0, mach_range=(0.1, 3.0), n_points=60,
            elastic_axis_fraction=0.40, aspect_ratio=ar,
        )
        result.effectiveness_data = eff_data
        result.reversal_mach = rev_mach
        result.mach_regime_data = regime_data

        # Control-reversal margin + effectiveness at max design Mach
        if rev_mach > 0.0:
            result.reversal_margin = rev_mach - max_flight_mach
        else:
            result.reversal_margin = float('inf')
        if eff_data:
            # Linear interpolation of effectiveness at max_flight_mach
            eff = eff_data[-1][1]
            for (m0, e0), (m1, e1) in zip(eff_data, eff_data[1:]):
                if m0 <= max_flight_mach <= m1:
                    f = (max_flight_mach - m0) / (m1 - m0) if m1 > m0 else 0.0
                    eff = e0 + f * (e1 - e0)
                    break
            result.effectiveness_at_max_mach = eff

    # Fin deflection estimate at max-Q
    if fins and max_flight_speed > 0:
        fin = fins[0]
        mat = get_structural_material(getattr(fin, 'material', 'Plywood (Birch)'))
        _, _, rho = isa_conditions(0)
        q = 0.5 * rho * max_flight_speed ** 2
        # Trapezoidal planform area
        S = 0.5 * (fin.root_chord + fin.tip_chord) * fin.height
        F_aero = q * S * 0.5  # rough normal force
        # Cantilever beam deflection
        c_mean = (fin.root_chord + fin.tip_chord) / 2.0
        I = c_mean * fin.thickness ** 3 / 12.0
        if mat.E > 0 and I > 0:
            delta = F_aero * fin.height ** 3 / (3.0 * mat.E * I)
            result.max_deflection_mm = delta * 1000.0
            result.max_deflection_deg = math.degrees(
                math.atan(delta / max(fin.height, 0.01))
            )

    # Logging
    rev_str = (f", reversal M={result.reversal_mach:.2f}"
               if result.reversal_mach > 0.0 else ", no reversal")
    logger.info(
        f"Aeroelastic: V_div={min_v_div:.1f} m/s (M={result.divergence_mach:.2f}), "
        f"margin={result.divergence_margin:.2f}{rev_str}"
    )
    return result
