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

- Aerodynamic centre position (drives the EA offset e):
    Subsonic:   x_AC = 0.25 c     (thin-aerofoil theory)
    Supersonic: x_AC → 0.50 c     (Ackeret — the AC moves aft through the
                                   transonic range)
  A flat-plate fin's elastic axis is at mid-chord, so e = (x_EA − x_AC)·c
  COLLAPSES in supersonic flow and torsional divergence disappears. Holding
  x_AC at 0.25 c everywhere predicted a supersonic divergence that does not
  physically occur.

- Divergence dynamic pressure of a uniform cantilever fin (Mach-dependent):
    q_div(M) = (π² / 4) × GJ / (L² × c × e(M) × CL_α(M))
  This is the exact continuous-beam eigenvalue, not the lumped single-spring
  form K_θ/(S·e·CL_α) with K_θ = GJ/L, which is 2.47x low in q (1.57x in V).

  Ref: Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §8-2 eq. 8-46;
       Hodges & Pierce, "Introduction to Structural Dynamics and
       Aeroelasticity", §4.1.

- Aeroelastic amplification (the flexible-to-rigid lift ratio):
    η(M) = 1 / (1 − q(M) / q_div(M))
  η → +∞ as q → q_div. η < 0 is NOT control reversal: it is the
  post-divergence branch, i.e. the fin has already failed. A fixed fin with
  no control surface has no reversal mode at all — reversal requires a
  deflectable surface whose hinge moment reverses (C_Mδ sign change).

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

# Chordwise elastic-axis position of a symmetric constant-thickness fin, as a
# fraction of chord. Mid-chord — matching flutter_analysis's x_ea = 0.5 for
# the same slab. Keep the two in step: they analyse the same panel.
_EA_FRACTION = 0.50


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
    # Mach number at which the divergence boundary is crossed (q = q_div).
    # Named ``reversal_mach`` for backward compatibility with existing UI and
    # export call-sites; it is the DIVERGENCE Mach, not a control reversal —
    # a fixed fin has no reversal mode. See the module docstring.
    reversal_mach: float = 0.0

    # Per-point regime identification: [(mach, regime_str), ...]
    mach_regime_data: list = field(default_factory=list)

    # Divergence-Mach margin (M_div - M_max; inf if not crossed in range)
    reversal_margin: float = float('inf')
    # Aeroelastic amplification at the max design Mach (interpolated)
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


def _x_ac(mach: float) -> float:
    """Chordwise aerodynamic-centre position as a fraction of chord.

    Thin-aerofoil theory puts the AC at the quarter chord subsonically; in
    linearised supersonic (Ackeret) flow the load is uniform over the chord
    and the AC sits at mid-chord. The shift happens across the transonic
    range, blended here with the same smoothstep used for CL_α so the two
    stay consistent.

    Ref: Anderson, "Fundamentals of Aerodynamics", §4.5 (subsonic) and
    §12.3 (supersonic); NACA Report 1135.
    """
    if mach < 0.80:
        return 0.25
    if mach > 1.20:
        return 0.50
    t = (mach - 0.80) / (1.20 - 0.80)
    sm = 3.0 * t ** 2 - 2.0 * t ** 3
    return 0.25 + 0.25 * sm


def _q_divergence(shear_modulus: float, chord: float, thickness: float,
                  span: float, e: float, cl_alpha: float) -> float:
    """Divergence dynamic pressure of a uniform cantilever fin (Pa).

        q_div = (π² / 4) · G·J / (L² · c · e · CL_α)

    with the thin-rectangle torsion constant J = c·t³/3. This is the exact
    first eigenvalue of the continuous torsional-divergence problem, and is
    2.47x (= π²/4 · 1) higher in q than the lumped-spring approximation
    K_θ/(S·e·CL_α) with K_θ = GJ/L that this module used before.

    Ref: BAH "Aeroelasticity" §8-2 eq. 8-46; Hodges & Pierce §4.1.
    """
    if span <= 0 or chord <= 0 or thickness <= 0 or e <= 0 or cl_alpha <= 0:
        return float('inf')
    J = chord * thickness ** 3 / 3.0
    GJ = shear_modulus * J
    denom = span ** 2 * chord * e * cl_alpha
    if denom <= 0.0 or GJ <= 0.0:
        return float('inf')
    return (math.pi ** 2 / 4.0) * GJ / denom


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
                     elastic_axis_fraction: float = 0.50,
                     aspect_ratio: float = None,
                     mach: float = 0.0) -> float:
    """Torsional divergence speed for a thin fin.

    V_div = √(2 · q_div / ρ),  q_div = (π²/4)·G·J / (L²·c·e·CL_α)

    Parameters
    ----------
    span : float                 Fin semi-span (m).
    chord : float                Mean aerodynamic chord (m).
    thickness : float            Fin thickness (m).
    shear_modulus : float        Shear modulus G (Pa).
    altitude_m : float           Flight altitude (m).
    elastic_axis_fraction : float
        Chordwise position of the elastic axis as a fraction of chord. The
        default is 0.50 — a symmetric flat/uncambered plate of uniform
        thickness has its shear centre at mid-chord, which is also what
        ``flutter_analysis._fin_section_properties`` assumes (x_ea = 0.5).
        The old 0.40 default disagreed with the flutter model on the same
        fin and, being closer to the AC, gave a 1.29x HIGHER (unconservative)
        divergence speed.
    aspect_ratio : float | None
        Fin aspect ratio for finite-span correction.  If None, computed
        from span²/(span×chord) = span/chord.
    mach : float
        Free-stream Mach for compressibility-corrected CL_α and for the
        aerodynamic-centre shift.  0 → incompressible.

    Returns
    -------
    float : Divergence speed (m/s). Infinite when the AC has moved back to
    the elastic axis (supersonic) so no divergent moment exists.

    Reference
    ---------
    Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §8.2–8.3.
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or chord <= 0 or thickness <= 0:
        return float('inf')

    _, T, rho = isa_conditions(altitude_m)

    # Elastic-axis offset AFT of the aerodynamic centre. The AC moves from
    # 0.25c to 0.50c through the transonic range, so e shrinks to zero for a
    # mid-chord EA and supersonic torsional divergence disappears.
    e = (elastic_axis_fraction - _x_ac(mach)) * chord
    if e <= 0.0:
        # EA at or ahead of AC → the aero moment is restoring, no divergence
        return float('inf')

    # Aspect ratio
    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    # Lift-curve slope (compressibility + finite span)
    cl_a = _cl_alpha(mach, aspect_ratio)

    q_div = _q_divergence(shear_modulus, chord, thickness, span, e, cl_a)

    # Divergence speed
    if q_div > 0 and math.isfinite(q_div) and rho > 0:
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
                              elastic_axis_fraction: float = 0.50,
                              aspect_ratio: float = None) -> list:
    """Compute aeroelastic effectiveness η vs Mach with compressibility.

    η(M) = 1 / (1 − q(M) / q_div(M))

    q_div now varies with Mach because CL_α(M) changes across subsonic,
    transonic, and supersonic regimes.

    η → +∞ as q → q_div (divergence). Negative η is the post-divergence
    branch — the fin has already failed, NOT control reversal.
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

    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    m_start, m_end = mach_range
    result = []

    for i in range(n_points):
        mach = m_start + (m_end - m_start) * i / max(n_points - 1, 1)
        V = mach * a
        q = 0.5 * rho * V ** 2

        # Mach-dependent divergence dynamic pressure (AC shift included)
        e = (elastic_axis_fraction - _x_ac(mach)) * chord
        cl_a = _cl_alpha(mach, aspect_ratio)
        q_div = (_q_divergence(shear_modulus, chord, thickness, span, e, cl_a)
                 if e > 0.0 else float('inf'))

        if math.isfinite(q_div) and q_div > 0.0:
            ratio = q / q_div
            eta_true = 1.0 / (1.0 - ratio) if abs(1.0 - ratio) > 1e-9 else float('inf')
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
    """Return (display_list, divergence_mach, regime_list).

    display_list : [(mach, eta_display), ...]
    divergence_mach : float  (0.0 if q never reaches q_div in the range)
    regime_list : [(mach, regime_str), ...]

    The crossing reported is where q(M) reaches q_div(M) — torsional
    DIVERGENCE. The previous version called it "control reversal", found it
    by watching η change sign, and then overwrote the whole curve with
    ``eta0·tanh((M_rev − M)/k)``: an invented function that made η glide
    smoothly through zero and hid the divergence singularity entirely. A
    fixed fin has no reversal mode, so nothing about that curve was
    physical.
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or chord <= 0 or thickness <= 0:
        pts = _linspace(mach_range[0], mach_range[1], n_points)
        return ([(m, 1.0) for m in pts], 0.0,
                [(m, _mach_regime(m)) for m in pts])

    _, T, rho = isa_conditions(altitude_m)
    a = math.sqrt(GAMMA_AIR * R_AIR * T)

    if aspect_ratio is None:
        aspect_ratio = span / chord if chord > 0 else 4.0

    m_start, m_end = mach_range
    display_list = []
    regime_list = []
    divergence_mach = 0.0
    prev_ratio = None
    prev_mach = None

    for i in range(n_points):
        mach = m_start + (m_end - m_start) * i / max(n_points - 1, 1)
        V = mach * a
        q = 0.5 * rho * V ** 2
        e = (elastic_axis_fraction - _x_ac(mach)) * chord
        cl_a = _cl_alpha(mach, aspect_ratio)
        q_div = (_q_divergence(shear_modulus, chord, thickness, span, e, cl_a)
                 if e > 0.0 else float('inf'))

        if math.isfinite(q_div) and q_div > 0.0:
            ratio = q / q_div
            eta_true = 1.0 / (1.0 - ratio) if abs(1.0 - ratio) > 1e-9 else float('inf')
        else:
            ratio = 0.0
            eta_true = 1.0

        # Divergence = q reaching q_div, i.e. the load ratio crossing 1.
        # Interpolating on the ratio is well behaved; interpolating on η is
        # not, because η is singular exactly at the crossing.
        if prev_ratio is not None and divergence_mach == 0.0:
            if prev_ratio < 1.0 <= ratio:
                dr = ratio - prev_ratio
                divergence_mach = (prev_mach + (1.0 - prev_ratio) / dr * (mach - prev_mach)
                                   if abs(dr) > 1e-12 else mach)
        prev_ratio, prev_mach = ratio, mach

        eta_display = max(-20.0, min(20.0, eta_true))
        display_list.append((mach, eta_display))
        regime_list.append((mach, _mach_regime(mach)))

    return display_list, divergence_mach, regime_list


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
                                 aspect_ratio=ar, mach=0.0)
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
            elastic_axis_fraction=_EA_FRACTION, aspect_ratio=ar,
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

    # ── Fin bending deflection at max-Q ─────────────────────────────────
    # Same model as structures.workstation.fin_analysis so the Dynamics and
    # Structures tabs report one number for one fin: the DIVERGENCE-critical
    # fin (was fins[0]), a finite-AR lift slope at a 5 deg gust AoA (was a
    # bare CN = 0.5), and a uniformly distributed load, delta = F L^3/(8EI)
    # (was a tip point load, F L^3/(3EI) — 2.67x high).
    defl_fin = critical_fin if critical_fin is not None else (fins[0] if fins else None)
    if defl_fin is not None and max_flight_speed > 0:
        mat = critical_mat if critical_mat is not None else get_structural_material(
            getattr(defl_fin, 'material', 'Plywood (Birch)'))
        _, _, rho = isa_conditions(0)
        q = 0.5 * rho * max_flight_speed ** 2
        root, tip = defl_fin.root_chord, defl_fin.tip_chord
        S = 0.5 * (root + tip) * defl_fin.height          # trapezoid planform
        c_mean = 0.5 * (root + tip)
        ar_d = defl_fin.height ** 2 / S if S > 0 else 4.0
        cn_alpha = 2.0 * math.pi * ar_d / (ar_d + 2.0)    # finite-AR lift slope
        F_aero = q * cn_alpha * math.radians(5.0) * S
        I = c_mean * defl_fin.thickness ** 3 / 12.0
        if mat.E > 0 and I > 0:
            delta = F_aero * defl_fin.height ** 3 / (8.0 * mat.E * I)
            result.max_deflection_mm = delta * 1000.0
            # Elastic twist of the tip section is what matters aeroelastically
            # (an incidence change), not atan(delta/span) — that is the bend
            # slope of the beam, which adds no angle of attack.
            J = c_mean * defl_fin.thickness ** 3 / 3.0
            GJ = mat.G * J if mat.G > 0 else 0.0
            e_arm = (_EA_FRACTION - 0.25) * c_mean
            if GJ > 0:
                twist = F_aero * e_arm * defl_fin.height / GJ   # rad, T·L/GJ
                result.max_deflection_deg = math.degrees(twist)

    # Logging
    rev_str = (f", divergence crossed at M={result.reversal_mach:.2f}"
               if result.reversal_mach > 0.0 else ", divergence not reached")
    logger.info(
        f"Aeroelastic: V_div={min_v_div:.1f} m/s (M={result.divergence_mach:.2f}), "
        f"margin={result.divergence_margin:.2f}{rev_str}"
    )
    return result
