"""
K2 Aerospace — Fin Flutter Analysis
=====================================
Two-level flutter assessment for rocket fin structures.

Level 1 — NACA TN-4197 empirical formula (unchanged legacy):
  V_f = a × √( G_E / (1.337 × AR³ × P × (λ+1) / (2 × (AR+2) × (t/c)³)) )

Level 2 — 2-DOF p-k flutter solver (bending-torsion coupling):
  Structural modes:
    Bending:  f_b = (1.875² / 2π) √(EI / (m L⁴))      [cantilever 1st mode]
    Torsion:  f_t = (1 / 2π) √(GJ / (I_α L))           [torsion fundamental]
  Aerodynamics:
    Theodorsen function C(k) via Jones rational approx:
      C(k) = 1 − 0.165/(1 − j·0.0455/k) − 0.335/(1 − j·0.3/k)
      Ref: Jones, R.T., "The Unsteady Lift of a Wing of Finite Aspect Ratio",
           NACA Report 681, 1940.
  Solution:
    p-k iteration at each velocity step; eigenvalues p = σ + jω yield
    damping g = 2σ/ω and frequency f = ω/(2π).
    Flutter = velocity where g crosses zero from negative to positive.

References
----------
- NACA TN-4197: Barmby, Cunningham & Garrick, "Panel Flutter", 1958
- Bisplinghoff, Ashley, Halfman: "Aeroelasticity", Dover, Ch. 6-9
- Theodorsen, T.: NACA Report 496, 1935
- Jones, R.T.: NACA Report 681, 1940
"""
from __future__ import annotations

import cmath
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("K2.Dynamics.Flutter")

# ── Physical constants ──────────────────────────────────────────────────────
GAMMA_AIR = 1.4
R_AIR = 287.05          # J/(kg·K)

# ── Cantilever eigenvalue for first bending mode ────────────────────────────
_BETA1_L = 1.8751040687  # first root of cos(βL)·cosh(βL) + 1 = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FlutterResult:
    """Results from flutter analysis.

    All original fields are preserved for backward compatibility.
    New fields added for p-k solver output and flight-envelope awareness.
    """
    # ── Legacy fields (unchanged) ───────────────────────────────────────────
    flutter_speed_mps: float = 0.0          # critical flutter speed (m/s)
    flutter_mach: float = 0.0               # flutter Mach number
    max_flight_speed: float = 0.0           # max expected flight speed (m/s)
    max_flight_mach: float = 0.0            # max expected Mach number
    flutter_margin: float = 0.0             # V_flutter / V_max (>1 = safe)
    safe: bool = True
    altitude_sweep: list = field(default_factory=list)
    # [(alt_m, V_flutter, M_flutter, fin_name), ...]
    mach_sweep: list = field(default_factory=list)
    # [(mach, margin, fin_name), ...]

    # ── New: p-k solver V-g / V-f plot data ─────────────────────────────────
    vg_data: list = field(default_factory=list)
    # [(velocity_mps, damping_g, mode_label), ...]
    vf_data: list = field(default_factory=list)
    # [(velocity_mps, freq_hz, mode_label), ...]

    # ── New: flutter characterisation ───────────────────────────────────────
    flutter_frequency_hz: float = 0.0       # frequency at flutter point (Hz)
    critical_mode_name: str = ""            # e.g. '1st Bending-Torsion'
    flutter_mechanism: str = ""             # 'bending-torsion' | 'fin_flutter'
    damping_margin: float = 0.0             # minimum damping ratio before zero-crossing

    # ── New: flight-envelope status ─────────────────────────────────────────
    envelope_status: str = "NOT_REACHED"
    # 'WITHIN_ENVELOPE' | 'OUTSIDE_ENVELOPE' | 'NOT_REACHED'

    # ── New: per-fin detail ─────────────────────────────────────────────────
    per_fin_results: list = field(default_factory=list)
    # list of dicts with per-fin p-k results

    # ── New: worst-case altitude ────────────────────────────────────────────
    worst_case_altitude_m: float = 0.0      # altitude with minimum margin
    worst_case_margin: float = float('inf') # margin at worst altitude


# ═══════════════════════════════════════════════════════════════════════════════
#  THEODORSEN AERODYNAMICS (Jones rational approximation)
# ═══════════════════════════════════════════════════════════════════════════════

def theodorsen_C(k: float) -> complex:
    """Theodorsen circulation function C(k) using Jones' two-pole approximation.

    C(k) = 1 − 0.165/(1 − j·0.0455/k) − 0.335/(1 − j·0.3/k)

    Parameters
    ----------
    k : float
        Reduced frequency  k = ω·c / (2V).  Must be > 0.

    Returns
    -------
    complex
        C(k) = F(k) + j·G(k)

    References
    ----------
    Jones, R.T., NACA Report 681, 1940.
    Bisplinghoff, Ashley, Halfman, "Aeroelasticity", §5-6.
    """
    if k <= 0:
        return complex(1.0, 0.0)  # quasi-steady limit
    jk_inv = complex(0.0, 1.0) / k   # = j/k
    C = (1.0
         - 0.165 / (1.0 - 0.0455 * jk_inv)
         - 0.335 / (1.0 - 0.3 * jk_inv))
    return C


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL NATURAL FREQUENCIES
# ═══════════════════════════════════════════════════════════════════════════════

def _cantilever_bending_freq(E: float, I_bend: float,
                             m_per_length: float, L: float) -> float:
    """First bending frequency of a cantilever beam.

    f_b = (β₁L)² / (2π) · √(EI / (m̄ · L⁴))

    where β₁L = 1.8751 (first eigenvalue of Euler-Bernoulli beam).

    Parameters
    ----------
    E : float           Young's modulus (Pa)
    I_bend : float      Second moment of area (m⁴)
    m_per_length : float  Mass per unit length (kg/m)
    L : float           Span length (m)

    Returns
    -------
    float : Frequency in Hz

    References
    ----------
    Blevins, "Formulas for Natural Frequency and Mode Shape", Table 8-1.
    """
    if L <= 0 or m_per_length <= 0 or E <= 0 or I_bend <= 0:
        return 1.0  # safe fallback
    beta_L_sq = _BETA1_L ** 2
    return (beta_L_sq / (2.0 * math.pi)) * math.sqrt(E * I_bend /
                                                       (m_per_length * L ** 4))


def _torsion_fundamental_freq(G: float, J: float,
                              I_alpha: float, L: float) -> float:
    """First torsional frequency of a cantilever with free end.

    f_t = (1 / 2π) √(GJ / (I_α · L))

    For a cantilever with one end free the first torsional mode shape is
    a quarter-wave; the effective stiffness is GJ/L.

    Parameters
    ----------
    G : float       Shear modulus (Pa)
    J : float       Torsional constant (m⁴)
    I_alpha : float Mass moment of inertia per unit length about EA (kg·m)
    L : float       Span (m)

    Returns
    -------
    float : Frequency in Hz

    References
    ----------
    Bisplinghoff, Ashley, Halfman, "Aeroelasticity", §3-3.
    """
    if L <= 0 or I_alpha <= 0 or G <= 0 or J <= 0:
        return 2.0  # safe fallback above bending
    return (1.0 / (2.0 * math.pi)) * math.sqrt(G * J / (I_alpha * L))


# ═══════════════════════════════════════════════════════════════════════════════
#  FIN SECTION PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def _fin_section_properties(root_chord: float, tip_chord: float,
                            thickness: float, height: float,
                            density: float):
    """Compute structural section properties for a trapezoidal fin panel.

    Uses mean chord for cross-section sizing (rectangular slab approximation).
    Returns a dict with all needed structural parameters.

    Parameters
    ----------
    root_chord, tip_chord : float   Chord lengths (m)
    thickness : float               Fin panel thickness (m)
    height : float                  Fin semi-span from root (m)
    density : float                 Material density (kg/m³)

    Returns
    -------
    dict with keys:
        c_mean    – mean chord (m)
        m_bar     – mass per unit span (kg/m)
        I_bend    – 2nd moment of area for bending about chord axis (m⁴)
        J         – torsional constant for thin rectangle (m⁴)
        I_alpha   – mass polar moment of inertia per unit span about EA (kg·m)
        x_ea      – elastic axis location as fraction of chord from LE
    """
    c_mean = 0.5 * (root_chord + tip_chord)
    if c_mean <= 0:
        c_mean = 0.01
    t = thickness if thickness > 0 else 0.001

    # Rectangular cross-section approximation
    # Bending about the chord-wise axis (out-of-plane): I = c·t³/12
    I_bend = c_mean * t ** 3 / 12.0                  # m⁴

    # Torsional constant for thin rectangle: J ≈ c·t³/3
    # (Ref: Roark's Formulas, Table 10.1)
    J = c_mean * t ** 3 / 3.0                        # m⁴

    # Mass per unit span
    m_bar = density * c_mean * t                      # kg/m

    # Elastic axis at mid-chord for symmetric section
    x_ea = 0.5  # fraction of chord from LE

    # Mass moment of inertia about elastic axis, per unit span
    # For uniform rectangular section:  I_α = m̄ · c² / 12
    I_alpha = m_bar * c_mean ** 2 / 12.0              # kg·m

    return {
        "c_mean": c_mean,
        "m_bar": m_bar,
        "I_bend": I_bend,
        "J": J,
        "I_alpha": I_alpha,
        "x_ea": x_ea,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2×2 COMPLEX EIGENVALUE SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def _solve_2x2_eigenvalues(a11: complex, a12: complex,
                            a21: complex, a22: complex):
    """Eigenvalues of a 2×2 complex matrix [[a11,a12],[a21,a22]].

    Uses the quadratic formula on the characteristic polynomial:
        λ² − (a11+a22)·λ + (a11·a22 − a12·a21) = 0

    Returns
    -------
    (λ₁, λ₂) : tuple of complex
    """
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    disc = cmath.sqrt(tr * tr - 4.0 * det)
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    return lam1, lam2


def _eigenvector_2x2(a11: complex, a12: complex,
                     a21: complex, a22: complex, lam: complex):
    """Eigenvector for eigenvalue lam of 2×2 matrix.

    Returns normalised (v1, v2).
    """
    # From (A - λI)v = 0, first row: (a11-λ)v1 + a12·v2 = 0
    off = a11 - lam
    if abs(a12) > abs(a21):
        if abs(a12) < 1e-30:
            return (complex(1, 0), complex(0, 0))
        v1 = -a12
        v2 = off
    else:
        if abs(a21) < 1e-30:
            return (complex(1, 0), complex(0, 0))
        v1 = a22 - lam
        v2 = -a21
    norm = cmath.sqrt(v1 * v1.conjugate() + v2 * v2.conjugate()).real
    if norm < 1e-30:
        return (complex(1, 0), complex(0, 0))
    return (v1 / norm, v2 / norm)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE TRACKING (simplified MAC)
# ═══════════════════════════════════════════════════════════════════════════════

def _mac_2dof(v_a, v_b):
    """Modal Assurance Criterion for 2-component complex eigenvectors.

    MAC = |{vA}ᴴ{vB}|² / (({vA}ᴴ{vA})({vB}ᴴ{vB}))

    Parameters
    ----------
    v_a, v_b : tuple of 2 complex values

    Returns
    -------
    float in [0, 1]
    """
    dot = v_a[0].conjugate() * v_b[0] + v_a[1].conjugate() * v_b[1]
    na = (abs(v_a[0]) ** 2 + abs(v_a[1]) ** 2)
    nb = (abs(v_b[0]) ** 2 + abs(v_b[1]) ** 2)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return (abs(dot) ** 2) / (na * nb)


# ═══════════════════════════════════════════════════════════════════════════════
#  P-K FLUTTER SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def pk_flutter_analysis(span: float, root_chord: float, tip_chord: float,
                        thickness: float, E: float, G: float,
                        density_mat: float, altitude_m: float = 0.0,
                        V_max_factor: float = 1.5,
                        max_flight_speed: float = 300.0,
                        n_steps: int = 120):
    """2-DOF p-k flutter solver for a trapezoidal cantilever fin.

    Solves the bending-torsion aeroelastic eigenvalue problem at each
    velocity step using Theodorsen unsteady aerodynamics (Jones approx.).

    The structural model treats the fin as a cantilever with:
      - 1st bending mode  (plunge DOF h)
      - 1st torsion mode  (pitch DOF α)

    At each velocity V the reduced-frequency-dependent aerodynamic matrix
    [A(k)] is combined with the structural mass and stiffness matrices.
    The eigenvalue problem is iterated in k until self-consistent (p-k method).

    Parameters
    ----------
    span : float            Fin semi-span (m)
    root_chord : float      Root chord (m)
    tip_chord : float       Tip chord (m)
    thickness : float       Fin thickness (m)
    E : float               Young's modulus (Pa)
    G : float               Shear modulus (Pa)
    density_mat : float     Material density (kg/m³)
    altitude_m : float      Flight altitude (m)
    V_max_factor : float    Scan up to V_max_factor × max_flight_speed
    max_flight_speed : float  Design max flight speed (m/s)
    n_steps : int           Number of velocity steps

    Returns
    -------
    dict with keys:
        flutter_speed_mps   : float
        flutter_mach        : float
        flutter_frequency_hz: float
        critical_mode_name  : str
        flutter_mechanism   : str
        damping_margin      : float
        envelope_status     : str
        vg_data             : list of (V, g, mode_label)
        vf_data             : list of (V, f, mode_label)

    References
    ----------
    Bisplinghoff, Ashley, Halfman, "Aeroelasticity", Ch. 6 (p-k method).
    Hassig, H.J., "An Approximate True Damping Solution of the Flutter
        Equation by Determinant Iteration", AIAA J. 1971.
    """
    from cfd.solvers.base import isa_conditions

    P, T, rho = isa_conditions(altitude_m)
    a_sound = math.sqrt(GAMMA_AIR * R_AIR * T)

    # ── Section properties ──────────────────────────────────────────────────
    props = _fin_section_properties(root_chord, tip_chord, thickness,
                                    span, density_mat)
    c = props["c_mean"]         # representative chord (m)
    m_bar = props["m_bar"]      # kg/m
    I_bend = props["I_bend"]    # m⁴
    J = props["J"]              # m⁴
    I_alpha = props["I_alpha"]  # kg·m (per unit span)
    b = c / 2.0                 # semi-chord (m)
    x_ea = props["x_ea"]       # EA location fraction from LE

    # ── Natural frequencies ─────────────────────────────────────────────────
    omega_h = 2.0 * math.pi * _cantilever_bending_freq(E, I_bend, m_bar, span)
    omega_a = 2.0 * math.pi * _torsion_fundamental_freq(G, J, I_alpha, span)

    if omega_h <= 0 or omega_a <= 0:
        logger.warning("pk_flutter: degenerate structural frequencies; "
                       "returning infinite flutter speed.")
        return _empty_pk_result(max_flight_speed, a_sound)

    logger.debug(f"pk_flutter: f_bend={omega_h/(2*math.pi):.2f} Hz, "
                 f"f_torsion={omega_a/(2*math.pi):.2f} Hz, "
                 f"chord={c:.4f} m, span={span:.4f} m")

    # ── Structural matrices (generalised coords) ────────────────────────────
    # DOF vector: {h/b, α}
    # Mass matrix M (non-dimensional by m_bar·b):
    #   M = [[μ,   x_α],
    #        [x_α, r_α²]]
    # where μ  = m_bar / (π·ρ·b²)  (mass ratio)
    #       x_α = static unbalance / (m_bar·b)   = (0.5 - x_ea)·c/b·... simplified
    #       r_α² = I_alpha / (m_bar·b²)
    mu = m_bar / (math.pi * rho * b ** 2) if (rho > 0 and b > 0) else 1e6
    # For symmetric section EA at mid-chord ⇒ x_α = 0 (no static unbalance)
    # In practice fins have small offset; use a_h = (x_ea - 0.5)*2
    # a_h is distance from mid-chord to EA in semi-chords (dimensionless)
    a_h = (x_ea - 0.5) * 2.0  # = 0 for mid-chord EA
    x_alpha = 0.0  # symmetric section ⇒ CG at EA ⇒ zero static moment
    r_alpha_sq = I_alpha / (m_bar * b ** 2) if (m_bar > 0 and b > 0) else 1.0

    # Frequency ratio
    omega_ratio_sq = (omega_h / omega_a) ** 2

    # ── Velocity sweep ──────────────────────────────────────────────────────
    V_scan_max = max(max_flight_speed * V_max_factor, 50.0)
    dV = V_scan_max / n_steps if n_steps > 0 else 5.0

    vg_data = []       # (V, g, mode_label)
    vf_data = []       # (V, f, mode_label)

    MODE_LABELS = ["1st Bending", "1st Torsion"]

    flutter_speed = float('inf')
    flutter_freq = 0.0
    flutter_mode_idx = -1
    damping_margin = 0.0  # most negative g before crossing

    # Previous step eigenvectors for MAC tracking
    prev_vecs = [None, None]
    # Previous step: (g, omega) per mode for crossing detection
    prev_g = [None, None]
    min_g_per_mode = [0.0, 0.0]  # track most negative damping

    # Start from small positive velocity to avoid k → ∞
    V_start = max(dV, 1.0)

    for i_step in range(n_steps):
        V = V_start + i_step * dV
        if V <= 0:
            continue

        # ── p-k iteration for self-consistent reduced frequency ─────────
        # Initial guess: use structural frequencies
        k_guesses = [omega_h * b / V, omega_a * b / V]
        eigenvalues = []
        eigenvectors = []

        for i_mode in range(2):
            # p-k iteration: solve at assumed k, update k from result
            k_current = k_guesses[i_mode]
            if k_current <= 0:
                k_current = 0.01

            for _pk_iter in range(25):
                C_k = theodorsen_C(k_current)

                # ── Build aeroelastic matrix ────────────────────────────
                # Non-dimensional flutter determinant (Ref: BAH eq 6-24):
                #
                # [M]{q̈} + [K]{q} = (ρ·V²·b/2)[A(k)]{q}
                #
                # Written as eigenvalue problem in p² = (σ+jω)² :
                #   det | [K] - p²[M] - ρ·V²·b·[A(k)]/2 | = 0
                #
                # For 2-DOF {h/b, α}:
                #
                # K11 = μ·ω_h²/ω_α²   K22 = r_α²
                # M11 = μ               M22 = r_α²
                # M12 = M21 = x_α
                #
                # Aero matrix [A(k)] from Theodorsen (thin airfoil):
                #   L = π·ρ·b²[ḧ + V·α̇ − b·a·α̈]
                #       + 2π·ρ·V·b·C(k)[ḣ + V·α + b(0.5−a)·α̇]
                #   M_α = π·ρ·b²[b·a·ḧ − V·b(0.5−a)·α̇ − b²(1/8+a²)·α̈]
                #       + 2π·ρ·V·b²·(a+0.5)·C(k)[ḣ + Vα + b(0.5−a)α̇]
                #
                # Non-dimensionalise by m_bar·b and ω_α²:
                a = a_h

                # Reduced velocity
                U_star = V / (b * omega_a) if (b * omega_a) > 0 else 1e6

                # Aero coefficients (per unit span, divided by π·ρ·b²·ω_α²)
                # Using p = jω approximation for the k-iteration:
                jk = complex(0, k_current)

                # Non-circulatory (apparent mass) terms
                # Circulatory terms via C(k)
                F_Ck = C_k.real
                G_Ck = C_k.imag

                # Build the effective dynamic matrix entries
                # Eigenvalue: λ = (p/ω_α)²   where p = σ + jω
                # Equation: ([K_eff] - λ[M_eff]){q} = 0
                #
                # Following Hassig's p-k formulation:
                # At assumed ω (from k = ω·b/V):
                omega_assumed = k_current * V / b if b > 0 else omega_a
                p_sq_norm = -(omega_assumed / omega_a) ** 2  # = -(ω/ω_α)²

                # Stiffness terms (structural)
                K11 = mu * omega_ratio_sq   # bending stiffness (non-dim)
                K22 = r_alpha_sq            # torsion stiffness (non-dim)

                # Mass terms (structural)
                M11 = mu
                M22 = r_alpha_sq
                M12 = x_alpha
                M21 = x_alpha

                # Aerodynamic terms (divided by π·ρ·b²·ω_α²):
                # They appear as  -1/μ * ... after dividing by structural mass
                # Direct formulation of the 2×2 eigenvalue problem:
                #   A_eff · q = λ · q
                #   where A_eff = M_eff⁻¹ · K_eff
                #
                # Simpler: use the flutter determinant directly.
                # Non-dimensional aero stiffness and damping per Theodorsen:

                # Quasi-steady + unsteady lift coefficient contributions
                # L_h = -π·ρ·b²·ω²·h - 2π·ρ·V·b·C(k)·(jω·h)
                # L_α = -π·ρ·b²·V·(jω)·α + 2π·ρ·V²·b·C(k)·α
                #        + 2π·ρ·V·b²·(0.5-a)·C(k)·(jω)·α
                #        + π·ρ·b³·a·ω²·α

                # For eigenvalue problem, collect into matrix form
                # and solve. Using the direct 2x2 approach:

                # Aero contributions (non-dim by π·ρ·b²):
                inv_mu = 1.0 / mu if mu > 1e-30 else 0.0

                # Aerodynamic stiffness matrix entries (non-dim)
                # From lift due to α (circulatory): 2·C(k)·V²/(b·ω_α²·b)
                # = 2·C(k)·U*²
                A_k_11 = 0.0           # no aero stiffness from h displacement
                A_k_12 = 2.0 * C_k * U_star ** 2  # lift from α, circulatory
                A_k_21 = 0.0
                A_k_22 = (2.0 * (a + 0.5) * C_k * U_star ** 2)  # moment from α

                # Aerodynamic damping matrix entries (non-dim)
                # proportional to jk (from velocity terms)
                A_d_11 = 2.0 * C_k * U_star * jk  # lift from ḣ, circulatory
                A_d_12_nc = -U_star * jk           # lift from α̇, non-circ
                A_d_12_c = 2.0 * (0.5 - a) * C_k * U_star * jk  # circ part
                A_d_12 = A_d_12_nc + A_d_12_c
                A_d_21 = 2.0 * (a + 0.5) * C_k * U_star * jk
                A_d_22_nc = U_star * (0.5 - a) * jk
                A_d_22_c = 2.0 * (a + 0.5) * (0.5 - a) * C_k * U_star * jk
                A_d_22 = A_d_22_nc + A_d_22_c

                # Apparent mass terms (proportional to -k²)
                k_sq = k_current ** 2
                A_m_11 = complex(k_sq, 0)           # apparent mass from ḧ
                A_m_12 = complex(-a * k_sq, 0)      # apparent mass from α̈
                A_m_21 = complex(-a * k_sq, 0)
                A_m_22 = complex((1.0/8.0 + a*a) * k_sq, 0)

                # Total aero contribution: [A_total] = [A_k] + [A_d] + [A_m]
                # These multiply 1/μ in the eigenvalue equation
                At11 = A_k_11 + A_d_11 + A_m_11
                At12 = A_k_12 + A_d_12 + A_m_12
                At21 = A_k_21 + A_d_21 + A_m_21
                At22 = A_k_22 + A_d_22 + A_m_22

                # Effective matrix: [K_struct + K_aero/μ] - λ·[M_struct + M_aero/μ]
                # Rearranging into standard A·q = λ·q:
                # [M_eff]⁻¹[K_eff]·q = λ·q  with  λ = (ω/ω_α)²
                #
                # M_eff = [M_struct + apparent_mass_aero/μ]
                # K_eff = [K_struct + aero_stiffness/μ + aero_damping/μ]
                #
                # For the p-k method we solve:
                #   det | K_total - λ·M_total | = 0
                # where K_total includes aero stiffness + damping (at assumed k)
                # and M_total includes structural + apparent mass

                Me11 = complex(M11, 0) + At11 * inv_mu
                Me12 = complex(M12, 0) + At12 * inv_mu
                Me21 = complex(M21, 0) + At21 * inv_mu
                Me22 = complex(M22, 0) + At22 * inv_mu

                Ke11 = complex(K11, 0) + (A_k_11 + A_d_11) * inv_mu
                Ke12 = complex(K12 if 'K12' in dir() else 0.0, 0) + (A_k_12 + A_d_12) * inv_mu
                Ke21 = complex(0.0, 0) + (A_k_21 + A_d_21) * inv_mu
                Ke22 = complex(K22, 0) + (A_k_22 + A_d_22) * inv_mu

                # Solve M_eff⁻¹ · K_eff eigenvalue
                det_M = Me11 * Me22 - Me12 * Me21
                if abs(det_M) < 1e-30:
                    break

                inv_det_M = 1.0 / det_M
                # M_eff inverse
                Mi11 = Me22 * inv_det_M
                Mi12 = -Me12 * inv_det_M
                Mi21 = -Me21 * inv_det_M
                Mi22 = Me11 * inv_det_M

                # A = M_eff⁻¹ · K_eff
                Ae11 = Mi11 * Ke11 + Mi12 * Ke21
                Ae12 = Mi11 * Ke12 + Mi12 * Ke22
                Ae21 = Mi21 * Ke11 + Mi22 * Ke21
                Ae22 = Mi21 * Ke12 + Mi22 * Ke22

                lam1, lam2 = _solve_2x2_eigenvalues(Ae11, Ae12, Ae21, Ae22)

                # λ = (ω/ω_α)²  ⇒  ω = ω_α·√λ
                # Extract ω from eigenvalue
                sqrt_lam = cmath.sqrt(lam1 if i_mode == 0 else lam2)
                omega_new = abs(sqrt_lam) * omega_a

                if omega_new <= 0:
                    break

                k_new = omega_new * b / V if V > 0 else k_current

                if abs(k_new - k_current) / max(k_current, 1e-10) < 0.005:
                    k_current = k_new
                    break
                k_current = 0.7 * k_current + 0.3 * k_new  # relaxation

            # ── Extract damping and frequency from converged eigenvalue ──
            lam_use = lam1 if i_mode == 0 else lam2
            p_complex = cmath.sqrt(lam_use) * omega_a  # p = σ + jω

            omega_result = abs(p_complex.imag) if abs(p_complex.imag) > 0 else abs(p_complex)
            sigma_result = p_complex.real

            freq_hz = omega_result / (2.0 * math.pi) if omega_result > 0 else 0.0
            g_damping = (2.0 * sigma_result / omega_result
                         if omega_result > 1e-10 else 0.0)

            eigenvalues.append((g_damping, freq_hz, omega_result))

            # Compute eigenvector for MAC tracking
            vec = _eigenvector_2x2(Ae11, Ae12, Ae21, Ae22, lam_use)
            eigenvectors.append(vec)

        if len(eigenvalues) < 2:
            continue

        # ── Mode tracking via MAC ───────────────────────────────────────
        assigned = [None, None]  # which computed mode maps to which tracked mode
        if prev_vecs[0] is not None:
            # Compute MAC between previous and current eigenvectors
            mac_matrix = [[0.0, 0.0], [0.0, 0.0]]
            for im in range(2):
                for jm in range(2):
                    mac_matrix[im][jm] = _mac_2dof(prev_vecs[im],
                                                    eigenvectors[jm])
            # Greedy assignment (2x2 is trivial)
            if mac_matrix[0][0] + mac_matrix[1][1] >= mac_matrix[0][1] + mac_matrix[1][0]:
                assigned = [0, 1]
            else:
                assigned = [1, 0]
        else:
            # First step: assign by frequency (lower = bending)
            if eigenvalues[0][1] <= eigenvalues[1][1]:
                assigned = [0, 1]
            else:
                assigned = [1, 0]

        # ── Store tracked results ───────────────────────────────────────
        for i_mode in range(2):
            j_comp = assigned[i_mode]  # which computed eigenvalue
            g_val = eigenvalues[j_comp][0]
            f_val = eigenvalues[j_comp][1]

            vg_data.append((V, g_val, MODE_LABELS[i_mode]))
            vf_data.append((V, f_val, MODE_LABELS[i_mode]))

            # Update tracking state
            prev_vecs[i_mode] = eigenvectors[j_comp]

            # Track minimum damping (most negative = most stable)
            if g_val < min_g_per_mode[i_mode]:
                min_g_per_mode[i_mode] = g_val

            # ── Flutter detection: g crosses zero from negative ─────
            if prev_g[i_mode] is not None:
                g_prev = prev_g[i_mode]
                if g_prev < 0 and g_val >= 0 and V < flutter_speed:
                    # Linear interpolation for exact crossing
                    dg = g_val - g_prev
                    if abs(dg) > 1e-12:
                        frac = -g_prev / dg
                        V_cross = (V - dV) + frac * dV
                    else:
                        V_cross = V
                    flutter_speed = V_cross
                    flutter_freq = f_val
                    flutter_mode_idx = i_mode
                    damping_margin = min_g_per_mode[i_mode]
                    logger.info(
                        f"pk_flutter: zero-crossing detected at "
                        f"V={V_cross:.1f} m/s, f={f_val:.2f} Hz, "
                        f"mode='{MODE_LABELS[i_mode]}'"
                    )

            prev_g[i_mode] = g_val

    # ── Determine envelope status ───────────────────────────────────────────
    if flutter_speed < float('inf'):
        flutter_mach = flutter_speed / a_sound if a_sound > 0 else 0.0
        if flutter_speed <= max_flight_speed:
            envelope_status = "WITHIN_ENVELOPE"
        else:
            envelope_status = "OUTSIDE_ENVELOPE"
        critical_mode = (MODE_LABELS[flutter_mode_idx]
                         if 0 <= flutter_mode_idx < 2 else "Unknown")
    else:
        flutter_mach = 0.0
        flutter_freq = 0.0
        envelope_status = "NOT_REACHED"
        critical_mode = ""
        damping_margin = min(min_g_per_mode)

    return {
        "flutter_speed_mps": flutter_speed,
        "flutter_mach": flutter_mach,
        "flutter_frequency_hz": flutter_freq,
        "critical_mode_name": critical_mode,
        "flutter_mechanism": "bending-torsion",
        "damping_margin": damping_margin,
        "envelope_status": envelope_status,
        "vg_data": vg_data,
        "vf_data": vf_data,
    }


def _empty_pk_result(max_flight_speed, a_sound):
    """Return a safe-default pk result dict when solver cannot run."""
    return {
        "flutter_speed_mps": float('inf'),
        "flutter_mach": 0.0,
        "flutter_frequency_hz": 0.0,
        "critical_mode_name": "",
        "flutter_mechanism": "bending-torsion",
        "damping_margin": 0.0,
        "envelope_status": "NOT_REACHED",
        "vg_data": [],
        "vf_data": [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  NACA TN-4197 EMPIRICAL FLUTTER SPEED  (Legacy — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def flutter_speed(span: float, root_chord: float, tip_chord: float,
                  thickness: float, shear_modulus: float,
                  altitude_m: float = 0.0) -> float:
    """Calculate flutter speed using NACA TN-4197 method.

    Parameters
    ----------
    span : float       Fin semi-span from body (m)
    root_chord : float Root chord (m)
    tip_chord : float  Tip chord (m)
    thickness : float  Fin thickness (m)
    shear_modulus : float  Shear modulus of fin material (Pa)
    altitude_m : float Altitude for atmospheric conditions (m)

    Returns
    -------
    float : Flutter speed (m/s)
    """
    from cfd.solvers.base import isa_conditions

    if span <= 0 or root_chord <= 0 or thickness <= 0:
        return float('inf')

    P, T, rho = isa_conditions(altitude_m)
    a = math.sqrt(1.4 * 287.05 * T)  # speed of sound

    # Planform area (trapezoidal)
    S = 0.5 * (root_chord + tip_chord) * span
    if S <= 0:
        return float('inf')

    # Aspect ratio
    AR = span ** 2 / S

    # Taper ratio
    lam = tip_chord / root_chord if root_chord > 0 else 0

    # Thickness-to-chord ratio (use mean chord)
    c_mean = (root_chord + tip_chord) / 2
    tc = thickness / c_mean if c_mean > 0 else 0.01

    if tc <= 0 or AR <= 0:
        return float('inf')

    # NACA TN-4197 flutter speed
    denom = 1.337 * AR**3 * P * (lam + 1) / (2 * (AR + 2) * tc**3)
    if denom <= 0:
        return float('inf')

    V_flutter = a * math.sqrt(shear_modulus / denom)
    return V_flutter


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def flutter_analysis(assembly, max_flight_speed: float = 300.0,
                     max_flight_mach: float = 1.0,
                     material_shear_modulus: float = None) -> FlutterResult:
    """Full flutter analysis for all fin sets in the assembly.

    Runs both the legacy NACA TN-4197 empirical method and the new 2-DOF
    p-k flutter solver on every fin set found.  The result reports the
    most critical (lowest flutter speed) fin across all altitudes.

    Parameters
    ----------
    assembly : RocketAssembly
    max_flight_speed : float  Maximum expected flight speed (m/s)
    max_flight_mach : float   Maximum expected Mach number
    material_shear_modulus : float  Override shear modulus (Pa); auto-detect if None
    """
    from core.components import TrapezoidalFinSet, BodyTube
    from structures.solvers.base import get_structural_material
    from cfd.solvers.base import isa_conditions

    result = FlutterResult()
    result.max_flight_speed = max_flight_speed
    result.max_flight_mach = max_flight_mach

    # ── Collect fin sets ────────────────────────────────────────────────────
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
        logger.warning("No fin sets found for flutter analysis.")
        result.flutter_speed_mps = float('inf')
        result.flutter_margin = float('inf')
        result.envelope_status = "NOT_REACHED"
        return result

    # ── Altitude sweep altitudes ────────────────────────────────────────────
    altitudes = [float(a) for a in range(0, 15001, 1000)]

    # ── Analyse each fin ────────────────────────────────────────────────────
    global_min_flutter = float('inf')
    global_best_pk = None
    global_worst_alt = 0.0
    global_worst_margin = float('inf')
    best_fin_name = ""

    for fin in fins:
        fin_name = getattr(fin, 'name', 'Fin Set')
        mat = get_structural_material(getattr(fin, 'material', 'Plywood (Birch)'))
        G = material_shear_modulus or mat.G
        E = mat.E
        density_mat = mat.density

        # ── NACA TN-4197 at sea level (legacy) ──────────────────────────
        V_f_naca = flutter_speed(
            span=fin.height,
            root_chord=fin.root_chord,
            tip_chord=fin.tip_chord,
            thickness=fin.thickness,
            shear_modulus=G,
            altitude_m=0.0,
        )

        # ── p-k solver at sea level ─────────────────────────────────────
        pk_sea = pk_flutter_analysis(
            span=fin.height,
            root_chord=fin.root_chord,
            tip_chord=fin.tip_chord,
            thickness=fin.thickness,
            E=E, G=G,
            density_mat=density_mat,
            altitude_m=0.0,
            max_flight_speed=max_flight_speed,
        )

        # Use the more conservative (lower) of NACA or p-k
        V_f_pk = pk_sea["flutter_speed_mps"]
        V_f_use = min(V_f_naca, V_f_pk)

        # Store per-fin results
        fin_result = {
            "fin_name": fin_name,
            "naca_flutter_speed_mps": V_f_naca,
            "pk_flutter_speed_mps": V_f_pk,
            "pk_flutter_freq_hz": pk_sea["flutter_frequency_hz"],
            "pk_critical_mode": pk_sea["critical_mode_name"],
            "pk_envelope_status": pk_sea["envelope_status"],
        }
        result.per_fin_results.append(fin_result)

        # Track global minimum
        if V_f_use < global_min_flutter:
            global_min_flutter = V_f_use
            global_best_pk = pk_sea
            best_fin_name = fin_name

        # ── Altitude sweep ──────────────────────────────────────────────
        for alt in altitudes:
            # NACA TN-4197 at this altitude
            V_f_alt = flutter_speed(fin.height, fin.root_chord, fin.tip_chord,
                                    fin.thickness, G, alt)

            P, T, _ = isa_conditions(alt)
            a_sound = math.sqrt(GAMMA_AIR * R_AIR * T)
            M_f = V_f_alt / a_sound if a_sound > 0 else 0.0
            margin_alt = V_f_alt / max_flight_speed if max_flight_speed > 0 else float('inf')

            result.altitude_sweep.append((alt, V_f_alt, M_f, fin_name))

            if margin_alt < global_worst_margin:
                global_worst_margin = margin_alt
                global_worst_alt = alt

            # Mach sweep entry
            result.mach_sweep.append((M_f, margin_alt, fin_name))

    # ── Populate result from global worst case ──────────────────────────────
    result.flutter_speed_mps = global_min_flutter

    _, T0, _ = isa_conditions(0)
    a0 = math.sqrt(GAMMA_AIR * R_AIR * T0)
    result.flutter_mach = global_min_flutter / a0 if (a0 > 0 and
                          global_min_flutter < float('inf')) else 0.0
    result.flutter_margin = (global_min_flutter / max_flight_speed
                             if max_flight_speed > 0 else float('inf'))
    result.safe = result.flutter_margin > 1.0

    # Worst-case altitude
    result.worst_case_altitude_m = global_worst_alt
    result.worst_case_margin = global_worst_margin

    # ── p-k specific results from most critical fin ─────────────────────────
    if global_best_pk is not None:
        result.vg_data = global_best_pk["vg_data"]
        result.vf_data = global_best_pk["vf_data"]
        result.flutter_frequency_hz = global_best_pk["flutter_frequency_hz"]
        result.critical_mode_name = global_best_pk["critical_mode_name"]
        result.flutter_mechanism = global_best_pk["flutter_mechanism"]
        result.damping_margin = global_best_pk["damping_margin"]
        result.envelope_status = global_best_pk["envelope_status"]

    # ── Logging ─────────────────────────────────────────────────────────────
    if global_min_flutter < float('inf'):
        logger.info(
            f"Flutter: V_f={global_min_flutter:.1f} m/s "
            f"(M={result.flutter_mach:.2f}), "
            f"margin={result.flutter_margin:.2f}, safe={result.safe}, "
            f"envelope={result.envelope_status}, "
            f"fin='{best_fin_name}', "
            f"f_flutter={result.flutter_frequency_hz:.1f} Hz, "
            f"mode='{result.critical_mode_name}', "
            f"worst_alt={global_worst_alt:.0f} m"
        )
    else:
        logger.info(
            f"Flutter: no flutter detected within analysis envelope; "
            f"margin={result.flutter_margin:.2f}, "
            f"envelope={result.envelope_status}"
        )

    return result
