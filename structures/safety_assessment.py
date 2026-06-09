"""
K2 Aerospace — Structural Safety Assessment
=============================================
Computes factor-of-safety (FoS) for every failure mode relevant to
aerospace thin-wall structures and identifies the governing mode.

Failure modes analysed
----------------------
1. **Yield**      — σ_yield / σ_max
2. **Ultimate**   — σ_ult  / σ_max
3. **Buckling**   — min(Euler column, NASA SP-8007 shell buckling)
4. **Fatigue**    — S_e(Marin) / σ_alternating
5. **Thermal**    — T_service_limit / T_max  (absolute kelvin)

References
----------
- Peterson's Stress Concentration Factors, 4th ed.
- NASA SP-8007, "Buckling of Thin-Walled Circular Cylinders", 1968
- Shigley & Mischke, *Mechanical Engineering Design*, ch. 6 (Marin eq.)
- Roark's Formulas for Stress and Strain, 8th ed.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("K2.Safety")


# ── Safety Assessment Dataclass ──────────────────────────────────────────────

@dataclass
class SafetyAssessment:
    """Comprehensive structural safety assessment across all failure modes.

    Each ``*_fos`` field stores the factor of safety for that mode.
    ``governing_mode`` / ``governing_fos`` identify the weakest link.

    Status thresholds (per NASA-STD-5001B Table 1 intent):
        SAFE      — FoS ≥ 3.0
        ADEQUATE  — 1.5 ≤ FoS < 3.0
        MARGINAL  — 1.0 ≤ FoS < 1.5
        FAILURE   — FoS < 1.0
    """
    yield_fos: float = 0.0           # σ_yield / σ_max
    ultimate_fos: float = 0.0        # σ_ultimate / σ_max
    buckling_fos: float = 0.0        # min(P_euler/P, σ_shell_crit/σ_axial)
    fatigue_fos: float = 0.0         # S_endurance / σ_alternating
    thermal_fos: float = 0.0         # T_service_limit / T_max (in K)

    governing_mode: str = ''         # 'yield'|'ultimate'|'buckling'|'fatigue'|'thermal'
    governing_fos: float = 0.0      # min of all FoS

    status: str = ''                 # 'SAFE'|'ADEQUATE'|'MARGINAL'|'FAILURE'
    status_color: str = ''           # hex color for UI

    margin_of_safety: float = 0.0   # governing_fos / SF_req - 1

    details: dict = field(default_factory=dict)  # per-mode breakdown


# ── Status Thresholds ────────────────────────────────────────────────────────

_STATUS_THRESHOLDS = [
    (3.0,  'SAFE',      '#2ecc71'),   # green
    (1.5,  'ADEQUATE',  '#f39c12'),   # amber
    (1.0,  'MARGINAL',  '#e67e22'),   # orange
    (0.0,  'FAILURE',   '#e74c3c'),   # red
]


def _classify_status(fos: float) -> tuple:
    """Return (status_str, hex_color) for a given FoS value.

    Thresholds
    ----------
    ≥ 3.0  → SAFE      (green  #2ecc71)
    ≥ 1.5  → ADEQUATE  (amber  #f39c12)
    ≥ 1.0  → MARGINAL  (orange #e67e22)
    < 1.0  → FAILURE   (red    #e74c3c)
    """
    for threshold, label, colour in _STATUS_THRESHOLDS:
        if fos >= threshold:
            return label, colour
    return 'FAILURE', '#e74c3c'


# ── Fatigue Endurance Limit (Marin Equation) ─────────────────────────────────

def _fatigue_endurance_limit(material, surface_finish: str = 'machined') -> float:
    r"""Compute modified endurance limit using the Marin equation.

    .. math::
        S_e = k_a \cdot k_b \cdot S_e'

    where the *un-modified* endurance limit is:

    - **Steel**:        ``S_e' = 0.5 × σ_ult``  (Shigley ch. 6)
    - **Aluminium / Titanium**: ``S_e' = 0.4 × σ_ult``
    - **Other**:        uses ``material.fatigue_endurance_factor × σ_ult``

    Surface-finish (Marin *k_a*) factors (machined ground surface):
        machined   → 0.70
        as-built   → 0.50
        polished   → 0.90

    Size factor *k_b* is taken as 0.85 for typical rocket airframe
    diameters (50–200 mm, Shigley Eq. 6-20).

    Parameters
    ----------
    material : StructuralMaterial
        Material dataclass (must have ``ultimate_strength``,
        optionally ``fatigue_endurance_factor``).
    surface_finish : str
        ``'machined'`` (default), ``'as-built'``, or ``'polished'``.

    Returns
    -------
    float
        Modified endurance limit S_e (Pa).
    """
    # ── Un-modified endurance limit S_e' ──
    sigma_ult = material.ultimate_strength

    # Use material-specific factor if available, else classify by name
    endurance_factor = getattr(material, 'fatigue_endurance_factor', None)
    if endurance_factor is not None and endurance_factor > 0:
        se_prime = endurance_factor * sigma_ult
    else:
        mat_lower = material.name.lower()
        if 'steel' in mat_lower:
            se_prime = 0.5 * sigma_ult       # Shigley ch. 6
        elif 'aluminum' in mat_lower or 'aluminium' in mat_lower or 'titanium' in mat_lower:
            se_prime = 0.4 * sigma_ult
        else:
            se_prime = 0.35 * sigma_ult       # conservative default

    # ── Marin surface-finish factor k_a ──
    surface_factors = {
        'polished': 0.90,
        'machined': 0.70,
        'as-built': 0.50,
    }
    k_a = surface_factors.get(surface_finish, 0.70)

    # ── Marin size factor k_b (Shigley Eq. 6-20, 50-200 mm range) ──
    k_b = 0.85

    se = k_a * k_b * se_prime
    return se


# ── Euler Column Buckling ────────────────────────────────────────────────────

def _euler_buckling_load(E: float, I: float, L: float,
                         k_eff: float = 2.0) -> float:
    r"""Classical Euler column buckling load.

    .. math::
        P_{cr} = \frac{\pi^2 E I}{(k_{eff} L)^2}

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    I : float
        Minimum second moment of area (m⁴).
    L : float
        Column length (m).
    k_eff : float
        Effective-length factor (2.0 = cantilever, 1.0 = pinned–pinned,
        0.5 = fixed–fixed).  Default 2.0 (rocket on launch rail ≈ cantilever).

    Returns
    -------
    float
        Critical Euler buckling load P_cr (N).  Returns ``inf`` when
        L ≤ 0.
    """
    if L <= 0:
        return float('inf')
    le = k_eff * L
    return (math.pi ** 2 * E * I) / (le ** 2)


# ── Shell Buckling (NASA SP-8007) ────────────────────────────────────────────

def _shell_buckling_stress(E: float, nu: float, R: float,
                           t: float) -> float:
    r"""Critical buckling stress for a thin cylindrical shell under
    axial compression with NASA SP-8007 empirical knockdown.

    Classical critical stress (Donnel, 1934):

    .. math::
        \sigma_{cl} = \frac{E}{\sqrt{3(1-\nu^2)}} \cdot \frac{t}{R}

    NASA SP-8007 empirical knockdown factor γ (Eq. 4):

    .. math::
        \gamma = 1 - 0.902 \left(1 - e^{-\frac{1}{16}\sqrt{\frac{R}{t}}}\right)

    Practical critical stress:

    .. math::
        \sigma_{cr} = \gamma \cdot \sigma_{cl}

    Parameters
    ----------
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    R : float
        Shell mid-surface radius (m).
    t : float
        Shell wall thickness (m).

    Returns
    -------
    float
        Critical shell buckling stress σ_cr (Pa).  Returns ``inf``
        when *R* or *t* is non-positive.
    """
    if R <= 0 or t <= 0:
        return float('inf')

    # Classical buckling stress
    sigma_cl = E / math.sqrt(3.0 * (1.0 - nu ** 2)) * (t / R)

    # NASA SP-8007 knockdown factor  (Eq. 4, 1968 revision)
    gamma = 1.0 - 0.902 * (1.0 - math.exp(-1.0 / 16.0 * math.sqrt(R / t)))

    sigma_cr = gamma * sigma_cl
    return sigma_cr


# ── Main Compute Function ────────────────────────────────────────────────────

def compute_safety(
    fem_result,
    material,
    config,
    *,
    surface_finish: str = 'machined',
    length_m: float = 1.0,
    radius_m: float = 0.05,
    thickness_m: float = 0.002,
    moment_of_inertia_m4: float = 0.0,
    alternating_stress_pa: float = 0.0,
) -> SafetyAssessment:
    """Compute a comprehensive safety assessment for a structural analysis.

    Evaluates five independent failure modes and identifies the governing
    (lowest) factor of safety.

    Parameters
    ----------
    fem_result : FEMResult
        Completed FEM result with stress / displacement / buckling fields.
    material : StructuralMaterial
        Material dataclass for the primary structure.
    config : FEMConfig
        Analysis configuration (carries ``safety_factor_required``).
    surface_finish : str
        ``'machined'``, ``'as-built'``, or ``'polished'`` for Marin fatigue
        surface-finish derating.
    length_m : float
        Representative structural length (m) for Euler buckling.
    radius_m : float
        Mid-surface shell radius (m) for shell buckling.
    thickness_m : float
        Wall thickness (m) for shell buckling.
    moment_of_inertia_m4 : float
        Cross-section second moment of area (m⁴).  If 0, estimated from
        thin-wall cylinder: ``I = π R³ t``.
    alternating_stress_pa : float
        Alternating stress amplitude for fatigue.  If 0, estimated as
        ``0.5 × max_von_mises`` (fully-reversed assumption).

    Returns
    -------
    SafetyAssessment
        Fully-populated assessment dataclass.
    """
    sa = SafetyAssessment()
    details: dict = {}
    sf_req = getattr(config, 'safety_factor_required', 2.0)

    sigma_max = max(fem_result.max_von_mises, 1e-6)  # avoid /0

    # ── 1. Yield FoS ────────────────────────────────────────────────────
    sigma_y = material.yield_strength
    sa.yield_fos = sigma_y / sigma_max
    details['yield'] = {
        'sigma_yield_Pa': sigma_y,
        'sigma_max_Pa': sigma_max,
        'fos': sa.yield_fos,
    }
    logger.debug("Yield FoS = %.2f  (σ_y=%.1f MPa, σ_max=%.1f MPa)",
                 sa.yield_fos, sigma_y / 1e6, sigma_max / 1e6)

    # ── 2. Ultimate FoS ────────────────────────────────────────────────
    sigma_ult = material.ultimate_strength
    sa.ultimate_fos = sigma_ult / sigma_max
    details['ultimate'] = {
        'sigma_ult_Pa': sigma_ult,
        'sigma_max_Pa': sigma_max,
        'fos': sa.ultimate_fos,
    }
    logger.debug("Ultimate FoS = %.2f", sa.ultimate_fos)

    # ── 3. Buckling FoS ────────────────────────────────────────────────
    # 3a. Euler column
    I_cs = moment_of_inertia_m4
    if I_cs <= 0 and radius_m > 0 and thickness_m > 0:
        # Thin-wall cylinder approximation  I = π R³ t
        I_cs = math.pi * radius_m ** 3 * thickness_m

    axial_force = getattr(fem_result, 'max_axial_stress', 0.0)
    # Use load-case axial force if available in config
    P_applied = 0.0
    lc = getattr(config, 'load_case', None)
    if lc is not None:
        P_applied = getattr(lc, 'axial_force', 0.0)

    # Effective length factor from modal_bc if available
    modal_bc = getattr(config, 'modal_bc', 'cantilever')
    k_eff_map = {
        'cantilever': 2.0,
        'pinned-pinned': 1.0,
        'fixed-fixed': 0.5,
        'fixed-pinned': 0.7,
    }
    k_eff = k_eff_map.get(modal_bc, 2.0)

    P_euler = _euler_buckling_load(material.E, I_cs, length_m, k_eff=k_eff)
    euler_fos = P_euler / max(abs(P_applied), 1.0)

    # 3b. Shell buckling (NASA SP-8007)
    sigma_axial = abs(fem_result.max_axial_stress) if fem_result.max_axial_stress != 0 else sigma_max
    sigma_shell_cr = _shell_buckling_stress(material.E, material.nu, radius_m, thickness_m)
    shell_fos = sigma_shell_cr / max(sigma_axial, 1.0)

    # 3c. FEM eigenvalue buckling factor if available
    blf = getattr(fem_result, 'buckling_load_factor', 0.0)

    sa.buckling_fos = min(euler_fos, shell_fos)
    if blf > 0:
        sa.buckling_fos = min(sa.buckling_fos, blf)

    details['buckling'] = {
        'euler_critical_load_N': P_euler,
        'euler_fos': euler_fos,
        'shell_critical_stress_Pa': sigma_shell_cr,
        'shell_fos': shell_fos,
        'fem_buckling_load_factor': blf,
        'combined_fos': sa.buckling_fos,
        'k_effective_length': k_eff,
    }
    logger.debug("Buckling FoS = %.2f  (Euler=%.2f, Shell=%.2f, BLF=%.2f)",
                 sa.buckling_fos, euler_fos, shell_fos, blf)

    # ── 4. Fatigue FoS (Marin equation) ─────────────────────────────────
    se = _fatigue_endurance_limit(material, surface_finish=surface_finish)

    sigma_alt = alternating_stress_pa
    if sigma_alt <= 0:
        # Fully-reversed assumption: σ_alt ≈ 0.5 × σ_max
        sigma_alt = 0.5 * sigma_max

    sa.fatigue_fos = se / max(sigma_alt, 1.0)
    details['fatigue'] = {
        'endurance_limit_Pa': se,
        'sigma_alternating_Pa': sigma_alt,
        'surface_finish': surface_finish,
        'fos': sa.fatigue_fos,
    }
    logger.debug("Fatigue FoS = %.2f  (S_e=%.1f MPa, σ_alt=%.1f MPa)",
                 sa.fatigue_fos, se / 1e6, sigma_alt / 1e6)

    # ── 5. Thermal FoS ─────────────────────────────────────────────────
    T_max = 293.15  # default ambient
    lc = getattr(config, 'load_case', None)
    if lc is not None:
        T_max = max(getattr(lc, 'wall_temp_K', 293.15), 293.15)

    # Also check station temperatures from FEM result
    if fem_result.station_temperatures:
        T_from_fem = max(T for _, T in fem_result.station_temperatures)
        T_max = max(T_max, T_from_fem)

    T_service = material.max_service_temp
    sa.thermal_fos = T_service / max(T_max, 1.0)
    details['thermal'] = {
        'T_service_K': T_service,
        'T_max_K': T_max,
        'fos': sa.thermal_fos,
    }
    logger.debug("Thermal FoS = %.2f  (T_service=%.0f K, T_max=%.0f K)",
                 sa.thermal_fos, T_service, T_max)

    # ── Governing Mode ──────────────────────────────────────────────────
    modes = {
        'yield':    sa.yield_fos,
        'ultimate': sa.ultimate_fos,
        'buckling': sa.buckling_fos,
        'fatigue':  sa.fatigue_fos,
        'thermal':  sa.thermal_fos,
    }

    governing = min(modes, key=modes.get)
    sa.governing_mode = governing
    sa.governing_fos = modes[governing]

    # ── Status Classification ───────────────────────────────────────────
    sa.status, sa.status_color = _classify_status(sa.governing_fos)

    # ── Margin of Safety (MoS = FoS / SF_req - 1) ──────────────────────
    sa.margin_of_safety = (sa.governing_fos / sf_req) - 1.0 if sf_req > 0 else 0.0

    sa.details = details

    logger.info(
        "Safety Assessment: %s  (governing=%s, FoS=%.2f, MoS=%+.2f)",
        sa.status, sa.governing_mode, sa.governing_fos, sa.margin_of_safety,
    )

    return sa
