"""
K2 Aerospace — High-Fidelity Aerodynamics Model (OpenRocket Port)
==================================================================
Engineering-grade aerodynamic calculations for rocket flight simulation.

Ported from OpenRocket's BarrowmanCalculator, FinSetCalc, and
SymmetricComponentCalc (GPL, Sampo Niskanen).

Implements:
    - Reynolds-based skin friction with Mach compressibility correction
    - Mach-dependent base drag and stagnation pressure drag
    - Full Barrowman method (nose + body + fins) for CN and CP
    - Supersonic fin CN with K1/K2/K3 tables (up to Mach 5)
    - Mach-dependent fin CP (quarter-chord → supersonic empirical)
    - Body lift (Galejs method)
    - Stall model (20° with graceful reduction)
    - Pitch/yaw damping (body + fin composite, OpenRocket method)
    - Roll damping moment
    - CD model: skin friction + base + pressure + wave + induced

References:
    - Barrowman, J.S. (1967)
    - Galejs body lift method
    - NASA TR-R-100 (nose pressure drag)
    - OpenRocket source code

All forces in Newtons, moments in N·m, angles in radians.
"""

import math
import numpy as np
import logging

from physics.drag_tables import (
    LinearInterpolator, stagnation_cd, base_cd,
    FIN_K1, FIN_K2, FIN_K3, SurfaceFinish, FinCrossSection,
    CNA_SUPERSONIC_MACH, GAMMA_AIR,
)

logger = logging.getLogger("K2.Aerodynamics")

# ── Constants ─────────────────────────────────────────────────────────────────
STALL_ANGLE = math.radians(20)      # 20° stall angle (OpenRocket)
CNA_SUBSONIC_MACH = 0.9             # subsonic regime boundary
BODY_LIFT_K = 1.1                   # Galejs body lift coefficient


# ══════════════════════════════════════════════════════════════════════════════
#  SKIN FRICTION (Reynolds-based, OpenRocket BarrowmanCalculator)
# ══════════════════════════════════════════════════════════════════════════════

def compute_skin_friction_cf(Re: float, mach: float,
                              perfect_finish: bool = False) -> float:
    """
    Skin friction coefficient using Schlichting's formula with Mach correction.
    Ported from OpenRocket's calculateFrictionCoefficient().
    """
    if Re < 1e4:
        Cf = 1.48e-2 if not perfect_finish else 1.33e-2
    elif perfect_finish and Re < 5.39e5:
        Cf = 1.328 / math.sqrt(Re)
    else:
        # Turbulent / transitional
        ln_Re = math.log(Re) if Re > 0 else 0
        Cf = 1.0 / (1.50 * ln_Re - 5.6) ** 2
        if perfect_finish and Re >= 5.39e5:
            Cf = Cf - 1700.0 / Re

    # Mach compressibility correction
    c1 = c2 = 1.0
    if perfect_finish:
        if mach < 1.1 and Re > 1e6:
            blend = min(1.0, (Re - 1e6) / 2e6)
            c1 = 1 - 0.1 * mach**2 * blend
        if mach > 0.9 and Re > 1e6:
            blend = min(1.0, (Re - 1e6) / 2e6)
            c2 = 1 + (1.0 / (1 + 0.045 * mach**2)**0.25 - 1) * blend
    else:
        if mach < 1.1:
            c1 = 1 - 0.1 * mach**2
        if mach > 0.9:
            c2 = 1.0 / (1 + 0.15 * mach**2)**0.58

    if mach < 0.9:
        Cf *= c1
    elif mach < 1.1:
        Cf *= c2 * (mach - 0.9) / 0.2 + c1 * (1.1 - mach) / 0.2
    else:
        Cf *= c2

    return max(Cf, 1e-6)


def roughness_limited_cf(roughness: float, body_length: float,
                          mach: float) -> float:
    """Roughness-limited friction coefficient (OpenRocket)."""
    if body_length <= 0 or roughness <= 0:
        return 0.0
    # Roughness correction for Mach
    if mach < 0.9:
        corr = 1 - 0.1 * mach**2
    elif mach > 1.1:
        corr = 1.0 / (1 + 0.18 * mach**2)
    else:
        c1 = 1 - 0.1 * 0.9**2
        c2 = 1.0 / (1 + 0.18 * 1.1**2)
        corr = c2 * (mach - 0.9) / 0.2 + c1 * (1.1 - mach) / 0.2
    return 0.032 * (roughness / body_length)**0.2 * corr


# ══════════════════════════════════════════════════════════════════════════════
#  NOSE CONE CN_alpha AND CP
# ══════════════════════════════════════════════════════════════════════════════

# Barrowman slender-body theory: nose-cone CN_alpha = 2.0 per radian for ALL
# shapes (it depends only on base area, not profile). Matches core.components.
NOSE_CN_MAP = {
    "conical": 2.0, "ogive": 2.0, "parabolic": 2.0,
    "elliptical": 2.0, "haack": 2.0, "power": 2.0,
}

def compute_nose_cn(nose_type: str = "ogive") -> float:
    """CN_alpha for the nose cone (per radian, slender body theory)."""
    return NOSE_CN_MAP.get(nose_type.lower(), 2.0)

def compute_nose_cp(nose_length: float, nose_type: str = "ogive") -> float:
    """CP location from nose tip (meters), Barrowman fractions of nose length.

    Cone CP = 2/3·L (0.666), ogive 0.466, parabolic 0.5, ellipsoid 0.333,
    LV-Haack 0.437. Cone value matches core.components.NoseCone (0.667).
    """
    fac = {"conical": 0.666, "ogive": 0.466, "parabolic": 0.500,
           "elliptical": 0.333, "haack": 0.437, "power": 0.400,
           }.get(nose_type.lower(), 0.466)
    return fac * nose_length


# ══════════════════════════════════════════════════════════════════════════════
#  BODY LIFT (Galejs method — OpenRocket SymmetricComponentCalc)
# ══════════════════════════════════════════════════════════════════════════════

def compute_body_lift_cn(planform_area: float, ref_area: float,
                          alpha: float, mach: float) -> float:
    """
    Body tube normal force coefficient from Galejs method.
    CN = K * A_planform / A_ref * sin(α)² / α
    """
    if ref_area <= 0 or abs(alpha) < 1e-9:
        return 0.0
    # Low-speed damping multiplier (OpenRocket: prevents instability at apogee)
    mul = 1.0
    if mach < 0.05 and abs(alpha) > math.pi / 4:
        mul = (mach / 0.05) ** 2
    sinc_alpha = math.sin(alpha) / alpha if abs(alpha) > 1e-9 else 1.0
    return mul * BODY_LIFT_K * planform_area / ref_area * math.sin(alpha) * sinc_alpha


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITION CN (Barrowman)
# ══════════════════════════════════════════════════════════════════════════════

def compute_transition_cn(fore_diam: float, aft_diam: float,
                           ref_area: float) -> float:
    """Barrowman CN_alpha for a conical transition."""
    if ref_area <= 0:
        return 0.0
    A0 = math.pi * (fore_diam / 2)**2
    A1 = math.pi * (aft_diam / 2)**2
    return 2.0 * (A1 - A0) / ref_area

def compute_transition_cp(length: float, fore_diam: float, aft_diam: float,
                            full_volume: float) -> float:
    """CP of transition from its fore end."""
    A0 = math.pi * (fore_diam / 2)**2
    A1 = math.pi * (aft_diam / 2)**2
    dA = A1 - A0
    if abs(dA) < 1e-12:
        return length / 2.0
    return (length * A1 - full_volume) / dA


# ══════════════════════════════════════════════════════════════════════════════
#  FIN SET CN_alpha (Sub/Trans/Supersonic — OpenRocket FinSetCalc)
# ══════════════════════════════════════════════════════════════════════════════

def compute_fin_cn_alpha(fin_count: int, fin_span: float, fin_root_chord: float,
                          fin_tip_chord: float, body_radius: float,
                          sweep_angle: float = 0.0, mach: float = 0.0,
                          alpha: float = 0.0) -> float:
    """
    Fin set CN_alpha using OpenRocket's FinSetCalc method.
    Supports subsonic, transonic, and supersonic regimes.
    """
    if fin_span <= 0 or body_radius <= 0 or fin_root_chord <= 0:
        return 0.0

    d = 2 * body_radius
    s = fin_span  # semi-span from body surface
    s_total = body_radius + s  # semi-span from centerline

    # Fin area and aspect ratio
    fin_area = 0.5 * (fin_root_chord + fin_tip_chord) * fin_span
    if fin_area < 1e-9:
        return 0.0
    ar = 2 * s**2 / fin_area

    # Mid-chord sweep
    sweep_len = s * math.tan(sweep_angle) if sweep_angle != 0 else 0.0
    lm = sweep_len + (fin_tip_chord / 2.0) - (fin_root_chord / 2.0)
    cos_gamma = s / math.sqrt(s**2 + lm**2) if s > 0 else 1.0
    if cos_gamma < 1e-9:
        return 0.0

    # Clamp alpha for stall
    eff_alpha = min(abs(alpha), math.pi - abs(alpha), STALL_ANGLE)
    ref_area_unit = math.pi * body_radius**2  # body cross-section

    # --- Subsonic regime ---
    if mach <= CNA_SUBSONIC_MACH:
        beta_sq = max(0.0, 1 - mach**2)
        denom = 1 + math.sqrt(1 + beta_sq * (s**2 / (fin_area * cos_gamma))**2)
        cna1 = 2 * math.pi * s**2 / denom
    # --- Supersonic regime ---
    elif mach >= CNA_SUPERSONIC_MACH:
        k1 = FIN_K1.get_value(mach)
        k2 = FIN_K2.get_value(mach)
        k3 = FIN_K3.get_value(mach)
        cna1 = fin_area * (k1 + k2 * eff_alpha + k3 * eff_alpha**2)
    # --- Transonic interpolation ---
    else:
        # Subsonic endpoint
        beta_sq_sub = max(0.0, 1 - CNA_SUBSONIC_MACH**2)
        sq = math.sqrt(1 + beta_sq_sub * (s**2 / (fin_area * cos_gamma))**2)
        sub_v = 2 * math.pi * s**2 / (1 + sq)
        # Supersonic endpoint
        k1s = FIN_K1.get_value(CNA_SUPERSONIC_MACH)
        k2s = FIN_K2.get_value(CNA_SUPERSONIC_MACH)
        k3s = FIN_K3.get_value(CNA_SUPERSONIC_MACH)
        sup_v = fin_area * (k1s + k2s * eff_alpha + k3s * eff_alpha**2)
        # Linear blend
        t = (mach - CNA_SUBSONIC_MACH) / (CNA_SUPERSONIC_MACH - CNA_SUBSONIC_MACH)
        cna1 = sub_v * (1 - t) + sup_v * t

    # Normalize to reference area
    if ref_area_unit > 0:
        cna1 /= ref_area_unit

    # Interference factor (Barrowman)
    tau = body_radius / s_total
    cna = cna1 * fin_count * (1 + tau)

    # Fin-fin interference for > 4 fins
    if fin_count == 5:
        cna *= 0.948
    elif fin_count == 6:
        cna *= 0.913
    elif fin_count == 7:
        cna *= 0.854
    elif fin_count >= 8:
        cna *= 0.81

    return cna


def compute_fin_cp(body_length: float, fin_root_chord: float,
                   fin_tip_chord: float, fin_span: float,
                   fin_leading_edge_sweep: float = 0.0,
                   mach: float = 0.0,
                   fin_position: float = 0.0) -> float:
    """
    Fin set CP location from nose tip.
    Mach-dependent: subsonic at quarter-MAC, supersonic empirical.
    """
    if fin_position > 0:
        x_f = fin_position
    else:
        x_f = body_length - fin_root_chord
        
    cr, ct, s = fin_root_chord, fin_tip_chord, fin_span

    # Compute MAC properties
    fin_area = 0.5 * (cr + ct) * s
    if fin_area < 1e-9 or s < 1e-9:
        return x_f + cr * 0.25

    ar = 2 * s**2 / fin_area
    sweep_len = s * math.tan(fin_leading_edge_sweep) if fin_leading_edge_sweep != 0 else 0.0
    lm = sweep_len + (ct / 2.0) - (cr / 2.0)

    # MAC lead position
    mac_lead = 0.0
    mac_length = 0.0
    area_sum = 0.0
    divs = 48
    dy = s / (divs - 1) if divs > 1 else s
    for i in range(divs):
        y_frac = i / (divs - 1) if divs > 1 else 0
        chord = cr + (ct - cr) * y_frac
        lead = sweep_len * y_frac
        mac_length += chord * chord
        mac_lead += lead * chord
        area_sum += chord
    if area_sum > 1e-9:
        mac_length = mac_length * dy / (area_sum * dy)
        mac_lead = mac_lead * dy / (area_sum * dy)
    else:
        mac_length = cr
        mac_lead = 0.0

    # CP position along MAC depends on Mach
    if mach <= 0.5:
        cp_frac = 0.25
    elif mach >= 2.0:
        beta = math.sqrt(max(1e-6, mach**2 - 1))
        cp_frac = (ar * beta - 0.67) / max(2 * ar * beta - 1, 0.01)
        cp_frac = max(0.1, min(0.6, cp_frac))
    else:
        # Interpolate between 0.25 and the supersonic value
        t = (mach - 0.5) / 1.5
        beta2 = math.sqrt(max(1e-6, 4.0 - 1))
        sup_frac = (ar * beta2 - 0.67) / max(2 * ar * beta2 - 1, 0.01)
        sup_frac = max(0.1, min(0.6, sup_frac))
        cp_frac = 0.25 * (1 - t) + sup_frac * t

    return x_f + mac_lead + cp_frac * mac_length


# ══════════════════════════════════════════════════════════════════════════════
#  COMPLETE CD MODEL (OpenRocket-style decomposition)
# ══════════════════════════════════════════════════════════════════════════════

def compute_cd(mach: float, alpha: float = 0.0, fineness_ratio: float = 10.0,
               base_area_ratio: float = 0.1, Re: float = 1e6,
               nose_type: str = "ogive",
               fin_thickness: float = 0.003, fin_span: float = 0.1,
               fin_mac_length: float = 0.1, fin_area: float = 0.01,
               ref_area: float = 0.005,
               surface_roughness: float = SurfaceFinish.NORMAL,
               body_length: float = 1.0) -> float:
    """Total drag coefficient with all OpenRocket drag components."""
    mach = max(0.0, mach)

    # 1. Skin friction
    Cf = compute_skin_friction_cf(Re, mach)
    Cf_rough = roughness_limited_cf(surface_roughness, body_length, mach)
    Cf = max(Cf, Cf_rough)
    # Body friction: Cf * wetted_area / ref_area, approximated
    cd_friction = Cf * (1 + 1.0 / (2 * fineness_ratio)) * 4 * fineness_ratio

    # 2. Fin friction (both sides)
    if fin_area > 0 and fin_mac_length > 0 and ref_area > 0:
        cd_fin_friction = Cf * (1 + 2 * fin_thickness / fin_mac_length) * 2 * fin_area / ref_area
    else:
        cd_fin_friction = 0.0

    # 3. Base drag
    cd_base = base_cd(mach) * base_area_ratio

    # 4. Induced drag from AoA — smooth in alpha (the old 1° on/off gate put a
    #    step discontinuity in CD that optimizer/sensitivity sweeps tripped on)
    cd_induced = 2.0 * math.sin(alpha)**2

    # 5. Transonic / supersonic wave + nose-pressure drag.
    #    Zero subsonic; rises through the transonic drag-divergence region and
    #    holds a supersonic plateau. Driven by the stagnation-pressure factor
    #    (captures the M=1 jump) and scaled inversely by slenderness — a longer,
    #    finer body sheds less wave drag. Referenced to frontal (ref) area.
    stag = stagnation_cd(mach)          # ~0.85 (M→0) → ~1.5 (supersonic)
    stag0 = 0.85                        # incompressible reference level
    fn = max(fineness_ratio, 3.0)
    if mach <= 0.8:
        cd_wave = 0.0
    elif mach < 1.1:
        t = (mach - 0.8) / 0.3
        cd_wave = (3 * t * t - 2 * t ** 3) * max(0.0, stag - stag0) / fn
    else:
        cd_wave = max(0.0, stag - stag0) / fn

    total_cd = cd_friction + cd_fin_friction + cd_base + cd_induced + cd_wave
    return min(total_cd, 3.0)  # Cap at 3.0 to prevent divergence


def compute_cn(cn_alpha_total: float, alpha: float) -> float:
    """Total normal force coefficient with stall model."""
    eff_alpha = min(abs(alpha), STALL_ANGLE)
    cn = cn_alpha_total * math.sin(eff_alpha)
    return cn * math.copysign(1, alpha)


# ══════════════════════════════════════════════════════════════════════════════
#  DAMPING MOMENTS (OpenRocket composite method)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pitch_damping(body_length: float, body_diameter: float,
                           cg: float, ref_area: float, ref_length: float,
                           fin_areas: list = None, fin_mid_positions: list = None,
                           fin_counts: list = None) -> float:
    """
    OpenRocket-style pitch damping multiplier.
    Combines body + fin contributions.
    """
    if ref_area <= 0 or ref_length <= 0:
        return 0.0

    # Body contribution
    d_avg = body_diameter
    mul = 0.275 * d_avg / (ref_area * ref_length)
    mul *= (cg**4 + (body_length - cg)**4)

    # Fin contributions
    if fin_areas and fin_mid_positions and fin_counts:
        for area, x_mid, count in zip(fin_areas, fin_mid_positions, fin_counts):
            n = min(count, 4)
            mul += 0.6 * n * area * abs(x_mid - cg)**3 / (ref_area * ref_length)

    # OpenRocket applies a 3x multiplier for more realistic apogee turn
    mul *= 3.0

    return mul


def compute_pitching_moment_coefficient(cn_alpha: float, cp: float, cg: float,
                                         ref_diam: float, alpha: float) -> float:
    """Pitching moment coefficient about the CG."""
    if ref_diam <= 0:
        return 0.0
    moment_arm = (cp - cg) / ref_diam
    return -cn_alpha * moment_arm * math.sin(min(abs(alpha), STALL_ANGLE)) * math.copysign(1, alpha)


def compute_pitch_damping_moment(damping_mul: float, pitch_rate: float,
                                  v_rel: float, q_dyn: float,
                                  ref_area: float, ref_length: float) -> float:
    """Pitch damping moment in N·m."""
    if v_rel <= 1.0:
        return 0.0
    return -math.copysign(1, pitch_rate) * min(
        damping_mul * (pitch_rate / v_rel)**2,
        abs(q_dyn * ref_area * ref_length * 0.5)
    ) * q_dyn * ref_area * ref_length


# ══════════════════════════════════════════════════════════════════════════════
#  AeroModel — Main Interface Class
# ══════════════════════════════════════════════════════════════════════════════

class AeroModel:
    """
    High-fidelity aerodynamic model for a rocket vehicle.
    Uses OpenRocket-equivalent physics for all subsystems.
    """

    def __init__(self, nose_type="ogive", nose_length=0.3,
                 body_length=2.0, body_diameter=0.08,
                 fin_count=4, fin_span=0.1,
                 fin_root_chord=0.15, fin_tip_chord=0.05,
                 fin_sweep=0.0, cmq=-20.0,
                 surface_finish="Normal",
                 fin_cross_section="Rounded",
                 fin_thickness=0.003,
                 fin_position=0.0):
        self.nose_type = nose_type
        self.nose_length = nose_length
        self.body_length = body_length
        self.body_diameter = body_diameter
        self.body_radius = body_diameter / 2.0
        self.fin_count = fin_count
        self.fin_span = fin_span
        self.fin_root_chord = fin_root_chord
        self.fin_tip_chord = fin_tip_chord
        self.fin_sweep = fin_sweep
        self.cmq = cmq
        self.fin_thickness = fin_thickness
        self.surface_finish = surface_finish
        self.fin_cross_section = fin_cross_section
        self.fin_position = fin_position

        self.ref_area = math.pi * self.body_radius ** 2
        self.fineness = body_length / body_diameter if body_diameter > 0 else 10.0

        # Body planform area (side projection)
        self.body_planform_area = body_length * body_diameter
        self.body_planform_center = body_length / 2.0

        # Fin area and MAC
        self.fin_area = 0.5 * (fin_root_chord + fin_tip_chord) * fin_span
        self.fin_mac_length = (fin_root_chord + fin_tip_chord) / 2.0

        # Pre-compute subsonic CN_alpha
        self._cn_alpha_nose = compute_nose_cn(nose_type)
        self._cp_nose = compute_nose_cp(nose_length, nose_type)

        # Damping parameters
        self._roughness = SurfaceFinish.get_roughness(surface_finish)

        self.cn_alpha_total = self._cn_alpha_nose  # updated in compute()

        logger.debug(
            f"AeroModel init: nose={nose_type}, L={body_length}m, "
            f"D={body_diameter}m, fins={fin_count}×{fin_span}m"
        )

    def cp_subsonic(self) -> float:
        """CP location at subsonic speed."""
        cn_fins = compute_fin_cn_alpha(
            self.fin_count, self.fin_span, self.fin_root_chord,
            self.fin_tip_chord, self.body_radius, self.fin_sweep, 0.0
        )
        cp_fins = compute_fin_cp(
            self.body_length, self.fin_root_chord, self.fin_tip_chord,
            self.fin_span, self.fin_sweep, 0.0, self.fin_position
        )
        cn_total = self._cn_alpha_nose + cn_fins
        if cn_total <= 0:
            return self.body_length * 0.5
        return (self._cn_alpha_nose * self._cp_nose + cn_fins * cp_fins) / cn_total

    def stability_margin(self, cp: float, cg: float) -> float:
        if self.body_diameter <= 0:
            return 0.0
        return (cp - cg) / self.body_diameter

    def compute(self, alpha, mach, q_dyn, pitch_rate, v_rel, cg) -> dict:
        """Compute all aerodynamic forces and moments."""
        A = self.ref_area
        d = self.body_diameter

        # Reynolds number from the LOCAL flow state (altitude-correct without
        # needing altitude passed in): recover ρ from q=½ρv², T from the local
        # speed of sound a=v_rel/mach, and μ via Sutherland. ν=μ/ρ, Re=v·L/ν.
        # Falls back to sea-level kinematic viscosity if the state is degenerate.
        if not hasattr(self, '_atm'):
            from environment.atmosphere_model import Atmosphere
            self._atm = Atmosphere()

        if v_rel > 1.0 and mach > 1e-3 and q_dyn > 1.0:
            a_sound = v_rel / mach
            T_local = a_sound ** 2 / (1.4 * 287.058)
            mu_local = 1.458e-6 * T_local ** 1.5 / (T_local + 110.4)  # Sutherland
            rho_local = 2.0 * q_dyn / (v_rel ** 2)
            nu = mu_local / rho_local if rho_local > 1e-9 else self._atm.kinematic_viscosity(0)
        else:
            nu = self._atm.kinematic_viscosity(0)
        Re = v_rel * self.body_length / max(nu, 1e-9)

        # Fin CN_alpha (Mach-aware)
        cn_fins = compute_fin_cn_alpha(
            self.fin_count, self.fin_span, self.fin_root_chord,
            self.fin_tip_chord, self.body_radius, self.fin_sweep,
            mach, alpha
        )
        cp_fins = compute_fin_cp(
            self.body_length, self.fin_root_chord, self.fin_tip_chord,
            self.fin_span, self.fin_sweep, mach, self.fin_position
        )

        # Body lift
        cn_body = compute_body_lift_cn(
            self.body_planform_area, A, alpha, mach
        )
        cp_body = self.body_planform_center

        # Nose
        cn_nose = self._cn_alpha_nose
        cp_nose_val = self._cp_nose

        # Total CN_alpha and CP
        cn_total = cn_nose + cn_fins + cn_body
        self.cn_alpha_total = cn_total

        if cn_total > 1e-9:
            cp = (cn_nose * cp_nose_val + cn_fins * cp_fins + cn_body * cp_body) / cn_total
        else:
            cp = self.body_length * 0.5
        # CP must lie on the physical body — the supersonic fin-CP / body-lift
        # terms can otherwise overshoot past the tail and inject a spurious
        # restoring-moment spike at transonic Mach.
        cp = max(0.0, min(cp, self.body_length))

        # CD (full model)
        cd = compute_cd(
            mach, alpha, self.fineness, 0.1, Re,
            self.nose_type, self.fin_thickness, self.fin_span,
            self.fin_mac_length, self.fin_area, A,
            self._roughness, self.body_length
        )

        # CN with stall
        cn = compute_cn(cn_total, alpha)

        # Cm
        cm = compute_pitching_moment_coefficient(cn_total, cp, cg, d, alpha)

        # Pitch damping — fin-dominated, applied in the standard linear form.
        #   Cmq = -2·CNα_fin·(arm/d)²   (per rad, about the CG)
        #   M_damp = Cmq · (ω·d / 2V) · q·A·d
        # The previous quadratic ∝(ω/V)² damping collapsed to ~zero at high
        # speed, leaving the high-q weathercock essentially undamped (ζ≈0.0006)
        # → divergent overshoot/tumbling in any crosswind. This linear Cmq form
        # gives a realistic ζ≈0.05 that keeps the airframe stable.
        cmq = self._pitch_damping_coeff(cn_fins, cp_fins, cg, d)
        m_damp = self._damping_moment(cmq, pitch_rate, v_rel, q_dyn, A, d, cn_total)

        # Forces
        F_drag = q_dyn * A * cd
        F_normal = q_dyn * A * abs(cn)
        M_pitch = cm * q_dyn * A * d + m_damp

        sm = self.stability_margin(cp, cg)

        return {
            "cd": cd, "cn": cn, "cm": cm, "cp": cp,
            "stability_margin": sm,
            "F_drag": F_drag, "F_normal": F_normal, "M_pitch": M_pitch,
            # nondim pitch-damping coefficient so the engine can damp yaw too
            "cmq": cmq, "cn_total": cn_total,
        }

    @staticmethod
    def _pitch_damping_coeff(cn_fins, cp_fins, cg, d) -> float:
        """Combined pitch damping derivative (Cmq + Cmα̇) for the fin set, per
        rad, about the CG. A tail fin damps through both pitch rate (Cmq) and
        the downwash-lag α̇ term; for a tail surface these are comparable, so
        the standard estimate is ≈4·CNα_fin·(arm/d)² total. A small body
        baseline is added. Negative = damping."""
        if d <= 0:
            return -1.0
        arm = (cp_fins - cg) / d
        return -4.0 * max(cn_fins, 0.0) * arm * arm - 1.0

    @staticmethod
    def _damping_moment(cmq, rate, v_rel, q_dyn, A, d, cn_total) -> float:
        """Standard linear damping moment from a nondimensional rate derivative,
        bounded by the restoring-moment scale to prevent single-step overshoot."""
        if v_rel <= 1.0 or q_dyn <= 0.0:
            return 0.0
        rate_hat = rate * d / (2.0 * v_rel)         # nondim pitch/yaw rate
        m = cmq * rate_hat * q_dyn * A * d           # opposes 'rate' (cmq<0)
        cap = abs(cn_total) * q_dyn * A * d          # restoring-moment scale
        if cap <= 0:
            return m
        return max(-cap, min(cap, m))

    @classmethod
    def from_state(cls, s) -> "AeroModel":
        """Construct AeroModel from the rocket state object."""
        # Use 'or' so zero-valued fields fall through to geometry-based defaults
        fin_span = getattr(s, 'fin_span', 0) or getattr(s, 'fin_height', 0) or s.diameter * 0.6
        fin_rc = getattr(s, 'fin_root_chord', 0) or s.length * 0.08
        fin_tc = getattr(s, 'fin_tip_chord', 0) or fin_rc * 0.5
        return cls(
            nose_type=getattr(s, 'nose_type', 'ogive'),
            nose_length=getattr(s, 'nose_length', 0) or s.length * 0.2,
            body_length=s.length,
            body_diameter=s.diameter,
            fin_count=getattr(s, 'fin_count', 4) or 4,
            fin_span=fin_span,
            fin_root_chord=fin_rc,
            fin_tip_chord=fin_tc,
            fin_sweep=getattr(s, 'fin_sweep_angle', 0.0),
            cmq=getattr(s, 'cmq', -20.0),
            surface_finish=getattr(s, 'surface_finish', 'Normal'),
            fin_cross_section=getattr(s, 'fin_cross_section', 'Rounded'),
            fin_thickness=getattr(s, 'fin_thickness', 0.003) or 0.003,
            fin_position=getattr(s, 'fin_position', 0.0),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY COMPATIBILITY WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_drag_coefficient(mach: float, fineness_ratio: float = 10.0) -> float:
    """Backward-compatible wrapper."""
    return compute_cd(mach, 0.0, fineness_ratio)

def compute_drag_force(velocity, altitude, diameter, cd) -> float:
    """Backward-compatible wrapper."""
    from environment.atmosphere_model import air_density_at_altitude
    rho = air_density_at_altitude(altitude)
    area = math.pi * (diameter / 2) ** 2
    return 0.5 * rho * velocity ** 2 * cd * area

def compute_cp_position(nose_length, body_length, fin_root_chord,
                         fin_height, fin_tip_chord, body_radius,
                         fin_count=4) -> float:
    """Backward-compatible CP wrapper."""
    cn_nose = compute_nose_cn("ogive")
    cn_fins = compute_fin_cn_alpha(fin_count, fin_height, fin_root_chord,
                                    fin_tip_chord, body_radius)
    cp_nose = compute_nose_cp(nose_length, "ogive")
    cp_fins = compute_fin_cp(body_length, fin_root_chord, fin_tip_chord, fin_height)
    cn_total = cn_nose + cn_fins
    if cn_total > 0:
        return (cn_nose * cp_nose + cn_fins * cp_fins) / cn_total
    return body_length * 0.5

def compute_stability_margin(cp, cg, diameter) -> float:
    """Backward-compatible stability margin wrapper."""
    if diameter <= 0:
        return 0.0
    return (cp - cg) / diameter
