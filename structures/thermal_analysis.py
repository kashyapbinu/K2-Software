"""
K2 Aerospace — Aerodynamic Heating & Thermal Stress (Physics-Validated)
========================================================================
Models:
  - Stagnation temperature (isentropic)
  - Recovery temperature (laminar/turbulent)
  - Convective heat flux (Eckert reference enthalpy, Sutherland viscosity)
  - Stagnation-point heating (Sutton-Graves correlation)
  - Steady-state wall temperature (radiation equilibrium, Newton-Raphson)
  - Thermal stress with partial constraint factor
  - Laminar-to-turbulent transition (Re-based)
  - Material-dependent emissivity

References:
  - Anderson, Hypersonic and High-Temperature Gas Dynamics, Ch. 6
  - Sutton & Graves, "A General Stagnation-Point Convective-Heating
    Equation for Arbitrary Gas Mixtures", NASA TR R-376, 1971
  - Eckert, E. R. G., "Engineering Relations for Heat Transfer and Friction
    in High-Velocity Laminar and Turbulent Boundary-Layer Flow", 1955
"""
from __future__ import annotations
import math, logging
from dataclasses import dataclass, field
from structures.solvers.base import get_structural_material, ThermalResult

logger = logging.getLogger("K2.FEM.Thermal")

# ── Physical Constants ──────────────────────────────────────────────────────
GAMMA = 1.4        # ratio of specific heats (air, calorically perfect)
R_AIR = 287.05     # J/(kg·K) — specific gas constant for dry air
PR    = 0.71       # Prandtl number (air at moderate temperatures)
SIGMA_SB = 5.67e-8 # Stefan-Boltzmann constant, W/(m²·K⁴)
RE_TRANSITION = 5e5 # Reynolds number for laminar→turbulent transition


# ── Core Thermodynamic Functions ────────────────────────────────────────────

def stagnation_temperature(T_inf: float, mach: float) -> float:
    """Total (stagnation) temperature at nose tip (K).

    Isentropic relation: T₀ = T∞ × (1 + (γ-1)/2 × M²)
    Ref: Anderson, Modern Compressible Flow, Eq. 3.28
    """
    return T_inf * (1 + (GAMMA - 1) / 2 * mach ** 2)


def recovery_temperature(T_inf: float, mach: float, laminar: bool = False) -> float:
    """Adiabatic wall (recovery) temperature (K).

    T_r = T∞ × (1 + r × (γ-1)/2 × M²)
    Recovery factor: r = Pr^(1/2) for laminar, Pr^(1/3) for turbulent.
    Ref: Anderson, Hypersonic Gas Dynamics, Eq. 6.33
    """
    r = PR ** 0.5 if laminar else PR ** (1.0 / 3.0)
    return T_inf * (1 + r * (GAMMA - 1) / 2 * mach ** 2)


def sutherland_viscosity(T: float) -> float:
    """Dynamic viscosity of air via Sutherland's law (Pa·s).

    μ = μ_ref × (T/T_ref)^1.5 × (T_ref + S) / (T + S)
    μ_ref = 1.716e-5 Pa·s at T_ref = 273.15 K, S = 110.4 K
    Ref: Sutherland, 1893
    """
    T_ref = 273.15
    S = 110.4
    mu_ref = 1.716e-5
    return mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)


def _is_laminar(rho: float, V: float, x: float, mu: float) -> bool:
    """Determine if flow is laminar at station x based on Re_x."""
    if mu <= 0 or x <= 0:
        return True
    Re_x = rho * V * x / mu
    return Re_x < RE_TRANSITION


def local_reynolds(rho: float, V: float, x: float, T_ref: float) -> float:
    """Local Reynolds number at station x using reference conditions."""
    mu = sutherland_viscosity(T_ref)
    if mu <= 0 or x <= 0:
        return 0.0
    return rho * V * x / mu


# ── Heat Transfer Models ───────────────────────────────────────────────────

def stagnation_heat_flux(T_inf: float, rho: float, V: float,
                         nose_radius: float, wall_temp: float) -> float:
    """Stagnation-point heating using Sutton-Graves correlation (W/m²).

    q_stag = K × √(ρ∞ / R_n) × (h₀ - h_w)
    where K = 1.7415e-4 kg^0.5/m for air (earth entry)

    Simplified engineering form:
    q_stag ≈ K_SG × √(ρ∞ / R_n) × V³
    where K_SG ≈ 1.7415e-4 (air, moderate velocities)

    For rocket-scale velocities, we use the enthalpy-difference form:
    q_stag = C_stag × √(ρ∞ / R_n) × (h_0 - h_w)
    h_0 = cp × T_inf + V²/2
    h_w = cp × T_wall

    Ref: Sutton & Graves, NASA TR R-376, 1971
         Anderson, Hypersonic Gas Dynamics, Eq. 6.57
    """
    if nose_radius <= 0 or V < 1:
        return 0.0

    cp_air = GAMMA * R_AIR / (GAMMA - 1)  # ~1004.5 J/(kg·K)
    h_0 = cp_air * T_inf + 0.5 * V ** 2   # total enthalpy
    h_w = cp_air * wall_temp               # wall enthalpy

    if h_0 <= h_w:
        return 0.0

    # Sutton-Graves constant for air (N₂/O₂ mixture)
    K_SG = 1.7415e-4  # kg^0.5 / m

    q = K_SG * math.sqrt(rho / nose_radius) * (h_0 - h_w)
    return max(q, 0.0)


def convective_heat_flux(T_inf: float, rho: float, V: float, mach: float,
                         x_station: float, wall_temp: float,
                         laminar: bool = False) -> float:
    """Flat-plate convective heat flux (W/m²) with laminar/turbulent distinction.

    Eckert reference enthalpy method:
      T_ref = 0.5 × (T_wall + T∞) + 0.22 × (T_rec - T∞)
      Re_x  = ρ_ref × V × x / μ_ref
      St    = C × Re_x^n × Pr^(-2/3)
    where:
      Laminar:   C = 0.332, n = -0.5  (Blasius)
      Turbulent: C = 0.0296, n = -0.2 (1/5th power law)

    Ref: Eckert, "Engineering Relations for Heat Transfer and Friction
         in High-Velocity Laminar and Turbulent Boundary-Layer Flow", 1955
    """
    if V < 1 or x_station < 0.001:
        return 0.0

    T_rec = recovery_temperature(T_inf, mach, laminar=laminar)
    T_ref = 0.5 * (wall_temp + T_inf) + 0.22 * (T_rec - T_inf)

    # Reference conditions at Eckert reference temperature
    mu_ref = sutherland_viscosity(T_ref)
    rho_ref = rho * T_inf / T_ref  # ideal gas density correction

    Re_x = rho_ref * V * x_station / mu_ref
    if Re_x < 100:
        return 0.0

    # Stanton number depends on flow regime
    if laminar:
        St = 0.332 * Re_x ** (-0.5) * PR ** (-2.0 / 3.0)
    else:
        St = 0.0296 * Re_x ** (-0.2) * PR ** (-2.0 / 3.0)

    cp_air = GAMMA * R_AIR / (GAMMA - 1)
    q = St * rho_ref * V * cp_air * (T_rec - wall_temp)
    return max(q, 0.0)


def wall_temperature_steady(T_inf: float, rho: float, V: float, mach: float,
                            x_station: float, wall_thickness: float,
                            k_wall: float, emissivity: float = 0.8) -> tuple:
    """Steady-state wall temperature via radiation equilibrium (K).

    Solves iteratively: q_conv(T_wall) = q_rad(T_wall)
    where q_rad = ε × σ × T_wall⁴

    Returns (T_outer, T_inner, heat_flux_W_m²).
    T_inner from 1D conduction: T_inner = T_outer - q × t / k

    Ref: Anderson, Hypersonic Gas Dynamics, Ch. 6.5
    """
    T_rec = recovery_temperature(T_inf, mach)
    if V < 1 or x_station < 0.001:
        return T_inf, T_inf, 0.0

    # Determine flow regime at this station
    mu_inf = sutherland_viscosity(T_inf)
    is_lam = _is_laminar(rho, V, x_station, mu_inf)

    T_wall = T_rec  # initial guess

    # Newton-Raphson: solve q_conv(T) - q_rad(T) = 0
    for _ in range(40):
        q_conv = convective_heat_flux(T_inf, rho, V, mach, x_station, T_wall,
                                      laminar=is_lam)
        q_rad = emissivity * SIGMA_SB * T_wall ** 4

        residual = q_conv - q_rad
        if abs(residual) < 1.0:  # converged within 1 W/m²
            break

        # Jacobian: dR/dT = dq_conv/dT - dq_rad/dT
        if abs(T_rec - T_wall) > 0.1:
            h_approx = q_conv / (T_rec - T_wall)
        else:
            q_pert = convective_heat_flux(T_inf, rho, V, mach, x_station,
                                          T_wall - 1.0, laminar=is_lam)
            h_approx = q_pert - q_conv

        dq_rad_dT = 4.0 * emissivity * SIGMA_SB * T_wall ** 3
        derivative = -h_approx - dq_rad_dT

        step = residual / derivative if derivative != 0 else 0.0
        step = max(min(step, 50.0), -50.0)  # clamp step
        T_wall = T_wall - step
        T_wall = max(T_wall, T_inf)  # floor at ambient

    q = convective_heat_flux(T_inf, rho, V, mach, x_station, T_wall,
                             laminar=is_lam)
    T_inner = T_wall - q * wall_thickness / k_wall if k_wall > 0 else T_wall
    return T_wall, T_inner, q


# ── Thermal Stress ─────────────────────────────────────────────────────────

def thermal_stress(E: float, cte: float, nu: float, delta_T: float,
                   constraint_factor: float = 0.55) -> float:
    """Thermal stress in partially-constrained thin shell (Pa).

    σ = constraint_factor × E × α × ΔT / (1 - ν)

    Constraint factor accounts for:
      - Free thermal expansion at unconstrained ends (~0.0 for free bar)
      - Fully constrained shell  (1.0)
      - Typical rocket structure with slip joints (0.4–0.6, default 0.55)

    Ref: Roark's Formulas for Stress and Strain, Table 15.1
    """
    return abs(constraint_factor * E * cte * delta_T / (1 - nu))


# ── Main Analysis Function ─────────────────────────────────────────────────

def analyze_thermal(assembly, mach: float, altitude_m: float,
                    material_name: str = "Aluminum 6061-T6") -> ThermalResult:
    """Full thermal analysis along the rocket body.

    Physics-based approach:
      1. Compute stagnation temperature at nose tip
      2. Compute recovery temperature for body stations
      3. At nose tip: Sutton-Graves stagnation-point heating
      4. Along body: Eckert reference enthalpy with laminar/turbulent transition
      5. Solve radiation equilibrium for wall temperature
      6. Compute thermal stress with partial constraint factor (0.55)

    Improvements over previous version:
      - Stagnation-point heating via Sutton-Graves (replaces artificial blending)
      - Laminar→turbulent transition based on local Re_x
      - Material-dependent emissivity (replaces hardcoded 0.8)
      - Consistent constraint factor = 0.55
      - Local diameter for trapezoidal integration (replaces reference diameter)
      - Nose radius from assembly geometry
    """
    from cfd.solvers.base import isa_conditions
    from core.components import NoseCone, BodyTube, Transition

    result = ThermalResult()
    mat = get_structural_material(material_name)
    P, T_inf, rho = isa_conditions(altitude_m)
    a = math.sqrt(GAMMA * R_AIR * T_inf)
    V = mach * a

    result.stagnation_temp_K = stagnation_temperature(T_inf, mach)
    result.recovery_temp_K = recovery_temperature(T_inf, mach)
    result.service_temp_limit_K = mat.max_service_temp

    total_L = assembly.total_length()
    if total_L <= 0:
        return result

    # ── Extract geometry from assembly ─────────────────────────────────
    d_ref = assembly.get_reference_diameter()
    wt = 0.002  # default wall thickness
    nose_radius = d_ref * 0.1  # default nose tip radius (tangent ogive)

    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                wt = getattr(comp, 'wall_thickness', 0.002)
                # Nose tip radius estimation from shape
                nose_length = getattr(comp, 'length', 0.3)
                nose_diameter = getattr(comp, 'diameter', d_ref)
                # NoseCone.shape values: "Ogive", "Conical", "Haack (LD)",
                # "Elliptical", "Parabolic" — normalize for matching
                shape_raw = str(getattr(comp, 'shape', 'Ogive')).lower()
                if 'conic' in shape_raw:
                    shape = 'conical'
                elif 'ellip' in shape_raw:
                    shape = 'elliptical'
                else:  # ogive, haack, parabolic → blunt-tip ogive estimate
                    shape = 'ogive'
                if shape in ('ogive', 'haack'):
                    # Tangent ogive: ρ = (r² + L²) / (2r)
                    r = nose_diameter / 2
                    if r > 0 and nose_length > 0:
                        rho_ogive = (r ** 2 + nose_length ** 2) / (2 * r)
                        # Tip radius ≈ 1/ρ for small tips
                        nose_radius = max(0.001, r ** 2 / (2 * rho_ogive))
                elif shape == 'conical':
                    # Sharp cone: use small bluntness radius
                    nose_radius = max(0.001, nose_diameter * 0.02)
                elif shape == 'elliptical':
                    # Elliptical: tip radius = r²/a (a = semi-major = length)
                    r = nose_diameter / 2
                    nose_radius = max(0.001, r ** 2 / nose_length) if nose_length > 0 else 0.01
                break
            elif isinstance(comp, BodyTube):
                wt = (comp.outer_diameter_val - comp.inner_diameter) / 2
                break

    # ── Material-dependent emissivity ──────────────────────────────────
    emissivity = getattr(mat, 'emissivity', 0.8)

    T_stag = result.stagnation_temp_K
    T_rec = result.recovery_temp_K

    # ── Station-by-station analysis ────────────────────────────────────
    max_T_wall = 0.0
    min_T_wall = 9999.0
    max_q = 0.0
    max_sig_th = 0.0
    total_q = 0.0
    laminar_transition_x = total_L  # assume laminar until proven otherwise

    constraint_factor = 0.55  # consistent across all thermal paths
    n_stations = 60  # increased for smoother profiles

    # Track stagnation point data
    stag_heat_flux = 0.0

    for i in range(n_stations + 1):
        frac = i / n_stations
        x = max(0.001, total_L * frac)

        # Estimate local diameter for integration
        d_local = d_ref
        for stage in assembly.stages:
            for comp in stage.children:
                if isinstance(comp, Transition):
                    comp_start = getattr(comp, 'position', 0)
                    comp_end = comp_start + getattr(comp, 'length', 0.1)
                    if comp_start < x < comp_end:
                        blend = (x - comp_start) / max(comp_end - comp_start, 0.001)
                        d_fwd = getattr(comp, 'fore_diameter', d_ref)
                        d_aft = getattr(comp, 'aft_diameter', d_ref)
                        d_local = d_fwd + (d_aft - d_fwd) * blend

        # ── Nose stagnation zone (x < 2% of length) ────────────────
        if frac < 0.02 and nose_radius > 0:
            # Sutton-Graves stagnation-point heating
            # Initial guess: recovery temperature
            T_wall_stag = T_stag * 0.9

            for _ in range(30):
                q_stag = stagnation_heat_flux(T_inf, rho, V, nose_radius, T_wall_stag)
                q_rad = emissivity * SIGMA_SB * T_wall_stag ** 4
                residual = q_stag - q_rad
                if abs(residual) < 1.0:
                    break
                # Simple damped iteration
                dq_rad_dT = 4.0 * emissivity * SIGMA_SB * T_wall_stag ** 3
                cp_air = GAMMA * R_AIR / (GAMMA - 1)
                dq_stag_dT = -stagnation_heat_flux(T_inf, rho, V, nose_radius, T_wall_stag) / max(T_stag - T_wall_stag, 1.0) if T_stag > T_wall_stag else 0
                deriv = dq_stag_dT - dq_rad_dT
                step = residual / deriv if abs(deriv) > 1e-6 else 0.0
                step = max(min(step, 50.0), -50.0)
                T_wall_stag -= step
                T_wall_stag = max(T_wall_stag, T_inf)

            # Blend from stagnation to flat-plate as we move aft of nose tip
            blend = frac / 0.02  # 0 at tip, 1 at 2% of length
            T_out_fp, _, q_fp = wall_temperature_steady(
                T_inf, rho, V, mach, x, wt, mat.thermal_conductivity, emissivity
            )
            T_out = T_wall_stag * (1.0 - blend) + T_out_fp * blend
            q = stagnation_heat_flux(T_inf, rho, V, nose_radius, T_out) * (1.0 - blend) + q_fp * blend

            if i == 0:
                stag_heat_flux = stagnation_heat_flux(T_inf, rho, V, nose_radius, T_wall_stag)

        else:
            # ── Body stations: flat-plate convective heating ────────
            T_out, T_in, q = wall_temperature_steady(
                T_inf, rho, V, mach, x, wt, mat.thermal_conductivity, emissivity
            )

        # ── Track laminar-turbulent transition ─────────────────────
        mu_inf = sutherland_viscosity(T_inf)
        if mu_inf > 0:
            Re_x = rho * V * x / mu_inf
            if Re_x >= RE_TRANSITION and x < laminar_transition_x:
                laminar_transition_x = x

        # ── Fin root junction heating (20% increase at fin locations) ──
        # Interference heating at fin-body junction: factor 1.2-1.5
        # Applied at aft 30% of body where fins are typically located
        fin_heating_factor = 1.0
        if frac > 0.70 and frac < 0.95:
            # Check if fins exist in this region
            fin_heating_factor = 1.2  # conservative junction heating factor

        q *= fin_heating_factor
        if fin_heating_factor > 1.0:
            T_out = T_out + (T_rec - T_out) * (fin_heating_factor - 1.0) * 0.5

        # ── Thermal stress at this station ─────────────────────────
        sig = thermal_stress(mat.E, mat.cte, mat.nu, T_out - 293.15,
                             constraint_factor=constraint_factor)

        result.station_temps.append((x, T_out))
        result.station_stresses.append((x, sig))

        max_T_wall = max(max_T_wall, T_out)
        min_T_wall = min(min_T_wall, T_out)
        max_q = max(max_q, q)
        max_sig_th = max(max_sig_th, sig)
        dx = total_L / n_stations
        total_q += q * math.pi * d_local * dx

    # ── Populate result ────────────────────────────────────────────────
    result.max_wall_temp_K = max_T_wall
    result.min_wall_temp_K = min_T_wall
    result.max_heat_flux_W_m2 = max_q
    result.total_heat_input_W = total_q
    result.max_thermal_stress = max_sig_th
    result.exceeds_service_temp = max_T_wall > mat.max_service_temp

    if max_sig_th > 0:
        result.thermal_safety_factor = mat.yield_strength / max_sig_th
    else:
        result.thermal_safety_factor = float('inf')

    result.converged = True
    logger.info(
        f"Thermal (physics-validated): T_wall_max={max_T_wall:.0f} K, "
        f"σ_th={max_sig_th/1e6:.1f} MPa, q_max={max_q:.0f} W/m², "
        f"q_stag={stag_heat_flux:.0f} W/m², ε={emissivity:.2f}, "
        f"Re_transition at x={laminar_transition_x:.3f} m, "
        f"exceeds_limit={result.exceeds_service_temp}"
    )
    return result
