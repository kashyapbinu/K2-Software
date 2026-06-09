"""
K2 Aerospace — Structural Analysis Module (Analytical)
=======================================================
Quick analytical stress calculations for real-time display.
Full FEM analysis is in structures/solvers/.
"""
import math, logging
logger = logging.getLogger("K2.Structures")

MATERIALS = {
    "Aluminum 6061-T6": {"yield": 276e6, "ultimate": 310e6, "E": 68.9e9, "density": 2700, "poisson": 0.33, "cte": 23.6e-6, "G": 26e9},
    "Carbon Fiber Composite": {"yield": 600e6, "ultimate": 700e6, "E": 70e9, "density": 1600, "poisson": 0.30, "cte": 2.0e-6, "G": 26.9e9},
    "Fiberglass (G10)": {"yield": 310e6, "ultimate": 380e6, "E": 18.6e9, "density": 1800, "poisson": 0.28, "cte": 14.0e-6, "G": 7.3e9},
    "Steel 4130": {"yield": 460e6, "ultimate": 560e6, "E": 200e9, "density": 7850, "poisson": 0.29, "cte": 11.2e-6, "G": 77.5e9},
    "Titanium Ti-6Al-4V": {"yield": 880e6, "ultimate": 950e6, "E": 114e9, "density": 4430, "poisson": 0.34, "cte": 8.6e-6, "G": 42.5e9},
}

def get_material(name):
    return MATERIALS.get(name, MATERIALS["Aluminum 6061-T6"])

def axial_stress(force, diameter, wall_thickness):
    area = math.pi * diameter * wall_thickness
    return abs(force) / area if area > 0 else 0.0

def hoop_stress(internal_pressure, diameter, wall_thickness):
    return internal_pressure * (diameter / 2) / wall_thickness if wall_thickness > 0 else 0.0

def longitudinal_stress(internal_pressure, diameter, wall_thickness):
    """Longitudinal stress in thin-walled pressure vessel (half of hoop)."""
    return internal_pressure * (diameter / 2) / (2 * wall_thickness) if wall_thickness > 0 else 0.0

def shear_stress(torque, diameter, wall_thickness):
    """Torsional shear stress in thin-walled tube."""
    r = diameter / 2
    J = 2 * math.pi * r**3 * wall_thickness  # thin-wall approx
    return abs(torque) * r / J if J > 0 else 0.0

def bending_stress(moment, diameter, wall_thickness):
    """Maximum bending stress in thin-walled tube."""
    r_o = diameter / 2
    r_i = r_o - wall_thickness
    I = (math.pi / 4) * (r_o**4 - r_i**4)
    return abs(moment) * r_o / I if I > 0 else 0.0

def von_mises(sigma_x, sigma_y=0, sigma_z=0, tau_xy=0, tau_xz=0, tau_yz=0):
    """Von Mises equivalent stress from general 3D stress state."""
    return math.sqrt(0.5 * ((sigma_x - sigma_y)**2 + (sigma_y - sigma_z)**2 +
                            (sigma_z - sigma_x)**2 + 6*(tau_xy**2 + tau_xz**2 + tau_yz**2)))

def von_mises_2d(sigma_axial, sigma_hoop, tau=0):
    """Simplified von Mises for thin-walled tube (plane stress)."""
    return math.sqrt(sigma_axial**2 - sigma_axial * sigma_hoop + sigma_hoop**2 + 3 * tau**2)

def thermal_stress_shell(E, cte, nu, delta_T, constraint_factor=1.0):
    """Thermal stress in partially-constrained thin shell (Pa).

    σ = constraint_factor × E × α × ΔT / (1 - ν)

    Constraint factor values:
      0.0  — fully free (no thermal stress)
      0.55 — typical rocket structure with slip joints (default for condition analysis)
      1.0  — fully constrained (conservative, default for generic analysis)

    Ref: Roark's Formulas for Stress and Strain, Table 15.1
    """
    return abs(constraint_factor * E * cte * delta_T / (1 - nu))

def euler_buckling(E, diameter, wall_thickness, length):
    r_o = diameter / 2
    r_i = r_o - wall_thickness
    I = (math.pi / 4) * (r_o**4 - r_i**4)
    if length <= 0:
        return float('inf')
    return (math.pi**2 * E * I) / (length**2)

def shell_buckling(E, diameter, wall_thickness, nu=0.3):
    """Critical axial stress for thin cylindrical shell buckling (NASA SP-8007)."""
    r = diameter / 2
    t = wall_thickness
    if r <= 0 or t <= 0:
        return float('inf')
    # Classical formula with knockdown factor
    sigma_cl = E * t / (r * math.sqrt(3 * (1 - nu**2)))
    # NASA knockdown factor for rocket structures (conservative)
    gamma = 1 - 0.901 * (1 - math.exp(-1 / 16 * math.sqrt(r / t)))
    return gamma * sigma_cl

def safety_factor(yield_strength, max_stress):
    return yield_strength / max_stress if max_stress > 0 else float('inf')

def margin_of_safety(yield_strength, max_stress, sf_required=1.0):
    """Margin of Safety: MoS = (σ_yield / (SF_req × σ_max)) - 1 = SF - 1 when
    SF_req = 1. Positive = safe. With the default SF_req = 1 the margin is
    never negative while the safety factor exceeds 1 (per validation spec)."""
    if max_stress <= 0:
        return float('inf')
    return (yield_strength / (sf_required * max_stress)) - 1.0

def compute_all(force, diameter, wall_thickness, length, material_name,
                internal_pressure=0.0, torque=0.0, bending_moment=0.0, delta_T=0.0,
                shear_force=0.0, thermal_constraint=1.0):
    """Comprehensive analytical stress analysis.

    ``thermal_constraint`` is the thin-shell thermal constraint factor: 1.0 =
    fully constrained (conservative generic default), ~0.55 for a real airframe
    with slip joints / free ends (matches the condition-specific thermal path).
    """
    mat = get_material(material_name)
    area = math.pi * diameter * wall_thickness
    ax = axial_stress(force, diameter, wall_thickness)
    hp = hoop_stress(internal_pressure, diameter, wall_thickness)
    lg = longitudinal_stress(internal_pressure, diameter, wall_thickness)
    sh_t = shear_stress(torque, diameter, wall_thickness)            # torsional
    tau_v = 2.0 * abs(shear_force) / area if area > 0 else 0.0       # transverse, τ_max≈2V/A
    sh = math.sqrt(sh_t ** 2 + tau_v ** 2)
    bd = bending_stress(bending_moment, diameter, wall_thickness)
    th = thermal_stress_shell(mat["E"], mat.get("cte", 23.6e-6), mat["poisson"], delta_T,
                              constraint_factor=thermal_constraint)
    # Combined axial = direct axial + longitudinal pressure + bending + thermal
    sigma_x = ax + lg + bd + th
    sigma_y = hp
    tau = sh
    vm = von_mises_2d(sigma_x, sigma_y, tau)
    bk = euler_buckling(mat["E"], diameter, wall_thickness, length)
    sbk = shell_buckling(mat["E"], diameter, wall_thickness, mat["poisson"])
    sf = safety_factor(mat["yield"], vm)
    mos = margin_of_safety(mat["yield"], vm)
    return {
        "axial": ax, "hoop": hp, "longitudinal": lg, "shear": sh,
        "bending": bd, "thermal": th, "von_mises": vm,
        "buckling": bk, "shell_buckling_stress": sbk,
        "max_stress": vm, "safety_factor": sf, "margin_of_safety": mos,
        "material": mat, "yield_utilization": vm / mat["yield"] if mat["yield"] > 0 else 0,
    }


# ── Condition-Specific Analysis ─────────────────────────────────────────────

def compute_for_condition(condition, diameter, wall_thickness, length, material_name,
                          force=0.0, internal_pressure=0.0, delta_T=0.0,
                          mach=0.0, altitude_m=0.0, vehicle_mass_kg=5.0,
                          angle_of_attack_deg=2.0, moment_arm_m=0.0):
    """Compute stresses for a specific flight condition.

    Returns a dict with physically distinct stress values per condition:
    - "max_thrust": compressive axial + hoop + bending from AoA
    - "recovery": tensile shock + DAF + stress concentration
    - "thermal": thermal expansion + gradient-driven stress

    ``moment_arm_m`` is the aerodynamic bending moment arm (|x_CP - x_CG|).
    When > 0 it is used directly; otherwise a length-fraction fallback applies.
    """
    if condition == "Max Thrust" or condition == "max_thrust":
        return _compute_max_thrust(force, diameter, wall_thickness, length,
                                    material_name, internal_pressure,
                                    mach, altitude_m, angle_of_attack_deg,
                                    moment_arm_m)
    elif condition == "Recovery Shock" or condition == "recovery":
        return _compute_recovery(diameter, wall_thickness, length,
                                  material_name, vehicle_mass_kg)
    elif condition == "Thermal" or condition == "thermal":
        return _compute_thermal(diameter, wall_thickness, length,
                                 material_name, mach, altitude_m)
    elif condition == "Max-Q":
        return _compute_max_q(force, diameter, wall_thickness, length,
                              material_name, mach, altitude_m,
                              angle_of_attack_deg, moment_arm_m)
    else:
        # Custom / fallback — use generic
        return compute_all(force, diameter, wall_thickness, length, material_name,
                           internal_pressure=internal_pressure, delta_T=delta_T)


def _compute_max_thrust(force, diameter, wall_thickness, length, material_name,
                         internal_pressure=0.0, mach=0.0, altitude_m=0.0,
                         angle_of_attack_deg=2.0, moment_arm_m=0.0):
    """Max thrust: compressive axial + hoop + AoA body bending + shear.

    Clean thin-walled-tube mechanics (Roark Ch. 9). Fin loads are analysed
    separately in structures.workstation.fin_analysis — they are NOT folded
    into the body von Mises here. A single documented stress-concentration
    factor Kt accounts for couplers / fin slots / rail-button cutouts.

        σ_axial = F / A,  A = π·d·t
        σ_hoop  = p·r / t
        σ_bend  = M·r_o / I,  I = (π/4)(r_o⁴ − r_i⁴)
        τ       ≈ 2V / A      (max shear, thin circular tube)
        σ_vm    = Kt·√(σ_x² − σ_x·σ_y + σ_y² + 3τ²)
    """
    mat = get_material(material_name)
    r = diameter / 2
    r_o = r + wall_thickness / 2
    r_i = r - wall_thickness / 2
    area = math.pi * diameter * wall_thickness
    I = (math.pi / 4) * (r_o**4 - r_i**4)

    # Single stress-concentration factor for real joints/cutouts (not 1.8 stacked)
    Kt = 1.5

    # 1. Compressive axial stress from thrust
    ax = abs(force) / area if area > 0 else 0.0
    # 2. Hoop + longitudinal from internal (motor chamber) pressure
    hp = internal_pressure * r / wall_thickness if wall_thickness > 0 else 0.0
    lg = hp / 2.0
    # 3. Aerodynamic body bending from angle of attack
    #    Pure axial thrust produces NO transverse shear on the cross-section,
    #    so shear starts at zero and is driven only by the aero side force.
    sigma_bend = 0.0
    tau = 0.0
    if mach > 0 and angle_of_attack_deg > 0 and I > 0:
        try:
            from cfd.solvers.base import isa_conditions
            P, T, rho = isa_conditions(altitude_m)
        except Exception:
            T, rho = 288.15, 1.225
        a = math.sqrt(1.4 * 287.05 * T)
        V = mach * a
        q = 0.5 * rho * V ** 2
        aoa_rad = math.radians(angle_of_attack_deg)
        CN_alpha = 2.0                       # slender-body normal-force slope (/rad)
        A_ref = math.pi * r ** 2
        F_N = q * CN_alpha * aoa_rad * A_ref  # aero side force
        # Bending moment arm = CP-to-CG distance when known, else ¼ length.
        arm = moment_arm_m if moment_arm_m > 0 else length * 0.25
        M_bend = F_N * arm
        sigma_bend = M_bend * r_o / I
        tau = 2.0 * F_N / area if area > 0 else 0.0  # τ_max ≈ 2V/A, thin tube

    # Combined stress state (thermal negligible during powered flight)
    sx = ax + lg + sigma_bend
    sy = hp
    vm = Kt * math.sqrt(sx**2 - sx * sy + sy**2 + 3 * tau**2)

    bk = euler_buckling(mat["E"], diameter, wall_thickness, length)
    sbk = shell_buckling(mat["E"], diameter, wall_thickness, mat["poisson"])
    sf = safety_factor(mat["yield"], vm)
    mos = margin_of_safety(mat["yield"], vm)

    return {
        "axial": ax, "hoop": hp, "longitudinal": lg,
        "shear": tau, "bending": sigma_bend, "thermal": 0.0,
        "von_mises": vm, "buckling": bk, "shell_buckling_stress": sbk,
        "max_stress": vm, "safety_factor": sf, "margin_of_safety": mos,
        "material": mat, "yield_utilization": vm / mat["yield"] if mat["yield"] > 0 else 0,
    }


def _compute_max_q(force, diameter, wall_thickness, length, material_name,
                    mach=0.8, altitude_m=3000.0, angle_of_attack_deg=3.0,
                    moment_arm_m=0.0):
    """Max-Q: aerodynamic bending dominated load case.

    Includes:
    1. Bending moment from AoA side force (DOMINANT)
    2. Fin root bending (fins see highest loads at max-Q)
    3. Dynamic amplification factor (gust response)
    4. Structural detail factor for joints/cutouts
    5. Axial compression from drag + thrust
    """
    mat = get_material(material_name)
    r = diameter / 2
    r_o = r + wall_thickness / 2
    r_i = r - wall_thickness / 2
    area = math.pi * diameter * wall_thickness
    I = (math.pi / 4) * (r_o**4 - r_i**4)

    Kt = 1.5         # joints / cutouts stress concentration
    DAF = 1.2        # gust dynamic amplification at max-Q

    # Atmospheric conditions at max-Q
    try:
        from cfd.solvers.base import isa_conditions
        P, T, rho = isa_conditions(altitude_m)
    except Exception:
        T, rho = 288.15, 1.225
    a = math.sqrt(1.4 * 287.05 * T)
    V = mach * a
    q_dyn = 0.5 * rho * V ** 2

    # 1. Axial: thrust + drag
    Cd = 0.5
    A_ref = math.pi * r ** 2
    F_drag = q_dyn * Cd * A_ref
    ax = (abs(force) + F_drag) / area if area > 0 else 0.0

    # 2. Body bending from AoA normal force (slender-body crossflow)
    aoa_rad = math.radians(angle_of_attack_deg)
    CN_alpha = 2.0
    F_N = q_dyn * CN_alpha * aoa_rad * A_ref
    # Bending moment arm = CP-to-CG distance when known, else ~0.3 L fallback.
    arm = moment_arm_m if moment_arm_m > 0 else length * 0.30
    M_bend = F_N * arm
    sigma_bend = M_bend * r_o / I if I > 0 else 0.0

    # 3. Hoop: only internal pressure (external aero pressure ≪ shell hoop)
    hp = 0.0

    # 4. Transverse shear (thin tube, τ_max ≈ 2V/A)
    tau = 2.0 * F_N / area if area > 0 else 0.0

    sx = ax + sigma_bend
    vm = Kt * DAF * math.sqrt(sx**2 - sx * hp + hp**2 + 3 * tau**2)

    bk = euler_buckling(mat["E"], diameter, wall_thickness, length)
    sbk = shell_buckling(mat["E"], diameter, wall_thickness, mat["poisson"])
    sf = safety_factor(mat["yield"], vm)
    mos = margin_of_safety(mat["yield"], vm)

    return {
        "axial": ax, "hoop": hp, "longitudinal": 0.0,
        "shear": tau, "bending": sigma_bend, "thermal": 0.0,
        "von_mises": vm, "buckling": bk, "shell_buckling_stress": sbk,
        "max_stress": vm, "safety_factor": sf, "margin_of_safety": mos,
        "material": mat, "yield_utilization": vm / mat["yield"] if mat["yield"] > 0 else 0,
    }


def _compute_recovery(diameter, wall_thickness, length, material_name,
                       vehicle_mass_kg=5.0, shock_g=15.0, daf=1.8, kt=2.5):
    """Recovery shock: tensile axial from parachute snap + stress concentrations."""
    mat = get_material(material_name)
    r = diameter / 2
    r_o = r + wall_thickness / 2
    r_i = r - wall_thickness / 2
    area = math.pi * diameter * wall_thickness
    I = (math.pi / 4) * (r_o**4 - r_i**4)

    # 1. Recovery shock force: F = m * a_shock * DAF
    F_recovery = vehicle_mass_kg * shock_g * 9.81 * daf

    # 2. TENSILE axial stress (not compressive!)
    ax = F_recovery / area if area > 0 else 0.0

    # 3. Localized stress at attachment point (stress concentration factor)
    ax_peak = ax * kt

    # 4. No hoop stress (no internal pressure during recovery)
    hp = 0.0

    # 5. Bending from transient snap-back oscillation
    # Recovery force applied off-axis → bending moment
    eccentricity = 0.01 * diameter  # 1% of diameter offset
    M_snapback = F_recovery * eccentricity
    sigma_bend = M_snapback * r_o / I if I > 0 else 0.0

    # 6. Shear from transient deceleration
    tau = F_recovery / (2 * math.pi * r * wall_thickness) * 0.3 if (r > 0 and wall_thickness > 0) else 0.0

    # 7. No thermal stress during recovery
    th = 0.0

    # Von Mises using peak (concentrated) axial stress
    sx = ax_peak + sigma_bend
    vm = math.sqrt(sx**2 + 3 * tau**2)

    bk = euler_buckling(mat["E"], diameter, wall_thickness, length)
    sbk = shell_buckling(mat["E"], diameter, wall_thickness, mat["poisson"])
    sf = safety_factor(mat["yield"], vm)
    mos = margin_of_safety(mat["yield"], vm)

    return {
        "axial": ax_peak, "hoop": hp, "longitudinal": 0.0,
        "shear": tau, "bending": sigma_bend, "thermal": th,
        "von_mises": vm, "buckling": bk, "shell_buckling_stress": sbk,
        "max_stress": vm, "safety_factor": sf, "margin_of_safety": mos,
        "material": mat, "yield_utilization": vm / mat["yield"] if mat["yield"] > 0 else 0,
    }


def _compute_thermal(diameter, wall_thickness, length, material_name,
                      mach=3.0, altitude_m=10000.0):
    """Thermal: aerodynamic heating with non-uniform temperature distribution.

    Uses partial constraint factor (0.55) to account for:
    - Free thermal expansion at unconstrained ends
    - Gradual heating (not instantaneous)
    - Stress relaxation in ductile materials
    """
    mat = get_material(material_name)
    r = diameter / 2

    # Get atmospheric conditions
    T_amb = 223.15  # default ~-50C at 10 km
    try:
        from cfd.solvers.base import isa_conditions
        P, T_amb, rho = isa_conditions(altitude_m)
    except Exception:
        pass

    gamma = 1.4
    r_recovery = 0.89  # turbulent recovery factor Pr^(1/3)

    # Recovery temperature (adiabatic wall temperature)
    T_recovery = T_amb * (1 + r_recovery * (gamma - 1) / 2 * mach**2)
    # Stagnation temperature (nose tip)
    T_stag = T_amb * (1 + (gamma - 1) / 2 * mach**2)

    # Non-uniform delta-T distribution (ref = 293.15 K stress-free assembly temp)
    dT_nose = T_stag - 293.15
    dT_body = T_recovery - 293.15
    # Aero skin heating governs the airframe; the motor casing is a separate
    # load path (internal combustion) and is NOT folded into the aero-skin
    # thermal stress here. The stagnation nose is the hottest aero station.
    dT_motor = dT_body  # retained for the display dict only; not a heat source

    dT_max = max(dT_nose, dT_body)

    # Partial constraint: real rocket structures are NOT fully constrained
    # Free ends, slip joints, gradual heating reduce effective stress
    constraint_factor = 0.55

    th_full = thermal_stress_shell(mat["E"], mat.get("cte", 23.6e-6), mat["poisson"], dT_max)
    th = th_full * constraint_factor

    # Thermal gradient-induced bending
    dT_gradient = abs(dT_nose - dT_body)
    th_bending = mat["E"] * mat.get("cte", 23.6e-6) * dT_gradient * wall_thickness / (2 * diameter) * constraint_factor if diameter > 0 else 0.0

    # No mechanical loads during purely thermal analysis
    ax = 0.0
    hp = 0.0
    tau = 0.0

    # Von Mises: thermal + thermal bending
    sx = th + th_bending
    vm = math.sqrt(sx**2)  # uniaxial thermal

    bk = euler_buckling(mat["E"], diameter, wall_thickness, length)
    sbk = shell_buckling(mat["E"], diameter, wall_thickness, mat["poisson"])
    sf = safety_factor(mat["yield"], vm)
    mos = margin_of_safety(mat["yield"], vm)

    return {
        "axial": ax, "hoop": hp, "longitudinal": 0.0,
        "shear": tau, "bending": th_bending, "thermal": th,
        "von_mises": vm, "buckling": bk, "shell_buckling_stress": sbk,
        "max_stress": vm, "safety_factor": sf, "margin_of_safety": mos,
        "material": mat, "yield_utilization": vm / mat["yield"] if mat["yield"] > 0 else 0,
        # Extra thermal data for display
        "dT_nose": dT_nose, "dT_body": dT_body, "dT_motor": dT_motor,
        "T_stag": T_stag, "T_recovery": T_recovery,
    }

