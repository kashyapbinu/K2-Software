"""
K2 Aerospace — Structures Workstation Backend
==============================================
Pure-Python analytical engine that turns a rocket (geometry + material +
flight history) into the full set of structural answers the Structures
workspace presents:

    • Flight-load import            (FlightLoads.from_history / from_state)
    • Worst-case condition search   (find_worst_case)
    • Recovery system loads         (recovery_loads)
    • Fin structural analysis       (fin_analysis)
    • Buckling (4 modes)            (buckling_analysis)
    • Mass efficiency               (mass_efficiency)
    • Structural safety score       (safety_score)
    • Thermal profile vs altitude   (thermal_profile)
    • Structural failure map        (failure_map)
    • Load-path force flow          (load_path)

No Qt, no FEM solver dependency — everything here is closed-form so it runs
in well under a second for a typical rocket. The Qt workspace consumes these
dataclasses; a standalone test at the bottom of the file exercises them.

References
----------
- NASA SP-8007  (shell buckling)
- NACA TN 4197  (fin flutter, the model-rocket form)
- Roark's Formulas for Stress and Strain, 8th ed.
- Shigley & Mischke, Mechanical Engineering Design
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from physics.structures import (
    compute_for_condition, euler_buckling, shell_buckling, get_material,
)
from structures.solvers.base import get_structural_material, StructuralMaterial

logger = logging.getLogger("K2.Workstation")

G = 9.80665


# ── Atmosphere helper (ISA, with graceful fallback) ──────────────────────────
def isa(alt_m: float) -> tuple[float, float, float]:
    """Return (pressure_Pa, temperature_K, density_kg_m3) at altitude."""
    try:
        from cfd.solvers.base import isa_conditions
        return isa_conditions(max(0.0, alt_m))
    except Exception:
        # Simple troposphere fallback
        T0, P0, L, R, g = 288.15, 101325.0, 0.0065, 287.05, 9.80665
        alt = max(0.0, min(alt_m, 11000.0))
        T = T0 - L * alt
        P = P0 * (T / T0) ** (g / (R * L))
        rho = P / (R * T)
        return P, T, rho


def speed_of_sound(T_K: float) -> float:
    return math.sqrt(1.4 * 287.05 * max(T_K, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# 3.  FLIGHT LOAD IMPORT
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FlightLoads:
    """Peak flight loads imported from a completed simulation (or state)."""
    available: bool = False
    source: str = "none"          # "simulation" | "state" | "manual"
    max_velocity: float = 0.0     # m/s
    max_mach: float = 0.0
    max_acceleration: float = 0.0  # m/s^2
    max_accel_g: float = 0.0
    max_dynamic_pressure: float = 0.0  # Pa
    max_thrust: float = 0.0       # N
    maxq_mach: float = 0.0
    maxq_altitude: float = 0.0    # m
    maxq_time: float = 0.0        # s
    maxmach_altitude: float = 0.0  # m (altitude at peak Mach — peak heating)
    maxthrust_mach: float = 0.0    # Mach at the peak-thrust instant
    maxthrust_altitude: float = 0.0  # m (altitude at peak thrust — early boost)
    wind_speed: float = 0.0       # m/s
    vehicle_mass: float = 5.0     # kg (at max-Q)
    moment_arm: float = 0.0       # m, |x_CP - x_CG| at max-Q (bending arm)

    @classmethod
    def from_history(cls, history, state=None) -> "FlightLoads":
        """Extract peak loads from a HistoryManager. Returns available=False
        when no flight data exists."""
        if history is None or len(history) == 0:
            return cls.from_state(state)

        t_q, q_max, idx_q = history.find_max("dynamic_pressure")
        _, v_max, _ = history.find_max("velocity")
        _, m_max, idx_m = history.find_max("mach")
        _, a_max, _ = history.find_max("acceleration")
        _, thr_max, idx_thr = history.find_max("thrust")

        snap = history.get_snapshot(idx_q) if idx_q is not None else {}
        snap_m = history.get_snapshot(idx_m) if idx_m is not None else {}
        snap_thr = history.get_snapshot(idx_thr) if idx_thr is not None else {}
        wind = getattr(state, "wind_speed", 0.0) if state else 0.0
        mass = snap.get("mass", 0.0) or (state.total_mass() if state else 5.0)
        # Bending moment arm at max-Q = distance between centre of pressure and
        # centre of gravity (recorded per-tick by the sim).
        arm = abs(snap.get("cp", 0.0) - snap.get("cg", 0.0))

        return cls(
            available=True, source="simulation",
            max_velocity=v_max, max_mach=m_max,
            max_acceleration=a_max, max_accel_g=a_max / G,
            max_dynamic_pressure=q_max, max_thrust=thr_max,
            maxq_mach=snap.get("mach", m_max),
            maxq_altitude=snap.get("altitude", 0.0),
            maxmach_altitude=snap_m.get("altitude", snap.get("altitude", 0.0)),
            maxthrust_mach=snap_thr.get("mach", 0.0),
            maxthrust_altitude=snap_thr.get("altitude", 0.0),
            maxq_time=t_q, wind_speed=wind,
            vehicle_mass=mass if mass > 0 else 5.0,
            moment_arm=arm,
        )

    @classmethod
    def from_state(cls, state) -> "FlightLoads":
        """Fallback: use whatever maxima the RocketState already tracks."""
        if state is None:
            return cls(available=False, source="none")
        has = (getattr(state, "max_velocity", 0.0) > 0
               or getattr(state, "max_mach", 0.0) > 0)
        mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0
        return cls(
            available=has, source="state" if has else "none",
            max_velocity=getattr(state, "max_velocity", 0.0),
            max_mach=getattr(state, "max_mach", 0.0),
            max_acceleration=getattr(state, "max_acceleration", 0.0),
            max_accel_g=getattr(state, "max_acceleration", 0.0) / G,
            max_dynamic_pressure=getattr(state, "dynamic_pressure", 0.0),
            max_thrust=getattr(state, "motor_max_thrust", 0.0) or getattr(state, "thrust", 0.0),
            maxq_mach=getattr(state, "max_mach", 0.0),
            maxq_altitude=getattr(state, "altitude", 0.0),
            maxmach_altitude=getattr(state, "altitude", 0.0),
            wind_speed=getattr(state, "wind_speed", 0.0),
            vehicle_mass=mass if mass and mass > 0 else 5.0,
            moment_arm=abs(getattr(state, "cp", 0.0) - getattr(state, "cg", 0.0)),
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4.  WORST-CASE LOAD SEARCH
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FlightEvent:
    name: str
    time: float = 0.0
    von_mises: float = 0.0        # Pa
    safety_factor: float = 0.0
    buckling_margin: float = 0.0  # applied / critical (>1 = buckled)
    load_factor_g: float = 0.0
    mach: float = 0.0
    altitude: float = 0.0
    dynamic_pressure: float = 0.0


@dataclass
class WorstCaseResult:
    events: list = field(default_factory=list)   # list[FlightEvent]
    critical_event: Optional[FlightEvent] = None  # lowest SF
    highest_stress_event: Optional[FlightEvent] = None
    highest_buckling_event: Optional[FlightEvent] = None
    available: bool = False


# Canonical event order requested by the spec
_EVENT_ORDER = [
    "Launch", "Rail Exit", "Max-Q", "Max Mach", "Burnout",
    "Coast", "Apogee", "Parachute Deployment", "Landing Impact",
]


def _event_indices(history) -> dict:
    """Map each named flight event to a history index."""
    n = len(history)
    if n == 0:
        return {}
    idx = {}
    idx["Launch"] = 0
    # Rail exit: first point past ~2 m travel (or 3% into flight)
    xs = history.get_values("x") or history.get_values("altitude")
    rail = 0
    for i, x in enumerate(xs):
        if x > 2.0:
            rail = i
            break
    idx["Rail Exit"] = rail
    _, _, iq = history.find_max("dynamic_pressure")
    idx["Max-Q"] = iq
    _, _, im = history.find_max("mach")
    idx["Max Mach"] = im
    # Burnout: where propellant_mass stops dropping
    prop = history.get_values("propellant_mass")
    burn = 0
    for i in range(1, len(prop)):
        if prop[i] <= 1e-6 and prop[i - 1] > 1e-6:
            burn = i
            break
    if burn == 0 and prop:
        burn = min(range(len(prop)), key=lambda i: prop[i])
    idx["Burnout"] = burn
    _, _, iap = history.find_max("altitude")
    idx["Apogee"] = iap
    idx["Coast"] = (burn + iap) // 2 if iap > burn else burn
    # Parachute deploy ~ just after apogee
    idx["Parachute Deployment"] = min(iap + 1, n - 1)
    idx["Landing Impact"] = n - 1
    return idx


def find_worst_case(state, history, material_name: str,
                    progress: Optional[Callable[[str, float], None]] = None) -> WorstCaseResult:
    """Evaluate structural response at every major flight event and identify
    the governing (lowest safety factor) condition."""
    res = WorstCaseResult()
    if history is None or len(history) == 0:
        return res

    d = state.diameter
    t = state.wall_thickness
    L = state.length
    mat = get_structural_material(material_name)
    indices = _event_indices(history)
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0

    events = []
    for k, name in enumerate(_EVENT_ORDER):
        if progress:
            progress(name, (k + 1) / len(_EVENT_ORDER))
        i = indices.get(name)
        if i is None:
            continue
        snap = history.get_snapshot(i)
        q = snap.get("dynamic_pressure", 0.0)
        mach = snap.get("mach", 0.0)
        alt = snap.get("altitude", 0.0)
        accel = snap.get("acceleration", 0.0)
        thrust = snap.get("thrust", 0.0)
        drag = snap.get("drag", 0.0)

        # Pick the governing physical condition for this event
        if name in ("Launch", "Rail Exit", "Burnout"):
            cond = "Max Thrust"
            force = max(thrust, abs(snap.get("net_force", 0.0)), 1.0)
        elif name in ("Max-Q", "Max Mach", "Coast"):
            cond = "Max-Q"
            force = max(thrust, drag, 1.0)
        elif name == "Apogee":
            cond = "Max Thrust"
            force = max(mass * G, 1.0)
        elif name == "Parachute Deployment":
            cond = "Recovery Shock"
            force = 0.0
        else:  # Landing Impact
            cond = "Recovery Shock"
            force = 0.0

        aoa = 3.0 if cond == "Max-Q" else 2.0
        arm = abs(snap.get("cp", 0.0) - snap.get("cg", 0.0))
        r = compute_for_condition(
            cond, d, t, L, material_name,
            force=force, mach=mach, altitude_m=alt,
            vehicle_mass_kg=mass, angle_of_attack_deg=aoa,
            moment_arm_m=arm,
        )
        vm = r["von_mises"]
        sf = r["safety_factor"]

        # Buckling margin: applied axial stress / critical shell stress
        area = math.pi * d * t if (d > 0 and t > 0) else 1.0
        sigma_axial = max(force, q * 0.5 * math.pi * (d / 2) ** 2) / area
        sigma_cr = shell_buckling(mat.E, d, t, mat.nu)
        buck_margin = sigma_axial / sigma_cr if sigma_cr > 0 else 0.0

        events.append(FlightEvent(
            name=name, time=snap.get("time", 0.0),
            von_mises=vm, safety_factor=sf, buckling_margin=buck_margin,
            load_factor_g=accel / G, mach=mach, altitude=alt,
            dynamic_pressure=q,
        ))

    if not events:
        return res
    res.events = events
    res.available = True
    res.critical_event = min(events, key=lambda e: e.safety_factor)
    res.highest_stress_event = max(events, key=lambda e: e.von_mises)
    res.highest_buckling_event = max(events, key=lambda e: e.buckling_margin)
    return res


# ═════════════════════════════════════════════════════════════════════════════
# 5.  RECOVERY SYSTEM LOAD ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class RecoveryLoads:
    drogue_shock_N: float = 0.0
    main_shock_N: float = 0.0
    harness_tension_N: float = 0.0
    nosecone_separation_N: float = 0.0
    bulkhead_load_N: float = 0.0
    eyebolt_load_N: float = 0.0
    peak_force_N: float = 0.0
    drogue_deploy_velocity: float = 0.0
    main_deploy_velocity: float = 0.0
    safety_factor: float = 0.0
    status: str = "—"
    status_color: str = "#8b949e"
    allowable_N: float = 0.0


# Opening-shock amplification (Knacke, "Parachute Recovery Systems", Cx ~ 1.4)
_SHOCK_FACTOR = 1.4
# Representative hardware allowables (forged steel eyebolt + glassed bulkhead)
_EYEBOLT_PROOF_N = 5300.0   # 1/4"-20 forged eyebolt working ~ 0.5 of proof


def recovery_loads(state, history=None,
                   harness_daf: float = 1.5,
                   eyebolt_kt: float = 2.0) -> RecoveryLoads:
    """Compute deployment shock loads for drogue + main and the reaction
    loads they impose on harness, bulkhead, eyebolt and nose-cone joint."""
    rl = RecoveryLoads()
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0
    if mass <= 0:
        mass = 5.0
    drogue_cda = getattr(state, "drogue_cd_area", 0.5) or 0.5
    main_cda = getattr(state, "main_cd_area", 3.0) or 3.0
    main_alt = getattr(state, "main_deploy_altitude", 300.0) or 300.0

    # ── Drogue deploys near apogee: use velocity there if we have history ──
    v_drogue = 30.0
    apogee_alt = getattr(state, "max_altitude", 0.0)
    if history is not None and len(history) > 0:
        _, apogee_alt, iap = history.find_max("altitude")
        snap = history.get_snapshot(min(iap + 1, len(history) - 1))
        v_drogue = max(abs(snap.get("velocity", 30.0)), 10.0)
    _, _, rho_apogee = isa(apogee_alt if apogee_alt > 0 else 1000.0)
    rl.drogue_deploy_velocity = v_drogue
    rl.drogue_shock_N = 0.5 * rho_apogee * v_drogue ** 2 * drogue_cda * _SHOCK_FACTOR

    # ── Main deploys low: terminal velocity under drogue is the inflow speed ──
    _, _, rho_main = isa(main_alt)
    v_main = math.sqrt(2 * mass * G / (rho_main * drogue_cda)) if drogue_cda > 0 else 30.0
    rl.main_deploy_velocity = v_main
    rl.main_shock_N = 0.5 * rho_main * v_main ** 2 * main_cda * _SHOCK_FACTOR

    # ── Reaction loads ──
    peak = max(rl.drogue_shock_N, rl.main_shock_N)
    rl.peak_force_N = peak
    rl.harness_tension_N = peak * harness_daf
    rl.nosecone_separation_N = peak           # joint must transmit full shock
    rl.bulkhead_load_N = peak                 # reacted through aft bulkhead
    rl.eyebolt_load_N = peak * eyebolt_kt     # stress concentration at thread

    # ── Safety factor vs hardware allowable (eyebolt governs) ──
    rl.allowable_N = _EYEBOLT_PROOF_N
    gov = max(rl.eyebolt_load_N, rl.harness_tension_N)
    rl.safety_factor = rl.allowable_N / gov if gov > 0 else float("inf")
    rl.status, rl.status_color = _status_from_sf(rl.safety_factor)
    return rl


# ═════════════════════════════════════════════════════════════════════════════
# 3b.  BEAM DEFLECTION  (Euler-Bernoulli)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class BeamDeflection:
    max_deflection_mm: float = 0.0
    tip_deflection_mm: float = 0.0
    location: str = "—"
    applied_normal_force_N: float = 0.0
    bending_moment_Nm: float = 0.0
    EI: float = 0.0


def beam_deflection(state, flight: "FlightLoads", material_name: str) -> BeamDeflection:
    """Lateral deflection of the airframe treated as a cantilever beam under
    the distributed aerodynamic side load at max-Q.

        Distributed load  w = F_N / L        (N/m)
        Tip deflection     δ = w·L⁴ / (8·E·I) = F_N·L³ / (8·E·I)

    (Euler-Bernoulli, uniformly distributed load on a cantilever — Roark
    Table 8.1 case 1d.) Returns a non-zero deflection whenever a lateral
    load exists, so the deformation view is never falsely flat."""
    bd = BeamDeflection()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0 or L <= 0:
        return bd
    r_o = d / 2 + t / 2
    r_i = d / 2 - t / 2
    I = (math.pi / 4) * (r_o ** 4 - r_i ** 4)
    EI = mat.E * I
    bd.EI = EI
    if EI <= 0:
        return bd

    # Aerodynamic normal force at max-Q + angle of attack
    q = flight.max_dynamic_pressure
    if q <= 0:
        P, T, rho = isa(flight.maxq_altitude or 3000.0)
        V = (flight.max_mach or 0.6) * speed_of_sound(T)
        q = 0.5 * rho * V ** 2
    aoa = math.radians(4.0)                 # conservative gust AoA
    A_ref = math.pi * (d / 2) ** 2
    F_N = q * 2.0 * aoa * A_ref             # CN_alpha ≈ 2 /rad (slender body)
    bd.applied_normal_force_N = F_N
    bd.bending_moment_Nm = F_N * L / 2.0

    delta_m = F_N * L ** 3 / (8.0 * EI)     # cantilever, distributed load
    bd.tip_deflection_mm = delta_m * 1000.0
    bd.max_deflection_mm = bd.tip_deflection_mm
    bd.location = "Forward Airframe / Nose"
    return bd


# ═════════════════════════════════════════════════════════════════════════════
# 4b.  MODAL ESTIMATE  (analytical cantilever beam)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ModalEstimate:
    f1_hz: float = 0.0          # 1st lateral bending
    f2_hz: float = 0.0          # 2nd lateral bending
    f3_hz: float = 0.0          # 3rd lateral bending
    total_mass_kg: float = 0.0
    warning: str = ""
    low_freq: bool = False


# Euler-Bernoulli cantilever eigenvalues (βL)
_BETA_L = (1.875104, 4.694091, 7.854757)


def modal_estimate(state, material_name: str) -> ModalEstimate:
    """First three lateral-bending natural frequencies of the airframe as a
    uniform cantilever beam:

        fₙ = (βₙL)² / (2π) · √( E·I / (m'·L⁴) )

    where m' = total mass / length. Flags fundamentals below 20 Hz for
    vehicles under 3 m (resonance / controllability concern)."""
    me = ModalEstimate()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0 or L <= 0:
        return me
    r_o = d / 2 + t / 2
    r_i = d / 2 - t / 2
    I = (math.pi / 4) * (r_o ** 4 - r_i ** 4)
    EI = mat.E * I
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0
    if mass <= 0:
        mass = 5.0
    me.total_mass_kg = mass
    m_per_L = mass / L
    if EI <= 0 or m_per_L <= 0:
        return me
    freqs = [(bl ** 2 / (2 * math.pi)) * math.sqrt(EI / (m_per_L * L ** 4))
             for bl in _BETA_L]
    me.f1_hz, me.f2_hz, me.f3_hz = freqs
    if L < 3.0 and me.f1_hz < 20.0:
        me.low_freq = True
        me.warning = (f"1st bending mode {me.f1_hz:.0f} Hz is below 20 Hz for a "
                      f"{L:.1f} m vehicle — soft airframe; check flutter / control coupling.")
    return me


# ═════════════════════════════════════════════════════════════════════════════
# 6.  FIN STRUCTURAL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class FinAnalysis:
    root_bending_MPa: float = 0.0
    root_shear_MPa: float = 0.0
    tip_deflection_mm: float = 0.0
    natural_frequency_Hz: float = 0.0
    flutter_speed_m_s: float = 0.0
    flutter_margin: float = 0.0
    fin_normal_force_N: float = 0.0
    safety_factor: float = 0.0
    max_deflection_mm: float = 0.0
    highest_loaded_fin: str = "Fin 1"
    status: str = "—"
    status_color: str = "#8b949e"
    deflection_profile: list = field(default_factory=list)  # [(span_frac, defl_mm)]


def fin_analysis(state, flight: FlightLoads, material_name: str) -> FinAnalysis:
    """Cantilever-plate fin analysis: root bending/shear, tip deflection,
    fundamental frequency, and NACA flutter margin."""
    fa = FinAnalysis()
    mat = get_structural_material(material_name)

    root = getattr(state, "fin_root_chord", 0.0) or state.length * 0.1
    tip = getattr(state, "fin_tip_chord", 0.0) or root * 0.5
    span = getattr(state, "fin_span", 0.0) or getattr(state, "fin_height", 0.0) \
        or state.diameter * 0.6
    thick = getattr(state, "fin_thickness", 0.003) or 0.003
    n_fins = getattr(state, "fin_count", 3) or 3
    if span <= 0 or root <= 0 or thick <= 0:
        return fa

    # ── Aerodynamic normal force per fin at max-Q ──
    q = flight.max_dynamic_pressure
    if q <= 0:
        P, T, rho = isa(flight.maxq_altitude or 3000.0)
        V = (flight.max_mach or 0.6) * speed_of_sound(T)
        q = 0.5 * rho * V ** 2
    A_fin = 0.5 * (root + tip) * span          # trapezoid planform
    aoa = math.radians(5.0)                    # conservative gust AoA
    CN_alpha = 2 * math.pi / (1 + 2 / max(span * 2 / (root + tip), 0.5))  # finite-AR lift slope
    F_fin = q * CN_alpha * aoa * A_fin
    fa.fin_normal_force_N = F_fin

    # ── Root bending stress: load centroid at ~0.4 span, rectangular root ──
    M_root = F_fin * (0.4 * span)
    Z_root = (root * thick ** 2) / 6.0          # section modulus of root rect
    fa.root_bending_MPa = (M_root / Z_root) / 1e6 if Z_root > 0 else 0.0

    # ── Root shear ──
    A_root = root * thick
    fa.root_shear_MPa = (F_fin / A_root) / 1e6 if A_root > 0 else 0.0

    # ── Tip deflection (cantilever plate, distributed load) ──
    I_root = (root * thick ** 3) / 12.0
    EI = mat.E * I_root
    fa.tip_deflection_mm = (F_fin * span ** 3 / (8 * EI)) * 1000.0 if EI > 0 else 0.0
    fa.max_deflection_mm = fa.tip_deflection_mm
    fa.deflection_profile = [
        (f, fa.tip_deflection_mm * (f ** 2) * (3 - f) / 2.0) for f in
        [i / 10 for i in range(11)]
    ]

    # ── Fundamental cantilever frequency (Euler-Bernoulli, β1L=1.875) ──
    m_per_len = mat.density * root * thick      # mass/length of root strip
    if m_per_len > 0 and EI > 0 and span > 0:
        fa.natural_frequency_Hz = (1.875 ** 2 / (2 * math.pi)) * \
            math.sqrt(EI / (m_per_len * span ** 4))

    # ── NACA TN-4197 flutter velocity (model-rocket form) ──
    P, T, rho = isa(flight.maxq_altitude or 3000.0)
    a_sound = speed_of_sound(T)
    AR = (span ** 2) / A_fin if A_fin > 0 else 1.0
    lam = tip / root if root > 0 else 0.5
    tc = thick / root if root > 0 else 0.03
    G_shear = mat.G if mat.G > 0 else mat.E / (2 * (1 + mat.nu))
    denom = (1.337 * (AR ** 3) * P * (lam + 1)) / (2 * (AR + 2) * (tc ** 3))
    if denom > 0:
        fa.flutter_speed_m_s = a_sound * math.sqrt(G_shear / denom)
    v_max = flight.max_velocity or (flight.max_mach * a_sound)
    fa.flutter_margin = fa.flutter_speed_m_s / v_max if v_max > 0 else float("inf")

    # ── Safety factor: bending vs yield ──
    sigma = fa.root_bending_MPa * 1e6
    fa.safety_factor = mat.yield_strength / sigma if sigma > 0 else float("inf")
    fa.highest_loaded_fin = "Fin 1"   # symmetric set — all equal; gust loads one most
    fa.status, fa.status_color = _status_from_sf(min(fa.safety_factor, fa.flutter_margin))
    return fa


# ═════════════════════════════════════════════════════════════════════════════
# 7.  BUCKLING ANALYSIS (4 modes)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class BucklingMode:
    name: str
    critical: float            # critical load (N) or stress (Pa)
    applied: float
    unit: str                  # "N" or "Pa"
    margin: float = 0.0        # critical / applied (>1 = safe)
    status: str = "—"
    status_color: str = "#8b949e"


@dataclass
class BucklingAnalysis:
    modes: list = field(default_factory=list)   # list[BucklingMode]
    governing: Optional[BucklingMode] = None
    applied_axial_N: float = 0.0
    status: str = "—"
    status_color: str = "#8b949e"


def buckling_analysis(state, flight: FlightLoads, material_name: str) -> BucklingAnalysis:
    """Euler column, NASA SP-8007 shell, flat-panel, and local crippling."""
    ba = BucklingAnalysis()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0:
        return ba
    r = d / 2
    area = math.pi * d * t

    # Applied axial compression: max thrust + drag at max-Q
    thrust = flight.max_thrust or getattr(state, "thrust", 0.0)
    q = flight.max_dynamic_pressure
    F_drag = q * 0.5 * math.pi * r ** 2
    P_applied = max(thrust + F_drag, getattr(state, "weight", 0.0), 1.0)
    sigma_applied = P_applied / area
    ba.applied_axial_N = P_applied

    # 1. Euler column buckling (cantilever on rail, k=2)
    P_euler = euler_buckling(mat.E, d, t, L * 2)
    ba.modes.append(BucklingMode("Euler Column", P_euler, P_applied, "N"))

    # 2. Shell buckling (NASA SP-8007)
    sigma_shell = shell_buckling(mat.E, d, t, mat.nu)
    ba.modes.append(BucklingMode("Shell Buckling", sigma_shell, sigma_applied, "Pa"))

    # 3. Panel buckling — skin panel between fins (flat-plate, simply supported)
    n_fins = getattr(state, "fin_count", 3) or 3
    b_panel = math.pi * d / max(n_fins, 1)          # arc width between fins
    k_plate = 4.0                                    # simply-supported uniaxial
    sigma_panel = (k_plate * math.pi ** 2 * mat.E /
                   (12 * (1 - mat.nu ** 2))) * (t / b_panel) ** 2 if b_panel > 0 else float("inf")
    ba.modes.append(BucklingMode("Panel Buckling", sigma_panel, sigma_applied, "Pa"))

    # 4. Local crippling (Gerard method for curved skin)
    sigma_cripple = 0.6 * mat.E * (t / r) if r > 0 else float("inf")
    sigma_cripple = min(sigma_cripple, mat.yield_strength)
    ba.modes.append(BucklingMode("Local Crippling", sigma_cripple, sigma_applied, "Pa"))

    for m in ba.modes:
        m.margin = (m.critical / m.applied) if m.applied > 0 else float("inf")
        m.status, m.status_color = _status_from_sf(m.margin)

    ba.governing = min(ba.modes, key=lambda m: m.margin)
    ba.status, ba.status_color = ba.governing.status, ba.governing.status_color
    return ba


# ═════════════════════════════════════════════════════════════════════════════
# 10.  MASS EFFICIENCY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class MassEfficiency:
    current_mass_kg: float = 0.0
    required_mass_kg: float = 0.0
    overbuilt_pct: float = 0.0
    efficiency_pct: float = 0.0
    optimization_potential: str = "—"     # LOW | MODERATE | HIGH
    potential_color: str = "#8b949e"


def mass_efficiency(state, assembly, governing_sf: float,
                    sf_target: float = 1.5) -> MassEfficiency:
    """Compare current structural mass against the minimum mass that would
    still meet the target safety factor (mass scales ~linearly with wall
    thickness, and SF scales ~linearly with thickness for stress-limited
    designs)."""
    me = MassEfficiency()

    # Current structural mass: load-bearing categories only
    struct_cats = {"Body", "Structure", "Inner"}
    cur = 0.0
    if assembly is not None:
        for c in assembly.all_components():
            if getattr(c, "category", "") in struct_cats:
                try:
                    cur += c.computed_mass()
                except Exception:
                    pass
    if cur <= 0:
        # Fallback: estimate shell mass from geometry
        mat = get_material(state.material_name)
        area = math.pi * state.diameter * state.wall_thickness
        cur = mat["density"] * area * state.length
    me.current_mass_kg = cur

    if governing_sf <= 0 or not math.isfinite(governing_sf):
        me.required_mass_kg = cur
        me.efficiency_pct = 100.0
        me.optimization_potential = "LOW"
        me.potential_color = "#7ee787"
        return me

    # Thickness scale to bring SF down to target. Floored at the minimum
    # practical wall gauge (0.5 mm) — you cannot thin the wall below what is
    # manufacturable/handleable regardless of how high the stress SF is.
    min_gauge = 0.0005
    t_cur = max(getattr(state, "wall_thickness", 0.002), min_gauge)
    gauge_floor = min_gauge / t_cur
    scale = sf_target / governing_sf
    scale = max(scale, gauge_floor)
    scale = min(scale, 1.0)
    me.required_mass_kg = cur * scale
    me.overbuilt_pct = min((cur - me.required_mass_kg) / me.required_mass_kg * 100.0
                           if me.required_mass_kg > 0 else 0.0, 400.0)
    me.efficiency_pct = me.required_mass_kg / cur * 100.0 if cur > 0 else 100.0

    if me.overbuilt_pct >= 30:
        me.optimization_potential, me.potential_color = "HIGH", "#f0883e"
    elif me.overbuilt_pct >= 12:
        me.optimization_potential, me.potential_color = "MODERATE", "#d29922"
    else:
        me.optimization_potential, me.potential_color = "LOW", "#7ee787"
    return me


# ═════════════════════════════════════════════════════════════════════════════
# 11.  STRUCTURAL SAFETY SCORE
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class SafetyScore:
    score: int = 0
    grade: str = "—"
    color: str = "#8b949e"
    yield_margin: float = 0.0
    buckling_margin: float = 0.0
    recovery_margin: float = 0.0
    fin_margin: float = 0.0
    thermal_margin: float = 0.0
    subscores: dict = field(default_factory=dict)


def _sf_to_subscore(sf: float) -> float:
    """Map a safety factor to a 0-100 sub-score (SF 3+ → 100, 1.5 → 70,
    1.0 → 40, <1 → linear to 0)."""
    if not math.isfinite(sf):
        return 100.0
    if sf >= 3.0:
        return 100.0
    if sf >= 1.5:
        return 70.0 + (sf - 1.5) / 1.5 * 30.0
    if sf >= 1.0:
        return 40.0 + (sf - 1.0) / 0.5 * 30.0
    return max(0.0, sf * 40.0)


def safety_score(yield_sf: float, buckling_sf: float, recovery_sf: float,
                 fin_sf: float, thermal_sf: float) -> SafetyScore:
    """Weighted 0-100 structural safety score across the five margins."""
    ss = SafetyScore()
    ss.yield_margin = yield_sf
    ss.buckling_margin = buckling_sf
    ss.recovery_margin = recovery_sf
    ss.fin_margin = fin_sf
    ss.thermal_margin = thermal_sf

    weights = {"yield": 0.30, "buckling": 0.25, "recovery": 0.15,
               "fin": 0.15, "thermal": 0.15}
    subs = {
        "yield": _sf_to_subscore(yield_sf),
        "buckling": _sf_to_subscore(buckling_sf),
        "recovery": _sf_to_subscore(recovery_sf),
        "fin": _sf_to_subscore(fin_sf),
        "thermal": _sf_to_subscore(thermal_sf),
    }
    ss.subscores = subs
    ss.score = int(round(sum(subs[k] * weights[k] for k in weights)))

    if ss.score > 85:
        ss.grade, ss.color = "EXCELLENT", "#2ecc71"
    elif ss.score >= 70:
        ss.grade, ss.color = "GOOD", "#f1c40f"
    elif ss.score >= 50:
        ss.grade, ss.color = "MARGINAL", "#e67e22"
    else:
        ss.grade, ss.color = "CRITICAL", "#e74c3c"
    return ss


# ═════════════════════════════════════════════════════════════════════════════
# 12.  THERMAL PROFILE
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ThermalProfile:
    skin_temp_K: float = 293.15
    internal_temp_K: float = 293.15
    gradient_K: float = 0.0
    expansion_mm: float = 0.0
    thermal_stress_MPa: float = 0.0
    max_thermal_stress_MPa: float = 0.0
    max_stress_altitude: float = 0.0
    service_limit_K: float = 423.0
    exceeds_limit: bool = False
    heat_flux_W_m2: float = 0.0                    # convective wall heat flux
    profile: list = field(default_factory=list)    # [(altitude_m, thermal_stress_MPa)]
    body_station_temps: list = field(default_factory=list)  # [(x_frac, T_K, label)]


def thermal_profile(state, flight: FlightLoads, history, material_name: str) -> ThermalProfile:
    """Aerodynamic-heating thermal stress, plus a thermal-stress-vs-altitude
    sweep over the ascent."""
    tp = ThermalProfile()
    mat = get_structural_material(material_name)
    tp.service_limit_K = mat.max_service_temp

    def thermal_at(mach, alt):
        r = compute_for_condition("Thermal", state.diameter, state.wall_thickness,
                                  state.length, material_name, mach=mach, altitude_m=alt)
        return r

    # Peak point: max mach (hottest skin). Use the altitude AT max Mach — not
    # the max-Q altitude — so the atmospheric state matches the Mach used.
    mach_pk = flight.max_mach or 1.0
    alt_pk = flight.maxmach_altitude or flight.maxq_altitude or 10000.0
    r = thermal_at(mach_pk, alt_pk)
    T_amb = isa(alt_pk)[1]
    T_recovery = r.get("T_recovery", T_amb)
    T_stag = r.get("T_stag", T_recovery)
    tp.skin_temp_K = T_recovery
    # Internal wall lags external by conduction through the shell — small for
    # a short transient; ~25% of the external rise reaches the interior.
    tp.internal_temp_K = T_amb + 0.25 * (T_recovery - T_amb)
    tp.gradient_K = T_stag - T_recovery     # nose-to-body gradient
    tp.expansion_mm = mat.cte * (tp.skin_temp_K - 293.15) * state.length * 1000.0
    tp.thermal_stress_MPa = r["thermal"] / 1e6
    tp.exceeds_limit = tp.skin_temp_K > tp.service_limit_K

    # ── Along-body wall temperature distribution (nose hottest → fins cooler) ──
    # Recovery (adiabatic-wall) temperature scales with the local recovery
    # factor; the stagnation nose tip sees T_stag, the cylindrical body the
    # turbulent recovery temperature, fins/aft slightly cooler.
    V_pk = mach_pk * speed_of_sound(T_amb)
    tp.body_station_temps = [
        (0.00, T_stag, "Nose tip (stagnation)"),
        (0.12, T_amb + 0.92 * (T_recovery - T_amb), "Nose shoulder"),
        (0.40, T_recovery, "Forward body"),
        (0.70, T_amb + 0.95 * (T_recovery - T_amb), "Aft body"),
        (0.95, T_amb + 0.85 * (T_recovery - T_amb), "Fins"),
    ]
    # Convective wall heat flux (turbulent flat plate, Stanton ≈ 0.0015):
    #   q" = St · ρ · V · cp · (T_recovery − T_wall)
    _, T_pk, rho_pk = isa(alt_pk)
    cp_air = 1005.0
    St = 0.0015
    T_wall = 293.15
    tp.heat_flux_W_m2 = max(0.0, St * rho_pk * V_pk * cp_air * (T_recovery - T_wall))

    # Sweep over ascent
    pts = []
    if history is not None and len(history) > 0:
        machs = history.get_values("mach")
        alts = history.get_values("altitude")
        step = max(1, len(machs) // 40)
        for i in range(0, len(machs), step):
            rr = thermal_at(machs[i], alts[i])
            pts.append((alts[i], rr["thermal"] / 1e6))
    else:
        for alt in range(0, 12000, 500):
            rr = thermal_at(mach_pk, alt)
            pts.append((float(alt), rr["thermal"] / 1e6))
    tp.profile = pts
    if pts:
        tp.max_thermal_stress_MPa = max(p[1] for p in pts)
        tp.max_stress_altitude = max(pts, key=lambda p: p[1])[0]
    else:
        tp.max_thermal_stress_MPa = tp.thermal_stress_MPa
    return tp


# ═════════════════════════════════════════════════════════════════════════════
# 9.  STRUCTURAL FAILURE MAP
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ComponentStatus:
    name: str
    subsystem: str
    margin: float = 0.0           # safety factor
    status: str = "—"             # SAFE | MARGIN | WARNING | FAILURE
    color: str = "#8b949e"
    detail: str = ""


@dataclass
class FailureMap:
    components: list = field(default_factory=list)   # list[ComponentStatus]
    weakest: Optional[ComponentStatus] = None


# Subsystem → governing-margin source key
_SUBSYSTEMS = [
    "Motor Mount", "Airframe", "Fins", "Couplers", "Bulkheads",
    "Recovery System", "Nose Cone", "Avionics Bay",
]


def _status_from_sf(sf: float) -> tuple[str, str]:
    """Spec status colors: Green safe, Yellow<1.5, Orange<1.2, Red<1.0."""
    if not math.isfinite(sf) or sf >= 1.5:
        return "SAFE", "#2ecc71"
    if sf >= 1.2:
        return "MARGIN", "#f1c40f"
    if sf >= 1.0:
        return "WARNING", "#e67e22"
    return "FAILURE", "#e74c3c"


def failure_map(state, body_sf: float, fin: FinAnalysis,
                recovery: RecoveryLoads, buckling: BucklingAnalysis,
                thermal: ThermalProfile) -> FailureMap:
    """Roll the analysis results up into a per-subsystem dashboard."""
    fm = FailureMap()
    thermal_sf = (thermal.service_limit_K / thermal.skin_temp_K
                  if thermal.skin_temp_K > 0 else 99.0)

    rows = {
        "Motor Mount": (min(body_sf, buckling.governing.margin if buckling.governing else body_sf),
                        "Thrust transfer + buckling"),
        "Airframe": (min(body_sf, *(m.margin for m in buckling.modes) if buckling.modes else (body_sf,)),
                     "Axial + bending + shell buckling"),
        "Fins": (min(fin.safety_factor, fin.flutter_margin), "Root bending + flutter"),
        "Couplers": (body_sf * 0.9, "Joint shear + bending"),
        "Bulkheads": (recovery.safety_factor, "Recovery shock reaction"),
        "Recovery System": (recovery.safety_factor, "Deployment shock"),
        "Nose Cone": (max(body_sf, thermal_sf), "Stagnation heating + separation"),
        "Avionics Bay": (body_sf * 1.2, "Vibration + acceleration"),
    }
    for name in _SUBSYSTEMS:
        sf, detail = rows[name]
        st, col = _status_from_sf(sf)
        fm.components.append(ComponentStatus(
            name=name, subsystem=name, margin=sf,
            status=st, color=col, detail=detail,
        ))
    fm.weakest = min(fm.components, key=lambda c: c.margin)
    return fm


# ═════════════════════════════════════════════════════════════════════════════
# 8.  LOAD-PATH VISUALIZATION DATA
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class LoadStation:
    name: str
    force_N: float = 0.0
    position: float = 0.0   # axial position (m, from nose)


@dataclass
class LoadPath:
    stations: list = field(default_factory=list)   # ordered nose→tail force flow
    peak_force_N: float = 0.0


def load_path(state, flight: FlightLoads) -> LoadPath:
    """Trace the primary compressive load path from the motor up through the
    airframe to the nose cone. Force decreases as distributed mass is shed
    above each station under acceleration."""
    lp = LoadPath()
    thrust = flight.max_thrust or getattr(state, "thrust", 0.0)
    if thrust <= 0:
        thrust = getattr(state, "motor_avg_thrust", 0.0) or 1.0
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0
    accel = thrust / mass if mass > 0 else G
    L = state.length or 1.0

    # Force at each station = thrust minus inertia of structure forward of it.
    # Model as fractions of vehicle mass carried at each interface.
    chain = [
        ("Motor", 1.00, 0.95 * L),
        ("Motor Mount", 0.95, 0.88 * L),
        ("Aft Airframe", 0.80, 0.70 * L),
        ("Coupler", 0.55, 0.55 * L),
        ("Recovery Bay", 0.40, 0.40 * L),
        ("Fwd Airframe", 0.25, 0.25 * L),
        ("Nose Cone", 0.08, 0.08 * L),
    ]
    for name, frac, pos in chain:
        lp.stations.append(LoadStation(name, thrust * frac, pos))
    lp.peak_force_N = thrust
    return lp


# ═════════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL AGGREGATOR
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class WorkstationReport:
    flight: FlightLoads = field(default_factory=FlightLoads)
    body_condition: dict = field(default_factory=dict)
    recovery: RecoveryLoads = field(default_factory=RecoveryLoads)
    fin: FinAnalysis = field(default_factory=FinAnalysis)
    buckling: BucklingAnalysis = field(default_factory=BucklingAnalysis)
    mass: MassEfficiency = field(default_factory=MassEfficiency)
    thermal: ThermalProfile = field(default_factory=ThermalProfile)
    score: SafetyScore = field(default_factory=SafetyScore)
    failure: FailureMap = field(default_factory=FailureMap)
    loads: LoadPath = field(default_factory=LoadPath)
    deflection: BeamDeflection = field(default_factory=BeamDeflection)
    modal: ModalEstimate = field(default_factory=ModalEstimate)
    warnings: list = field(default_factory=list)   # physics-consistency warnings
    verdict: str = "—"
    verdict_color: str = "#8b949e"


def full_analysis(state, assembly, history, material_name: str,
                  condition: str = "Max-Q") -> WorkstationReport:
    """Run the complete workstation analysis suite and assemble a report.
    Designed to complete in well under a second."""
    rep = WorkstationReport()
    rep.flight = FlightLoads.from_history(history, state)

    # Governing body stress for the selected condition
    mass = state.total_mass() if callable(getattr(state, "total_mass", None)) else 5.0
    force = max(rep.flight.max_thrust, getattr(state, "thrust", 0.0), mass * G, 1.0)
    rep.body_condition = compute_for_condition(
        condition, state.diameter, state.wall_thickness, state.length,
        material_name, force=force, mach=rep.flight.maxq_mach,
        altitude_m=rep.flight.maxq_altitude, vehicle_mass_kg=mass,
        angle_of_attack_deg=3.0 if condition == "Max-Q" else 2.0,
        moment_arm_m=rep.flight.moment_arm,
    )
    body_sf = rep.body_condition["safety_factor"]

    rep.recovery = recovery_loads(state, history)
    rep.fin = fin_analysis(state, rep.flight, material_name)
    rep.buckling = buckling_analysis(state, rep.flight, material_name)
    rep.thermal = thermal_profile(state, rep.flight, history, material_name)
    rep.mass = mass_efficiency(state, assembly,
                               min(body_sf, rep.buckling.governing.margin
                                   if rep.buckling.governing else body_sf))

    thermal_sf = (rep.thermal.service_limit_K / rep.thermal.skin_temp_K
                  if rep.thermal.skin_temp_K > 0 else 99.0)
    buck_sf = rep.buckling.governing.margin if rep.buckling.governing else 99.0
    rep.score = safety_score(body_sf, buck_sf, rep.recovery.safety_factor,
                             min(rep.fin.safety_factor, rep.fin.flutter_margin),
                             thermal_sf)
    rep.failure = failure_map(state, body_sf, rep.fin, rep.recovery,
                              rep.buckling, rep.thermal)
    rep.loads = load_path(state, rep.flight)
    rep.deflection = beam_deflection(state, rep.flight, material_name)
    rep.modal = modal_estimate(state, material_name)

    # Physics-consistency validation (imported lazily to avoid cycles)
    try:
        from structures.validation import validate_report
        rep.warnings = validate_report(state, rep)
    except Exception as e:
        logger.debug(f"validation skipped: {e}")

    # Final verdict
    gov = rep.failure.weakest.margin if rep.failure.weakest else body_sf
    if gov >= 1.5 and rep.score.score >= 85:
        rep.verdict, rep.verdict_color = "PASS", "#2ecc71"
    elif gov >= 1.0:
        rep.verdict, rep.verdict_color = "PASS WITH MARGIN", "#f1c40f"
    else:
        rep.verdict, rep.verdict_color = "FAIL", "#e74c3c"
    return rep


# ═════════════════════════════════════════════════════════════════════════════
#  STANDALONE SMOKE TEST
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time as _time

    class _State:
        diameter = 0.10
        length = 1.5
        wall_thickness = 0.002
        material_name = "Aluminum 6061-T6"
        dry_mass = 3.0
        propellant_mass = 1.1
        fin_root_chord = 0.15
        fin_tip_chord = 0.07
        fin_span = 0.06
        fin_height = 0.06
        fin_thickness = 0.003
        fin_count = 3
        thrust = 1200.0
        weight = 40.0
        motor_max_thrust = 1500.0
        motor_avg_thrust = 1100.0
        max_altitude = 3000.0
        max_velocity = 280.0
        max_mach = 0.85
        max_acceleration = 120.0
        dynamic_pressure = 48000.0
        temperature_ambient = 288.15
        drogue_cd_area = 0.5
        main_cd_area = 3.0
        main_deploy_altitude = 300.0
        wind_speed = 5.0

        def total_mass(self):
            return self.dry_mass + self.propellant_mass

    s = _State()
    t0 = _time.time()
    rep = full_analysis(s, None, None, "Aluminum 6061-T6", "Max-Q")
    dt = _time.time() - t0
    print(f"full_analysis ran in {dt*1000:.1f} ms")
    print(f"Body σ_vm = {rep.body_condition['von_mises']/1e6:.1f} MPa, "
          f"SF = {rep.body_condition['safety_factor']:.2f}")
    print(f"Recovery: main={rep.recovery.main_shock_N:.0f} N, "
          f"harness={rep.recovery.harness_tension_N:.0f} N, "
          f"SF={rep.recovery.safety_factor:.2f} ({rep.recovery.status})")
    print(f"Fin: root bend={rep.fin.root_bending_MPa:.1f} MPa, "
          f"tip defl={rep.fin.tip_deflection_mm:.2f} mm, "
          f"f1={rep.fin.natural_frequency_Hz:.0f} Hz, "
          f"flutter={rep.fin.flutter_speed_m_s:.0f} m/s (margin {rep.fin.flutter_margin:.2f})")
    print(f"Buckling governing: {rep.buckling.governing.name} "
          f"margin={rep.buckling.governing.margin:.2f}")
    print(f"Mass: cur={rep.mass.current_mass_kg:.2f} kg, "
          f"req={rep.mass.required_mass_kg:.2f} kg, "
          f"overbuilt={rep.mass.overbuilt_pct:.0f}% ({rep.mass.optimization_potential})")
    print(f"Thermal: skin={rep.thermal.skin_temp_K:.0f} K, "
          f"σ_th={rep.thermal.thermal_stress_MPa:.1f} MPa")
    print(f"Safety score: {rep.score.score}/100 ({rep.score.grade})")
    print(f"Weakest: {rep.failure.weakest.name} SF={rep.failure.weakest.margin:.2f}")
    print(f"VERDICT: {rep.verdict}")
