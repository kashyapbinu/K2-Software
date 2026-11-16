"""
K2 AeroSim — Structures Workstation Backend
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


def _site_altitude(state) -> float:
    """Launch-site elevation above sea level (m). Flight altitudes are above
    the pad; ISA wants them above sea level."""
    try:
        return float(getattr(state, "launch_altitude", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
    # The *_altitude fields are above SEA LEVEL (the ISA lookups here need
    # that): the flight's height above the pad plus the launch-site elevation.
    maxq_altitude: float = 0.0    # m
    maxq_time: float = 0.0        # s
    maxq_thrust: float = 0.0      # N, thrust at the max-Q instant
    maxq_drag: float = 0.0        # N, drag at the max-Q instant
    maxmach_altitude: float = 0.0  # m (altitude at peak Mach — peak heating)
    maxthrust_mach: float = 0.0    # Mach at the peak-thrust instant
    maxthrust_altitude: float = 0.0  # m (altitude at peak thrust — early boost)
    wind_speed: float = 0.0       # m/s
    vehicle_mass: float = 5.0     # kg (at max-Q)
    moment_arm: float = 0.0       # m, |x_CP - x_CG| at max-Q (bending arm)

    @classmethod
    def from_history(cls, history, state=None, summary=None) -> "FlightLoads":
        """Extract peak loads from a HistoryManager. Returns available=False
        when no flight data exists.

        ``summary`` is that flight's FlightSummary when there is one. Its
        peak acceleration is the sensed one (what loads the airframe, and
        what the Results tab and the report give). Without it the peak is
        the recorder's d|v|/dt, about 1 g less on the way up: the Structures
        tab showed a load factor of 19.0 G for a flight the Results tab gave
        20.0 g.
        """
        if history is None or len(history) == 0:
            return cls.from_state(state)

        t_q, q_max, idx_q = history.find_max("dynamic_pressure")
        _, v_max, _ = history.find_max("velocity")
        _, m_max, idx_m = history.find_max("mach")
        _, a_max, _ = history.find_max("acceleration")
        if getattr(summary, "max_acceleration", None):
            a_max = summary.max_acceleration
        _, thr_max, idx_thr = history.find_max("thrust")

        snap = history.get_snapshot(idx_q) if idx_q is not None else {}
        snap_m = history.get_snapshot(idx_m) if idx_m is not None else {}
        snap_thr = history.get_snapshot(idx_thr) if idx_thr is not None else {}
        site = _site_altitude(state)
        wind = getattr(state, "wind_speed", 0.0) if state else 0.0
        mass = snap.get("mass", 0.0) or (vehicle_mass(state) if state else 5.0)
        # Bending moment arm at max-Q = distance between centre of pressure and
        # centre of gravity (recorded per-tick by the sim).
        arm = abs(snap.get("cp", 0.0) - snap.get("cg", 0.0))

        return cls(
            available=True, source="simulation",
            max_velocity=v_max, max_mach=m_max,
            max_acceleration=a_max, max_accel_g=a_max / G,
            max_dynamic_pressure=q_max, max_thrust=thr_max,
            maxq_mach=snap.get("mach", m_max),
            maxq_altitude=site + snap.get("altitude", 0.0),
            maxmach_altitude=site + snap_m.get("altitude", snap.get("altitude", 0.0)),
            maxthrust_mach=snap_thr.get("mach", 0.0),
            maxthrust_altitude=site + snap_thr.get("altitude", 0.0),
            maxq_time=t_q, wind_speed=wind,
            maxq_thrust=snap.get("thrust", 0.0), maxq_drag=snap.get("drag", 0.0),
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
        mass = vehicle_mass(state)
        return cls(
            available=has, source="state" if has else "none",
            max_velocity=getattr(state, "max_velocity", 0.0),
            max_mach=getattr(state, "max_mach", 0.0),
            max_acceleration=getattr(state, "max_acceleration", 0.0),
            max_accel_g=getattr(state, "max_acceleration", 0.0) / G,
            max_dynamic_pressure=getattr(state, "dynamic_pressure", 0.0),
            max_thrust=getattr(state, "motor_max_thrust", 0.0) or getattr(state, "thrust", 0.0),
            maxq_mach=getattr(state, "max_mach", 0.0),
            maxq_altitude=_site_altitude(state) + getattr(state, "altitude", 0.0),
            maxmach_altitude=_site_altitude(state) + getattr(state, "altitude", 0.0),
            wind_speed=getattr(state, "wind_speed", 0.0),
            vehicle_mass=mass if mass and mass > 0 else 5.0,
            moment_arm=abs(getattr(state, "cp", 0.0) - getattr(state, "cg", 0.0)),
        )

    def max_speed(self) -> float:
        """Fastest the rocket flew (m/s), 0 when no flight is known. The speed
        every flutter margin in the Structures workspace is judged against."""
        if self.max_velocity:
            return self.max_velocity
        return self.max_mach * speed_of_sound(isa(self.maxq_altitude or 3000.0)[1])


def vehicle_mass(state) -> float:
    """Liftoff mass of the vehicle (kg): airframe, motor case and the full
    propellant load (RocketState.liftoff_mass).

    The load cases, the modal estimate and the reports took
    ``state.total_mass()``, which is the live mass: the liftoff mass before
    a flight and the burnt-out mass after one. The same rocket gave two sets
    of structural numbers depending on whether it had been flown (canonical
    rocket: 7.4 kg, then 6.0 kg). An instant of the flight that recorded its
    own mass (FlightLoads.from_history) still uses that.

    5 kg for an object that carries no mass at all (a bare stand-in state).
    """
    for name in ("liftoff_mass", "total_mass"):
        fn = getattr(state, name, None)
        if callable(fn):
            mass = fn()
            if isinstance(mass, (int, float)) and mass > 0:
                return float(mass)
    return 5.0


def aero_drag(state, mach: float, altitude_asl: float) -> float:
    """Drag (N) of this rocket at zero angle of attack, at ``mach`` and
    ``altitude_asl`` (m above sea level, ISA), from the aero model the
    flight uses (physics.aerodynamics.AeroModel). For a load case typed in
    by hand, where no flight recorded the drag."""
    if mach <= 0:
        return 0.0
    from physics.aerodynamics import AeroModel
    _, T, rho = isa(altitude_asl)
    v = mach * speed_of_sound(T)
    q = 0.5 * rho * v * v
    out = AeroModel.from_state(state).compute(0.0, mach, q, 0.0, v, getattr(state, "cg", 0.0))
    return abs(float(out["F_drag"]))


def flight_known(state, history) -> bool:
    """True when there is a flight to take the loads from: a recorded
    history, or the maxima a flight left on the state.

    The functions below still run without one, on stand-in conditions (fins
    and deflection at Mach 0.6 and 3000 m, heating at Mach 1.0 and 10 km,
    canopies opening at 30 m/s from 1000 m). Those are not this rocket's
    loads; a caller that shows results to the user asks this first.
    """
    return FlightLoads.from_history(history, state).available


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


def _event_indices(history, state=None) -> dict:
    """Map each named flight event to a history index."""
    n = len(history)
    if n == 0:
        return {}
    idx = {}
    idx["Launch"] = 0
    # Rail exit: first sample above the launch rod/rail length. Uses
    # altitude — history "x" is DOWNRANGE distance, which the old code read
    # as travel along the rail and fired only once the rocket had drifted 2 m
    # sideways (typically near burnout).
    rod_len = 1.0
    if state is not None:
        rod_len = getattr(state, "launch_rod_length", 1.0) or 1.0
    alts = history.get_values("altitude")
    rail = 0
    for i, z in enumerate(alts):
        if z > rod_len:
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
    indices = _event_indices(history, state)
    mass = vehicle_mass(state)
    # One recovery-shock model for the whole report: harness tension from the
    # real CdA + deploy velocity (recovery_loads), not a separate 15 g guess.
    rec_force = recovery_loads(state, history).harness_tension_N

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
            force = thrust
        elif name == "Apogee":
            cond = "Max Thrust"
            force = max(mass * G, 1.0)
        elif name == "Parachute Deployment":
            cond = "Recovery Shock"
            force = rec_force
        else:  # Landing Impact
            cond = "Recovery Shock"
            force = rec_force

        aoa = 3.0 if cond == "Max-Q" else 2.0
        arm = abs(snap.get("cp", 0.0) - snap.get("cg", 0.0))
        r = compute_for_condition(
            cond, d, t, L, material_name,
            force=force, mach=mach, altitude_m=alt,
            vehicle_mass_kg=mass, angle_of_attack_deg=aoa,
            moment_arm_m=arm, drag_force_N=drag, q_dyn=q,
        )
        vm = r["von_mises"]
        sf = r["safety_factor"]

        # Buckling margin: applied axial compression / critical shell stress.
        # The compression is the larger of thrust and the drag the flight
        # recorded (it was thrust or 0.5 x q x frontal area). A parachute
        # pulls the airframe in tension, which does not buckle it.
        area = math.pi * d * t if (d > 0 and t > 0) else 1.0
        compression = 0.0 if cond == "Recovery Shock" else max(force, drag)
        sigma_axial = compression / area
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
    # None: not assessed. K2 holds no rated strength for the harness,
    # eyebolt or bulkhead, so there is nothing to divide the load by.
    safety_factor: Optional[float] = None
    rating_N: float = 0.0          # RocketState.recovery_hardware_rating
    status: str = "—"
    status_color: str = "#8b949e"
    # Per canopy, in opening order: name, deploy_altitude (m above pad),
    # inflow_velocity (m/s), shock_N.
    canopies: list = field(default_factory=list)


# Opening-shock amplification (Knacke, "Parachute Recovery Systems", Cx ~ 1.4)
_SHOCK_FACTOR = 1.4


def recovery_loads(state, history=None,
                   harness_daf: float = 1.5) -> RecoveryLoads:
    """Opening-shock loads of every canopy the vehicle carries, and the
    reaction loads they impose on harness, bulkhead, eyebolt and nose-cone
    joint.

    The canopies are the ones the flight flies (RecoverySystem.for_state),
    taken in the order they open. A canopy's inflow speed is the terminal
    speed under the canopies already open at its deploy height; the first
    one opens at apogee speed (from the flight history when there is one),
    or, if nothing is open above an altitude trigger, at the free-fall speed
    from apogee (drag ignored: an upper bound).

    The safety factor is the harness tension against
    ``state.recovery_hardware_rating``, the rated strength the user entered
    for the weakest part of the load path; with none entered there is no
    safety factor. The result used to carry one against a 5300 N eyebolt
    nobody had entered (a 1/4"-20 forged eyebolt, assumed),
    and an "eyebolt load" of twice the shock (a stress concentration applied
    to a force), so any rocket with a parachute was told its recovery
    hardware was SAFE (6.99 on the canonical rocket). The eyebolt carries
    the harness tension.

    This used to read the two Simulation-tab slots through ``or 0.5`` and
    ``or 3.0``, so a design with one parachute, or none, was checked against
    canopies it does not have.
    """
    from recovery.parachute_dynamics import APOGEE, RecoverySystem

    rl = RecoveryLoads()
    mass = vehicle_mass(state)
    if mass <= 0:
        mass = 5.0
    # The mass that hangs from the canopies: the burnt-out vehicle. It is
    # read as dry mass + motor case. It used to be total_mass() minus the
    # propellant load, and after a flight total_mass() is already the
    # burnt-out mass, so the propellant came off twice: the canonical rocket
    # hung 4.6 kg instead of 6.0 kg and its main opening shock read 379 N
    # where the same rocket, before flying, read 494 N.
    dry = getattr(state, "dry_mass", None)
    if dry is not None and dry > 0:
        descent_mass = dry + (getattr(state, "motor_dry_mass", 0.0) or 0.0)
    else:
        descent_mass = mass - (getattr(state, "propellant_mass_initial", 0.0) or 0.0)
    if descent_mass <= 0:
        descent_mass = mass

    v_apogee = 30.0          # horizontal speed at apogee when no flight was run
    apogee_alt = getattr(state, "max_altitude", 0.0)
    if history is not None and len(history) > 0:
        _, apogee_alt, iap = history.find_max("altitude")
        snap = history.get_snapshot(min(iap + 1, len(history) - 1))
        v_apogee = max(abs(snap.get("velocity", 30.0)), 10.0)
    if apogee_alt <= 0:
        apogee_alt = 1000.0
    site = _site_altitude(state)

    canopies = RecoverySystem.for_state(state).canopies
    order = sorted((c for c in canopies if c.event == APOGEE), key=lambda c: c.delay) + \
        sorted((c for c in canopies if c.event != APOGEE), key=lambda c: -c.altitude)
    open_area = 0.0
    for c in order:
        alt = apogee_alt if c.event == APOGEE else min(c.altitude, apogee_alt)
        _, _, rho = isa(site + alt)
        if open_area > 0.0:
            v_in = math.sqrt(2 * descent_mass * G / (rho * open_area))
        elif c.event == APOGEE:
            v_in = v_apogee
        else:
            v_in = max(v_apogee, math.sqrt(2 * G * max(apogee_alt - alt, 0.0)))
        shock = 0.5 * rho * v_in ** 2 * c.cd_area * _SHOCK_FACTOR
        rl.canopies.append({"name": c.name, "deploy_altitude": alt,
                            "inflow_velocity": v_in, "shock_N": shock})
        open_area += c.cd_area

    # The two summary slots the panel shows: the first canopy to open, and
    # the hardest-opening one after it.
    if rl.canopies:
        first = rl.canopies[0]
        rl.drogue_shock_N, rl.drogue_deploy_velocity = first["shock_N"], first["inflow_velocity"]
        rest = rl.canopies[1:]
        if rest:
            worst = max(rest, key=lambda r: r["shock_N"])
            rl.main_shock_N, rl.main_deploy_velocity = worst["shock_N"], worst["inflow_velocity"]
    else:
        rl.status = "No parachute"
        return rl

    # ── Reaction loads ──
    peak = max(r["shock_N"] for r in rl.canopies)
    rl.peak_force_N = peak
    rl.harness_tension_N = peak * harness_daf
    rl.nosecone_separation_N = peak           # joint must transmit full shock
    rl.bulkhead_load_N = peak                 # reacted through aft bulkhead
    rl.eyebolt_load_N = rl.harness_tension_N  # the harness pulls on it

    # Safety factor against the rating the user entered for the weakest part
    # of the recovery load path. None when there is none.
    rating = getattr(state, "recovery_hardware_rating", 0.0) or 0.0
    rl.rating_N = rating
    if rating > 0 and rl.harness_tension_N > 0:
        rl.safety_factor = rating / rl.harness_tension_N
        rl.status, rl.status_color = _status_from_sf(rl.safety_factor)
    else:
        rl.status = "Enter the hardware rating to get a safety factor"
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
    """Lateral (elastic) deflection of the airframe under the aerodynamic side
    load at max-Q.

    A rocket in free flight is a FREE-FREE beam: nothing clamps it. The aero
    normal force F_N is balanced by the distributed inertia of the vehicle
    (w = F_N / L), so the elastic bend is the deflection of the ends relative
    to the load point — two back-to-back cantilevers of length L/2 under the
    uniform inertial load:

        δ_end = w·(L/2)⁴ / (8·E·I) = F_N·L³ / (128·E·I)

    (Euler-Bernoulli, Roark Table 8.1 case 1d applied to each half.) The old
    cantilever-from-the-tail form F_N·L³/(8EI) over-predicted by 16×.
    Returns a non-zero deflection whenever a lateral load exists, so the
    deformation view is never falsely flat."""
    bd = BeamDeflection()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0 or L <= 0:
        return bd
    r_o = d / 2                 # d = outer diameter
    r_i = d / 2 - t
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

    delta_m = F_N * L ** 3 / (128.0 * EI)   # free-free, inertia-balanced
    bd.tip_deflection_mm = delta_m * 1000.0
    bd.max_deflection_mm = bd.tip_deflection_mm
    bd.location = "Nose / Tail (relative to mid-body)"
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


# Euler-Bernoulli FREE-FREE eigenvalues (βL). A rocket in flight is not
# clamped anywhere; the cantilever set (1.875, 4.694, 7.855) under-predicts
# the fundamental by (4.730/1.875)² ≈ 6.4×.
_BETA_L = (4.730041, 7.853205, 10.995608)
_BETA_L_BY_BC = {
    "free-free": _BETA_L,
    "cantilever": (1.875104, 4.694091, 7.854757),   # benchmark vs clamped FE only
}


def modal_estimate(state, material_name: str,
                   boundary: str = "free-free") -> ModalEstimate:
    """First three lateral-bending natural frequencies of the airframe as a
    uniform FREE-FREE beam (free flight — no clamped end):

        fₙ = (βₙL)² / (2π) · √( E·I / (m'·L⁴) )

    where m' = total mass / length. Flags fundamentals below 50 Hz for
    vehicles under 3 m (resonance / controllability concern).

    ``boundary`` selects the eigenvalue set: "free-free" (flight, default)
    or "cantilever" (only for comparing against a clamped FE model)."""
    me = ModalEstimate()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0 or L <= 0:
        return me
    r_o = d / 2                 # d = outer diameter
    r_i = d / 2 - t
    I = (math.pi / 4) * (r_o ** 4 - r_i ** 4)
    EI = mat.E * I
    mass = vehicle_mass(state)
    if mass <= 0:
        mass = 5.0
    me.total_mass_kg = mass
    m_per_L = mass / L
    if EI <= 0 or m_per_L <= 0:
        return me
    betas = _BETA_L_BY_BC.get(boundary, _BETA_L)
    freqs = [(bl ** 2 / (2 * math.pi)) * math.sqrt(EI / (m_per_L * L ** 4))
             for bl in betas]
    me.f1_hz, me.f2_hz, me.f3_hz = freqs
    if boundary == "free-free" and L < 3.0 and me.f1_hz < 50.0:
        me.low_freq = True
        me.warning = (f"1st bending mode {me.f1_hz:.0f} Hz is below 50 Hz for a "
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
    present: bool = True       # False: the rocket has no fins to analyse
    # Flutter on the scale every flutter readout shares
    # (dynamics.flutter_analysis): the verdict, and that verdict on the
    # safety-factor scale the status, failure map and score are judged on.
    flutter_verdict: str = "—"
    flutter_sf: float = 0.0

    @property
    def governing_sf(self) -> float:
        """What the fins are judged on: the lower of root bending and flutter."""
        return min(self.safety_factor, self.flutter_sf)

    @classmethod
    def no_fins(cls) -> "FinAnalysis":
        """Result for a finless rocket: nothing to fail, nothing to report."""
        return cls(present=False, safety_factor=float("inf"),
                   flutter_margin=float("inf"), flutter_sf=float("inf"),
                   highest_loaded_fin="—", status="NO FINS")


def fin_root_bending(span: float, root: float, tip: float, thick: float,
                     q: float, aoa_rad: float) -> tuple:
    """Normal force on one fin and the bending stress it puts in the root.

    Finite-aspect-ratio lift slope on the trapezoid planform, load centroid
    at 0.4 span, rectangular root section (root chord x thickness). Returns
    ``(force_N, stress_Pa)``. Shared by fin_analysis and the CalculiX
    solver's no-solver fallback, which used to size a made-up fin from the
    airframe wall.
    """
    if span <= 0 or root <= 0 or thick <= 0 or q <= 0:
        return 0.0, 0.0
    area = 0.5 * (root + tip) * span
    cn_alpha = 2 * math.pi / (1 + 2 / max(span * 2 / (root + tip), 0.5))
    force = q * cn_alpha * aoa_rad * area
    section_modulus = root * thick ** 2 / 6.0
    return force, force * (0.4 * span) / section_modulus


def _fin_material_name(assembly, fallback: str) -> str:
    """Material of the first fin set in the assembly, else the body material.

    Fins are commonly a different material from the airframe (plywood or G10
    fins on a fibreglass tube). The flat RocketState carries only one
    material, so without the assembly the fin analysis silently used the
    BODY's E, G and yield — and the flutter speed it reported then disagreed
    with the Dynamics workspace, which reads fin.material directly."""
    if assembly is None:
        return fallback
    try:
        from core.components import TrapezoidalFinSet
        for comp in assembly.all_components():
            if isinstance(comp, TrapezoidalFinSet):
                name = getattr(comp, "material", None)
                if name:
                    return name
    except Exception as e:
        logger.debug(f"fin material lookup failed: {e}")
    return fallback


def fin_analysis(state, flight: FlightLoads, material_name: str) -> FinAnalysis:
    """Cantilever-plate fin analysis: root bending/shear, tip deflection,
    fundamental frequency, and flutter margin.

    ``material_name`` should be the FIN material (see _fin_material_name),
    not the airframe's, when the two differ.

    The fins analysed are the ones on the state. Missing dimensions used to
    be filled in from the body (root 0.1 L, span 0.6 D, three fins), so a
    finless rocket was reported with a fin stress, a flutter speed and a
    FAILURE status for fins it does not have, and a delta fin (tip chord 0)
    was analysed with a tip of half its root."""
    mat = get_structural_material(material_name)

    root = getattr(state, "fin_root_chord", 0.0) or 0.0
    tip = getattr(state, "fin_tip_chord", 0.0) or 0.0
    span = getattr(state, "fin_span", 0.0) or getattr(state, "fin_height", 0.0) or 0.0
    thick = getattr(state, "fin_thickness", 0.0) or 0.0
    count = getattr(state, "fin_count", None)
    if (count is not None and count <= 0) or span <= 0 or root <= 0:
        return FinAnalysis.no_fins()
    if thick <= 0:
        # No thickness, no section to analyse. It used to be taken as 3 mm.
        fa = FinAnalysis.no_fins()
        fa.status = "NO FIN THICKNESS"
        return fa
    fa = FinAnalysis()

    # ── Aerodynamic normal force per fin at max-Q ──
    q = flight.max_dynamic_pressure
    if q <= 0:
        P, T, rho = isa(flight.maxq_altitude or 3000.0)
        V = (flight.max_mach or 0.6) * speed_of_sound(T)
        q = 0.5 * rho * V ** 2
    aoa = math.radians(5.0)                    # conservative gust AoA
    # ── Root bending stress: load centroid at ~0.4 span, rectangular root ──
    F_fin, sigma_root = fin_root_bending(span, root, tip, thick, q, aoa)
    fa.fin_normal_force_N = F_fin
    fa.root_bending_MPa = sigma_root / 1e6

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

    # ── Flutter velocity ──
    # The same number the Dynamics workspace reports for this fin: the lower
    # of NACA TN-4197 and the p-k solve, at sea level, from one shared
    # function. This tab used the NACA value alone, at the max-Q altitude,
    # and so quoted a higher flutter speed than Dynamics for the same fin
    # (452 against 282 m/s for a 4 mm plywood fin).
    from dynamics.flutter_analysis import (
        flutter_verdict, governing_flutter_speed, margin_safety_factor)
    G_shear = mat.G if mat.G > 0 else mat.E / (2 * (1 + mat.nu))
    v_max = flight.max_speed()
    v_f, _v_naca, _pk = governing_flutter_speed(
        span, root, tip, thick, mat.E, G_shear, mat.density,
        max_flight_speed=v_max if v_max > 0 else 300.0)
    fa.flutter_speed_m_s = v_f if math.isfinite(v_f) else 0.0
    fa.flutter_margin = v_f / v_max if v_max > 0 else float("inf")
    # Judged as the Dynamics workspace judges it. The ratio used to go
    # straight onto the safety-factor scale (SAFE from 1.5x), so a fin 30 %
    # clear of flutter read MARGIN here and SAFE there, and one 5 % clear
    # read WARNING here (structure: PASS WITH MARGIN) and UNSAFE there.
    fa.flutter_verdict, margin_pct = flutter_verdict(v_f, v_max)
    fa.flutter_sf = margin_safety_factor(margin_pct)

    # ── Safety factor: bending vs yield ──
    sigma = fa.root_bending_MPa * 1e6
    fa.safety_factor = mat.yield_strength / sigma if sigma > 0 else float("inf")
    fa.highest_loaded_fin = "Fin 1"   # symmetric set — all equal; gust loads one most
    fa.status, fa.status_color = _status_from_sf(fa.governing_sf)
    return fa


def weakest_fin_analysis(state, assembly, flight: FlightLoads,
                         material_name: str) -> FinAnalysis:
    """fin_analysis of every fin set, each in its own material; the rocket's
    fins are judged on the weakest one (lowest governing_sf).

    The Dynamics workspace takes the lowest flutter speed over all fin sets;
    this tab analysed the first set only, so canards that flutter long before
    the tail fins read SAFE here and UNSAFE there. Without an assembly the
    state's own fin set is analysed."""
    import types
    from core.components import TrapezoidalFinSet
    from core.staging import fin_set_fields

    sets = ([c for c in assembly.all_components() if isinstance(c, TrapezoidalFinSet)]
            if assembly is not None else [])
    if not sets:
        return fin_analysis(state, flight, _fin_material_name(assembly, material_name))
    results = []
    for fins in sets:
        fa = fin_analysis(types.SimpleNamespace(**fin_set_fields(fins)), flight,
                          getattr(fins, "material", None) or material_name)
        if len(sets) > 1:
            fa.highest_loaded_fin = fins.name
        results.append(fa)
    return min(results, key=lambda fa: fa.governing_sf)


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


def buckling_analysis(state, flight: FlightLoads, material_name: str,
                      bending_stress_pa: float = 0.0) -> BucklingAnalysis:
    """Euler column, NASA SP-8007 axial shell, flat-panel, and shell buckling
    under bending (the max-Q governing load). ``bending_stress_pa`` is the
    peak fibre bending stress from the body condition."""
    ba = BucklingAnalysis()
    mat = get_structural_material(material_name)
    d, t, L = state.diameter, state.wall_thickness, state.length
    if d <= 0 or t <= 0:
        return ba
    r = d / 2
    area = math.pi * d * t

    # Applied axial compression: the larger of peak thrust and the drag the
    # flight recorded at max-Q (see physics.structures._compute_max_q). It
    # was peak thrust plus 0.5 x q x frontal area, two loads from different
    # instants added together with an assumed drag coefficient.
    thrust = flight.max_thrust or getattr(state, "thrust", 0.0)
    P_applied = max(thrust, flight.maxq_drag, getattr(state, "weight", 0.0), 1.0)
    sigma_applied = P_applied / area
    ba.applied_axial_N = P_applied

    # 1. Euler column buckling (cantilever on rail, k=2)
    P_euler = euler_buckling(mat.E, d, t, L * 2)
    ba.modes.append(BucklingMode("Euler Column", P_euler, P_applied, "N"))

    # 2. Shell buckling (NASA SP-8007)
    sigma_shell = shell_buckling(mat.E, d, t, mat.nu)
    ba.modes.append(BucklingMode("Shell Buckling", sigma_shell, sigma_applied, "Pa"))

    # 3. Panel buckling — skin panel between fins (flat-plate, simply supported)
    #    A finless rocket has no such panel; it used to be given three fins.
    n_fins = getattr(state, "fin_count", 0) or 0
    if n_fins > 0:
        b_panel = math.pi * d / n_fins               # arc width between fins
        k_plate = 4.0                                # simply-supported uniaxial
        sigma_panel = (k_plate * math.pi ** 2 * mat.E /
                       (12 * (1 - mat.nu ** 2))) * (t / b_panel) ** 2
        ba.modes.append(BucklingMode("Panel Buckling", sigma_panel, sigma_applied, "Pa"))

    # 4. Shell buckling under pure bending (NASA SP-8007 sec. 4.2.2). The old
    #    "Local Crippling" entry was 0.6·E·t/r — the classical axial shell
    #    value with no knockdown, i.e. the same mode as (2) at ×4–5 the
    #    allowable, so it never governed. Bending is the max-Q load, so it
    #    gets its own check: γ_b = 1 − 0.731(1 − e^(−√(r/t)/16)) is less
    #    severe than the axial knockdown, and the applied stress is the
    #    outer-fibre bending stress.
    gamma_b = 1.0 - 0.731 * (1.0 - math.exp(-math.sqrt(r / t) / 16.0)) if t > 0 else 1.0
    sigma_cl = mat.E * t / (r * math.sqrt(3.0 * (1.0 - mat.nu ** 2))) if r > 0 else float("inf")
    sigma_bend_cr = gamma_b * sigma_cl
    ba.modes.append(BucklingMode("Bending Buckling", sigma_bend_cr,
                                 max(bending_stress_pa, 0.0), "Pa"))

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
    recovery_margin: Optional[float] = None   # None: not assessed
    fin_margin: float = 0.0
    thermal_margin: Optional[float] = None    # None: not assessed
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


def safety_score(yield_sf: float, buckling_sf: float, recovery_sf: Optional[float],
                 fin_sf: float, thermal_sf: Optional[float]) -> SafetyScore:
    """Weighted 0-100 structural safety score across the margins that were
    assessed. A margin of None (recovery with no hardware rating, thermal
    with no flight history) is left out and the other weights are scaled up
    to sum to 1; it is not scored as a pass."""
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
        "fin": _sf_to_subscore(fin_sf),
    }
    if recovery_sf is not None:
        subs["recovery"] = _sf_to_subscore(recovery_sf)
    if thermal_sf is not None:
        subs["thermal"] = _sf_to_subscore(thermal_sf)
    ss.subscores = subs
    total_weight = sum(weights[k] for k in subs)
    ss.score = int(round(sum(subs[k] * weights[k] for k in subs) / total_weight))

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
    available: bool = False                # False: no flight history to heat the wall with
    initial_temp_K: float = 0.0            # wall on the pad = air at the pad
    skin_temp_K: float = 0.0               # peak outer-face temperature
    internal_temp_K: float = 0.0           # inner face at that instant
    gradient_K: float = 0.0                # peak outer - inner difference
    expansion_mm: float = 0.0              # growth in length at the peak mean wall temperature
    thermal_stress_MPa: float = 0.0        # peak wall stress
    max_thermal_stress_MPa: float = 0.0
    max_stress_altitude: float = 0.0       # m above the pad
    surface_minus_mean_K: float = 0.0      # outer face above the wall mean, at the peak stress
    stagnation_temp_K: float = 0.0         # air stagnation temperature at peak Mach
    service_limit_K: float = 423.0
    exceeds_limit: bool = False
    heat_flux_W_m2: float = 0.0                    # peak convective wall heat flux
    profile: list = field(default_factory=list)    # [(altitude_m, thermal_stress_MPa)]
    body_station_temps: list = field(default_factory=list)  # [(x_frac, peak T_K, label)]


_WALL_NODES = 12
_thermal_cache: dict = {}      # the last thermal_profile: {"key": ..., "value": ...}


def wall_heating(times, speeds, machs, air_temps, air_densities,
                 x_station: float, thickness: float, mat) -> dict:
    """Temperature through the airframe wall over a flight, at one station.

    One-dimensional conduction through the wall thickness (implicit finite
    differences, ``_WALL_NODES`` nodes): the outer face is heated by the
    boundary layer (Eckert reference-temperature flat plate at ``x_station``
    metres from the nose, structures.thermal_analysis.convective_heat_flux),
    the inner face is insulated, and the wall starts at the air temperature
    of the first sample. Conductivity, specific heat and density are the
    material's.

    Returns arrays over the samples: ``outer``, ``inner``, ``mean`` (K) and
    ``flux`` (W/m^2).
    """
    import numpy as np
    from structures.thermal_analysis import convective_heat_flux

    n = _WALL_NODES
    dx = thickness / (n - 1)
    k, rho_c = mat.thermal_conductivity, mat.density * mat.specific_heat
    T = np.full(n, float(air_temps[0]))
    outer, inner, mean, flux = [T[0]], [T[-1]], [T[0]], [0.0]
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0:
            outer.append(T[0]); inner.append(T[-1]); mean.append(float(T.mean())); flux.append(flux[-1])
            continue
        q = convective_heat_flux(air_temps[i], air_densities[i], speeds[i], machs[i],
                                 x_station, T[0])
        r = k * dt / (rho_c * dx * dx)
        A = np.zeros((n, n))
        b = T.copy()
        for j in range(1, n - 1):
            A[j, j - 1], A[j, j], A[j, j + 1] = -r, 1 + 2 * r, -r
        # Half cells at the faces: flux in at the outer one, none at the inner.
        A[0, 0], A[0, 1] = 1 + 2 * r, -2 * r
        b[0] += 2 * q * dt / (rho_c * dx)
        A[-1, -1], A[-1, -2] = 1 + 2 * r, -2 * r
        T = np.linalg.solve(A, b)
        # Trapezoid mean over the wall (half weight on the faces).
        m = (T[0] / 2 + T[1:-1].sum() + T[-1] / 2) / (n - 1)
        outer.append(float(T[0])); inner.append(float(T[-1])); mean.append(float(m)); flux.append(q)
    return {"outer": np.array(outer), "inner": np.array(inner),
            "mean": np.array(mean), "flux": np.array(flux)}


def thermal_profile(state, flight: FlightLoads, history, material_name: str) -> ThermalProfile:
    """Aerodynamic heating of the airframe wall over the recorded ascent.

    The wall temperature is integrated through the flight (wall_heating) at
    five stations along the body; the hottest is the one nearest the nose,
    where the boundary layer is thinnest. The stress is the one the
    through-wall temperature difference puts in the outer face,
    E * alpha * (T_outer - T_mean) / (1 - nu).

    Needs the flight history. Without one the result is ``available=False``
    and carries no temperatures.

    The skin used to be put at the steady recovery temperature of the peak
    Mach, as if the rocket held that speed until the wall soaked through,
    and the stations along the body were fixed fractions of it (92 %, 95 %,
    85 %). The stress sweep took the air temperature of each instant as the
    wall temperature, so its maximum was at apogee, in the cold.
    """
    tp = ThermalProfile()
    mat = get_structural_material(material_name)
    tp.service_limit_K = mat.max_service_temp
    L, wall = state.length, state.wall_thickness
    if history is None or len(history) < 3 or L <= 0 or wall <= 0:
        return tp

    # The solve takes about 0.2 s and the Structures tab asks on every state
    # change, so the last answer is kept for the same flight, wall and
    # material.
    key = (id(history), len(history), history.get_values("time")[-1],
           history.find_max("mach")[1], history.find_max("altitude")[1],
           history.get_values("atm_temperature")[0], material_name, L, wall)
    if _thermal_cache.get("key") == key:
        return _thermal_cache["value"]
    _thermal_cache.update(key=key, value=tp)

    _, _, iap = history.find_max("altitude")
    step = max(1, (iap + 1) // 400)
    idx = list(range(0, iap + 1, step))
    if len(idx) < 3:
        return tp
    col = lambda name: [history.get_values(name)[i] for i in idx]
    times, speeds, machs, alts = col("time"), col("velocity"), col("mach"), col("altitude")
    air_T, air_rho = col("atm_temperature"), col("atm_density")
    if not air_T[0] or not air_rho[0]:
        return tp

    stations = [(f, wall_heating(times, [abs(v) for v in speeds], machs, air_T, air_rho,
                                 f * L, wall, mat))
                for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    tp.body_station_temps = [(f, float(w["outer"].max()), f"{f * L:.2f} m from nose")
                             for f, w in stations]
    w = stations[0][1]                      # the hottest station
    ipk = int(w["outer"].argmax())
    tp.available = True
    tp.initial_temp_K = float(air_T[0])
    tp.skin_temp_K = float(w["outer"][ipk])
    tp.internal_temp_K = float(w["inner"][ipk])
    tp.gradient_K = float((w["outer"] - w["inner"]).max())
    tp.expansion_mm = mat.cte * (float(w["mean"].max()) - tp.initial_temp_K) * L * 1000.0
    tp.heat_flux_W_m2 = float(w["flux"].max())
    tp.exceeds_limit = tp.skin_temp_K > tp.service_limit_K
    im = max(range(len(machs)), key=lambda i: machs[i])
    tp.stagnation_temp_K = air_T[im] * (1 + 0.2 * machs[im] ** 2)

    stress = mat.E * mat.cte * (w["outer"] - w["mean"]) / (1 - mat.nu)
    tp.profile = [(alts[i], abs(float(stress[i])) / 1e6) for i in range(len(idx))]
    ist = max(range(len(idx)), key=lambda i: abs(stress[i]))
    tp.max_thermal_stress_MPa = tp.thermal_stress_MPa = abs(float(stress[ist])) / 1e6
    tp.max_stress_altitude = alts[ist]
    tp.surface_minus_mean_K = float(w["outer"][ist] - w["mean"][ist])
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
    not_modelled: list = field(default_factory=list)  # subsystem names


# Subsystems this module has no model of. They are named on the map so the
# gap is visible; they carry no safety factor.
NOT_MODELLED = ["Motor Mount", "Couplers", "Bulkheads", "Recovery Hardware",
                "Nose Cone", "Avionics Bay"]


def thermal_margin(thermal: "ThermalProfile") -> Optional[float]:
    """Thermal safety factor = allowable temperature RISE / actual rise of
    the skin above the temperature it left the pad at. None when the heating
    was not solved (no flight history). No rise → ∞.

    The rise was measured from a fixed 20 °C, whatever the day. Before that
    the form was T_limit / T_skin, an absolute-Kelvin ratio, which gave
    SF ≈ 1.35 for an unheated Mach-0.8 skin at 314 K."""
    if not thermal.available:
        return None
    rise = thermal.skin_temp_K - thermal.initial_temp_K
    allow = thermal.service_limit_K - thermal.initial_temp_K
    if rise <= 0.0:
        return float("inf")
    if allow <= 0.0:
        return 0.0
    return allow / rise


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
    """The subsystems that were analysed, each with its governing margin.

    Only the airframe and the fins are. The map used to show eight tiles:
    Couplers at 0.9 x the body safety factor, Avionics Bay at 1.2 x, Nose
    Cone at the larger of body and thermal, Motor Mount as a copy of the
    airframe, and Bulkheads and Recovery System judged against an assumed
    eyebolt. None of those had an analysis behind it."""
    fm = FailureMap(not_modelled=list(NOT_MODELLED))
    rows = [("Airframe",
             min(body_sf, *(m.margin for m in buckling.modes)) if buckling.modes else body_sf,
             "Axial + bending stress, buckling")]
    if fin.present:
        rows.append(("Fins", fin.governing_sf,
                     "Root bending + flutter" if fin.safety_factor <= fin.flutter_sf else
                     f"Flutter governs: margin {(fin.flutter_margin - 1.0) * 100.0:+.0f}% on "
                     f"max flight speed ({fin.flutter_verdict})"))
    for name, sf, detail in rows:
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


def load_path(state, flight: FlightLoads, assembly=None) -> LoadPath:
    """Axial compression at each airframe joint at peak thrust, nose to tail.

    The force through a joint is what accelerates everything forward of it:
    F = thrust x (mass forward of the joint / liftoff mass), drag left out
    (peak thrust is at low speed). The masses and joints are the design's:
    each component the Design tab holds, at its own position, and the motor
    and propellant where the state puts them.

    Without a design there are no joints, and no stations are returned. The
    chart used to be drawn for seven fixed interfaces (Motor Mount, Coupler,
    Recovery Bay ...) carrying 95 %, 80 %, 55 %, 40 %, 25 % and 8 % of the
    thrust at 88 %, 70 % ... of the length, for every rocket."""
    lp = LoadPath()
    thrust = flight.max_thrust or getattr(state, "motor_max_thrust", 0.0) or 0.0
    if thrust <= 0 or assembly is None:
        return lp
    from core.components import Stage

    def counted(c):
        if isinstance(c, Stage):
            return False
        p = getattr(c, "parent", None)
        return not (p is not None and getattr(p, "_override_includes_children", False)
                    and getattr(p, "override_mass", None) is not None)

    masses = []                                   # (centre from nose, kg)
    for c in assembly.all_components():
        if counted(c):
            try:
                m = float(c.computed_mass())
            except Exception:
                m = 0.0
            if m > 0:
                masses.append((float(c.cg_position()), m))
    motor = (getattr(state, "motor_dry_mass", 0.0) or 0.0) + \
        (getattr(state, "propellant_mass_initial", 0.0) or 0.0)
    if motor > 0:
        aft = getattr(state, "motor_position", 0.0) or getattr(state, "length", 0.0) or 0.0
        masses.append((aft - (getattr(state, "motor_length", 0.0) or 0.0) / 2.0, motor))
    total = sum(m for _, m in masses)
    if total <= 0:
        return lp

    # Joints: the aft end of every airframe section but the last.
    sections = sorted((c for stage in assembly.stages for c in stage.children
                       if c.component_length() > 0 and c.outer_diameter() > 0),
                      key=lambda c: c.position)
    for sec in sections[:-1]:
        x = float(sec.position) + float(sec.component_length())
        forward = sum(m for cg, m in masses if cg < x)
        lp.stations.append(LoadStation(f"Aft of {sec.name}", thrust * forward / total, x))
    if sections:
        last = sections[-1]
        lp.stations.append(LoadStation(
            "Motor thrust", thrust, float(last.position) + float(last.component_length())))
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


def body_condition_from_fem(result) -> dict:
    """A static FEMResult in the shape of a compute_for_condition() result.

    full_analysis(), the PDF report, the physics checks and the 3D stress
    view all read the body condition by those keys. The Structures workspace
    built this dict inline with the stress components only, so full_analysis
    raised KeyError('safety_factor') after every CalculiX run and the
    workstation tabs never updated (and before that, the report printed an
    FEM run's safety factor as 0.00).

    The margin follows the workstation convention MoS = SF − 1, as
    compute_for_condition reports it. The FEM result's own margin divides by
    FEMConfig.safety_factor_required (2.0) first, which the physics checks
    would flag as a negative margin at SF > 1.
    """
    vm = result.max_von_mises
    # No stress → no failure. The analytical FEM fallback leaves SF at 0.0 in
    # that case, which would read as the worst possible result.
    sf = result.safety_factor if vm > 0 else float("inf")
    return {
        "axial": result.max_axial_stress, "hoop": result.max_hoop_stress,
        "bending": result.max_bending_stress, "shear": result.max_shear_stress,
        "thermal": result.max_thermal_stress, "von_mises": vm,
        "safety_factor": sf,
        "margin_of_safety": sf - 1.0,
        "yield_utilization": result.yield_utilization,
    }


def full_analysis(state, assembly, history, material_name: str,
                  condition: str = "Max-Q",
                  body_condition: Optional[dict] = None) -> WorkstationReport:
    """Run the complete workstation analysis suite and assemble a report.
    Designed to complete in well under a second.

    ``body_condition`` — an already-computed compute_for_condition() result
    (e.g. the one the workspace shows in its stress panel). When given it is
    the single source for body SF, so the score / failure map / mass
    efficiency agree with the numbers on screen instead of being derived
    from a second, differently-loaded evaluation."""
    rep = WorkstationReport()
    rep.flight = FlightLoads.from_history(history, state)
    rep.recovery = recovery_loads(state, history)

    # Governing body stress for the selected condition
    mass = vehicle_mass(state)
    if body_condition:
        rep.body_condition = body_condition
    else:
        fl = rep.flight
        extra = {}
        if condition == "Recovery Shock":
            force = rep.recovery.harness_tension_N
        elif condition == "Max-Q" and fl.source == "simulation":
            # The thrust, drag and dynamic pressure of the max-Q instant, as
            # the worst-case table uses. Peak thrust (an earlier instant)
            # was paired with max-Q here: 7.5 MPa on this panel against
            # 5.3 MPa in the table for the same event.
            force = fl.maxq_thrust
            extra = dict(drag_force_N=fl.maxq_drag, q_dyn=fl.max_dynamic_pressure)
        else:
            force = max(fl.max_thrust, getattr(state, "thrust", 0.0), mass * G, 1.0)
        if condition == "Thermal":
            extra = dict(wall_delta_T=thermal_profile(
                state, fl, history, material_name).surface_minus_mean_K)
        rep.body_condition = compute_for_condition(
            condition, state.diameter, state.wall_thickness, state.length,
            material_name, force=force, mach=fl.maxq_mach,
            altitude_m=fl.maxq_altitude, vehicle_mass_kg=mass,
            angle_of_attack_deg=3.0 if condition == "Max-Q" else 2.0,
            moment_arm_m=fl.moment_arm, **extra,
        )
    body_sf = rep.body_condition["safety_factor"]

    rep.fin = weakest_fin_analysis(state, assembly, rep.flight, material_name)
    rep.buckling = buckling_analysis(state, rep.flight, material_name,
                                     bending_stress_pa=rep.body_condition.get("bending", 0.0))
    rep.thermal = thermal_profile(state, rep.flight, history, material_name)
    rep.mass = mass_efficiency(state, assembly,
                               min(body_sf, rep.buckling.governing.margin
                                   if rep.buckling.governing else body_sf))

    thermal_sf = thermal_margin(rep.thermal)
    buck_sf = rep.buckling.governing.margin if rep.buckling.governing else 99.0
    rep.score = safety_score(body_sf, buck_sf, rep.recovery.safety_factor,
                             rep.fin.governing_sf, thermal_sf)
    rep.failure = failure_map(state, body_sf, rep.fin, rep.recovery,
                              rep.buckling, rep.thermal)
    rep.loads = load_path(state, rep.flight, assembly)
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
          f"{rep.recovery.status}")
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
