"""
Turn a RocketState (and optional assembly) into a compact, model-readable
summary. Curated, not a dump: the model should never see 200 fields, and it
should never do the physics itself — K2's numbers are the ground truth.
"""

from __future__ import annotations

import json
import math


def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def liftoff_mass(state) -> float:
    """Pad mass: airframe + motor casing + full propellant load.

    The casing (``motor_dry_mass``) rides along the whole flight and on a
    high-power motor weighs about as much as the propellant (median 0.95× for
    the catalog's L motors). Leaving it out overstated thrust-to-weight by 14%
    for an L1090W on a 6 kg airframe.
    """
    return state.liftoff_mass()


def zero_lift_cd(state):
    """The rocket's drag coefficient at zero incidence and Mach 0.3, from the
    aero model the flight uses; None when no airframe is drawn.

    The assistant was given ``state.cd`` as "cd0". That is 0.45 for every
    rocket, whatever its nose and fins (the field's default; nothing sets it
    from the geometry), and a flight used to leave its last instant's value
    there: 1.57 under the canopy for the canonical rocket.
    """
    if not ((state.length or 0.0) > 0 and (state.diameter or 0.0) > 0):
        return None
    try:
        from physics.aerodynamics import AeroModel
        return AeroModel.from_state(state).zero_lift_cd(0.3)
    except Exception:
        return None


def _stability_verdict(margin_cal: float) -> str:
    """The verdict the Design tab shows (core.stability), in words for the
    model. It had its own marks: over-stable only above 3 cal, where the
    screen says OVERSTABLE from 2.5."""
    from core import stability as S
    return {
        S.UNSTABLE: "UNSTABLE (CP ahead of CG)",
        S.MARGINAL: f"marginal (<{S.MARGINAL_BELOW:g} cal)",
        S.STABLE: "stable",
        S.OVERSTABLE: f"over-stable (>{S.OVERSTABLE_ABOVE:g} cal, weathercocking risk)",
    }.get(S.stability_verdict(margin_cal), "unknown")


def summarize_state(state, assembly=None, flight=None) -> dict:
    """Structured summary; ``render_context`` turns it into prompt text.
    ``flight`` is the last flight's ``FlightSummary`` when there is one."""
    d = state.diameter or 0.0
    # The design's balance is the rocket on the pad. state.cg and
    # state.stability_margin are live: after a flight they are the burnt-out
    # rocket's, and the assistant was told the canonical rocket had 6.78 cal
    # (5.30 on the pad).
    margin = _f(state.pad_stability_margin(), 2)
    geom = {
        "name": state.name,
        "length_m": _f(state.length),
        "diameter_m": _f(state.diameter),
        "nose": {"type": state.nose_type, "length_m": _f(state.nose_length)},
        "fins": {
            "count": state.fin_count,
            "root_chord_m": _f(state.fin_root_chord),
            "tip_chord_m": _f(state.fin_tip_chord),
            "span_m": _f(state.fin_span or state.fin_height),
            "sweep_deg": _f(math.degrees(state.fin_sweep_angle), 1)
            if abs(state.fin_sweep_angle) < 2 * math.pi else _f(state.fin_sweep_angle, 1),
            "thickness_m": _f(state.fin_thickness, 4),
            "cross_section": state.fin_cross_section,
            "position_from_nose_m": _f(state.fin_position),
        },
        "surface_finish": state.surface_finish,
    }
    mass = {
        "dry_mass_kg": _f(state.dry_mass),
        "propellant_mass_kg": _f(state.propellant_mass_initial or state.propellant_mass),
        "liftoff_mass_kg": _f(liftoff_mass(state)),
        "cg_from_nose_m": _f(state.liftoff_cg()),
        "cp_from_nose_m": _f(state.cp),
        "stability_margin_cal": margin,
        "stability_verdict": _stability_verdict(margin),
        "cd0": _f(zero_lift_cd(state)),
    }
    motor = {
        "designation": state.motor_designation,
        "avg_thrust_N": _f(state.motor_avg_thrust, 1),
        "max_thrust_N": _f(state.motor_max_thrust, 1),
        "total_impulse_Ns": _f(state.motor_total_impulse, 1),
        "burn_time_s": _f(state.motor_burn_time, 2),
        "isp_s": _f(state.motor_isp, 1),
    }
    liftoff = liftoff_mass(state)
    if liftoff > 0 and state.motor_avg_thrust:
        motor["thrust_to_weight"] = _f(state.motor_avg_thrust / (liftoff * 9.80665), 2)

    launch = {
        "launch_angle_deg": _f(state.launch_angle, 1),
        "rail_azimuth_deg": _f(state.launch_azimuth, 0),
        "rod_length_m": _f(state.launch_rod_length, 2),
        "site_elevation_m": _f(state.launch_altitude, 0),
        "wind_speed_m_s": _f(state.wind_speed, 1),
        "wind_from_deg": _f(state.wind_direction, 0),
        # 0 on the state means a standard day at the site elevation.
        "ground_temp_K": (_f(state.ground_temperature, 1)
                          if state.ground_temperature > 0 else "ISA"),
    }
    recovery = {
        "drogue_cd_area_m2": _f(state.drogue_cd_area, 2),
        "main_cd_area_m2": _f(state.main_cd_area, 2),
        "main_deploy_alt_m": _f(state.main_deploy_altitude, 0),
        "drogue_delay_s": _f(state.drogue_deploy_delay, 1),
    }
    structure = {
        "material": state.material_name,
        "wall_thickness_m": _f(state.wall_thickness, 4),
        "safety_factor": _f(state.safety_factor, 2) if state.safety_factor else None,
        "max_stress_MPa": _f(state.max_stress / 1e6, 1) if state.max_stress else None,
        "flutter_speed_m_s": _f(state.flutter_speed, 1) if state.flutter_speed else None,
        "flutter_exceeded_in_flight": bool(state.flutter_exceeded),
    }

    out = {"geometry": geom, "mass_and_stability": mass, "motor": motor,
           "launch_conditions": launch, "recovery": recovery, "structure": structure}

    if state.stages_config:
        out["multistage"] = {"stage_count": len(state.stages_config)}

    if state.max_altitude or state.sim_phase not in ("Pre-Launch", "", None):
        out["last_flight"] = last_flight_summary(state, flight)

    if state.cfd_converged or state.cfd_cd:
        out["cfd"] = {"converged": bool(state.cfd_converged), "mach": _f(state.cfd_mach, 2),
                      "cd": _f(state.cfd_cd, 3), "cl": _f(state.cfd_cl, 3),
                      "cp_from_nose_m": _f(state.cfd_cp_location, 3)}

    if assembly is not None:
        comps = _assembly_components(assembly)
        if comps:
            out["components"] = comps
    return out


def last_flight_summary(state, flight=None) -> dict:
    """The last flight as the model is told it. ``flight`` is its
    ``FlightSummary`` (``SimulationEngine.summary``).

    Peak dynamic pressure and the rail-exit numbers come only from
    ``flight``; without it they are left out. "max_dynamic_pressure_Pa" used
    to be ``state.dynamic_pressure``, which is the pressure at this instant:
    0 Pa once the rocket has landed, so every finished flight was reported
    with a max-Q of zero.
    """
    out = {
        "phase_at_end": state.sim_phase,
        "sim_time_s": _f(state.sim_time, 1),
        "apogee_m": _f(state.max_altitude, 1),
        "max_velocity_m_s": _f(state.max_velocity, 1),
        "max_mach": _f(state.max_mach, 2),
        "max_acceleration_m_s2": _f(state.max_acceleration, 1),
        "drogue_descent_rate_m_s": _f(state.drogue_descent_rate, 1),
        "main_descent_rate_m_s": _f(state.main_descent_rate, 1),
        "landing_drift_m": _f(state.landing_drift, 0),
        "descent_time_s": _f(state.descent_time, 0),
        "recovery_shock_N": _f(state.recovery_shock_force, 0),
        "parachute_deployed": bool(state.parachute_deployed),
        "flutter_exceeded": bool(state.flutter_exceeded),
    }
    if flight is not None:
        # The sensed peak, as the Results tab and the report give it.
        # state.max_acceleration is the peak of d|v|/dt, about 1 g less on
        # the way up (185.9 against 195.8 m/s² for the canonical rocket).
        if flight.max_acceleration is not None:
            out["max_acceleration_m_s2"] = _f(flight.max_acceleration, 1)
            out["max_acceleration_g"] = _f(flight.max_acceleration / 9.80665, 2)
        out["max_dynamic_pressure_Pa"] = _f(flight.max_q, 0)
        if flight.rail_exit_velocity is not None:
            out["rail_exit_velocity_m_s"] = _f(flight.rail_exit_velocity, 1)
        if flight.stability_at_rail_exit is not None:
            out["stability_at_rail_exit_cal"] = _f(flight.stability_at_rail_exit, 2)
    return out


def _assembly_components(assembly, limit: int = 40) -> list[dict]:
    """Flat list of (type, name, mass, length, position) from the UI assembly tree."""
    rows = []

    def walk(node, depth=0):
        if len(rows) >= limit:
            return
        row = {"type": type(node).__name__, "name": getattr(node, "name", "")}
        for attr, key in (("mass", "mass_kg"), ("length", "length_m"),
                          ("position", "position_m"), ("material", "material")):
            v = getattr(node, attr, None)
            if v is None:
                continue
            row[key] = v if isinstance(v, str) else _f(v)
        if depth:
            row["depth"] = depth
        rows.append(row)
        for child in getattr(node, "children", None) or []:
            walk(child, depth + 1)

    root = getattr(assembly, "root", None) or assembly
    try:
        walk(root)
    except Exception:
        return []
    return rows


def render_context(summary: dict) -> str:
    return "CURRENT ROCKET (computed by K2 — authoritative):\n" + json.dumps(
        summary, indent=1, ensure_ascii=False)
