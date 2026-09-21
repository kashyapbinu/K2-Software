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


def _stability_verdict(margin_cal: float) -> str:
    if margin_cal is None:
        return "unknown"
    if margin_cal < 0:
        return "UNSTABLE (CP ahead of CG)"
    if margin_cal < 1.0:
        return "marginal (<1 cal)"
    if margin_cal > 3.0:
        return "over-stable (>3 cal, weathercocking risk)"
    return "stable"


def summarize_state(state, assembly=None) -> dict:
    """Structured summary; ``render_context`` turns it into prompt text."""
    d = state.diameter or 0.0
    margin = _f(state.stability_margin, 2)
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
        "liftoff_mass_kg": _f((state.dry_mass or 0) + (state.propellant_mass_initial or state.propellant_mass or 0)),
        "cg_from_nose_m": _f(state.cg),
        "cp_from_nose_m": _f(state.cp),
        "stability_margin_cal": margin,
        "stability_verdict": _stability_verdict(margin),
        "cd0": _f(state.cd),
    }
    motor = {
        "designation": state.motor_designation,
        "avg_thrust_N": _f(state.motor_avg_thrust, 1),
        "max_thrust_N": _f(state.motor_max_thrust, 1),
        "total_impulse_Ns": _f(state.motor_total_impulse, 1),
        "burn_time_s": _f(state.motor_burn_time, 2),
        "isp_s": _f(state.motor_isp, 1),
    }
    liftoff = (state.dry_mass or 0) + (state.propellant_mass_initial or 0)
    if liftoff > 0 and state.motor_avg_thrust:
        motor["thrust_to_weight"] = _f(state.motor_avg_thrust / (liftoff * 9.80665), 2)

    launch = {
        "launch_angle_deg": _f(state.launch_angle, 1),
        "rod_length_m": _f(state.launch_rod_length, 2),
        "wind_speed_m_s": _f(state.wind_speed, 1),
        "wind_from_deg": _f(state.wind_direction, 0),
        "ground_temp_K": _f(state.ground_temperature, 1),
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
        out["last_flight"] = last_flight_summary(state)

    if state.cfd_converged or state.cfd_cd:
        out["cfd"] = {"converged": bool(state.cfd_converged), "mach": _f(state.cfd_mach, 2),
                      "cd": _f(state.cfd_cd, 3), "cl": _f(state.cfd_cl, 3),
                      "cp_from_nose_m": _f(state.cfd_cp_location, 3)}

    if assembly is not None:
        comps = _assembly_components(assembly)
        if comps:
            out["components"] = comps
    return out


def last_flight_summary(state) -> dict:
    return {
        "phase_at_end": state.sim_phase,
        "sim_time_s": _f(state.sim_time, 1),
        "apogee_m": _f(state.max_altitude, 1),
        "max_velocity_m_s": _f(state.max_velocity, 1),
        "max_mach": _f(state.max_mach, 2),
        "max_acceleration_m_s2": _f(state.max_acceleration, 1),
        "max_dynamic_pressure_Pa": _f(state.dynamic_pressure, 0) if state.dynamic_pressure else None,
        "drogue_descent_rate_m_s": _f(state.drogue_descent_rate, 1),
        "main_descent_rate_m_s": _f(state.main_descent_rate, 1),
        "landing_drift_m": _f(state.landing_drift, 0),
        "descent_time_s": _f(state.descent_time, 0),
        "recovery_shock_N": _f(state.recovery_shock_force, 0),
        "parachute_deployed": bool(state.parachute_deployed),
        "flutter_exceeded": bool(state.flutter_exceeded),
    }


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
