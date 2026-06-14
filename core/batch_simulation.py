"""
K2 AeroSim — Headless Batch Simulation Runner
=================================================
Thread-safe, non-Qt 6DOF trajectory simulator for Monte Carlo analysis.

Replicates the physics from ``SimulationEngine._derivatives`` and ``_step``
but runs in a simple ``while``-loop with a per-run ``numpy.random.Generator``
instead of QTimer and global ``np.random`` state.  No PyQt6 imports — safe
for ``concurrent.futures`` / ``multiprocessing`` pools.

Usage::

    from core.batch_simulation import BatchSimConfig, run_batch_simulation

    cfg = BatchSimConfig.from_rocket_state(state_engine.state)
    result = run_batch_simulation(cfg, seed=42)
    print(result.apogee, result.max_velocity)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from core.constants import G_EARTH, gravity_at_altitude
from environment.atmosphere_model import Atmosphere
from core.flight_phases import FlightPhase, PhaseManager
from core.integrators import get_integrator
from physics.aerodynamics import AeroModel
from environment.wind_model import WindModel, MultiLevelWindModel

logger = logging.getLogger("K2.BatchSim")


# ══════════════════════════════════════════════════════════════════════════════
#  BatchSimConfig — all inputs for a single simulation run
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchSimConfig:
    """Complete parameter set for one headless trajectory simulation."""

    # ── Geometry ──────────────────────────────────────────────────
    length: float = 0.0
    diameter: float = 0.0
    nose_length: float = 0.0
    fin_height: float = 0.0
    fin_root_chord: float = 0.0
    fin_tip_chord: float = 0.0
    fin_count: int = 4
    nose_type: str = "ogive"
    fin_span: float = 0.0
    fin_sweep_angle: float = 0.0
    fin_thickness: float = 0.003
    fin_position: float = 0.0
    surface_finish: str = "Normal"
    fin_cross_section: str = "Rounded"

    # ── Mass ──────────────────────────────────────────────────────
    dry_mass: float = 0.0
    propellant_mass: float = 0.0
    motor_dry_mass: float = 0.0

    # ── Stability / CG / CP ───────────────────────────────────────
    cg: float = 0.0
    dry_cg: float = 0.0
    cp: float = 0.0
    cd: float = 0.45
    motor_position: float = 0.0
    motor_length: float = 0.0

    # ── Motor ─────────────────────────────────────────────────────
    motor_designation: str = "None"
    motor_avg_thrust: float = 0.0
    motor_max_thrust: float = 0.0
    motor_total_impulse: float = 0.0
    motor_burn_time: float = 0.0
    motor_isp: float = 0.0
    custom_thrust_curve: list = field(default_factory=list)

    # ── Environment ───────────────────────────────────────────────
    launch_angle: float = 90.0
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    wind_gust_intensity: float = 0.0
    wind_mode: str = "average"            # "average" | "multi_level"
    wind_layers: list = field(default_factory=list)  # [(alt_m, speed_m_s, dir_deg)]

    # ── Recovery ──────────────────────────────────────────────────
    drogue_deploy_delay: float = 1.0
    main_deploy_altitude: float = 300.0
    drogue_cd_area: float = 0.5
    main_cd_area: float = 3.0

    # ── Simulation ────────────────────────────────────────────────
    sim_dt: float = 0.01
    integrator_name: str = "rk4"

    # ── Factory ───────────────────────────────────────────────────

    @classmethod
    def from_rocket_state(cls, state) -> "BatchSimConfig":
        """
        Build a BatchSimConfig from a RocketState (or any object exposing
        the same attribute names).
        """
        return cls(
            # Geometry
            length=state.length,
            diameter=state.diameter,
            nose_length=getattr(state, "nose_length", 0.0),
            fin_height=getattr(state, "fin_height", 0.0),
            fin_root_chord=getattr(state, "fin_root_chord", 0.0),
            fin_tip_chord=getattr(state, "fin_tip_chord", 0.0),
            fin_count=getattr(state, "fin_count", 4),
            nose_type=getattr(state, "nose_type", "ogive"),
            fin_span=getattr(state, "fin_span", 0.0),
            fin_sweep_angle=getattr(state, "fin_sweep_angle", 0.0),
            fin_thickness=getattr(state, "fin_thickness", 0.003),
            fin_position=getattr(state, "fin_position", 0.0),
            surface_finish=getattr(state, "surface_finish", "Normal"),
            fin_cross_section=getattr(state, "fin_cross_section", "Rounded"),
            # Mass
            dry_mass=state.dry_mass,
            propellant_mass=getattr(state, "propellant_mass_initial",
                                    getattr(state, "propellant_mass", 0.0)),
            motor_dry_mass=getattr(state, "motor_dry_mass", 0.0),
            # Stability
            cg=state.cg,
            dry_cg=getattr(state, "dry_cg", 0.0),
            cp=state.cp,
            cd=getattr(state, "cd", 0.45),
            motor_position=getattr(state, "motor_position", 0.0),
            motor_length=getattr(state, "motor_length", 0.0),
            # Motor
            motor_designation=getattr(state, "motor_designation", "None"),
            motor_avg_thrust=getattr(state, "motor_avg_thrust", 0.0),
            motor_max_thrust=getattr(state, "motor_max_thrust", 0.0),
            motor_total_impulse=getattr(state, "motor_total_impulse", 0.0),
            motor_burn_time=getattr(state, "motor_burn_time", 0.0),
            motor_isp=getattr(state, "motor_isp", 0.0),
            custom_thrust_curve=list(getattr(state, "custom_thrust_curve", [])),
            # Environment
            launch_angle=getattr(state, "launch_angle", 90.0),
            wind_speed=getattr(state, "wind_speed", 0.0),
            wind_direction=getattr(state, "wind_direction", 0.0),
            wind_gust_intensity=getattr(state, "wind_gust_intensity", 0.0),
            wind_mode=getattr(state, "wind_mode", "average"),
            wind_layers=[tuple(l) for l in getattr(state, "wind_layers", [])],
            # Recovery
            drogue_deploy_delay=getattr(state, "drogue_deploy_delay", 1.0),
            main_deploy_altitude=getattr(state, "main_deploy_altitude", 300.0),
            drogue_cd_area=getattr(state, "drogue_cd_area", 0.5),
            main_cd_area=getattr(state, "main_cd_area", 3.0),
            # Simulation
            sim_dt=getattr(state, "sim_dt", 0.01),
            integrator_name=getattr(state, "integrator_name", "rk4"),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BatchSimResult — all outputs from one simulation run
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchSimResult:
    """Summary statistics produced by a single batch simulation."""

    apogee: float = 0.0
    max_velocity: float = 0.0
    max_mach: float = 0.0
    max_acceleration: float = 0.0

    landing_x: float = 0.0
    landing_y: float = 0.0
    landing_distance: float = 0.0

    flight_time: float = 0.0
    rail_exit_velocity: float = 0.0

    min_stability_margin: float = float("inf")
    max_dynamic_pressure: float = 0.0

    final_phase: str = ""
    success: bool = True
    failure_reasons: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_thrust_curve(cfg: BatchSimConfig) -> List[tuple]:
    """Build a simplified trapezoidal thrust curve from motor params,
    or return the custom curve verbatim."""
    if cfg.custom_thrust_curve:
        return [(t, v) for t, v in cfg.custom_thrust_curve]

    bt = cfg.motor_burn_time
    avg_t = cfg.motor_avg_thrust
    max_t = cfg.motor_max_thrust if cfg.motor_max_thrust > 0 else avg_t * 1.4

    if bt <= 0:
        return []

    # Rated impulse the curve must integrate to (so apogee isn't biased high).
    total_impulse = cfg.motor_total_impulse if cfg.motor_total_impulse > 0 else avg_t * bt

    # Trapezoid (0 → peak → plateau → 0) with 10% ramp/tail. Solve the plateau
    # so the area equals total_impulse exactly:
    #   area = 0.5·(1-ramp_frac)·bt·(max_t + plateau) = total_impulse
    # The old code hard-coded plateau = avg_t, which over-delivered ~3.5%.
    ramp_frac = 0.1
    ramp = bt * ramp_frac
    plateau = 2.0 * total_impulse / ((1.0 - ramp_frac) * bt) - max_t

    if plateau < 0.0:
        # Peak alone exceeds the impulse budget — fall back to a triangle whose
        # area (0.5·max·bt) matches the rated impulse.
        max_t = 2.0 * total_impulse / bt
        return [(0.0, 0.0), (ramp, max_t), (bt, 0.0)]

    return [
        (0.0, 0.0),
        (ramp, max_t),
        (bt - ramp, plateau),
        (bt, 0.0),
    ]


def _get_thrust(t: float, curve: List[tuple]) -> float:
    """Interpolate thrust at time *t* from a piecewise-linear curve."""
    if not curve or t < 0:
        return 0.0
    if t >= curve[-1][0]:
        return 0.0
    for i in range(len(curve) - 1):
        t0, v0 = curve[i]
        t1, v1 = curve[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            return v0 + frac * (v1 - v0)
    return 0.0


def _estimate_inertia(mass: float, length: float) -> float:
    """Fallback pitch inertia estimate: thin rod Iyy = m·L²/12."""
    L = length if length > 0 else 2.0
    return mass * L ** 2 / 12.0


class _StateProxy:
    """Lightweight adapter so ``AeroModel.from_state()`` can consume a
    ``BatchSimConfig``.  Only the geometry / fin attributes are needed."""

    __slots__ = (
        "length", "diameter", "nose_length", "nose_type",
        "fin_height", "fin_root_chord", "fin_tip_chord", "fin_count",
        "fin_span", "fin_sweep_angle", "fin_thickness", "fin_position",
        "surface_finish", "fin_cross_section",
    )

    def __init__(self, cfg: BatchSimConfig):
        self.length = cfg.length
        self.diameter = cfg.diameter
        self.nose_length = cfg.nose_length
        self.nose_type = cfg.nose_type
        self.fin_height = cfg.fin_height
        self.fin_root_chord = cfg.fin_root_chord
        self.fin_tip_chord = cfg.fin_tip_chord
        self.fin_count = cfg.fin_count
        self.fin_span = cfg.fin_span
        self.fin_sweep_angle = cfg.fin_sweep_angle
        self.fin_thickness = cfg.fin_thickness
        self.fin_position = cfg.fin_position
        self.surface_finish = cfg.surface_finish
        self.fin_cross_section = cfg.fin_cross_section


# ══════════════════════════════════════════════════════════════════════════════
#  run_batch_simulation — main entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_batch_simulation(
    config: BatchSimConfig,
    seed: Optional[int] = None,
) -> BatchSimResult:
    """
    Execute one full 6DOF trajectory simulation headlessly.

    Parameters
    ----------
    config : BatchSimConfig
        All rocket, motor, environment, and simulation parameters.
    seed : int, optional
        RNG seed for reproducibility.  Each call creates its own
        ``numpy.random.Generator`` so concurrent runs are thread-safe.

    Returns
    -------
    BatchSimResult
        Summary statistics (apogee, max-V, landing point, …).
    """
    # ── Thread-local RNG ──────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # ── Core modules (one per run — no shared state) ──────────────
    atmosphere = Atmosphere()
    phase_mgr = PhaseManager()
    phase_mgr.reset()
    integrator = get_integrator(config.integrator_name, nan_guard=False)

    # ── Aero model ────────────────────────────────────────────────
    proxy = _StateProxy(config)
    try:
        aero_model: Optional[AeroModel] = AeroModel.from_state(proxy)
    except Exception:
        aero_model = None

    # ── Cd scale factor for MC perturbation ───────────────────────
    # AeroModel computes Cd from geometry (skin friction, base drag, etc.)
    # and ignores config.cd. To propagate Cd perturbations from Monte Carlo,
    # we compute the ratio config.cd / aero_baseline_cd and scale all
    # drag forces by this factor.
    cd_scale = 1.0
    if aero_model is not None and config.cd > 0.01:
        try:
            # Get baseline Cd at Mach 0.3 (representative subsonic flight)
            aero_baseline = aero_model.compute(0.0, 0.3, 1000.0, 0.0, 100.0, config.cg)
            aero_cd0 = aero_baseline.get("cd", 0)
            if aero_cd0 > 0.01:
                cd_scale = config.cd / aero_cd0
        except Exception:
            cd_scale = 1.0

    # ── Wind (proper pink-noise model, seeded for reproducibility) ─
    wind_seed = int(rng.integers(0, 2**31)) if seed is not None else None
    if config.wind_mode == "multi_level" and config.wind_layers:
        wind_model = MultiLevelWindModel(
            config.wind_layers,
            turbulence_intensity=config.wind_gust_intensity,
            seed=wind_seed,
        )
    else:
        wind_model = WindModel(
            base_speed=config.wind_speed,
            direction=config.wind_direction,
            gust_intensity=config.wind_gust_intensity,
            seed=wind_seed,
        )

    # ── Thrust curve ──────────────────────────────────────────────
    thrust_curve = _build_thrust_curve(config)

    # ── Initial 6DOF state vector ─────────────────────────────────
    launch_pitch = math.radians(config.launch_angle)
    total_mass = config.dry_mass + config.motor_dry_mass + config.propellant_mass

    # [x, y, z, vx, vy, vz, pitch, yaw, roll,
    #  pitch_rate, yaw_rate, roll_rate, mass]
    state_vec: List[float] = [
        0.0,            # 0  x
        0.0,            # 1  y (lateral)
        0.0,            # 2  z (altitude)
        0.0,            # 3  vx
        0.0,            # 4  vy
        0.0,            # 5  vz
        launch_pitch,   # 6  pitch
        0.0,            # 7  yaw
        0.0,            # 8  roll
        0.0,            # 9  pitch_rate
        0.0,            # 10 yaw_rate
        0.0,            # 11 roll_rate
        total_mass,     # 12 mass
    ]

    # ── Tracking state ────────────────────────────────────────────
    phase = FlightPhase.PRELAUNCH
    drogue_deployed = False
    main_deployed = False
    prev_dt = config.sim_dt
    initial_prop_mass = config.propellant_mass
    rail_exit_detected = False
    # Peak force-based acceleration (F/m, excluding gravity). Tracked as a
    # running max across ALL derivative evaluations — RK4 calls _derivatives
    # 4× per step, so capturing only the last stage missed the true peak.
    _peak_force_accel = 0.0

    result = BatchSimResult()

    # ── Cached config scalars ─────────────────────────────────────
    dt_base = config.sim_dt
    diameter = config.diameter
    body_length = config.length
    dry_mass = config.dry_mass
    burnout_mass = config.dry_mass + config.motor_dry_mass
    ref_area = math.pi * (diameter / 2.0) ** 2

    # ── Dynamic CG helper (mirrors RocketStateEngine._recompute_derived) ──

    def _compute_cg(current_mass: float) -> float:
        """Recompute CG as propellant burns away (motor case + prop act at motor CG)."""
        motor_m = max(0.0, current_mass - dry_mass)
        motor_cg = max(0.0, config.motor_position - 0.5 * config.motor_length)
        if current_mass > 0:
            return (dry_mass * config.dry_cg + motor_m * motor_cg) / current_mass
        return config.cg

    # ── Derivatives function (closed over local state) ────────────

    def _derivatives(t_: float, sv: list) -> list:
        """6DOF equations of motion — mirrors SimulationEngine._derivatives."""

        x, y, z, vx, vy, vz, pitch, yaw, roll, \
            pitch_rate_, yaw_rate_, roll_rate_, mass = sv

        z = max(0.0, z)
        mass = max(0.01, mass)

        # Rail constraint
        on_rail = z < body_length
        if on_rail or phase == FlightPhase.PRELAUNCH:
            pitch = launch_pitch
            yaw = 0.0
            pitch_rate_ = yaw_rate_ = roll_rate_ = 0.0

        # Atmosphere
        rho = atmosphere.density(z)
        a_sound = atmosphere.speed_of_sound(z)

        # Altitude-dependent gravity
        g = gravity_at_altitude(z)

        # Wind (uses the proper WindModel with pink-noise turbulence)
        wind_vx, wind_vy, wind_vz = wind_model.get_wind_velocity(z, t_)

        # Wind-relative velocity
        vrel_x = vx - wind_vx
        vrel_y = vy - wind_vy
        vrel_z = vz - wind_vz
        v_rel = math.sqrt(vrel_x**2 + vrel_y**2 + vrel_z**2)
        mach = v_rel / a_sound if a_sound > 0 else 0.0

        # Angle of attack (in pitch plane)
        if v_rel > 0.5:
            vel_angle = math.atan2(vrel_z, math.sqrt(vrel_x**2 + vrel_y**2))
        else:
            vel_angle = pitch
        alpha = pitch - vel_angle
        alpha = max(-math.radians(45), min(math.radians(45), alpha))

        # Sideslip (yaw plane)
        if v_rel > 0.5:
            beta_angle = math.atan2(vrel_y, math.sqrt(vrel_x**2 + vrel_z**2))
        else:
            beta_angle = 0.0

        # Dynamic pressure
        q_dyn = 0.5 * rho * v_rel**2

        # Current CG (recomputed as propellant burns)
        cg = _compute_cg(mass)

        # Aerodynamic model
        if aero_model is not None:
            aero = aero_model.compute(alpha, mach, q_dyn, pitch_rate_, v_rel, cg)
            F_drag = aero["F_drag"] * cd_scale  # Apply MC Cd perturbation
            F_normal = aero["F_normal"]
            M_pitch = aero["M_pitch"]
            # Yaw moment (symmetric to pitch for axisymmetric rocket)
            M_yaw = -aero.get("cm", 0) * q_dyn * ref_area * diameter * math.sin(beta_angle)
        else:
            cd_val = config.cd
            # Basic transonic/supersonic drag rise when aero_model unavailable
            if 0.8 < mach < 1.2:
                cd_val *= (1.0 + 2.0 * (mach - 0.8))  # Transonic drag rise
            elif mach >= 1.2:
                cd_val *= (1.0 + 0.8 + 0.2 / max(mach - 0.8, 0.01))  # Supersonic
            F_drag = q_dyn * ref_area * cd_val
            F_normal = 0.0
            M_pitch = 0.0
            M_yaw = 0.0

        # Recovery drag override
        if drogue_deployed and not main_deployed and vz < 0:
            F_drag = 0.5 * rho * v_rel**2 * config.drogue_cd_area
            F_normal = M_pitch = M_yaw = 0.0
        elif main_deployed and vz < 0:
            F_drag = 0.5 * rho * v_rel**2 * config.main_cd_area
            F_normal = M_pitch = M_yaw = 0.0

        # Roll damping (simple model)
        M_roll = -0.01 * roll_rate_ * q_dyn * ref_area * diameter if v_rel > 1.0 else 0.0

        # Random perturbation to prevent over-perfect flight (OpenRocket technique)
        # CRITICAL: uses thread-local rng, NOT np.random globally
        M_pitch += rng.normal(0, 0.0005) * q_dyn * ref_area * diameter
        M_yaw += rng.normal(0, 0.0005) * q_dyn * ref_area * diameter

        # Thrust
        thrust = _get_thrust(t_, thrust_curve)
        tx = thrust * math.cos(pitch) * math.cos(yaw)
        ty = thrust * math.cos(pitch) * math.sin(yaw)
        tz = thrust * math.sin(pitch)

        # Drag opposes relative airflow
        if v_rel > 0:
            drag_x = -F_drag * vrel_x / v_rel
            drag_y = -F_drag * vrel_y / v_rel
            drag_z = -F_drag * vrel_z / v_rel
        else:
            drag_x = drag_y = drag_z = 0.0

        # Normal force in pitch plane
        if v_rel > 0.5:
            perp_angle = vel_angle + math.pi / 2
            normal_x = F_normal * math.cos(perp_angle) * math.copysign(1, alpha)
            normal_z = F_normal * math.sin(perp_angle) * math.copysign(1, alpha)
        else:
            normal_x = normal_z = 0.0
        normal_y = 0.0

        weight = mass * g
        ax = (tx + drag_x + normal_x) / mass
        ay = (ty + drag_y + normal_y) / mass
        az = (tz + drag_z + normal_z - weight) / mass

        # Safety clamp: cap acceleration at 500 G
        MAX_ACCEL = 500.0 * 9.81
        ax = max(-MAX_ACCEL, min(MAX_ACCEL, ax))
        ay = max(-MAX_ACCEL, min(MAX_ACCEL, ay))
        az = max(-MAX_ACCEL, min(MAX_ACCEL, az))

        # Track peak force-based acceleration (excludes gravity). Running max
        # over every stage evaluation, not just the last RK4 stage.
        nonlocal _peak_force_accel
        force_ax = (tx + drag_x + normal_x) / mass
        force_ay = (ty + drag_y + normal_y) / mass
        force_az = (tz + drag_z + normal_z) / mass
        fa = math.sqrt(force_ax**2 + force_ay**2 + force_az**2)
        if fa > _peak_force_accel:
            _peak_force_accel = fa

        # Rotational dynamics
        if on_rail:
            pitch_accel = yaw_accel = roll_accel = 0.0
        else:
            inertia = _estimate_inertia(mass, body_length)
            ixx = inertia * 0.1  # Roll inertia much smaller
            pitch_accel = M_pitch / inertia
            yaw_accel = M_yaw / inertia
            roll_accel = M_roll / max(ixx, 0.01)

            # Clamp angular accelerations
            MAX_ROT = 100.0  # rad/s²
            pitch_accel = max(-MAX_ROT, min(MAX_ROT, pitch_accel))
            yaw_accel = max(-MAX_ROT, min(MAX_ROT, yaw_accel))
            roll_accel = max(-MAX_ROT, min(MAX_ROT, roll_accel))

        # Mass flow
        isp = config.motor_isp
        if isp <= 0:
            if config.motor_total_impulse > 0 and config.propellant_mass > 0:
                isp = config.motor_total_impulse / (config.propellant_mass * g)
        if thrust > 0 and (mass - burnout_mass) > 1e-3:
            if isp > 10:
                dm_dt = -thrust / (isp * g)
            elif config.motor_burn_time > 0:
                dm_dt = -initial_prop_mass / config.motor_burn_time
            else:
                dm_dt = 0.0
        else:
            dm_dt = 0.0

        return [vx, vy, vz, ax, ay, az,
                pitch_rate_, yaw_rate_, roll_rate_,
                pitch_accel, yaw_accel, roll_accel, dm_dt]

    # ── Main simulation loop ──────────────────────────────────────
    t = 0.0
    prev_velocity = 0.0
    failure_reasons: List[str] = []

    try:
        while t < 600.0:
            # ── Phase-aware adaptive time stepping ─────────────────
            # During powered flight, thrust changes rapidly — need small dt
            # for accurate energy integration. During coast/descent, dynamics
            # are smooth — 5× larger dt is safe and gives big speedup.
            thrust_now_check = _get_thrust(t, thrust_curve)
            in_powered_phase = thrust_now_check > 0 or t < config.motor_burn_time + 0.5
            if in_powered_phase:
                # Powered phase (+ 0.5s buffer after burnout for transients)
                dt_phase = dt_base
            else:
                # Coast / descent — dynamics are smooth, safe to use 5× dt
                dt_phase = min(dt_base * 5.0, 0.05)

            dt_candidates = [dt_phase]
            # Limit by pitch rate (max 3° per step)
            if abs(state_vec[9]) > 0.01:
                dt_candidates.append(math.radians(3) / abs(state_vec[9]))
            # Limit by roll rate (max ~57° per step)
            if abs(state_vec[11]) > 0.01:
                dt_candidates.append(1.0 / abs(state_vec[11]))
            # Growth limiter only during powered flight (coast can jump immediately)
            if in_powered_phase:
                dt_candidates.append(1.5 * prev_dt)
            # Minimum step floor
            dt_min = dt_base / 20.0
            adaptive_dt = max(dt_min, min(dt_candidates))
            prev_dt = adaptive_dt

            # ── Integrate ─────────────────────────────────────────
            try:
                new_vec = integrator.step(state_vec, t, adaptive_dt, _derivatives)
            except Exception as exc:
                logger.warning(f"Batch sim integration error at t={t:.3f}s: {exc}")
                failure_reasons.append(f"Integration error: {exc}")
                break

            # ── Clamp at ground ───────────────────────────────────
            if new_vec[2] < 0.0:
                new_vec[2] = 0.0
                if new_vec[5] < 0.0:
                    new_vec[5] = 0.0

            # Enforce mass floor (structure + spent motor casing)
            new_vec[12] = max(burnout_mass, new_vec[12])

            state_vec = new_vec
            t += adaptive_dt

            # ── Unpack relevant quantities ────────────────────────
            cur_x = state_vec[0]
            cur_y = state_vec[1]
            cur_z = max(0.0, state_vec[2])
            cur_vx = state_vec[3]
            cur_vy = state_vec[4]
            cur_vz = state_vec[5]
            cur_mass = state_vec[12]

            cur_speed = math.sqrt(cur_vx**2 + cur_vy**2 + cur_vz**2)
            signed_vel = cur_vz  # Use vertical velocity for phase detection

            a_sound = atmosphere.speed_of_sound(cur_z)
            cur_mach = cur_speed / a_sound if a_sound > 0 else 0.0
            rho = atmosphere.density(cur_z)
            cur_q = 0.5 * rho * cur_speed**2

            thrust_now = _get_thrust(t, thrust_curve)

            # Force-based acceleration (F/m, excludes gravity weight) — peak so far
            cur_accel = _peak_force_accel

            # ── Track maxima ──────────────────────────────────────
            if cur_z > result.apogee:
                result.apogee = cur_z
            if cur_speed > result.max_velocity:
                result.max_velocity = cur_speed
            if cur_mach > result.max_mach:
                result.max_mach = cur_mach
            if cur_accel > result.max_acceleration:
                result.max_acceleration = cur_accel
            if cur_q > result.max_dynamic_pressure:
                result.max_dynamic_pressure = cur_q

            # ── Divergence detection ──────────────────────────────
            if cur_speed > 2000.0:  # > ~Mach 6 at sea level
                failure_reasons.append("Simulation divergence: velocity exceeded 2000 m/s")
                break
            if cur_z > 100_000:  # Above Kármán line
                failure_reasons.append("Simulation divergence: altitude exceeded 100 km")
                break

            # Stability margin tracking (only during powered/coast phases)
            if phase in (FlightPhase.IGNITION, FlightPhase.BOOST, FlightPhase.COAST):
                if aero_model is not None and diameter > 0:
                    cg_now = _compute_cg(cur_mass)
                    sm = aero_model.stability_margin(aero_model.cp_subsonic(), cg_now)
                    if sm < result.min_stability_margin:
                        result.min_stability_margin = sm

            # Rail exit velocity: first time altitude exceeds rocket length
            if not rail_exit_detected and cur_z >= body_length and body_length > 0:
                result.rail_exit_velocity = cur_speed
                rail_exit_detected = True

            # ── Phase evaluation ──────────────────────────────────
            prev_phase = phase
            phase = phase_mgr.evaluate(
                phase, t, cur_z, signed_vel,
                thrust_now, config.main_deploy_altitude,
                drogue_delay=config.drogue_deploy_delay,
            )

            # ── Recovery deployment ───────────────────────────────
            if phase == FlightPhase.DROGUE_DESCENT and not drogue_deployed:
                drogue_deployed = True

            if phase == FlightPhase.MAIN_DESCENT and not main_deployed:
                main_deployed = True

            # ── Ground check ──────────────────────────────────────
            descent_phases = (
                FlightPhase.DROGUE_DESCENT,
                FlightPhase.MAIN_DESCENT,
                FlightPhase.APOGEE,
            )
            if cur_z <= 0 and phase in descent_phases:
                phase = FlightPhase.LANDED
            elif cur_z <= 0 and t > config.motor_burn_time + 5.0:
                phase = FlightPhase.LANDED
            # Universal ground check: ballistic descent regardless of phase
            elif cur_z <= 0 and cur_vz < 0 and t > 2.0:
                phase = FlightPhase.LANDED

            prev_velocity = signed_vel

            if phase == FlightPhase.LANDED:
                break

    except Exception as exc:
        failure_reasons.append(f"Simulation error: {exc}")
        logger.error(f"Batch simulation failed at t={t:.3f}s: {exc}")

    # ── Finalize result ───────────────────────────────────────────
    result.flight_time = t
    result.landing_x = state_vec[0]
    result.landing_y = state_vec[1]
    result.landing_distance = math.sqrt(state_vec[0]**2 + state_vec[1]**2)
    result.final_phase = phase.value

    # If stability margin was never updated, reset sentinel
    if result.min_stability_margin == float("inf"):
        result.min_stability_margin = 0.0

    # Determine success
    if failure_reasons:
        result.success = False
        result.failure_reasons = failure_reasons
    elif phase == FlightPhase.TIMEOUT:
        result.success = False
        result.failure_reasons = ["Simulation timed out"]
    elif result.apogee < 1.0:
        result.success = False
        result.failure_reasons = ["Apogee below 1 m — likely no thrust"]
    else:
        result.success = True
        result.failure_reasons = []

    logger.debug(
        f"Batch sim complete — Apogee: {result.apogee:.1f} m, "
        f"Max V: {result.max_velocity:.1f} m/s, "
        f"Max Mach: {result.max_mach:.3f}, "
        f"Landing dist: {result.landing_distance:.1f} m, "
        f"Phase: {result.final_phase}"
    )
    return result
