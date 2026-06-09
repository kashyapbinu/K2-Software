"""
K2 Aerospace — Simulation Engine v3 (6DOF, OpenRocket Physics)
================================================================
Real-time trajectory simulation using modular core components.

Architecture:
    RK4Integrator   →  state propagation (6DOF)
    Atmosphere       →  density, Mach, dynamic pressure, Reynolds
    PhaseManager    →  automatic phase transitions
    EventManager    →  event-driven subsystem notifications
    HistoryManager  →  full flight data recording

Physics (6DOF state vector):
    state_vec = [x, y, z, vx, vy, vz, pitch, yaw, roll,
                 pitch_rate, yaw_rate, roll_rate, mass]

    Aerodynamics:  OpenRocket-grade Barrowman + Galejs body lift
    Drag:          Reynolds-based skin friction, Mach-dependent base/pressure
    Wind:          Pink noise turbulence (IIR α=5/3)
    Gravity:       Altitude-dependent (inverse-square law)
    Time stepping: Adaptive multi-criteria (OpenRocket method)
"""

import math
import logging
import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from core.constants import G_EARTH
from environment.atmosphere_model import Atmosphere
from core.flight_phases import FlightPhase, PhaseManager
from core.integrators import get_integrator
from core.event_manager import EventManager, SimEvent
from core.history_manager import HistoryManager
from physics.aerodynamics import AeroModel, compute_drag_coefficient, compute_drag_force
from environment.wind_model import WindModel
from vehicle.builder import build_vehicle
from vehicle.motor import Motor

logger = logging.getLogger("K2.SimEngine")


class SimulationEngine(QObject):
    """
    Real-time flight simulation engine.
    Runs a physics loop via QTimer, updating RocketStateEngine each tick.
    """

    sim_started = pyqtSignal()
    sim_paused = pyqtSignal()
    sim_resumed = pyqtSignal()
    sim_finished = pyqtSignal()
    sim_tick = pyqtSignal(float)  # current sim time

    def __init__(self, state_engine):
        super().__init__()
        self.engine = state_engine
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer_tick)
        self._running = False
        self._paused = False

        # ── Core modules ──
        self.atmosphere = Atmosphere()
        self.phase_mgr = PhaseManager()
        self.event_mgr = EventManager()
        self.history = HistoryManager()
        self.integrator = get_integrator("rk4")

        # ── Vehicle & Aerodynamics ──
        self.vehicle    = None
        self.aero_model = None
        self.wind_model = WindModel()          # default: calm
        self._initial_prop_mass = 0.0
        self._thrust_curve = []
        self._state_vec = None
        self._prev_dt = 0.01
        self._last_aero = {}

        # ── Tracking ──
        self._phase = FlightPhase.PRELAUNCH
        self._max_q = 0.0
        self._max_q_fired = False
        self._apogee_time = 0.0
        self._drogue_deployed = False
        self._main_deployed = False

    # ── Public API ────────────────────────────────────────────────

    def start(self):
        """Start or restart the simulation."""
        s = self.engine.state

        if s.motor_designation == "None":
            logger.warning("No motor selected — cannot simulate")
            self.engine.log_message.emit("⚠ No motor selected. Please select a motor in the Design tab.")
            return

        # Select integrator
        self.integrator = get_integrator(s.integrator_name)
        logger.info(f"Using {self.integrator.name} integrator")

        # Reset flight state
        self.engine.update(
            altitude=0.0, velocity=0.0, acceleration=0.0,
            mach_number=0.0, thrust=0.0, drag=0.0,
            sim_time=0.0, sim_running=True, sim_paused=False,
            sim_phase=FlightPhase.PRELAUNCH.value,
            parachute_deployed=False,
            flight_computer_state="ARMED",
            max_altitude=0.0, max_velocity=0.0,
            max_acceleration=0.0, max_mach=0.0,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            propellant_mass=s.propellant_mass_initial,
            dynamic_pressure=0.0,
            emit=False
        )

        # Build canonical vehicle from UI assembly
        if hasattr(self.engine, '_assembly') and self.engine._assembly:
            self.vehicle = build_vehicle(self.engine._assembly)
        else:
            logger.warning("No assembly on state engine — using state-only vehicle.")
            self.vehicle = None

        # Inject motor into the active stage
        if self.vehicle and self.vehicle.stages:
            sim_motor = Motor(
                s.motor_designation,
                empty_mass=s.dry_mass * 0.08,
                propellant_mass=s.propellant_mass_initial,
                length=0.3, diameter=0.03
            )
            self.vehicle.stages[-1].set_motor(sim_motor)

        # Build AeroModel from current state geometry
        self.aero_model = AeroModel.from_state(s)
        logger.info(
            f"AeroModel: CN_alpha={self.aero_model.cn_alpha_total:.2f}, "
            f"CP={self.aero_model.cp_subsonic():.3f}m, "
            f"Fineness={self.aero_model.fineness:.1f}"
        )

        # Build WindModel from simulation settings
        wind_speed = getattr(s, 'wind_speed', 0.0)
        wind_dir   = getattr(s, 'wind_direction', 0.0)
        wind_gust  = getattr(s, 'wind_gust_intensity', 0.0)
        self.wind_model = WindModel(wind_speed, wind_dir, wind_gust)

        # Reset modules
        self.phase_mgr.reset()
        self.event_mgr.clear_log()
        self.history.clear()
        self._phase = FlightPhase.PRELAUNCH
        self._max_q = 0.0
        self._max_q_fired = False
        self._apogee_time = 0.0
        self._drogue_deployed = False
        self._main_deployed = False
        self._state_vec = None

        self._initial_prop_mass = s.propellant_mass_initial
        self._build_thrust_curve()

        # Fire SIM_START event
        self.event_mgr.fire(SimEvent.SIM_START, {
            "time": 0.0,
            "motor": s.motor_designation,
            "integrator": self.integrator.name,
        })

        # Timer interval based on sim speed
        interval_ms = max(1, int(s.sim_dt * 1000 / s.sim_speed))
        self._timer.setInterval(interval_ms)

        self._running = True
        self._paused = False
        self._timer.start()

        self.sim_started.emit()
        logger.info(f"Simulation started (dt={s.sim_dt}s, speed={s.sim_speed}x, integrator={self.integrator.name})")
        self.engine.log_message.emit(f"🚀 Simulation started — Motor: {s.motor_designation} | Integrator: {self.integrator.name}")

    def pause(self):
        if self._running and not self._paused:
            self._timer.stop()
            self._paused = True
            self.engine.update(sim_paused=True)
            self.sim_paused.emit()
            logger.info("Simulation paused")

    def resume(self):
        if self._running and self._paused:
            self._paused = False
            self.engine.update(sim_paused=False)
            self._timer.start()
            self.sim_resumed.emit()
            logger.info("Simulation resumed")

    def stop(self):
        self._timer.stop()
        self._running = False
        self._paused = False
        self._phase = FlightPhase.TERMINATED
        self.engine.update(sim_running=False, sim_paused=False,
                          sim_phase=FlightPhase.TERMINATED.value)
        self.event_mgr.fire(SimEvent.SIM_END, {
            "time": self.engine.state.sim_time,
            "reason": "user_stop",
        })
        self.sim_finished.emit()
        logger.info("Simulation stopped")
        self.engine.log_message.emit("Simulation stopped by user")

    def set_speed(self, speed: float):
        s = self.engine.state
        self.engine.update(sim_speed=speed, emit=False)
        if self._running and not self._paused:
            interval_ms = max(1, int(s.sim_dt * 1000 / speed))
            self._timer.setInterval(interval_ms)

    @property
    def is_running(self):
        return self._running

    @property
    def is_paused(self):
        return self._paused

    # ── Thrust Curve ──────────────────────────────────────────────

    def _build_thrust_curve(self):
        """Build a simplified trapezoidal thrust curve from motor params, or use custom curve."""
        s = self.engine.state
        
        if hasattr(s, "custom_thrust_curve") and s.custom_thrust_curve:
            self._thrust_curve = list(s.custom_thrust_curve)
            return

        bt = s.motor_burn_time
        avg_t = s.motor_avg_thrust
        max_t = s.motor_max_thrust if s.motor_max_thrust > 0 else avg_t * 1.4

        if bt <= 0:
            self._thrust_curve = []
            return

        ramp = bt * 0.1
        self._thrust_curve = [
            (0.0, 0.0),
            (ramp, max_t),
            (bt - ramp, avg_t),
            (bt, 0.0),
        ]

    def _get_thrust(self, t: float) -> float:
        """Interpolate thrust at time t."""
        if not self._thrust_curve or t < 0:
            return 0.0
        curve = self._thrust_curve
        if t >= curve[-1][0]:
            return 0.0
        for i in range(len(curve) - 1):
            t0, v0 = curve[i]
            t1, v1 = curve[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if (t1 - t0) > 0 else 0
                return v0 + frac * (v1 - v0)
        return 0.0

    # ── Physics: Derivatives Function ─────────────────────────────

    def _derivatives(self, t: float, state_vec: list) -> list:
        """
        6DOF equations of motion (OpenRocket-grade).
        state_vec = [x, y, z, vx, vy, vz, pitch, yaw, roll,
                     pitch_rate, yaw_rate, roll_rate, mass]
        Returns derivatives of all 13 state variables.
        """
        x, y, z, vx, vy, vz, pitch, yaw, roll, \
            pitch_rate, yaw_rate, roll_rate, mass = state_vec

        if any(not math.isfinite(v) for v in state_vec):
            raise ValueError(f"Non-finite state at t={t:.4f}s")

        z = max(0.0, z)
        mass = max(0.01, mass)
        s = self.engine.state

        # Rail constraint
        on_rail = z < s.length
        if on_rail or self._phase == FlightPhase.PRELAUNCH:
            launch_pitch = math.radians(getattr(s, 'launch_angle', 90.0))
            pitch = launch_pitch
            yaw = 0.0
            pitch_rate = yaw_rate = roll_rate = 0.0

        # Atmosphere
        rho = self.atmosphere.density(z)
        a_sound = self.atmosphere.speed_of_sound(z)

        # Altitude-dependent gravity
        from core.constants import gravity_at_altitude
        g = gravity_at_altitude(z)

        # Wind
        wind_vx, wind_vy, wind_vz = self.wind_model.get_wind_velocity(z, t)

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
        ref_area = math.pi * (s.diameter / 2)**2

        # Aerodynamic model
        cg = s.cg
        if self.aero_model is not None:
            aero = self.aero_model.compute(alpha, mach, q_dyn, pitch_rate, v_rel, cg)
            F_drag = aero["F_drag"]
            F_normal = aero["F_normal"]
            M_pitch = aero["M_pitch"]
            cd = aero["cd"]
            cp = aero["cp"]
            stab_margin = aero["stability_margin"]
            # Yaw moment (symmetric to pitch for axisymmetric rocket) + yaw
            # damping mirroring the pitch-damping fix (else yaw weathercock is
            # undamped and tumbles in crosswind just like pitch did).
            M_yaw = -aero.get("cm", 0) * q_dyn * ref_area * s.diameter * math.sin(beta_angle)
            M_yaw += self.aero_model._damping_moment(
                aero.get("cmq", -1.0), yaw_rate, v_rel, q_dyn,
                ref_area, s.diameter, aero.get("cn_total", 2.0))
        else:
            from physics.aerodynamics import compute_cd as _cd
            cd = _cd(mach, alpha, s.length / max(s.diameter, 0.01))
            F_drag = q_dyn * ref_area * cd
            F_normal = 0.0
            M_pitch = 0.0
            M_yaw = 0.0
            cp = s.cp
            stab_margin = (cp - cg) / max(s.diameter, 0.01)

        # Recovery drag override
        if self._drogue_deployed and not self._main_deployed and vz < 0:
            F_drag = 0.5 * rho * v_rel**2 * getattr(s, 'drogue_cd_area', 0.4)
            F_normal = M_pitch = M_yaw = 0.0
        elif self._main_deployed and vz < 0:
            F_drag = 0.5 * rho * v_rel**2 * getattr(s, 'main_cd_area', 1.5)
            F_normal = M_pitch = M_yaw = 0.0

        # Roll damping (simple model)
        M_roll = -0.01 * roll_rate * q_dyn * ref_area * s.diameter if v_rel > 1.0 else 0.0

        # Gust torque ("prevent over-perfect flight"). Must be a SMOOTH, low-
        # frequency, time-correlated perturbation — NOT per-step white noise.
        # White noise has spectral energy at the pitch/yaw natural frequency
        # (~10-15 Hz at high q), which resonantly pumps the lightly-damped
        # attitude mode into a divergent tumble in any crosswind. Low-frequency
        # gusts (<0.5 Hz, well below ω_n) perturb without exciting resonance.
        # q_dyn capped so the gust does not scale with the vehicle's own v².
        if not hasattr(self, "_gust_w"):
            self._gust_w = [0.7, 1.7, 3.1]   # rad/s, all ≪ attitude ω_n
            self._gust_ph_p = list(np.random.uniform(0, 2 * math.pi, 3))
            self._gust_ph_y = list(np.random.uniform(0, 2 * math.pi, 3))
        q_gust = min(q_dyn, 2000.0)
        gust_amp = 0.0006 * q_gust * ref_area * s.diameter
        gp = sum(math.sin(w * t + ph) for w, ph in zip(self._gust_w, self._gust_ph_p)) / 3.0
        gy = sum(math.sin(w * t + ph) for w, ph in zip(self._gust_w, self._gust_ph_y)) / 3.0
        M_pitch += gust_amp * gp
        M_yaw += gust_amp * gy

        # Thrust
        thrust = self._get_thrust(t)
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
        # Lateral (yaw-plane) normal force from sideslip — mirrors the pitch-plane
        # term so a crosswind produces side translation, not just a yaw moment.
        # Same CNα·sin(β) form with the 20° stall clamp; acts to oppose sideslip.
        _deployed = self._drogue_deployed or self._main_deployed
        if v_rel > 0.5 and abs(beta_angle) > 1e-6 and not _deployed:
            cn_total_now = aero.get("cn_total", 2.0) if self.aero_model is not None else 0.0
            eff_beta = min(abs(beta_angle), math.radians(20))
            F_normal_y = q_dyn * ref_area * cn_total_now * math.sin(eff_beta)
            normal_y = -math.copysign(F_normal_y, beta_angle)
        else:
            normal_y = 0.0

        weight = mass * g
        ax = (tx + drag_x + normal_x) / mass
        ay = (ty + drag_y + normal_y) / mass
        az = (tz + drag_z + normal_z - weight) / mass

        # Safety clamp: cap acceleration at 500G to prevent divergence
        MAX_ACCEL = 500.0 * 9.81
        ax = max(-MAX_ACCEL, min(MAX_ACCEL, ax))
        ay = max(-MAX_ACCEL, min(MAX_ACCEL, ay))
        az = max(-MAX_ACCEL, min(MAX_ACCEL, az))

        # Rotational dynamics
        if on_rail:
            pitch_accel = yaw_accel = roll_accel = 0.0
        else:
            if self.vehicle is not None:
                ixx_v, iyy, _ = self.vehicle.inertia_tensor()
                inertia = iyy if iyy > 0.1 else self._estimate_inertia(mass, s)
                # Real roll inertia from the vehicle; fall back to a solid-
                # cylinder estimate (m·r²/2) — NOT a fixed fraction of pitch.
                ixx = ixx_v if ixx_v > 1e-4 else self._estimate_roll_inertia(mass, s)
            else:
                inertia = self._estimate_inertia(mass, s)
                ixx = self._estimate_roll_inertia(mass, s)
            pitch_accel = M_pitch / inertia
            yaw_accel = M_yaw / inertia
            roll_accel = M_roll / max(ixx, 0.01)

            # Clamp angular accelerations
            MAX_ROT = 100.0  # rad/s²
            pitch_accel = max(-MAX_ROT, min(MAX_ROT, pitch_accel))
            yaw_accel = max(-MAX_ROT, min(MAX_ROT, yaw_accel))
            roll_accel = max(-MAX_ROT, min(MAX_ROT, roll_accel))

        # Mass flow
        isp = getattr(s, 'motor_isp', 0.0)
        if isp <= 0:
            total_impulse = getattr(s, 'motor_total_impulse', 0.0)
            prop_mass = getattr(s, 'propellant_mass_initial', 0.0)
            if total_impulse > 0 and prop_mass > 0:
                isp = total_impulse / (prop_mass * g)
        if thrust > 0 and (mass - s.dry_mass) > 1e-3:
            if isp > 10:
                dm_dt = -thrust / (isp * g)
            elif s.motor_burn_time > 0:
                dm_dt = -self._initial_prop_mass / s.motor_burn_time
            else:
                dm_dt = 0.0
        else:
            dm_dt = 0.0

        self._last_aero = {
            "cd": cd, "cp": cp, "alpha": alpha, "mach": mach,
            "q_dyn": q_dyn, "F_drag": F_drag, "stab_margin": stab_margin,
            "wind_vx": wind_vx, "thrust": thrust, "F_normal": F_normal,
        }

        return [vx, vy, vz, ax, ay, az,
                pitch_rate, yaw_rate, roll_rate,
                pitch_accel, yaw_accel, roll_accel, dm_dt]

    def _estimate_inertia(self, mass: float, s) -> float:
        """Fallback pitch inertia estimate: thin rod Iyy = m*L^2/12."""
        L = getattr(s, 'length', 2.0)
        return mass * L ** 2 / 12.0

    def _estimate_roll_inertia(self, mass: float, s) -> float:
        """Fallback roll inertia estimate: solid cylinder Ixx = m*r^2/2."""
        r = getattr(s, 'diameter', 0.1) / 2.0
        return max(mass * r ** 2 / 2.0, 1e-4)

    # ── Main Simulation Step ──────────────────────────────────────

    def _on_timer_tick(self):
        """Called by the QTimer to advance the simulation by one UI frame."""
        if not self._running or self._paused:
            return
            
        s = self.engine.state
        # Advance simulation time by one standard UI tick (multiplied by sim speed)
        # s.sim_dt is usually 0.01s. If sim_speed is 1x, target is current + 0.01s.
        target_time = s.sim_time + (s.sim_dt * s.sim_speed)
        
        # In adaptive stepping, if the pitch rate is very high, adaptive_dt might drop to 0.0005s.
        # This while loop ensures we take 20 micro-steps to catch up to the 10ms UI tick,
        # rather than advancing 0.0005s per UI tick (which makes it run at 0.05x speed).
        # We cap at 100 loops to prevent UI freezing if the physics gets catastrophically bogged down.
        max_loops = 100
        loops = 0
        
        while self._running and not self._paused and s.sim_time < target_time and loops < max_loops:
            self._step()
            loops += 1

    def _step(self):
        """Single simulation time step with 6DOF state and adaptive stepping."""
        s = self.engine.state
        dt = s.sim_dt
        t = s.sim_time

        # Initialise 6DOF state vector on first tick
        if t == 0 or self._state_vec is None:
            launch_pitch = math.radians(s.launch_angle)
            self._state_vec = [
                0.0,               # x
                0.0,               # y (lateral)
                s.altitude,        # z (altitude)
                0.0,               # vx
                0.0,               # vy
                s.velocity,        # vz
                launch_pitch,      # pitch
                0.0,               # yaw
                0.0,               # roll
                0.0,               # pitch_rate
                0.0,               # yaw_rate
                0.0,               # roll_rate
                s.total_mass(),    # mass
            ]
            self._prev_dt = dt

        # ── Adaptive time stepping (OpenRocket-style) ──
        dt_candidates = [dt]
        # Limit by pitch rate (max 3° per step)
        if abs(self._state_vec[9]) > 0.01:  # pitch_rate
            dt_candidates.append(math.radians(3) / abs(self._state_vec[9]))
        # Limit by roll rate (max ~57° per step)
        if abs(self._state_vec[11]) > 0.01:  # roll_rate
            dt_candidates.append(1.0 / abs(self._state_vec[11]))
        # Limit growth to 1.5x previous step
        dt_candidates.append(1.5 * self._prev_dt)
        # Limit by the attitude (weathercock) oscillation period at the current
        # dynamic pressure: ω_n = √(CNα·q·A·|CP−CG| / I). The pitch mode reaches
        # ~75 rad/s at high q; a coarse user dt (e.g. 0.05 s) gives ω·dt≈3.8,
        # beyond RK4's stable range → numerical tumbling that a stray gust
        # triggers. Cap dt so ω_n·dt ≤ 0.5 (mode stays well-resolved).
        aero = getattr(self, "_last_aero", {})
        q_dyn = aero.get("q_dyn", 0.0)
        if q_dyn > 0 and self.aero_model is not None:
            cna = getattr(self.aero_model, "cn_alpha_total", 2.0)
            A_ref = math.pi * (s.diameter / 2.0) ** 2
            margin = abs(aero.get("cp", s.cp) - s.cg)
            I_est = max(self._state_vec[12] * s.length ** 2 / 12.0, 1e-6)
            K = cna * q_dyn * A_ref * margin
            if K > 0:
                wn = math.sqrt(K / I_est)
                if wn > 0:
                    dt_candidates.append(0.5 / wn)
        # Minimum step floor
        dt_min = dt / 50.0
        adaptive_dt = max(dt_min, min(dt_candidates))
        self._prev_dt = adaptive_dt

        # ── Integrate ──
        try:
            new_vec = self.integrator.step(self._state_vec, t, adaptive_dt, self._derivatives)
        except Exception as e:
            logger.error(f"Integration error at t={t:.3f}s: {e}")
            self.stop()
            self.engine.log_message.emit(f"❌ Simulation error at T+{t:.2f}s: {e}")
            return

        # ── Clamp at ground ──
        old_mass = self._state_vec[12]
        if new_vec[2] < 0.0:        # z (altitude)
            new_vec[2] = 0.0
            if new_vec[5] < 0.0:    # vz
                new_vec[5] = 0.0

        # ── Recovery attitude stabilization ──
        # Under a deployed parachute the vehicle hangs from the canopy and stops
        # tumbling; zero the rates so the descent stays stable in the viewer.
        if self._drogue_deployed or self._main_deployed:
            new_vec[6] = math.pi / 2.0   # pitch upright
            new_vec[7] = 0.0             # yaw
            new_vec[9] = 0.0             # pitch_rate
            new_vec[10] = 0.0            # yaw_rate
            new_vec[11] = 0.0            # roll_rate

        self._state_vec = new_vec

        # ── Unpack 6DOF state ──
        new_x          = new_vec[0]
        new_y          = new_vec[1]
        new_altitude   = max(0.0, new_vec[2])
        new_vx         = new_vec[3]
        new_vy         = new_vec[4]
        new_vz         = new_vec[5]
        new_pitch      = new_vec[6]
        new_yaw        = new_vec[7]
        new_roll       = new_vec[8]
        new_pitch_rate = new_vec[9]
        new_yaw_rate   = new_vec[10]
        new_roll_rate  = new_vec[11]
        new_mass       = max(s.dry_mass, new_vec[12])
        new_prop_mass  = max(0.0, new_mass - s.dry_mass)

        # Update canonical vehicle propellant consumption
        if self.vehicle is not None:
            consumed = max(0.0, old_mass - new_mass)
            self.vehicle.consume_propellant(consumed)

        # Signed velocity magnitude
        new_velocity = math.sqrt(new_vx**2 + new_vy**2 + new_vz**2)
        if new_vz < 0:
            new_velocity = -new_velocity

        # ── Derived quantities ──
        thrust       = self._get_thrust(t)
        atm_temp     = self.atmosphere.temperature(new_altitude)
        atm_press    = self.atmosphere.pressure(new_altitude)
        atm_rho      = self.atmosphere.density(new_altitude)
        a_sound      = self.atmosphere.speed_of_sound(new_altitude)
        mach         = abs(new_velocity) / a_sound if a_sound > 0 else 0.0
        dyn_pressure = 0.5 * atm_rho * new_velocity ** 2

        # Use aero data from _derivatives if available
        aero = getattr(self, '_last_aero', {})
        cd = aero.get('cd', s.cd)
        ref_area   = math.pi * (s.diameter / 2) ** 2
        drag_force = aero.get('F_drag', 0.5 * atm_rho * new_velocity**2 * cd * ref_area)

        if self._main_deployed and new_velocity < 0:
            drag_force = 0.5 * atm_rho * new_velocity ** 2 * getattr(s, 'main_cd_area', 1.5)
        elif self._drogue_deployed and new_velocity < 0:
            drag_force = 0.5 * atm_rho * new_velocity ** 2 * getattr(s, 'drogue_cd_area', 0.4)

        # Rate of change of SPEED (unsigned) — using the signed pseudo-magnitude
        # here produced a spurious spike at apogee, where the sign convention
        # flips from +|v| to -|v| while the horizontal speed is still finite.
        accel = (abs(new_velocity) - abs(s.velocity)) / adaptive_dt if adaptive_dt > 0 else 0.0

        # ── Phase evaluation ──
        prev_phase = self._phase
        self._phase = self.phase_mgr.evaluate(
            self._phase, t + adaptive_dt, new_altitude, new_velocity,
            thrust, getattr(s, 'main_deploy_altitude', 300.0)
        )

        # ── Events ──
        if prev_phase != self._phase:
            self.event_mgr.fire(SimEvent.PHASE_CHANGE, {
                "time": t + adaptive_dt, "from": prev_phase.value,
                "to": self._phase.value, "altitude": new_altitude,
            })

            if self._phase == FlightPhase.COAST and prev_phase == FlightPhase.BOOST:
                self.event_mgr.fire(SimEvent.MOTOR_BURNOUT, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                    "velocity": new_velocity,
                })

            if self._phase == FlightPhase.APOGEE:
                self._apogee_time = t + adaptive_dt
                self.event_mgr.fire(SimEvent.APOGEE, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                    "max_velocity": s.max_velocity,
                })

            if self._phase == FlightPhase.DROGUE_DESCENT and not self._drogue_deployed:
                self._drogue_deployed = True
                self.event_mgr.fire(SimEvent.DROGUE_DEPLOY, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                })
                logger.info(f"Drogue deployed at {new_altitude:.1f}m")

            if self._phase == FlightPhase.MAIN_DESCENT and not self._main_deployed:
                self._main_deployed = True
                self.event_mgr.fire(SimEvent.MAIN_DEPLOY, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                })
                logger.info(f"Main chute deployed at {new_altitude:.1f}m")

        # Max-Q tracking
        if dyn_pressure > self._max_q:
            self._max_q = dyn_pressure
        elif not self._max_q_fired and dyn_pressure < self._max_q * 0.95 and self._max_q > 100:
            self._max_q_fired = True
            self.event_mgr.fire(SimEvent.MAX_Q, {
                "time": t + adaptive_dt, "max_q": self._max_q,
                "altitude": new_altitude, "mach": mach,
            })

        # ── Ground check ──
        _descent_phases = (
            FlightPhase.DROGUE_DESCENT,
            FlightPhase.MAIN_DESCENT,
            FlightPhase.APOGEE,
        )
        if new_altitude <= 0 and self._phase in _descent_phases:
            new_altitude = 0.0
            new_velocity = 0.0
            accel        = 0.0
            self._phase  = FlightPhase.LANDED
        elif new_altitude <= 0 and t > s.motor_burn_time + 5.0:
            new_altitude = 0.0
            new_velocity = 0.0
            accel        = 0.0
            self._phase  = FlightPhase.LANDED

        # ── Gyro simulation ──
        # A rate gyro reads the body angular rates (rad/s), not acceleration.
        # Map roll/pitch/yaw rate to the three axes and add a small sensor-noise
        # floor (~0.02 rad/s ≈ 1°/s, typical MEMS).
        gyro_x_val = new_roll_rate  + np.random.normal(0, 0.02)
        gyro_y_val = new_pitch_rate + np.random.normal(0, 0.02)
        gyro_z_val = new_yaw_rate   + np.random.normal(0, 0.02)

        # ── Structural loads & Dynamics ──
        from core.constants import gravity_at_altitude
        from physics.structures import compute_all
        from structures.thermal_analysis import recovery_temperature
        net_force    = thrust - drag_force - new_mass * gravity_at_altitude(new_altitude)

        # Aerodynamic skin heating: adiabatic-wall (recovery) temperature is the
        # conservative skin temp; thermal stress is driven by its rise above the
        # 293.15 K stress-free assembly temperature.
        wall_temp = recovery_temperature(atm_temp, mach) if mach > 0 else atm_temp
        delta_T = wall_temp - 293.15

        # Aerodynamic body bending: normal force acts at the CP, reacted by
        # inertia about the CG → bending moment M = |F_normal| · |x_CP - x_CG|.
        # Clamp single-tick gust spikes: a real airframe sheds load past large
        # AoA / stall, so cap the normal force used for structural stress at the
        # value a credible gust AoA (12°) would produce at the current q. This
        # leaves the trajectory dynamics untouched — only the stress input.
        F_normal = aero.get('F_normal', 0.0)
        q_dyn_now = aero.get('q_dyn', 0.0)
        cn_alpha = getattr(self.aero_model, 'cn_alpha_total', 2.0) if self.aero_model else 2.0
        F_normal_cap = q_dyn_now * cn_alpha * math.radians(12.0) * ref_area
        F_struct = min(abs(F_normal), F_normal_cap) if F_normal_cap > 0 else abs(F_normal)
        bend_arm = abs(aero.get('cp', s.cp) - s.cg)
        bending_moment = F_struct * bend_arm

        # Calculate full analytical stress state
        struct_res = compute_all(
            force=max(abs(net_force), abs(thrust), new_mass * gravity_at_altitude(new_altitude)),
            diameter=s.diameter,
            wall_thickness=getattr(s, 'wall_thickness', 0.002),
            length=s.length,
            material_name=getattr(s, 'material_name', 'Aluminum 6061-T6'),
            internal_pressure=0.0,
            bending_moment=bending_moment,
            shear_force=F_struct,
            delta_T=delta_T,
            thermal_constraint=0.55  # real airframe: slip joints / free ends
        )
        
        # Calculate flutter margin (simplified for simulation loop)
        flutter_margin = getattr(s, 'flutter_speed', float('inf')) / new_velocity if new_velocity > 0 else float('inf')

        # ── Update state (6DOF) ──
        self.engine.update_sim_tick(
            sim_time=t + adaptive_dt,
            x_position=new_x,
            y_position=new_y,
            altitude=new_altitude,
            velocity_x=new_vx,
            velocity_y=new_vy,
            velocity_z=new_vz,
            velocity=new_velocity,
            pitch=new_pitch,
            yaw=new_yaw,
            roll=new_roll,
            pitch_rate=new_pitch_rate,
            yaw_rate=new_yaw_rate,
            roll_rate=new_roll_rate,
            acceleration=accel,
            mach_number=mach,
            thrust=thrust,
            drag=abs(drag_force),
            net_force=net_force,
            propellant_mass=new_prop_mass,
            sim_phase=self._phase.value,
            flight_computer_state=self._phase.fc_state,
            parachute_deployed=self._drogue_deployed or self._main_deployed,
            cd=cd,
            axial_stress=struct_res['axial'],
            hoop_stress=struct_res['hoop'],
            von_mises_stress=struct_res['von_mises'],
            shear_stress=struct_res['shear'],
            thermal_stress=struct_res['thermal'],
            safety_factor=struct_res['safety_factor'],
            margin_of_safety=struct_res['margin_of_safety'],
            wall_temperature=wall_temp,
            flutter_margin=flutter_margin,
            max_stress=max(s.max_stress, struct_res['von_mises']),
            atm_temperature=atm_temp,
            atm_pressure=atm_press,
            atm_density=atm_rho,
            dynamic_pressure=dyn_pressure,
            gyro_x=gyro_x_val,
            gyro_y=gyro_y_val,
            gyro_z=gyro_z_val,
        )

        # ── Record history ──
        self.history.record(
            time=t + adaptive_dt,
            x=new_x,
            y=new_y,
            altitude=new_altitude,
            vx=new_vx,
            vy=new_vy,
            vz=new_vz,
            velocity=new_velocity,
            pitch=new_pitch,
            yaw=new_yaw,
            roll=new_roll,
            pitch_rate=new_pitch_rate,
            yaw_rate=new_yaw_rate,
            roll_rate=new_roll_rate,
            acceleration=accel,
            mach=mach,
            thrust=thrust,
            drag=abs(drag_force),
            net_force=net_force,
            mass=new_mass,
            cg=s.cg,
            cp=aero.get('cp', s.cp),
            stability_margin=aero.get('stab_margin', s.stability_margin),
            phase=self._phase.value,
            cd=cd,
            atm_temperature=atm_temp,
            atm_pressure=atm_press,
            atm_density=atm_rho,
            dynamic_pressure=dyn_pressure,
            axial_stress=struct_res['axial'],
            hoop_stress=struct_res['hoop'],
            von_mises_stress=struct_res['von_mises'],
            shear_stress=struct_res['shear'],
            thermal_stress=struct_res['thermal'],
            safety_factor=struct_res['safety_factor'],
            margin_of_safety=struct_res['margin_of_safety'],
            wall_temperature=wall_temp,
            flutter_margin=flutter_margin,
            propellant_mass=new_prop_mass,
        )

        self.sim_tick.emit(t + adaptive_dt)

        # ── Termination ──
        if self._phase == FlightPhase.LANDED:
            self._timer.stop()
            self._running = False
            self.engine.update(sim_running=False, sim_phase=FlightPhase.LANDED.value)
            self.event_mgr.fire(SimEvent.LANDING, {
                "time": t + dt, "flight_time": t + dt,
                "max_altitude": s.max_altitude,
                "max_velocity": s.max_velocity,
                "max_mach": s.max_mach,
            })
            self.event_mgr.fire(SimEvent.SIM_END, {
                "time": t + dt, "reason": "landed",
                "data_points": self.history.count,
            })
            self.sim_finished.emit()
            logger.info(
                f"Flight complete — Apogee: {s.max_altitude:.1f}m, "
                f"Max V: {s.max_velocity:.1f}m/s, "
                f"Max Mach: {s.max_mach:.3f}"
            )
            self.engine.log_message.emit(
                f"✅ Flight complete — Apogee: {s.max_altitude:.1f}m | "
                f"Max Velocity: {s.max_velocity:.1f}m/s | "
                f"Max Mach: {s.max_mach:.3f} | "
                f"Flight Time: {t+dt:.2f}s | "
                f"Events: {len(self.event_mgr.event_log)}"
            )

        # Safety timeout in SIM time. Must exceed the slow drogue descent of a
        # high-apogee flight (e.g. a ~14 km descent under a small drogue takes
        # ~18 min) or the flight is cut off mid-recovery and reported as a
        # spurious TIMEOUT instead of LANDED.
        if t + dt > 3600:
            self._timer.stop()
            self._running = False
            self._phase = FlightPhase.TIMEOUT
            self.engine.update(sim_running=False, sim_phase=FlightPhase.TIMEOUT.value)
            self.event_mgr.fire(SimEvent.SIM_END, {"time": t + dt, "reason": "timeout"})
            self.sim_finished.emit()
            logger.warning("Simulation timeout (3600s)")
