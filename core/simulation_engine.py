"""
K2 AeroSim — Simulation Engine v3 (6DOF, OpenRocket Physics)
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
from physics.aerodynamics import (AeroModel, compute_drag_coefficient,
                                  compute_drag_force,
                                  compute_pitching_moment_coefficient)
from environment.wind_model import WindModel, MultiLevelWindModel
from vehicle.builder import build_vehicle
from vehicle.motor import Motor
from recovery.parachute_dynamics import RecoverySystem
from core.staging import StageManager, StageConfig

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
    sim_failed = pyqtSignal(str, str)   # (title, detail) — flight diverged/unstable or bad config
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

        # ── Staging ──
        self.stage_mgr = None           # StageManager (per-stage motors + separation)
        self.spent_stages = []          # separation-state snapshots of dropped stages
        self.spent_stage_results = []   # ballistic landing results for dropped stages
        self._flutter_warned = False    # one-shot in-flight flutter warning

        # ── Recovery ──
        self.recovery = None            # RecoverySystem (inflation-aware drag)
        self._recovery_shock_peak = 0.0  # max opening-shock drag force (N)
        self._last_descent_vz = 0.0      # |vz| of most recent descent tick
        self._drogue_descent_rate = 0.0  # captured at main deploy

    # ── Public API ────────────────────────────────────────────────

    def start(self):
        """Start or restart the simulation."""
        s = self.engine.state

        # Multistage flights carry their motors in stages_config; only require a
        # scalar motor when no stage config is present (single-stage path).
        if s.motor_designation == "None" and not getattr(s, 'stages_config', None):
            logger.warning("No motor selected — cannot simulate")
            self.engine.log_message.emit("No motor selected. Please select a motor in the Design tab.")
            return

        # Degenerate geometry guard: zero diameter/length means zero drag
        # reference area — the flight would be ballistic nonsense (happens when
        # state was reset and geometry never re-synced from the design).
        if s.diameter <= 0.0 or s.length <= 0.0:
            logger.warning(f"Invalid geometry (L={s.length}, D={s.diameter}) — cannot simulate")
            self.engine.log_message.emit(
                "Rocket geometry is empty (diameter/length = 0). "
                "Open the Design tab so the airframe is loaded, then run again.")
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
                empty_mass=s.motor_dry_mass if s.motor_dry_mass > 0 else s.dry_mass * 0.08,
                propellant_mass=s.propellant_mass_initial,
                length=0.3, diameter=0.03
            )
            self.vehicle.stages[-1].set_motor(sim_motor)

        # Build the staging manager. With no stages_config this wraps the
        # single scalar motor as one stage (numerically identical to the legacy
        # single-body path); with a config it drives true multistage flight.
        stages_cfg = getattr(s, 'stages_config', None)
        if stages_cfg:
            self.stage_mgr = StageManager(
                [StageConfig.from_dict(d) for d in stages_cfg])
            logger.info(f"Multistage: {self.stage_mgr.num_stages} stages")
        else:
            self.stage_mgr = StageManager.from_state(s)

        # Build AeroModel. Multistage rebuilds it from the active stack geometry
        # (and again at each separation); single-stage uses the full state.
        if self.stage_mgr.is_multistage:
            self.aero_model = AeroModel.from_state(self.stage_mgr.aero_config())
        else:
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
        wind_mode  = getattr(s, 'wind_mode', 'average')
        wind_layers = getattr(s, 'wind_layers', [])
        if wind_mode == 'multi_level' and wind_layers:
            self.wind_model = MultiLevelWindModel(
                wind_layers, turbulence_intensity=wind_gust)
            logger.info(f"Wind: multi-level, {len(wind_layers)} layers, "
                        f"TI={wind_gust:.2f}")
        else:
            self.wind_model = WindModel(wind_speed, wind_dir, wind_gust)

        # Launch-site temperature → ISA+ΔT atmosphere offset
        self.atmosphere.temperature_offset = (
            getattr(s, 'ground_temperature', 288.15) - 288.15)

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

        # Build inflation-aware recovery model from the configured Cd×A values.
        # Replaces the old instant-full-drag CdA constants: a real canopy fills
        # over ~0.5–1 s, so the drag (and the structural opening shock) ramps in.
        self.recovery = RecoverySystem.from_cd_areas(
            drogue_cd_area=getattr(s, 'drogue_cd_area', 0.5),
            main_cd_area=getattr(s, 'main_cd_area', 3.0),
            main_deploy_altitude=getattr(s, 'main_deploy_altitude', 300.0),
            drogue_delay=getattr(s, 'drogue_deploy_delay', 1.0),
            inflation_time=getattr(s, 'recovery_inflation_time', 0.6),
        )
        self._recovery_shock_peak = 0.0
        self._last_descent_vz = 0.0
        self._drogue_descent_rate = 0.0
        self.spent_stages = []
        self.spent_stage_results = []
        self._flutter_warned = False
        self.engine.update(flutter_exceeded=False, emit=False)

        # ── Flight-sanity monitor accumulators ──
        self._unstable_time = 0.0       # sustained negative-static-margin time (s)
        self._tumble_time = 0.0         # sustained high-body-rate time (s)
        self._aborted = False           # one-shot guard so abort fires once

        self._initial_prop_mass = s.propellant_mass_initial
        self._build_thrust_curve()

        # Pre-flight config check — catch unflyable setups before the timer runs
        # (negative liftoff thrust-to-weight, etc.) so the user gets a clear
        # reason instead of a rocket that sits on the pad or tumbles instantly.
        ok, title, detail = self._validate_config(s)
        if not ok:
            logger.warning(f"Config rejected: {title} — {detail}")
            self.engine.update(sim_running=False, emit=False)
            self.engine.log_message.emit(f"❌ {title}: {detail}")
            self.sim_failed.emit(title, detail)
            return

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
        self.engine.log_message.emit(f"Simulation started — Motor: {s.motor_designation} | Integrator: {self.integrator.name}")

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

    # ── Flight sanity / abort ─────────────────────────────────────────
    # Thresholds for declaring a flight broken. Sustained-time gates avoid
    # false aborts from a single transient gust tick.
    _TUMBLE_RATE = 15.0          # body pitch/yaw rate (rad/s) ⇒ tumbling (~860°/s)
    _TUMBLE_HOLD = 0.20          # sustained s above _TUMBLE_RATE to abort
    _UNSTABLE_HOLD = 0.30        # sustained s of negative static margin to abort
    _MIN_AIRSPEED_STAB = 8.0     # below this, static margin/AoA is meaningless
    _VEL_DIVERGE = 8000.0        # |v| (m/s) past which the integration has run away

    def _validate_config(self, s) -> tuple:
        """Pre-flight check for unflyable setups. Returns (ok, title, detail)."""
        # Liftoff thrust-to-weight. Use the first burning stage's average thrust
        # vs the full stacked liftoff weight — T/W ≤ 1 never leaves the pad.
        from core.constants import G_EARTH
        try:
            mass0 = self.stage_mgr.total_mass()
            thr0 = self.stage_mgr.stages[0].motor_avg_thrust
        except Exception:
            mass0 = s.total_mass()
            thr0 = getattr(s, 'motor_avg_thrust', 0.0)
        weight0 = mass0 * G_EARTH
        if thr0 <= 0:
            return (False, "No thrust",
                    "The selected motor produces zero average thrust. "
                    "Pick a real motor or check the custom-motor inputs.")
        if weight0 > 0 and (thr0 / weight0) < 1.0:
            return (False, "Thrust-to-weight below 1.0",
                    f"Liftoff thrust {thr0:.0f} N cannot lift the "
                    f"{mass0:.2f} kg rocket (weight {weight0:.0f} N, "
                    f"T/W = {thr0 / weight0:.2f}). Use a more powerful motor "
                    f"or reduce dry mass.")
        return (True, "", "")

    def _abort(self, t, title, detail):
        """Halt the sim on a diverged/unstable flight with a clear diagnosis."""
        if self._aborted:
            return
        self._aborted = True
        self._timer.stop()
        self._running = False
        self._paused = False
        self._phase = FlightPhase.ABORTED
        self.engine.update(sim_running=False,
                           sim_phase=FlightPhase.ABORTED.value, emit=False)
        self.event_mgr.fire(SimEvent.SIM_ABORT, {
            "time": t, "title": title, "detail": detail,
        })
        logger.error(f"Flight ABORTED at T+{t:.2f}s — {title}: {detail}")
        self.engine.log_message.emit(f"❌ ABORTED at T+{t:.2f}s — {title}: {detail}")
        self.sim_failed.emit(f"{title} (T+{t:.2f}s)", detail)
        self.sim_finished.emit()

    def _check_flight_sanity(self, t, new_vec, new_altitude, new_velocity,
                             aero, mach) -> bool:
        """Inspect the just-computed step for divergence / loss of control.

        Returns True if the flight was aborted (caller must stop the step).
        Only the ascent (powered + coast to apogee) is policed: under canopy the
        body rates are legitimately odd and post-apogee airspeed is too low for
        a meaningful static margin."""
        # 1) Numerical divergence — non-finite state or runaway speed.
        if not all(math.isfinite(x) for x in new_vec):
            self._abort(t, "Numerical divergence",
                        "The integrator produced a non-finite state (NaN/Inf). "
                        "Try a smaller time step or the RK4 integrator.")
            return True
        if abs(new_velocity) > self._VEL_DIVERGE:
            self._abort(t, "Velocity diverged",
                        f"Speed reached {abs(new_velocity):.0f} m/s — the "
                        "integration has run away (unstable step). Reduce the "
                        "time step or check for a bad motor/mass configuration.")
            return True

        # Only police attitude during the ascent, above a usable airspeed.
        ascent = self._phase in (FlightPhase.BOOST, FlightPhase.COAST)
        if not ascent or self._drogue_deployed or self._main_deployed:
            self._unstable_time = 0.0
            self._tumble_time = 0.0
            return False

        dt = self._prev_dt
        speed = abs(new_velocity)

        # 2) Tumbling — sustained very high body pitch/yaw rate.
        body_rate = max(abs(new_vec[9]), abs(new_vec[10]))   # pitch_rate, yaw_rate
        if body_rate > self._TUMBLE_RATE and speed > self._MIN_AIRSPEED_STAB:
            self._tumble_time += dt
        else:
            self._tumble_time = 0.0
        if self._tumble_time >= self._TUMBLE_HOLD:
            self._abort(t, "Rocket tumbling",
                        f"Body rate {math.degrees(body_rate):.0f}°/s sustained — "
                        "the rocket has lost attitude control and is tumbling. "
                        "Usually a stability problem: move the CP aft (larger / "
                        "further-aft fins) or the CG forward (nose ballast).")
            return True

        # 3) Static instability — CP ahead of CG (negative margin) during ascent.
        margin = aero.get("stab_margin", None)
        if margin is not None and math.isfinite(margin) \
                and speed > self._MIN_AIRSPEED_STAB:
            if margin < 0.0:
                self._unstable_time += dt
            else:
                self._unstable_time = 0.0
            if self._unstable_time >= self._UNSTABLE_HOLD:
                cp = aero.get("cp", 0.0)
                cg = self.engine.state.cg
                self._abort(t, "Statically unstable",
                            f"Static margin is negative ({margin:.2f} cal): the "
                            f"centre of pressure (CP {cp:.3f} m) is ahead of the "
                            f"centre of gravity (CG {cg:.3f} m), so any disturbance "
                            "diverges and the rocket weathercocks out of control. "
                            "Add fin area / move fins aft, or add nose mass.")
                return True
        return False

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
        curve = [
            (0.0, 0.0),
            (ramp, max_t),
            (bt - ramp, avg_t),
            (bt, 0.0),
        ]
        # Normalize so the curve integrates to the motor's true total impulse
        # (avg_thrust × burn_time). The raw trapezoid overshoots by ~8% when
        # max_t = 1.4·avg_t, which inflated every burn's delivered impulse.
        impulse = sum(
            0.5 * (curve[i][1] + curve[i + 1][1]) * (curve[i + 1][0] - curve[i][0])
            for i in range(len(curve) - 1)
        )
        target = avg_t * bt
        if impulse > 0 and target > 0:
            scale = target / impulse
            curve = [(t, f * scale) for t, f in curve]
        self._thrust_curve = curve

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

        # Rail constraint — guided until the rocket clears the launch rod/rail
        rod_len = getattr(s, 'launch_rod_length', 1.0) or 1.0
        on_rail = z < rod_len
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

        # Sideslip (yaw plane): effective sideslip is the angle between the
        # body axis and the relative wind in the yaw plane — body yaw minus
        # the lateral flow angle — mirroring alpha = pitch - vel_angle. Using
        # the flow angle alone left the yaw attitude itself unrestored: a
        # yawed rocket felt no correcting moment until its velocity drifted.
        if v_rel > 0.5:
            flow_beta = math.atan2(vrel_y, math.sqrt(vrel_x**2 + vrel_z**2))
            beta_angle = yaw - flow_beta
            beta_angle = max(-math.radians(45), min(math.radians(45), beta_angle))
        else:
            beta_angle = 0.0

        # Reference geometry. Multistage shrinks with each separation, so take
        # the active-stack diameter/CG from the stage manager; single-stage
        # uses the static design values (identical numbers).
        ms = self.stage_mgr is not None and self.stage_mgr.is_multistage
        ref_diameter = self.stage_mgr.active_diameter() if ms else s.diameter

        # Dynamic pressure
        q_dyn = 0.5 * rho * v_rel**2
        ref_area = math.pi * (ref_diameter / 2)**2

        # Aerodynamic model
        cg = self.stage_mgr.active_cg() if ms else s.cg
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
            # undamped and tumbles in crosswind just like pitch did). Built
            # from CNα and the effective sideslip directly — the pitch cm
            # already contains sin(α), so scaling it by sin(β) made yaw
            # stiffness vanish whenever the pitch AoA was near zero.
            cm_yaw = compute_pitching_moment_coefficient(
                aero.get("cn_total", 2.0), cp, cg, s.diameter, beta_angle)
            M_yaw = cm_yaw * q_dyn * ref_area * s.diameter
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

        # Recovery drag override. Drag now comes from the inflation-aware
        # RecoverySystem (cubic fill curve), so the canopy ramps from zero to
        # full CdA over its inflation time instead of snapping on instantly.
        # Body aero (normal force, moments) is zeroed: under canopy the airframe
        # hangs and no longer weathercocks. Drag still opposes v_rel (which
        # includes wind), so the descent drifts downwind.
        if (self._drogue_deployed or self._main_deployed) and vz < 0 \
                and self.recovery is not None:
            F_drag = self.recovery.get_drag_force(rho, v_rel, t)
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

        # Thrust — from the active (burning) stage. The stage state machine is
        # advanced once per accepted step in _step, so within this RK4 sub-eval
        # the burning stage and ignition time are constant.
        thrust = self.stage_mgr.thrust(t)
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
        # Same CNα·sin(β) form with the 20° stall clamp; acts toward the side
        # the nose points relative to the flow (lift), like the pitch term.
        _deployed = self._drogue_deployed or self._main_deployed
        if v_rel > 0.5 and abs(beta_angle) > 1e-6 and not _deployed:
            cn_total_now = aero.get("cn_total", 2.0) if self.aero_model is not None else 0.0
            eff_beta = min(abs(beta_angle), math.radians(20))
            F_normal_y = q_dyn * ref_area * cn_total_now * math.sin(eff_beta)
            normal_y = math.copysign(F_normal_y, beta_angle)
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
            if ms:
                # Multistage: the canonical vehicle isn't stage-synced, so take
                # pitch inertia from the active stack (drops sharply at
                # separation); roll from a solid-cylinder estimate.
                inertia = self.stage_mgr.pitch_inertia()
                ixx = self._estimate_roll_inertia(mass, s)
            elif self.vehicle is not None:
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

        # Mass flow — Isp is defined against standard g0, not local gravity.
        # Isp and the burnout-mass floor come from the active (burning) stage,
        # so propellant depletes against the correct stage and the floor tracks
        # the active stack (single-stage values are identical to before).
        isp = self.stage_mgr.active_isp()
        burnout_floor = self.stage_mgr.active_burnout_mass()
        if thrust > 0 and (mass - burnout_floor) > 1e-3:
            if isp > 10:
                dm_dt = -thrust / (isp * G_EARTH)
            elif self.stage_mgr.active_burn_time() > 0:
                dm_dt = -self.stage_mgr.active_propellant_mass() \
                    / self.stage_mgr.active_burn_time()
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
                self.stage_mgr.total_mass(),   # mass (all stages stacked)
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
            # Use the ACTIVE-stack geometry/inertia for multistage — the scalar
            # state values describe the whole stacked rocket, which gives the
            # wrong attitude natural frequency and lets the pitch mode go
            # under-resolved (numerical tumble → divergence) after the model is
            # rebuilt for a shorter, lighter stack.
            ms_dt = self.stage_mgr is not None and self.stage_mgr.is_multistage
            ref_d = self.stage_mgr.active_diameter() if ms_dt else s.diameter
            cg_dt = self.stage_mgr.active_cg() if ms_dt else s.cg
            A_ref = math.pi * (ref_d / 2.0) ** 2
            margin = abs(aero.get("cp", s.cp) - cg_dt)
            if ms_dt:
                I_est = self.stage_mgr.pitch_inertia()
            else:
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
            self.engine.log_message.emit(f"Simulation error at T+{t:.2f}s: {e}")
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
        burnout_mass   = self.stage_mgr.active_burnout_mass()
        new_mass       = max(burnout_mass, new_vec[12])

        # Sync propellant depletion to the burning stage (+ canonical vehicle).
        consumed = max(0.0, old_mass - new_mass)
        self.stage_mgr.consume_propellant(consumed)
        if self.vehicle is not None:
            self.vehicle.consume_propellant(consumed)

        # ── Staging state machine ──
        # Advanced once per accepted step. On separation the spent stage's mass
        # leaves the stack — resync the integrator mass and rebuild the aero
        # model for the now-shorter active stack (CP/CG/drag all change).
        for ev, idx in self.stage_mgr.update(t + adaptive_dt):
            if ev == "separation":
                # Snapshot the dropped stage's state so its ballistic descent
                # can be tracked to the ground (range safety).
                dropped_cfg = self.stage_mgr.stages[idx]
                self.spent_stages.append({
                    "stage": idx, "name": dropped_cfg.name,
                    "t": t + adaptive_dt,
                    "x": new_x, "y": new_y, "z": new_vec[2],
                    "vx": new_vx, "vy": new_vy, "vz": new_vz,
                    "mass": dropped_cfg.total_mass(),
                    "length": dropped_cfg.length, "diameter": dropped_cfg.diameter,
                })
                new_mass = self.stage_mgr.total_mass()
                self._state_vec[12] = new_mass
                burnout_mass = self.stage_mgr.active_burnout_mass()
                self.aero_model = AeroModel.from_state(self.stage_mgr.aero_config())
                self.event_mgr.fire(SimEvent.STAGE_SEPARATION, {
                    "time": t + adaptive_dt, "stage": idx,
                    "altitude": new_vec[2], "new_mass": new_mass,
                })
                logger.info(f"Stage {idx} separated at {new_vec[2]:.1f}m, "
                            f"mass now {new_mass:.2f}kg")
            elif ev == "ignition":
                self.event_mgr.fire(SimEvent.STAGE_IGNITION, {
                    "time": t + adaptive_dt, "stage": idx,
                    "altitude": new_vec[2],
                })
                logger.info(f"Stage {idx} ignition at {new_vec[2]:.1f}m")

        new_prop_mass  = max(0.0, new_mass - burnout_mass)

        # Signed velocity magnitude
        new_velocity = math.sqrt(new_vx**2 + new_vy**2 + new_vz**2)
        if new_vz < 0:
            new_velocity = -new_velocity

        # ── Derived quantities ──
        thrust       = self.stage_mgr.thrust(t + adaptive_dt)
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

        # Recovery drag for reporting/structures — mirror the inflation-aware
        # force used in the integrator (_derivatives) so the two stay consistent.
        if (self._drogue_deployed or self._main_deployed) and new_velocity < 0 \
                and self.recovery is not None:
            drag_force = self.recovery.get_drag_force(
                atm_rho, abs(new_velocity), t + adaptive_dt)
            # Opening shock: the inflation transient produces a transient drag
            # spike (peaks near full fill at the highest descent speed). Track
            # the peak for the structural margin on the recovery harness/bulkhead.
            if drag_force > self._recovery_shock_peak:
                self._recovery_shock_peak = drag_force
            self._last_descent_vz = abs(new_vz)

        # Rate of change of SPEED (unsigned) — using the signed pseudo-magnitude
        # here produced a spurious spike at apogee, where the sign convention
        # flips from +|v| to -|v| while the horizontal speed is still finite.
        accel = (abs(new_velocity) - abs(s.velocity)) / adaptive_dt if adaptive_dt > 0 else 0.0

        # ── Phase evaluation ──
        prev_phase = self._phase
        self._phase = self.phase_mgr.evaluate(
            self._phase, t + adaptive_dt, new_altitude, new_velocity,
            thrust, getattr(s, 'main_deploy_altitude', 300.0),
            drogue_delay=getattr(s, 'drogue_deploy_delay', 1.0),
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
                if self.recovery is not None:
                    self.recovery.drogue.deploy(t + adaptive_dt)
                    self.recovery.drogue_deployed = True
                self.event_mgr.fire(SimEvent.DROGUE_DEPLOY, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                })
                logger.info(f"Drogue deployed at {new_altitude:.1f}m")

            if self._phase == FlightPhase.MAIN_DESCENT and not self._main_deployed:
                self._main_deployed = True
                # Descent rate reached under the drogue (touchdown velocity is
                # captured separately at landing). |vz| just before the main fills.
                self._drogue_descent_rate = abs(new_vz)
                if self.recovery is not None:
                    self.recovery.main.deploy(t + adaptive_dt)
                    self.recovery.main_deployed = True
                self.event_mgr.fire(SimEvent.MAIN_DEPLOY, {
                    "time": t + adaptive_dt, "altitude": new_altitude,
                })
                logger.info(f"Main chute deployed at {new_altitude:.1f}m")

        # ── Flight-sanity / divergence guard ──
        # Abort early (before the heavy structural solve) if the flight has gone
        # numerically divergent or lost attitude control, with a clear reason.
        if self._check_flight_sanity(t + adaptive_dt, new_vec, new_altitude,
                                     new_velocity, aero, mach):
            return

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
        # During recovery the dominant axial load is the parachute opening
        # shock (drag_force), reacted through the harness — include it so the
        # descent structural margin reflects the snatch load, not just weight.
        struct_res = compute_all(
            force=max(abs(net_force), abs(thrust),
                      new_mass * gravity_at_altitude(new_altitude),
                      drag_force if (self._drogue_deployed or self._main_deployed) else 0.0),
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

        # Live flutter check: if the fin flutter speed (from the Dynamics
        # workspace) is exceeded in flight, fire a one-shot warning. Only active
        # when a finite flutter speed has been computed and pushed to state.
        f_speed = getattr(s, 'flutter_speed', 0.0)
        if (f_speed > 0 and abs(new_velocity) > f_speed
                and not self._flutter_warned):
            self._flutter_warned = True
            self.event_mgr.fire(SimEvent.FLUTTER_WARNING, {
                "time": t + adaptive_dt, "velocity": abs(new_velocity),
                "flutter_speed": f_speed, "altitude": new_altitude, "mach": mach,
            })
            self.engine.update(flutter_exceeded=True, emit=False)
            self.engine.log_message.emit(
                f"⚠ FLUTTER at T+{t+adaptive_dt:.2f}s — V={abs(new_velocity):.0f} m/s "
                f"exceeds fin flutter speed {f_speed:.0f} m/s (Mach {mach:.2f})")

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
            # Static margin is meaningless past apogee: at near-zero airspeed
            # the AoA pegs at the 45° clamp and drags CP to a garbage position
            # (margin collapses toward zero), and under a canopy the attitude
            # is frozen entirely. Record NaN from apogee onward so plots show
            # a clean gap instead of an artificial collapse.
            stability_margin=(float('nan')
                              if (self._drogue_deployed or self._main_deployed
                                  or self._phase not in (FlightPhase.PRELAUNCH,
                                                         FlightPhase.BOOST,
                                                         FlightPhase.COAST))
                              else aero.get('stab_margin', s.stability_margin)),
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

            # Recovery summary: landing point, drift radius, descent rates,
            # opening shock, descent duration (apogee → touchdown).
            landing_drift = math.sqrt(new_x ** 2 + new_y ** 2)
            touchdown_rate = self._last_descent_vz
            descent_time = max(0.0, (t + adaptive_dt) - self._apogee_time) \
                if self._apogee_time > 0 else 0.0
            self.engine.update(
                sim_running=False, sim_phase=FlightPhase.LANDED.value,
                landing_x=new_x, landing_y=new_y, landing_drift=landing_drift,
                drogue_descent_rate=self._drogue_descent_rate,
                main_descent_rate=touchdown_rate,
                recovery_shock_force=self._recovery_shock_peak,
                descent_time=descent_time,
            )
            self.event_mgr.fire(SimEvent.LANDING, {
                "time": t + dt, "flight_time": t + dt,
                "max_altitude": s.max_altitude,
                "max_velocity": s.max_velocity,
                "max_mach": s.max_mach,
                "landing_x": new_x, "landing_y": new_y,
                "landing_drift": landing_drift,
                "drogue_descent_rate": self._drogue_descent_rate,
                "touchdown_rate": touchdown_rate,
                "recovery_shock_force": self._recovery_shock_peak,
                "descent_time": descent_time,
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
                f"Flight complete — Apogee: {s.max_altitude:.1f}m | "
                f"Max Velocity: {s.max_velocity:.1f}m/s | "
                f"Max Mach: {s.max_mach:.3f} | "
                f"Flight Time: {t+dt:.2f}s | "
                f"Events: {len(self.event_mgr.event_log)}"
            )
            # Spent-stage ballistic tracking: propagate each dropped stage from
            # its separation state to the ground for a landing footprint.
            if self.spent_stages:
                from core.spent_stage import propagate_all
                from core.constants import gravity_at_altitude as _g
                self.spent_stage_results = propagate_all(
                    self.spent_stages, self.atmosphere, self.wind_model, _g)
                for snap, r in zip(self.spent_stages, self.spent_stage_results):
                    self.engine.log_message.emit(
                        f"Spent stage '{snap.get('name','')}' landed "
                        f"{r['drift']:.0f}m from pad @ {r['impact_velocity']:.0f}m/s")

            self.engine.log_message.emit(
                f"Recovery — Touchdown: {touchdown_rate:.1f}m/s | "
                f"Drogue descent: {self._drogue_descent_rate:.1f}m/s | "
                f"Drift: {landing_drift:.0f}m | "
                f"Opening shock: {self._recovery_shock_peak:.0f}N | "
                f"Descent time: {descent_time:.1f}s"
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
