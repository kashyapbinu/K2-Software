"""
K2 Aerospace — Centralized Rocket State Engine
================================================
The RocketState is the single source of truth for the entire simulation.
Every module (propulsion, aero, structures, trajectory, visualization)
reads from and writes to this centralized state object.

Architecture:
    UI fields → RocketStateEngine.update() → state_changed signal
    → All workspace panels refresh
    → 3D viewer refreshes
    → Console logs changes
"""

from dataclasses import dataclass, field, asdict
from PyQt6.QtCore import QObject, pyqtSignal
import logging

logger = logging.getLogger("K2.RocketState")


@dataclass
class RocketState:
    """
    Complete rocket state snapshot. All values in SI units.
    """
    # ── Identity ──────────────────────────────────────────────────
    name: str = "Untitled Rocket"

    # ── Geometry ──────────────────────────────────────────────────
    length: float = 0.0
    diameter: float = 0.0
    nose_length: float = 0.0
    fin_height: float = 0.0
    fin_root_chord: float = 0.0
    fin_tip_chord: float = 0.0
    fin_count: int = 0
    nose_type: str = "ogive"
    fin_span: float = 0.0
    fin_sweep_angle: float = 0.0
    fin_thickness: float = 0.003
    fin_position: float = 0.0   # Distance from nose tip to fin root leading edge
    surface_finish: str = "Normal"
    fin_cross_section: str = "Rounded"

    # ── Mass ──────────────────────────────────────────────────────
    dry_mass: float = 0.0
    propellant_mass: float = 0.0
    propellant_mass_initial: float = 0.0

    # ── Center of Gravity / Pressure ──────────────────────────────
    cg: float = 0.0
    dry_cg: float = 0.0
    cp: float = 0.0
    motor_position: float = 0.0

    # ── Dynamics (6DOF) ───────────────────────────────────────────
    x_position: float = 0.0
    y_position: float = 0.0     # lateral drift
    altitude: float = 0.0
    velocity_x: float = 0.0
    velocity_y: float = 0.0     # lateral velocity
    velocity_z: float = 0.0
    velocity: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0            # yaw angle (rad)
    roll: float = 0.0           # roll angle (rad)
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0       # yaw rate (rad/s)
    roll_rate: float = 0.0      # roll rate (rad/s)
    acceleration: float = 0.0
    mach_number: float = 0.0
    max_altitude: float = 0.0
    max_velocity: float = 0.0
    max_acceleration: float = 0.0
    max_mach: float = 0.0

    # ── Forces ────────────────────────────────────────────────────
    thrust: float = 0.0
    drag: float = 0.0
    weight: float = 0.0
    net_force: float = 0.0

    # ── Structural ────────────────────────────────────────────────
    wall_thickness: float = 0.002
    material_name: str = "Aluminum 6061-T6"
    yield_strength: float = 276e6
    elastic_modulus: float = 68.9e9
    material_density: float = 2700.0
    axial_stress: float = 0.0
    hoop_stress: float = 0.0
    buckling_stress: float = 0.0
    max_stress: float = 0.0
    max_strain: float = 0.0
    safety_factor: float = 0.0
    von_mises_stress: float = 0.0
    shear_stress: float = 0.0
    bending_stress: float = 0.0
    thermal_stress: float = 0.0
    wall_temperature: float = 293.15
    margin_of_safety: float = 0.0
    yield_utilization: float = 0.0

    # ── Dynamics / Flutter ────────────────────────────────────────
    flutter_speed: float = 0.0
    flutter_margin: float = 0.0
    divergence_speed: float = 0.0
    modal_freq_1: float = 0.0
    modal_freq_2: float = 0.0
    modal_freq_3: float = 0.0

    # ── Stability ─────────────────────────────────────────────────
    stability_margin: float = 0.0
    cd: float = 0.45

    # ── Simulation ────────────────────────────────────────────────
    sim_time: float = 0.0
    sim_running: bool = False
    sim_paused: bool = False
    sim_phase: str = "Pre-Launch"
    sim_dt: float = 0.01
    sim_speed: float = 1.0
    integrator_name: str = "rk4"

    # ── Environment ───────────────────────────────────────────────
    launch_angle: float = 90.0
    wind_speed: float = 0.0
    temperature_ambient: float = 288.15

    # ── Motor ─────────────────────────────────────────────────────
    motor_designation: str = "None"
    motor_avg_thrust: float = 0.0
    motor_max_thrust: float = 0.0
    motor_total_impulse: float = 0.0
    motor_burn_time: float = 0.0
    motor_isp: float = 0.0
    motor_mass_flow: float = 0.0
    motor_chamber_pressure: float = 0.0
    custom_thrust_curve: list = field(default_factory=list)

    # ── Avionics ──────────────────────────────────────────────────
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    parachute_deployed: bool = False
    flight_computer_state: str = "IDLE"

    # ── Recovery Configuration ────────────────────────────────────
    drogue_deploy_delay: float = 1.0      # seconds after apogee
    main_deploy_altitude: float = 300.0   # meters AGL
    drogue_cd_area: float = 0.5           # Cd × A for drogue (m²)
    main_cd_area: float = 3.0             # Cd × A for main chute (m²)

    # ── Atmosphere Snapshot (updated each tick) ───────────────────
    atm_temperature: float = 288.15
    atm_pressure: float = 101325.0
    atm_density: float = 1.225
    dynamic_pressure: float = 0.0

    # ── CFD Results (injected from CFD workspace) ─────────────────
    cfd_cd: float = 0.0
    cfd_cl: float = 0.0
    cfd_cm: float = 0.0
    cfd_cp_location: float = 0.0
    cfd_converged: bool = False
    cfd_dynamic_pressure: float = 0.0
    cfd_force_axial: float = 0.0
    cfd_force_normal: float = 0.0
    cfd_mach: float = 0.0
    cfd_reynolds: float = 0.0

    def total_mass(self) -> float:
        # Prefer the live propellant_mass, but fall back to the initial load when
        # it has not been synced yet (e.g. a headless caller that only set
        # propellant_mass_initial before the sim engine populated propellant_mass).
        prop = self.propellant_mass if self.propellant_mass > 0 else self.propellant_mass_initial
        return self.dry_mass + prop

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove non-serializable / transient fields
        for key in ['sim_running', 'sim_paused']:
            d.pop(key, None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "RocketState":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class RocketStateEngine(QObject):
    """
    Reactive wrapper around RocketState.
    Emits Qt signals whenever the state changes.
    """

    state_changed = pyqtSignal(object)
    log_message = pyqtSignal(str)
    telemetry_tick = pyqtSignal(object)

    def __init__(self, state: RocketState = None):
        super().__init__()
        self._state = state or RocketState()
        self.auto_estimate_properties = True
        self._recompute_derived()

    @property
    def state(self) -> RocketState:
        return self._state

    def update(self, emit=True, **kwargs) -> None:
        changed = []
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                old_value = getattr(self._state, key)
                if old_value != value:
                    setattr(self._state, key, value)
                    changed.append(key)
            else:
                logger.warning(f"Unknown state property: {key}")

        if changed:
            self._recompute_derived()
            if emit:
                logger.info(f"State updated: {', '.join(changed)}")
                self.log_message.emit(f"State updated: {', '.join(changed)}")
                self.state_changed.emit(self._state)

    def update_sim_tick(self, **kwargs) -> None:
        """High-frequency update during simulation — emits telemetry_tick instead of state_changed."""
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)
        self._recompute_derived()
        self.telemetry_tick.emit(self._state)

    def set_state(self, state: RocketState) -> None:
        self._state = state
        self._recompute_derived()
        self.log_message.emit("Project state loaded")
        self.state_changed.emit(self._state)

    def reset(self) -> None:
        self._state = RocketState()
        self._recompute_derived()
        self.log_message.emit("State reset to defaults")
        self.state_changed.emit(self._state)

    def _recompute_derived(self) -> None:
        s = self._state

        # ── Dynamic CG calculation ──
        # Even with high-fidelity assembly (auto_estimate_properties=False),
        # the propellant mass changes, so total CG must be updated.
        total_m = s.dry_mass + s.propellant_mass
        if total_m > 0:
            s.cg = (s.dry_mass * s.dry_cg + s.propellant_mass * s.motor_position) / total_m

        # Skip estimations if we have a high-fidelity assembly active
        if not self.auto_estimate_properties:
            if s.diameter > 0:
                s.stability_margin = (s.cp - s.cg) / s.diameter
            from core.constants import G_EARTH
            s.weight = total_m * G_EARTH
            s.net_force = s.thrust - s.drag - s.weight
            self._track_maxima(s)
            return

        # ── Geometry derived ──
        s.nose_length = s.length * 0.2
        s.fin_root_chord = s.length * 0.1
        s.fin_tip_chord = s.fin_root_chord * 0.5
        s.fin_height = s.diameter * 0.6
        # Keep fin_span in sync (used by AeroModel)
        if s.fin_span <= 0:
            s.fin_span = s.fin_height

        # ── CG estimation ──
        body_cg = s.length * 0.45
        motor_cg = s.length * 0.85
        total_mass = s.total_mass()
        if total_mass > 0:
            s.cg = (s.dry_mass * body_cg + s.propellant_mass * motor_cg) / total_mass

        # ── CP and Stability estimation (via AeroModel) ──
        try:
            from physics.aerodynamics import AeroModel
            aero = AeroModel.from_state(s)
            s.cp = aero.cp_subsonic()
            if s.diameter > 0:
                s.stability_margin = (s.cp - s.cg) / s.diameter
        except Exception:
            # Fallback if AeroModel fails to init
            nose_cp = 0.466 * s.nose_length
            cn_nose = 2.0
            fin_cp_from_base = s.fin_root_chord * 0.25
            fin_cp = max(0, s.length - fin_cp_from_base)
            cn_fin = s.fin_count * 2.0
            cn_total = cn_nose + cn_fin
            if cn_total > 0:
                s.cp = (cn_nose * nose_cp + cn_fin * fin_cp) / cn_total
            if s.diameter > 0:
                s.stability_margin = (s.cp - s.cg) / s.diameter

        # ── Weight ──
        from core.constants import G_EARTH
        s.weight = total_mass * G_EARTH

        # ── Net force ──
        s.net_force = s.thrust - s.drag - s.weight

        self._track_maxima(s)

    def _track_maxima(self, s):
        # ── Track maxima ──
        if s.altitude > s.max_altitude:
            s.max_altitude = s.altitude
        if abs(s.velocity) > s.max_velocity:
            s.max_velocity = abs(s.velocity)
        if abs(s.acceleration) > s.max_acceleration:
            s.max_acceleration = abs(s.acceleration)
        if s.mach_number > s.max_mach:
            s.max_mach = s.mach_number
