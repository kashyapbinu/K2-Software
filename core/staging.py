"""
K2 AeroSim — Multistage Manager
==================================
Drives staged flight: per-stage motors, burnout → separation → next-stage
ignition, mass drop, and a rebuilt aerodynamic config for the active stack.

Stage ordering
--------------
`StageManager.stages` is held in IGNITION ORDER: index 0 is the first motor
lit (the bottom booster), the last index is the sustainer/upper stage that
carries the nose. Physically the stack runs nose(top)→tail(bottom) = the
*reverse* of ignition order.

`bottom_index` is the bottom of the currently-attached stack. Spent stages
that have separated are `stages[:bottom_index]` — they have fallen away from
the tail, so the attached stack is always `stages[bottom_index:]` and its
nose stays at x=0.

Staging timeline for a stage that burns out with another stage above it:
    ignition → burn → burnout → (separation_delay) → SEPARATION (mass drops)
             → (next stage ignition_delay) → next ignition → …

Single-stage back-compat
------------------------
`StageManager.from_state(s)` wraps a one-motor RocketState as a single stage,
so the manager reproduces today's single-body flight exactly (no separation,
thrust/mass identical to the legacy scalar path).
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("K2.Staging")

G_EARTH = 9.80665


@dataclass
class StageConfig:
    """Geometry + motor for one stage (ignition-order element)."""
    name: str = "Stage"

    # ── Motor ──
    motor_designation: str = "None"
    motor_avg_thrust: float = 0.0
    motor_max_thrust: float = 0.0
    motor_total_impulse: float = 0.0
    motor_burn_time: float = 0.0
    motor_isp: float = 0.0
    motor_dry_mass: float = 0.0        # casing mass, survives burnout
    propellant_mass: float = 0.0       # initial propellant
    motor_length: float = 0.3
    motor_diameter: float = 0.038
    custom_thrust_curve: list = field(default_factory=list)

    # ── Structure (this stage only) ──
    dry_mass: float = 0.0              # airframe dry mass (excl. motor casing)
    length: float = 0.0
    diameter: float = 0.0

    # ── Fins on this stage ──
    fin_count: int = 0
    fin_span: float = 0.0
    fin_root_chord: float = 0.0
    fin_tip_chord: float = 0.0
    fin_sweep_angle: float = 0.0
    fin_thickness: float = 0.003
    fin_position: float = 0.0          # from this stage's own top
    fin_cross_section: str = "Rounded"

    # ── Nose (typically only the top stage) ──
    nose_type: str = "ogive"
    nose_length: float = 0.0
    surface_finish: str = "Normal"

    # ── Staging ──
    separation_delay: float = 0.0      # burnout → physical separation (s)
    ignition_delay: float = 0.0        # separation → this stage's ignition (s)

    # ── Runtime (mutated during sim) ──
    current_propellant_mass: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self.current_propellant_mass = self.propellant_mass

    @classmethod
    def from_dict(cls, d: dict) -> "StageConfig":
        """Build from a plain dict, ignoring unknown keys (forward-compatible)."""
        valid = {f for f in cls.__dataclass_fields__ if f != "current_propellant_mass"}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def total_mass(self) -> float:
        """Dry airframe + motor casing + remaining propellant."""
        return self.dry_mass + self.motor_dry_mass + self.current_propellant_mass


def build_thrust_curve(cfg: StageConfig) -> list:
    """Trapezoidal thrust curve normalized to true total impulse (avg×burn).

    Mirrors SimulationEngine._build_thrust_curve so a single-stage manager
    produces the identical curve to the legacy scalar path.
    """
    if cfg.custom_thrust_curve:
        return sorted((float(t), float(f)) for t, f in cfg.custom_thrust_curve)

    bt = cfg.motor_burn_time
    avg = cfg.motor_avg_thrust
    mx = cfg.motor_max_thrust if cfg.motor_max_thrust > 0 else avg * 1.4
    if bt <= 0:
        return []

    ramp = bt * 0.1
    curve = [(0.0, 0.0), (ramp, mx), (bt - ramp, avg), (bt, 0.0)]
    impulse = sum(
        0.5 * (curve[i][1] + curve[i + 1][1]) * (curve[i + 1][0] - curve[i][0])
        for i in range(len(curve) - 1)
    )
    target = avg * bt
    if impulse > 0 and target > 0:
        scale = target / impulse
        curve = [(t, f * scale) for t, f in curve]
    return curve


def _find_child(component, cls):
    """Depth-first search for the first descendant of type `cls`."""
    for c in getattr(component, "children", []):
        if isinstance(c, cls):
            return c
        found = _find_child(c, cls)
        if found is not None:
            return found
    return None


_NOSE_SHAPE_MAP = {
    "ogive": "ogive", "haack (ld)": "ogive", "haack": "ogive",
    "conical": "conical", "cone": "conical",
    "elliptical": "elliptical", "ellipsoid": "elliptical",
    "parabolic": "parabolic",
}


def _extract_stage_geometry(stage) -> dict:
    """Pull StageConfig geometry fields from one assembly Stage (UI component)."""
    from core.components import NoseCone, TrapezoidalFinSet

    length = stage.component_length()
    diameter = stage.outer_diameter()
    dry_mass = stage.total_mass()          # airframe only — motors live separately

    geom = dict(length=length, diameter=diameter, dry_mass=dry_mass)

    nose = _find_child(stage, NoseCone)
    if nose is not None:
        geom["nose_type"] = _NOSE_SHAPE_MAP.get(
            str(getattr(nose, "shape", "ogive")).lower(), "ogive")
        geom["nose_length"] = nose.component_length()

    fins = _find_child(stage, TrapezoidalFinSet)
    if fins is not None:
        geom["fin_count"] = fins.fin_count
        geom["fin_root_chord"] = fins.root_chord
        geom["fin_tip_chord"] = fins.tip_chord
        geom["fin_span"] = fins.height
        geom["fin_sweep_angle"] = fins.sweep_angle
        geom["fin_thickness"] = getattr(fins, "thickness", 0.003)
        # fin_position is stage-LOCAL (from the stage's own top): assembly
        # positions are absolute from the nose tip, so subtract the stage top.
        geom["fin_position"] = max(0.0, fins.position - stage.position)
    return geom


def build_stages_config(assembly) -> list:
    """Build the ignition-order stages_config list from a UI RocketAssembly.

    Assembly stages are stacked nose(top)→tail(bottom) in `.stages` order, so
    ignition order (bottom booster first) is the reverse. Each stage's motor +
    separation/ignition delays come from the UI Stage component.

    Returns [] when there are fewer than 2 stages OR no stage has a motor —
    the caller then uses the single-stage scalar path unchanged.
    """
    stages = list(getattr(assembly, "stages", []) or [])
    if len(stages) < 2:
        return []

    ignition_order = list(reversed(stages))
    if not any(getattr(st, "motor", None) for st in ignition_order):
        return []

    configs = []
    for st in ignition_order:
        motor = dict(getattr(st, "motor", None) or {})
        cfg = dict(name=st.name)
        cfg.update(_extract_stage_geometry(st))
        cfg.update(motor)                                  # motor_* keys
        cfg["separation_delay"] = getattr(st, "separation_delay", 0.0)
        cfg["ignition_delay"] = getattr(st, "ignition_delay", 0.0)
        configs.append(cfg)
    return configs


class _StackAeroConfig:
    """Duck-typed state-like object for AeroModel.from_state(active stack).

    Exposes exactly the attributes AeroModel.from_state reads, computed for
    the currently-attached stack rather than the whole vehicle.
    """
    __slots__ = ("length", "diameter", "nose_type", "nose_length",
                 "fin_count", "fin_span", "fin_root_chord", "fin_tip_chord",
                 "fin_sweep_angle", "fin_thickness", "fin_position",
                 "fin_cross_section", "surface_finish", "cmq")

    def __init__(self, active: list[StageConfig]):
        # Physical top→bottom is the reverse of ignition order.
        top = active[-1]      # carries the nose
        bottom = active[0]    # carries the aft fins (stability) + is burning
        self.length = sum(s.length for s in active) or bottom.length or 1.0
        self.diameter = max((s.diameter for s in active), default=0.0) \
            or bottom.diameter
        self.nose_type = top.nose_type
        self.nose_length = top.nose_length or self.length * 0.2
        # Aft fins belong to the bottom (burning) stage; place them on its
        # segment near the tail of the stack.
        self.fin_count = bottom.fin_count
        self.fin_span = bottom.fin_span
        self.fin_root_chord = bottom.fin_root_chord
        self.fin_tip_chord = bottom.fin_tip_chord
        self.fin_sweep_angle = bottom.fin_sweep_angle
        self.fin_thickness = bottom.fin_thickness or 0.003
        self.fin_cross_section = bottom.fin_cross_section
        self.surface_finish = top.surface_finish
        self.fin_position = max(0.0, self.length - bottom.length
                                + bottom.fin_position)
        self.cmq = -20.0


class StageManager:
    """State machine + data provider for staged flight."""

    def __init__(self, stages: list[StageConfig]):
        if not stages:
            raise ValueError("StageManager needs at least one stage")
        self.stages = stages
        self._curves = [build_thrust_curve(s) for s in stages]
        self.reset()

    # ── back-compat constructor ──────────────────────────────────────
    @classmethod
    def from_state(cls, s) -> "StageManager":
        """Wrap a single-motor RocketState as one stage (legacy behavior).

        Mass split mirrors RocketState.total_mass() exactly
        (dry_mass + motor_dry_mass + propellant) so single-stage flights are
        numerically identical to the legacy scalar path.
        """
        prop = getattr(s, 'propellant_mass_initial', 0.0)
        motor_dry = getattr(s, 'motor_dry_mass', 0.0)
        cfg = StageConfig(
            name=getattr(s, 'name', 'Stage'),
            motor_designation=getattr(s, 'motor_designation', 'None'),
            motor_avg_thrust=getattr(s, 'motor_avg_thrust', 0.0),
            motor_max_thrust=getattr(s, 'motor_max_thrust', 0.0),
            motor_total_impulse=getattr(s, 'motor_total_impulse', 0.0),
            motor_burn_time=getattr(s, 'motor_burn_time', 0.0),
            motor_isp=getattr(s, 'motor_isp', 0.0),
            motor_dry_mass=motor_dry,
            propellant_mass=prop,
            motor_length=getattr(s, 'motor_length', 0.0) or 0.3,
            motor_diameter=getattr(s, 'diameter', 0.038),
            custom_thrust_curve=list(getattr(s, 'custom_thrust_curve', []) or []),
            dry_mass=getattr(s, 'dry_mass', 0.0),
            length=getattr(s, 'length', 0.0),
            diameter=getattr(s, 'diameter', 0.0),
            fin_count=getattr(s, 'fin_count', 0),
            fin_span=getattr(s, 'fin_span', 0.0),
            fin_root_chord=getattr(s, 'fin_root_chord', 0.0),
            fin_tip_chord=getattr(s, 'fin_tip_chord', 0.0),
            fin_sweep_angle=getattr(s, 'fin_sweep_angle', 0.0),
            fin_thickness=getattr(s, 'fin_thickness', 0.003),
            fin_position=getattr(s, 'fin_position', 0.0),
            fin_cross_section=getattr(s, 'fin_cross_section', 'Rounded'),
            nose_type=getattr(s, 'nose_type', 'ogive'),
            nose_length=getattr(s, 'nose_length', 0.0),
            surface_finish=getattr(s, 'surface_finish', 'Normal'),
        )
        return cls([cfg])

    # ── lifecycle ────────────────────────────────────────────────────
    def reset(self):
        for s in self.stages:
            s.current_propellant_mass = s.propellant_mass
        self.bottom_index = 0
        self.is_burning = True          # first stage lit at t=0
        self.ign_time = 0.0
        self._burnout_time = None
        self._awaiting_sep = False
        self._awaiting_ign = False
        self._sep_time = 0.0
        self._ign_time_pending = 0.0

    @property
    def is_multistage(self) -> bool:
        return len(self.stages) > 1

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    # ── state machine ────────────────────────────────────────────────
    def update(self, t: float) -> list[tuple]:
        """Advance the staging state machine to time t.

        Returns a list of events that fired this call, each a tuple:
            ("burnout", stage_index)
            ("separation", dropped_stage_index)
            ("ignition", stage_index)
        """
        events: list[tuple] = []

        if self.is_burning:
            bstage = self.stages[self.bottom_index]
            if (t - self.ign_time) >= bstage.motor_burn_time or \
                    bstage.current_propellant_mass <= 0:
                self.is_burning = False
                self._burnout_time = t
                events.append(("burnout", self.bottom_index))
                if self.bottom_index < len(self.stages) - 1:
                    self._sep_time = t + bstage.separation_delay
                    self._awaiting_sep = True
            return events

        # coasting (not burning)
        if self._awaiting_sep and t >= self._sep_time:
            dropped = self.bottom_index
            self.bottom_index += 1            # mass drops here
            self._awaiting_sep = False
            self._awaiting_ign = True
            self._ign_time_pending = (
                self._sep_time + self.stages[self.bottom_index].ignition_delay)
            events.append(("separation", dropped))

        if self._awaiting_ign and t >= self._ign_time_pending:
            self.is_burning = True
            self.ign_time = self._ign_time_pending
            self._awaiting_ign = False
            events.append(("ignition", self.bottom_index))

        return events

    @property
    def staging_complete(self) -> bool:
        """True once the final stage is lit (or burning) — no more stages."""
        return self.bottom_index >= len(self.stages) - 1

    # ── physics queries ──────────────────────────────────────────────
    def thrust(self, t: float) -> float:
        """Thrust of the active (burning) stage at absolute time t."""
        if not self.is_burning:
            return 0.0
        stage = self.stages[self.bottom_index]
        if stage.current_propellant_mass <= 0:
            return 0.0
        curve = self._curves[self.bottom_index]
        if not curve:
            return 0.0
        local = t - self.ign_time
        if local < 0 or local >= curve[-1][0]:
            return 0.0
        for i in range(len(curve) - 1):
            t0, f0 = curve[i]
            t1, f1 = curve[i + 1]
            if t0 <= local <= t1:
                frac = (local - t0) / (t1 - t0) if (t1 - t0) > 1e-12 else 0.0
                return max(0.0, f0 + frac * (f1 - f0))
        return 0.0

    def active_stages(self) -> list[StageConfig]:
        return self.stages[self.bottom_index:]

    def total_mass(self) -> float:
        return sum(s.total_mass() for s in self.active_stages())

    def active_propellant_mass(self) -> float:
        """Remaining propellant of the currently-burning stage."""
        return self.stages[self.bottom_index].current_propellant_mass

    def active_burnout_mass(self) -> float:
        """Mass of the attached stack with the burning stage's propellant gone."""
        return self.total_mass() - self.active_propellant_mass()

    def consume_propellant(self, amount: float):
        """Deplete propellant from the burning stage (clamped at zero)."""
        stage = self.stages[self.bottom_index]
        stage.current_propellant_mass = max(
            0.0, stage.current_propellant_mass - amount)

    def active_isp(self) -> float:
        """Isp of the burning stage; derive from impulse if not given."""
        stage = self.stages[self.bottom_index]
        if stage.motor_isp > 0:
            return stage.motor_isp
        if stage.motor_total_impulse > 0 and stage.propellant_mass > 0:
            return stage.motor_total_impulse / (stage.propellant_mass * G_EARTH)
        return 0.0

    def active_burn_time(self) -> float:
        return self.stages[self.bottom_index].motor_burn_time

    def aero_config(self):
        """state-like config object for AeroModel.from_state of active stack."""
        return _StackAeroConfig(self.active_stages())

    def active_length(self) -> float:
        return sum(s.length for s in self.active_stages())

    def active_diameter(self) -> float:
        a = self.active_stages()
        return max((s.diameter for s in a), default=0.0)

    def active_cg(self) -> float:
        """Mass-weighted CG of the active stack, measured from the nose (x=0).

        Physical order is top→bottom = reversed ignition order. Each stage's
        local CG: airframe at mid-length, motor (casing+prop) biased aft.
        """
        active = self.active_stages()
        phys = list(reversed(active))   # top → bottom
        x_top = 0.0
        moment = 0.0
        mass = 0.0
        for st in phys:
            L = st.length or 0.0
            m_dry = st.dry_mass
            m_motor = st.motor_dry_mass + st.current_propellant_mass
            cg_dry = x_top + L * 0.5
            cg_motor = x_top + L * 0.85          # motor sits aft
            moment += m_dry * cg_dry + m_motor * cg_motor
            mass += m_dry + m_motor
            x_top += L
        return moment / mass if mass > 0 else 0.0

    def pitch_inertia(self) -> float:
        """Pitch inertia of the active stack about its CG (thin-rod + parallel
        axis per stage). Coarse but tracks the large drop at separation."""
        active = self.active_stages()
        phys = list(reversed(active))
        cg = self.active_cg()
        x_top = 0.0
        iyy = 0.0
        for st in phys:
            L = st.length or 0.0
            m = st.total_mass()
            stage_cg = x_top + L * 0.5
            iyy += m * L ** 2 / 12.0 + m * (stage_cg - cg) ** 2
            x_top += L
        return max(iyy, 1e-6)
