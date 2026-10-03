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
    fin_body_radius: float = 0.0       # tube the fins sit on (0 = stack radius)
    # Further fin sets and diameter changes on this stage (fin_set_fields /
    # transition_fields dicts, positions from this stage's own top)
    extra_fin_sets: list = field(default_factory=list)
    transitions: list = field(default_factory=list)

    # ── Nose (typically only the top stage) ──
    nose_type: str = "ogive"
    nose_length: float = 0.0
    nose_diameter: float = 0.0         # nose base diameter (0 = stack diameter)
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


def aero_nose_type(shape) -> str:
    """Aero-model nose type for a NoseCone.shape ("Haack (LD)" → "ogive")."""
    return _NOSE_SHAPE_MAP.get(str(shape or "ogive").lower(), "ogive")


def _descendants(component):
    """Every descendant of `component`, depth-first, in tree order (the order
    RocketAssembly.all_components walks one stage)."""
    for c in getattr(component, "children", []):
        yield c
        yield from _descendants(c)


def fin_set_fields(fins, origin: float = 0.0) -> dict:
    """One TrapezoidalFinSet as the flat RocketState fin fields.

    These are the keys the state carries for its first fin set and the keys
    of each entry in ``extra_fin_sets``. ``origin`` is subtracted from the
    position: 0 for a position from the nose tip, the stage top for a
    stage-local one.
    """
    parent = getattr(fins, "parent", None)
    return {
        "fin_count": fins.fin_count,
        "fin_root_chord": fins.root_chord,
        "fin_tip_chord": fins.tip_chord,
        "fin_span": fins.height,
        # DEGREES on the component, RADIANS on the state/StageConfig — the aero
        # model does a bare math.tan on this field. Passing it through unchanged
        # made a 30-degree fin reach AeroModel as 30 radians (tan = -6.4, a fin
        # swept forward past its own root chord) on every multistage flight.
        "fin_sweep_angle": math.radians(fins.sweep_angle),
        "fin_thickness": getattr(fins, "thickness", 0.003),
        "fin_cross_section": getattr(fins, "cross_section", "Rounded"),
        "fin_position": max(0.0, fins.position - origin),
        # Radius of the tube the fins are on (0: not on a tube, use the body's)
        "fin_body_radius": (parent.outer_diameter() / 2.0) if parent is not None else 0.0,
    }


def transition_fields(component, origin: float = 0.0):
    """A diameter change as the aero model takes it, or None if `component`
    is not one: a Transition, or a Boat-Tail nozzle (the only nozzle type
    that is an external surface)."""
    from core.components import Nozzle, Transition
    if isinstance(component, Transition):
        fore, aft = component.fore_diameter, component.aft_diameter
    elif isinstance(component, Nozzle) and component.nozzle_type == "Boat-Tail":
        fore, aft = component.inlet_diameter, component.exit_diameter
    else:
        return None
    return {"position": max(0.0, component.position - origin),
            "length": component.length,
            "fore_diameter": fore, "aft_diameter": aft}


def _extract_stage_geometry(stage) -> dict:
    """Pull StageConfig geometry fields from one assembly Stage (UI component)."""
    from core.components import NoseCone, TrapezoidalFinSet

    length = stage.component_length()
    diameter = stage.outer_diameter()
    dry_mass = stage.total_mass()          # airframe only — motors live separately

    geom = dict(length=length, diameter=diameter, dry_mass=dry_mass)

    nose = _find_child(stage, NoseCone)
    if nose is not None:
        geom["nose_type"] = aero_nose_type(getattr(nose, "shape", "ogive"))
        geom["nose_length"] = nose.component_length()
        geom["nose_diameter"] = nose.outer_diameter()

    # fin_position is stage-LOCAL (from the stage's own top): assembly
    # positions are absolute from the nose tip, so subtract the stage top.
    # The first fin set fills the flat fin_* fields; any others follow whole.
    fin_sets = [c for c in _descendants(stage) if isinstance(c, TrapezoidalFinSet)]
    if fin_sets:
        geom.update(fin_set_fields(fin_sets[0], stage.position))
        geom["extra_fin_sets"] = [fin_set_fields(f, stage.position)
                                  for f in fin_sets[1:]]
    geom["transitions"] = [t for t in (transition_fields(c, stage.position)
                                       for c in _descendants(stage)) if t]
    return geom


# Flat RocketState fields that describe the nose and the fin set, with the
# value each takes when the assembly has no such component.
_AERO_GEOMETRY_DEFAULTS = {
    "nose_type": "ogive", "nose_length": 0.0, "nose_diameter": 0.0,
    "fin_span": 0.0, "fin_height": 0.0,
    "fin_root_chord": 0.0, "fin_tip_chord": 0.0,
    "fin_sweep_angle": 0.0, "fin_thickness": 0.003,
    "fin_cross_section": "Rounded", "fin_body_radius": 0.0,
}


def assembly_aero_geometry(assembly) -> dict:
    """Nose and fin dimensions of a UI RocketAssembly as flat RocketState fields.

    AeroModel.from_state reads these flat fields and nothing else, and where
    one is zero it substitutes a generic fin sized from the body. So whoever
    copies an assembly into the state has to copy these too, or the flight
    sim, Monte Carlo and the optimizer all fly a rocket the user never drew.

    The nose comes from the first stage that has one and the fins from the
    first fin set in the assembly, the same one RocketAssembly.fin_count()
    reports. Every key is always present, so a field left over from a previous
    design cannot survive a sync. fin_sweep_angle is in RADIANS.

    ``extra_fin_sets`` holds every further fin set and ``transitions`` every
    diameter change, positions from the nose tip. RocketAssembly.compute_cp
    counts all of them, so the sim has to be told about all of them: with
    only the first fin set a canard, a booster's fins or a boat-tail moved
    the Design tab's CP and left the flight's where it was.
    """
    from core.components import TrapezoidalFinSet

    # A nose or fin tube as wide as the body is written as 0 ("the body's"),
    # so it follows the body diameter when that is edited on the state (the
    # optimizer's diameter variable) instead of keeping the drawn radius.
    body = assembly.max_diameter() if hasattr(assembly, "max_diameter") else 0.0

    def _unless_body(value, body_value):
        return 0.0 if abs(value - body_value) <= 1e-12 else value

    geom = dict(_AERO_GEOMETRY_DEFAULTS)
    geom["extra_fin_sets"], geom["transitions"] = [], []
    have_nose = have_fins = False
    for stage in getattr(assembly, "stages", []) or []:
        stage_geom = _extract_stage_geometry(stage)
        if not have_nose and "nose_length" in stage_geom:
            have_nose = True
            for key in ("nose_type", "nose_length", "nose_diameter"):
                geom[key] = stage_geom[key]
            geom["nose_diameter"] = _unless_body(geom["nose_diameter"], body)
        for fins in (c for c in _descendants(stage) if isinstance(c, TrapezoidalFinSet)):
            fields = fin_set_fields(fins)
            fields["fin_body_radius"] = _unless_body(fields["fin_body_radius"], body / 2.0)
            if have_fins:
                geom["extra_fin_sets"].append(fields)
                continue
            have_fins = True
            for key in ("fin_span", "fin_root_chord", "fin_tip_chord",
                        "fin_sweep_angle", "fin_thickness", "fin_cross_section",
                        "fin_body_radius"):
                geom[key] = fields[key]
            geom["fin_height"] = fields["fin_span"]
        geom["transitions"] += [t for t in (transition_fields(c)
                                            for c in _descendants(stage)) if t]
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
                 "nose_diameter", "fin_count", "fin_span", "fin_root_chord",
                 "fin_tip_chord", "fin_sweep_angle", "fin_thickness",
                 "fin_position", "fin_cross_section", "fin_body_radius",
                 "extra_fin_sets", "transitions", "surface_finish", "cmq")

    _FIN_KEYS = ("fin_count", "fin_span", "fin_root_chord", "fin_tip_chord",
                 "fin_sweep_angle", "fin_thickness", "fin_cross_section",
                 "fin_position", "fin_body_radius")

    def __init__(self, active: list[StageConfig]):
        # Physical top→bottom is the reverse of ignition order.
        top = active[-1]      # carries the nose
        bottom = active[0]    # carries the aft fins (stability) + is burning
        self.length = sum(s.length for s in active) or bottom.length or 1.0
        self.diameter = max((s.diameter for s in active), default=0.0) \
            or bottom.diameter
        self.nose_type = top.nose_type
        self.nose_length = top.nose_length or self.length * 0.2
        self.nose_diameter = top.nose_diameter
        # The stack's fins are the aft-most fin set it carries: the bottom
        # (burning) stage's, or the next stage up when the bottom one has
        # none. AeroModel no longer invents fins for a finless state, so a
        # finless booster must not hide the fins of the stage above it.
        idx = next((i for i, st in enumerate(active)
                    if st.fin_count > 0 and st.fin_span > 0
                    and st.fin_root_chord > 0), None)
        finned = active[idx] if idx is not None else bottom
        self.fin_count = finned.fin_count if idx is not None else 0
        self.fin_span = finned.fin_span
        self.fin_root_chord = finned.fin_root_chord
        self.fin_tip_chord = finned.fin_tip_chord
        self.fin_sweep_angle = finned.fin_sweep_angle
        self.fin_thickness = finned.fin_thickness or 0.003
        self.fin_cross_section = finned.fin_cross_section
        self.fin_body_radius = finned.fin_body_radius
        self.surface_finish = top.surface_finish
        # Stage-local fin position → from the nose: everything stacked above
        # that stage (later in ignition order) comes first.
        above = sum(st.length for st in active[(idx or 0) + 1:])
        self.fin_position = max(0.0, above + finned.fin_position)
        self.cmq = -20.0

        # Everything else the stack carries: the other stages' fin sets, each
        # stage's further fin sets and its diameter changes, moved from
        # stage-local positions to positions from the nose. The sustainer's
        # fins fly with the booster attached; without them the sim's CP sat
        # where the booster's fins alone put it.
        self.extra_fin_sets, self.transitions = [], []
        for i, st in enumerate(active):
            top_of_stage = sum(s.length for s in active[i + 1:])
            sets = list(st.extra_fin_sets or [])
            if i != idx and st.fin_count > 0 and st.fin_span > 0 \
                    and st.fin_root_chord > 0:
                sets.insert(0, {k: getattr(st, k) for k in self._FIN_KEYS})
            for f in sets:
                f = dict(f)
                f["fin_position"] = max(0.0, top_of_stage + f.get("fin_position", 0.0))
                self.extra_fin_sets.append(f)
            for tr in st.transitions or []:
                tr = dict(tr)
                tr["position"] = top_of_stage + tr.get("position", 0.0)
                self.transitions.append(tr)


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
            fin_body_radius=getattr(s, 'fin_body_radius', 0.0),
            extra_fin_sets=[dict(f) for f in getattr(s, 'extra_fin_sets', None) or []],
            transitions=[dict(t) for t in getattr(s, 'transitions', None) or []],
            nose_type=getattr(s, 'nose_type', 'ogive'),
            nose_length=getattr(s, 'nose_length', 0.0),
            nose_diameter=getattr(s, 'nose_diameter', 0.0),
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
