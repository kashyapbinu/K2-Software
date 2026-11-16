"""
K2 AeroSim — Parachute Dynamics
====================================
High-fidelity recovery system drag calculations including
inflation transients, steady-state drag, and failure modes.
"""
import math
import logging

logger = logging.getLogger("K2.Recovery")

G0 = 9.80665


class ParachuteDynamics:
    """
    Models inflation transient and steady-state descent drag for a parachute.
    """

    def __init__(self, cd: float, diameter: float,
                 inflation_time: float = 0.5,
                 fill_factor: float = 1.0):
        self.cd             = cd
        self.diameter       = diameter
        self.area           = math.pi * (diameter / 2) ** 2
        self.cd_area_nominal = cd * self.area * fill_factor
        self.inflation_time = max(inflation_time, 0.01)
        self.deployed       = False
        self._deploy_time   = None

    @classmethod
    def from_cd_area(cls, cd_area: float, inflation_time: float = 0.5):
        """Build directly from a Cd×A product (m²).

        The K2 UI stores recovery sizing as a single drag area (Cd×A) rather
        than separate Cd + diameter, so reconstruct an equivalent canopy with
        cd=1.0 and a diameter that yields the requested area. Drag depends only
        on cd_area_nominal, so the cd/diameter split is cosmetic here.
        """
        diameter = math.sqrt(max(cd_area, 0.0) * 4.0 / math.pi)
        return cls(cd=1.0, diameter=diameter, inflation_time=inflation_time)

    def deploy(self, t: float):
        self.deployed    = True
        self._deploy_time = t
        logger.debug(f"Parachute deployed at t={t:.2f}s")

    def effective_cd_area(self, t: float) -> float:
        """Returns CdA accounting for inflation transient."""
        if not self.deployed or self._deploy_time is None:
            return 0.0
        elapsed = t - self._deploy_time
        if elapsed <= 0:
            return 0.0
        if elapsed >= self.inflation_time:
            return self.cd_area_nominal
        # Smooth cubic inflation curve
        frac = elapsed / self.inflation_time
        smooth = 3 * frac**2 - 2 * frac**3
        return self.cd_area_nominal * smooth

    def get_drag_force(self, rho: float, v_mag: float, t: float) -> float:
        cda = self.effective_cd_area(t)
        return 0.5 * rho * v_mag**2 * cda

    def terminal_velocity(self, mass: float, rho: float) -> float:
        """Steady-state terminal velocity (m/s)."""
        cda = self.cd_area_nominal
        if cda <= 0 or rho <= 0:
            return 999.0
        return math.sqrt(2 * mass * G0 / (rho * cda))


APOGEE = "apogee"
ALTITUDE = "altitude"
DEPLOY_EVENTS = (APOGEE, ALTITUDE)

# Canopies this large or larger fill over twice the base inflation time.
# (The old two-slot model always gave the "main" 2x and the "drogue" 1x.)
_SLOW_FILL_CDA = 1.0


class Canopy:
    """One parachute or streamer: its drag area, when it opens, and what it
    did in the flight."""

    def __init__(self, name: str, cd_area: float, event: str = APOGEE,
                 altitude: float = 0.0, delay: float = 0.0,
                 inflation_time: float = 0.5):
        if event not in DEPLOY_EVENTS:
            raise ValueError(f"unknown deploy event {event!r}")
        self.name = name
        self.cd_area = max(0.0, float(cd_area))
        self.event = event
        self.altitude = float(altitude)      # m above the pad, ALTITUDE event
        self.delay = max(0.0, float(delay))  # s after the event
        self.dynamics = ParachuteDynamics.from_cd_area(self.cd_area, inflation_time)
        self.reset()

    def reset(self):
        self.dynamics.deployed = False
        self.dynamics._deploy_time = None
        self.triggered_at = None             # time the event fired
        self.deploy_time = None
        self.deploy_altitude = None
        self.deploy_speed = None
        self.peak_force = 0.0
        self.descent_rate = None             # |vz| when the next canopy opened / at touchdown

    @property
    def deployed(self) -> bool:
        return self.dynamics.deployed

    def report(self) -> dict:
        return {"name": self.name, "cd_area": self.cd_area, "event": self.event,
                "trigger_altitude": self.altitude if self.event == ALTITUDE else None,
                "delay": self.delay, "deploy_time": self.deploy_time,
                "deploy_altitude": self.deploy_altitude,
                "deploy_speed": self.deploy_speed, "peak_force": self.peak_force,
                "descent_rate": self.descent_rate}


class RecoverySystem:
    """Every canopy on the vehicle, each with its own trigger.

    Both flight engines call :meth:`step` once per accepted step and
    :meth:`get_drag_force` from their derivatives, so a design deploys the
    same way in the Simulation tab and in Monte Carlo.

    Triggers: APOGEE opens *delay* s after apogee; ALTITUDE opens *delay* s
    after the vehicle, descending after apogee, passes *altitude* (above the
    pad). An ALTITUDE canopy whose altitude is above the apogee opens at
    apogee. Apogee is the step where vertical speed turns from up to down.
    """

    def __init__(self, canopies=None):
        self.canopies = [c for c in (canopies or []) if c.cd_area > 0.0]
        self.reset()

    # ── construction ──
    @classmethod
    def from_canopies(cls, rows, inflation_time: float = 0.5) -> "RecoverySystem":
        """From state dicts: {name, cd_area, event, altitude, delay}."""
        out = []
        for r in rows or []:
            area = float(r.get("cd_area", 0.0))
            fill = inflation_time * (2.0 if area >= _SLOW_FILL_CDA else 1.0)
            out.append(Canopy(str(r.get("name", "Parachute")), area,
                              str(r.get("event", APOGEE)),
                              float(r.get("altitude", 0.0)),
                              float(r.get("delay", 0.0)), fill))
        return cls(out)

    @classmethod
    def from_cd_areas(cls, drogue_cd_area: float, main_cd_area: float,
                      main_deploy_altitude: float = 300.0,
                      drogue_delay: float = 0.0,
                      inflation_time: float = 0.5) -> "RecoverySystem":
        """The Simulation tab's two slots: a drogue at apogee + delay and a
        main at the main-deploy altitude (main fill time 2x base)."""
        return cls([
            Canopy("Drogue", drogue_cd_area, APOGEE, 0.0, drogue_delay, inflation_time),
            Canopy("Main", main_cd_area, ALTITUDE, main_deploy_altitude, 0.0,
                   inflation_time * 2.0),
        ])

    @classmethod
    def for_state(cls, s) -> "RecoverySystem":
        """The recovery a RocketState (or BatchSimConfig) describes: the
        Design tab's canopies when it has any, else the two slots."""
        fill = getattr(s, "recovery_inflation_time", 0.6) or 0.6
        rows = getattr(s, "recovery_canopies", None) or []
        if rows:
            return cls.from_canopies(rows, fill)
        return cls.from_cd_areas(getattr(s, "drogue_cd_area", 0.0),
                                 getattr(s, "main_cd_area", 0.0),
                                 getattr(s, "main_deploy_altitude", 300.0),
                                 getattr(s, "drogue_deploy_delay", 1.0), fill)

    def reset(self):
        for c in self.canopies:
            c.reset()
        self.apogee_time = None
        self._prev_vz = 0.0
        self._open = []          # canopies in deploy order
        self.peak_total_force = 0.0   # largest summed canopy drag (N): the opening shock

    # ── flight ──
    @property
    def any_deployed(self) -> bool:
        return bool(self._open)

    @property
    def deployed(self) -> list:
        return list(self._open)

    def step(self, t: float, altitude: float, vz: float,
             rho: float = 0.0, speed: float = 0.0) -> list:
        """Advance one accepted step. Returns the canopies that opened now."""
        if self.apogee_time is None and self._prev_vz > 0.0 and vz <= 0.0 and altitude > 0.0:
            self.apogee_time = t
        self._prev_vz = vz

        opened = []
        if self.apogee_time is not None:
            for c in self.canopies:
                if c.deployed:
                    continue
                if c.triggered_at is None:
                    if c.event == APOGEE or altitude <= c.altitude:
                        c.triggered_at = t
                if c.triggered_at is not None and t - c.triggered_at >= c.delay:
                    # A canopy that opened on an earlier step measures its
                    # descent rate here. One opening in this same step has
                    # not come down under itself alone, so it has none yet
                    # (touchdown() gives it the landing rate). It used to be
                    # stamped too: two canopies at apogee gave the first a
                    # "descent rate" of the 0.3 m/s the vehicle had there.
                    for prev in self._open:
                        if prev.descent_rate is None and prev.deploy_time < t:
                            prev.descent_rate = abs(vz)
                    c.dynamics.deploy(t)
                    c.deploy_time, c.deploy_altitude = t, altitude
                    c.deploy_speed = speed
                    self._open.append(c)
                    opened.append(c)
                    logger.info(f"{c.name} deployed at t={t:.2f}s alt={altitude:.1f}m")

        if rho > 0.0 and self._open:
            total = 0.0
            for c in self._open:
                f = c.dynamics.get_drag_force(rho, speed, t)
                total += f
                if f > c.peak_force:
                    c.peak_force = f
            if total > self.peak_total_force:
                self.peak_total_force = total
        return opened

    def touchdown(self, vz: float) -> None:
        """Record the descent rate of every canopy still open at landing."""
        for c in self._open:
            if c.descent_rate is None:
                c.descent_rate = abs(vz)

    def get_drag_force(self, rho: float, v_mag: float, t: float) -> float:
        """Total recovery drag force (N)."""
        return sum(c.dynamics.get_drag_force(rho, v_mag, t) for c in self._open)

    def reports(self) -> list:
        return [c.report() for c in self.canopies]


def final_descent_rate(state, rho: float = 1.225):
    """Steady descent rate (m/s) of the burnt-out vehicle under every canopy
    it carries (RecoverySystem.for_state: the Design tab's parachutes, else
    the two Simulation-tab slots), in air of density *rho*. None when it has
    no canopy. The airframe's own drag is left out, so this is a little
    faster than the touchdown the flight reports.

    For the landing envelope the 3D views draw before and during a flight.
    Each worked its own figure out as sqrt(2 m g / (rho x main_cd_area)) with
    m = dry mass + propellant (no motor case, and propellant that has burnt
    by then) and the area floored at 0.1 m². A design with one parachute has
    no main slot, so its envelope was drawn for a fall under an invented
    0.1 m² canopy: 34 m/s for the canonical rocket on one 2.5 m² parachute,
    which comes down at 6.2 m/s.
    """
    area = sum(c.cd_area for c in RecoverySystem.for_state(state).canopies)
    mass = (getattr(state, "dry_mass", 0.0) or 0.0) + (getattr(state, "motor_dry_mass", 0.0) or 0.0)
    if area <= 0.0 or mass <= 0.0 or rho <= 0.0:
        return None
    return math.sqrt(2.0 * mass * G0 / (rho * area))


def recovery_phase_parameters(system: RecoverySystem) -> tuple:
    """(drogue_delay, main_deploy_altitude) for core.flight_phases, so the
    Drogue/Main descent phases shown in the UI follow the canopies actually
    on the vehicle: the shortest apogee delay and the highest altitude
    trigger (0 when there is none of that kind)."""
    apogee = [c.delay for c in system.canopies if c.event == APOGEE]
    alt = [c.altitude for c in system.canopies if c.event == ALTITUDE]
    return (min(apogee) if apogee else 0.0, max(alt) if alt else 0.0)
