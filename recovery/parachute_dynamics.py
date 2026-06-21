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


class RecoverySystem:
    """
    Combined drogue + main parachute system with deployment logic.
    """

    def __init__(self,
                 drogue_cd: float = 1.5, drogue_diameter: float = 0.5,
                 main_cd: float = 2.2, main_diameter: float = 2.0,
                 main_deploy_altitude: float = 300.0,
                 drogue_delay: float = 0.0,
                 inflation_time: float = 0.5):
        self.drogue = ParachuteDynamics(drogue_cd, drogue_diameter, inflation_time)
        self.main   = ParachuteDynamics(main_cd, main_diameter, inflation_time * 2)
        self.main_deploy_altitude = main_deploy_altitude
        self.drogue_delay         = drogue_delay
        self._apogee_time         = None
        self.drogue_deployed      = False
        self.main_deployed        = False

    @classmethod
    def from_cd_areas(cls, drogue_cd_area: float, main_cd_area: float,
                      main_deploy_altitude: float = 300.0,
                      drogue_delay: float = 0.0,
                      inflation_time: float = 0.5):
        """Build from Cd×A products (the form the K2 state engine stores).

        The main canopy is much larger than the drogue, so its inflation
        transient is scaled up proportionally (2× base) — the shock-loading
        peak is dominated by the main, and a too-fast main fill exaggerates it.
        """
        sys = cls.__new__(cls)
        sys.drogue = ParachuteDynamics.from_cd_area(drogue_cd_area, inflation_time)
        sys.main   = ParachuteDynamics.from_cd_area(main_cd_area, inflation_time * 2)
        sys.main_deploy_altitude = main_deploy_altitude
        sys.drogue_delay         = drogue_delay
        sys._apogee_time         = None
        sys.drogue_deployed      = False
        sys.main_deployed        = False
        return sys

    def update(self, t: float, altitude: float, velocity_z: float, phase: str):
        """Check deployment conditions and deploy chutes."""
        if phase in ("Apogee", "Drogue Descent") and not self.drogue_deployed:
            if self._apogee_time is None:
                self._apogee_time = t
            if t - self._apogee_time >= self.drogue_delay:
                self.drogue.deploy(t)
                self.drogue_deployed = True
                logger.info(f"Drogue deployed at t={t:.2f}s alt={altitude:.1f}m")

        if self.drogue_deployed and not self.main_deployed:
            if altitude <= self.main_deploy_altitude and velocity_z < 0:
                self.main.deploy(t)
                self.main_deployed = True
                logger.info(f"Main deployed at t={t:.2f}s alt={altitude:.1f}m")

    def get_drag_force(self, rho: float, v_mag: float, t: float) -> float:
        """Total recovery drag force (N)."""
        drag = 0.0
        if self.drogue_deployed:
            drag += self.drogue.get_drag_force(rho, v_mag, t)
        if self.main_deployed:
            drag += self.main.get_drag_force(rho, v_mag, t)
        return drag

    def reset(self):
        self.drogue_deployed = self.main_deployed = False
        self._apogee_time    = None
        self.drogue.deployed = self.main.deployed = False
        self.drogue._deploy_time = self.main._deploy_time = None
