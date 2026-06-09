"""
K2 Aerospace — Motor Model
============================
High-fidelity motor representation that is ALSO a VehicleComponent
so it integrates cleanly with stage.children and inertia_tensor().
"""

import math
import logging
from vehicle.component import VehicleComponent

logger = logging.getLogger("K2.Motor")

G0 = 9.80665  # m/s²


class Motor(VehicleComponent):
    """
    Canonical motor representation for simulation.

    Supports:
        - Arbitrary thrust curve [(t, thrust)] with linear interpolation
        - Isp-based propellant mass flow (dm/dt = -T / (Isp * g0))
        - Chamber pressure estimation
        - Propellant depletion tracking
        - Thrust misalignment angle
    """

    def __init__(self, designation: str,
                 empty_mass: float = 0.1,
                 propellant_mass: float = 0.5,
                 length: float = 0.3,
                 diameter: float = 0.038,
                 isp: float = 200.0,
                 thrust_curve: list | None = None,
                 avg_thrust: float = 0.0,
                 max_thrust: float = 0.0,
                 burn_time: float = 0.0,
                 thrust_misalignment: float = 0.0):
        """
        Args:
            designation:          Motor name / NFPA code (e.g. 'L1120').
            empty_mass:           Motor casing mass after burn (kg).
            propellant_mass:      Initial propellant mass (kg).
            length:               Motor length (m).
            diameter:             Motor diameter (m).
            isp:                  Specific impulse (s).
            thrust_curve:         List of (time_s, thrust_N) tuples. If None,
                                  a trapezoidal curve is generated from avg/max/burn_time.
            avg_thrust:           Average thrust (N) — used if no curve provided.
            max_thrust:           Peak thrust (N) — used if no curve provided.
            burn_time:            Burn duration (s) — used if no curve provided.
            thrust_misalignment:  Misalignment angle (rad) — small off-axis thrust.
        """
        # Initialise VehicleComponent base (motor sits at the aft end of the rocket)
        # Position = 0 here; caller (Stage/builder) sets the real position after.
        super().__init__(
            name=designation,
            mass=empty_mass,              # structural (casing) mass
            length=length,
            cg_local=length * 0.5,        # CG roughly at midpoint
            position=0.0,
        )

        self.designation         = designation
        self.empty_mass          = empty_mass
        self.propellant_mass_initial = propellant_mass
        self.current_propellant_mass = propellant_mass
        self.diameter            = diameter
        self.isp                 = isp
        self.thrust_misalignment = thrust_misalignment

        # Build or store thrust curve
        if thrust_curve and len(thrust_curve) >= 2:
            self._curve = sorted(thrust_curve, key=lambda p: p[0])
        else:
            self._curve = self._build_trapezoidal(avg_thrust, max_thrust, burn_time)

        self.burn_time = self._curve[-1][0] if self._curve else 0.0

        logger.debug(
            f"Motor '{designation}': Isp={isp}s, mp={propellant_mass:.2f}kg, "
            f"burn={self.burn_time:.2f}s, points={len(self._curve)}"
        )

    # ── Internal builders ─────────────────────────────────────────────────────

    @staticmethod
    def _build_trapezoidal(avg_thrust: float, max_thrust: float,
                            burn_time: float) -> list:
        """Generate a smooth trapezoidal thrust curve."""
        if burn_time <= 0 or avg_thrust <= 0:
            return [(0.0, 0.0), (0.001, 0.0)]
        if max_thrust <= 0:
            max_thrust = avg_thrust * 1.35  # typical peak factor

        ramp  = burn_time * 0.08   # 8% ramp-up
        decay = burn_time * 0.05   # 5% tail-off
        return [
            (0.0,                      0.0),
            (ramp,                     max_thrust),
            (burn_time - decay,        avg_thrust * 0.95),
            (burn_time,                0.0),
        ]

    # ── Thrust API ────────────────────────────────────────────────────────────

    def get_thrust(self, t: float) -> float:
        """
        Interpolated thrust at time t (seconds from ignition).

        Returns 0 if propellant is exhausted or t is outside the curve.
        """
        if self.current_propellant_mass <= 0 or not self._curve:
            return 0.0
        if t < 0 or t >= self._curve[-1][0]:
            return 0.0

        # Linear interpolation
        for i in range(len(self._curve) - 1):
            t0, f0 = self._curve[i]
            t1, f1 = self._curve[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-12 else 0.0
                return max(0.0, f0 + frac * (f1 - f0))
        return 0.0

    def mass_flow_rate(self, t: float) -> float:
        """
        Propellant mass flow rate at time t (kg/s).
        Uses Isp-based calculation: dm/dt = -T / (Isp * g0).
        """
        T = self.get_thrust(t)
        if T <= 0 or self.isp <= 0:
            return 0.0
        return -T / (self.isp * G0)

    def consume_propellant(self, amount: float):
        """Deplete propellant by amount (kg). Clamps to zero."""
        self.current_propellant_mass = max(0.0, self.current_propellant_mass - amount)

    def consume_propellant_dt(self, t: float, dt: float):
        """Consume propellant over time step dt using current thrust."""
        mdot = self.mass_flow_rate(t)  # negative
        delta = abs(mdot) * dt
        self.consume_propellant(delta)

    # ── VehicleComponent overrides ─────────────────────────────────────────────

    def total_mass(self) -> float:
        """Current motor mass (casing + remaining propellant) in kg."""
        return self.empty_mass + self.current_propellant_mass

    def cg_global(self) -> float:
        """Global CG of motor. Propellant CG shifts aft as it depletes."""
        if self.total_mass() <= 0:
            return self.position + self.cg_local
        # Casing CG at midpoint, propellant CG at 55% of length
        m_case = self.empty_mass
        m_prop = self.current_propellant_mass
        cg_case = self.position + self.length * 0.5
        cg_prop = self.position + self.length * 0.55
        return (m_case * cg_case + m_prop * cg_prop) / self.total_mass()

    def moment_of_inertia_local(self) -> tuple:
        """
        Pitch moment of inertia of the motor about its own CG.
        Approximated as a solid cylinder: Iyy = m*(3r^2 + L^2)/12
        """
        m = self.total_mass()
        r = self.diameter / 2
        L = self.length
        iyy = m * (3 * r**2 + L**2) / 12.0 if m > 0 else 0.0
        ixx = 0.5 * m * r**2 if m > 0 else 0.0   # axial spin
        return (ixx, iyy, iyy)

    def propellant_fraction_remaining(self) -> float:
        """Fraction of propellant remaining (0.0 → 1.0)."""
        if self.propellant_mass_initial <= 0:
            return 0.0
        return self.current_propellant_mass / self.propellant_mass_initial

    def is_burning(self, t: float) -> bool:
        return self.get_thrust(t) > 0 and self.current_propellant_mass > 0

    # ── Chamber pressure (simplified) ─────────────────────────────────────────

    def chamber_pressure(self, t: float,
                          ambient_pressure: float = 101325.0) -> float:
        """
        Simplified chamber pressure estimate (Pa).
        Assumes Pc ∝ thrust and a fixed pressure ratio.
        """
        T = self.get_thrust(t)
        if T <= 0:
            return ambient_pressure
        # Empirical: Pc ~ T / (nozzle_throat_area * Cf)
        # Simplified as linear relationship with peak thrust
        T_max = max(f for _, f in self._curve) if self._curve else 1.0
        pc_max = 5.0e6   # 50 bar typical HPR motor
        return ambient_pressure + (T / max(T_max, 1.0)) * (pc_max - ambient_pressure)

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Motor({self.designation}, Isp={self.isp}s, "
                f"mp={self.current_propellant_mass:.3f}/{self.propellant_mass_initial:.3f}kg, "
                f"burn={self.burn_time:.2f}s)")
