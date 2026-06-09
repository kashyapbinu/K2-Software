"""
K2 Aerospace — Nozzle Model
============================
Canonical nozzle representation for simulation.  Models nozzle geometry,
expansion ratio, and (for propulsion nozzles) isentropic thrust coefficient.
"""

import math
from vehicle.component import VehicleComponent


class Nozzle(VehicleComponent):
    """
    Vehicle-layer nozzle component.

    Supports three nozzle types:

        * ``"propulsion"`` — A full rocket propulsion nozzle (converging-diverging
          De Laval design).  Provides isentropic thrust-coefficient calculation
          via ``thrust_coefficient()``.

        * ``"aerospike"`` — An aerospike / plug nozzle.  Geometry is stored but
          the thrust-coefficient model is not implemented; ``thrust_coefficient()``
          returns 0.0.

        * ``"conical"`` — A simple conical nozzle.  Often used as a lightweight
          approximation.  ``thrust_coefficient()`` returns 0.0.

    Geometry Parameters
    -------------------
    throat_diameter : float
        Diameter of the nozzle throat (m).
    exit_diameter : float
        Diameter of the nozzle exit plane (m).
    inlet_diameter : float
        Diameter of the nozzle inlet / convergent entry (m).
    half_angle : float
        Divergent half-angle of the nozzle cone (radians).
    wall_thickness : float
        Nozzle wall thickness (m).
    """

    def __init__(self, name: str, mass: float,
                 nozzle_type: str = "propulsion",
                 throat_diameter: float = 0.02,
                 exit_diameter: float = 0.05,
                 inlet_diameter: float = 0.04,
                 length: float = 0.10,
                 half_angle: float = 0.2618,      # ~15°
                 wall_thickness: float = 0.002,
                 position: float = 0.0):
        """
        Args:
            name:             Component name.
            mass:             Nozzle mass (kg).
            nozzle_type:      One of ``"propulsion"``, ``"aerospike"``, ``"conical"``.
            throat_diameter:  Throat diameter (m).
            exit_diameter:    Exit-plane diameter (m).
            inlet_diameter:   Inlet / convergent-section diameter (m).
            length:           Nozzle axial length (m).
            half_angle:       Divergent half-angle (rad).
            wall_thickness:   Wall thickness (m).
            position:         Axial position from rocket tip (m).
        """
        # CG approximated at 60 % of length (toward the heavier exit end)
        super().__init__(
            name=name,
            mass=mass,
            length=length,
            cg_local=length * 0.6,
            position=position,
        )

        self.nozzle_type = nozzle_type
        self.throat_diameter = throat_diameter
        self.exit_diameter = exit_diameter
        self.inlet_diameter = inlet_diameter
        self.half_angle = half_angle
        self.wall_thickness = wall_thickness

    # ── Geometry properties ──────────────────────────────────────────────────

    @property
    def expansion_ratio(self) -> float:
        """Area expansion ratio  ε = A_exit / A_throat = (d_e / d_t)²."""
        if self.throat_diameter <= 0:
            return 1.0
        return (self.exit_diameter / self.throat_diameter) ** 2

    @property
    def throat_area(self) -> float:
        """Throat cross-sectional area (m²)."""
        return math.pi * (self.throat_diameter / 2) ** 2

    @property
    def exit_area(self) -> float:
        """Exit cross-sectional area (m²)."""
        return math.pi * (self.exit_diameter / 2) ** 2

    # ── VehicleComponent overrides ───────────────────────────────────────────

    def moment_of_inertia_local(self) -> tuple[float, float, float]:
        """
        Principal moments of inertia about the nozzle CG.

        Approximated as a thin-walled frustum shell with inner radii r1
        (inlet end) and r2 (exit end):

            Iyy = m * (3*(r1² + r2²) + L²) / 12
        """
        m = self.total_mass()
        if m <= 0:
            return (0.0, 0.0, 0.0)

        r1 = self.inlet_diameter / 2
        r2 = self.exit_diameter / 2
        L = self.length

        iyy = m * (3.0 * (r1 ** 2 + r2 ** 2) + L ** 2) / 12.0
        ixx = 0.5 * m * (r1 ** 2 + r2 ** 2)  # axial spin
        return (ixx, iyy, iyy)

    def outer_diameter(self) -> float:
        """Outer diameter of the nozzle (exit plane)."""
        return self.exit_diameter

    def get_aerodynamic_properties(self) -> tuple[float, float]:
        """Nozzles contribute negligible aerodynamic normal force."""
        return (0.0, 0.0)

    # ── Propulsion ───────────────────────────────────────────────────────────

    def thrust_coefficient(self, gamma: float = 1.2,
                           pressure_ratio: float | None = None) -> float:
        """
        Isentropic thrust coefficient (Cf) for a converging-diverging nozzle.

        Only meaningful for ``nozzle_type == "propulsion"``.  Other types
        return 0.0.

        The calculation assumes ideal quasi-1-D isentropic expansion to the
        nozzle exit, with no separation or losses:

            Cf = sqrt( (2γ²/(γ-1)) * (2/(γ+1))^((γ+1)/(γ-1))
                       * (1 - (Pe/Pc)^((γ-1)/γ)) )
                 + ε * (Pe/Pc)

        where ε = expansion ratio and Pe/Pc = pressure_ratio.

        Args:
            gamma:          Ratio of specific heats (default 1.2 for solid
                            propellant products).
            pressure_ratio: Exit-to-chamber pressure ratio Pe/Pc.  If *None*,
                            an estimate is derived from the expansion ratio
                            using the isentropic area–Mach relation.

        Returns:
            Thrust coefficient (dimensionless).
        """
        if self.nozzle_type != "Full Propulsion":
            return 0.0

        eps = self.expansion_ratio
        if eps <= 1.0:
            return 0.0

        gm1 = gamma - 1.0
        gp1 = gamma + 1.0

        # ── Estimate Pe/Pc from expansion ratio if not provided ──────────
        if pressure_ratio is None:
            # Newton iteration on isentropic area–Mach relation to find
            # the supersonic Mach number for the given expansion ratio.
            M = 3.0  # initial guess (supersonic branch)
            exp = gp1 / (2.0 * gm1)
            for _ in range(100):
                t = 1.0 + 0.5 * gm1 * M ** 2
                u = (2.0 / gp1) * t
                A_ratio = u ** exp / M
                # d(A/A*)/dM via chain rule on u^exp / M
                du_dM = (2.0 / gp1) * gm1 * M
                dA = (exp * u ** (exp - 1.0) * du_dM) / M - u ** exp / M ** 2
                if abs(dA) < 1e-15:
                    break
                M = M - (A_ratio - eps) / dA
                M = max(M, 1.001)  # keep supersonic

            pressure_ratio = (1.0 + 0.5 * gm1 * M ** 2) ** (-gamma / gm1)

        # ── Momentum thrust term ─────────────────────────────────────────
        coeff_sq = ((2.0 * gamma ** 2) / gm1) \
            * (2.0 / gp1) ** (gp1 / gm1) \
            * (1.0 - pressure_ratio ** (gm1 / gamma))

        if coeff_sq < 0:
            return 0.0

        cf = math.sqrt(coeff_sq) + eps * pressure_ratio
        return cf

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"Nozzle({self.name!r}, type={self.nozzle_type}, "
                f"eps={self.expansion_ratio:.2f}, "
                f"d_t={self.throat_diameter*1000:.1f}mm, "
                f"d_e={self.exit_diameter*1000:.1f}mm)")
