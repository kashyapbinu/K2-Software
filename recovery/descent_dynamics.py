"""
K2 AeroSim — Descent Dynamics
"""

import math

class DescentDynamics:
    """
    Computes descent rates and forces during recovery.
    """
    @staticmethod
    def compute_terminal_velocity(mass: float, rho: float, cd_area: float) -> float:
        """
        Calculates the theoretical terminal velocity.
        """
        if cd_area <= 0 or rho <= 0:
            return 0.0
        # mg = 1/2 * rho * v^2 * CdA -> v = sqrt(2mg / (rho * CdA))
        return math.sqrt((2 * mass * 9.80665) / (rho * cd_area))

    @staticmethod
    def get_recovery_drag(drogue, main, rho: float, velocity_mag: float) -> float:
        """
        Gets the total drag force from deployed recovery systems.
        """
        drag = 0.0
        if drogue:
            drag += drogue.get_drag_force(rho, velocity_mag)
        if main:
            drag += main.get_drag_force(rho, velocity_mag)
        return drag
