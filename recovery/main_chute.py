"""
K2 Aerospace — Main Parachute
"""

class MainChute:
    """
    Main parachute for final low-velocity landing.
    """
    def __init__(self, cd: float, diameter: float):
        self.cd = cd
        self.diameter = diameter
        self.area = 3.14159 * (diameter / 2) ** 2
        self.cd_area = cd * self.area
        self.deployed = False

    def deploy(self):
        self.deployed = True

    def get_drag_force(self, rho: float, velocity_mag: float) -> float:
        if not self.deployed:
            return 0.0
        return 0.5 * rho * velocity_mag**2 * self.cd_area
