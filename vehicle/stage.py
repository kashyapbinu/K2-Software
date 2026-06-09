"""
K2 Aerospace — Canonical Stage
"""

from vehicle.component import VehicleComponent
from vehicle.motor import Motor

class Stage(VehicleComponent):
    """
    A full stage of the rocket, containing a motor and other components.
    """
    def __init__(self, name: str, position: float = 0.0):
        super().__init__(name, mass=0.0, length=0.0, cg_local=0.0, position=position)
        self.motor: Motor | None = None
        self.is_active = True

    def set_motor(self, motor: Motor):
        self.motor = motor
        self.add_child(motor)

    def total_mass(self) -> float:
        if not self.is_active:
            return 0.0
        return sum(c.total_mass() for c in self.children)
