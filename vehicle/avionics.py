"""
K2 Aerospace — Canonical Avionics
"""

from vehicle.component import VehicleComponent

class AvionicsPackage(VehicleComponent):
    """
    Representation of the avionics bay mass and properties.
    """
    def __init__(self, name: str, mass: float, length: float, position: float = 0.0):
        super().__init__(name, mass, length, length / 2, position)
