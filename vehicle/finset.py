"""
K2 Aerospace — Canonical Fin Set
"""

from vehicle.component import VehicleComponent
import math

class FinSet(VehicleComponent):
    """
    Aerodynamic representation of a set of fins.
    """

    def __init__(self, name: str, mass: float, count: int, root_chord: float,
                 tip_chord: float, height: float, sweep_angle: float, position: float = 0.0):
        # Fin CG is approximated at 1/3 root chord
        super().__init__(name, mass, root_chord, root_chord / 3, position)
        self.count = count
        self.root_chord = root_chord
        self.tip_chord = tip_chord
        self.height = height
        self.sweep_angle = sweep_angle

    def get_aerodynamic_properties(self) -> tuple[float, float]:
        """Barrowman approximation for fins."""
        if self.root_chord <= 0:
            return (0.0, 0.0)
        # CN depends on fin geometry, very roughly proportional to count
        cn = self.count * 2.0
        # CP is approximated
        cp_local = self.root_chord * 0.25
        return (cn, cp_local)
