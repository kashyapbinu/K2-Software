"""
K2 Aerospace — Canonical Base Component
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

class VehicleComponent(ABC):
    """
    Abstract base class for all canonical vehicle components.
    These are read-only physics representations, distinct from UI components.
    """

    def __init__(self, name: str, mass: float, length: float,
                 cg_local: float, position: float = 0.0):
        self.name = name
        self.mass = mass                # kg
        self.length = length            # m
        self.cg_local = cg_local        # m (relative to component top)
        self.position = position        # m (relative to rocket tip)
        self.children: List['VehicleComponent'] = []

    def add_child(self, child: 'VehicleComponent'):
        self.children.append(child)

    def total_mass(self) -> float:
        return self.mass + sum(c.total_mass() for c in self.children)

    def cg_global(self) -> float:
        """Global CG of this component and its children."""
        tm = self.total_mass()
        if tm == 0:
            return self.position + self.cg_local
        
        moment = self.mass * (self.position + self.cg_local)
        for c in self.children:
            moment += c.total_mass() * c.cg_global()
        return moment / tm

    def moment_of_inertia_local(self) -> tuple[float, float, float]:
        """
        Returns principal moments of inertia (Ixx, Iyy, Izz) around this component's local CG.
        Assume z is axial, x/y are lateral.
        """
        return (0.0, 0.0, 0.0)

    def outer_diameter(self) -> float:
        return 0.0

    def get_aerodynamic_properties(self) -> tuple[float, float]:
        """Returns (CN, CP_local) for this component."""
        return (0.0, 0.0)
