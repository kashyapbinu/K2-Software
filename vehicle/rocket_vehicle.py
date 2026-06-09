"""
K2 Aerospace — Rocket Vehicle (The Canonical Simulation Object)
"""

from vehicle.component import VehicleComponent
from vehicle.stage import Stage
import numpy as np

class RocketVehicle:
    """
    The canonical, read-only representation of the rocket for the simulation engine.
    This is built from the UI's component tree but provides a physics-first interface.
    """

    def __init__(self, name: str):
        self.name = name
        self.stages: list[Stage] = []
        self.components: list[VehicleComponent] = []
        self.reference_area = 0.0
        self.wetted_area = 0.0

    def add_stage(self, stage: Stage):
        self.stages.append(stage)

    def add_component(self, component: VehicleComponent):
        self.components.append(component)

    def get_active_stages(self) -> list[Stage]:
        return [s for s in self.stages if s.is_active]

    def total_mass(self) -> float:
        return sum(s.total_mass() for s in self.get_active_stages()) + \
               sum(c.total_mass() for c in self.components)

    def propellant_mass(self) -> float:
        mass = 0.0
        for s in self.get_active_stages():
            if s.motor:
                mass += s.motor.current_propellant_mass
        return mass

    def cg_global(self) -> float:
        tm = self.total_mass()
        if tm == 0:
            return 0.0

        moment = 0.0
        for s in self.get_active_stages():
            moment += s.total_mass() * s.cg_global()
        for c in self.components:
            moment += c.total_mass() * c.cg_global()
        return moment / tm

    def cp_global(self) -> float:
        """Barrowman center of pressure estimation."""
        cn_total = 0.0
        cp_weighted = 0.0

        # Stages
        for s in self.get_active_stages():
            # For this simple prototype, we assume the stage itself doesn't have a direct CN,
            # but its children might.
            for c in s.children:
                cn, cp = c.get_aerodynamic_properties()
                cn_total += cn
                cp_weighted += cn * (c.position + cp)
                
        # Top-level components
        for c in self.components:
             cn, cp = c.get_aerodynamic_properties()
             cn_total += cn
             cp_weighted += cn * (c.position + cp)

        if cn_total == 0:
            return 0.0
        return cp_weighted / cn_total

    def inertia_tensor(self) -> tuple[float, float, float]:
        """
        Calculates the principal moments of inertia (Ixx, Iyy, Izz)
        about the global CG using the parallel axis theorem.
        """
        cg = self.cg_global()
        ixx_total = iyy_total = izz_total = 0.0

        def _add_component_inertia(comp: VehicleComponent):
            nonlocal ixx_total, iyy_total, izz_total
            ixx, iyy, izz = comp.moment_of_inertia_local()
            m = comp.total_mass()
            d = comp.cg_global() - cg # axial distance from local CG to global CG

            # Parallel axis theorem. Ixx is the ROLL (longitudinal) axis — an
            # axial CG offset does NOT shift roll inertia (components are on the
            # centerline), so add no m·d² there. Iyy and Izz are the transverse
            # pitch/yaw axes and both pick up the axial-offset term.
            ixx_total += ixx
            iyy_total += iyy + m * d**2
            izz_total += izz + m * d**2

            for child in comp.children:
                _add_component_inertia(child)

        for s in self.get_active_stages():
            _add_component_inertia(s)
        for c in self.components:
            _add_component_inertia(c)

        return (ixx_total, iyy_total, izz_total)

    def consume_propellant(self, amount: float):
        """Consumes propellant from the active stage motor."""
        stages = self.get_active_stages()
        if stages and stages[-1].motor:
            stages[-1].motor.consume_propellant(amount)
