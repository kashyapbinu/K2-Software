"""
K2 Aerospace — Vehicle Builder Pipeline
Translates the UI RocketAssembly into the canonical RocketVehicle representation.
"""

from core.components import RocketAssembly, RocketComponent
import core.components as ui_comp
from vehicle.rocket_vehicle import RocketVehicle
from vehicle.stage import Stage
from vehicle.motor import Motor
from vehicle.finset import FinSet
from vehicle.avionics import AvionicsPackage
from vehicle.nozzle import Nozzle as CanonNozzle
from vehicle.component import VehicleComponent

def build_vehicle(assembly: RocketAssembly) -> RocketVehicle:
    """
    Builds a canonical RocketVehicle from the UI's RocketAssembly.
    """
    vehicle = RocketVehicle(assembly.name)

    # We assume the UI assembly stages are ordered bottom-to-top (or top-to-bottom, we need to check).
    # Usually UI stages are Sustainer (top) -> Booster 1 -> Booster 2.
    # Let's preserve the order.
    
    for ui_stage in assembly.stages:
        canon_stage = Stage(ui_stage.name, ui_stage.position)
        
        # In a real system, we'd recursively traverse children.
        # Here we do a simplified translation for the simulation prototype.
        _traverse_and_build(ui_stage, canon_stage)
        
        vehicle.add_stage(canon_stage)

    # Some UI components might be at the top level of assembly if not in a stage (though UI enforces stages)
    # We can handle them here if needed.

    return vehicle

def _traverse_and_build(ui_parent: RocketComponent, canon_parent: VehicleComponent):
    for ui_child in ui_parent.children:
        canon_child = None
        
        mass = ui_child.computed_mass()
        length = ui_child.component_length()
        pos = ui_child.position
        
        if isinstance(ui_child, ui_comp.TrapezoidalFinSet):
            canon_child = FinSet(
                name=ui_child.name,
                mass=mass,
                count=ui_child.fin_count,
                root_chord=ui_child.root_chord,
                tip_chord=ui_child.tip_chord,
                height=ui_child.height,
                sweep_angle=ui_child.sweep_angle,
                position=pos
            )
        elif isinstance(ui_child, ui_comp.Nozzle):
            canon_child = CanonNozzle(
                name=ui_child.name,
                mass=mass,
                nozzle_type=ui_child.nozzle_type,
                throat_diameter=ui_child.throat_diameter,
                exit_diameter=ui_child.exit_diameter,
                inlet_diameter=ui_child.inlet_diameter,
                length=length,
                half_angle=ui_child.half_angle,
                wall_thickness=ui_child.wall_thickness,
                position=pos
            )
        elif getattr(ui_child, 'is_motor_mount', False):
            # If it's a motor mount, let's pretend it has a motor for now
            # In the real system, motor selection is attached to the stage or mount.
            # We'll need a way to pass the actual selected motor.
            # For this prototype, we'll let the simulation engine set the motor explicitly later.
            pass
        elif isinstance(ui_child, ui_comp.NoseCone) or isinstance(ui_child, ui_comp.BodyTube) or isinstance(ui_child, ui_comp.Transition):
            # Generic structural component
            canon_child = VehicleComponent(
                name=ui_child.name,
                mass=mass,
                length=length,
                cg_local=length / 2, # simplified
                position=pos
            )
        
        if canon_child:
            canon_parent.add_child(canon_child)
            # Recurse if the component has children (like fins on a body tube)
            _traverse_and_build(ui_child, canon_child)
        else:
            # If we didn't create a canonical child, still traverse the UI children 
            # and attach them to the current canonical parent (flattening).
            _traverse_and_build(ui_child, canon_parent)
