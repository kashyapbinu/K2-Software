"""
Sharp-cone geometry for the Taylor–Maccoll SU2 verification case.
"""
from __future__ import annotations

import math


def cone_assembly(half_angle_deg: float = 10.0, length: float = 0.5):
    """A sharp cone of the given half-angle as a RocketAssembly (conical nose).

    A pure cone is a conical nose cone with no body tube. base_radius =
    length·tan(half_angle); diameter = 2·base_radius.
    """
    from core.components import RocketAssembly, NoseCone

    base_r = length * math.tan(math.radians(half_angle_deg))
    asm = RocketAssembly()
    asm.name = f"cone-{half_angle_deg:g}deg"

    nose = NoseCone()
    nose.shape = "Conical"
    nose.length = length
    nose.diameter = 2.0 * base_r
    nose.wall_thickness = 0.002
    nose.material = "Aluminum 6061-T6"
    asm.add_component(asm.stages[0], nose)
    return asm
