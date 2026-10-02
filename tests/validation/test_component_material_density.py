"""
A component's mass must use the density of the material it is made of.

``RocketComponent._get_density`` looked the name up in the Design list
(core.components.MATERIALS) and fell back to Cardboard for anything else,
silently. The structural library uses different names for the same materials,
and the validation rockets, which also feed the FEM solver, use those: an
airframe set to "Aluminum 6061-T6" was weighed at 680 kg/m3 instead of 2700.
The canonical validation rocket came out at 1.13 kg instead of 3.98 kg, which
moved its CG 0.3 m aft and took its stability margin from +3.3 to +0.35 cal.
"""
import logging

import pytest

from core.components import (MATERIALS, BodyTube, material_density)
from structures.solvers.base import STRUCTURAL_MATERIALS
from validation.cases.rocket_canonical import canonical_assembly


def test_structural_names_keep_their_own_density():
    for name, material in STRUCTURAL_MATERIALS.items():
        if name in MATERIALS:
            continue    # the Design list is the authority for names in both
        assert material_density(name) == pytest.approx(material.density), name


@pytest.mark.parametrize("name", sorted(MATERIALS))
def test_design_list_densities_are_unchanged(name):
    assert material_density(name) == MATERIALS[name]["density"]


def test_aluminium_tube_weighs_the_same_under_either_name():
    design, structural = BodyTube(), BodyTube()
    design.material = "Aluminum 6061"
    structural.material = "Aluminum 6061-T6"
    assert structural.computed_mass() == pytest.approx(design.computed_mass())
    assert structural.computed_mass() > BodyTube().computed_mass() * 2.5


def test_validation_rocket_has_an_aluminium_airframe_mass():
    # Tube 3.376 kg + nose 0.436 kg in aluminium, plywood fins 0.169 kg.
    assert canonical_assembly().total_mass() == pytest.approx(3.981, abs=0.005)


def test_unknown_material_is_reported_once(caplog):
    with caplog.at_level(logging.WARNING, logger="K2.Components"):
        first = material_density("Unobtainium-test")
        second = material_density("Unobtainium-test")

    assert first == second == MATERIALS["Cardboard"]["density"]
    reports = [r for r in caplog.records if "Unobtainium-test" in r.getMessage()]
    assert len(reports) == 1
