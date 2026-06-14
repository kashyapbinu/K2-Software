"""
K2 AeroSim — Structures Package
===================================
Structural analysis: FEM (CalculiX), thermal, materials.
"""
from structures.solvers.base import (
    FEMConfig, FEMResult, ModalResult, ThermalResult,
    LoadCase, StructuralMaterial, get_structural_material,
    STRUCTURAL_MATERIALS,
)
from structures.fem_interface import FEMInterface

__all__ = [
    "FEMConfig", "FEMResult", "ModalResult", "ThermalResult",
    "LoadCase", "StructuralMaterial", "get_structural_material",
    "STRUCTURAL_MATERIALS", "FEMInterface",
]
