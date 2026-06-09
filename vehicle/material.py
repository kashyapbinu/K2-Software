"""
K2 Aerospace — Materials Database
"""

from dataclasses import dataclass

@dataclass
class Material:
    name: str
    density: float  # kg/m^3
    color: str

MATERIALS = {
    "Cardboard": Material("Cardboard", 680, "#a0826d"),
    "Kraft Phenolic": Material("Kraft Phenolic", 960, "#8B7355"),
    "Fiberglass": Material("Fiberglass", 1800, "#d4e4bc"),
    "Carbon Fiber": Material("Carbon Fiber", 1600, "#2a2a2a"),
    "Aluminum 6061": Material("Aluminum 6061", 2700, "#b0b0b0"),
    "Polycarbonate": Material("Polycarbonate", 1200, "#c8dce8"),
    "Balsa Wood": Material("Balsa Wood", 170, "#e8d8b0"),
    "Plywood (Birch)": Material("Plywood (Birch)", 630, "#c4a66a"),
    "ABS Plastic": Material("ABS Plastic", 1050, "#e0e0e0"),
    "Nylon": Material("Nylon", 1150, "#f0f0f0"),
    "Ripstop Nylon": Material("Ripstop Nylon", 40, "#ff6633"),
}

def get_material(name: str) -> Material:
    return MATERIALS.get(name, MATERIALS["Cardboard"])
