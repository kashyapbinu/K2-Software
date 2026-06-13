"""
K2 Aerospace — Component Data Model
Hierarchical rocket component system with multi-stage support.
"""
import math, uuid, logging
from dataclasses import dataclass, field
from typing import Optional
logger = logging.getLogger("K2.Components")

# ── Material database ──
MATERIALS = {
    "Cardboard": {"density": 680, "color": "#a0826d"},
    "Kraft Phenolic": {"density": 960, "color": "#8B7355"},
    "Fiberglass": {"density": 1800, "color": "#d4e4bc"},
    "Carbon Fiber": {"density": 1600, "color": "#2a2a2a"},
    "Aluminum 6061": {"density": 2700, "color": "#b0b0b0"},
    "Polycarbonate": {"density": 1200, "color": "#c8dce8"},
    "Balsa Wood": {"density": 170, "color": "#e8d8b0"},
    "Plywood (Birch)": {"density": 630, "color": "#c4a66a"},
    "ABS Plastic": {"density": 1050, "color": "#e0e0e0"},
    "Nylon": {"density": 1150, "color": "#f0f0f0"},
    "Ripstop Nylon": {"density": 40, "color": "#ff6633"},
}

NOSE_SHAPES = ["Conical", "Ogive", "Elliptical", "Parabolic", "Haack (LD)"]
FIN_SHAPES = ["Trapezoidal", "Elliptical", "Swept"]
TRANSITION_SHAPES = ["Conical", "Ogive", "Elliptical"]
NOZZLE_TYPES = ["Convergent-Divergent", "Boat-Tail", "Full Propulsion"]


class RocketComponent:
    """Base class for all rocket components."""
    component_type = "Component"
    can_have_children = False
    category = "Other"

    def __init__(self, name=""):
        self.id = str(uuid.uuid4())[:8]
        self.name = name or self.component_type
        self.children: list['RocketComponent'] = []
        self.parent: Optional['RocketComponent'] = None
        self.comment = ""
        self.override_mass: Optional[float] = None
        self.color = "#808080"
        self.material = "Cardboard"
        self.surface_finish = "Normal"  # Polished/Smooth/Normal/Rough/Very Rough
        self._position = 0.0  # computed position from top

    @property
    def position(self):
        return self._position

    def _get_density(self):
        """Get the best density: ORK XML density > K2 material database."""
        ork_d = getattr(self, '_ork_density', None)
        if ork_d is not None and ork_d > 0:
            return ork_d
        mat = MATERIALS.get(self.material, MATERIALS["Cardboard"])
        return mat["density"]

    def computed_mass(self) -> float:
        if self.override_mass is not None:
            return self.override_mass
        return self._calc_mass()

    def _calc_mass(self) -> float:
        return 0.0

    def total_mass(self) -> float:
        m = self.computed_mass()
        for c in self.children:
            m += c.total_mass()
        return m

    def component_length(self) -> float:
        return 0.0

    def outer_diameter(self) -> float:
        return 0.0

    def cg_position(self) -> float:
        return self._position + self.component_length() / 2.0

    def cp_contribution(self, d_ref: float) -> tuple:
        """Returns (CN, CP_position) referenced to d_ref."""
        return (0.0, 0.0)

    def to_dict(self) -> dict:
        d = {"type": self.component_type, "id": self.id, "name": self.name,
             "material": self.material, "comment": self.comment, "color": self.color}
        if self.override_mass is not None:
            d["override_mass"] = self.override_mass
        # Preserve ORK-import positioning so reload doesn't fall back to auto-stacking
        if getattr(self, "_ork_pos", None) is not None:
            d["ork_pos"] = self._ork_pos
            d["ork_rel"] = getattr(self, "_ork_rel", "top")
        if getattr(self, "_ork_density", None) is not None:
            d["ork_density"] = self._ork_density
        d["properties"] = self._props_dict()
        d["children"] = [c.to_dict() for c in self.children]
        return d

    def _props_dict(self) -> dict:
        return {}

    def display_info(self) -> str:
        return f"{self.name} — {self.component_type}"

    def __repr__(self):
        return f"<{self.component_type}: {self.name}>"


# ═══════════════════════════════════════════════════════════════
#  STRUCTURAL COMPONENTS
# ═══════════════════════════════════════════════════════════════

class NoseCone(RocketComponent):
    component_type = "Nose Cone"
    category = "Body"

    def __init__(self, name="Nose Cone"):
        super().__init__(name)
        self.shape = "Ogive"
        self.length = 0.15
        self.diameter = 0.066
        self.wall_thickness = 0.002
        self.shoulder_length = 0.0
        self.shoulder_diameter = 0.0
        self.shoulder_thickness = 0.0
        self.material = "Fiberglass"
        self.color = "#3a8fd6"

    def component_length(self):
        # Aerodynamic length only for safety during import
        return self.length

    def outer_diameter(self):
        return self.diameter

    def cg_position(self) -> float:
        """Mass centroid of the (thin-shell) nose, shape-dependent fraction of
        length from the tip. Conical shell centroid = 2/3·L; ogive/Haack ≈ 0.47;
        ellipsoid ≈ 0.40; parabolic ≈ 0.5. Overrides the generic L/2 midpoint."""
        frac = {"Conical": 0.667, "Ogive": 0.47, "Haack (LD)": 0.47,
                "Elliptical": 0.40, "Parabolic": 0.50}.get(self.shape, 0.50)
        return self._position + self.length * frac

    def _calc_mass(self):
        density = self._get_density()
        r = self.diameter / 2
        # Ogive cone surface area
        area = math.pi * r * math.sqrt(r**2 + self.length**2)
        mass = area * self.wall_thickness * density
        # Shoulder tube mass
        if self.shoulder_length > 0 and self.shoulder_diameter > 0:
            r_sh = self.shoulder_diameter / 2
            r_sh_i = r_sh - self.shoulder_thickness if self.shoulder_thickness > 0 else r_sh - self.wall_thickness
            r_sh_i = max(r_sh_i, 0)
            mass += math.pi * (r_sh**2 - r_sh_i**2) * self.shoulder_length * density
        return mass

    def cp_contribution(self, d_ref: float):
        # CN_alpha = 2.0 referenced to base area
        # Scaled to d_ref: CN = 2.0 * (d_base / d_ref)^2
        cn = 2.0 * (self.diameter / d_ref)**2 if d_ref > 0 else 2.0
        
        if self.shape == "Conical":
            cp = self._position + self.length * 0.667
        elif self.shape in ["Ogive", "Haack (LD)"]:
            cp = self._position + self.length * 0.466
        else:
            cp = self._position + self.length * 0.5
        return (cn, cp)

    def _props_dict(self):
        return {"shape": self.shape, "length": self.length, "diameter": self.diameter,
                "wall_thickness": self.wall_thickness,
                "shoulder_length": self.shoulder_length,
                "shoulder_diameter": self.shoulder_diameter,
                "shoulder_thickness": self.shoulder_thickness}


class BodyTube(RocketComponent):
    component_type = "Body Tube"
    can_have_children = True
    category = "Body"

    def __init__(self, name="Body Tube"):
        super().__init__(name)
        self.length = 0.30
        self.outer_diameter_val = 0.066
        self.inner_diameter = 0.063
        self.material = "Kraft Phenolic"
        self.color = "#2d6aa5"

    def component_length(self):
        return self.length

    def outer_diameter(self):
        return self.outer_diameter_val

    def _calc_mass(self):
        density = self._get_density()
        r_o, r_i = self.outer_diameter_val / 2, self.inner_diameter / 2
        vol = math.pi * (r_o**2 - r_i**2) * self.length
        return vol * density

    def planform_area(self) -> float:
        """Side-view projected area for body lift (Galejs method)."""
        return self.length * self.outer_diameter_val

    def cp_contribution(self, d_ref: float):
        """
        Classic Barrowman method assumes cylindrical body tubes produce zero normal force.
        (Galejs body lift is only applied during dynamic simulation flight).
        """
        return (0.0, 0.0)

    def _props_dict(self):
        return {"length": self.length, "outer_diameter": self.outer_diameter_val,
                "inner_diameter": self.inner_diameter}


class Transition(RocketComponent):
    component_type = "Transition"
    category = "Body"

    def __init__(self, name="Transition"):
        super().__init__(name)
        self.shape = "Conical"
        self.length = 0.06
        self.fore_diameter = 0.066
        self.aft_diameter = 0.054
        self.wall_thickness = 0.002
        self.material = "Fiberglass"
        self.color = "#2d8aa5"

    def component_length(self):
        return self.length

    def outer_diameter(self):
        return max(self.fore_diameter, self.aft_diameter)

    def _calc_mass(self):
        density = self._get_density()
        r1, r2 = self.fore_diameter / 2, self.aft_diameter / 2
        s = math.sqrt((r1 - r2)**2 + self.length**2)
        area = math.pi * (r1 + r2) * s
        return area * self.wall_thickness * density

    def cp_contribution(self, d_ref: float):
        """Full Barrowman transition CP contribution."""
        d_f = self.fore_diameter
        d_a = self.aft_diameter
        if d_ref <= 0: return (0.0, 0.0)
        
        # CN_alpha referenced to d_ref
        cn = 2.0 * ((d_a / d_ref)**2 - (d_f / d_ref)**2)
        
        # CP location from fore end
        if abs(d_a - d_f) < 1e-6:
            cp_loc = self.length / 2.0
        else:
            # Barrowman formula for conical transition CP
            # x = L/3 * [1 + (1 - df/da)/(1 - (df/da)^2)]
            ratio = d_f / d_a
            cp_loc = (self.length / 3.0) * (1.0 + (1.0 - ratio) / (1.0 - ratio**2 + 1e-9))
            
        return (cn, self._position + cp_loc)

    def _props_dict(self):
        return {"shape": self.shape, "length": self.length,
                "fore_diameter": self.fore_diameter, "aft_diameter": self.aft_diameter,
                "wall_thickness": self.wall_thickness}


class TrapezoidalFinSet(RocketComponent):
    component_type = "Trapezoidal Fins"
    category = "Fins"

    def __init__(self, name="Fin Set"):
        super().__init__(name)
        self.fin_count = 4
        self.root_chord = 0.10
        self.tip_chord = 0.05
        self.height = 0.05
        self.sweep_angle = 30.0
        self.thickness = 0.003
        self.cross_section = "Rounded"  # Square/Rounded/Airfoil
        self.material = "Plywood (Birch)"
        self.color = "#d94f3b"

    def component_length(self):
        return self.root_chord

    def outer_diameter(self):
        return 0.0

    def _calc_mass(self):
        density = self._get_density()
        area = 0.5 * (self.root_chord + self.tip_chord) * self.height
        return area * self.thickness * density * self.fin_count

    def cp_contribution(self, d_ref: float):
        """Full Barrowman method for trapezoidal fin CP."""
        if self.root_chord <= 0 or self.height <= 0 or d_ref <= 0:
            return (0.0, 0.0)

        Cr = self.root_chord
        Ct = self.tip_chord
        a = self.height  # span from body wall
        N = self.fin_count

        # Find body radius at fin position
        R = 0.0
        if self.parent is not None:
            R = self.parent.outer_diameter() / 2.0
        if R <= 0: R = d_ref / 2.0

        s_total = R + a  # semi-span from centerline

        # Sweep length (leading edge offset)
        sweep_len = a * math.tan(math.radians(self.sweep_angle)) if self.sweep_angle > 0 else 0

        # Mid-chord line sweep
        lm = sweep_len + (Ct / 2.0) - (Cr / 2.0)
        
        # Mid-chord line length l = sqrt(a^2 + lm^2)
        l_mid = math.sqrt(a**2 + lm**2)

        # Barrowman CN_alpha for N fins (referenced to d_ref)
        denom = (self.root_chord + self.tip_chord)
        if denom < 1e-6:
            return 0.0, 0.0
            
        try:
            val_to_sqrt = 1.0 + (2.0 * l_mid / denom)**2
            cn_alpha = (4.0 * N * (a / d_ref)**2) / (1.0 + math.sqrt(max(0, val_to_sqrt)))
            if not math.isfinite(cn_alpha): cn_alpha = 0.0
        except:
            cn_alpha = 0.0
            
        # CP of fins from root LE
        if (self.root_chord + self.tip_chord) > 1e-6:
            x_f = (sweep_len * (self.root_chord + 2.0 * self.tip_chord) / (3.0 * (self.root_chord + self.tip_chord))
                   + (1.0 / 6.0) * (self.root_chord + self.tip_chord - self.root_chord * self.tip_chord / (self.root_chord + self.tip_chord)))
        else:
            x_f = self.root_chord * 0.25
            
        if not math.isfinite(x_f): x_f = 0.0
        
        # Interference factor: K_fb = 1 + R/s_total
        K_fb = 1.0 + R / s_total
        cn = cn_alpha * K_fb
        return (cn, self._position + x_f)

    def _props_dict(self):
        return {"fin_count": self.fin_count, "root_chord": self.root_chord,
                "tip_chord": self.tip_chord, "height": self.height,
                "sweep_angle": self.sweep_angle, "thickness": self.thickness,
                "cross_section": self.cross_section}


# ═══════════════════════════════════════════════════════════════
#  INNER COMPONENTS
# ═══════════════════════════════════════════════════════════════

class InnerTube(RocketComponent):
    component_type = "Inner Tube"
    category = "Inner"

    def __init__(self, name="Motor Mount"):
        super().__init__(name)
        self.length = 0.15
        self.outer_diameter_val = 0.029
        self.inner_diameter = 0.027
        self.material = "Kraft Phenolic"
        self.is_motor_mount = True
        self.color = "#8B7355"

    def component_length(self):
        return self.length

    def outer_diameter(self):
        return self.outer_diameter_val

    def _calc_mass(self):
        density = self._get_density()
        r_o, r_i = self.outer_diameter_val / 2, self.inner_diameter / 2
        return math.pi * (r_o**2 - r_i**2) * self.length * density

    def _props_dict(self):
        return {"length": self.length, "outer_diameter": self.outer_diameter_val,
                "inner_diameter": self.inner_diameter, "is_motor_mount": self.is_motor_mount}


class CenteringRing(RocketComponent):
    component_type = "Centering Ring"
    category = "Inner"

    def __init__(self, name="Centering Ring"):
        super().__init__(name)
        self.outer_diameter_val = 0.063
        self.inner_diameter = 0.029
        self.thickness = 0.003
        self.material = "Plywood (Birch)"
        self.color = "#c4a66a"

    def component_length(self):
        return self.thickness

    def _calc_mass(self):
        density = self._get_density()
        r_o, r_i = self.outer_diameter_val / 2, self.inner_diameter / 2
        return math.pi * (r_o**2 - r_i**2) * self.thickness * density

    def _props_dict(self):
        return {"outer_diameter": self.outer_diameter_val,
                "inner_diameter": self.inner_diameter, "thickness": self.thickness}


class Bulkhead(RocketComponent):
    component_type = "Bulkhead"
    category = "Inner"

    def __init__(self, name="Bulkhead"):
        super().__init__(name)
        self.diameter = 0.063
        self.thickness = 0.003

    def component_length(self):
        return self.thickness

    def _calc_mass(self):
        density = self._get_density()
        return math.pi * (self.diameter / 2)**2 * self.thickness * density

    def _props_dict(self):
        return {"diameter": self.diameter, "thickness": self.thickness}


class TubeCoupler(RocketComponent):
    component_type = "Tube Coupler"
    category = "Body"

    def __init__(self, name="Tube Coupler"):
        super().__init__(name)
        self.outer_diameter_val = 0.060
        self.inner_diameter = 0.056
        self.length = 0.05
        self.thickness = 0.002

    def outer_diameter(self):
        return self.outer_diameter_val

    def component_length(self):
        return self.length

    def _calc_mass(self):
        density = self._get_density()
        r_o, r_i = self.outer_diameter_val / 2, self.inner_diameter / 2
        return math.pi * (r_o**2 - r_i**2) * self.length * density

    def _props_dict(self):
        return {"outer_diameter": self.outer_diameter_val,
                "inner_diameter": self.inner_diameter,
                "length": self.length}


class EngineBlock(RocketComponent):
    component_type = "Engine Block"
    category = "Inner"

    def __init__(self, name="Engine Block"):
        super().__init__(name)
        self.diameter = 0.029
        self.thickness = 0.003
        self.material = "Plywood (Birch)"
        self.color = "#444444"

    def component_length(self):
        return self.thickness

    def _calc_mass(self):
        density = self._get_density()
        return math.pi * (self.diameter / 2)**2 * self.thickness * density

    def _props_dict(self):
        return {"diameter": self.diameter, "thickness": self.thickness}


# ═══════════════════════════════════════════════════════════════
#  RECOVERY
# ═══════════════════════════════════════════════════════════════

class Parachute(RocketComponent):
    component_type = "Parachute"
    category = "Recovery"

    def __init__(self, name="Parachute"):
        super().__init__(name)
        self.diameter = 0.60
        self.cd = 1.5
        self.material = "Ripstop Nylon"
        self.line_count = 6
        self.line_length = 0.45
        self.packed_length = 0.05
        self.color = "#ff6633"

    def component_length(self):
        return self.packed_length

    def _calc_mass(self):
        mat = MATERIALS.get(self.material, MATERIALS["Cardboard"])
        area = math.pi * (self.diameter / 2)**2
        canopy = area * 0.00005 * mat["density"]
        lines = self.line_count * self.line_length * 0.001
        return canopy + lines

    def _props_dict(self):
        return {"diameter": self.diameter, "cd": self.cd, "line_count": self.line_count,
                "line_length": self.line_length, "packed_length": self.packed_length}


class ShockCord(RocketComponent):
    component_type = "Shock Cord"
    category = "Recovery"

    def __init__(self, name="Shock Cord"):
        super().__init__(name)
        self.length = 0.60
        self.material = "Nylon"
        self.color = "#dddd44"

    def component_length(self):
        return 0.02

    def _calc_mass(self):
        return self.length * 0.005

    def _props_dict(self):
        return {"length": self.length}


class MassComponent(RocketComponent):
    component_type = "Mass Component"
    category = "Mass"

    def __init__(self, name="Mass Object"):
        super().__init__(name)
        self.mass = 0.010
        self.mass_length = 0.02
        self.color = "#999999"

    def component_length(self):
        return self.mass_length

    def _calc_mass(self):
        return self.mass

    def _props_dict(self):
        return {"mass": self.mass, "mass_length": self.mass_length}


class LaunchLug(RocketComponent):
    component_type = "Launch Lug"
    category = "Mass"

    def __init__(self, name="Launch Lug"):
        super().__init__(name)
        self.length = 0.05
        self.outer_diameter_val = 0.006
        self.inner_diameter = 0.005
        self.material = "Cardboard"
        self.color = "#888888"

    def component_length(self):
        return self.length

    def _calc_mass(self):
        density = self._get_density()
        r_o, r_i = self.outer_diameter_val / 2, self.inner_diameter / 2
        return math.pi * (r_o**2 - r_i**2) * self.length * density

    def _props_dict(self):
        return {"length": self.length, "outer_diameter": self.outer_diameter_val,
                "inner_diameter": self.inner_diameter}


class RailButton(RocketComponent):
    component_type = "Rail Button"
    category = "Mass"

    def __init__(self, name="Rail Button"):
        super().__init__(name)
        self.height = 0.01
        self.base_diameter = 0.01
        self.material = "ABS Plastic"
        self.color = "#555555"

    def component_length(self):
        return self.height

    def _calc_mass(self):
        return 0.005

    def _props_dict(self):
        return {"height": self.height, "base_diameter": self.base_diameter}



# ═══════════════════════════════════════════════════════════════
#  PROPULSION COMPONENTS
# ═══════════════════════════════════════════════════════════════

class Nozzle(RocketComponent):
    """Rocket nozzle component with three configurable types.

    Types:
        Convergent-Divergent (De Laval): Standard CD nozzle with throat/exit
            geometry for mass, inertia, and thrust coefficient calculation.
        Boat-Tail: Aft taper for base drag reduction — geometry only, no
            thrust modeling.
        Full Propulsion: CD nozzle + chamber pressure, exit pressure, and
            altitude-compensated thrust coefficient tied to the Motor.
    """
    component_type = "Nozzle"
    can_have_children = False
    category = "Body"

    def __init__(self, name="Nozzle"):
        super().__init__(name)
        self.nozzle_type = "Convergent-Divergent"
        self.length = 0.06
        self.throat_diameter = 0.020
        self.exit_diameter = 0.040
        self.inlet_diameter = 0.050
        self.half_angle = 15.0       # divergent half-angle (degrees)
        self.wall_thickness = 0.002
        self.material = "Aluminum 6061"
        self.color = "#555555"

        # Full Propulsion only
        self.design_chamber_pressure = 5.0e6   # Pa
        self.design_exit_pressure = 101325.0   # Pa (sea-level)

    @property
    def expansion_ratio(self) -> float:
        """Area ratio: A_exit / A_throat."""
        if self.throat_diameter <= 0:
            return 1.0
        return (self.exit_diameter / self.throat_diameter) ** 2

    @property
    def throat_area(self) -> float:
        return math.pi * (self.throat_diameter / 2) ** 2

    @property
    def exit_area(self) -> float:
        return math.pi * (self.exit_diameter / 2) ** 2

    def component_length(self):
        return self.length

    def outer_diameter(self):
        return max(self.exit_diameter, self.inlet_diameter)

    def _calc_mass(self):
        """Mass from frustum shell geometry (convergent + divergent sections)."""
        density = self._get_density()
        if self.nozzle_type == "Boat-Tail":
            # Single frustum from inlet to exit
            r1, r2 = self.inlet_diameter / 2, self.exit_diameter / 2
            s = math.sqrt((r1 - r2) ** 2 + self.length ** 2)
            area = math.pi * (r1 + r2) * s
            return area * self.wall_thickness * density
        else:
            # Convergent section: inlet → throat (40% of length)
            L_conv = self.length * 0.4
            L_div = self.length * 0.6
            r_in = self.inlet_diameter / 2
            r_th = self.throat_diameter / 2
            r_ex = self.exit_diameter / 2

            s1 = math.sqrt((r_in - r_th) ** 2 + L_conv ** 2)
            a1 = math.pi * (r_in + r_th) * s1

            s2 = math.sqrt((r_ex - r_th) ** 2 + L_div ** 2)
            a2 = math.pi * (r_th + r_ex) * s2

            return (a1 + a2) * self.wall_thickness * density

    def cp_contribution(self, d_ref: float):
        """Nozzle CP contribution.

        In standard Barrowman analysis the nozzle is an *internal*
        propulsion component, not an exposed external aerodynamic
        surface.  Its converging/diverging geometry does not produce
        normal force in the same way a body transition does, so we
        return zero contribution.  Only a Boat-Tail (pure external
        taper for base-drag reduction) contributes a small negative CN.
        """
        if d_ref <= 0:
            return (0.0, 0.0)

        if self.nozzle_type == "Boat-Tail":
            d_f = self.inlet_diameter
            d_a = self.exit_diameter
            cn = 2.0 * ((d_a / d_ref) ** 2 - (d_f / d_ref) ** 2)
            return (cn, self._position + self.length / 3.0)

        # CD / Full-Propulsion nozzles: no aerodynamic CN contribution
        return (0.0, 0.0)

    def _props_dict(self):
        d = {
            "nozzle_type": self.nozzle_type,
            "length": self.length,
            "throat_diameter": self.throat_diameter,
            "exit_diameter": self.exit_diameter,
            "inlet_diameter": self.inlet_diameter,
            "half_angle": self.half_angle,
            "wall_thickness": self.wall_thickness,
        }
        if self.nozzle_type == "Full Propulsion":
            d["design_chamber_pressure"] = self.design_chamber_pressure
            d["design_exit_pressure"] = self.design_exit_pressure
        return d


# ═══════════════════════════════════════════════════════════════
#  STAGE & ASSEMBLY
# ═══════════════════════════════════════════════════════════════

class Stage(RocketComponent):
    component_type = "Stage"
    can_have_children = True
    category = "Structure"

    def __init__(self, name="Sustainer"):
        super().__init__(name)
        self.color = "#58a6ff"
        self.separation_delay = 0.0
        self.separation_event = "burnout"

    def component_length(self):
        # Total length of a stage is the sum of its structural components
        total = 0.0
        for c in self.children:
            if isinstance(c, (BodyTube, NoseCone, Transition, Nozzle)):
                total += c.component_length()
        return total

    def outer_diameter(self):
        for c in self.children:
            d = c.outer_diameter()
            if d > 0:
                return d
        return 0.0

    def _props_dict(self):
        return {"separation_delay": self.separation_delay,
                "separation_event": self.separation_event}


# Component type registry for deserialization
COMPONENT_TYPES = {
    "Nose Cone": NoseCone, "Body Tube": BodyTube, "Transition": Transition,
    "Trapezoidal Fins": TrapezoidalFinSet, "Inner Tube": InnerTube,
    "Centering Ring": CenteringRing, "Bulkhead": Bulkhead,
    "Engine Block": EngineBlock, "Parachute": Parachute,
    "Shock Cord": ShockCord, "Mass Component": MassComponent,
    "Launch Lug": LaunchLug, "Rail Button": RailButton, "Stage": Stage,
    "Tube Coupler": TubeCoupler, "Nozzle": Nozzle,
}


class RocketAssembly:
    """Top-level rocket containing stages and their components."""

    def __init__(self):
        self.name = "Untitled Rocket"
        self.stages: list[Stage] = []
        self.reference_diameter: Optional[float] = None
        self.add_stage("Sustainer")

    def add_stage(self, name="New Stage") -> Stage:
        stage = Stage(name)
        stage.parent = None
        self.stages.append(stage)
        self._recompute_positions()
        return stage

    def remove_stage(self, stage: Stage):
        if stage in self.stages and len(self.stages) > 1:
            self.stages.remove(stage)
            self._recompute_positions()

    def add_component(self, parent, component: RocketComponent):
        if parent is None:
            if self.stages:
                parent = self.stages[0]
            else:
                return
        component.parent = parent
        parent.children.append(component)
        self._recompute_positions()
        logger.debug(f"Added {component.component_type}: {component.name}")

    def remove_component(self, component: RocketComponent):
        if component.parent:
            component.parent.children.remove(component)
            component.parent = None
            self._recompute_positions()

    def move_up(self, component: RocketComponent):
        if component.parent:
            siblings = component.parent.children
            idx = siblings.index(component)
            if idx > 0:
                siblings[idx], siblings[idx - 1] = siblings[idx - 1], siblings[idx]
                self._recompute_positions()

    def move_down(self, component: RocketComponent):
        if component.parent:
            siblings = component.parent.children
            idx = siblings.index(component)
            if idx < len(siblings) - 1:
                siblings[idx], siblings[idx + 1] = siblings[idx + 1], siblings[idx]
                self._recompute_positions()

    def duplicate_component(self, component: RocketComponent) -> Optional[RocketComponent]:
        if not component.parent:
            return None
        import copy
        dup = copy.deepcopy(component)
        dup.id = str(uuid.uuid4())[:8]
        dup.name = f"{component.name} (copy)"
        parent = component.parent
        idx = parent.children.index(component)
        parent.children.insert(idx + 1, dup)
        dup.parent = parent
        self._recompute_positions()
        return dup

    def _recompute_positions(self):
        pos = 0.0
        for stage in self.stages:
            stage._position = pos
            self._position_children(stage, pos)
            pos += stage.component_length()

    def _position_children(self, parent, start_pos):
        pos = start_pos
        for child in parent.children:
            # ── Detect explicit ORK positioning data ──
            # Only use ORK offsets if the component actually has them
            # (set by the ORK importer).  UI-created components don't
            # have these attributes and should use auto-positioning.
            ork_pos = getattr(child, '_ork_pos', None)
            has_explicit_pos = ork_pos is not None
            if not has_explicit_pos and hasattr(child, '_ork_offset'):
                ork_pos = child._ork_offset
                has_explicit_pos = True

            # Check both possible tag names for consistency
            rel_to = getattr(child, '_ork_rel', getattr(child, '_ork_rel_to', 'top')).lower()

            if has_explicit_pos and not (isinstance(child, (BodyTube, NoseCone, Transition, Nozzle)) and ork_pos == 0 and rel_to == "top"):
                # Explicit ORK position. OpenRocket offsets locate the child's
                # corresponding edge: bottom = child's aft edge from parent's aft
                # edge, middle = child's center from parent's center.
                if rel_to == "bottom":
                    child._position = (start_pos + parent.component_length()
                                       + ork_pos - child.component_length())
                elif rel_to == "top":
                    child._position = start_pos + ork_pos
                elif rel_to == "middle":
                    child._position = (start_pos + ork_pos
                                       + (parent.component_length() - child.component_length()) / 2.0)
                elif rel_to == "absolute":
                    child._position = ork_pos
                else:
                    child._position = start_pos + ork_pos
            elif isinstance(child, (BodyTube, NoseCone, Transition, Nozzle)):
                # Auto-stacking for structural components
                child._position = pos
                # Nose cones only push by their aerodynamic length
                if isinstance(child, NoseCone):
                    pos += child.length
                else:
                    pos += child.component_length()
            elif isinstance(child, TrapezoidalFinSet):
                # Auto-position fins at the trailing edge of the last
                # structural component (aft body tube end).  'pos' tracks the
                # running aft-edge of structural siblings; when the fin is a
                # child of a body tube (no structural siblings) 'pos' is still
                # the parent's START, so fall back to the parent's aft end.
                anchor = pos if pos > start_pos else start_pos + parent.component_length()
                child._position = max(0, anchor - child.root_chord)
            else:
                # Default for internal components (at parent start)
                child._position = start_pos

            if child.can_have_children:
                self._position_children(child, child._position)

    def total_length(self) -> float:
        """Total length is the distance from nose tip to the aftmost structural point."""
        max_z = 0.0
        for c in self.all_components():
            if c.category in ["Body", "Fins", "Structure"]:
                # End position of this component
                end_z = c.position + c.component_length()
                if end_z > max_z:
                    max_z = end_z
        return max_z - self.stages[0].position if self.stages else max_z

    def get_reference_diameter(self) -> float:
        ref = getattr(self, 'reference_diameter', None)
        if ref is not None and ref > 1e-6:
            return ref
        d = self.max_diameter()
        return d if d > 1e-6 else 0.1 # Absolute fallback to prevent div-by-zero

    def max_diameter(self) -> float:
        d = 0.0
        for s in self.stages:
            for c in s.children:
                d = max(d, c.outer_diameter())
        return d

    def total_mass(self) -> float:
        """Total mass, handling override_includes_children flag."""
        tm = 0.0
        for c in self.all_components():
            if isinstance(c, Stage):
                continue
            # Skip children if parent's override includes them
            if (c.parent and hasattr(c.parent, '_override_includes_children')
                    and c.parent._override_includes_children
                    and c.parent.override_mass is not None):
                continue
            tm += c.computed_mass()
        return tm

    def compute_cg(self) -> float:
        """Compute center of gravity, properly handling override_includes_children."""
        tm, wp = 0.0, 0.0
        for c in self.all_components():
            if isinstance(c, Stage):
                continue
            # Skip children if parent's override includes them
            if (c.parent and hasattr(c.parent, '_override_includes_children')
                    and c.parent._override_includes_children
                    and c.parent.override_mass is not None):
                continue
            m = c.computed_mass()
            if m > 0:
                tm += m
                wp += m * c.cg_position()
        return wp / tm if tm > 0 else 0.0

    def compute_cp(self) -> float:
        """Compute center of pressure using Barrowman method.

        Safety: if total CN is non-positive (no net restoring force)
        the rocket has no meaningful aerodynamic CP, so we fall back
        to the nose-tip position (0) rather than returning a negative
        or wildly-large value that would produce impossible stability
        margins.
        """
        cn_total, cp_weighted = 0.0, 0.0
        d_ref = self.get_reference_diameter()
        if d_ref <= 0:
            return 0.0

        for c in self.all_components():
            if isinstance(c, Stage):
                continue
            cn, cp = c.cp_contribution(d_ref)
            if abs(cn) > 1e-9:
                cn_total += cn
                cp_weighted += cn * cp

        if cn_total <= 1e-9:
            # No net positive normal force — CP is undefined / at nose
            return 0.0

        cp = cp_weighted / cn_total

        # Sanity clamp: CP must lie within the physical rocket body
        total_len = self.total_length()
        if total_len > 0:
            cp = max(0.0, min(cp, total_len))

        return cp

    def fin_count(self) -> int:
        for c in self.all_components():
            if isinstance(c, TrapezoidalFinSet):
                return c.fin_count
        return 0

    def all_components(self):
        for stage in self.stages:
            yield stage
            yield from self._iter_children(stage)

    def _iter_children(self, parent):
        for child in parent.children:
            yield child
            if child.can_have_children:
                yield from self._iter_children(child)

    def _cg_recurse(self, comp, total_mass, weighted_pos):
        pass  # handled in compute_cg via all_components

    def to_dict(self) -> dict:
        return {"name": self.name,
                "stages": [s.to_dict() for s in self.stages]}

    @classmethod
    def from_dict(cls, data: dict) -> 'RocketAssembly':
        asm = cls.__new__(cls)
        asm.name = data.get("name", "Rocket")
        asm.stages = []
        for sd in data.get("stages", []):
            stage = Stage(sd.get("name", "Stage"))
            stage.id = sd.get("id", stage.id)
            _load_children(stage, sd.get("children", []))
            asm.stages.append(stage)
        if not asm.stages:
            asm.add_stage("Sustainer")
        asm._recompute_positions()
        return asm


def _load_children(parent, children_data):
    for cd in children_data:
        ctype = cd.get("type", "")
        cls = COMPONENT_TYPES.get(ctype)
        if not cls:
            continue
        comp = cls(cd.get("name", ctype))
        comp.id = cd.get("id", comp.id)
        comp.material = cd.get("material", comp.material)
        comp.comment = cd.get("comment", "")
        comp.color = cd.get("color", comp.color)
        if "override_mass" in cd:
            comp.override_mass = cd["override_mass"]
        if "ork_pos" in cd:
            comp._ork_pos = cd["ork_pos"]
            comp._ork_rel = cd.get("ork_rel", "top")
        if "ork_density" in cd:
            comp._ork_density = cd["ork_density"]
        props = cd.get("properties", {})
        for k, v in props.items():
            # Never clobber methods (e.g. BodyTube saves "outer_diameter" but
            # stores the value in outer_diameter_val — outer_diameter() is a method)
            if callable(getattr(comp, k, None)):
                alt = k + "_val"
                if hasattr(comp, alt) and not callable(getattr(comp, alt)):
                    setattr(comp, alt, v)
                continue
            if hasattr(comp, k):
                setattr(comp, k, v)
        comp.parent = parent
        parent.children.append(comp)
        if comp.can_have_children and "children" in cd:
            _load_children(comp, cd["children"])
