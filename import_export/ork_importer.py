"""
K2 Aerospace — OpenRocket (.ork) Importer
Parses .ork files (ZIP containing XML) into K2 RocketAssembly.

Key design decisions:
  - Material density from ORK XML is stored on each component as `_ork_density`
    and used in `_calc_mass()` instead of K2's material database density.
  - Explicit `<overridemass>` tags are respected as `override_mass`.
  - `<overridesubcomponentsmass>` flag is tracked so mass rollup is correct.
  - We do NOT artificially compute masses from density — we let each component's
    `_calc_mass()` do the work, using the ORK density if available.
"""
import zipfile, logging, math
import xml.etree.ElementTree as ET
from pathlib import Path
from core.components import (
    RocketAssembly, Stage, NoseCone, BodyTube, Transition,
    TrapezoidalFinSet, InnerTube, CenteringRing, Bulkhead,
    EngineBlock, Parachute, ShockCord, MassComponent, LaunchLug, RailButton
)

logger = logging.getLogger("K2.ORKImport")

# Map OpenRocket XML tags to K2 component classes
TAG_MAP = {
    "nosecone": NoseCone,
    "bodytube": BodyTube,
    "transition": Transition,
    "trapezoidfinset": TrapezoidalFinSet,
    "ellipticalfinset": TrapezoidalFinSet,
    "freeformfinset": TrapezoidalFinSet,
    "innertube": InnerTube,
    "tubecoupler": InnerTube,
    "centeringring": CenteringRing,
    "bulkhead": Bulkhead,
    "engineblock": EngineBlock,
    "parachute": Parachute,
    "streamer": ShockCord,
    "shockcord": ShockCord,
    "masscomponent": MassComponent,
    "launchlug": LaunchLug,
    "railbutton": RailButton,
}

NOSE_SHAPE_MAP = {
    "ogive": "Ogive", "conical": "Conical", "ellipsoid": "Elliptical",
    "power": "Parabolic", "parabolic": "Parabolic", "haack": "Haack (LD)",
}


def _float(elem, tag, default=0.0):
    """Safely extract a float from a child element. Handles 'auto X.XX' format."""
    child = elem.find(tag)
    if child is not None and child.text:
        text = child.text.strip()
        # Handle "auto 0.0765" format from OpenRocket
        if text.startswith("auto"):
            text = text.replace("auto", "").strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            pass
    return default


def _text(elem, tag, default=""):
    child = elem.find(tag)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _map_material(ork_material):
    """Best-effort map from OpenRocket material names to K2 materials."""
    m = ork_material.lower()
    if "carbon" in m or "cf" in m:
        return "Carbon Fiber"
    elif "fiberglass" in m or "glass" in m:
        return "Fiberglass"
    elif "aluminum" in m or "aluminium" in m:
        return "Aluminum 6061"
    elif "balsa" in m:
        return "Balsa Wood"
    elif "birch" in m or "plywood" in m or "ply" in m:
        return "Plywood (Birch)"
    elif "phenolic" in m or "kraft" in m:
        return "Kraft Phenolic"
    elif "abs" in m or "plastic" in m or "pla" in m:
        return "ABS Plastic"
    elif "nylon" in m or "ripstop" in m:
        return "Ripstop Nylon"
    elif "polycarbonate" in m or "lexan" in m:
        return "Polycarbonate"
    elif "cardboard" in m or "paper" in m:
        return "Cardboard"
    elif "mylar" in m or "polyester" in m:
        return "Ripstop Nylon"  # closest match
    return "Cardboard"


def _parse_component(elem, tag_lower):
    """Parse a single OpenRocket XML component element into a K2 component."""
    comp_class = TAG_MAP.get(tag_lower)
    if comp_class is None:
        return None

    name = _text(elem, "name") or comp_class.component_type
    comp = comp_class(name)

    # ── Material & density ──
    mat_elem = elem.find("material")
    ork_density = None
    ork_mat_type = None
    if mat_elem is not None:
        mat_text = mat_elem.text or mat_elem.get("type", "")
        comp.material = _map_material(mat_text)
        ork_mat_type = mat_elem.get("type", "bulk")  # "bulk", "surface", "line"
        density_str = mat_elem.get("density", "")
        if density_str:
            try:
                ork_density = float(density_str)
            except ValueError:
                pass

    # Store ORK density and type for _calc_mass() to use
    comp._ork_density = ork_density
    comp._ork_mat_type = ork_mat_type

    # ── Mass override ──
    mass_elem = elem.find("overridemass")
    if mass_elem is not None and mass_elem.text:
        try:
            comp.override_mass = float(mass_elem.text)
        except ValueError:
            pass

    # Track whether override includes subcomponents
    override_sub = _text(elem, "overridesubcomponentsmass", "false")
    comp._override_includes_children = (override_sub.lower() == "true")

    # ── Positioning ──
    pos_elem = elem.find("position")
    if pos_elem is None:
        pos_elem = elem.find("axialoffset")
        
    if pos_elem is not None:
        try:
            val = float(pos_elem.text)
            comp._ork_pos = val
            # OpenRocket default logic: 
            # If positionrelativeto is missing:
            # - Fins default to "bottom"
            # - Internal components with negative values often imply "bottom" in practice
            rel_text = _text(elem, "positionrelativeto", "").lower()
            if not rel_text:
                if isinstance(comp, TrapezoidalFinSet):
                    rel_text = "bottom"
                elif val < 0:
                    rel_text = "bottom"
                else:
                    rel_text = "top"
            comp._ork_rel = rel_text
        except (ValueError, TypeError):
            comp._ork_pos = None
            comp._ork_rel = "top"
    else:
        comp._ork_pos = None
        comp._ork_rel = "top"

    # ── Type-specific geometry parsing ──
    if isinstance(comp, NoseCone):
        comp.length = _float(elem, "length", 0.15)
        aftrad = _float(elem, "aftradius", 0.033)
        comp.diameter = aftrad * 2 if aftrad > 0 else _float(elem, "aftouterdiameter", 0.066)
        comp.wall_thickness = _float(elem, "thickness", 0.002)
        comp.shoulder_length = _float(elem, "aftshoulderlength", 0.0)
        comp.shoulder_diameter = _float(elem, "aftshoulderradius", 0.0) * 2
        comp.shoulder_thickness = _float(elem, "aftshoulderthickness", 0.0)
        shape = _text(elem, "shape").lower()
        comp.shape = NOSE_SHAPE_MAP.get(shape, "Ogive")

    elif isinstance(comp, BodyTube):
        comp.length = _float(elem, "length", 0.30)
        comp.outer_diameter_val = _float(elem, "radius", 0.033) * 2
        if comp.outer_diameter_val <= 0:
            comp.outer_diameter_val = _float(elem, "outerradius", 0.033) * 2
        comp.inner_diameter = comp.outer_diameter_val - 2 * _float(elem, "thickness", 0.0015)

    elif isinstance(comp, Transition):
        comp.length = _float(elem, "length", 0.06)
        comp.fore_diameter = _float(elem, "foreradius", 0.038) * 2
        comp.aft_diameter = _float(elem, "aftradius", 0.033) * 2
        comp.wall_thickness = _float(elem, "thickness", 0.002)

    elif isinstance(comp, TrapezoidalFinSet):
        comp.fin_count = int(_float(elem, "fincount", 4))
        comp.root_chord = _float(elem, "rootchord", 0.10)
        comp.tip_chord = _float(elem, "tipchord", 0.05)
        comp.height = _float(elem, "height", 0.05)
        sweep = _float(elem, "sweeplength", 0.0)
        if sweep > 0 and comp.height > 0:
            comp.sweep_angle = math.degrees(math.atan(sweep / comp.height))
        comp.thickness = _float(elem, "thickness", 0.003)
        # Store ORK positioning metadata
        comp._ork_offset = _float(elem, "axialoffset", 0.0)
        comp._ork_rel_to = _text(elem, "positionrelativeto", "bottom").lower()
        comp._ork_offset = _float(elem, "axialoffset", 0.0)
        comp._ork_rel_to = _text(elem, "positionrelativeto", "bottom").lower()

    elif isinstance(comp, InnerTube):
        comp.length = _float(elem, "length", 0.15)
        comp.outer_diameter_val = _float(elem, "outerradius", 0.016) * 2
        thickness = _float(elem, "thickness", 0.0005)
        comp.inner_diameter = comp.outer_diameter_val - 2 * thickness
        if comp.inner_diameter <= 0:
            comp.inner_diameter = _float(elem, "innerradius", 0.0145) * 2
        motor_elem = elem.find("motormount")
        if motor_elem is not None:
            comp.is_motor_mount = True

    elif isinstance(comp, CenteringRing):
        comp.outer_diameter_val = _float(elem, "outerradius", 0.032) * 2
        comp.inner_diameter = _float(elem, "innerradius", 0.016) * 2
        comp.thickness = _float(elem, "length", 0.003)

    elif isinstance(comp, (Bulkhead, EngineBlock)):
        or_text = _text(elem, "outerradius")
        if or_text and or_text != "auto":
            try:
                comp.diameter = float(or_text) * 2
            except ValueError:
                comp.diameter = 0.063
        else:
            comp.diameter = 0.063  # auto-sized later
        comp.thickness = _float(elem, "length", 0.003)

    elif isinstance(comp, Parachute):
        comp.diameter = _float(elem, "diameter", 0.60)
        cd_val = _float(elem, "cd", 0.0)
        if cd_val > 0:
            comp.cd = cd_val
        else:
            comp.cd = 1.5  # default when "auto"
        comp.line_count = int(_float(elem, "linecount", 6))
        comp.line_length = _float(elem, "linelength", 0.45)

    elif isinstance(comp, ShockCord):
        comp.length = _float(elem, "cordlength", 0.60)

    elif isinstance(comp, MassComponent):
        comp.mass = _float(elem, "mass", 0.01)

    elif isinstance(comp, LaunchLug):
        comp.length = _float(elem, "length", 0.05)
        comp.outer_diameter_val = _float(elem, "radius", 0.003) * 2
        comp.inner_diameter = _float(elem, "innerradius", 0.0025) * 2

    return comp


def _parse_children(xml_elem, parent_comp, assembly):
    """Recursively parse child XML elements into K2 components."""
    for child_elem in xml_elem:
        tag = child_elem.tag.lower().replace("-", "").replace("_", "")

        if tag == "subcomponents":
            _parse_children(child_elem, parent_comp, assembly)
            continue

        if tag in TAG_MAP:
            comp = _parse_component(child_elem, tag)
            if comp:
                assembly.add_component(parent_comp, comp)
                # Parse nested children (e.g., inner components inside body tube)
                sub = child_elem.find("subcomponents")
                if sub is not None and comp.can_have_children:
                    _parse_children(sub, comp, assembly)
                elif comp.can_have_children:
                    _parse_children(child_elem, comp, assembly)


def _auto_size_bulkheads(assembly):
    """Set auto-sized bulkhead diameters to match their parent body tube."""
    from core.components import Bulkhead, EngineBlock, BodyTube, InnerTube
    for comp in assembly.all_components():
        if isinstance(comp, (Bulkhead, EngineBlock)):
            if abs(comp.diameter - 0.063) < 0.001:  # was set to auto default
                # Walk up to find enclosing body tube
                parent = comp.parent
                while parent is not None:
                    if isinstance(parent, BodyTube):
                        comp.diameter = parent.inner_diameter
                        break
                    elif isinstance(parent, InnerTube):
                        comp.diameter = parent.inner_diameter
                        break
                    parent = getattr(parent, 'parent', None)


def import_ork(filepath: str) -> RocketAssembly:
    """
    Import an OpenRocket .ork file and return a RocketAssembly.
    .ork files are ZIP archives containing rocket.ork (XML).
    """
    filepath = str(filepath)
    logger.info(f"Importing OpenRocket file: {filepath}")

    xml_content = None

    # Try as ZIP first
    try:
        with zipfile.ZipFile(filepath, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.ork') or name.endswith('.xml'):
                    xml_content = zf.read(name)
                    break
            if xml_content is None and zf.namelist():
                xml_content = zf.read(zf.namelist()[0])
    except zipfile.BadZipFile:
        # Try as plain XML
        with open(filepath, 'r', encoding='utf-8') as f:
            xml_content = f.read().encode('utf-8')

    if xml_content is None:
        raise ValueError("Could not read .ork file content")

    root = ET.fromstring(xml_content)

    # Navigate to the rocket element
    rocket_elem = root.find(".//rocket")
    if rocket_elem is None:
        rocket_elem = root

    assembly = RocketAssembly()
    assembly.stages.clear()

    # Get rocket name and reference diameter
    name = _text(rocket_elem, "name")
    if name:
        assembly.name = name
    
    ref_diam = _float(rocket_elem, "referencediameter", 0.0)
    if ref_diam > 0:
        assembly.reference_diameter = ref_diam

    # Find stages — ONLY look in rocket > subcomponents > stage
    # NOT in motorconfiguration > stage (those are motor config refs)
    stages_found = False
    rocket_subcomponents = rocket_elem.find("subcomponents")
    if rocket_subcomponents is not None:
        for child in rocket_subcomponents:
            if child.tag.lower() == "stage":
                stage_name = _text(child, "name") or "Sustainer"
                stage = assembly.add_stage(stage_name)
                sub = child.find("subcomponents")
                if sub is not None:
                    _parse_children(sub, stage, assembly)
                else:
                    _parse_children(child, stage, assembly)
                stages_found = True

    # If no explicit stages found, treat all components as single stage
    if not stages_found:
        stage = assembly.add_stage("Sustainer")
        _parse_children(rocket_elem, stage, assembly)

    # Auto-size bulkheads that have "auto" outer radius
    _auto_size_bulkheads(assembly)

    assembly._recompute_positions()

    total_comps = sum(1 for _ in assembly.all_components()) - len(assembly.stages)
    logger.info(f"Imported '{assembly.name}' — {len(assembly.stages)} stage(s), {total_comps} components")

    return assembly
