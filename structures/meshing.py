"""
K2 Aerospace — Structural Meshing (Gmsh)
==========================================
Generates shell meshes of the rocket for CalculiX FEM analysis.
Output: CalculiX .inp mesh file with node/element sets per component.
"""
from __future__ import annotations
import logging, math, re
from pathlib import Path
logger = logging.getLogger("K2.FEM.Meshing")

_REFINEMENT = {
    "coarse":     {"axial_per_cal": 4,  "circum": 16, "fin_div": 4},
    "medium":     {"axial_per_cal": 8,  "circum": 24, "fin_div": 8},
    "fine":       {"axial_per_cal": 16, "circum": 36, "fin_div": 12},
    "very_fine":  {"axial_per_cal": 32, "circum": 48, "fin_div": 20},
    "ultra_fine": {"axial_per_cal": 64, "circum": 72, "fin_div": 32},
}

def build_structural_mesh(
    assembly, output_path: Path, refinement="medium", element_type="shell",
    custom_circum: int | None = None,
    custom_axial_per_cal: int | None = None,
) -> Path:
    """Generate a CalculiX .inp mesh from a K2 RocketAssembly."""
    from core.components import NoseCone, BodyTube, Transition, TrapezoidalFinSet, Nozzle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref = _REFINEMENT.get(refinement, _REFINEMENT["medium"])
    # Custom overrides
    if custom_circum is not None and custom_circum > 0:
        ref = dict(ref)  # copy to avoid mutating the preset
        ref["circum"] = custom_circum
    if custom_axial_per_cal is not None and custom_axial_per_cal > 0:
        ref = dict(ref) if not isinstance(ref, dict) else ref
        ref["axial_per_cal"] = custom_axial_per_cal
    n_circ = ref["circum"]
    nodes, elements, nsets, elsets = [], [], {}, {}
    nid, eid = 1, 1
    total_L = assembly.total_length()
    if total_L <= 0:
        raise ValueError("Assembly has zero length.")
    d_ref = assembly.get_reference_diameter()

    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                nid, eid = _mesh_axisym(comp, nodes, elements, nsets, elsets,
                    nid, eid, n_circ, ref["axial_per_cal"], d_ref, "nose")
            elif isinstance(comp, BodyTube):
                nid, eid = _mesh_axisym(comp, nodes, elements, nsets, elsets,
                    nid, eid, n_circ, ref["axial_per_cal"], d_ref, "tube")
                for child in comp.children:
                    if isinstance(child, TrapezoidalFinSet):
                        nid, eid = _mesh_fin_set(child, comp, nodes, elements,
                            nsets, elsets, nid, eid, ref["fin_div"])
            elif isinstance(comp, Transition):
                nid, eid = _mesh_axisym(comp, nodes, elements, nsets, elsets,
                    nid, eid, n_circ, ref["axial_per_cal"], d_ref, "transition")
            elif isinstance(comp, Nozzle):
                if comp.nozzle_type == "Boat-Tail":
                    nid, eid = _mesh_axisym(comp, nodes, elements, nsets, elsets,
                        nid, eid, n_circ, ref["axial_per_cal"], d_ref, "nozzle_bt")
                else:
                    # Mesh convergent and divergent sections
                    nid, eid = _mesh_axisym(comp, nodes, elements, nsets, elsets,
                        nid, eid, n_circ, ref["axial_per_cal"], d_ref, "nozzle_cd")

    if not nodes:
        raise ValueError("No meshable components found.")
    _write_inp(output_path, nodes, elements, nsets, elsets)
    logger.info(f"Structural mesh: {len(nodes)} nodes, {len(elements)} elems → {output_path}")
    return output_path


def _radius_at_frac(comp, frac, comp_type):
    """Return radius at fractional position along a component."""
    if comp_type == "tube":
        return comp.outer_diameter_val / 2
    elif comp_type == "transition":
        r_fore = comp.fore_diameter / 2
        r_aft = comp.aft_diameter / 2
        return r_fore + frac * (r_aft - r_fore)
    elif comp_type == "nozzle_bt":
        # Boat-tail: linear taper from inlet to exit
        r_in = comp.inlet_diameter / 2
        r_ex = comp.exit_diameter / 2
        return r_in + frac * (r_ex - r_in)
    elif comp_type == "nozzle_cd":
        # Convergent-Divergent: inlet → throat (40%) → exit (60%)
        r_in = comp.inlet_diameter / 2
        r_th = comp.throat_diameter / 2
        r_ex = comp.exit_diameter / 2
        if frac <= 0.4:
            t = frac / 0.4
            return r_in + t * (r_th - r_in)
        else:
            t = (frac - 0.4) / 0.6
            return r_th + t * (r_ex - r_th)
    else:  # nose
        r = comp.diameter / 2
        shape = getattr(comp, 'shape', 'Ogive')
        L = comp.length
        if shape in ("Ogive", "Haack (LD)") and r > 0 and L > 0:
            rho = (r**2 + L**2) / (2 * r)
            d = frac * L
            r_local = math.sqrt(max(rho**2 - (L - d)**2, 0)) - (rho - r)
            return max(r_local, 0.001 * r)
        elif shape == "Elliptical":
            return max(r * math.sqrt(max(1 - (1 - frac)**2, 0)), 0.001 * r)
        else:
            return max(r * frac, 0.001 * r)


def _mesh_axisym(comp, nodes, elements, nsets, elsets,
                 nid, eid, n_circ, axial_per_cal, d_ref, comp_type):
    """Mesh an axi-symmetric component as quad shell elements."""
    L = comp.component_length()
    pos = comp.position
    n_axial = max(4, int(axial_per_cal * L / max(d_ref, 0.01)))
    cname = _unique_name(_safe_name(comp.name), nsets)
    cn, ce = [], []
    grid = []
    for i in range(n_axial + 1):
        frac = i / n_axial
        z = pos + frac * L
        r = _radius_at_frac(comp, frac, comp_type)
        ring = []
        for j in range(n_circ):
            theta = 2 * math.pi * j / n_circ
            nodes.append((nid, r * math.cos(theta), r * math.sin(theta), z))
            cn.append(nid); ring.append(nid); nid += 1
        grid.append(ring)
    for i in range(n_axial):
        for j in range(n_circ):
            j1 = (j + 1) % n_circ
            elements.append((eid, "S4R", [grid[i][j], grid[i][j1], grid[i+1][j1], grid[i+1][j]]))
            ce.append(eid); eid += 1
    nsets[cname] = cn; elsets[cname] = ce
    return nid, eid


def _mesh_fin_set(finset, parent, nodes, elements, nsets, elsets, nid, eid, fin_div):
    """Mesh trapezoidal fins as flat shell quads."""
    n_fins = finset.fin_count
    h, Cr, Ct = finset.height, finset.root_chord, finset.tip_chord
    sweep_deg = finset.sweep_angle
    pos = finset.position
    body_r = parent.outer_diameter_val / 2
    sweep_off = h * math.tan(math.radians(sweep_deg))
    ns, nc = max(3, fin_div), max(3, fin_div)
    cname = _unique_name(_safe_name(finset.name), nsets)
    cn, ce = [], []
    for fi in range(n_fins):
        ang = 2 * math.pi * fi / n_fins
        ca, sa = math.cos(ang), math.sin(ang)
        grid = []
        for si in range(ns + 1):
            sf = si / ns
            rl = body_r + sf * h
            chord = Cr + sf * (Ct - Cr)
            le_off = sf * sweep_off
            row = []
            for ci in range(nc + 1):
                cf = ci / nc
                z = pos + le_off + cf * chord
                nodes.append((nid, rl * ca, rl * sa, z))
                cn.append(nid); row.append(nid); nid += 1
            grid.append(row)
        for si in range(ns):
            for ci in range(nc):
                elements.append((eid, "S4R", [grid[si][ci], grid[si][ci+1],
                    grid[si+1][ci+1], grid[si+1][ci]]))
                ce.append(eid); eid += 1
    nsets[cname] = cn; elsets[cname] = ce
    return nid, eid


def _write_inp(fp: Path, nodes, elements, nsets, elsets):
    # Merge coincident nodes (connects components)
    merged_nodes = []
    node_map = {}
    TOL = 1e-6  # 1 micron tolerance (prevents merging nose tip ring)
    
    for nid, x, y, z in nodes:
        found = False
        for mnid, mx, my, mz in merged_nodes:
            if abs(x-mx) < TOL and abs(y-my) < TOL and abs(z-mz) < TOL:
                node_map[nid] = mnid
                found = True
                break
        if not found:
            merged_nodes.append((nid, x, y, z))
            node_map[nid] = nid
            
    # Update elements with merged node IDs
    new_elements = []
    for eid, etype, en in elements:
        new_elements.append((eid, etype, [node_map[n] for n in en]))
        
    # Update sets with merged node IDs
    for name in nsets:
        nsets[name] = list(set(node_map[n] for n in nsets[name]))
        
    nodes = merged_nodes
    elements = new_elements

    with open(fp, "w", encoding="ascii", errors="replace") as f:
        f.write("** K2 Aerospace — Structural Mesh\n**\n")
        f.write("*NODE, NSET=NALL\n")
        for nid, x, y, z in nodes:
            f.write(f"{nid}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write("*ELEMENT, TYPE=S4R, ELSET=EALL\n")
        for eid, _, en in elements:
            f.write(f"{eid}, {', '.join(str(n) for n in en)}\n")
        for name, ids in nsets.items():
            f.write(f"*NSET, NSET={name}\n")
            _write_set(f, ids)
        for name, ids in elsets.items():
            f.write(f"*ELSET, ELSET={name}\n")
            _write_set(f, ids)

        # Generate boundary node sets: NAFT (aft ring) and NFWD (forward tip)
        if nodes:
            z_vals = [n[3] for n in nodes]
            z_max = max(z_vals)
            z_min = min(z_vals)
            z_range = z_max - z_min if z_max > z_min else 1.0
            tol = 0.02 * z_range  # 2% tolerance

            aft_ids = [n[0] for n in nodes if abs(n[3] - z_max) < tol]
            fwd_ids = [n[0] for n in nodes if abs(n[3] - z_min) < tol]

            if aft_ids:
                f.write("*NSET, NSET=NAFT\n")
                _write_set(f, aft_ids)
            else:
                # Fallback: use last node
                f.write(f"*NSET, NSET=NAFT\n{nodes[-1][0]}\n")

            if fwd_ids:
                f.write("*NSET, NSET=NFWD\n")
                _write_set(f, fwd_ids)
            else:
                f.write(f"*NSET, NSET=NFWD\n{nodes[0][0]}\n")

        f.write("*SURFACE, NAME=INNER_SURFACE, TYPE=ELEMENT\nEALL, SNEG\n")
        f.write("*SURFACE, NAME=OUTER_SURFACE, TYPE=ELEMENT\nEALL, SPOS\n")


def _write_set(f, ids):
    for i, v in enumerate(ids):
        if i > 0 and i % 8 == 0: f.write("\n")
        elif i > 0: f.write(", ")
        f.write(str(v))
    f.write("\n")


def _safe_name(name: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_]', '_', name).strip('_')[:30]
    if not s: s = "COMP"
    if s[0].isdigit(): s = "C_" + s
    return s.upper()


def _unique_name(base: str, existing: dict) -> str:
    """Components sharing a display name (e.g. two 'Body Tube') would
    otherwise overwrite each other's node/element sets."""
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"
