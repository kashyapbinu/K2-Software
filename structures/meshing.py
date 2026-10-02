"""
K2 AeroSim — Structural Meshing
==================================
Generates the CalculiX shell mesh of the rocket (.inp with node/element sets
per component) and a MeshInfo describing it — per-component wall thickness
and material, the fin sets, and the rings where loads enter — which the
solver uses to write sections and loads.

Frame: z along the body axis from the nose tip (z = 0) aft; x, y lateral.

Connectivity rules — every part must share nodes with its neighbours, or
the solve is singular / the part floats:
  * every axisymmetric component is meshed as rings of ``n_circ`` nodes at
    the same angles, so adjacent components share their end rings;
  * a diameter step between components is closed by an annulus of elements;
  * ``n_circ`` is a multiple of every fin count and each body tube gets
    rings at its fins' root-chord stations, so every fin-root node IS a tube
    node (they used to coincide only at the aft corner — one node per fin).

Elements are S4, which CalculiX expands to C3D8I: it bends through the wall
thickness. S4R expands to C3D8R with ONE integration point, so the inner
and outer wall stresses came out identical and plate bending (fin roots,
local shell bending) read as zero.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("K2.FEM.Meshing")

ELEMENT_TYPE = "S4"

_REFINEMENT = {
    "coarse":     {"axial_per_cal": 4,  "circum": 16, "fin_div": 4},
    "medium":     {"axial_per_cal": 8,  "circum": 24, "fin_div": 8},
    "fine":       {"axial_per_cal": 16, "circum": 36, "fin_div": 12},
    "very_fine":  {"axial_per_cal": 32, "circum": 48, "fin_div": 20},
    "ultra_fine": {"axial_per_cal": 64, "circum": 72, "fin_div": 32},
}

_MERGE_TOL = 1e-6       # m — coincident-node tolerance (keeps the nose-tip ring)


@dataclass
class ShellSection:
    """One element/node set of the mesh and what it is made of."""
    name: str                   # ELSET and NSET name
    kind: str                   # nose | tube | transition | nozzle | fin | step
    thickness: float            # wall / fin thickness (m)
    material: str = ""          # the component's own material name
    component: object = None    # the RocketComponent (in-process only)
    fin_count: int = 0


@dataclass
class MeshInfo:
    sections: list = field(default_factory=list)       # [ShellSection]
    n_circ: int = 0
    nose_ring: list = field(default_factory=list)      # forward-most ring (nose tip)
    aft_ring: list = field(default_factory=list)       # aft-most airframe ring
    thrust_ring: list = field(default_factory=list)    # aft ring of the aft-most body tube
    body_stations: list = field(default_factory=list)  # [(z, [node ids])] airframe rings
    unattached_fin_roots: int = 0
    pieces: list = field(default_factory=list)         # [[set names]] per connected piece

    @property
    def fin_sections(self):
        return [s for s in self.sections if s.kind == "fin"]

    def section(self, name):
        return next((s for s in self.sections if s.name == name), None)


def build_structural_mesh(
    assembly, output_path: Path, refinement="medium", element_type="shell",
    custom_circum: int | None = None,
    custom_axial_per_cal: int | None = None,
) -> Path:
    """Generate a CalculiX .inp mesh from a K2 RocketAssembly."""
    return build_structural_mesh_info(assembly, output_path, refinement, element_type,
                                      custom_circum, custom_axial_per_cal)[0]


def build_structural_mesh_info(
    assembly, output_path: Path, refinement="medium", element_type="shell",
    custom_circum: int | None = None,
    custom_axial_per_cal: int | None = None,
):
    """Mesh the assembly; returns (path to the .inp, MeshInfo)."""
    from core.components import NoseCone, BodyTube, Transition, TrapezoidalFinSet, Nozzle
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ref = dict(_REFINEMENT.get(refinement, _REFINEMENT["medium"]))
    if custom_circum is not None and custom_circum > 0:
        ref["circum"] = custom_circum
    if custom_axial_per_cal is not None and custom_axial_per_cal > 0:
        ref["axial_per_cal"] = custom_axial_per_cal
    total_L = assembly.total_length()
    if total_L <= 0:
        raise ValueError("Assembly has zero length.")
    d_ref = assembly.get_reference_diameter()

    def fins_of(tube):
        return [c for c in tube.children
                if isinstance(c, TrapezoidalFinSet) and c.fin_count > 0
                and c.root_chord > 0 and c.height > 0]

    parts = []
    for stage in assembly.stages:
        for comp in stage.children:
            if isinstance(comp, NoseCone):
                parts.append((comp, "nose"))
            elif isinstance(comp, BodyTube):
                parts.append((comp, "tube"))
            elif isinstance(comp, Transition):
                parts.append((comp, "transition"))
            elif isinstance(comp, Nozzle):
                parts.append((comp, "nozzle_bt" if comp.nozzle_type == "Boat-Tail"
                              else "nozzle_cd"))
    fin_counts = [fs.fin_count for comp, kind in parts if kind == "tube"
                  for fs in fins_of(comp)]
    n_circ = _ring_count(ref["circum"], fin_counts)
    if n_circ != ref["circum"]:
        logger.info(f"Circumferential divisions {ref['circum']} -> {n_circ} so every "
                    f"fin root lies on a ring node (fin counts {sorted(set(fin_counts))})")

    b = _Builder(n_circ)
    prev = None                 # aft end of the previous airframe component
    tube_rings = []             # (aft z, aft ring) of every body tube
    for comp, kind in parts:
        L = comp.component_length()
        if L <= 0:
            continue
        z0 = comp.position
        # A small axial gap / overlap to the previous component is snapped
        # shut; otherwise the two would share no nodes.
        if prev is not None and abs(z0 - prev[0]) < 0.05 * max(d_ref, 0.01):
            L += z0 - prev[0]
            z0 = prev[0]
        n_axial = max(4, int(ref["axial_per_cal"] * L / max(d_ref, 0.01)))
        extra = []
        if kind == "tube":
            for fs in fins_of(comp):
                nc = max(3, ref["fin_div"])
                extra += [fs.position + (ci / nc) * fs.root_chord for ci in range(nc + 1)]
        zs = _stations(z0, L, n_axial, extra)
        sec = ShellSection(_unique_name(_safe_name(comp.name), b.nsets), _section_kind(kind),
                           _wall_thickness(comp, kind), getattr(comp, "material", ""), comp)
        first, last = b.axisym(sec, zs, lambda z: _radius_at_frac(comp, (z - z0) / L, kind))
        if prev is not None and abs(first[0] - prev[0]) < 1e-9:
            b.join(prev, first, sec)
        prev = last
        if kind == "tube":
            tube_rings.append(last)
            for fs in fins_of(comp):
                fsec = ShellSection(_unique_name(_safe_name(fs.name), b.nsets), "fin",
                                    fs.thickness if fs.thickness > 0 else 0.003,
                                    getattr(fs, "material", ""), fs, fs.fin_count)
                b.fin_set(fsec, fs, comp.outer_diameter_val / 2, max(3, ref["fin_div"]))

    if not b.nodes:
        raise ValueError("No meshable components found.")

    info = MeshInfo(sections=b.sections, n_circ=n_circ)
    nodes, node_map = _merge_nodes(b.nodes)
    elements = []
    for eid, en in b.elements:
        en = [node_map[n] for n in en]
        if len(set(en)) == 4:
            elements.append((eid, en))
    nsets = {name: sorted({node_map[n] for n in ids}) for name, ids in b.nsets.items()}
    elsets = dict(b.elsets)
    remap = lambda ring: [node_map[n] for n in ring]
    info.nose_ring = remap(b.first_ring[1]) if b.first_ring else []
    info.aft_ring = remap(prev[1]) if prev else []
    info.thrust_ring = remap(max(tube_rings, key=lambda r: r[0])[1]) if tube_rings \
        else list(info.aft_ring)
    body_nodes = set()
    for sec in b.sections:
        if sec.kind != "fin":
            body_nodes.update(nsets[sec.name])
    stations = {}
    for z, ring in b.rings:
        stations.setdefault(round(z, 9), set()).update(remap(ring))
    info.body_stations = [(z, sorted(ids)) for z, ids in sorted(stations.items())]
    roots = [node_map[n] for n in b.fin_roots]
    info.unattached_fin_roots = sum(1 for n in roots if n not in body_nodes)
    if info.unattached_fin_roots:
        logger.warning(f"{info.unattached_fin_roots} of {len(roots)} fin-root nodes are not "
                       "on the body tube (fin overhangs the tube end?)")
    info.pieces = _pieces(elements, {e: s.name for s in b.sections
                                     for e in elsets.get(s.name, ())})
    if len(info.pieces) > 1:
        logger.warning("Structural mesh is in %d unconnected pieces: %s", len(info.pieces),
                       "; ".join(", ".join(p) for p in info.pieces))

    _write_inp(output_path, nodes, elements, nsets, elsets, info)
    logger.info(f"Structural mesh: {len(nodes)} nodes, {len(elements)} {ELEMENT_TYPE} elems "
                f"→ {output_path}")
    return output_path, info


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _section_kind(kind):
    return {"nozzle_bt": "nozzle", "nozzle_cd": "nozzle"}.get(kind, kind)


def _wall_thickness(comp, kind):
    if kind == "tube":
        t = (comp.outer_diameter_val - comp.inner_diameter) / 2
    else:
        t = getattr(comp, "wall_thickness", 0.0)
    return t if t and t > 0 else 0.002


def _ring_count(circum, fin_counts):
    """Smallest multiple of every fin count that is ≥ ``circum`` (and ≥ 8)."""
    n = max(int(circum), 8)
    step = 1
    for k in fin_counts:
        step = step * k // math.gcd(step, k)
    return int(math.ceil(n / step) * step)


def _stations(z0, L, n, extra=()):
    """Axial ring stations of a component: uniform, plus the ``extra``
    stations inside it (fin-root chord points). Uniform stations closer than
    0.3 of the spacing to an extra one are dropped so no sliver rows form."""
    z1 = z0 + L
    base = [z0 + L * i / n for i in range(n)] + [z1]
    ins = sorted(e for e in extra if z0 + 1e-9 < e < z1 - 1e-9)
    if ins:
        dz = L / n
        base = [z for k, z in enumerate(base)
                if k in (0, n) or min(abs(z - e) for e in ins) > 0.3 * dz]
    zs = sorted(base + ins)
    out = [zs[0]]
    for z in zs[1:]:
        if z - out[-1] > 1e-9:
            out.append(z)
    out[-1] = z1
    return out


def _radius_at_frac(comp, frac, comp_type):
    """Return radius at fractional position along a component."""
    frac = min(max(frac, 0.0), 1.0)
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


# ── Mesh builder ──────────────────────────────────────────────────────────────

class _Builder:
    def __init__(self, n_circ):
        self.n_circ = n_circ
        self.nodes, self.elements = [], []     # [(nid, x, y, z)], [(eid, [n1..n4])]
        self.nsets, self.elsets = {}, {}
        self.sections = []
        self.rings = []                        # [(z, ring node ids)] airframe rings
        self.first_ring = None
        self.fin_roots = []
        self._nid = self._eid = 1

    def _node(self, x, y, z):
        nid = self._nid
        self.nodes.append((nid, x, y, z))
        self._nid += 1
        return nid

    def _elem(self, sec, en):
        eid = self._eid
        self.elements.append((eid, en))
        self.elsets.setdefault(sec.name, []).append(eid)
        self._eid += 1

    def _ring(self, sec, z, r):
        ring = []
        for j in range(self.n_circ):
            t = 2 * math.pi * j / self.n_circ
            ring.append(self._node(r * math.cos(t), r * math.sin(t), z))
        self.nsets.setdefault(sec.name, []).extend(ring)
        self.rings.append((z, ring))
        return ring

    def axisym(self, sec, zs, radius):
        """Rings at stations ``zs``; returns the (z, ring, r) of both ends."""
        self.sections.append(sec)
        grid = [(z, self._ring(sec, z, radius(z)), radius(z)) for z in zs]
        if self.first_ring is None:
            self.first_ring = grid[0][:2]
        n = self.n_circ
        for (_, a, _), (_, b, _) in zip(grid, grid[1:]):
            for j in range(n):
                j1 = (j + 1) % n
                self._elem(sec, [a[j], a[j1], b[j1], b[j]])
        return grid[0], grid[-1]

    def join(self, prev, first, sec):
        """Connect two components' end rings (same z). Radii within 2 % are
        snapped together — imported designs often differ by a fraction of a
        millimetre, which would leave the parts unconnected, and an annulus
        that thin is a degenerate element. A real diameter step is closed
        with an annulus."""
        (_, a, ra), (_, b, rb) = prev, first
        if abs(ra - rb) <= _MERGE_TOL:
            return
        if abs(ra - rb) < 0.02 * max(ra, rb):
            for na, nb in zip(a, b):
                self.nodes[nb - 1] = (nb,) + self.nodes[na - 1][1:]
            return
        step = ShellSection(_unique_name("STEP", self.nsets), "step", sec.thickness,
                            sec.material, sec.component)
        self.sections.append(step)
        self.nsets.setdefault(step.name, []).extend(a + b)
        n = self.n_circ
        for j in range(n):
            j1 = (j + 1) % n
            self._elem(step, [a[j], a[j1], b[j1], b[j]])

    def fin_set(self, sec, finset, body_r, div):
        """Flat quads per fin; fin k sits at angle 2πk/N, its root row on the
        body surface at the tube's fin stations."""
        self.sections.append(sec)
        n_fins = finset.fin_count
        h, Cr, Ct = finset.height, finset.root_chord, finset.tip_chord
        sweep_off = h * math.tan(math.radians(finset.sweep_angle))
        pos = finset.position
        ns = nc = max(3, div)
        cn = self.nsets.setdefault(sec.name, [])
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
                    z = pos + le_off + (ci / nc) * chord
                    nid = self._node(rl * ca, rl * sa, z)
                    row.append(nid)
                    if si == 0:
                        self.fin_roots.append(nid)
                cn.extend(row)
                grid.append(row)
            for si in range(ns):
                for ci in range(nc):
                    self._elem(sec, [grid[si][ci], grid[si][ci + 1],
                                     grid[si + 1][ci + 1], grid[si + 1][ci]])


def _pieces(elements, elset_of):
    """Connected pieces of the mesh (elements sharing a node), each as the
    sorted names of the sets it contains."""
    parent = {}

    def find(n):
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for _, en in elements:
        r = find(en[0])
        for n in en[1:]:
            parent[find(n)] = r
    groups = {}
    for eid, en in elements:
        groups.setdefault(find(en[0]), set()).add(elset_of.get(eid, "?"))
    return [sorted(g) for g in groups.values()]


def _merge_nodes(nodes, tol=_MERGE_TOL):
    """Merge coincident nodes (|Δ| < tol per axis) with a spatial hash —
    O(n), where the pairwise scan it replaces was O(n²)."""
    inv = 1.0 / tol
    grid, merged, node_map = {}, [], {}
    for nid, x, y, z in nodes:
        k = (math.floor(x * inv), math.floor(y * inv), math.floor(z * inv))
        hit = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for mid, mx, my, mz in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if abs(x - mx) < tol and abs(y - my) < tol and abs(z - mz) < tol:
                            hit = mid
                            break
                    if hit is not None:
                        break
                if hit is not None:
                    break
        if hit is None:
            grid.setdefault(k, []).append((nid, x, y, z))
            merged.append((nid, x, y, z))
            node_map[nid] = nid
        else:
            node_map[nid] = hit
    return merged, node_map


def _write_inp(fp: Path, nodes, elements, nsets, elsets, info: MeshInfo):
    with open(fp, "w", encoding="ascii", errors="replace") as f:
        f.write("** K2 AeroSim - Structural Mesh\n**\n")
        f.write("*NODE, NSET=NALL\n")
        for nid, x, y, z in nodes:
            f.write(f"{nid}, {x:.8e}, {y:.8e}, {z:.8e}\n")
        f.write(f"*ELEMENT, TYPE={ELEMENT_TYPE}, ELSET=EALL\n")
        for eid, en in elements:
            f.write(f"{eid}, {', '.join(str(n) for n in en)}\n")
        for name, ids in nsets.items():
            f.write(f"*NSET, NSET={name}\n")
            _write_set(f, ids)
        for name, ids in elsets.items():
            f.write(f"*ELSET, ELSET={name}\n")
            _write_set(f, ids)
        # NAFT: the thrust ring (clamped for the cantilever modal case).
        # It used to be every node within 2 % of the length of the tail —
        # several rings plus the trailing 11 nodes of every fin.
        for name, ids in (("NAFT", info.thrust_ring), ("NFWD", info.nose_ring)):
            if ids:
                f.write(f"*NSET, NSET={name}\n")
                _write_set(f, sorted(set(ids)))
        # All fin nodes, so post-processing can tell fin motion from airframe
        # motion without guessing from user-chosen component names.
        fin_ids = sorted({n for s in info.fin_sections for n in nsets.get(s.name, ())})
        if fin_ids:
            f.write("*NSET, NSET=NFINS\n")
            _write_set(f, fin_ids)


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
    otherwise overwrite each other's node/element sets. Reserved solver set
    names are never handed out."""
    reserved = {"NALL", "EALL", "NAFT", "NFWD", "NFINS", "NSUPPORT"}
    if base not in existing and base not in reserved:
        return base
    i = 2
    while f"{base}_{i}" in existing or f"{base}_{i}" in reserved:
        i += 1
    return f"{base}_{i}"
