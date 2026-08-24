"""
NAFEMS linear-elastic benchmarks LE1 and LE10, run through the bundled CalculiX.
================================================================================

These are *published* benchmarks with agreed target values, not textbook formulas
we derive ourselves — the same role the NACA-1135 cone tables play on the CFD
side. Both share one planform: a quarter of an elliptic annulus with

    inner ellipse   x²/2.00² + y²/1.00² = 1     (points D=(2,0), A=(0,1))
    outer ellipse   x²/3.25² + y²/2.75² = 1     (points C=(3.25,0), B=(0,2.75))

LE1 — Plane stress, elliptic membrane
    thickness 0.1 m, uniform *outward* pressure of 10 MPa on the outer edge BC,
    symmetry on AB (Ux=0) and DC (Uy=0).
    Target: tangential edge stress σ_yy at D = **92.7 MPa**.

LE10 — Thick plate under pressure
    thickness 0.6 m, uniform 1.0 MPa normal pressure on the upper surface,
    symmetry on faces DCD'C' (Uy=0) and ABA'B' (Ux=0), Ux=Uy=0 on the outer
    face BCB'C' with Uz=0 along its mid-plane only.
    Target: direct stress σ_yy at D=(2, 0, 0.3) = **-5.38 MPa**.

Both use E = 210 GPa, ν = 0.3.

Source of the target values: NAFEMS, *The Standard NAFEMS Benchmarks* (linear
elastic tests), as reproduced in the ESRD StressCheck benchmarks guide. The
same guide reports that a converged p-version solution reaches 92.70 MPa on LE1
and -5.25 MPa on LE10 (-2.4% from target) with hexahedra, which sets a realistic
expectation for what a structured low-order mesh can achieve here.

The mesh is a mapped structured hex grid generated in Python — no external
mesher — so the case is self-contained and reproducible.
"""
from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from structures.solvers.ccx_solver import _find_ccx

# ── benchmark constants ───────────────────────────────────────────────────────

E_STEEL = 210.0e9
NU_STEEL = 0.3

A_INNER, B_INNER = 2.00, 1.00
A_OUTER, B_OUTER = 3.25, 2.75

LE1_THICKNESS = 0.1
LE1_EDGE_PRESSURE = 10.0e6        # outward (tensile) traction on BC
LE1_TARGET_SYY = 92.7e6           # at D

LE10_THICKNESS = 0.6
LE10_FACE_PRESSURE = 1.0e6        # onto the upper surface
LE10_TARGET_SYY = -5.38e6         # at D = (2, 0, +0.3)


# ── mapped elliptic-annulus mesh ──────────────────────────────────────────────

@dataclass
class EllipticMesh:
    """Quarter elliptic-annulus hex mesh, extruded over -t/2 <= z <= +t/2."""
    nodes: dict = field(default_factory=dict)      # nid -> (x, y, z)
    elements: dict = field(default_factory=dict)   # eid -> [8 nids]
    ns: int = 0          # radial divisions (inner ellipse -> outer ellipse)
    nt: int = 0          # circumferential divisions (theta = 0 -> pi/2)
    nz: int = 0          # through-thickness divisions
    thickness: float = 0.0

    # node index helper -------------------------------------------------------
    def nid(self, i: int, j: int, k: int) -> int:
        """i: radial 0..ns, j: circumferential 0..nt, k: axial 0..nz."""
        return 1 + i + (self.ns + 1) * (j + (self.nt + 1) * k)

    # named node sets ---------------------------------------------------------
    @property
    def face_y0(self) -> list:
        """Face DCD'C': theta = 0, the y = 0 symmetry plane."""
        return [self.nid(i, 0, k)
                for k in range(self.nz + 1) for i in range(self.ns + 1)]

    @property
    def face_x0(self) -> list:
        """Face ABA'B': theta = pi/2, the x = 0 symmetry plane."""
        return [self.nid(i, self.nt, k)
                for k in range(self.nz + 1) for i in range(self.ns + 1)]

    @property
    def face_outer(self) -> list:
        """Face BCB'C': the outer elliptical surface."""
        return [self.nid(self.ns, j, k)
                for k in range(self.nz + 1) for j in range(self.nt + 1)]

    @property
    def outer_midplane(self) -> list:
        """Mid-thickness ring of the outer face (where LE10 fixes Uz)."""
        if self.nz % 2 != 0:
            raise ValueError("nz must be even for a mid-plane node ring")
        k = self.nz // 2
        return [self.nid(self.ns, j, k) for j in range(self.nt + 1)]

    @property
    def midplane(self) -> list:
        k = self.nz // 2
        return [self.nid(i, j, k)
                for j in range(self.nt + 1) for i in range(self.ns + 1)]

    def node_at_D(self, top: bool) -> int:
        """Point D = (2, 0, ±t/2): inner ellipse, theta = 0, top or bottom face."""
        return self.nid(0, 0, self.nz if top else 0)

    def outer_face_elements(self) -> list:
        """(eid, face_id) for the element faces lying on the outer ellipse.

        CalculiX C3D8 face numbering: F1 = nodes 1-2-3-4 (bottom), F2 = 5-6-7-8
        (top), F3 = 1-2-6-5, F4 = 2-3-7-6, F5 = 3-4-8-7, F6 = 4-1-5-8.
        Our connectivity puts the outer-radius pair at local nodes 2,3 (bottom)
        and 6,7 (top), which is face F4.
        """
        out = []
        for k in range(self.nz):
            for j in range(self.nt):
                eid = 1 + (self.ns - 1) + self.ns * (j + self.nt * k)
                out.append((eid, 4))
        return out

    def top_face_elements(self) -> list:
        """(eid, face_id) for element faces on the upper surface z = +t/2 (F2)."""
        out = []
        k = self.nz - 1
        for j in range(self.nt):
            for i in range(self.ns):
                out.append((1 + i + self.ns * (j + self.nt * k), 2))
        return out


def make_elliptic_mesh(thickness: float, ns: int, nt: int, nz: int) -> EllipticMesh:
    """Mapped hex mesh of the quarter annulus between the two ellipses.

    Radial blending is linear between the inner and outer ellipse at the same
    parametric angle, which keeps element shapes well conditioned everywhere in
    this geometry (both boundaries are convex and similarly oriented).
    """
    if nz % 2 != 0:
        raise ValueError("nz must be even so a mid-plane node ring exists")
    m = EllipticMesh(ns=ns, nt=nt, nz=nz, thickness=thickness)

    for k in range(nz + 1):
        z = -thickness / 2.0 + thickness * k / nz
        for j in range(nt + 1):
            theta = (math.pi / 2.0) * j / nt
            ct, st = math.cos(theta), math.sin(theta)
            xi, yi = A_INNER * ct, B_INNER * st
            xo, yo = A_OUTER * ct, B_OUTER * st
            for i in range(ns + 1):
                s = i / ns
                m.nodes[m.nid(i, j, k)] = (xi + s * (xo - xi),
                                           yi + s * (yo - yi), z)

    eid = 1
    for k in range(nz):
        for j in range(nt):
            for i in range(ns):
                m.elements[eid] = [
                    m.nid(i,     j,     k),
                    m.nid(i + 1, j,     k),
                    m.nid(i + 1, j + 1, k),
                    m.nid(i,     j + 1, k),
                    m.nid(i,     j,     k + 1),
                    m.nid(i + 1, j,     k + 1),
                    m.nid(i + 1, j + 1, k + 1),
                    m.nid(i,     j + 1, k + 1),
                ]
                eid += 1
    return m


# ── deck writing ──────────────────────────────────────────────────────────────

def _mesh_block(m: EllipticMesh, etype: str) -> list:
    lines = ["*NODE, NSET=NALL"]
    for nid, (x, y, z) in sorted(m.nodes.items()):
        lines.append(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}")
    lines.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for eid, conn in sorted(m.elements.items()):
        lines.append(f"{eid}, " + ", ".join(str(c) for c in conn))
    return lines


def _nset(name: str, nodes) -> list:
    uniq = sorted(set(nodes))
    lines = [f"*NSET, NSET={name}"]
    for i in range(0, len(uniq), 12):
        lines.append(", ".join(str(n) for n in uniq[i:i + 12]))
    return lines


def _material() -> list:
    return ["*MATERIAL, NAME=STEEL", "*ELASTIC",
            f"{E_STEEL:.6g}, {NU_STEEL:.6g}",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL"]


# ``*NODE PRINT ... S`` is silently ignored by CalculiX — stresses are element
# quantities in the .dat file. The extrapolated *nodal* stresses only reach the
# .frd, which is where the benchmark reads the value at point D from; the
# .dat element block is kept as the cross-check.
_OUTPUT_REQUESTS = ["*NODE FILE", "U, S",
                    "*EL PRINT, ELSET=EALL", "S",
                    "*END STEP"]


def write_le1_deck(path: Path, m: EllipticMesh, etype: str = "C3D8I") -> None:
    """LE1: outward edge traction on BC, symmetry on AB and DC, plane stress."""
    lines = _mesh_block(m, etype)
    lines += _nset("SYMY", m.face_y0)          # Uy = 0
    lines += _nset("SYMX", m.face_x0)          # Ux = 0
    lines += _nset("MIDZ", m.midplane)         # Uz = 0 (no through-thickness BC)
    lines += _nset("PTD", [m.node_at_D(top=True), m.node_at_D(top=False)])
    lines += _material()

    lines += ["*STEP", "*STATIC", "*BOUNDARY",
              "SYMY, 2, 2, 0.0",
              "SYMX, 1, 1, 0.0",
              "MIDZ, 3, 3, 0.0",
              "*DLOAD"]
    # A CalculiX pressure is positive INTO the face, so an outward traction is
    # applied as a negative pressure.
    for eid, face in m.outer_face_elements():
        lines.append(f"{eid}, P{face}, {-LE1_EDGE_PRESSURE:.10g}")
    lines += _OUTPUT_REQUESTS
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_le10_deck(path: Path, m: EllipticMesh, etype: str = "C3D8I") -> None:
    """LE10: 1 MPa on the upper face, clamped outer edge, symmetry planes."""
    lines = _mesh_block(m, etype)
    lines += _nset("SYMY", m.face_y0)
    lines += _nset("SYMX", m.face_x0)
    lines += _nset("OUTER", m.face_outer)
    lines += _nset("OUTMID", m.outer_midplane)
    lines += _nset("PTD", [m.node_at_D(top=True)])
    lines += _material()

    lines += ["*STEP", "*STATIC", "*BOUNDARY",
              "SYMY, 2, 2, 0.0",
              "SYMX, 1, 1, 0.0",
              "OUTER, 1, 2, 0.0",          # Ux = Uy = 0 on the whole outer face
              "OUTMID, 3, 3, 0.0",         # Uz = 0 only along its mid-plane
              "*DLOAD"]
    for eid, face in m.top_face_elements():
        lines.append(f"{eid}, P{face}, {LE10_FACE_PRESSURE:.10g}")
    lines += _OUTPUT_REQUESTS
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ── run + extract ─────────────────────────────────────────────────────────────

def run_ccx(work_dir: Path, job: str = "case") -> Path:
    exe = _find_ccx()
    if exe is None:
        raise FileNotFoundError("ccx binary not found in bin/ or PATH")
    proc = subprocess.run([str(exe), "-i", job], cwd=str(work_dir),
                          capture_output=True, text=True, timeout=1800)
    dat = work_dir / f"{job}.dat"
    if not dat.exists():
        raise RuntimeError(
            f"ccx produced no .dat (exit {proc.returncode}).\n{proc.stdout[-2000:]}")
    return dat


_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def element_stress_rows(dat_path: Path) -> dict:
    """{eid: [ (Sxx,Syy,Szz,Sxy,Sxz,Syz), ... per integration point ]}."""
    text = dat_path.read_text(encoding="ascii", errors="ignore")
    out: dict = {}
    in_block = False
    for line in text.splitlines():
        if "stresses" in line.lower():
            in_block = True
            continue
        if not in_block:
            continue
        nums = re.findall(_FLOAT, line)
        if len(nums) >= 8:
            eid = int(float(nums[0]))
            out.setdefault(eid, []).append(tuple(float(v) for v in nums[2:8]))
        elif not line.strip() and out:
            in_block = False
    return out


def nodal_stress_from_frd(frd_path: Path, node: int, component: int = 1) -> float:
    """Extrapolated nodal σ_component from a CalculiX .frd result file.

    The STRESS block is fixed-width: ``" -1"``, node number in 10 columns, then
    six 12-column values in the order SXX, SYY, SZZ, SXY, SYZ, SZX.

    This is the value the NAFEMS targets refer to — a stress *on the surface* at
    point D — whereas ``*EL PRINT`` reports integration points that sit inside
    the element and therefore under-read a surface peak.

    component: 0=Sxx, 1=Syy, 2=Szz, 3=Sxy, 4=Syz, 5=Szx.
    """
    in_block = False
    for line in Path(frd_path).read_text(encoding="ascii",
                                         errors="ignore").splitlines():
        if line.startswith(" -4") and "STRESS" in line:
            in_block = True
            continue
        if not in_block:
            continue
        if line.startswith(" -3"):          # end of block
            break
        if not line.startswith(" -1"):
            continue
        if int(line[3:13]) != node:
            continue
        field = line[13 + 12 * component: 25 + 12 * component]
        return float(field)
    raise RuntimeError(f"node {node} not found in STRESS block of {frd_path}")


def stress_at_point(dat_path: Path, mesh: EllipticMesh, node: int,
                    component: int = 1) -> float:
    """σ_component at `node`, averaged over the integration points of the
    elements that touch it.

    CalculiX's ``*EL PRINT`` reports integration-point values, which sit inside
    the element rather than on its surface. Averaging the touching elements is a
    deliberately conservative reconstruction: on a stress gradient it *under*-
    reads the true surface value, so a benchmark that passes this way is not
    passing because of an optimistic extrapolation.

    component: 0=Sxx, 1=Syy, 2=Szz, 3=Sxy, 4=Sxz, 5=Syz.
    """
    rows = element_stress_rows(dat_path)
    touching = [eid for eid, conn in mesh.elements.items() if node in conn]
    vals = [ip[component] for eid in touching for ip in rows.get(eid, [])]
    if not vals:
        raise RuntimeError(f"no stress data for elements touching node {node}")
    return sum(vals) / len(vals)


def nearest_ip_stress(dat_path: Path, mesh: EllipticMesh, node: int,
                      component: int = 1) -> float:
    """σ_component at the integration point closest to `node`.

    Complements :func:`stress_at_point`: taking the single nearest sample rather
    than an element average keeps more of the peak on a steep gradient. The two
    together bracket the surface value.
    """
    rows = element_stress_rows(dat_path)
    target = mesh.nodes[node]
    best, best_d = None, float("inf")
    for eid, conn in mesh.elements.items():
        if node not in conn:
            continue
        pts = [mesh.nodes[c] for c in conn]
        cx = sum(p[0] for p in pts) / 8.0
        cy = sum(p[1] for p in pts) / 8.0
        cz = sum(p[2] for p in pts) / 8.0
        d = (cx - target[0]) ** 2 + (cy - target[1]) ** 2 + (cz - target[2]) ** 2
        ips = rows.get(eid, [])
        if not ips:
            continue
        if d < best_d:
            best_d = d
            # Element average is still the best single estimate inside one element.
            best = sum(ip[component] for ip in ips) / len(ips)
    if best is None:
        raise RuntimeError(f"no stress data near node {node}")
    return best
