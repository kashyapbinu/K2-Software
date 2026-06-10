"""
Direct CalculiX verification cases (textbook-exact).
====================================================

Drives the *bundled* ``bin/ccx.exe`` with hand-generated ``.inp`` decks for
canonical solid-mechanics problems whose closed-form answer is exact. This is
the structural analogue of the Taylor–Maccoll cone check on the CFD side: it
proves the solver binary, the material card, the units, the boundary conditions
and the load application are all correct — independent of K2's rocket-airframe
meshing pipeline.

Cases:
    * uniaxial bar in tension  → σ=F/A, δ=FL/EA  (constant-strain, FE-exact)
    * cantilever bending        → δ_tip=FL³/3EI  (C3D8I incompatible modes vs
                                  Euler–Bernoulli; a few % FE error expected)

A small structured hexahedral mesh is generated in Python (no external mesher),
so the cases are self-contained and reproducible.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from structures.solvers.ccx_solver import _find_ccx

E_AL = 68.9e9
NU_AL = 0.33


# ── mesh generation (structured hex grid of C3D8 / C3D8I) ─────────────────────

@dataclass
class HexMesh:
    nodes: dict          # nid -> (x, y, z)
    elements: dict       # eid -> [8 node ids]
    nset_x0: list        # nodes on x=0 face
    nset_xL: list        # nodes on x=L face


def make_beam_mesh(L: float, b: float, h: float,
                   nx: int, ny: int, nz: int) -> HexMesh:
    """Structured C3D8 grid: length L along x, width b along y, height h along z."""
    dx, dy, dz = L / nx, b / ny, h / nz

    def nid(ix, iy, iz):
        return 1 + ix + (nx + 1) * (iy + (ny + 1) * iz)

    nodes = {}
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nodes[nid(ix, iy, iz)] = (ix * dx, iy * dy, iz * dz)

    elements = {}
    eid = 1
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # C3D8 connectivity: bottom face CCW then top face CCW.
                n = [
                    nid(ix,     iy,     iz),
                    nid(ix + 1, iy,     iz),
                    nid(ix + 1, iy + 1, iz),
                    nid(ix,     iy + 1, iz),
                    nid(ix,     iy,     iz + 1),
                    nid(ix + 1, iy,     iz + 1),
                    nid(ix + 1, iy + 1, iz + 1),
                    nid(ix,     iy + 1, iz + 1),
                ]
                elements[eid] = n
                eid += 1

    tol = 1e-9
    nset_x0 = [i for i, (x, _, _) in nodes.items() if abs(x) < tol]
    nset_xL = [i for i, (x, _, _) in nodes.items() if abs(x - L) < tol]
    return HexMesh(nodes, elements, nset_x0, nset_xL)


# ── deck writing ──────────────────────────────────────────────────────────────

def _write_deck(path: Path, mesh: HexMesh, etype: str,
                load_nodes: list, load_dof: int, load_per_node: float,
                E: float = E_AL, nu: float = NU_AL) -> None:
    lines = ["*NODE, NSET=NALL"]
    for nid, (x, y, z) in sorted(mesh.nodes.items()):
        lines.append(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}")

    lines.append(f"*ELEMENT, TYPE={etype}, ELSET=EALL")
    for eid, conn in sorted(mesh.elements.items()):
        lines.append(f"{eid}, " + ", ".join(str(c) for c in conn))

    lines.append("*NSET, NSET=FIXED")
    lines.append(", ".join(str(n) for n in mesh.nset_x0))

    lines.append("*MATERIAL, NAME=AL")
    lines.append("*ELASTIC")
    lines.append(f"{E:.6g}, {nu:.6g}")
    lines.append("*SOLID SECTION, ELSET=EALL, MATERIAL=AL")

    lines.append("*STEP")
    lines.append("*STATIC")
    lines.append("*BOUNDARY")
    lines.append("FIXED, 1, 3, 0.0")              # clamp all 3 DOF on x=0 face
    lines.append("*CLOAD")
    for n in load_nodes:
        lines.append(f"{n}, {load_dof}, {load_per_node:.10g}")
    # Print nodal displacement + elemental stress to the .dat file.
    lines.append("*NODE PRINT, NSET=NALL")
    lines.append("U")
    lines.append("*EL PRINT, ELSET=EALL")
    lines.append("S")
    lines.append("*END STEP")

    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ── run + parse ───────────────────────────────────────────────────────────────

def run_ccx(work_dir: Path, job: str = "case") -> Path:
    """Run the bundled ccx on <work_dir>/<job>.inp; return the .dat path."""
    exe = _find_ccx()
    if exe is None:
        raise FileNotFoundError("ccx binary not found in bin/ or PATH")
    proc = subprocess.run([str(exe), "-i", job], cwd=str(work_dir),
                          capture_output=True, text=True, timeout=600)
    dat = work_dir / f"{job}.dat"
    if not dat.exists():
        raise RuntimeError(
            f"ccx produced no .dat (exit {proc.returncode}).\n{proc.stdout[-2000:]}")
    return dat


_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_displacements(dat_path: Path) -> dict:
    """Parse the 'displacements' block of a .dat file → {nid: (ux,uy,uz)}."""
    text = dat_path.read_text(encoding="ascii", errors="ignore")
    out = {}
    in_block = False
    for line in text.splitlines():
        low = line.lower()
        if "displacements" in low:
            in_block = True
            continue
        if in_block:
            nums = re.findall(_FLOAT, line)
            if len(nums) >= 4:
                out[int(float(nums[0]))] = (float(nums[1]),
                                            float(nums[2]), float(nums[3]))
            elif line.strip() == "" and out:
                in_block = False
    return out


def parse_mean_stress(dat_path: Path, component: int = 1) -> float:
    """Volume-average S_component over all integration points.

    For a uniform bar under an end load, equilibrium forces ∫σ_xx dA = F at every
    section, so the average axial stress equals F/A exactly — unlike the *max*,
    which is inflated by the 3-D Poisson constraint at the clamped face.
    """
    text = dat_path.read_text(encoding="ascii", errors="ignore")
    in_block = False
    vals = []
    for line in text.splitlines():
        if "stresses" in line.lower():
            in_block = True
            continue
        if in_block:
            nums = re.findall(_FLOAT, line)
            if len(nums) >= 8:
                vals.append(float(nums[1 + component]))
            elif line.strip() == "" and vals:
                in_block = False
    return sum(vals) / len(vals) if vals else 0.0


def parse_max_abs_stress(dat_path: Path, component: int = 1) -> float:
    """Max |S_component| from the 'stresses' block. component 1 = Sxx (1-indexed)."""
    text = dat_path.read_text(encoding="ascii", errors="ignore")
    in_block = False
    best = 0.0
    for line in text.splitlines():
        low = line.lower()
        if "stresses" in low:
            in_block = True
            continue
        if in_block:
            nums = re.findall(_FLOAT, line)
            # rows: elem, intpt, Sxx,Syy,Szz,Sxy,Sxz,Syz  → >=8 numbers
            if len(nums) >= 8:
                val = abs(float(nums[1 + component]))
                best = max(best, val)
            elif line.strip() == "" and in_block and best > 0:
                in_block = False
    return best
