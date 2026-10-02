"""
The CalculiX static model must carry the real loads, and everything the
Structures workspace shows must be read off that one solution.

Before (measured on the audit's rocket, 2026-09-23):
* the deck loaded only the shell's own weight (GRAV, pointing at the nose —
  tension where thrust compresses); every flight load was a hand formula
  added afterwards: raw CalculiX 0.2 MPa was displayed as 21.6 MPa;
* the tail was clamped AND the nose tip pinned, so the raw peak sat at the
  nose tip;
* one *SHELL SECTION gave every element the nose-cone wall (4 mm fins
  analysed as 2.5 mm), S4R elements (one integration point through the
  wall) could not bend, and each fin shared ONE node with the tube;
* the 3D contour never read the .frd at all.
"""
import math
from pathlib import Path

import numpy as np
import pytest

from structures.fe_loads import FEMesh, build_static_loads
from structures.frd import read_frd
from structures.meshing import build_structural_mesh_info
from structures.solvers.base import LoadCase, get_structural_material
from validation.cases.rocket_canonical import canonical_assembly

_ALU = get_structural_material("Aluminum 6061-T6")

# Vehicle = airframe + motor. The total was a literal 2.63 kg, sized around an
# airframe that weighed 1.13 kg only because its aluminium was being weighed
# as cardboard. The real airframe is 3.98 kg, more than that total, which left
# no mass for the motor at all.
_MOTOR_KG = 1.5
_VEHICLE_KG = canonical_assembly().total_mass() + _MOTOR_KG


def _mesh(tmp_path, asm=None, refinement="medium"):
    asm = asm or canonical_assembly()
    path, info = build_structural_mesh_info(asm, tmp_path / "structure_mesh.inp", refinement)
    return asm, FEMesh.read(path), info, path


def _flight_case(name="Max-Q"):
    if name == "Max-Q":
        lc = LoadCase.max_q(thrust=500.0, q_dyn=31.4e3, mach=0.8, alt=3000.0, aoa=3.0)
    else:
        lc = LoadCase.max_thrust(thrust=1500.0, angle_of_attack_deg=2.0, mach=0.3,
                                 altitude_m=500.0)
    lc.vehicle_mass_kg, lc.motor_aft_m, lc.motor_length_m = _VEHICLE_KG, 2.0, 0.30
    return lc


# ── Mesh ──────────────────────────────────────────────────────────────────────

def test_every_fin_root_node_is_a_tube_node(tmp_path):
    _, mesh, info, _ = _mesh(tmp_path)
    assert info.unattached_fin_roots == 0
    assert len(info.pieces) == 1
    fins = set(mesh.nsets["NFINS"])
    tube = set(mesh.nsets["BODY_TUBE"])
    # 4 fins × (8 + 1) root chord points — was 4 (one aft corner per fin)
    assert len(fins & tube) == 4 * 9


def test_ring_count_is_a_multiple_of_the_fin_count(tmp_path):
    asm = canonical_assembly()
    fins = next(c for c in asm.all_components() if type(c).__name__ == "TrapezoidalFinSet")
    fins.fin_count = 3
    _, _, info, _ = _mesh(tmp_path, asm, "coarse")        # coarse = 16 around
    assert info.n_circ % 3 == 0 and info.n_circ >= 16
    assert info.unattached_fin_roots == 0


def test_each_component_keeps_its_own_wall(tmp_path):
    _, _, info, path = _mesh(tmp_path)
    t = {s.kind: s.thickness for s in info.sections}
    assert t["fin"] == pytest.approx(0.004)             # was the 2.5 mm nose wall
    assert t["tube"] == pytest.approx(0.0025) and t["nose"] == pytest.approx(0.0025)
    text = path.read_text()
    assert "TYPE=S4," in text and "S4R" not in text     # S4R cannot bend through the wall


def test_diameter_steps_and_small_gaps_stay_connected(tmp_path):
    from core.components import BodyTube, RocketAssembly, NoseCone
    asm = RocketAssembly()
    st = asm.stages[0]
    nose = NoseCone(); nose.length, nose.diameter = 0.3, 0.1000
    t1 = BodyTube(); t1.length, t1.outer_diameter_val, t1.inner_diameter = 0.6, 0.1004, 0.0964
    t2 = BodyTube(); t2.length, t2.outer_diameter_val, t2.inner_diameter = 0.5, 0.0760, 0.0720
    for c in (nose, t1, t2):
        asm.add_component(st, c)
    t2._ork_pos, t2._ork_rel = 0.902, "top"   # starts 2 mm after t1 ends (0.9 m)
    asm._recompute_positions()
    _, _, info, _ = _mesh(tmp_path, asm)
    assert len(info.pieces) == 1, info.pieces
    assert any(s.kind == "step" for s in info.sections)    # 0.1004 → 0.076 annulus


# ── Load set ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", ["Max-Q", "Max Thrust", "Recovery Shock", "Pressure"])
def test_load_set_is_a_balanced_free_body(tmp_path, case):
    asm, mesh, info, _ = _mesh(tmp_path)
    if case == "Recovery Shock":
        lc = LoadCase.recovery(vehicle_mass_kg=_VEHICLE_KG)
    elif case == "Pressure":
        lc = LoadCase(name="Custom", axial_force=300.0, internal_pressure=2e5)
    else:
        lc = _flight_case(case)
    lm = build_static_loads(mesh, info, asm, lc, lambda s: _ALU.density)
    scale = max(np.abs(lm.forces).sum(), 1.0)
    d = mesh.xyz - mesh.xyz.mean(axis=0)
    assert np.linalg.norm(lm.forces.sum(axis=0)) < 1e-9 * scale
    assert np.linalg.norm(np.cross(d, lm.forces).sum(axis=0)) < 1e-9 * scale
    A, B, C = lm.support
    assert len({A, B, C}) == 3 and all(n in info.aft_ring for n in (A, B, C))


def test_load_set_carries_the_design_loads(tmp_path):
    asm, mesh, info, _ = _mesh(tmp_path)
    lc = _flight_case("Max-Q")
    lm = build_static_loads(mesh, info, asm, lc, lambda s: _ALU.density)
    s = lm.summary
    assert s["thrust_N"] == 500.0
    assert s["vehicle_mass_kg"] == pytest.approx(_VEHICLE_KG, rel=1e-6)
    # The motor is what the airframe does not account for. It is clamped at
    # zero, so a total below the airframe mass would pass silently otherwise.
    assert s["motor_mass_kg"] == pytest.approx(_MOTOR_KG, rel=1e-6)
    assert s["normal_force_N"] > 0 and s["daf"] == 1.3
    assert s["accel_axial_g"] == pytest.approx(
        (500.0 - s["drag_N"]) / (_VEHICLE_KG * 9.80665), rel=1e-6)
    assert s["fin_normal_N"] > 0 and s["drag_N"] > 0


# ── .frd reader ───────────────────────────────────────────────────────────────

def test_frd_fields_are_read_by_column(tmp_path):
    frd = tmp_path / "a.frd"
    frd.write_text("\n".join([
        "    1C", "    2C                             2                                     1",
        " -1         7-1.16145E-03 0.00000E+00 3.04093E-04",
        " -1         9 1.26345E-03-2.50000E-01-3.04093E-04",
        " -3",
        "    3C                             1                                     1",
        " -1         1    1    0    1",
        " -2         7         9         7         9         7         9         7         9",
        " -3",
        "  100CL  101 1.000000000           2                     0    1           1",
        " -4  STRESS      6    1",
        " -5  SXX         1    4    1    1",
        " -1         7 7.03620E+03-8.22566E+03 1.75449E+04-1.14147E-09-3.08546E-10-7.50535E+03",
        " -1         9" + "".join(f"{v:12.5E}" for v in (-1, 2, 3, 4, 5, 6)),
        " -3", " 9999", ""]))
    r = read_frd(frd)
    assert r.nodes[9] == pytest.approx((1.26345e-3, -0.25, -3.04093e-4))
    assert r.elements[1] == (1, [7, 9, 7, 9, 7, 9, 7, 9])
    assert r.blocks["STRESS"][7] == pytest.approx(
        [7036.2, -8225.66, 17544.9, -1.14147e-9, -3.08546e-10, -7505.35])
    assert r.blocks["STRESS"][9] == pytest.approx([-1, 2, 3, 4, 5, 6])


# ── Real CalculiX solve (slow) ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def maxq_result(tmp_path_factory):
    from structures.solvers.ccx_solver import _find_ccx
    if _find_ccx() is None:
        pytest.skip("CalculiX not available")
    from structures.fem_interface import FEMInterface
    work = tmp_path_factory.mktemp("fe_static")
    lc = _flight_case("Max-Q")
    return FEMInterface(work_dir=work).analyze(canonical_assembly(), lc, "Aluminum 6061-T6",
                                               "medium", "static")


@pytest.mark.slow
@pytest.mark.structures
def test_panel_profile_and_contour_are_one_field(maxq_result):
    r, f = maxq_result, maxq_result.fe_field
    assert f is not None
    assert r.max_von_mises == pytest.approx(float(f.von_mises.max()))
    assert max(v for _, v in r.element_stresses) == pytest.approx(r.max_von_mises)
    assert r.safety_factor == pytest.approx(
        float((f.yield_pa / (f.kt * f.von_mises)).min()), rel=1e-9)
    assert r.max_axial_stress == pytest.approx(float(np.abs(f.axial).max()))
    assert r.max_displacement_mm == pytest.approx(
        float(np.linalg.norm(f.displacement, axis=1).max() * 1000))
    # the 3-2-1 support carries nothing: the load set balances
    assert r.applied_loads["support_reaction_N"] < 1e-3


@pytest.mark.slow
@pytest.mark.structures
def test_fe_bending_matches_beam_statics_of_its_own_loads(maxq_result):
    """Ring bending stress vs M·r/I and membrane stress vs N/A of the same
    nodal load set, at mid-body stations away from load introduction."""
    r = maxq_result
    f = r.fe_field
    asm = canonical_assembly()
    work = Path(r.result_vtk).parent
    mesh = FEMesh.read(work / "structure_mesh.inp")
    _, info = build_structural_mesh_info(asm, work / "check.inp", "medium")
    lm = build_static_loads(mesh, info, asm, _flight_case("Max-Q"), lambda s: _ALU.density)
    rm, t = 0.051, 0.0025
    I, A = math.pi * rm ** 3 * t, 2 * math.pi * rm * t
    for zc in (0.6, 1.0, 1.4):
        fwd = mesh.xyz[:, 2] < zc
        M = float(np.sum(lm.forces[fwd, 1] * (zc - mesh.xyz[fwd, 2])))
        N = float(np.sum(lm.forces[fwd, 2]))
        sel = (~f.is_fin) & (np.abs(f.points[:, 2] - zc) < 0.007)
        th = np.arctan2(f.points[sel, 1], f.points[sel, 0])
        ring = f.axial[sel]
        amp = math.hypot(2 * np.mean(ring * np.cos(th)), 2 * np.mean(ring * np.sin(th)))
        assert amp == pytest.approx(abs(M) * rm / I, rel=0.10), zc
        assert -float(np.mean(ring)) == pytest.approx(N / A, rel=0.05), zc


@pytest.mark.slow
@pytest.mark.structures
def test_the_old_artefacts_are_gone(maxq_result):
    f = maxq_result.fe_field
    k = int(np.argmax(f.von_mises))
    assert f.points[k, 2] > 0.1 * f.length          # not the pinned nose tip any more
    # S4 bends through the wall: inner and outer surface differ on the fins
    fin_vm = np.sort(f.von_mises[f.is_fin])
    assert fin_vm[-1] > 0 and len(np.unique(np.round(fin_vm, 3))) > len(fin_vm) // 3


# ── The 3D views draw the field itself ────────────────────────────────────────

class _Plotter:
    """Records what a view hands to pyvista (no Qt / GL needed)."""

    def __init__(self):
        self.meshes, self.labels = [], []

    def add_mesh(self, mesh, **kw):
        self.meshes.append((mesh, kw))

    def add_point_labels(self, pts, labels, **kw):
        self.labels.extend(labels)

    def __getattr__(self, name):
        return lambda *a, **k: None


def _one_hex_field():
    from structures.solvers.base import FEField
    pts = np.array([[0.05, 0, 0.1], [0.05, 0, 0.2], [0.051, 0.01, 0.2], [0.051, 0.01, 0.1],
                    [0.052, 0, 0.1], [0.052, 0, 0.2], [0.053, 0.01, 0.2], [0.053, 0.01, 0.1]])
    vm = np.arange(1.0, 9.0) * 1e6
    return FEField(points=pts, cells=np.arange(8)[None, :], von_mises=vm,
                   axial=-vm, hoop=vm / 2, shear=vm / 4,
                   displacement=np.tile([0.0, 0.0, 1e-4], (8, 1)) * np.arange(1, 9)[:, None],
                   yield_pa=np.full(8, 276e6), is_fin=np.zeros(8, bool),
                   cell_is_fin=np.zeros(1, bool), kt=1.8, length=0.3)


def test_stress_viewer_draws_the_calculix_values():
    pytest.importorskip("pyvista")
    from ui.widgets.stress_viewer import StressViewer, fe_grid
    f = _one_hex_field()
    sv = StressViewer.__new__(StressViewer)
    sv._fe, sv._fe_grid, sv._component, sv._total_len = f, fe_grid(f), "Entire Vehicle", 0.3
    for mode, peak, label in (("Von Mises Stress", 8.0, "Peak: 8.00 MPa"),
                              ("Axial Stress", -8.0, "Peak: -8.00 MPa"),
                              ("Safety Factor", 276 / (1.8 * 8), "Min SF: 19.17")):
        sv.plotter, sv._mode = _Plotter(), mode
        sv._render()
        mesh, kw = sv.plotter.meshes[0]
        v = np.asarray(mesh.point_data["value"])
        assert (v.min() if peak < 0 or mode == "Safety Factor" else v.max()) == pytest.approx(peak)
        assert sv.plotter.labels[0].startswith(label)


def test_deformation_view_draws_the_calculix_displacement():
    pytest.importorskip("pyvista")
    from ui.widgets.deformation_viewer import DeformationViewer
    dv = DeformationViewer.__new__(DeformationViewer)
    dv.plotter, dv._empty = _Plotter(), _Plotter()      # _empty: a stand-in label
    peak = dv.set_fe_deflection(_one_hex_field(), exaggeration=10)
    assert peak == pytest.approx(0.8)                    # 8 × 1e-4 m in mm
    deformed = [m for m, kw in dv.plotter.meshes if kw.get("scalars") == "Displacement (mm)"][0]
    assert float(np.max(deformed["Displacement (mm)"])) == pytest.approx(0.8)
