"""
What the Structures / Dynamics views draw must be the CalculiX result they
are labelled with.

Audit of 2026-09-23, every item measured on real ccx output:

* The modal parser never recognised CalculiX's letter-spaced table titles
  ("P A R T I C I P A T I O N  F A C T O R S"). Participation-factor rows were
  read as extra frequencies, their tiny positive Z values inflated the
  rigid-mode offset, and the shape list was sliced from the wrong mode: on a
  real rocket "Mode 1, 62 Hz" animated mode 7 (469 Hz) and modes 7–10 had no
  shape. The installed app logged "10 modes parsed, 5–7 shapes" on every run.
* Both workspaces loaded the mode mesh from a relative "fem_run/..." path that
  does not exist in installed builds, so the Structures viewer stayed blank
  and Dynamics always showed a synthetic cylinder.
* Dynamics vibration gave cantilever Γ by list position to FE modes: shell
  modes with 0 % effective mass got Γ 0.25 / 0.18 (phantom resonance peaks).
* The 3D stress contour was pinned per region (every region peaked at the
  panel σ_vm) on a body tube with two axial stations, so the peak sat at the
  nose/body joint instead of mid-body; the 2D stress plot peaked at the tail.
"""
import math
from pathlib import Path

import pytest

from structures.solvers.base import FEMConfig, ModalResult
from structures.solvers.ccx_solver import CalculiXSolver

_ROOT = Path(__file__).resolve().parents[2]

# ── A small CalculiX-format modal result ──────────────────────────────────────
# Tube: 4 rings (z = 0 … 0.3 m) × 8 nodes at r = 0.05 m → nodes 1–32.
# One fin in the θ = 0 plane, 3 × 3 nodes (33–41), listed in NFINS.
_R, _NC, _ZS = 0.05, 8, (0.0, 0.1, 0.2, 0.3)
_FREQS = (62.1, 62.1, 321.2, 350.1, 400.0, 500.0)
_KINDS = ("Bending-Y", "Bending-X", "Shell", "Torsional", "Fin", "Axial")
_DESCS = ("1st Lateral Bending (Y)", "1st Lateral Bending (X)", "1st Shell (Ovalling)",
          "1st Torsional", "1st Fin Mode", "1st Axial")


def _nodes():
    nodes = {}
    for i, z in enumerate(_ZS):
        for j in range(_NC):
            t = 2 * math.pi * j / _NC
            nodes[i * _NC + j + 1] = (_R * math.cos(t), _R * math.sin(t), z)
    k = len(nodes) + 1
    for s in range(3):                       # span
        for c in range(3):                   # chord
            nodes[k] = (0.06 + 0.045 * s, 0.0, 0.2 + 0.05 * c)
            k += 1
    return nodes


def _shape(mode, nid, x, y, z):
    fin = nid > len(_ZS) * _NC
    if mode == 1:
        return (0.0, (z + 0.1) ** 2, 0.0)            # bending, Y
    if mode == 2:
        return ((z + 0.1) ** 2, 0.0, 0.0)            # bending, X
    if mode == 5:                                    # fin flap, body still
        return (0.0, 10.0 * (x - 0.05), 0.0) if fin else (0.0, 0.0, 0.0)
    if fin:
        return (0.0, 0.0, 0.0)
    t = math.atan2(y, x)
    if mode == 3:                                    # ovalling: u_r ∝ cos 2θ
        a = math.cos(2 * t) * (z + 0.1)
        return (a * math.cos(t), a * math.sin(t), 0.0)
    if mode == 4:                                    # roll: u = φ(z) (−y, x, 0)
        phi = 10.0 * (z + 0.1)
        return (-y * phi, x * phi, 0.0)
    return (0.0, 0.0, z + 0.1)                       # axial


def _write_case(work: Path):
    nodes = _nodes()
    lines = ["** test mesh", "*NODE, NSET=NALL"]
    lines += [f"{n}, {x:.8e}, {y:.8e}, {z:.8e}" for n, (x, y, z) in nodes.items()]
    lines.append("*ELEMENT, TYPE=S4R, ELSET=EALL")
    eid = 1
    for i in range(len(_ZS) - 1):
        for j in range(_NC):
            a, b = i * _NC + j + 1, i * _NC + (j + 1) % _NC + 1
            lines.append(f"{eid}, {a}, {b}, {b + _NC}, {a + _NC}")
            eid += 1
    for s in range(2):
        for c in range(2):
            a = 33 + 3 * s + c
            lines.append(f"{eid}, {a}, {a + 1}, {a + 4}, {a + 3}")
            eid += 1
    lines += ["*NSET, NSET=BODY_TUBE", ", ".join(str(n) for n in range(1, 33)),
              "*NSET, NSET=MY_FINS", ", ".join(str(n) for n in range(33, 42)),
              "*NSET, NSET=NAFT", ", ".join(str(n) for n in range(25, 33)),
              "*NSET, NSET=NFWD", ", ".join(str(n) for n in range(1, 9)),
              "*NSET, NSET=NFINS", ", ".join(str(n) for n in range(33, 42))]
    mesh = work / "structure_mesh.inp"
    mesh.write_text("\n".join(lines) + "\n", encoding="ascii")

    e = lambda v: f"{v:15.7E}"
    # Z participation of modes 1, 3, 5 is a tiny POSITIVE number, exactly
    # the values the old scanner mistook for frequencies.
    part = [(-0.13, 1.94, 1.7e-12), (1.94, 0.13, -8.6e-13), (1e-11, 1e-11, 2.4e-13),
            (1e-12, 1e-12, -3e-13), (1e-3, 0.32, 9.8e-13), (1e-12, 1e-12, 2.0)]
    meff = [(0.0161, 3.768, 0.0), (3.768, 0.0161, 0.0), (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0), (0.0, 0.1024, 0.0), (0.0, 0.0, 4.0)]
    rz = [0.0, 0.0, 0.0, 0.031, 0.0, 0.0]
    d = ["                        S T E P       1", "", "",
         "     E I G E N V A L U E   O U T P U T", "",
         " MODE NO    EIGENVALUE                       FREQUENCY   ",
         "                                     REAL PART            IMAGINARY PART",
         "                           (RAD/TIME)      (CYCLES/TIME     (RAD/TIME)", ""]
    for k, f in enumerate(_FREQS, 1):
        w = 2 * math.pi * f
        d.append(f"{k:7d}{e(w * w)}{e(w)}{e(f)}{e(0.0)}")
    d += ["", "     P A R T I C I P A T I O N   F A C T O R S", "",
          "MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      "
          "Y-ROTATION      Z-ROTATION", ""]
    for k, (gx, gy, gz) in enumerate(part, 1):
        d.append(f"{k:7d} {e(gx)} {e(gy)} {e(gz)} {e(0.1)} {e(0.1)} {e(rz[k - 1] ** 0.5)}")
    d += ["", "     E F F E C T I V E   M O D A L   M A S S", "",
          "MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      "
          "Y-ROTATION      Z-ROTATION", ""]
    for k, (mx, my, mz) in enumerate(meff, 1):
        d.append(f"{k:7d} {e(mx)} {e(my)} {e(mz)} {e(0.01)} {e(0.01)} {e(rz[k - 1])}")
    d += [f"TOTAL   {e(7.5)} {e(7.5)} {e(4.0)} {e(0.02)} {e(0.02)} {e(0.031)}", "",
          "     T O T A L   E F F E C T I V E   M A S S", "",
          "MODE NO.   X-COMPONENT     Y-COMPONENT     Z-COMPONENT     X-ROTATION      "
          "Y-ROTATION      Z-ROTATION", "",
          f"          {e(6.78)} {e(6.78)} {e(6.78)} {e(7.27)} {e(7.27)} {e(0.0484)}", ""]
    for k in range(1, len(_FREQS) + 1):
        d += ["", f"                    E I G E N V A L U E    N U M B E R {k:5d}", "", "",
              " displacements (vx,vy,vz) for set NALL and time  0.1000000E+01", ""]
        for n, (x, y, z) in nodes.items():
            ux, uy, uz = _shape(k, n, x, y, z)
            d.append(f"{n:10d} {ux:13.6E} {uy:13.6E} {uz:13.6E}")
    (work / "analysis.dat").write_text("\n".join(d) + "\n", encoding="utf-8")
    return mesh, nodes


def _parse(tmp_path):
    mesh, nodes = _write_case(tmp_path)
    solver = CalculiXSolver(FEMConfig(analysis_type="modal", num_modes=10,
                                      work_dir=tmp_path))
    solver._mesh_path = mesh
    return solver._parse_modal_results(), nodes


@pytest.mark.structures
def test_each_mode_shape_stays_with_its_own_frequency(tmp_path):
    r, nodes = _parse(tmp_path)
    assert r.frequencies_hz == pytest.approx(list(_FREQS), rel=1e-6)
    assert r.mode_numbers == list(range(1, len(_FREQS) + 1))
    assert len(r.mode_shapes) == len(_FREQS)
    for k, shape in enumerate(r.mode_shapes, 1):
        for n, (x, y, z) in nodes.items():
            assert shape[n] == pytest.approx(_shape(k, n, x, y, z), abs=1e-6), (k, n)


@pytest.mark.structures
def test_modes_are_named_from_their_shapes(tmp_path):
    r, _ = _parse(tmp_path)
    assert r.mode_classifications == list(_KINDS)
    # the two planes of one bending mode are ONE mode — both "1st"
    assert r.descriptions == list(_DESCS)


@pytest.mark.structures
def test_calculix_mass_tables_are_reported_per_mode(tmp_path):
    r, _ = _parse(tmp_path)
    assert r.total_mass_kg == pytest.approx(6.78)
    assert r.effective_modal_mass[0]["y"] == pytest.approx(100 * 3.768 / 6.78, abs=0.1)
    assert r.effective_modal_mass[3]["rz"] == pytest.approx(100 * 0.031 / 0.0484, abs=0.1)
    assert r.participation_factors[0]["y"] == pytest.approx(1.94)
    assert len(r.effective_modal_mass) == len(r.participation_factors) == len(_FREQS)


@pytest.mark.structures
def test_mode_viewer_gets_the_mesh_from_the_result(tmp_path):
    """The mesh travels with the result — no file path is read."""
    r, nodes = _parse(tmp_path)
    assert set(r.mesh_nodes) == set(nodes)
    assert len(r.mesh_elements) == 3 * _NC + 4
    (tmp_path / "structure_mesh.inp").unlink()      # a later run could replace it

    pytest.importorskip("pyvista")
    from ui.widgets import mode_shape_viewer as msv

    class _Plotter:                                  # no Qt / GL needed
        def __getattr__(self, name):
            return lambda *a, **k: None

    class _Label:
        text = ""

        def setText(self, t):
            self.text = t

    v = msv.ModeShapeViewer.__new__(msv.ModeShapeViewer)
    v.plotter, v.mode_shape, v.lbl_mode_info = _Plotter(), None, _Label()
    assert v.load_result_mesh(r)
    assert v.base_points.shape == (len(nodes), 3)
    v.set_mode_shape(r.mode_shapes[0], freq_hz=r.frequencies_hz[0],
                     description=r.descriptions[0], mode_index=1)
    assert "62.1 Hz" in v.lbl_mode_info.text
    # contour is |φ| relative to its peak — not the eigenvector ×1000 as "mm"
    amp = v._grid[msv._AMP]
    assert 0.0 < amp.max() <= 1.0


def test_workspaces_do_not_read_a_relative_fem_path():
    import ast
    for rel in ("ui/workspaces/structures_workspace.py", "ui/workspaces/dynamics_workspace.py"):
        tree = ast.parse((_ROOT / rel).read_text(encoding="utf-8"))
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        assert not [s for s in strings if "fem_run" in s], rel
        calls = {c.func.attr for c in ast.walk(tree)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)}
        names = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)}
        assert "load_result_mesh" in calls | names, rel


# ── Dynamics: vibration weighted by CalculiX's effective masses ───────────────

def _user_rocket_modal():
    """The mode list of the rocket in the audit (installed-app run)."""
    m = ModalResult()
    m.frequencies_hz = [62.1, 62.1, 321.2, 328.8, 350.1, 350.1, 469.3, 504.1, 517.2, 725.0]
    z = [0.0] * 6
    m.effective_mass_kg = [[0.190, 3.595, 0, 0.845, 0.045, 0], [3.595, 0.190, 0, 0.045, 0.845, 0],
                           z, z, [0.782, 0.487, 0, 0, 0, 0], [0.487, 0.782, 0, 0, 0, 0],
                           z, z, [0, 0, 0, 0, 0, 0.0308], z]
    m.total_effective_mass_kg = (6.784, 6.784, 6.784, 7.267, 7.267, 0.0484)
    m.descriptions = ["1st Lateral Bending (Y)", "1st Lateral Bending (X)", "1st Shell (Ovalling)",
                      "2nd Shell (Ovalling)", "2nd Lateral Bending (X)", "2nd Lateral Bending (Y)",
                      "3rd Shell (Ovalling)", "4th Shell (Ovalling)", "1st Torsional",
                      "5th Shell (Ovalling)"]
    return m


def test_vibration_inputs_come_from_calculix_effective_mass():
    from dynamics.vibration_analysis import fem_modal_inputs
    freqs, gammas, names = fem_modal_inputs(_user_rocket_modal())
    assert freqs == pytest.approx([62.1, 350.1])
    # √(effective-mass fraction) of each merged degenerate pair
    assert gammas[0] == pytest.approx(math.sqrt((0.190 + 3.595) / 6.784), rel=1e-6)
    assert gammas[1] == pytest.approx(math.sqrt((0.782 + 0.487) / 6.784), rel=1e-6)
    assert names == ["Mode 1+2 — 1st Lateral Bending", "Mode 5+6 — 2nd Lateral Bending"]


def test_zero_mass_modes_draw_no_resonance_peak():
    from dynamics.vibration_analysis import vibration_from_modal
    r = vibration_from_modal(_user_rocket_modal(), 0.02, 0.04)
    peaks = sorted(f for f, _, _ in r.modal_markers)
    # index Γ also drew peaks at 319, 467, 518 and 733 Hz — modes with no mass
    assert peaks == pytest.approx([62.1, 350.1], rel=0.02)
    assert [n for _, _, n in sorted(r.modal_markers)] == [
        "Mode 1+2 — 1st Lateral Bending", "Mode 5+6 — 2nd Lateral Bending"]


def test_result_without_mass_tables_keeps_the_default_gamma():
    from dynamics.vibration_analysis import fem_modal_inputs, vibration_from_modal
    m = ModalResult(frequencies_hz=[50.0, 140.0])
    assert fem_modal_inputs(m) is None
    assert vibration_from_modal(m).participation_factors_used == pytest.approx([0.783, 0.434])


# ── Structures: one stress field, peak where the bending peaks ────────────────

_BC = dict(axial=0.78e6, hoop=0.64e6, bending=50.84e6, shear=1.32e6, thermal=0.0,
           von_mises=96.4e6)


def _viewer():
    pytest.importorskip("pyvista")
    from ui.widgets.stress_viewer import StressViewer, build_rocket_regions
    from validation.cases.rocket_canonical import canonical_assembly, canonical_state
    sv = StressViewer.__new__(StressViewer)          # no Qt / GL needed
    sv._region_meshes, sv._total_len = build_rocket_regions(canonical_state(),
                                                            canonical_assembly())
    sv._fin_stress_pa, sv._yield_pa = 4.4e6, 276e6
    return sv


@pytest.mark.structures
def test_stress_contour_peaks_once_at_the_panel_value_mid_body():
    import numpy as np
    sv = _viewer()
    L = sv._total_len
    peaks = {}
    for region, mesh in sv._region_meshes.items():
        f = sv._field_for_mesh(mesh, region, "Von Mises Stress", _BC)
        j = int(np.argmax(f))
        peaks[region] = (float(f[j]), float((L - mesh.points[j, 2]) / L))
    top = max(peaks.values())
    assert top[0] == pytest.approx(96.4, rel=1e-3)          # = panel σ_vm (MPa)
    assert 0.4 < top[1] < 0.6                               # bending peak, mid-body
    # the nose cone is NOT pinned to the panel value on its own any more
    assert peaks["nose"][0] < 0.8 * top[0]


@pytest.mark.structures
def test_body_tube_has_axial_stations():
    import numpy as np
    sv = _viewer()
    z = sv._region_meshes["airframe"].points[:, 2]
    assert len(np.unique(np.round(z, 6))) > 50      # pv.Cylinder gave 2
