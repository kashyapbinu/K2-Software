"""
Regression gate: the fin cross-section a component declares must reach the mesh.

``cfd.meshing._add_fins`` used to extrude the planform into a slab whatever
``TrapezoidalFinSet.cross_section`` said, so every fin in every CFD run had a
square leading edge. On the AGARD-B validation wing that is a blunt face 4% of
chord deep across the whole span — the flow stagnates on it instead of forming
the leading-edge suction peak, and no amount of mesh refinement recovers the
lift that peak carries.

The tests measure the solid gmsh actually built: a shaped section removes
material relative to the slab, and the polygon it is built from must be the same
shape at root and tip (a mismatch is not a solid, and OCC only reports it much
later as ``BOPAlgo_AlertTooFewArguments`` from an unrelated boolean).
"""
import pytest

gmsh = pytest.importorskip("gmsh")

from cfd.meshing import _add_fins, _fin_section_2d   # noqa: E402

ROOT_CHORD, TIP_CHORD, HEIGHT, THICK = 0.20, 0.08, 0.15, 0.008
BODY_R, TOTAL_L = 0.05, 1.00


def _fin_volume(section: str, tip_chord: float = TIP_CHORD) -> tuple:
    rocket = {
        "fin_count": 1,
        "fin_root": ROOT_CHORD,
        "fin_tip": tip_chord,
        "fin_height": HEIGHT,
        "fin_sweep_deg": 45.0,
        "fin_thick": THICK,
        "fin_cross_section": section,
        "body_radius": BODY_R,
        "body_length": TOTAL_L,
        "fin_z_base_k2": 0.0,
        "profile": [(0.0, BODY_R), (TOTAL_L, BODY_R)],
    }
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("fin_section_probe")
        occ = gmsh.model.occ
        parts = _add_fins(occ, rocket, TOTAL_L)
        occ.synchronize()
        return sum(occ.getMass(d, t) for d, t in parts), len(parts)
    finally:
        gmsh.finalize()


@pytest.mark.cfd
def test_airfoil_section_is_thinner_than_the_slab():
    """A double-wedge removes a quarter of the slab; a square section keeps it."""
    v_square, n_square = _fin_volume("Square")
    v_airfoil, n_airfoil = _fin_volume("Airfoil")
    assert n_square == n_airfoil == 1

    # Section area: slab = c*t, double wedge with full thickness over the middle
    # half = 0.75*c*t. The ratio is independent of the planform.
    assert v_airfoil == pytest.approx(0.75 * v_square, rel=0.05), (
        f"airfoil fin volume {v_airfoil:.4e} m^3 against slab {v_square:.4e} — "
        f"the declared section did not reach the mesh"
    )


@pytest.mark.cfd
def test_rounded_section_is_between_the_slab_and_the_wedge():
    v_square, _ = _fin_volume("Square")
    v_round, n = _fin_volume("Rounded")
    assert n == 1
    # Rounding both ends removes (2 - pi/2)*(t/2)^2 per section — a small bite
    # out of a long chord, so it must be under the slab and over the wedge.
    assert 0.75 * v_square < v_round < v_square


@pytest.mark.cfd
def test_sharp_delta_tip_keeps_the_extruded_triangle():
    """No tip chord means no second section to loft to — planform still wins."""
    vol, n = _fin_volume("Airfoil", tip_chord=0.0)
    assert n == 1
    triangle = 0.5 * ROOT_CHORD * HEIGHT * THICK
    assert vol == pytest.approx(triangle, rel=0.15)


@pytest.mark.parametrize("section", ["Square", "Airfoil", "Rounded"])
def test_section_polygon_is_shape_stable_across_chords(section):
    """Root and tip must be the same polygon or the loft is not a solid."""
    long_chord = _fin_section_2d(0.20, THICK, section)
    short_chord = _fin_section_2d(0.01, THICK, section)
    assert len(long_chord) == len(short_chord)
    # And the thickness clamp must keep a short chord from turning inside out.
    assert max(abs(z) for _, z in short_chord) <= 0.2 * 0.01 + 1e-12
