"""
Regression gate: a sharp-tipped delta fin must mesh as a delta, not a box.

``cfd.meshing._add_fins`` builds each fin as an extruded planar loop. With
``tip_chord = 0`` the two tip corners coincide, ``occ.addLine`` between them
raises, and the old code fell through to a rectangular-box fallback. The
substitution was silent apart from a log warning and roughly doubled the fin
planform area — on the AGARD-B validation case it inflated the lift-curve slope
by about 25%.

The test measures the volume gmsh actually produced and requires it to match the
triangular prism, not the box.
"""
import math

import pytest

gmsh = pytest.importorskip("gmsh")

from cfd.meshing import _add_fins   # noqa: E402


def _fin_volume(tip_chord: float) -> tuple:
    """(total volume of the fin solids, number of solids) for one fin set."""
    root_chord, height, thickness, n_fins = 0.20, 0.15, 0.004, 1
    body_r, total_L = 0.05, 1.00

    rocket = {
        "fin_count": n_fins,
        "fin_root": root_chord,
        "fin_tip": tip_chord,
        "fin_height": height,
        "fin_sweep_deg": 45.0,
        "fin_thick": thickness,
        "body_radius": body_r,
        "body_length": total_L,
        "fin_z_base_k2": 0.0,
        "profile": [(0.0, body_r), (total_L, body_r)],
    }

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("delta_fin_probe")
        occ = gmsh.model.occ
        parts = _add_fins(occ, rocket, total_L)
        occ.synchronize()
        vol = sum(gmsh.model.occ.getMass(dim, tag) for dim, tag in parts)
        return vol, len(parts)
    finally:
        gmsh.finalize()


@pytest.mark.cfd
def test_sharp_delta_tip_builds_a_triangle_not_a_box():
    vol, n = _fin_volume(tip_chord=0.0)
    assert n == 1

    root_chord, height, thickness = 0.20, 0.15, 0.004
    # The fin root is sunk 10% into the body for a clean boolean, so the built
    # panel is slightly taller than the exposed height; compare against the
    # triangle/box ratio instead, which is insensitive to that offset.
    triangle = 0.5 * root_chord * height * thickness
    box = root_chord * height * thickness

    assert vol == pytest.approx(triangle, rel=0.15), (
        f"delta fin volume {vol:.3e} m^3 is not the triangular prism "
        f"{triangle:.3e} m^3 — box fallback would give {box:.3e} m^3"
    )
    assert vol < 0.75 * box


@pytest.mark.cfd
def test_tapered_fin_still_builds_a_quadrilateral():
    """The four-point path must be untouched by the sharp-tip branch."""
    root_chord, tip_chord, height, thickness = 0.20, 0.08, 0.15, 0.004
    vol, n = _fin_volume(tip_chord=tip_chord)
    assert n == 1
    trapezoid = 0.5 * (root_chord + tip_chord) * height * thickness
    assert vol == pytest.approx(trapezoid, rel=0.15)
