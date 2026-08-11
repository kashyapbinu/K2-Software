"""Wall-resolution gates: the mesh must not lie about where its wall is.

Two separate failures live here, both of which produced confident-looking
numbers and no error.

The first is the standing one: the mesh is tet-only, the first cell sits near
y+ 3500, and skin friction is an order of magnitude below flat plate. That is
known and reported at config time.

The second is what an attempt to fix it produced. Prism extrusion DOES run on
the OCC boolean-cut domain on Gmsh 4.15.2 — the old "it crashes generate(3)"
claim no longer reproduces — but the tets then fill the boundary-layer region
as well, so prisms and tets overlap and the wall stops being a boundary. Every
downstream signal stayed healthy: element counts, Gmsh quality metrics, SU2's
own mesh-quality table, the .su2 file. The only symptom was a solve whose
residual never fell below its starting value before going NaN, across five
different numerics variants.

So these tests are mostly about the cheap topology audit that catches that,
because the expensive geometric ones did not.
"""
import pytest

from cfd.boundary_layer import predict_wall_yplus
from cfd.solvers.su2_solver import _wall_spacing_from_su2_mesh

# M = 0.8 at 3000 m ISA, the case every measured number in these comments used.
V_INF, RHO, MU, REF_L = 265.0, 0.9093, 1.694e-5, 1.0


def _su2_mesh(path, elements, points, wall_tris):
    path.write_text(
        "NDIME= 3\n"
        f"NELEM= {len(elements)}\n"
        + "".join(f"{t} {' '.join(str(n) for n in ns)} {i}\n"
                 for i, (t, ns) in enumerate(elements))
        + f"NPOIN= {len(points)}\n"
        + "".join(f"{x} {y} {z} {i}\n" for i, (x, y, z) in enumerate(points))
        + "NMARK= 1\nMARKER_TAG= rocket_wall\n"
        f"MARKER_ELEMS= {len(wall_tris)}\n"
        + "".join(f"5 {' '.join(str(n) for n in t)}\n" for t in wall_tris)
    )
    return path


# ── the audit that catches an overlapping mesh ───────────────────────────────

def test_manifold_audit_accepts_a_wall_with_one_cell_behind_it():
    """A valid wall face bounds exactly one cell: fluid on one side, nothing
    on the other. This is the shape of every correct mesh."""
    from cfd.meshing import _audit_wall_is_manifold

    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.model.add("manifold_ok")
        occ = gmsh.model.occ
        box = occ.addBox(-2, -2, -2, 8, 4, 4)
        body = occ.addCylinder(0, 0, 0, 2, 0, 0, 0.3)
        out, _ = occ.cut([(3, box)], [(3, body)])
        occ.synchronize()
        vol = out[0][1]
        wall = [
            t for d, t in gmsh.model.getBoundary([(3, vol)], oriented=False)
            if abs(occ.getCenterOfMass(d, t)[0]) < 2.0
            and (occ.getCenterOfMass(d, t)[1] ** 2
                 + occ.getCenterOfMass(d, t)[2] ** 2) ** 0.5 < 1.0
        ]
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.15)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
        gmsh.model.mesh.generate(3)

        assert _audit_wall_is_manifold(gmsh, wall) == 0
    finally:
        gmsh.finalize()


def test_manifold_audit_catches_cells_on_both_sides_of_the_wall():
    """The failure that shipped past every other check.

    Extruding a boundary layer without rebuilding the volume leaves the tet
    pass free to fill the same region, so wall faces end up with a cell on
    each side. Measured on the real coarse mesh, all 6066 wall triangles were
    like this, and nothing except the diverging solve noticed.
    """
    from cfd.meshing import _audit_wall_is_manifold

    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.model.add("manifold_bad")
        occ = gmsh.model.occ
        box = occ.addBox(-2, -2, -2, 8, 4, 4)
        body = occ.addCylinder(0, 0, 0, 2, 0, 0, 0.3)
        out, _ = occ.cut([(3, box)], [(3, body)])
        occ.synchronize()
        vol = out[0][1]
        wall = [
            t for d, t in gmsh.model.getBoundary([(3, vol)], oriented=False)
            if abs(occ.getCenterOfMass(d, t)[0]) < 2.0
            and (occ.getCenterOfMass(d, t)[1] ** 2
                 + occ.getCenterOfMass(d, t)[2] ** 2) ** 0.5 < 1.0
        ]
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.15)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1.0)
        gmsh.model.mesh.generate(2)
        gmsh.model.geo.extrudeBoundaryLayer(
            [(2, t) for t in wall], [1] * 4,
            [1e-4, 2.2e-4, 3.6e-4, 5.4e-4], True,
        )
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        buried = _audit_wall_is_manifold(gmsh, wall)
        assert buried and buried > 0, (
            "the audit passed an overlapping prism/tet mesh — this is exactly "
            "the mesh SU2 cannot converge on, and nothing else detects it"
        )
    finally:
        gmsh.finalize()


# ── measuring the wall spacing the y+ prediction is built on ─────────────────

def test_wall_spacing_uses_the_facet_size_on_a_tet_mesh(tmp_path):
    """With no prisms there is no wall-normal direction to measure, so the
    facet size is the only proxy available — and it must be labelled as such,
    because it is an approximation standing in for a real measurement."""
    edge = 1.0e-2
    mesh = _su2_mesh(
        tmp_path / "tet.su2",
        elements=[(10, [0, 1, 2, 3])],
        points=[(0, 0, 0), (edge, 0, 0), (0, edge, 0), (0, 0, edge)],
        wall_tris=[(0, 1, 2)],
    )

    out = _wall_spacing_from_su2_mesh(mesh)
    assert out["source"] == "tet"
    assert out["normal"] == pytest.approx(edge, rel=1e-6)
    assert not predict_wall_yplus(out["normal"], V_INF, RHO, MU, REF_L)["wall_resolved"]


def test_wall_spacing_would_read_a_prism_stack_wall_normal(tmp_path):
    """y+ is a wall-NORMAL quantity, and the two spacings differ by ~1000x.

    No mesher here emits prisms today, but the probe handles them because
    reading the triangle edge on a prism mesh would report a tet-mesh y+ —
    calling the wall unresolved on exactly the mesh that resolves it. Keeping
    the distinction means a future prism mesh is measured, not assumed.
    """
    first_h, edge = 4.0e-6, 1.0e-2
    mesh = _su2_mesh(
        tmp_path / "prism.su2",
        elements=[(13, [0, 1, 2, 3, 4, 5])],
        points=[(0, 0, 0), (edge, 0, 0), (0, edge, 0),
                (0, 0, first_h), (edge, 0, first_h), (0, edge, first_h)],
        wall_tris=[(0, 1, 2)],
    )

    out = _wall_spacing_from_su2_mesh(mesh)
    assert out["source"] == "prism"
    assert out["normal"] == pytest.approx(first_h, rel=1e-6)
    assert out["tangential"] == pytest.approx(edge, rel=1e-6)
    assert predict_wall_yplus(out["normal"], V_INF, RHO, MU, REF_L)["wall_resolved"]


def test_yplus_regimes_are_reported_not_smoothed_over():
    """The buffer layer must be named as invalid rather than rounded toward
    whichever neighbour looks better; it is the one band where neither wall
    resolution nor a wall function applies."""
    fine = predict_wall_yplus(1e-6, V_INF, RHO, MU, REF_L)
    assert fine["wall_resolved"] and fine["regime"] == "wall-resolved"

    coarse = predict_wall_yplus(1e-2, V_INF, RHO, MU, REF_L)
    assert not coarse["wall_resolved"] and coarse["regime"] == "under-resolved"
    assert coarse["y_plus"] > 300

    # Bracket the buffer layer by construction rather than by a guessed spacing.
    target = predict_wall_yplus(1e-6, V_INF, RHO, MU, REF_L)
    buffer_spacing = 1e-6 * (10.0 / target["y_plus"])
    assert "buffer" in predict_wall_yplus(
        buffer_spacing, V_INF, RHO, MU, REF_L)["regime"]


# ── end to end ───────────────────────────────────────────────────────────────

@pytest.mark.cfd
@pytest.mark.slow
def test_shipped_mesh_is_tet_only_and_its_wall_is_a_real_boundary(tmp_path):
    """The mesh the pipeline actually produces, checked for both faults.

    Tet-only is the current, documented limitation — if prisms ever appear here
    without the volume being rebuilt against them, the manifold check inside
    build_wind_tunnel_mesh logs MESH INVALID and this catches the type-13
    elements directly.
    """
    from cfd.meshing import build_wind_tunnel_mesh

    rocket = {
        "length": 1.0, "body_radius": 0.05, "nose_radius": 0.05,
        "nose_length": 0.3, "body_length": 0.7, "max_diameter": 0.1,
        "fin_count": 4, "fin_height": 0.1, "fin_root": 0.15, "fin_thick": 0.003,
    }
    stl = tmp_path / "geometry.stl"
    stl.touch()

    out = build_wind_tunnel_mesh(
        stl_path=stl, output_path=tmp_path / "m", refinement="coarse",
        domain_length_scale=6.0, domain_radius_scale=20.0,
        geometry_dict=rocket,
    )
    assert out.is_file()

    text = out.read_text(errors="replace")
    assert "rocket_wall" in text and "farfield" in text

    # Count VTK type 13 (wedge/prism) inside the element block only. Matching
    # "13" against the whole file also hits point coordinates.
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("NELEM="))
    n_elem = int(lines[start].split("=")[1].split()[0])
    types = [l.split()[0] for l in lines[start + 1:start + 1 + n_elem] if l.split()]
    assert types.count("13") == 0, (
        "prisms appeared in a mesh built without a volume rebuild — see "
        "cfd/meshing.py step 7; this mesh will not converge"
    )
    assert types.count("10") > 1000, "expected a tetrahedral volume mesh"

    spacing = _wall_spacing_from_su2_mesh(out)
    assert spacing["source"] == "tet"
    assert spacing["normal"] > 0
