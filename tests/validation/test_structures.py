"""
Structural validation gates.

The closed-form formula gate runs fast and always. The CalculiX cases need the
bundled ccx.exe and are marked slow (minutes).
"""
import pytest

from validation.structures import benchmarks as B
from tests.validation.conftest import assert_benchmark


@pytest.mark.structures
def test_closed_form_formulas():
    assert_benchmark(B.bench_closed_form_formulas())


@pytest.mark.structures
@pytest.mark.slow
def test_bar_tension():
    assert_benchmark(B.bench_bar_tension())


@pytest.mark.structures
@pytest.mark.slow
def test_cantilever_bending():
    assert_benchmark(B.bench_cantilever_bending())


@pytest.mark.structures
@pytest.mark.slow
def test_modal_vs_ccx():
    assert_benchmark(B.bench_modal_vs_ccx())


@pytest.mark.structures
@pytest.mark.slow
def test_nafems_le1():
    assert_benchmark(B.bench_nafems_le1())


@pytest.mark.structures
@pytest.mark.slow
def test_nafems_le10():
    assert_benchmark(B.bench_nafems_le10())


@pytest.mark.structures
def test_nafems_mesh_lands_on_the_published_geometry():
    """The mapped mesh must actually reach the points the targets refer to.

    A benchmark that quietly meshed a slightly different ellipse would still
    'pass' at a loose tolerance, so pin the corners the NAFEMS definition names:
    D=(2,0), C=(3.25,0), A=(0,1), B=(0,2.75).
    """
    from validation.structures import nafems as N

    m = N.make_elliptic_mesh(N.LE10_THICKNESS, ns=4, nt=8, nz=2)
    x, y, z = m.nodes[m.node_at_D(top=True)]
    assert (round(x, 9), round(y, 9)) == (2.0, 0.0)
    assert round(z, 9) == round(N.LE10_THICKNESS / 2, 9)

    corner = {
        m.nid(m.ns, 0, 0): (3.25, 0.0),          # C
        m.nid(0, m.nt, 0): (0.0, 1.0),           # A
        m.nid(m.ns, m.nt, 0): (0.0, 2.75),       # B
    }
    for nid, (ex, ey) in corner.items():
        gx, gy, _ = m.nodes[nid]
        assert abs(gx - ex) < 1e-9 and abs(gy - ey) < 1e-9

    # Every node must sit between the two ellipses, i.e. the mapping never folds.
    for gx, gy, _ in m.nodes.values():
        inner = (gx / N.A_INNER) ** 2 + (gy / N.B_INNER) ** 2
        outer = (gx / N.A_OUTER) ** 2 + (gy / N.B_OUTER) ** 2
        assert inner >= 1.0 - 1e-9
        assert outer <= 1.0 + 1e-9
