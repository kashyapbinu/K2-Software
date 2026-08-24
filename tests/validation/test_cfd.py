"""
CFD validation gates.

The Taylor–Maccoll reference gate is fast and always runs. The SU2 cases are
slow (gmsh + SU2, minutes) and skip cleanly if the head-less pipeline fails.
"""
import pytest

from validation.cfd import benchmarks as B
from tests.validation.conftest import assert_benchmark


@pytest.mark.cfd
def test_taylor_maccoll_reference():
    assert_benchmark(B.bench_taylor_maccoll_reference())


@pytest.mark.cfd
@pytest.mark.slow
def test_su2_cone_vs_taylor_maccoll():
    assert_benchmark(B.bench_su2_cone())


@pytest.mark.cfd
@pytest.mark.slow
def test_barrowman_vs_su2():
    assert_benchmark(B.bench_barrowman_vs_su2())


@pytest.mark.cfd
def test_barrowman_vs_agardb_experiment():
    assert_benchmark(B.bench_agardb_barrowman())


@pytest.mark.cfd
@pytest.mark.slow
def test_su2_vs_agardb_experiment():
    assert_benchmark(B.bench_agardb_su2())


@pytest.mark.cfd
@pytest.mark.slow
def test_onera_m6_vs_experiment():
    assert_benchmark(B.bench_onera_m6())


@pytest.mark.cfd
def test_agardb_geometry_matches_the_published_specification():
    """AGARD-B is defined in body diameters; check the built assembly hits them.

    The dimensions below are the specification (AGARD AG-4/M3), not values read
    back from our own builder, so this fails if the geometry ever drifts.
    """
    from validation.cfd.agardb_geometry import (
        agardb_assembly, reference_quantities, DEFAULT_DIAMETER_M as D,
    )
    from validation.data.agardb_aedc import nose_radius
    from visualization.viewer_3d import nose_profile

    asm = agardb_assembly()
    assert abs(asm.total_length() / D - 8.5) < 1e-9

    ref = reference_quantities()
    assert abs(ref["s_ref_m2"] / D ** 2 - 6.928203) < 1e-5     # 4*sqrt(3)
    assert abs(ref["c_ref_m"] / D - 2.309401) < 1e-5           # 4*sqrt(3)/3

    # The nose must follow the published ogive equation, not K2's default
    # tangent ogive — the two differ by ~7% of the radius at mid-nose.
    zs, rs = nose_profile("AGARD-B", 3 * D, D / 2, 41)
    for z, r in zip(zs, rs):
        assert abs(r - nose_radius((3 * D - z) / D) * D) < 1e-12


@pytest.mark.cfd
def test_agardb_delta_tip_is_not_degenerate():
    """A zero tip chord makes the OCC fin builder fall back to a box.

    That fallback is silent and inflates lift by ~25%, so the truncation that
    avoids it is load-bearing and gets its own gate.
    """
    from validation.cfd.agardb_geometry import agardb_assembly
    from core.components import TrapezoidalFinSet

    fins = [c for c in agardb_assembly().all_components()
            if isinstance(c, TrapezoidalFinSet)]
    assert len(fins) == 1
    assert fins[0].tip_chord > 0.0
    # Tip chord and height must still lie on the true delta line: extrapolating
    # the taper back to zero chord has to recover the full 1.5 D semi-span.
    fin = fins[0]
    taper = (fin.root_chord - fin.tip_chord) / fin.height
    assert abs(fin.root_chord / taper - 0.15) < 1e-9   # 1.5 D with D = 0.1 m


@pytest.mark.cfd
def test_onera_m6_planform_reproduces_published_values():
    """Root/tip chords are derived from MAC and taper; check the derivation.

    Aspect ratio and trailing-edge sweep are *not* used to build the wing, so
    recovering the published 3.8 and 15.8 deg is an independent check.
    """
    from validation.cfd import onera_m6 as M6

    span_full = 2 * M6.SEMI_SPAN
    area_full = (M6.root_chord() + M6.tip_chord()) * M6.SEMI_SPAN
    assert abs(span_full ** 2 / area_full - 3.8) < 0.01
    assert abs(M6.te_sweep_deg() - M6.TE_SWEEP_DEG_PUBLISHED) < 0.1


@pytest.mark.cfd
def test_onera_m6_experimental_data_is_present_and_sane():
    from validation.cfd import onera_m6 as M6

    for section in M6.SECTIONS:
        for upper in (True, False):
            rows = M6.load_experimental_cp(section, upper)
            assert len(rows) >= 10
            assert all(0.0 <= x <= 1.05 for x, _ in rows)
            # Transonic upper surface: suction well below the M=0.84 vacuum bound.
            assert all(-2.0 < cp < 1.5 for _, cp in rows)


@pytest.mark.cfd
def test_onera_m6_cp_sign_convention():
    """The archive files store -Cp; the loader must return Cp.

    Worth its own gate because the error is invisible to a range check — both
    signs sit inside any plausible Cp band — and it silently turns the whole
    benchmark into a comparison against a mirror image of the measurement.
    """
    from validation.cfd import onera_m6 as M6

    for section in M6.SECTIONS:
        upper = M6.load_experimental_cp(section, True)
        lower = M6.load_experimental_cp(section, False)

        # Upper surface of a wing at 3 deg is suction-dominated.
        assert min(cp for _, cp in upper) < -0.8
        # Nothing anywhere may exceed the stagnation bound; Cp > 1 on the upper
        # surface is the signature of the un-negated file.
        assert max(cp for _, cp in upper) < 1.0
        # The forwardmost tap on each surface sits in the leading-edge
        # compression, so both read positive.
        assert upper[0][1] > 0.0
        assert lower[0][1] > 0.0
        # And the wing lifts: the upper surface carries the suction, by a margin
        # no plausible sign confusion survives. (The lower surface does go mildly
        # negative aft of ~10% chord — the section is symmetric, so thickness
        # accelerates the flow there too. That is why the *forward* tap and this
        # margin are the checks, and not the sign of the lower surface as a whole.)
        assert min(cp for _, cp in upper) < min(cp for _, cp in lower) - 0.3


@pytest.mark.cfd
def test_onera_m6_span_frame_finds_the_root_on_both_builds():
    """Span stations are measured off the solution, so the root must be located.

    The CAD import recentres what it is given, and the root sits mid-span on the
    mirrored build but at an end on the semi-span one — so neither end nor origin
    can be assumed.
    """
    pv = pytest.importorskip("pyvista")
    from validation.cfd import onera_m6 as M6

    def frame(mirror, shift):
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        stl = M6.build_m6_stl(d / "w.stl", n_span=40, mirror=mirror)
        mesh = pv.read(str(stl))
        mesh.points[:, 1] += shift          # stand in for the import's recentring
        return M6.span_frame(mesh)

    root, tip = frame(mirror=True, shift=0.0)
    assert abs(root) < 0.05 * M6.SEMI_SPAN
    assert abs(abs(tip) - M6.SEMI_SPAN) < 0.02 * M6.SEMI_SPAN

    # Semi-span, recentred the way analyze_cad recentres it: root at -b/2.
    root, tip = frame(mirror=False, shift=-0.5 * M6.SEMI_SPAN)
    assert abs(root - (-0.5 * M6.SEMI_SPAN)) < 0.05 * M6.SEMI_SPAN
    assert abs(tip - 0.5 * M6.SEMI_SPAN) < 0.02 * M6.SEMI_SPAN
    assert abs(tip - root) == pytest.approx(M6.SEMI_SPAN, rel=0.02)
