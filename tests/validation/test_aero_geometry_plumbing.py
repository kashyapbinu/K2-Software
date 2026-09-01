"""
The analytic aero model must be fed the geometry the benchmark claims to test.

Both failures guarded here were silent: they produced plausible numbers for the
wrong vehicle, and the validation report presented them as method error.

1. ``AeroModel.from_state`` reads FLAT fields off the state object and never
   looks at ``state.assembly``. Its ``or``-fallbacks then invent a generic
   4-fin rocket from the body dimensions, so a benchmark that sets only the
   assembly measures fins it never specified. This is how the AGARD-B
   Barrowman comparison reported 63% error against wind-tunnel data while
   actually modelling a 0.6 D unswept fin.

2. ``fin_sweep_angle`` is RADIANS on ``RocketState`` (``compute_fin_cn_alpha``
   does a bare ``math.tan``) and DEGREES on ``TrapezoidalFinSet``
   (``components.py`` converts with ``math.radians``). A degree value stored
   straight into the state field gives ``tan(30) = -6.4``: a fin swept forward
   several root chords.
"""
import math

import pytest

from core.components import TrapezoidalFinSet
from physics.aerodynamics import AeroModel


def _find(assembly, cls):
    stack = list(getattr(assembly, "stages", []) or [])
    while stack:
        node = stack.pop(0)
        if isinstance(node, cls):
            return node
        stack.extend(getattr(node, "children", []) or [])
    return None


def test_agardb_barrowman_uses_the_agardb_wing():
    """The AGARD-B delta must reach AeroModel, not a fabricated default fin."""
    from validation.cfd.agardb import barrowman_cl_alpha_per_deg  # noqa: F401
    from validation.cfd import agardb
    from validation.cfd.agardb_geometry import agardb_assembly, DEFAULT_DIAMETER_M

    captured = {}
    real_from_state = AeroModel.from_state

    def spy(state):
        model = real_from_state(state)
        captured["model"] = model
        return model

    AeroModel.from_state = staticmethod(spy)
    try:
        agardb.barrowman_cl_alpha_per_deg(mach=0.2)
    finally:
        AeroModel.from_state = real_from_state

    model = captured["model"]
    wing = _find(agardb_assembly(DEFAULT_DIAMETER_M), TrapezoidalFinSet)

    assert model.fin_count == wing.fin_count == 2
    assert model.fin_span == pytest.approx(wing.height, rel=1e-9)
    assert model.fin_root_chord == pytest.approx(wing.root_chord, rel=1e-9)
    assert model.fin_tip_chord == pytest.approx(wing.tip_chord, rel=1e-9)
    # Degrees on the component, radians on the model.
    assert model.fin_sweep == pytest.approx(math.radians(wing.sweep_angle), rel=1e-9)


def test_canonical_state_stores_fin_sweep_in_radians():
    """State sweep must be the radian form of the assembly's degree value."""
    from validation.cases.rocket_canonical import canonical_state, canonical_assembly

    state = canonical_state()
    fins = _find(canonical_assembly(), TrapezoidalFinSet)

    assert state.fin_sweep_angle == pytest.approx(math.radians(fins.sweep_angle))
    # A degrees-as-radians value sweeps the fin forward: tan < 0 for 30 rad.
    assert math.tan(state.fin_sweep_angle) > 0.0


def test_stage_geometry_converts_fin_sweep_to_radians():
    """The multistage path shares the state's radian convention."""
    from core.staging import _extract_stage_geometry
    from validation.cases.rocket_canonical import canonical_assembly

    asm = canonical_assembly()
    fins = _find(asm, TrapezoidalFinSet)
    geom = _extract_stage_geometry(asm.stages[0])

    assert geom["fin_sweep_angle"] == pytest.approx(math.radians(fins.sweep_angle))


def test_cruciform_set_lifts_with_two_panels_not_four():
    """Only fins presenting a surface to the cross-flow carry normal force.

    A 4-fin cruciform set has two lifting panels and two edge-on to the
    cross-flow, so it must produce the same CN_alpha as a 2-fin planar set of
    the same panel -- not twice it. Counting all four put the canonical rocket
    58% above SU2 while the 2-panel AGARD-B wing sat 21% below wind-tunnel data.
    """
    from physics.aerodynamics import compute_fin_cn_alpha

    kw = dict(fin_span=0.12, fin_root_chord=0.20, fin_tip_chord=0.08,
              body_radius=0.051, sweep_angle=math.radians(30.0), mach=0.5)

    two = compute_fin_cn_alpha(fin_count=2, **kw)
    four = compute_fin_cn_alpha(fin_count=4, **kw)
    three = compute_fin_cn_alpha(fin_count=3, **kw)
    one = compute_fin_cn_alpha(fin_count=1, **kw)

    assert four == pytest.approx(two, rel=1e-9)
    assert three == pytest.approx(two * 0.75, rel=1e-9)
    assert one == pytest.approx(two / 2.0, rel=1e-9)


def test_canonical_normal_force_is_within_ten_percent_of_su2():
    """The analytic model must agree with its own RANS solver to 10%.

    SU2 reported CL = 1.302, CD = 0.4046 for the canonical rocket at M=0.5,
    alpha=4 deg (validation/report/benchmarks.json). Rotated into body axes and
    with both halves of the wing-body interference present, Barrowman lands
    within 10% with no one-sided bias left.

    Drag is deliberately NOT asserted here: K2 keeps OpenRocket's base-drag
    model, which the AGARD-B C_D0 diagnostic shows running well high.
    """
    from validation.cases.rocket_canonical import canonical_state
    from physics.aerodynamics import AeroModel
    from environment.atmosphere_model import Atmosphere

    state = canonical_state()
    atm = Atmosphere()
    v = 0.5 * atm.speed_of_sound(3000.0)
    q = 0.5 * atm.density(3000.0) * v * v
    aoa = math.radians(4.0)

    k2 = AeroModel.from_state(state).compute(
        alpha=aoa, mach=0.5, q_dyn=q, pitch_rate=0.0, v_rel=v,
        cg=state.cg or 1.2)
    cn_su2 = 1.302 * math.cos(aoa) + 0.4046 * math.sin(aoa)

    assert abs(k2["cn"] - cn_su2) / cn_su2 < 0.10


def test_wing_body_carryover_matches_the_agardb_wind_tunnel():
    """Both halves of the wing-body interference, checked against measurement.

    Barrowman's K_fb = 1 + tau is only the fin-in-presence-of-body factor; the
    fin's carryover lift onto the body is the other (1 + tau). With just the
    first the analytic slope is 21% below AEDC-TR-70-100 at every Mach, which
    is a missing term, not scatter. With (1 + tau)^2 it is inside 5%.
    """
    from validation.cfd.agardb import barrowman_cl_alpha_per_deg
    from validation.data.agardb_aedc import AEDC_TR_70_100 as REF

    for mach in (0.2, 0.4, 0.6, 0.8, 0.9):
        k2 = barrowman_cl_alpha_per_deg(mach=mach)
        ref = REF[mach]["cl_alpha_per_deg"]
        assert abs(k2 - ref) / ref < 0.10, f"M={mach}: {k2:.5f} vs {ref:.5f}"


def test_interference_factor_follows_openrocket_across_mach():
    """(1+tau)^2 subsonic, (1+tau) supersonic, continuous blend between.

    OpenRocket's FinSetCalc.calculateBodyFinInterferenceFactor blends the
    body-carryover half out over the transonic CNa interval, because the
    Mach-cone geometry it needs is not modelled there. K2 matches that shape
    rather than applying (1+tau)^2 everywhere.
    """
    from physics.aerodynamics import (body_fin_interference_factor,
                                      CNA_SUBSONIC_MACH)
    from physics.drag_tables import CNA_SUPERSONIC_MACH

    tau = 0.051 / (0.051 + 0.12)
    sub, sup = (1 + tau) ** 2, 1 + tau

    assert body_fin_interference_factor(tau, 0.0) == pytest.approx(sub)
    assert body_fin_interference_factor(tau, CNA_SUBSONIC_MACH) == pytest.approx(sub)
    assert body_fin_interference_factor(tau, 3.0) == pytest.approx(sup)
    assert body_fin_interference_factor(tau, CNA_SUPERSONIC_MACH) == pytest.approx(sup)

    # Continuous and monotone through the blend, no step at either end.
    mid = 0.5 * (CNA_SUBSONIC_MACH + CNA_SUPERSONIC_MACH)
    assert sup < body_fin_interference_factor(tau, mid) < sub
    prev = sub
    m = CNA_SUBSONIC_MACH
    while m <= CNA_SUPERSONIC_MACH + 1e-9:
        cur = body_fin_interference_factor(tau, m)
        assert cur <= prev + 1e-12
        prev = cur
        m += 0.05
    assert prev == pytest.approx(sup, rel=1e-6)

    # tau -> 0 (vanishing body) leaves no interference at all.
    assert body_fin_interference_factor(0.0, 0.5) == pytest.approx(1.0)


def test_transonic_cna_interpolation_matches_openrocket_constraints():
    """The quartic must satisfy all five constraints OpenRocket imposes.

    FinSetCalc builds its PolyInterpolator over
    {values at 0.9 and 1.5}, {derivatives at 0.9 and 1.5}, {2nd deriv at 0.9},
    then calls interpolate(mach, subV, superV, subD, superD, 0). Five
    constraints, so a quartic -- not the straight line K2 had, which was
    continuous in value but cornered in slope at both ends.
    """
    from physics.aerodynamics import _cna_transonic, CNA_SUBSONIC_MACH
    from physics.drag_tables import CNA_SUPERSONIC_MACH

    sub_v, sup_v, sub_d, sup_d = 2.5, 0.8, 1.6, -4.4
    x1, x2 = CNA_SUBSONIC_MACH, CNA_SUPERSONIC_MACH
    f = lambda m: _cna_transonic(m, sub_v, sup_v, sub_d, sup_d)
    h = 1e-6

    assert f(x1) == pytest.approx(sub_v, rel=1e-9)
    assert f(x2) == pytest.approx(sup_v, rel=1e-9)
    assert (f(x1 + h) - f(x1 - h)) / (2 * h) == pytest.approx(sub_d, rel=1e-5)
    assert (f(x2 + h) - f(x2 - h)) / (2 * h) == pytest.approx(sup_d, rel=1e-5)
    # Second differences divide by h**2, so h=1e-6 is rounding-dominated here
    # (it reports 1.4e-2 for a quantity that is zero). 1e-4 is the minimum of
    # the rounding/truncation curve for this function.
    h2 = 1e-4
    d2 = (f(x1 + h2) - 2 * f(x1) + f(x1 - h2)) / h2 ** 2
    assert abs(d2) < 1e-3, f"second derivative at {x1} should vanish, got {d2}"


def test_fin_cna_is_smooth_across_the_subsonic_joint():
    """No slope discontinuity at M=0.9 in cna1 (the linear blend had one)."""
    from physics.aerodynamics import (compute_fin_cn_alpha,
                                      body_fin_interference_factor,
                                      CNA_SUBSONIC_MACH)

    kw = dict(fin_count=4, fin_span=0.12, fin_root_chord=0.20,
              fin_tip_chord=0.08, body_radius=0.051,
              sweep_angle=math.radians(30.0))
    tau = 0.051 / (0.051 + 0.12)

    def cna1(m):   # strip the fin-count and interference factors
        return (compute_fin_cn_alpha(mach=m, **kw)
                / (2.0 * body_fin_interference_factor(tau, m)))

    h = 1e-5
    m = CNA_SUBSONIC_MACH
    left = (cna1(m) - cna1(m - h)) / h
    right = (cna1(m + h) - cna1(m)) / h
    assert abs(cna1(m + 1e-9) - cna1(m - 1e-9)) < 1e-6      # value
    assert abs(right - left) < 1e-3                          # slope
