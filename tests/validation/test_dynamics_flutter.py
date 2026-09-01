"""
Fin flutter gates.

The dynamics section had no test coverage at all, and carried two independent
implementations of the same NACA TN-4197 equation:

    dynamics/flutter_analysis.py   t/c referenced to the MEAN chord
    structures/workstation.py      t/c referenced to the ROOT chord

Since V_f scales with (t/c)^1.5 these agreed only for an untapered fin and
diverged with taper — 1.54x apart at taper 0.5, 2.15x at taper 0.2 — so the
Dynamics tab and the Structures tab reported different flutter speeds for the
same fin. The root chord is now the single convention: it is the larger chord,
so the lower V_f, and fin flutter is sudden and destructive enough that
under-predicting its onset is the safe direction to be wrong in.

structures/workstation.py now calls flutter_speed() instead of re-deriving it.
"""
import math

import pytest

from cfd.solvers.base import isa_conditions
from dynamics.flutter_analysis import flutter_speed

ALU_G = 26e9
FIN = dict(span=0.10, root_chord=0.15, tip_chord=0.075,
           thickness=0.003, shear_modulus=ALU_G)


# ── the two implementations must not drift again ─────────────────────────────

@pytest.mark.parametrize("tip", [0.150, 0.110, 0.075, 0.030])
def test_workstation_and_dynamics_agree_at_every_taper(tip):
    """Taper is what separated them, so sweep it."""
    import structures.workstation as wks
    from structures.solvers.base import get_structural_material

    class State:
        length, diameter = 1.68, 0.155
        fin_root_chord, fin_tip_chord = 0.15, tip
        fin_span, fin_thickness, fin_count = 0.10, 0.003, 4

    flight = wks.FlightLoads(available=True, source="test", max_velocity=300.0,
                             max_mach=0.9, max_dynamic_pressure=40000.0,
                             maxq_altitude=3000.0)
    fa = wks.fin_analysis(State(), flight, "Aluminum 6061-T6")

    mat = get_structural_material("Aluminum 6061-T6")
    g = mat.G if mat.G > 0 else mat.E / (2 * (1 + mat.nu))
    expected = flutter_speed(0.10, 0.15, tip, 0.003, g, altitude_m=3000.0)

    assert fa.flutter_speed_m_s == pytest.approx(expected, rel=1e-12), (
        "structures and dynamics disagree on flutter speed again"
    )


def test_thickness_ratio_uses_the_root_chord():
    """Pin the convention: t/c must be t/root, not t/mean.

    A tapered fin makes the two differ; recomputing by hand with the root
    chord must reproduce flutter_speed() exactly.
    """
    span, root, tip, thick, alt = 0.10, 0.15, 0.075, 0.003, 3000.0
    P, T, _rho = isa_conditions(alt)
    a = math.sqrt(1.4 * 287.05 * T)
    S = 0.5 * (root + tip) * span
    AR = span ** 2 / S
    lam = tip / root
    tc_root = thick / root
    den = (1.337 * AR ** 3 * P * (lam + 1)) / (2 * (AR + 2) * tc_root ** 3)
    expected = a * math.sqrt(ALU_G / den)

    got = flutter_speed(span, root, tip, thick, ALU_G, altitude_m=alt)
    assert got == pytest.approx(expected, rel=1e-12)

    # ...and is measurably NOT the mean-chord form it used to be.
    tc_mean = thick / ((root + tip) / 2)
    den_mean = (1.337 * AR ** 3 * P * (lam + 1)) / (2 * (AR + 2) * tc_mean ** 3)
    assert got != pytest.approx(a * math.sqrt(ALU_G / den_mean), rel=1e-3)


def test_untapered_fin_is_where_the_two_conventions_coincide():
    """Explains why this hid: a square fin cannot show the bug."""
    square = flutter_speed(0.10, 0.15, 0.15, 0.003, ALU_G, altitude_m=3000.0)
    P, T, _ = isa_conditions(3000.0)
    a = math.sqrt(1.4 * 287.05 * T)
    S = 0.15 * 0.10
    AR = 0.10 ** 2 / S
    den = (1.337 * AR ** 3 * P * 2.0) / (2 * (AR + 2) * (0.003 / 0.15) ** 3)
    assert square == pytest.approx(a * math.sqrt(ALU_G / den), rel=1e-12)


# ── physical invariants of the formula ───────────────────────────────────────

def test_flutter_speed_rises_with_thickness_as_t_to_the_three_halves():
    v1 = flutter_speed(**{**FIN, "thickness": 0.003})
    v2 = flutter_speed(**{**FIN, "thickness": 0.006})
    assert v2 / v1 == pytest.approx(2 ** 1.5, rel=1e-9)


def test_flutter_speed_rises_with_the_square_root_of_shear_modulus():
    v1 = flutter_speed(**{**FIN, "shear_modulus": 26e9})
    v2 = flutter_speed(**{**FIN, "shear_modulus": 104e9})
    assert v2 / v1 == pytest.approx(2.0, rel=1e-9)


def test_flutter_speed_rises_with_altitude():
    """P falls with altitude, so the true-airspeed flutter onset rises."""
    speeds = [flutter_speed(**FIN, altitude_m=h) for h in (0, 5000, 10000)]
    assert speeds == sorted(speeds)


def test_degenerate_fins_do_not_produce_a_finite_flutter_speed():
    """Guarded inputs return inf rather than 0, so they cannot read as a
    real, very low flutter speed."""
    for key in ("span", "thickness", "root_chord"):
        assert flutter_speed(**{**FIN, key: 0.0}) == float("inf")


def test_a_realistic_aluminium_fin_flutters_well_above_flight_speed():
    """Order-of-magnitude sanity: a 3 mm alu fin is not marginal at Mach 1."""
    v = flutter_speed(**FIN, altitude_m=3000.0)
    assert 500.0 < v < 5000.0, v
