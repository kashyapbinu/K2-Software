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


# ── structural modal frequencies feeding the p-k solver ──────────────────────

def _props(span=0.10, root=0.15, tip=0.075, thick=0.003, dens=2700.0):
    from dynamics.flutter_analysis import _fin_section_properties
    return _fin_section_properties(root, tip, thick, span, dens)


def test_torsion_frequency_is_dimensionally_a_frequency():
    """f_t must scale as 1/L with GJ/I_alpha held fixed.

    The old form was (1/2pi)*sqrt(GJ/(I_alpha*L)). With I_alpha per unit span
    (kg*m, which is what _fin_section_properties returns) that radicand is
    N*m^2/(kg*m^2) = m/s^2, so its square root is not a frequency at all. The
    resulting error was a factor of (2/pi)*sqrt(L) — span-dependent, so not
    even a constant calibration offset.
    """
    from dynamics.flutter_analysis import _torsion_fundamental_freq
    f1 = _torsion_fundamental_freq(26e9, 1e-9, 1e-3, 0.10)
    f4 = _torsion_fundamental_freq(26e9, 1e-9, 1e-3, 0.40)
    assert f1 / f4 == pytest.approx(4.0, rel=1e-12)


def test_torsion_frequency_matches_the_quarter_wave_closed_form():
    from dynamics.flutter_analysis import _torsion_fundamental_freq
    G, J, I_a, L = 26e9, 1.0125e-9, 9.61084e-4, 0.10
    expected = (1.0 / (4.0 * L)) * math.sqrt(G * J / I_a)
    assert _torsion_fundamental_freq(G, J, I_a, L) == pytest.approx(expected, rel=1e-12)


def test_torsion_sits_above_first_bending():
    """Bending-torsion flutter is a lower bending branch coalescing with a
    higher torsion branch. The old formula inverted that ordering."""
    from dynamics.flutter_analysis import (_cantilever_bending_freq,
                                           _torsion_fundamental_freq)
    p = _props()
    f_b = _cantilever_bending_freq(70e9, p["I_bend"], p["m_bar"], 0.10)
    f_t = _torsion_fundamental_freq(26e9, p["J"], p["I_alpha"], 0.10)
    assert f_t > f_b, f"torsion {f_t:.1f} Hz must exceed bending {f_b:.1f} Hz"


def test_theodorsen_hits_its_known_limits():
    """C(0)=1 (quasi-steady), C(inf)=0.5, and the lag term is negative."""
    from dynamics.flutter_analysis import theodorsen_C
    assert theodorsen_C(0.0) == complex(1.0, 0.0)
    assert abs(theodorsen_C(1e-8) - 1.0) < 1e-3
    assert theodorsen_C(1e6).real == pytest.approx(0.5, abs=1e-6)
    assert theodorsen_C(0.5).imag < 0.0


# ── p-k / k-method solver ────────────────────────────────────────────────────
#
# The solver previously returned inf for every fin: the plunge row of the
# aeroelastic system was scaled by the mass ratio mu while its aerodynamic
# terms were scaled by 1/mu, so aerodynamic coupling in that row was ~mu^2
# (5600x for a typical fin) too weak and damping could never cross zero.
# Rebuilt as the classical k-method, where every Theodorsen term carries a
# factor Omega = (w/wa)^2 and the aerodynamics folds into the mass side.

ALU = dict(span=0.10, root_chord=0.15, tip_chord=0.075, thickness=0.003,
           E=70e9, G=26e9, density_mat=2700.0)


def _pk(**kw):
    from dynamics.flutter_analysis import pk_flutter_analysis
    args = dict(ALU, altitude_m=0.0, max_flight_speed=8000.0)
    args.update(kw)
    return pk_flutter_analysis(**args)


def _divergence_speed_analytic(span=0.10, root=0.15, tip=0.075, thick=0.003,
                               dens=2700.0, G=26e9, rho_air=1.225):
    """Static divergence of the typical section: U*_D^2 = mu*r_a^2/(2(a+1/2)).

    Falls out of the torsion equation at k -> 0, where C(0) = 1. With the EA
    at mid-chord (a = 0) this is a closed form the solver must reproduce.
    """
    from dynamics.flutter_analysis import (_fin_section_properties,
                                           _torsion_fundamental_freq)
    p = _fin_section_properties(root, tip, thick, span, dens)
    b = p["c_mean"] / 2.0
    mu = p["m_bar"] / (math.pi * rho_air * b ** 2)
    r_a2 = p["I_alpha"] / (p["m_bar"] * b ** 2)
    omega_a = 2.0 * math.pi * _torsion_fundamental_freq(G, p["J"], p["I_alpha"], span)
    return math.sqrt(mu * r_a2 / (2.0 * 0.5)) * b * omega_a


def test_radius_of_gyration_matches_the_uniform_slab():
    """r_alpha^2 = I_alpha/(m*b^2) = c^2/(12 b^2) = 1/3 for a uniform plate."""
    from dynamics.flutter_analysis import _fin_section_properties
    p = _fin_section_properties(0.15, 0.075, 0.003, 0.10, 2700.0)
    b = p["c_mean"] / 2.0
    assert p["I_alpha"] / (p["m_bar"] * b ** 2) == pytest.approx(1.0 / 3.0, rel=1e-12)


def test_solver_finds_flutter_for_a_realistic_fin():
    """The whole point: it used to return inf for every fin ever tried."""
    r = _pk()
    assert math.isfinite(r["flutter_speed_mps"]), "solver found no flutter again"
    assert r["flutter_speed_mps"] > 0
    assert r["flutter_frequency_hz"] > 0
    assert r["envelope_status"] != "NOT_REACHED"


def test_damping_actually_crosses_zero():
    r = _pk()
    assert any(g > 0 for _v, g, _n in r["vg_data"]), (
        "no positive damping anywhere in the V-g sweep"
    )


def test_solver_reproduces_the_analytic_divergence_asymptote():
    """At k -> 0 one branch must run out to the static divergence speed.

    This is the load-bearing check on the aerodynamic matrix: divergence has
    a closed form, so any error in the Theodorsen assembly shows up here.
    """
    r = _pk()
    v_div = _divergence_speed_analytic()
    v_max_branch = max(v for v, _g, _n in r["vg_data"])
    # the plunge branch runs away as w -> 0; the torsion branch asymptotes
    branch_maxima = sorted(
        max(v for v, _g, n in r["vg_data"] if n == lbl)
        for lbl in {n for _v, _g, n in r["vg_data"]}
    )
    assert branch_maxima[0] == pytest.approx(v_div, rel=0.05), (
        f"divergence asymptote {branch_maxima[0]:.1f} vs analytic {v_div:.1f}"
    )
    assert v_max_branch > v_div


def test_flutter_precedes_divergence():
    r = _pk()
    assert r["flutter_speed_mps"] < _divergence_speed_analytic()


def test_flutter_speed_falls_with_thinner_fins():
    speeds = [_pk(thickness=t)["flutter_speed_mps"]
              for t in (0.004, 0.003, 0.002, 0.001)]
    assert speeds == sorted(speeds, reverse=True), speeds
    assert all(math.isfinite(s) for s in speeds)


def test_flutter_speed_rises_with_altitude():
    """Thinner air -> higher mass ratio -> higher onset speed."""
    speeds = [_pk(altitude_m=h)["flutter_speed_mps"] for h in (0, 5000, 10000)]
    assert speeds == sorted(speeds), speeds


def test_pk_is_more_conservative_than_the_naca_correlation():
    """The 2-DOF unsteady solve should sit below the empirical formula.

    Not a tight calibration - just the ordering, which is what makes taking
    min(NACA, p-k) meaningful rather than a no-op.
    """
    r = _pk()
    naca = flutter_speed(ALU["span"], ALU["root_chord"], ALU["tip_chord"],
                         ALU["thickness"], ALU["G"], altitude_m=0.0)
    assert 0.3 < r["flutter_speed_mps"] / naca < 1.0


def test_bending_above_torsion_suppresses_binary_flutter():
    """A classical result, and a guard against the solver crying wolf.

    Plywood's low G/E puts first bending above first torsion; binary
    bending-torsion flutter is then not the mechanism and inf is the honest
    answer, not a failure to converge.
    """
    from dynamics.flutter_analysis import (_fin_section_properties,
                                           _cantilever_bending_freq,
                                           _torsion_fundamental_freq)
    span, root, tip, thick, E, G, dens = 0.12, 0.18, 0.09, 0.003, 8e9, 0.7e9, 680.0
    p = _fin_section_properties(root, tip, thick, span, dens)
    f_b = _cantilever_bending_freq(E, p["I_bend"], p["m_bar"], span)
    f_t = _torsion_fundamental_freq(G, p["J"], p["I_alpha"], span)
    assert f_b > f_t, "test case no longer has bending above torsion"

    r = _pk(span=span, root_chord=root, tip_chord=tip, thickness=thick,
            E=E, G=G, density_mat=dens)
    assert not math.isfinite(r["flutter_speed_mps"])
    assert r["envelope_status"] == "NOT_REACHED"
