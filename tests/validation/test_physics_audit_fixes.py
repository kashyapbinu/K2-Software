"""
Regression gates for the 2026-09-12 physics audit.

Each test pins a bug that shipped and was measured, so the specific wrong
formula cannot come back:

1. Wind-bearing isotropy — the relative wind was resolved with
   ``atan2(vrel_z, hypot(vrel_x, vrel_y))``, which discards the wind's compass
   bearing.
2. Nozzle area ratio — Sutton eq. 3-25 was written ``t1/(t2·t3)`` instead of
   ``1/(t1·t2·t3)`` in three places, scaling ε by ((γ+1)/2)^(2/(γ-1)).
3. Sutton-Graves stagnation heating — the V³ correlation constant multiplied
   an enthalpy difference, which is dimensionally Pa rather than W/m².
4. Engine parity — ``core.batch_simulation`` is a hand-copy of
   ``SimulationEngine._derivatives`` that had drifted out of sync.
"""
import math

import pytest

from core.flight_dynamics import resolve_aero_frame


# ── 1. Wind-bearing isotropy ────────────────────────────────────────────────

_BEARINGS = [0, 15, 30, 45, 60, 90, 135, 180, 225, 270, 315]


def _frame_at(bearing_deg, v_up=100.0, v_cross=17.5):
    """Vertical rocket, steady climb, relative wind from *bearing_deg*."""
    r = math.radians(bearing_deg)
    vrel = (v_cross * math.cos(r), v_cross * math.sin(r), v_up)
    return resolve_aero_frame(vrel, math.hypot(v_cross, v_up),
                              math.pi / 2, 0.0)


@pytest.mark.sim
@pytest.mark.parametrize("bearing", _BEARINGS)
def test_total_incidence_is_independent_of_wind_bearing(bearing):
    """Rotating the wind about the vertical axis is a rotation of the whole
    problem: total angle of attack cannot change."""
    ref = _frame_at(0).alpha_total
    assert _frame_at(bearing).alpha_total == pytest.approx(ref, rel=1e-12)


@pytest.mark.sim
def test_pure_crosswind_is_pure_sideslip():
    """A crosswind square to the body must produce zero pitch-plane AoA.

    It used to produce nearly full AoA on the global X axis, so pitch angular
    acceleration came out identical for wind from 0, 90 and 180 degrees.
    """
    assert _frame_at(90).alpha == pytest.approx(0.0, abs=1e-12)
    assert abs(_frame_at(90).beta) > math.radians(5)
    # ...and the mirror case: a headwind produces zero sideslip.
    assert _frame_at(0).beta == pytest.approx(0.0, abs=1e-12)
    assert abs(_frame_at(0).alpha) > math.radians(5)


@pytest.mark.sim
def test_normal_force_direction_reverses_with_the_flow():
    """Wind from the north and wind from the south must push opposite ways.

    ``hypot(vrel_x, vrel_y)`` threw away the sign of vrel_x, so the X-axis
    normal force could never change sign — wind from the south weathercocked
    the rocket south.
    """
    n0 = _frame_at(0).normal_dir
    n180 = _frame_at(180).normal_dir
    assert n0[0] * n180[0] < 0.0
    assert n0[0] == pytest.approx(-n180[0], rel=1e-9)


@pytest.mark.sim
@pytest.mark.parametrize("bearing", _BEARINGS)
def test_normal_direction_is_a_unit_vector_perpendicular_to_the_flow(bearing):
    fr = _frame_at(bearing)
    n = fr.normal_dir
    assert math.sqrt(sum(c * c for c in n)) == pytest.approx(1.0, rel=1e-12)
    r = math.radians(bearing)
    vrel = (17.5 * math.cos(r), 17.5 * math.sin(r), 100.0)
    dot = sum(a * b for a, b in zip(n, vrel)) / math.hypot(17.5, 100.0)
    assert dot == pytest.approx(0.0, abs=1e-9)


# ── 2. Nozzle area ratio ────────────────────────────────────────────────────

def _eps_from_mach(M, g):
    """Isentropic area-Mach relation — independent of the pressure-ratio form."""
    return (1.0 / M) * ((2.0 / (g + 1)) * (1 + (g - 1) / 2 * M * M)) \
        ** ((g + 1) / (2 * (g - 1)))


def _pe_pc_from_mach(M, g):
    return (1 + (g - 1) / 2 * M * M) ** (-g / (g - 1))


@pytest.mark.sim
@pytest.mark.parametrize("gamma", [1.15, 1.20, 1.25])
@pytest.mark.parametrize("mach", [2.0, 2.5, 3.0, 3.5])
def test_area_ratio_matches_the_isentropic_area_mach_relation(gamma, mach):
    """ε(Pe/Pc) must agree with ε(M) evaluated at the same exit state."""
    from physics.internal_ballistics import area_ratio_for_pressure_ratio

    eps_ref = _eps_from_mach(mach, gamma)
    eps = area_ratio_for_pressure_ratio(_pe_pc_from_mach(mach, gamma), gamma)
    assert eps == pytest.approx(eps_ref, rel=1e-9)


@pytest.mark.sim
@pytest.mark.parametrize("gamma", [1.15, 1.20, 1.25])
@pytest.mark.parametrize("eps", [4.0, 10.0, 25.0])
def test_exit_pressure_ratio_inverts_the_area_ratio(gamma, eps):
    """Round-trip: solving for Pe/Pc then recomputing ε must return ε."""
    from physics.internal_ballistics import (area_ratio_for_pressure_ratio,
                                             exit_pressure_ratio)

    pe_pc = exit_pressure_ratio(eps, gamma)
    assert area_ratio_for_pressure_ratio(pe_pc, gamma) == pytest.approx(eps, rel=1e-6)
    # Must land on the SUPERSONIC branch, below the throat pressure ratio.
    assert pe_pc < (2.0 / (gamma + 1)) ** (gamma / (gamma - 1))


@pytest.mark.sim
@pytest.mark.parametrize("gamma", [1.15, 1.20, 1.25])
def test_solid_and_liquid_nozzle_models_agree(gamma):
    """The two propulsion modules solve the same nozzle physics; they used to
    disagree by exactly ((γ+1)/2)^(2/(γ-1))."""
    from physics.internal_ballistics import optimum_expansion_ratio
    from physics.liquid_propulsion import expansion_ratio_for_exit_pressure

    pc, pa = 70e5, 101325.0
    assert optimum_expansion_ratio(pa, pc, gamma) == pytest.approx(
        expansion_ratio_for_exit_pressure(pa, pc, gamma), rel=1e-6)


# ── 3. Sutton-Graves stagnation heating ─────────────────────────────────────

@pytest.mark.structures
@pytest.mark.parametrize("V", [600.0, 1500.0, 2400.0])
def test_stagnation_heating_matches_the_v_cubed_correlation(V):
    """In the cold-wall limit the result must be K·√(ρ/R_n)·V³ exactly.

    The shipped form multiplied K by an enthalpy difference instead of V³,
    under-predicting by ~4000x and returning exactly 0 below about Mach 2.5.
    """
    from structures.thermal_analysis import (stagnation_heat_flux,
                                             stagnation_heat_flux_cold_wall)

    rho, R_n = 0.018, 0.02
    expected = 1.7415e-4 * math.sqrt(rho / R_n) * V ** 3
    assert stagnation_heat_flux_cold_wall(rho, V, R_n) == pytest.approx(expected, rel=1e-12)
    # Hot wall only ever reduces the flux, and never below zero.
    hot = stagnation_heat_flux(226.0, rho, V, R_n, 500.0)
    assert 0.0 <= hot <= expected


@pytest.mark.structures
def test_stagnation_heating_is_in_watts_per_square_metre():
    """Dimensional sanity: a 20 mm nose at Mach 5 sits in the hundreds of
    kW/m², not the hundreds of W/m²."""
    from structures.thermal_analysis import stagnation_heat_flux

    q = stagnation_heat_flux(226.0, 0.018, 1509.0, 0.02, 300.0)
    assert 1e5 < q < 1e7


# ── 4. Engine parity ────────────────────────────────────────────────────────

@pytest.mark.sim
def test_both_integrators_share_one_aero_frame():
    """Both 6DOF integrators must import the shared frame resolver rather than
    hand-rolling the relative-wind maths, so a fix to one reaches the other."""
    import core.batch_simulation as batch
    import core.simulation_engine as engine

    assert batch.resolve_aero_frame is resolve_aero_frame
    assert engine.resolve_aero_frame is resolve_aero_frame


# ── 5. Yaw kinematic metric factor ──────────────────────────────────────────

@pytest.mark.sim
def test_yaw_euler_rate_carries_the_inverse_cosine_factor():
    """dψ/dt = r / cos(θ). Returning r directly (what both integrators did)
    made the yaw weathercock fade out near vertical, where rockets fly."""
    from core.flight_dynamics import yaw_euler_rate

    for pitch_deg in (0.0, 30.0, 45.0, 60.0):
        p = math.radians(pitch_deg)
        assert yaw_euler_rate(1.0, p) == pytest.approx(1.0 / math.cos(p), rel=1e-12)


@pytest.mark.sim
@pytest.mark.parametrize("pitch_deg", [89.0, 89.9, 90.0, 90.1, 91.0, -90.0])
def test_yaw_euler_rate_stays_finite_at_the_pole(pitch_deg):
    """cos(θ) → 0 at vertical — the nominal launch attitude — so the relation
    must be floored rather than allowed to produce inf/NaN."""
    from core.flight_dynamics import yaw_euler_rate

    r = yaw_euler_rate(0.5, math.radians(pitch_deg))
    assert math.isfinite(r)


@pytest.mark.sim
def test_nose_angular_rate_is_recovered_from_the_euler_rate():
    """The physical rate the nose swings is dψ/dt·cos(θ), which must return the
    body rate r for any pitch away from the floored region."""
    from core.flight_dynamics import yaw_euler_rate

    for pitch_deg in (10.0, 45.0, 80.0, 89.0):
        p = math.radians(pitch_deg)
        assert yaw_euler_rate(0.7, p) * math.cos(p) == pytest.approx(0.7, rel=1e-9)


@pytest.mark.sim
@pytest.mark.parametrize("angle", [0.0, 3.0, -3.0, 7.5, -7.5, 100.0])
def test_wrap_angle_is_bounded_and_preserves_direction(angle):
    from core.flight_dynamics import wrap_angle

    w = wrap_angle(angle)
    assert -math.pi < w <= math.pi + 1e-12
    assert math.cos(w) == pytest.approx(math.cos(angle), abs=1e-9)
    assert math.sin(w) == pytest.approx(math.sin(angle), abs=1e-9)


@pytest.mark.sim
@pytest.mark.parametrize("bearing", [0, 45, 90, 135, 180, 225, 270, 315])
def test_apogee_does_not_depend_on_wind_bearing(bearing):
    """End-to-end isotropy gate. A vertical launch of an axisymmetric rocket
    cannot care which compass bearing a uniform wind comes from — rotating the
    wind about the vertical axis rotates the whole problem.

    This shipped at a 5.2% apogee spread (39% in landing range). The tolerance
    below is set by what the decoupled-Euler moment split can deliver, which is
    about 1%.
    """
    import core.batch_simulation as bs
    from core.batch_simulation import BatchSimConfig, run_batch_simulation

    def make(wdir):
        return BatchSimConfig(
            length=1.2, diameter=0.054, nose_length=0.25,
            fin_height=0.06, fin_span=0.06, fin_root_chord=0.12,
            fin_tip_chord=0.06, fin_count=4, dry_mass=1.2, propellant_mass=0.25,
            cg=0.70, dry_cg=0.70, cp=0.95, motor_position=1.05,
            motor_length=0.20, motor_designation="H128",
            motor_avg_thrust=128.0, motor_max_thrust=170.0,
            motor_total_impulse=250.0, motor_burn_time=1.95, motor_isp=180.0,
            launch_angle=90.0, wind_speed=8.0, wind_direction=wdir,
            wind_gust_intensity=0.0, main_deploy_altitude=200.0,
            drogue_cd_area=0.2, main_cd_area=1.2,
            sim_dt=0.01, integrator_name="rk4")

    saved = bs._GUST_AMP_COEFF
    bs._GUST_AMP_COEFF = 0.0          # the gust is a deliberate perturbation
    try:
        ref = run_batch_simulation(make(0), seed=7)
        got = run_batch_simulation(make(bearing), seed=7)
    finally:
        bs._GUST_AMP_COEFF = saved

    assert got.apogee == pytest.approx(ref.apogee, rel=0.02)
    # ...and the vehicle must still fly upwind, rotated to match the bearing.
    r = math.radians(bearing)
    assert got.landing_x * math.cos(r) + got.landing_y * math.sin(r) < 0.0


# ── 7. Supersonic fin coefficients ──────────────────────────────────────────

_FIN_KW = dict(fin_count=4, fin_span=0.09, fin_root_chord=0.16,
               fin_tip_chord=0.07, body_radius=0.038)


@pytest.mark.sim
@pytest.mark.parametrize("mach", [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
def test_busemann_k1_matches_two_over_beta_across_the_table(mach):
    """K1 is the Ackeret slope 2/β. It used to freeze at M4.9 — the table was
    built by ``range(int((5.0 - 1.5) * 10))``, one step short of its own stated
    limit — so fin lift stopped decaying: 1.23x high at M6, 2.5x at M12.
    """
    from physics.drag_tables import FIN_K1

    assert FIN_K1.get_value(mach) == pytest.approx(
        2.0 / math.sqrt(mach * mach - 1.0), rel=1e-3)


@pytest.mark.sim
def test_fin_table_endpoint_is_inclusive():
    """The top of the tabulated range must actually be tabulated, not one
    step short of it."""
    from physics.drag_tables import FIN_K1, FIN_K_TABLE_MAX_MACH

    top = FIN_K_TABLE_MAX_MACH
    assert FIN_K1.get_value(top) == pytest.approx(
        2.0 / math.sqrt(top * top - 1.0), rel=1e-6)


@pytest.mark.sim
def test_out_of_range_mach_is_reported_not_silent():
    """Past the table the interpolator clamps, which over-predicts fin
    authority and therefore stability margin. That must be detectable."""
    from physics.drag_tables import fin_k_out_of_range, FIN_K_TABLE_MAX_MACH

    assert not fin_k_out_of_range(FIN_K_TABLE_MAX_MACH)
    assert not fin_k_out_of_range(4.9)
    assert fin_k_out_of_range(FIN_K_TABLE_MAX_MACH + 0.1)


@pytest.mark.sim
def test_fin_normal_force_decays_monotonically_through_supersonic():
    """Physical trend: fin CN_alpha must fall with Mach across the whole
    supersonic range, with no plateau from a table running out."""
    from physics.aerodynamics import compute_fin_cn_alpha

    a = math.radians(2.0)
    machs = [1.6 + 0.2 * i for i in range(43)]      # 1.6 → 10.0
    vals = [compute_fin_cn_alpha(mach=m, alpha=a, **_FIN_KW) for m in machs]
    for m, lo, hi in zip(machs[1:], vals[:-1], vals[1:]):
        assert hi < lo, f"fin CN_alpha stopped decaying at M={m:.1f}"


@pytest.mark.sim
@pytest.mark.parametrize("edge", [0.9, 1.5])
def test_cn_alpha_is_continuous_across_the_regime_joins(edge):
    """The subsonic / transonic-quartic / Busemann branches must join without a
    step — a jump here would put a discontinuity in the flight derivative."""
    from physics.aerodynamics import compute_fin_cn_alpha

    a = math.radians(2.0)
    below = compute_fin_cn_alpha(mach=edge - 1e-4, alpha=a, **_FIN_KW)
    above = compute_fin_cn_alpha(mach=edge + 1e-4, alpha=a, **_FIN_KW)
    assert below == pytest.approx(above, rel=2e-3)


# ── 6. Wind model ───────────────────────────────────────────────────────────

@pytest.mark.sim
def test_zero_gust_means_zero_turbulence():
    """An explicit 0 must mean calm air.

    ``turbulence_intensity`` defaulted to 0.1 and was only overridden ``if
    gust_intensity > 0``, so 0 fell through to 10%. The UI spinbox defaults to
    0%, making that the default experience, and it was non-monotonic: 0% came
    out noisier than 5%.
    """
    from environment.wind_model import WindModel

    assert WindModel(8.0, 0.0, 0.0).turbulence_intensity == 0.0
    assert WindModel(8.0, 0.0, gust_intensity=0.0).turbulence_intensity == 0.0
    assert WindModel(8.0, 0.0, turbulence_intensity=0.0).turbulence_intensity == 0.0
    # A real request still comes through, and the explicit keyword wins.
    assert WindModel(8.0, 0.0, 0.15).turbulence_intensity == pytest.approx(0.15)
    assert WindModel(8.0, 0.0, 0.15,
                     turbulence_intensity=0.02).turbulence_intensity == pytest.approx(0.02)
    # Neither specified -> the documented default still applies.
    assert WindModel(8.0, 0.0).turbulence_intensity == pytest.approx(
        WindModel._DEFAULT_TURBULENCE)


@pytest.mark.sim
def test_zero_turbulence_is_deterministic():
    """With no turbulence the wind must be a pure function of altitude."""
    from environment.wind_model import WindModel

    a = WindModel(10.0, 0.0, 0.0, seed=1)
    b = WindModel(10.0, 0.0, 0.0, seed=999)
    for t in (0.0, 1.0, 5.0, 30.0):
        assert a.get_wind_velocity(250.0, t) == b.get_wind_velocity(250.0, t)


@pytest.mark.sim
def test_turbulence_is_monotonic_in_intensity():
    """More requested turbulence must mean more scatter, at every step."""
    import statistics
    from environment.wind_model import WindModel

    spreads = []
    for ti in (0.0, 0.05, 0.15, 0.30):
        w = WindModel(10.0, 0.0, ti, seed=4)
        sp = [math.hypot(*w.get_wind_velocity(200.0, i * 0.02)[:2])
              for i in range(4000)]
        spreads.append(statistics.pstdev(sp))
    assert spreads[0] == pytest.approx(0.0, abs=1e-12)
    assert spreads[0] < spreads[1] < spreads[2] < spreads[3]


@pytest.mark.sim
def test_wind_profile_is_continuous_at_the_ground():
    """No step in the wind at z = 0.

    A hard zero for ``altitude <= 0`` combined with a 0.1 m floor on the power
    law put a 52%-of-reference jump at exactly z = 0, which RK4 sub-steps
    straddle at lift-off and touchdown.
    """
    from environment.wind_model import WindModel

    w = WindModel(10.0, 0.0, 0.0)
    prev = 0.0
    for z in (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0):
        v = math.hypot(*w.get_wind_velocity(z, 1.0)[:2])
        assert v >= prev - 1e-12, f"profile not monotonic at z={z}"
        prev = v
    assert math.hypot(*w.get_wind_velocity(0.0, 1.0)[:2]) == 0.0
    # Continuity: approaching the ground the speed must tend to zero, with no
    # jump left behind by a floor.
    assert math.hypot(*w.get_wind_velocity(1e-9, 1.0)[:2]) < 0.5
    assert math.hypot(*w.get_wind_velocity(10.0, 1.0)[:2]) == pytest.approx(10.0)


@pytest.mark.sim
def test_wind_power_law_is_one_seventh_referenced_to_ten_metres():
    from environment.wind_model import WindModel

    w = WindModel(10.0, 0.0, 0.0)
    v10 = math.hypot(*w.get_wind_velocity(10.0, 1.0)[:2])
    v100 = math.hypot(*w.get_wind_velocity(100.0, 1.0)[:2])
    assert v10 == pytest.approx(10.0)
    assert v100 / v10 == pytest.approx(10.0 ** 0.143, rel=1e-9)


@pytest.mark.sim
@pytest.mark.parametrize("bearing", [0.0, 90.0, 180.0, 270.0])
def test_wind_direction_is_the_bearing_it_blows_from(bearing):
    """Must match the landing estimator, which drifts toward ``dir + 180``."""
    from environment.wind_model import WindModel

    w = WindModel(10.0, bearing, 0.0)
    vx, vy, _ = w.get_wind_velocity(100.0, 1.0)
    blow_to = math.radians(bearing + 180.0)
    mag = math.hypot(vx, vy)
    assert vx / mag == pytest.approx(math.cos(blow_to), abs=1e-9)
    assert vy / mag == pytest.approx(math.sin(blow_to), abs=1e-9)


@pytest.mark.sim
def test_batch_uses_the_rail_length_not_the_body_length():
    """The batch config must carry the launch rod length; it used to treat the
    rocket's own length as the rail."""
    from core.batch_simulation import BatchSimConfig

    cfg = BatchSimConfig.from_rocket_state(
        type("S", (), {"length": 2.0, "diameter": 0.1, "dry_mass": 5.0,
                       "cg": 1.0, "cp": 1.4, "launch_rod_length": 3.5})()
    )
    assert cfg.launch_rod_length == pytest.approx(3.5)
