"""
Canonical validation rocket — one geometry shared by sim, CFD and structures.
=============================================================================

A realistic single-stage 4-inch high-power rocket on a 54 mm **L1090W** motor
(1090 N avg, 2671 N·s, 2.45 s burn — a real ThrustCurve entry). Using one fixed
vehicle everywhere means the sim/CFD/structures benchmarks all describe the same
physical object, so cross-domain results are directly comparable and the report
tells a single coherent story.

The numbers here are intentionally frozen constants, not pulled live from the UI
or ``motors.json``, so a benchmark run is reproducible regardless of catalog
edits. ``motors.json`` was only used once to pick credible motor values.
"""
from __future__ import annotations

from core.rocket_state import RocketState

WALL_T = 0.0025          # airframe wall thickness (m)
STRUCT_MATERIAL = "Aluminum 6061-T6"


# ── Motor (real ThrustCurve L1090W, 54 mm) ────────────────────────────────────
MOTOR = dict(
    designation="L1090W",
    avg_thrust=1090.0,        # N
    max_thrust=1430.0,        # N (typical regressive-ish peak)
    total_impulse=2671.0,     # N·s
    burn_time=2.45,           # s
    propellant_mass=1.40,     # kg
)


def canonical_state() -> RocketState:
    """Return a fresh RocketState for the canonical validation rocket.

    Fresh each call so a benchmark can mutate it (launch angle, integrator, dt)
    without leaking into the next run.
    """
    return RocketState(
        name="K2-Validation-Canonical",
        # ── Geometry (4" airframe, 2 m long) ──
        length=2.0,
        diameter=0.102,
        nose_length=0.40,
        nose_type="ogive",
        fin_count=4,
        fin_span=0.12,
        fin_height=0.12,
        fin_root_chord=0.20,
        fin_tip_chord=0.08,
        fin_sweep_angle=30.0,
        fin_thickness=0.004,
        fin_position=1.70,        # nose tip → fin root LE
        surface_finish="Normal",
        fin_cross_section="Rounded",
        # ── Mass ──
        dry_mass=6.0,
        propellant_mass=MOTOR["propellant_mass"],
        propellant_mass_initial=MOTOR["propellant_mass"],
        # ── Structures (airframe wall) ──
        wall_thickness=0.0025,
        material_name="Aluminum 6061-T6",
        yield_strength=276e6,
        elastic_modulus=68.9e9,
        material_density=2700.0,
        # ── Motor ──
        motor_designation=MOTOR["designation"],
        motor_avg_thrust=MOTOR["avg_thrust"],
        motor_max_thrust=MOTOR["max_thrust"],
        motor_total_impulse=MOTOR["total_impulse"],
        motor_burn_time=MOTOR["burn_time"],
        # ── Flight / recovery ──
        launch_angle=90.0,
        main_deploy_altitude=300.0,
        drogue_cd_area=0.5,
        main_cd_area=3.0,
        # ── Integration ──
        sim_dt=0.01,
        sim_speed=1.0,
        integrator_name="rk4",
    )


def canonical_assembly():
    """Build a RocketAssembly matching the canonical rocket, for FEM/CFD meshing.

    Aluminium nose + body tube (0.102 m OD, 2.5 mm wall) totalling 2.0 m — the
    same airframe the closed-form structures formulas describe, so the modal
    closed-form-vs-CalculiX comparison is on one physical object.
    """
    from core.components import RocketAssembly, NoseCone, BodyTube

    asm = RocketAssembly()
    asm.name = "K2-Validation-Canonical"
    stage = asm.stages[0]

    nose = NoseCone()
    nose.shape = "Ogive"
    nose.length = 0.40
    nose.diameter = 0.102
    nose.wall_thickness = WALL_T
    nose.material = STRUCT_MATERIAL
    asm.add_component(stage, nose)

    tube = BodyTube()
    tube.length = 1.60
    tube.outer_diameter_val = 0.102
    tube.inner_diameter = 0.102 - 2 * WALL_T
    tube.material = STRUCT_MATERIAL
    asm.add_component(stage, tube)

    # Fins matching canonical_state() so the SU2 mesh and the Barrowman AeroModel
    # describe the *same* rocket (extract_cfd_geometry fabricates fins otherwise).
    from core.components import TrapezoidalFinSet
    fins = TrapezoidalFinSet()
    fins.fin_count = 4
    fins.root_chord = 0.20
    fins.tip_chord = 0.08
    fins.height = 0.12
    fins.sweep_angle = 30.0
    fins.thickness = 0.004
    asm.add_component(tube, fins)

    return asm
