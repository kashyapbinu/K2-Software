"""
AGARD-B calibration model as a K2 RocketAssembly.
=================================================

Geometry per AGARD Memorandum AG-4/M3 (1955), in body diameters D:

    body        8.5 D long: a 3 D ogive nose (see ``agardb_aedc.nose_radius``)
                followed by a 5.5 D cylinder
    wing        delta in the form of an equilateral triangle, span 4 D, root
                chord 2*sqrt(3) D = 3.4641 D, leading-edge sweep 60 deg,
                trailing edge flush with the body base
    section     symmetric circular arc, t/c = 0.04 streamwise

K2 models the wing as a two-panel ``TrapezoidalFinSet``. The panel dimensions
are the *exposed* ones: the delta's chord shrinks linearly outboard, so at the
body surface (y = 0.5 D) the chord is already down to 0.75 * root, and the
exposed semi-span is (4 D - 1 D)/2 = 1.5 D.

Known idealisations, which the benchmark reports rather than hides:

  * K2's fin cross-sections are flat/rounded/wedge, not a circular arc, so the
    4% section is approximated by a constant-thickness panel of the same t/c at
    the exposed root.
  * The real model has 0.002 D rounded leading and trailing edges.
  * The sting and its windshield are not modelled; the tunnel data is corrected
    to a zero-base-drag or measured-base-pressure condition, which is why the
    lift-curve slope — not the axial force — is the primary comparison.
"""
from __future__ import annotations

import math

from validation.data.agardb_aedc import (
    BODY_LENGTH_D, NOSE_LENGTH_D, WING_SPAN_D, WING_ROOT_CHORD_D,
    WING_THICKNESS_RATIO, SREF_D2, CREF_D,
)

# The AEDC test used a 1.956-in-diameter model; K2's mesher and the SU2 case are
# happier at a metre-ish scale, and every coefficient here is non-dimensional.
DEFAULT_DIAMETER_M = 0.100

# Fraction of the exposed semi-span trimmed off the delta tip so the fin solid
# has a finite tip chord. See the comment in :func:`agardb_assembly`.
TIP_TRUNCATION = 0.01


def exposed_root_chord_d() -> float:
    """Wing chord where it meets the body, in D.

    The equilateral delta's local chord is c(y) = c_root * (1 - y/(b/2)); at the
    body surface y = 0.5 D with b/2 = 2 D that leaves 75% of the root chord.
    """
    return WING_ROOT_CHORD_D * (1.0 - 0.5 / (WING_SPAN_D / 2.0))


def exposed_semi_span_d() -> float:
    """Exposed panel height, in D: (span - body diameter) / 2."""
    return (WING_SPAN_D - 1.0) / 2.0


def reference_quantities(diameter_m: float = DEFAULT_DIAMETER_M) -> dict:
    """Reference area/length the AEDC coefficients are normalised by.

    Note this is the **wing planform area**, not the body frontal area K2's own
    aerodynamics uses — converting between the two is the single most common way
    to get an apparently 10x wrong answer out of this comparison.
    """
    d = diameter_m
    return {
        "s_ref_m2": SREF_D2 * d ** 2,
        "c_ref_m": CREF_D * d,
        "body_area_m2": math.pi * (d / 2.0) ** 2,
        "diameter_m": d,
        "length_m": BODY_LENGTH_D * d,
    }


def agardb_assembly(diameter_m: float = DEFAULT_DIAMETER_M):
    """Build the AGARD-B model as a RocketAssembly."""
    from core.components import RocketAssembly, BodyTube, NoseCone, TrapezoidalFinSet

    d = diameter_m
    asm = RocketAssembly()
    asm.name = "AGARD-B"

    nose = NoseCone()
    nose.name = "AGARD-B ogive"
    nose.shape = "AGARD-B"
    nose.length = NOSE_LENGTH_D * d
    nose.diameter = d
    nose.wall_thickness = 0.002 * d
    nose.material = "Aluminum 6061-T6"
    asm.add_component(asm.stages[0], nose)

    tube = BodyTube()
    tube.name = "AGARD-B cylinder"
    tube.length = (BODY_LENGTH_D - NOSE_LENGTH_D) * d
    tube.outer_diameter_val = d
    tube.inner_diameter = d - 4.0 * 0.002 * d
    tube.material = "Aluminum 6061-T6"
    asm.add_component(asm.stages[0], tube)

    root = exposed_root_chord_d() * d
    fins = TrapezoidalFinSet()
    fins.name = "AGARD-B delta wing"
    fins.fin_count = 2                     # a wing, i.e. one panel per side
    fins.root_chord = root
    # Truncate the outermost 1% of the span rather than running a mathematically
    # sharp tip. The tip chord and height below stay exactly on the true delta
    # planform line, so the removed area is 0.01% of the panel — but a real tip
    # chord also keeps this case off the degenerate-edge path in the mesher.
    #
    # That path is why the truncation exists at all: a tip_chord of exactly 0
    # used to make the OCC fin builder fail on a zero-length edge and fall back
    # to a BOX, silently meshing a rectangular wing and inflating the lift-curve
    # slope by ~25%. ``cfd.meshing._add_fins`` now builds the triangle properly
    # (see tests/validation/test_cfd_delta_fin.py), so this is belt-and-braces.
    fins.tip_chord = TIP_TRUNCATION * root
    fins.height = (1.0 - TIP_TRUNCATION) * exposed_semi_span_d() * d
    # Leading-edge sweep of the equilateral delta, measured from the body normal.
    fins.sweep_angle = 60.0
    fins.thickness = WING_THICKNESS_RATIO * root
    fins.cross_section = "Airfoil"
    fins.material = "Aluminum 6061-T6"
    # A fin set added to a body tube auto-positions with its trailing edge flush
    # with the tube's aft end, which is where the AGARD-B wing sits.
    asm.add_component(tube, fins)
    return asm
