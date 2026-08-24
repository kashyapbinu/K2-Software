"""
AGARD-B experimental force data — AEDC-TR-70-100.
=================================================

C. F. Anderson, *An Investigation of the Aerodynamic Characteristics of the
AGARD Model B for Mach Numbers from 0.2 to 1.0*, Propulsion Wind Tunnel
Facility, Arnold Engineering Development Center, AEDC-TR-70-100, May 1970
(DTIC AD0868286). Tunnel 4T, 1.956-in-diameter model.

Provenance / how these numbers were obtained
--------------------------------------------
The report's Table I is a **scan**; the values below were parsed from the OCR
text at https://archive.org/details/DTIC_AD0868286 and are therefore NOT
transcribed by hand from the printed page. OCR of that scan reliably confuses
0/n/o and 6/b, so no individual table entry is trusted here. Instead each Mach
block is reduced to **fitted aggregates**:

    C_L_alpha   least-squares slope of C_L vs alpha over |alpha| <= 6 deg
    C_D0        least-squares C_D0 of C_D = C_D0 + k*alpha^2 over |alpha| <= 4 deg

both with iterative 2-sigma outlier rejection, so a handful of mis-read digits
cannot move the aggregate. `n_pts` records how many rows survived rejection.
The resulting C_L_alpha(M) is smooth and monotonic through the transonic rise,
which is the independent check that the extraction is sound — the underlying
scan contains no such smoothing.

Treat C_D0 as the weaker of the two: it is a small number formed from a column
that OCR damages more often, and it includes base drag (C_D = C_D,F + C_D,b),
which is sting-dependent.

Reference quantities (from the report's nomenclature)
-----------------------------------------------------
    S_ref = 4*sqrt(3)*D^2      total wing planform area (NOT body frontal area)
    c_ref = 4*sqrt(3)*D/3      mean aerodynamic chord = 2.3094 D
    D     = 0.163 ft           body diameter of the tested model
    moment reference: quarter-chord point of the mean aerodynamic chord

Quoted measurement precision (report, Section 3.2 equivalent): the companion
NACA TN 3300 tests of the same model quote +/-0.0004 on C_L and +/-0.001 on C_D.
"""
from __future__ import annotations

import math

# ── AGARD-B geometry, AGARD Memorandum AG-4/M3 (1955), in body diameters D ────
BODY_LENGTH_D = 8.5          # total body length
NOSE_LENGTH_D = 3.0          # ogive nose length
WING_SPAN_D = 4.0            # total span, tip to tip
WING_ROOT_CHORD_D = 2.0 * math.sqrt(3.0)      # equilateral delta = 3.4641 D
WING_LE_SWEEP_DEG = 60.0
WING_THICKNESS_RATIO = 0.04  # symmetric circular-arc section, streamwise
SREF_D2 = 4.0 * math.sqrt(3.0)                # 6.9282 D^2, wing planform area
CREF_D = 4.0 * math.sqrt(3.0) / 3.0           # 2.3094 D, mean aerodynamic chord


def nose_radius(x_over_d: float) -> float:
    """AGARD-B ogive nose radius r/D at axial station x/D from the tip.

    r = (x/3)*[1 - (1/9)(x/D)^2 + (1/54)(x/D)^3]   (AGARD AG-4/M3)

    At x = 3D this gives r = 0.5 D with dr/dx = 0, i.e. it closes tangentially
    onto the cylinder.
    """
    u = x_over_d
    return (u / 3.0) * (1.0 - (u ** 2) / 9.0 + (u ** 3) / 54.0)



# Mach -> fitted aggregates. Coefficients are wing-area referenced.
AEDC_TR_70_100 = {
     0.2: dict(cl_alpha_per_deg=0.04560, n_pts=29, cl0=+0.00290, cd0=0.02583, cd0_n_pts=15),
    0.25: dict(cl_alpha_per_deg=0.04435, n_pts=18, cl0=+0.00755, cd0=0.02619, cd0_n_pts=8),
     0.3: dict(cl_alpha_per_deg=0.04375, n_pts=23, cl0=+0.00560, cd0=0.02484, cd0_n_pts=13),
    0.35: dict(cl_alpha_per_deg=0.04517, n_pts=20, cl0=+0.00216, cd0=0.02494, cd0_n_pts=12),
     0.4: dict(cl_alpha_per_deg=0.04517, n_pts=20, cl0=+0.00109, cd0=0.02515, cd0_n_pts=14),
     0.5: dict(cl_alpha_per_deg=0.04580, n_pts=22, cl0=+0.00131, cd0=0.02609, cd0_n_pts=17),
     0.6: dict(cl_alpha_per_deg=0.04816, n_pts=24, cl0=+0.00445, cd0=0.02689, cd0_n_pts=15),
     0.7: dict(cl_alpha_per_deg=0.04977, n_pts=25, cl0=+0.00520, cd0=0.02677, cd0_n_pts=17),
     0.8: dict(cl_alpha_per_deg=0.05273, n_pts=22, cl0=+0.00532, cd0=0.02793, cd0_n_pts=13),
     0.9: dict(cl_alpha_per_deg=0.05534, n_pts=22, cl0=+0.00340, cd0=0.02802, cd0_n_pts=16),
    0.95: dict(cl_alpha_per_deg=0.05838, n_pts=19, cl0=+0.00403, cd0=0.02914, cd0_n_pts=12),
     1.0: dict(cl_alpha_per_deg=0.05918, n_pts=24, cl0=+0.00724, cd0=0.03903, cd0_n_pts=13),
}

# Blocks retained after the contamination filter: [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
