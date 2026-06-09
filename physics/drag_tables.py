"""
K2 Aerospace — Empirical Drag Tables
======================================
Ported from OpenRocket's BarrowmanCalculator and SymmetricComponentCalc.

Data Sources:
    - NASA TR-R-100: "Collection of Zero-Lift Drag Data on Bodies of
      Revolution from Free-Flight Investigations" (page 16)
    - Barrowman, J.S. (1967) thesis
    - OpenRocket source (GPL, Sampo Niskanen)

All tables are for fineness-ratio 3 and are extrapolated for other ratios.
"""

import math
import bisect


class LinearInterpolator:
    """
    Piecewise-linear interpolation table.
    Values outside the range are clamped to the boundary values.
    """

    def __init__(self, x_points=None, y_points=None):
        self._x = list(x_points) if x_points else []
        self._y = list(y_points) if y_points else []

    def add_point(self, x: float, y: float):
        idx = bisect.bisect_left(self._x, x)
        # Replace if duplicate x
        if idx < len(self._x) and abs(self._x[idx] - x) < 1e-12:
            self._y[idx] = y
        else:
            self._x.insert(idx, x)
            self._y.insert(idx, y)

    def get_value(self, x: float) -> float:
        if not self._x:
            return 0.0
        if x <= self._x[0]:
            return self._y[0]
        if x >= self._x[-1]:
            return self._y[-1]
        idx = bisect.bisect_right(self._x, x) - 1
        x0, x1 = self._x[idx], self._x[idx + 1]
        y0, y1 = self._y[idx], self._y[idx + 1]
        t = (x - x0) / (x1 - x0) if (x1 - x0) > 1e-15 else 0.0
        return y0 + t * (y1 - y0)

    @property
    def x_points(self):
        return self._x


# ── Stagnation CD ──────────────────────────────────────────────────────────────

def stagnation_cd(mach: float) -> float:
    """
    Stagnation pressure coefficient (OpenRocket BarrowmanCalculator).
    """
    if mach <= 1.0:
        pressure = 1.0 + mach**2 / 4.0 + mach**4 / 40.0
    else:
        pressure = (1.84 - 0.76 / mach**2 + 0.166 / mach**4
                    + 0.035 / mach**6)
    return 0.85 * pressure


# ── Base CD ────────────────────────────────────────────────────────────────────

def base_cd(mach: float) -> float:
    """
    Base drag coefficient (OpenRocket BarrowmanCalculator).
    """
    if mach <= 1.0:
        return 0.12 + 0.13 * mach**2
    else:
        return 0.25 / mach


# ── Nose Cone Pressure Drag Tables (NASA TR-R-100) ────────────────────────────

# Each table: Mach → CD at fineness ratio 3

ELLIPSOID_TABLE = LinearInterpolator(
    [1.2, 1.25, 1.3, 1.4, 1.6, 2.0, 2.4],
    [0.110, 0.128, 0.140, 0.148, 0.152, 0.159, 0.162],
)

POWER_X14_TABLE = LinearInterpolator(
    [1.2, 1.3, 1.4, 1.6, 1.8, 2.2, 2.6, 3.0, 3.6],
    [0.140, 0.156, 0.169, 0.192, 0.206, 0.227, 0.241, 0.249, 0.252],
)

POWER_X12_TABLE = LinearInterpolator(
    [0.925, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.7, 2.0],
    [0.0, 0.014, 0.050, 0.060, 0.059, 0.081, 0.084, 0.085, 0.078],
)

POWER_X34_TABLE = LinearInterpolator(
    [0.8, 0.9, 1.0, 1.06, 1.2, 1.4, 1.6, 2.0, 2.8, 3.4],
    [0.0, 0.015, 0.078, 0.121, 0.110, 0.098, 0.090, 0.084, 0.078, 0.074],
)

VON_KARMAN_TABLE = LinearInterpolator(
    [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0, 3.0],
    [0.0, 0.010, 0.027, 0.055, 0.070, 0.081, 0.095, 0.097, 0.091, 0.083],
)

LV_HAACK_TABLE = LinearInterpolator(
    [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6, 2.0],
    [0.0, 0.010, 0.024, 0.066, 0.084, 0.100, 0.114, 0.117, 0.113],
)

PARABOLIC_TABLE = LinearInterpolator(
    [0.95, 0.975, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7],
    [0.0, 0.016, 0.041, 0.092, 0.109, 0.119, 0.113, 0.108],
)

PARABOLIC_12_TABLE = LinearInterpolator(
    [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.3, 1.5, 1.8],
    [0.0, 0.016, 0.042, 0.100, 0.126, 0.125, 0.100, 0.090, 0.088],
)

PARABOLIC_34_TABLE = LinearInterpolator(
    [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7],
    [0.0, 0.023, 0.073, 0.098, 0.107, 0.106, 0.089, 0.082],
)

# Blunt body: stagnation CD at each Mach
_blunt_x = [m * 0.05 for m in range(60)]  # 0 to 3.0
_blunt_y = [stagnation_cd(m * 0.05) for m in range(60)]
BLUNT_TABLE = LinearInterpolator(_blunt_x, _blunt_y)


# ── Supersonic Fin Coefficients (K1, K2, K3) ──────────────────────────────────

GAMMA_AIR = 1.4
CNA_SUPERSONIC_MACH = 1.5

def _build_fin_k_tables():
    """Pre-compute K1, K2, K3 tables for supersonic fin CN_alpha."""
    n = int((5.0 - CNA_SUPERSONIC_MACH) * 10)
    x = []
    k1_vals, k2_vals, k3_vals = [], [], []
    for i in range(n):
        M = CNA_SUPERSONIC_MACH + i * 0.1
        beta = math.sqrt(max(1e-12, M * M - 1.0))
        x.append(M)
        k1_vals.append(2.0 / beta)
        k2_vals.append(
            ((GAMMA_AIR + 1) * M**4 - 4 * beta**2) / (4 * beta**4)
        )
        k3_vals.append(
            ((GAMMA_AIR + 1) * M**8
             + (2 * GAMMA_AIR**2 - 7 * GAMMA_AIR - 5) * M**6
             + 10 * (GAMMA_AIR + 1) * M**4 + 8)
            / (6 * beta**7)
        )
    return (
        LinearInterpolator(x, k1_vals),
        LinearInterpolator(x, k2_vals),
        LinearInterpolator(x, k3_vals),
    )


FIN_K1, FIN_K2, FIN_K3 = _build_fin_k_tables()


# ── Ogive/Conical Nose Pressure Interpolator ──────────────────────────────────

def build_ogive_nose_interpolator(shape_param: float, sinphi: float) -> LinearInterpolator:
    """
    Build a Mach-vs-CD interpolator for ogive/conical nose cones.
    shape_param: 0 = conical, higher = more ogive
    sinphi: sine of the half-angle at the tip
    """
    interp = LinearInterpolator()
    mul = 0.72 * (shape_param - 0.5)**2 + 0.82

    # Transonic M = 1.0 ... 1.3: polynomial fit
    cd_m1 = sinphi
    cd_m13 = 2.1 * sinphi**2 + 0.6019 * sinphi
    # Derivative at M=1
    d_m1 = 4.0 / (GAMMA_AIR + 1) * (1.0 - 0.5 * cd_m1)
    # Derivative at M=1.3
    d_m13 = -1.1341 * sinphi

    # Simple cubic Hermite interpolation for M in [1, 1.3]
    for m_i in range(0, 16):
        m = 1.0 + m_i * 0.02
        t = (m - 1.0) / 0.3
        # Hermite basis
        h00 = 2*t**3 - 3*t**2 + 1
        h10 = t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 = t**3 - t**2
        cd = h00 * cd_m1 + h10 * 0.3 * d_m1 + h01 * cd_m13 + h11 * 0.3 * d_m13
        interp.add_point(m, mul * max(0, cd))

    # Supersonic M > 1.3: direct formula
    for m_i in range(0, 135):
        m = 1.32 + m_i * 0.02
        beta_sq = max(1e-12, m * m - 1.0)
        cd = 2.1 * sinphi**2 + 0.5 * sinphi / math.sqrt(beta_sq)
        interp.add_point(m, mul * cd)

    return interp


# ── Surface Finish Roughness Heights ──────────────────────────────────────────

class SurfaceFinish:
    """Surface finish roughness heights (meters) from OpenRocket."""
    POLISHED     = 0.5e-6    # 0.5 μm
    SMOOTH       = 2.0e-6    # 2 μm
    NORMAL       = 10.0e-6   # 10 μm  (unfinished fiberglass)
    ROUGH        = 50.0e-6   # 50 μm  (rough paint)
    VERY_ROUGH   = 200.0e-6  # 200 μm (bare cardboard)

    NAMES = {
        "Polished":   POLISHED,
        "Smooth":     SMOOTH,
        "Normal":     NORMAL,
        "Rough":      ROUGH,
        "Very Rough": VERY_ROUGH,
    }

    @classmethod
    def get_roughness(cls, name: str) -> float:
        return cls.NAMES.get(name, cls.NORMAL)


# ── Fin Cross-Section Types ───────────────────────────────────────────────────

class FinCrossSection:
    SQUARE  = "Square"
    ROUNDED = "Rounded"
    AIRFOIL = "Airfoil"

    ALL = [SQUARE, ROUNDED, AIRFOIL]
