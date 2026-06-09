"""
K2 Aerospace — Physical Constants & Unit Definitions
=====================================================
Centralized physical constants used across all simulation modules.
All values in SI units.

Atmosphere functions have been moved to core.atmosphere.
Backward-compatible wrappers are provided below.
"""

# ─── Gravitational ──────────────────────────────────────────────────────────
G_EARTH = 9.80665          # m/s² — standard gravitational acceleration
G_UNIVERSAL = 6.67430e-11  # N⋅m²/kg² — universal gravitational constant
EARTH_RADIUS = 6.371e6     # m — mean Earth radius
EARTH_MASS = 5.972e24      # kg

# ─── Atmospheric (Sea Level, ISA) ────────────────────────────────────────────
RHO_SEA_LEVEL = 1.225      # kg/m³ — air density at sea level
P_SEA_LEVEL = 101325.0     # Pa — atmospheric pressure at sea level
T_SEA_LEVEL = 288.15       # K — temperature at sea level (15 °C)
LAPSE_RATE = 0.0065        # K/m — temperature lapse rate (troposphere)

# ─── Gas Properties (Air) ───────────────────────────────────────────────────
GAMMA_AIR = 1.4            # ratio of specific heats (Cp/Cv)
R_SPECIFIC_AIR = 287.058   # J/(kg⋅K) — specific gas constant for air
SPEED_OF_SOUND_SL = 343.0  # m/s — speed of sound at sea level (ISA)

# ─── Conversions ────────────────────────────────────────────────────────────
DEG_TO_RAD = 0.017453292519943295
RAD_TO_DEG = 57.29577951308232
LBF_TO_N = 4.44822
KG_TO_LB = 2.20462
M_TO_FT = 3.28084
FT_TO_M = 0.3048
PA_TO_PSI = 0.000145038
IN_TO_M = 0.0254


# ─── Backward-Compatible Atmosphere Wrappers ────────────────────────────────
# These delegate to core.atmosphere. New code should import directly from there.

def air_density_at_altitude(altitude: float) -> float:
    """Compute air density at altitude. Delegates to core.atmosphere."""
    from environment.atmosphere_model import air_density_at_altitude as _impl
    return _impl(altitude)


def speed_of_sound_at_altitude(altitude: float) -> float:
    """Compute speed of sound at altitude. Delegates to core.atmosphere."""
    from environment.atmosphere_model import speed_of_sound_at_altitude as _impl
    return _impl(altitude)


# ─── Altitude-Dependent Gravity ─────────────────────────────────────────────
# Ported from OpenRocket: g(h) = g0 * (R / (R + h))²
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s — Earth's angular velocity

def gravity_at_altitude(altitude: float) -> float:
    """
    Gravitational acceleration at a given altitude (m) using inverse-square law.

    At sea level: 9.80665 m/s²
    At 10 km:     ~9.776 m/s²  (0.3% reduction)
    At 100 km:    ~9.50 m/s²   (3% reduction)
    """
    ratio = EARTH_RADIUS / (EARTH_RADIUS + max(0.0, altitude))
    return G_EARTH * ratio * ratio
