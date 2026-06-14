"""
K2 AeroSim — International Standard Atmosphere (ISA)
=======================================================
Atmosphere model valid from sea level to 86 km.

Layers:
    Troposphere    :  0 – 11 km   (lapse rate −6.5 K/km)
    Tropopause     : 11 – 20 km   (isothermal 216.65 K)
    Stratosphere   : 20 – 32 km   (lapse rate +1.0 K/km)
    Upper strato   : 32 – 47 km   (lapse rate +2.8 K/km)
    Stratopause    : 47 – 51 km   (isothermal 270.65 K)
    Mesosphere     : 51 – 71 km   (lapse rate −2.8 K/km)
    Upper meso     : 71 – 86 km   (lapse rate −2.0 K/km)

All values in SI units.
"""

import math

# Sea-level reference
T0 = 288.15      # K
P0 = 101325.0    # Pa
RHO0 = 1.225     # kg/m³
G0 = 9.80665     # m/s²
R_AIR = 287.058   # J/(kg·K)
GAMMA = 1.4       # Cp/Cv for air

# ISA layer definitions: (base_alt_m, base_temp_K, lapse_rate_K_per_m)
_LAYERS = [
    (0,     288.15,  -0.0065),   # Troposphere
    (11000, 216.65,   0.0),      # Tropopause
    (20000, 216.65,   0.001),    # Lower Stratosphere
    (32000, 228.65,   0.0028),   # Upper Stratosphere
    (47000, 270.65,   0.0),      # Stratopause
    (51000, 270.65,  -0.0028),   # Mesosphere
    (71000, 214.65,  -0.002),    # Upper Mesosphere
]

_LAYER_TOPS = [11000, 20000, 32000, 47000, 51000, 71000, 86000]


class Atmosphere:
    """International Standard Atmosphere model.

    Supports the ISA+ΔT convention via ``temperature_offset``: temperature
    (and therefore density and speed of sound) is shifted by a constant
    offset while pressure keeps the standard profile.
    """

    def __init__(self, temperature_offset: float = 0.0):
        self.temperature_offset = temperature_offset
        # Pre-compute base pressures for each layer
        self._base_pressures = [P0]
        self._base_densities = [RHO0]
        for i in range(len(_LAYERS) - 1):
            h_base, T_base, lapse = _LAYERS[i]
            h_top = _LAYER_TOPS[i]
            dh = h_top - h_base
            p_base = self._base_pressures[i]

            if abs(lapse) < 1e-10:
                # Isothermal layer
                p_top = p_base * math.exp(-G0 * dh / (R_AIR * T_base))
            else:
                T_top = T_base + lapse * dh
                p_top = p_base * (T_top / T_base) ** (-G0 / (lapse * R_AIR))

            self._base_pressures.append(p_top)
            self._base_densities.append(p_top / (R_AIR * _LAYERS[i + 1][1]))

    def _get_layer(self, altitude: float) -> int:
        """Return layer index for a given altitude."""
        for i, h_top in enumerate(_LAYER_TOPS):
            if altitude < h_top:
                return i
        return len(_LAYERS) - 1

    def temperature(self, altitude: float) -> float:
        """Temperature in Kelvin at given altitude (m), including ISA+ΔT offset."""
        altitude = max(0.0, min(altitude, 86000.0))
        i = self._get_layer(altitude)
        h_base, T_base, lapse = _LAYERS[i]
        return max(1.0, T_base + lapse * (altitude - h_base) + self.temperature_offset)

    def pressure(self, altitude: float) -> float:
        """Pressure in Pascals at given altitude (m)."""
        altitude = max(0.0, min(altitude, 86000.0))
        i = self._get_layer(altitude)
        h_base, T_base, lapse = _LAYERS[i]
        p_base = self._base_pressures[i]
        dh = altitude - h_base

        if abs(lapse) < 1e-10:
            return p_base * math.exp(-G0 * dh / (R_AIR * T_base))
        else:
            T = T_base + lapse * dh
            return p_base * (T / T_base) ** (-G0 / (lapse * R_AIR))

    def density(self, altitude: float) -> float:
        """Air density in kg/m³ at given altitude (m)."""
        T = self.temperature(altitude)
        P = self.pressure(altitude)
        if T <= 0:
            return 0.0
        return P / (R_AIR * T)

    def speed_of_sound(self, altitude: float) -> float:
        """Speed of sound in m/s at given altitude (m)."""
        T = self.temperature(altitude)
        if T <= 0:
            return 0.0
        return math.sqrt(GAMMA * R_AIR * T)

    def mach_number(self, velocity: float, altitude: float) -> float:
        """Mach number for given velocity and altitude."""
        a = self.speed_of_sound(altitude)
        return abs(velocity) / a if a > 0 else 0.0

    def dynamic_pressure(self, velocity: float, altitude: float) -> float:
        """Dynamic pressure (q = ½ρv²) in Pascals."""
        rho = self.density(altitude)
        return 0.5 * rho * velocity ** 2

    def kinematic_viscosity(self, altitude: float) -> float:
        """Kinematic viscosity approximation (m²/s)."""
        T = self.temperature(altitude)
        rho = self.density(altitude)
        if rho <= 0:
            return 0.0
        # Sutherland's law for dynamic viscosity
        mu = 1.458e-6 * T**1.5 / (T + 110.4)
        return mu / rho

    def kinematic_viscosity_linear(self, altitude: float) -> float:
        """
        OpenRocket's linear approximation for kinematic viscosity (m²/s).
        Accurate in the -40 to +40°C range. Faster than Sutherland's.

        ν = (3.7291e-6 + 4.9944e-8 * T) / ρ
        """
        T = self.temperature(altitude)
        rho = self.density(altitude)
        if rho <= 0:
            return 0.0
        mu_approx = 3.7291e-6 + 4.9944e-8 * T
        return mu_approx / rho

    def reynolds_number(self, velocity: float, characteristic_length: float,
                        altitude: float) -> float:
        """
        Reynolds number: Re = V * L / ν

        Args:
            velocity:              Wind-relative speed (m/s).
            characteristic_length: Rocket body length (m).
            altitude:              Current altitude (m).

        Returns:
            Reynolds number (dimensionless).
        """
        nu = self.kinematic_viscosity(altitude)
        if nu <= 0:
            return 0.0
        return abs(velocity) * characteristic_length / nu


# ── Module-level convenience functions (backward-compatible) ──

_DEFAULT = Atmosphere()

def air_density_at_altitude(altitude: float) -> float:
    """Backward-compatible wrapper."""
    return _DEFAULT.density(altitude)

def speed_of_sound_at_altitude(altitude: float) -> float:
    """Backward-compatible wrapper."""
    return _DEFAULT.speed_of_sound(altitude)

def temperature_at_altitude(altitude: float) -> float:
    return _DEFAULT.temperature(altitude)

def pressure_at_altitude(altitude: float) -> float:
    return _DEFAULT.pressure(altitude)

def reynolds_number(velocity: float, length: float, altitude: float) -> float:
    """Module-level Reynolds number convenience function."""
    return _DEFAULT.reynolds_number(velocity, length, altitude)
