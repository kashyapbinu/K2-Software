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


class ProfileAtmosphere(Atmosphere):
    """An atmosphere from measured levels (a weather sounding), with ISA
    above and below the measurements.

    *levels*: ``[(altitude_m above sea level, pressure_Pa, temperature_K),
    ...]``. Between levels temperature is interpolated linearly and pressure
    log-linearly in altitude (exact for an isothermal slab, very close for a
    lapse layer). Above the top level the ISA shape continues from the top
    level's values; below the bottom one, ISA's shape continues down from it.
    Density follows from the ideal-gas law, as in Atmosphere.
    """

    def __init__(self, levels):
        super().__init__(0.0)
        clean = sorted((float(h), float(p), float(t)) for h, p, t in levels)
        if len(clean) < 2:
            raise ValueError("an atmosphere profile needs at least two levels")
        for (h0, _, _), (h1, _, _) in zip(clean, clean[1:]):
            if h1 <= h0:
                raise ValueError("profile altitudes must all differ")
        if any(p <= 0.0 or t <= 0.0 for _, p, t in clean):
            raise ValueError("profile pressure and temperature must be positive")
        self.levels = clean
        self._log_p = [math.log(p) for _, p, _ in clean]

    def _isa(self, altitude: float):
        return (Atmosphere.temperature(self, altitude),
                Atmosphere.pressure(self, altitude))

    def _anchored(self, altitude: float, anchor: tuple) -> tuple:
        """ISA's shape, shifted to pass through *anchor* (h, p, T)."""
        h_a, p_a, t_a = anchor
        t_isa_a, p_isa_a = self._isa(h_a)
        t_isa, p_isa = self._isa(altitude)
        return (max(1.0, t_isa + (t_a - t_isa_a)), p_isa * (p_a / p_isa_a))

    def _tp(self, altitude: float) -> tuple:
        lv = self.levels
        if altitude >= lv[-1][0]:
            return self._anchored(altitude, lv[-1])
        if altitude <= lv[0][0]:
            return self._anchored(altitude, lv[0])
        lo, hi = 0, len(lv) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if lv[mid][0] <= altitude:
                lo = mid
            else:
                hi = mid
        h0, _, t0 = lv[lo]
        h1, _, t1 = lv[hi]
        f = (altitude - h0) / (h1 - h0)
        t = t0 + f * (t1 - t0)
        p = math.exp(self._log_p[lo] + f * (self._log_p[hi] - self._log_p[lo]))
        return t, p

    def temperature(self, altitude: float) -> float:
        return self._tp(altitude)[0] + self.temperature_offset

    def pressure(self, altitude: float) -> float:
        return self._tp(altitude)[1]


class SiteAtmosphere:
    """An Atmosphere read by height above the launch pad.

    The flight engines carry altitude above ground level (the pad is z = 0).
    Every atmosphere lookup they make goes through this, which adds the site
    elevation, so a flight from a 1,400 m pad meets the air at 1,400 m and
    above instead of sea-level air. Before launch sites existed every flight
    started at sea level.
    """

    def __init__(self, atmosphere: "Atmosphere", site_altitude: float = 0.0):
        self.atmosphere = atmosphere
        self.site_altitude = float(site_altitude)

    @property
    def temperature_offset(self) -> float:
        return self.atmosphere.temperature_offset

    @temperature_offset.setter
    def temperature_offset(self, value: float) -> None:
        self.atmosphere.temperature_offset = value

    def above_sea_level(self, height: float) -> float:
        """Height above the pad (m) → altitude above sea level (m)."""
        return height + self.site_altitude

    def temperature(self, height: float) -> float:
        return self.atmosphere.temperature(height + self.site_altitude)

    def pressure(self, height: float) -> float:
        return self.atmosphere.pressure(height + self.site_altitude)

    def density(self, height: float) -> float:
        return self.atmosphere.density(height + self.site_altitude)

    def speed_of_sound(self, height: float) -> float:
        return self.atmosphere.speed_of_sound(height + self.site_altitude)

    def mach_number(self, velocity: float, height: float) -> float:
        return self.atmosphere.mach_number(velocity, height + self.site_altitude)

    def dynamic_pressure(self, velocity: float, height: float) -> float:
        return self.atmosphere.dynamic_pressure(velocity, height + self.site_altitude)

    def kinematic_viscosity(self, height: float) -> float:
        return self.atmosphere.kinematic_viscosity(height + self.site_altitude)

    def reynolds_number(self, velocity: float, characteristic_length: float,
                        height: float) -> float:
        return self.atmosphere.reynolds_number(
            velocity, characteristic_length, height + self.site_altitude)


def launch_site_atmosphere(site_altitude: float = 0.0,
                           ground_temperature: float = 0.0,
                           profile=None) -> SiteAtmosphere:
    """The atmosphere a flight from this site flies through, read by height
    above the pad. Both flight engines build theirs here.

    *profile*: measured levels ``[(altitude_m ASL, pressure_Pa,
    temperature_K), ...]`` from a weather sounding. When given it is the air,
    and *ground_temperature* is not used (the sounding already says how warm
    the pad is).

    *ground_temperature* is the air temperature at the pad (K); 0 means ISA.
    The ISA+ΔT offset is taken against ISA **at the site elevation**, so the
    pad reads exactly the temperature entered. (The offset used to be taken
    against the sea-level 288.15 K, which was the same thing only while every
    pad was at sea level.)
    """
    if profile:
        return SiteAtmosphere(ProfileAtmosphere(profile), site_altitude)
    atm = Atmosphere()
    if ground_temperature and ground_temperature > 0.0:
        atm.temperature_offset = ground_temperature - atm.temperature(site_altitude)
    return SiteAtmosphere(atm, site_altitude)


def isa_temperature(altitude: float) -> float:
    """Standard-day temperature (K) at *altitude* above sea level."""
    return _DEFAULT.temperature(altitude)


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
