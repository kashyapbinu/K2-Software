"""
K2 AeroSim — Weather Profiles
"""

from environment.wind_model import WindModel


class WeatherProfile:
    """A named wind preset.

    ``turbulence_intensity`` is the RATIO σ / mean speed, not a gust speed in
    m/s. The presets below used to pass 2.0 / 5.0 / 8.0 here — plainly intended
    as m/s — which the wind model read as 200-800% turbulence: "Storm" asked
    for a 120 m/s standard deviation on a 15 m/s wind.
    """

    def __init__(self, name: str, base_wind_speed: float, wind_direction: float,
                 turbulence_intensity: float):
        self.name = name
        self.turbulence_intensity = turbulence_intensity
        self.wind = WindModel(base_wind_speed, wind_direction,
                              turbulence_intensity=turbulence_intensity)


PROFILES = {
    "Calm":   WeatherProfile("Calm",   0.0,  0.0, 0.00),
    "Breezy": WeatherProfile("Breezy", 5.0, 45.0, 0.10),
    "Gusty":  WeatherProfile("Gusty",  8.0, 90.0, 0.20),
    "Storm":  WeatherProfile("Storm", 15.0, 180.0, 0.30),
}

def get_weather_profile(name: str) -> WeatherProfile:
    return PROFILES.get(name, PROFILES["Calm"])
