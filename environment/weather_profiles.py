"""
K2 AeroSim — Weather Profiles
"""

from environment.wind_model import WindModel

class WeatherProfile:
    def __init__(self, name: str, base_wind_speed: float, wind_direction: float, gust_intensity: float):
        self.name = name
        self.wind = WindModel(base_wind_speed, wind_direction, gust_intensity)

PROFILES = {
    "Calm": WeatherProfile("Calm", 0.0, 0.0, 0.0),
    "Breezy": WeatherProfile("Breezy", 5.0, 45.0, 2.0),
    "Gusty": WeatherProfile("Gusty", 8.0, 90.0, 5.0),
    "Storm": WeatherProfile("Storm", 15.0, 180.0, 8.0),
}

def get_weather_profile(name: str) -> WeatherProfile:
    return PROFILES.get(name, PROFILES["Calm"])
