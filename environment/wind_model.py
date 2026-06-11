"""
K2 Aerospace — Advanced Wind Model
====================================
Ported from OpenRocket's PinkNoiseWindModel.java.

Supports:
    - Constant wind with altitude power-law profile
    - Pink noise turbulence (IIR filter, α=5/3)
    - Turbulence intensity classification
    - Configurable wind direction
"""

import math
import random
import numpy as np


# ── Pink Noise Generator (Voss-McCartney IIR filter) ─────────────────────────

class PinkNoise:
    """
    Pink noise generator using an IIR filter (ported from OpenRocket).
    α = 5/3 (Kolmogorov spectrum), 2 poles.
    """

    def __init__(self, alpha: float = 5.0/3.0, poles: int = 2, rng=None):
        self._rng = rng or random.Random()
        self._poles = poles
        # Pre-compute IIR filter multipliers from the Voss-McCartney method
        self._multipliers = []
        a = 1.0
        for i in range(1, poles + 1):
            a *= (i - 1.0 - alpha / 2.0) / i
            self._multipliers.append(a)
        self._history = [0.0] * poles

    def next_value(self) -> float:
        """Generate the next pink noise sample."""
        white = self._rng.gauss(0, 1)
        output = white
        for i in range(self._poles):
            output -= self._multipliers[i] * self._history[i]
        # Shift history
        for i in range(self._poles - 1, 0, -1):
            self._history[i] = self._history[i - 1]
        self._history[0] = output
        return output


# ── Wind Model ────────────────────────────────────────────────────────────────

class WindModel:
    """
    High-fidelity wind model with pink noise turbulence.

    Wind speed follows a power-law altitude profile and has pink-noise
    turbulence layered on top. The wind direction is constant with altitude.

    Ported from OpenRocket's PinkNoiseWindModel.

    Args:
        base_speed:       Mean wind speed at 10m altitude (m/s).
        direction:        Wind direction in degrees (0=North, 90=East).
                          This is the direction the wind is BLOWING FROM.
        gust_intensity:   Legacy parameter (ignored if turbulence_intensity > 0).
        turbulence_intensity: Standard deviation / mean speed (0.0 to 0.3 typical).
                          0.0  = calm, 0.1 = moderate, 0.2 = high
        seed:             Random seed for reproducibility.
    """

    # Pink noise parameters (from OpenRocket)
    _ALPHA = 5.0 / 3.0
    _POLES = 2
    _STDDEV = 2.252    # standard deviation of the raw pink noise output
    _DELTA_T = 0.05    # time between noise samples (s)

    # Turbulence intensity classifications
    INTENSITY_LABELS = [
        (0.001, "None"),
        (0.05,  "Very Low"),
        (0.10,  "Low"),
        (0.15,  "Medium"),
        (0.20,  "High"),
        (0.25,  "Very High"),
        (1.00,  "Extreme"),
    ]

    def __init__(self, base_speed: float = 0.0, direction: float = 0.0,
                 gust_intensity: float = 0.0,
                 turbulence_intensity: float = 0.1,
                 seed: int = None):
        self.base_speed = base_speed
        self.direction = math.radians(direction)
        self.gust_intensity = gust_intensity
        # Callers (sim engine, batch sim, weather profiles) pass the user's
        # gust setting positionally as gust_intensity — honor it as the
        # turbulence intensity instead of silently using the 0.1 default.
        if gust_intensity > 0:
            turbulence_intensity = gust_intensity
        self.turbulence_intensity = turbulence_intensity

        # Pink noise state
        seed = seed if seed is not None else random.randint(0, 2**31)
        self._seed = seed ^ 0x7343AA03  # OpenRocket seed randomization
        self._noise = None
        self._time1 = 0.0
        self._value1 = 0.0
        self._value2 = 0.0
        self._reset_noise()

    def _reset_noise(self):
        """Initialize or reset the pink noise generator."""
        self._noise = PinkNoise(self._ALPHA, self._POLES,
                                random.Random(self._seed))
        self._time1 = 0.0
        self._value1 = self._noise.next_value()
        self._value2 = self._noise.next_value()

    @property
    def standard_deviation(self) -> float:
        return self.turbulence_intensity * self.base_speed

    def get_intensity_label(self) -> str:
        """Get the human-readable turbulence intensity label."""
        ti = self.turbulence_intensity
        for threshold, label in self.INTENSITY_LABELS:
            if ti < threshold:
                return label
        return "Extreme"

    def get_wind_velocity(self, altitude: float, time: float) -> tuple[float, float, float]:
        """
        Returns (wind_vx, wind_vy, wind_vz) at the given altitude and time.

        Uses:
            1. Power-law altitude profile: v = v0 * (z/10)^0.143
            2. Pink noise turbulence around the mean
            3. Wind direction decomposition into X and Y

        Returns:
            Tuple of (wind_vx, wind_vy, wind_vz).
        """
        if altitude <= 0 or self.base_speed <= 0:
            return (0.0, 0.0, 0.0)

        # ── Altitude-scaled mean speed ──
        z = max(0.1, altitude)
        mean_speed = self.base_speed * (z / 10.0) ** 0.143

        # ── Pink noise turbulence ──
        if self.turbulence_intensity > 0.001 and self.base_speed > 0:
            speed = self._get_pink_noise_speed(time, mean_speed)
        else:
            speed = mean_speed

        # Ensure non-negative
        speed = max(0.0, speed)

        # Decompose into X and Y components. `direction` is the bearing the
        # wind blows FROM (meteorological convention, matches the landing
        # estimator in visualization.mission.flight_envelope), so the air-mass
        # velocity vector points the opposite way (toward direction + 180°).
        vx = -speed * math.cos(self.direction)
        vy = -speed * math.sin(self.direction)
        # Vertical wind component — negligible for typical rocketry
        vz = 0.0

        return (vx, vy, vz)

    def _get_pink_noise_speed(self, time: float, mean_speed: float) -> float:
        """
        Compute wind speed with pink noise turbulence (OpenRocket method).
        Linearly interpolates between noise samples taken at _DELTA_T intervals.
        """
        if time < 0:
            time = 0

        # If time went backwards (e.g. reset), re-initialize
        if time < self._time1:
            self._reset_noise()

        # Advance noise samples until we bracket `time`
        while self._time1 + self._DELTA_T < time:
            self._value1 = self._value2
            self._value2 = self._noise.next_value()
            self._time1 += self._DELTA_T

        # Linear interpolation between samples
        a = (time - self._time1) / self._DELTA_T
        a = max(0.0, min(1.0, a))
        noise_val = self._value1 * (1 - a) + self._value2 * a

        # Scale noise to desired standard deviation
        sigma = self.standard_deviation
        return mean_speed + noise_val * sigma / self._STDDEV

    def reset(self):
        """Reset the wind model state."""
        self._reset_noise()


# ── Multi-Level Wind Model ────────────────────────────────────────────────────

class MultiLevelWindModel(WindModel):
    """
    Wind model defined by altitude layers, each with its own speed and
    direction. Speed and direction are linearly interpolated between layers
    (direction along the shortest arc, so a 350°→10° transition passes
    through 0°, not 180°). Below the lowest layer and above the highest
    layer the nearest layer's values are held constant.

    Pink-noise turbulence (inherited from WindModel) is applied on top of
    the interpolated mean speed, scaled by turbulence_intensity.

    Args:
        layers: List of (altitude_m, speed_m_s, direction_deg) tuples.
                Direction is the bearing the wind blows FROM (meteorological
                convention, same as WindModel).
        turbulence_intensity: σ/mean (0.0 = smooth interpolated wind).
        seed:   Random seed for reproducibility.
    """

    def __init__(self, layers: list, turbulence_intensity: float = 0.0,
                 seed: int = None):
        # Sort by altitude and drop malformed rows
        clean = sorted(
            (float(a), max(0.0, float(s)), float(d) % 360.0)
            for a, s, d in layers
        )
        if not clean:
            clean = [(0.0, 0.0, 0.0)]
        self.layers = clean
        # base_speed only seeds the parent's turbulence bookkeeping; the
        # per-altitude σ is computed in get_wind_velocity instead.
        super().__init__(base_speed=clean[0][1], direction=clean[0][2],
                         turbulence_intensity=turbulence_intensity, seed=seed)

    def _interpolate(self, altitude: float) -> tuple[float, float]:
        """Return (mean_speed, direction_rad) at the given altitude."""
        layers = self.layers
        if altitude <= layers[0][0]:
            _, s, d = layers[0]
            return s, math.radians(d)
        if altitude >= layers[-1][0]:
            _, s, d = layers[-1]
            return s, math.radians(d)

        for i in range(len(layers) - 1):
            a0, s0, d0 = layers[i]
            a1, s1, d1 = layers[i + 1]
            if a0 <= altitude <= a1:
                f = (altitude - a0) / (a1 - a0) if a1 > a0 else 0.0
                speed = s0 + f * (s1 - s0)
                # Shortest-arc direction interpolation
                delta = ((d1 - d0 + 180.0) % 360.0) - 180.0
                direction = (d0 + f * delta) % 360.0
                return speed, math.radians(direction)

        _, s, d = layers[-1]
        return s, math.radians(d)

    def get_wind_velocity(self, altitude: float, time: float) -> tuple[float, float, float]:
        """Returns (wind_vx, wind_vy, wind_vz) at the given altitude and time."""
        mean_speed, direction = self._interpolate(max(0.0, altitude))
        if mean_speed <= 0:
            return (0.0, 0.0, 0.0)

        # Pink-noise turbulence around the interpolated mean. The parent's
        # standard_deviation property uses base_speed, so temporarily point
        # it at the local mean for correct σ scaling.
        if self.turbulence_intensity > 0.001:
            self.base_speed = mean_speed
            speed = self._get_pink_noise_speed(time, mean_speed)
        else:
            speed = mean_speed
        speed = max(0.0, speed)

        # Same FROM-bearing convention as WindModel
        vx = -speed * math.cos(direction)
        vy = -speed * math.sin(direction)
        return (vx, vy, 0.0)
