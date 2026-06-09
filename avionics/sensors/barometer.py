"""
K2 Aerospace — High-Fidelity Simulated Barometer
==================================================
Models a realistic barometric altimeter with:
  - Configurable noise (std dev)
  - Bias drift over time
  - Update rate limiting (discrete sampling)
  - First-order lag (sensor response time)
  - Altitude quantization (LSB resolution)
"""

import random
import logging
from avionics.sensors.base_sensor import BaseSensor
from environment.atmosphere_model import Atmosphere

logger = logging.getLogger("K2.Barometer")
_atm = Atmosphere()


class Barometer(BaseSensor):
    """
    Realistic barometric pressure sensor with altitude derivation.

    Simulates:
        - Gaussian pressure noise (Pa)
        - Constant altitude bias (m)
        - Slow bias drift (random walk)
        - First-order lag filter (sensor response time)
        - Discrete update rate (Hz)
        - Altitude quantization (LSB in meters)
    """

    def __init__(self,
                 noise_std: float = 1.0,
                 bias: float = 0.5,
                 sample_rate: float = 50.0,
                 lag_tau: float = 0.05,
                 drift_rate: float = 0.001,
                 quantization: float = 0.1):
        """
        Args:
            noise_std:     Pressure noise standard deviation (Pa).
            bias:          Fixed altitude bias (m).
            sample_rate:   Sensor update rate (Hz). Default 50Hz.
            lag_tau:       First-order lag time constant (s). Default 50ms.
            drift_rate:    Bias drift rate (m/s random walk std).
            quantization:  Altitude resolution (m). Default 0.1m.
        """
        super().__init__(
            name="Barometer",
            units="Pa",
            noise_std=noise_std,
            bias=bias,
            sample_rate=sample_rate,
            range_min=1000.0,
            range_max=110000.0,
        )
        self._lag_tau      = lag_tau
        self._drift_rate   = drift_rate
        self._quantization = quantization
        self._ref_pressure = 101325.0

        # Internal state
        self._lagged_alt    = 0.0    # output of the lag filter
        self._current_bias  = bias   # drifting bias
        self._last_t        = 0.0
        self._rng           = random.Random()

    def _preset_scale(self) -> float:
        return 1.0

    def set_reference_pressure(self, p0: float):
        self._ref_pressure = p0

    def read_altitude(self, true_altitude: float, t: float = 0.0) -> float:
        """
        Return a simulated altitude reading at time t.

        Args:
            true_altitude:  True altitude (m).
            t:              Current simulation time (s).

        Returns:
            Simulated altitude reading (m) with noise, lag, drift, quantization.
        """
        dt = max(0.001, t - self._last_t)
        self._last_t = t

        # 1. Gaussian noise
        noise = self._rng.gauss(0, self.noise_std * 0.1)   # pressure noise -> ~0.1m altitude

        # 2. Bias drift (random walk)
        self._current_bias += self._rng.gauss(0, self._drift_rate * dt)

        # 3. Raw noisy altitude
        raw_alt = true_altitude + noise + self._current_bias

        # 4. First-order lag: y[n] = y[n-1] + (dt/tau)*(raw - y[n-1])
        alpha = min(1.0, dt / max(self._lag_tau, 1e-6))
        self._lagged_alt += alpha * (raw_alt - self._lagged_alt)

        # 5. Quantization (round to LSB)
        if self._quantization > 0:
            out = round(self._lagged_alt / self._quantization) * self._quantization
        else:
            out = self._lagged_alt

        return max(0.0, out)

    # Backward-compatible pressure-based path
    def read_altitude_from_pressure(self, true_pressure: float) -> float:
        """Read pressure and convert to altitude estimate (no lag)."""
        noisy = self.read(true_pressure)
        if noisy <= 0 or self._ref_pressure <= 0:
            return 0.0
        ratio = noisy / self._ref_pressure
        T0, L, R, g = 288.15, 0.0065, 287.058, 9.80665
        return max(0.0, (T0 / L) * (1 - ratio ** (L * R / g)))
