"""
K2 AeroSim — Turbulence Model
================================
Bandlimited Dryden turbulence for rocket flight simulation.
"""
import math
import random
import logging

logger = logging.getLogger("K2.Turbulence")


class TurbulenceModel:
    """First-order Dryden shaping filter for correlated wind gusts."""

    def __init__(self, intensity: float = 0.0, scale_length: float = 533.0, seed=None):
        self.intensity    = intensity
        self.scale_length = max(scale_length, 1.0)
        self._rng = random.Random(seed)
        self._u_state = 0.0
        self._w_state = 0.0

    def get_turbulence(self, altitude: float, velocity: float, dt: float):
        if self.intensity <= 0 or velocity < 1.0:
            return 0.0, 0.0
        sigma = self.intensity * math.exp(-altitude / 600.0)
        if sigma < 1e-4:
            return 0.0, 0.0
        tau   = self.scale_length / max(velocity, 1.0)
        alpha = max(0.0, 1.0 - dt / tau)
        drive = math.sqrt(max(0.0, 2.0 * sigma ** 2 * dt / tau))
        self._u_state = alpha * self._u_state + drive * self._rng.gauss(0, 1)
        self._w_state = alpha * self._w_state + drive * self._rng.gauss(0, 1)
        return self._u_state, self._w_state

    def reset(self):
        self._u_state = self._w_state = 0.0
