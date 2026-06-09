"""
K2 Aerospace — Base Sensor Model
==================================
Abstract base class for all simulated sensors.
Provides configurable noise, bias, and sample rate.
"""

import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("K2.Sensors")


# ── Preset Sensor Profiles ──────────────────────────────────────

SENSOR_PRESETS = {
    "ideal": {
        "noise_std": 0.0,
        "bias": 0.0,
        "description": "Perfect sensor (no noise)",
    },
    "consumer": {
        "noise_std": 1.0,
        "bias": 0.1,
        "description": "Consumer grade (phone-level)",
    },
    "hobby": {
        "noise_std": 0.3,
        "bias": 0.05,
        "description": "Hobby rocketry (MPU6050, BMP280)",
    },
    "industrial": {
        "noise_std": 0.05,
        "bias": 0.01,
        "description": "Industrial grade (ADXL375, MS5611)",
    },
}


class BaseSensor(ABC):
    """
    Abstract simulated sensor with noise model.
    
    Subclasses define what 'true_value' means and what units to use.
    """

    def __init__(self, name: str, units: str = "",
                 noise_std: float = 0.0, bias: float = 0.0,
                 sample_rate: float = 100.0,
                 range_min: float = -1e9, range_max: float = 1e9):
        self.name = name
        self.units = units
        self.noise_std = noise_std
        self.bias = bias
        self.sample_rate = sample_rate
        self.range_min = range_min
        self.range_max = range_max
        self._last_reading = 0.0
        self._last_true = 0.0

    def read(self, true_value: float) -> float:
        """
        Simulate a sensor reading with noise and bias.
        
        Args:
            true_value: The actual physical value.
            
        Returns:
            Noisy sensor reading.
        """
        self._last_true = true_value
        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        reading = true_value + self.bias + noise
        reading = np.clip(reading, self.range_min, self.range_max)
        self._last_reading = reading
        return reading

    def configure(self, noise_std: float = None, bias: float = None,
                  sample_rate: float = None):
        """Update sensor parameters."""
        if noise_std is not None:
            self.noise_std = noise_std
        if bias is not None:
            self.bias = bias
        if sample_rate is not None:
            self.sample_rate = sample_rate

    def apply_preset(self, preset_name: str):
        """Apply a named preset profile."""
        preset = SENSOR_PRESETS.get(preset_name)
        if preset:
            self.noise_std = preset["noise_std"] * self._preset_scale()
            self.bias = preset["bias"] * self._preset_scale()
            logger.info(f"Sensor '{self.name}' preset: {preset_name}")

    @abstractmethod
    def _preset_scale(self) -> float:
        """Scale factor for preset noise/bias values (varies by sensor type)."""
        ...

    @property
    def last_reading(self) -> float:
        return self._last_reading

    @property
    def last_error(self) -> float:
        return self._last_reading - self._last_true

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', noise={self.noise_std:.3f}, bias={self.bias:.3f})"
