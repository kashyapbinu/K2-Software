"""
K2 Aerospace — Simulated Gyroscope
====================================
Measures angular rates around body axes.
Hobby: MPU6050 (~0.5 °/s noise)
Industrial: BMI088 (~0.05 °/s)
"""

from avionics.sensors.base_sensor import BaseSensor


class Gyroscope(BaseSensor):
    """Single-axis gyroscope."""

    def __init__(self, noise_std: float = 0.5, bias: float = 0.1):
        super().__init__(
            name="Gyroscope",
            units="°/s",
            noise_std=noise_std,
            bias=bias,
            sample_rate=1000.0,
            range_min=-2000.0,
            range_max=2000.0,
        )

    def _preset_scale(self) -> float:
        return 0.5  # °/s per unit noise


class Gyroscope3Axis:
    """Three-axis gyroscope bundle."""

    def __init__(self, noise_std: float = 0.5, bias: float = 0.1):
        self.x = Gyroscope(noise_std, bias)
        self.x.name = "Gyro_X"
        self.y = Gyroscope(noise_std, bias)
        self.y.name = "Gyro_Y"
        self.z = Gyroscope(noise_std, bias)
        self.z.name = "Gyro_Z"

    def read(self, gx: float, gy: float, gz: float) -> tuple:
        return self.x.read(gx), self.y.read(gy), self.z.read(gz)

    def configure_all(self, noise_std: float = None, bias: float = None):
        for s in [self.x, self.y, self.z]:
            s.configure(noise_std=noise_std, bias=bias)
