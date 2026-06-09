"""
K2 Aerospace — Simulated Accelerometer
========================================
Measures acceleration along rocket body axis.
Hobby profile: MPU6050 (~0.5 m/s² noise)
Industrial: ADXL375 (~0.05 m/s²)
"""

from avionics.sensors.base_sensor import BaseSensor


class Accelerometer(BaseSensor):
    """3-axis accelerometer (reports body-axis acceleration)."""

    def __init__(self, noise_std: float = 0.5, bias: float = 0.1):
        super().__init__(
            name="Accelerometer",
            units="m/s²",
            noise_std=noise_std,
            bias=bias,
            sample_rate=1000.0,
            range_min=-200 * 9.81,  # ±200g
            range_max=200 * 9.81,
        )

    def _preset_scale(self) -> float:
        return 0.5  # m/s² per unit noise


class Accelerometer3Axis:
    """Three-axis accelerometer bundle."""

    def __init__(self, noise_std: float = 0.5, bias: float = 0.1):
        self.x = Accelerometer(noise_std, bias)
        self.x.name = "Accel_X"
        self.y = Accelerometer(noise_std, bias)
        self.y.name = "Accel_Y"
        self.z = Accelerometer(noise_std, bias)
        self.z.name = "Accel_Z"

    def read(self, ax: float, ay: float, az: float) -> tuple:
        return self.x.read(ax), self.y.read(ay), self.z.read(az)

    def configure_all(self, noise_std: float = None, bias: float = None):
        for s in [self.x, self.y, self.z]:
            s.configure(noise_std=noise_std, bias=bias)
