"""
K2 Aerospace — Simulated GPS Receiver
=======================================
Provides position and velocity with configurable noise.
Consumer: ~3m position noise, ~0.5 m/s velocity noise
Industrial: ~0.5m position, ~0.1 m/s velocity
"""

from avionics.sensors.base_sensor import BaseSensor


class GPS(BaseSensor):
    """GPS position sensor (altitude channel)."""

    def __init__(self, noise_std: float = 3.0, bias: float = 0.0):
        super().__init__(
            name="GPS_Alt",
            units="m",
            noise_std=noise_std,
            bias=bias,
            sample_rate=10.0,  # typical 10 Hz
            range_min=-100.0,
            range_max=100000.0,
        )

    def _preset_scale(self) -> float:
        return 3.0  # meters per unit noise


class GPSVelocity(BaseSensor):
    """GPS velocity sensor."""

    def __init__(self, noise_std: float = 0.5, bias: float = 0.0):
        super().__init__(
            name="GPS_Vel",
            units="m/s",
            noise_std=noise_std,
            bias=bias,
            sample_rate=10.0,
            range_min=-1000.0,
            range_max=1000.0,
        )

    def _preset_scale(self) -> float:
        return 0.5  # m/s per unit noise


class GPSReceiver:
    """Bundled GPS receiver with position and velocity channels."""

    def __init__(self, pos_noise: float = 3.0, vel_noise: float = 0.5):
        self.altitude = GPS(pos_noise)
        self.velocity = GPSVelocity(vel_noise)

    def read(self, true_alt: float, true_vel: float) -> tuple:
        return self.altitude.read(true_alt), self.velocity.read(true_vel)

    def configure(self, pos_noise: float = None, vel_noise: float = None):
        if pos_noise is not None:
            self.altitude.configure(noise_std=pos_noise)
        if vel_noise is not None:
            self.velocity.configure(noise_std=vel_noise)
