"""
K2 Aerospace — 1D Kalman Filter
=================================
Fuses barometer altitude and accelerometer-integrated velocity
to produce a low-noise altitude/velocity estimate for the flight computer.
"""
import logging

logger = logging.getLogger("K2.Kalman")


class KalmanFilter1D:
    """
    Linear Kalman filter for altitude + vertical velocity fusion.

    State: [altitude (m), velocity (m/s)]
    Measurement: barometric altitude (m)
    Input: acceleration (m/s²)

    Process model:
        z[k+1] = z[k] + v[k]*dt + 0.5*a*dt^2
        v[k+1] = v[k] + a*dt

    Measurement model:
        y = z + noise
    """

    def __init__(self, q_altitude: float = 0.1, q_velocity: float = 1.0,
                 r_baro: float = 2.0):
        """
        Args:
            q_altitude:  Process noise variance for altitude (m²).
            q_velocity:  Process noise variance for velocity (m²/s²).
            r_baro:      Measurement noise variance for barometer (m²).
        """
        self.q_alt = q_altitude
        self.q_vel = q_velocity
        self.r_baro = r_baro

        # State: [altitude, velocity]
        self.x = [0.0, 0.0]

        # Covariance matrix P (2x2, stored flat: [p00, p01, p10, p11])
        self.P = [10.0, 0.0, 0.0, 10.0]

        self.initialized = False

    def init(self, altitude: float, velocity: float = 0.0):
        """Initialize filter state."""
        self.x = [altitude, velocity]
        self.P = [1.0, 0.0, 0.0, 1.0]
        self.initialized = True

    def predict(self, accel: float, dt: float):
        """
        Propagate state forward using IMU acceleration.
        """
        if not self.initialized:
            return

        z, v = self.x
        dt2 = dt * dt

        # State prediction
        z_new = z + v * dt + 0.5 * accel * dt2
        v_new = v + accel * dt
        self.x = [z_new, v_new]

        # Covariance prediction: P = F*P*F' + Q
        # F = [[1, dt], [0, 1]]
        p00, p01, p10, p11 = self.P
        q00 = p00 + dt * (p10 + p01) + dt2 * p11 + self.q_alt
        q01 = p01 + dt * p11
        q10 = p10 + dt * p11
        q11 = p11 + self.q_vel
        self.P = [q00, q01, q10, q11]

    def update(self, baro_altitude: float):
        """
        Correct state using barometric altitude measurement.
        """
        if not self.initialized:
            self.init(baro_altitude)
            return

        z, v = self.x
        p00, p01, p10, p11 = self.P

        # Innovation: y = baro - z_predicted
        y = baro_altitude - z

        # Innovation covariance: S = H*P*H' + R  (H = [1, 0])
        S = p00 + self.r_baro
        if abs(S) < 1e-10:
            return

        # Kalman gain: K = P*H'/S
        k0 = p00 / S
        k1 = p10 / S

        # State update
        self.x = [z + k0 * y, v + k1 * y]

        # Covariance update: P = (I - K*H)*P
        p00_new = (1 - k0) * p00
        p01_new = (1 - k0) * p01
        p10_new = p10 - k1 * p00
        p11_new = p11 - k1 * p01
        self.P = [p00_new, p01_new, p10_new, p11_new]

    @property
    def altitude(self) -> float:
        return self.x[0]

    @property
    def velocity(self) -> float:
        return self.x[1]
