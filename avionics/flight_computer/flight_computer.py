"""
K2 Aerospace — Flight Computer
================================
State machine that reads sensors and commands recovery actions.

States:
    IDLE → ARMED → BOOST → COAST → APOGEE → RECOVERY → LANDED

The flight computer is decoupled from the simulation engine.
It reads sensor data and makes independent decisions.
"""

import logging
from core.event_manager import EventManager, SimEvent
from avionics.sensors.accelerometer import Accelerometer3Axis
from avionics.sensors.barometer import Barometer
from avionics.sensors.gyroscope import Gyroscope3Axis
from avionics.sensors.gps import GPSReceiver

logger = logging.getLogger("K2.FlightComputer")


class FlightComputer:
    """
    Simulated on-board flight computer.
    Reads from sensors and publishes events.
    """

    STATES = ["IDLE", "ARMED", "BOOST", "COAST", "APOGEE", "RECOVERY", "LANDED"]

    def __init__(self, event_manager: EventManager = None):
        self.event_mgr = event_manager or EventManager()
        self.state = "IDLE"

        # ── Sensors ──
        self.accelerometer = Accelerometer3Axis(noise_std=0.5, bias=0.1)
        self.barometer = Barometer(noise_std=1.0, bias=0.5)
        self.gyroscope = Gyroscope3Axis(noise_std=0.5, bias=0.1)
        self.gps = GPSReceiver(pos_noise=3.0, vel_noise=0.5)

        # ── Sensor readings (latest) ──
        self.readings = {
            "accel_x": 0.0, "accel_y": 0.0, "accel_z": 0.0,
            "baro_alt": 0.0, "baro_pressure": 0.0,
            "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
            "gps_alt": 0.0, "gps_vel": 0.0,
        }

        # ── Internal tracking ──
        self._armed = False
        self._max_baro_alt = 0.0

    def arm(self):
        """Arm the flight computer for launch detection."""
        self.state = "ARMED"
        self._armed = True
        self._max_baro_alt = 0.0
        self.barometer.set_reference_pressure(101325.0)
        logger.info("Flight computer ARMED")

    def reset(self):
        """Reset to IDLE."""
        self.state = "IDLE"
        self._armed = False
        self._max_baro_alt = 0.0
        self.readings = {k: 0.0 for k in self.readings}

    def tick(self, true_accel: float, true_pressure: float,
             true_altitude: float, true_velocity: float,
             t: float = 0.0, gyro_rates: tuple = (0, 0, 0)):
        """
        Process one tick of sensor data.

        Args:
            true_accel:    True body-axis acceleration (m/s²).
            true_pressure: True atmospheric pressure (Pa).
            true_altitude: True altitude (m).
            true_velocity: True velocity (m/s).
            t:             Simulation time (s).
            gyro_rates:    True angular rates (°/s) as (x, y, z).
        """
        # Read sensors
        ax, ay, az = self.accelerometer.read(0, 0, true_accel)
        # Barometer now expects altitude for the high-fidelity model
        baro_alt = self.barometer.read_altitude(true_altitude, t)
        gx, gy, gz = self.gyroscope.read(*gyro_rates)
        gps_alt, gps_vel = self.gps.read(true_altitude, true_velocity)

        self.readings.update({
            "accel_x": ax, "accel_y": ay, "accel_z": az,
            "baro_alt": baro_alt, "baro_pressure": self.barometer.last_reading,
            "gyro_x": gx, "gyro_y": gy, "gyro_z": gz,
            "gps_alt": gps_alt, "gps_vel": gps_vel,
        })

        # Track max barometric altitude
        if baro_alt > self._max_baro_alt:
            self._max_baro_alt = baro_alt

    def set_sensor_preset(self, preset_name: str):
        """Apply a preset profile to all sensors."""
        for sensor in [self.accelerometer.x, self.accelerometer.y, self.accelerometer.z,
                       self.barometer,
                       self.gyroscope.x, self.gyroscope.y, self.gyroscope.z,
                       self.gps.altitude, self.gps.velocity]:
            sensor.apply_preset(preset_name)
        logger.info(f"All sensors set to '{preset_name}' preset")

    def get_sensor_config(self) -> dict:
        """Return current sensor configuration for UI display."""
        return {
            "accelerometer": {"noise": self.accelerometer.x.noise_std, "bias": self.accelerometer.x.bias},
            "barometer": {"noise": self.barometer.noise_std, "bias": self.barometer.bias},
            "gyroscope": {"noise": self.gyroscope.x.noise_std, "bias": self.gyroscope.x.bias},
            "gps_position": {"noise": self.gps.altitude.noise_std, "bias": self.gps.altitude.bias},
            "gps_velocity": {"noise": self.gps.velocity.noise_std, "bias": self.gps.velocity.bias},
        }
