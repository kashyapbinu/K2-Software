"""
K2 Aerospace — Avionics Workspace
Simulated telemetry dashboard with sensor configuration and flight computer state.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QGridLayout, QFrame, QComboBox, QDoubleSpinBox, QFormLayout)
from PyQt6.QtCore import Qt
from ui.widgets.gauge_widget import GaugeWidget

logger = logging.getLogger("K2.AvionicsWS")


class TelemetryLabel(QLabel):
    def __init__(self, t="—", parent=None):
        super().__init__(t, parent)
        self.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; "
            "font-size: 15px; font-weight: 600; padding: 4px 8px; "
            "background-color: #161b22; border: 1px solid #21262d; border-radius: 6px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class StatusLight(QLabel):
    def __init__(self, label, parent=None):
        super().__init__(label, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: #484f58; font-weight: 600; font-size: 12px; "
            "padding: 6px 12px; background-color: #161b22; border: 1px solid #21262d; border-radius: 6px;")

    def set_active(self, active, color="#7ee787"):
        if active:
            self.setStyleSheet(f"color: {color}; font-weight: 700; font-size: 12px; "
                f"padding: 6px 12px; background-color: #0d1117; "
                f"border: 2px solid {color}; border-radius: 6px;")
        else:
            self.setStyleSheet("color: #484f58; font-weight: 600; font-size: 12px; "
                "padding: 6px 12px; background-color: #161b22; border: 1px solid #21262d; border-radius: 6px;")


class AvionicsWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._flight_computer = None
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)
        self.engine.telemetry_tick.connect(self._on_telemetry)

    def set_flight_computer(self, fc):
        """Connect to the simulation's flight computer."""
        self._flight_computer = fc

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Title bar
        title = QLabel("FLIGHT COMPUTER TELEMETRY")
        title.setStyleSheet("color: #58a6ff; font-size: 16px; font-weight: 700; letter-spacing: 2px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Gauges row
        gauge_layout = QHBoxLayout()
        gauge_layout.setSpacing(16)

        self.alt_gauge = GaugeWidget("ALT", "m", 0, 5000)
        self.vel_gauge = GaugeWidget("VEL", "m/s", 0, 500)
        self.accel_gauge = GaugeWidget("ACCEL", "m/s²", 0, 200)
        self.mach_gauge = GaugeWidget("MACH", "", 0, 3)
        self.thrust_gauge = GaugeWidget("THRUST", "N", 0, 2000)

        for g in [self.alt_gauge, self.vel_gauge, self.accel_gauge, self.mach_gauge, self.thrust_gauge]:
            gauge_layout.addWidget(g)
        layout.addLayout(gauge_layout)

        # Telemetry readouts
        telem_group = QGroupBox("Live Telemetry Data")
        tg = QGridLayout(); tg.setSpacing(8)

        labels = ["Time", "Altitude", "Velocity", "Acceleration", "Mach", "Thrust", "Mass", "Drag"]
        self.telem_labels = {}
        for i, name in enumerate(labels):
            header = QLabel(name)
            header.setStyleSheet("color: #58a6ff; font-weight: 600; font-size: 11px;")
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tg.addWidget(header, 0, i)
            val = TelemetryLabel("—")
            self.telem_labels[name] = val
            tg.addWidget(val, 1, i)

        telem_group.setLayout(tg)
        layout.addWidget(telem_group)

        # Bottom: FC state + Gyro + Sensor Config
        bottom = QHBoxLayout()
        bottom.setSpacing(14)

        # FC state
        fc_group = QGroupBox("Flight Computer")
        fc_layout = QVBoxLayout(); fc_layout.setSpacing(8)
        self.fc_state = TelemetryLabel("IDLE")
        self.fc_state.setStyleSheet(self.fc_state.styleSheet() + "font-size: 20px;")
        fc_layout.addWidget(self.fc_state)

        self.phase_label = TelemetryLabel("Pre-Launch")
        fc_layout.addWidget(self.phase_label)

        # Status lights
        sl = QHBoxLayout(); sl.setSpacing(6)
        self.light_armed = StatusLight("ARMED")
        self.light_boost = StatusLight("BOOST")
        self.light_coast = StatusLight("COAST")
        self.light_drogue = StatusLight("DROGUE")
        self.light_main = StatusLight("MAIN")
        self.light_land = StatusLight("LANDED")
        for l in [self.light_armed, self.light_boost, self.light_coast,
                  self.light_drogue, self.light_main, self.light_land]:
            sl.addWidget(l)
        fc_layout.addLayout(sl)

        fc_group.setLayout(fc_layout)
        bottom.addWidget(fc_group, 2)

        # Gyro
        gyro_group = QGroupBox("IMU / Gyroscope")
        gl = QGridLayout(); gl.setSpacing(6)
        self.gyro_labels = {}
        for i, axis in enumerate(["X", "Y", "Z"]):
            h = QLabel(axis); h.setStyleSheet("color: #58a6ff; font-weight: 600;")
            h.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(h, 0, i)
            v = TelemetryLabel("0.00")
            self.gyro_labels[axis] = v
            gl.addWidget(v, 1, i)
        gyro_group.setLayout(gl)
        bottom.addWidget(gyro_group, 1)

        # Sensor configuration
        sensor_group = QGroupBox("Sensor Configuration")
        sensor_layout = QFormLayout(); sensor_layout.setSpacing(6)

        self.sensor_preset = QComboBox()
        self.sensor_preset.addItems(["Hobby (MPU6050/BMP280)", "Consumer", "Industrial", "Ideal (No Noise)"])
        self.sensor_preset.setCurrentIndex(0)
        self.sensor_preset.currentIndexChanged.connect(self._on_preset_changed)
        sensor_layout.addRow("Preset:", self.sensor_preset)

        self.accel_noise_spin = QDoubleSpinBox()
        self.accel_noise_spin.setRange(0, 10); self.accel_noise_spin.setValue(0.5)
        self.accel_noise_spin.setSuffix(" m/s²"); self.accel_noise_spin.setDecimals(2)
        sensor_layout.addRow("Accel Noise:", self.accel_noise_spin)

        self.baro_noise_spin = QDoubleSpinBox()
        self.baro_noise_spin.setRange(0, 50); self.baro_noise_spin.setValue(1.0)
        self.baro_noise_spin.setSuffix(" Pa"); self.baro_noise_spin.setDecimals(1)
        sensor_layout.addRow("Baro Noise:", self.baro_noise_spin)

        self.gyro_noise_spin = QDoubleSpinBox()
        self.gyro_noise_spin.setRange(0, 10); self.gyro_noise_spin.setValue(0.5)
        self.gyro_noise_spin.setSuffix(" °/s"); self.gyro_noise_spin.setDecimals(2)
        sensor_layout.addRow("Gyro Noise:", self.gyro_noise_spin)

        self.gps_noise_spin = QDoubleSpinBox()
        self.gps_noise_spin.setRange(0, 50); self.gps_noise_spin.setValue(3.0)
        self.gps_noise_spin.setSuffix(" m"); self.gps_noise_spin.setDecimals(1)
        sensor_layout.addRow("GPS Noise:", self.gps_noise_spin)

        sensor_group.setLayout(sensor_layout)
        bottom.addWidget(sensor_group, 1)

        layout.addLayout(bottom)
        layout.addStretch()

    def _on_preset_changed(self, index):
        presets = [
            (0.5, 1.0, 0.5, 3.0),    # Hobby
            (1.0, 5.0, 1.0, 5.0),    # Consumer
            (0.05, 0.3, 0.05, 0.5),  # Industrial
            (0.0, 0.0, 0.0, 0.0),    # Ideal
        ]
        if 0 <= index < len(presets):
            a, b, g, gps = presets[index]
            self.accel_noise_spin.setValue(a)
            self.baro_noise_spin.setValue(b)
            self.gyro_noise_spin.setValue(g)
            self.gps_noise_spin.setValue(gps)

            if self._flight_computer:
                self._flight_computer.accelerometer.configure_all(noise_std=a)
                self._flight_computer.barometer.configure(noise_std=b)
                self._flight_computer.gyroscope.configure_all(noise_std=g)
                self._flight_computer.gps.configure(pos_noise=gps)

    def _update(self, s):
        self.alt_gauge.set_value(s.altitude)
        self.vel_gauge.set_value(abs(s.velocity))
        self.accel_gauge.set_value(abs(s.acceleration))
        self.mach_gauge.set_value(s.mach_number)
        self.thrust_gauge.set_value(s.thrust)

        self.telem_labels["Time"].setText(f"{s.sim_time:.2f} s")
        self.telem_labels["Altitude"].setText(f"{s.altitude:.1f} m")
        self.telem_labels["Velocity"].setText(f"{s.velocity:.1f} m/s")
        self.telem_labels["Acceleration"].setText(f"{s.acceleration:.1f} m/s²")
        self.telem_labels["Mach"].setText(f"{s.mach_number:.3f}")
        self.telem_labels["Thrust"].setText(f"{s.thrust:.1f} N")
        self.telem_labels["Mass"].setText(f"{s.total_mass():.3f} kg")
        self.telem_labels["Drag"].setText(f"{s.drag:.1f} N")

        self.fc_state.setText(s.flight_computer_state)
        self.phase_label.setText(s.sim_phase)

        self.gyro_labels["X"].setText(f"{s.gyro_x:.2f}")
        self.gyro_labels["Y"].setText(f"{s.gyro_y:.2f}")
        self.gyro_labels["Z"].setText(f"{s.gyro_z:.2f}")

        # Status lights
        fc = s.flight_computer_state
        self.light_armed.set_active(fc in ["ARMED", "BOOST", "COAST", "APOGEE", "RECOVERY", "LANDED"], "#58a6ff")
        self.light_boost.set_active(fc == "BOOST", "#f0883e")
        self.light_coast.set_active(fc in ["COAST", "APOGEE"], "#d29922")
        self.light_drogue.set_active(s.sim_phase in ["Drogue Descent", "Main Descent"], "#7ee787")
        self.light_main.set_active(s.sim_phase == "Main Descent", "#3fb950")
        self.light_land.set_active(fc == "LANDED", "#7ee787")

    def _on_state_changed(self, s):
        self._update(s)

    def _on_telemetry(self, s):
        self._update(s)
