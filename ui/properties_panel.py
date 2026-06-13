"""
K2 Aerospace — Properties & Output Panel
==========================================
Right dock panel displaying computed/derived rocket properties.
Auto-refreshes when RocketState changes.
"""

import logging, math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QLabel, QScrollArea, QFrame, QProgressBar
)
from PyQt6.QtCore import Qt

logger = logging.getLogger("K2.Properties")


class ValueLabel(QLabel):
    def __init__(self, text="—", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            "color: #e6edf3; font-family: 'Cascadia Code', 'Consolas', monospace; "
            "font-size: 13px; font-weight: 600; padding: 2px 4px; "
            "background-color: #161b22; border-radius: 4px;"
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)


class StabilityIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value_label = ValueLabel("0.00 cal")
        layout.addWidget(self.value_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        layout.addWidget(self.bar)
        self.status_label = QLabel("Unknown")
        self.status_label.setStyleSheet("font-size: 10px; color: #8b949e;")
        layout.addWidget(self.status_label)

    def update_stability(self, margin):
        if not math.isfinite(margin):
            margin = 0.0

        self.value_label.setText(f"{margin:.2f} cal")

        # Progress bar: map [-2, 4] range onto 0-100%
        clamped = min(max(margin, -2.0), 4.0)
        progress = int((clamped + 2.0) / 6.0 * 100)
        self.bar.setValue(min(max(progress, 0), 100))

        if margin < 0:
            status, color = "UNSTABLE", "#f85149"
        elif margin < 0.5:
            status, color = "MARGINAL–", "#d29922"
        elif margin < 1.0:
            status, color = "MARGINAL", "#d29922"
        elif margin <= 2.5:
            status, color = "✓ STABLE", "#7ee787"
        else:
            status, color = "OVERSTABLE", "#d29922"

        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: 600;")
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}")


class PropertiesPanel(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._setup_ui()
        self.engine.state_changed.connect(self._on_state_changed)

    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        ml = QVBoxLayout(content)
        ml.setContentsMargins(8, 8, 8, 8)
        ml.setSpacing(12)

        # Stability
        g = QGroupBox("Stability Analysis")
        lo = QVBoxLayout()
        self.stability_indicator = StabilityIndicator()
        lo.addWidget(self.stability_indicator)
        f = QFormLayout(); f.setSpacing(6)
        self.cg_label = ValueLabel("0.000 m"); f.addRow("CG:", self.cg_label)
        self.cp_label = ValueLabel("0.000 m"); f.addRow("CP:", self.cp_label)
        self.total_length_label = ValueLabel("0.000 m"); f.addRow("Total Length:", self.total_length_label)
        lo.addLayout(f); g.setLayout(lo); ml.addWidget(g)

        # Mass
        g = QGroupBox("Mass Properties"); f = QFormLayout(); f.setSpacing(6)
        self.total_mass_label = ValueLabel("0.000 kg"); f.addRow("Total:", self.total_mass_label)
        self.weight_label = ValueLabel("0.000 N"); f.addRow("Weight:", self.weight_label)
        g.setLayout(f); ml.addWidget(g)



        # Motor
        g = QGroupBox("Active Motor"); f = QFormLayout(); f.setSpacing(6)
        self.motor_name_label = ValueLabel("None"); f.addRow("Motor:", self.motor_name_label)
        self.motor_impulse_label = ValueLabel("—"); f.addRow("Impulse:", self.motor_impulse_label)
        self.motor_thrust_label = ValueLabel("—"); f.addRow("Thrust:", self.motor_thrust_label)
        self.motor_burn_label = ValueLabel("—"); f.addRow("Burn:", self.motor_burn_label)
        g.setLayout(f); ml.addWidget(g)

        ml.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_state_changed(self, state):
        self.stability_indicator.update_stability(state.stability_margin)
        self.cg_label.setText(f"{state.cg:.3f} m")
        self.cp_label.setText(f"{state.cp:.3f} m")
        self.total_length_label.setText(f"{state.length:.3f} m")
        self.total_mass_label.setText(f"{state.total_mass():.3f} kg")
        self.weight_label.setText(f"{state.weight:.3f} N")
        self.motor_name_label.setText(state.motor_designation)
        if state.motor_total_impulse > 0:
            self.motor_impulse_label.setText(f"{state.motor_total_impulse:.1f} N·s")
            self.motor_thrust_label.setText(f"{state.motor_avg_thrust:.1f} N")
            self.motor_burn_label.setText(f"{state.motor_burn_time:.2f} s")
        else:
            self.motor_impulse_label.setText("—")
            self.motor_thrust_label.setText("—")
            self.motor_burn_label.setText("—")
