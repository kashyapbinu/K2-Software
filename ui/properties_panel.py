"""
K2 AeroSim — Properties & Output Panel
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

from ui import theme
from ui.widgets.collapsible import CollapsibleSection

logger = logging.getLogger("K2.Properties")


class ValueLabel(QLabel):
    def __init__(self, text="—", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color: {theme.TEXT_BRIGHT}; font-family: {theme.MONO}; "
            f"font-size: 13px; padding: 2px 0px; background: transparent;"
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)


class StabilityIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value_label = ValueLabel("0.00 cal")
        self.value_label.setStyleSheet(
            f"color: {theme.ACCENT}; font-family: {theme.MONO}; font-size: 17px;"
            f" font-weight: 600; padding: 2px 0px; background: transparent;")
        layout.addWidget(self.value_label)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        layout.addWidget(self.bar)
        self.status_label = QLabel("Unknown")
        self.status_label.setStyleSheet(f"font-size: 11px; color: {theme.TEXT_DIM};")
        layout.addWidget(self.status_label)

    def update_stability(self, margin):
        if not math.isfinite(margin):
            margin = 0.0

        self.value_label.setText(f"{margin:.2f} cal")

        # Progress bar: map [-2, 4] range onto 0-100%
        clamped = min(max(margin, -2.0), 4.0)
        progress = int((clamped + 2.0) / 6.0 * 100)
        self.bar.setValue(min(max(progress, 0), 100))

        # The one verdict (core.stability), as the Design tab shows it.
        from core.stability import LABELS, stability_verdict
        verdict = stability_verdict(margin)
        status, color = LABELS.get(verdict, "—"), theme.stability_color(verdict)

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
        ml.setSpacing(8)

        # Stability
        g = CollapsibleSection("Stability (current configuration)")
        lo = QVBoxLayout()
        self.stability_indicator = StabilityIndicator()
        lo.addWidget(self.stability_indicator)
        f = QFormLayout(); f.setSpacing(6)
        self.cg_label = ValueLabel("0.000 m"); f.addRow("CG:", self.cg_label)
        self.cp_label = ValueLabel("0.000 m"); f.addRow("CP:", self.cp_label)
        self.total_length_label = ValueLabel("0.000 m"); f.addRow("Total Length:", self.total_length_label)
        lo.addLayout(f); g.set_content_layout(lo); ml.addWidget(g)

        # Mass
        g = CollapsibleSection("Mass properties"); f = QFormLayout(); f.setSpacing(6)
        self.total_mass_label = ValueLabel("0.000 kg"); f.addRow("Total:", self.total_mass_label)
        self.weight_label = ValueLabel("0.000 N"); f.addRow("Weight:", self.weight_label)
        g.set_content_layout(f); ml.addWidget(g)



        # Motor
        g = CollapsibleSection("Active motor"); f = QFormLayout(); f.setSpacing(6)
        self.motor_name_label = ValueLabel("None"); f.addRow("Motor:", self.motor_name_label)
        self.motor_impulse_label = ValueLabel("—"); f.addRow("Impulse:", self.motor_impulse_label)
        self.motor_thrust_label = ValueLabel("—"); f.addRow("Thrust:", self.motor_thrust_label)
        self.motor_burn_label = ValueLabel("—"); f.addRow("Burn:", self.motor_burn_label)
        g.set_content_layout(f); ml.addWidget(g)

        ml.addStretch()
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_state_changed(self, state):
        # The rocket on the pad (RocketState.liftoff_*): this panel sits
        # beside the Design tab and describes the design. It read the live
        # mass, CG, margin and weight, so once a flight had run it showed the
        # burnt-out rocket (6.001 kg and 6.78 cal for the canonical rocket's
        # 7.400 kg and 5.30 cal) next to an unchanged drawing.
        from core.constants import G_EARTH
        margin = state.pad_stability_margin()
        mass = state.liftoff_mass()
        self.stability_indicator.update_stability(margin if margin is not None else 0.0)
        self.cg_label.setText(f"{state.liftoff_cg():.3f} m")
        self.cp_label.setText(f"{state.cp:.3f} m")
        self.total_length_label.setText(f"{state.length:.3f} m")
        self.total_mass_label.setText(f"{mass:.3f} kg")
        self.weight_label.setText(f"{mass * G_EARTH:.3f} N")
        self.motor_name_label.setText(state.motor_designation)
        if state.motor_total_impulse > 0:
            self.motor_impulse_label.setText(f"{state.motor_total_impulse:.1f} N·s")
            self.motor_thrust_label.setText(f"{state.motor_avg_thrust:.1f} N")
            self.motor_burn_label.setText(f"{state.motor_burn_time:.2f} s")
        else:
            self.motor_impulse_label.setText("—")
            self.motor_thrust_label.setText("—")
            self.motor_burn_label.setText("—")
