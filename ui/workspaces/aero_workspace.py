"""
K2 AeroSim — Aerodynamics Workspace
CD vs Mach, stability analysis, wind settings, future CFD placeholders.
"""
import numpy as np, logging
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
    QFormLayout, QLabel, QDoubleSpinBox, QSplitter, QFrame, QScrollArea, QTabWidget)
from PyQt6.QtCore import Qt
from ui.widgets.plot_widget import PlotWidget
from physics.aerodynamics import compute_drag_coefficient

logger = logging.getLogger("K2.AeroWS")

class ValueLabel(QLabel):
    def __init__(self, t="—", parent=None):
        super().__init__(t, parent)
        self.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; font-size: 13px; "
            "font-weight: 600; padding: 2px 4px; background-color: #161b22; border-radius: 4px;")


class AeroWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._setup_ui()
        self.engine.state_changed.connect(self._refresh)
        self._refresh(engine.state)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: config + readouts
        left = QScrollArea(); left.setWidgetResizable(True); left.setMaximumWidth(340)
        left.setFrameShape(QFrame.Shape.NoFrame)
        lw = QWidget(); ll = QVBoxLayout(lw); ll.setContentsMargins(12,12,12,12); ll.setSpacing(12)

        g1 = QGroupBox("Environment")
        f1 = QFormLayout(); f1.setSpacing(6)
        self.wind_spin = QDoubleSpinBox(); self.wind_spin.setRange(0, 50); self.wind_spin.setValue(0); self.wind_spin.setSuffix(" m/s")
        self.wind_spin.valueChanged.connect(lambda v: self.engine.update(wind_speed=v))
        f1.addRow("Wind Speed:", self.wind_spin)
        g1.setLayout(f1); ll.addWidget(g1)

        g2 = QGroupBox("Aerodynamic Properties")
        f2 = QFormLayout(); f2.setSpacing(6)
        self.lbl_cd = ValueLabel(); f2.addRow("CD:", self.lbl_cd)
        self.lbl_ref_area = ValueLabel(); f2.addRow("Ref Area:", self.lbl_ref_area)
        self.lbl_mach = ValueLabel(); f2.addRow("Mach:", self.lbl_mach)
        self.lbl_drag = ValueLabel(); f2.addRow("Drag Force:", self.lbl_drag)
        g2.setLayout(f2); ll.addWidget(g2)

        g3 = QGroupBox("Stability Analysis")
        f3 = QFormLayout(); f3.setSpacing(6)
        self.lbl_cp = ValueLabel(); f3.addRow("CP:", self.lbl_cp)
        self.lbl_cg = ValueLabel(); f3.addRow("CG:", self.lbl_cg)
        self.lbl_margin = ValueLabel(); f3.addRow("Margin:", self.lbl_margin)
        self.lbl_status = QLabel("—")
        self.lbl_status.setStyleSheet("font-weight: 600; font-size: 13px;")
        f3.addRow("Status:", self.lbl_status)
        g3.setLayout(f3); ll.addWidget(g3)

        # CFD placeholder
        g4 = QGroupBox("CFD Integration")
        cf = QVBoxLayout()
        placeholder = QLabel("CFD module will be available in a future release.\n\n"
            "Planned features:\n• Mesh generation\n• Pressure contours\n• Flow visualization\n• External solver coupling")
        placeholder.setWordWrap(True)
        placeholder.setStyleSheet("color: #484f58; padding: 8px;")
        cf.addWidget(placeholder)
        g4.setLayout(cf); ll.addWidget(g4)

        ll.addStretch()
        left.setWidget(lw)
        splitter.addWidget(left)

        # Right: plots
        right_tabs = QTabWidget()
        self.cd_plot = PlotWidget(title="CD vs Mach", xlabel="Mach Number", ylabel="CD")
        right_tabs.addTab(self.cd_plot, "CD vs Mach")

        self.stability_plot = PlotWidget(title="CP & CG vs Length", xlabel="Body Length (m)", ylabel="Position (m)")
        right_tabs.addTab(self.stability_plot, "Stability")

        splitter.addWidget(right_tabs)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        self._plot_cd_curve()

    def _plot_cd_curve(self):
        s = self.engine.state
        fineness = s.length / s.diameter if s.diameter > 0 else 10
        mach = np.linspace(0, 3, 200)
        cd = [compute_drag_coefficient(m, fineness) for m in mach]
        self.cd_plot.update_plot(mach, cd, "CD vs Mach Number", "Mach", "CD", "#f0883e")

    def _refresh(self, state=None):
        s = state or self.engine.state
        import math
        ref_area = math.pi * (s.diameter / 2) ** 2
        self.lbl_cd.setText(f"{s.cd:.4f}")
        self.lbl_ref_area.setText(f"{ref_area:.6f} m²")
        self.lbl_mach.setText(f"{s.mach_number:.3f}")
        self.lbl_drag.setText(f"{s.drag:.2f} N")
        self.lbl_cp.setText(f"{s.cp:.3f} m")
        self.lbl_cg.setText(f"{s.cg:.3f} m")
        self.lbl_margin.setText(f"{s.stability_margin:.2f} cal")

        if s.stability_margin < 0.5:
            self.lbl_status.setText("UNSTABLE"); self.lbl_status.setStyleSheet("color: #f85149; font-weight: 600;")
        elif s.stability_margin < 1.0:
            self.lbl_status.setText("MARGINAL"); self.lbl_status.setStyleSheet("color: #d29922; font-weight: 600;")
        elif s.stability_margin <= 2.5:
            self.lbl_status.setText("✓ STABLE"); self.lbl_status.setStyleSheet("color: #7ee787; font-weight: 600;")
        else:
            self.lbl_status.setText("OVERSTABLE"); self.lbl_status.setStyleSheet("color: #d29922; font-weight: 600;")
