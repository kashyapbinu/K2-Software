"""
K2 AeroSim — Gauge Widget for avionics dashboard.
Custom-painted circular analog gauge.
"""
import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QConicalGradient


class GaugeWidget(QWidget):
    """Circular analog gauge with dark theme styling."""

    def __init__(self, title="", unit="", min_val=0, max_val=100, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self._value = 0.0
        self.setMinimumSize(140, 140)
        self.setMaximumSize(200, 200)

    def set_value(self, value):
        self._value = max(self.min_val, min(value, self.max_val))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h)
        p.translate(w / 2, h / 2)

        # Background circle
        p.setPen(QPen(QColor("#30363d"), 2))
        p.setBrush(QColor("#0d1117"))
        r = side * 0.42
        p.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        # Arc track
        p.setPen(QPen(QColor("#21262d"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        arc_rect = QRectF(-r * 0.85, -r * 0.85, r * 1.7, r * 1.7)
        p.drawArc(arc_rect, 225 * 16, -270 * 16)

        # Value arc
        rng = self.max_val - self.min_val
        frac = (self._value - self.min_val) / rng if rng > 0 else 0
        arc_span = -270 * frac

        if frac < 0.6:
            arc_color = QColor("#58a6ff")
        elif frac < 0.85:
            arc_color = QColor("#d29922")
        else:
            arc_color = QColor("#f85149")

        p.setPen(QPen(arc_color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(arc_rect, 225 * 16, int(arc_span * 16))

        # Needle
        angle_deg = 225 - 270 * frac
        angle_rad = math.radians(angle_deg)
        nx = r * 0.65 * math.cos(angle_rad)
        ny = -r * 0.65 * math.sin(angle_rad)
        p.setPen(QPen(QColor("#e6edf3"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(0, 0, int(nx), int(ny))

        # Center dot
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(arc_color)
        p.drawEllipse(QRectF(-4, -4, 8, 8))

        # Value text
        p.setPen(QColor("#e6edf3"))
        font = QFont("Cascadia Code", int(side * 0.09), QFont.Weight.Bold)
        p.setFont(font)
        if abs(self._value) >= 1000:
            val_text = f"{self._value:.0f}"
        elif abs(self._value) >= 10:
            val_text = f"{self._value:.1f}"
        else:
            val_text = f"{self._value:.2f}"
        p.drawText(QRectF(-r, r * 0.1, r * 2, r * 0.4), Qt.AlignmentFlag.AlignCenter, val_text)

        # Unit
        p.setPen(QColor("#484f58"))
        font2 = QFont("Segoe UI", int(side * 0.06))
        p.setFont(font2)
        p.drawText(QRectF(-r, r * 0.4, r * 2, r * 0.3), Qt.AlignmentFlag.AlignCenter, self.unit)

        # Title
        p.setPen(QColor("#58a6ff"))
        font3 = QFont("Segoe UI", int(side * 0.055), QFont.Weight.Bold)
        p.setFont(font3)
        p.drawText(QRectF(-r, -r * 0.65, r * 2, r * 0.3), Qt.AlignmentFlag.AlignCenter, self.title)

        p.end()
