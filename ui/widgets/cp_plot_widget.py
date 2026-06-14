"""
K2 AeroSim — Cp Distribution Plot Widget
=============================================
Aerospace-standard pressure coefficient distribution plot with inverted
Y-axis (Cp convention: negative up = suction), hover cursor, and section
highlighting for nose/body/fin regions.
"""
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class CpPlotWidget(QWidget):
    """Aerospace-standard Cp vs x/L plot with inverted Y-axis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor("#0d1117")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._annotation = None
        self._x_data = None
        self._y_data = None
        self._style_axis()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        # Connect mouse motion for hover cursor
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _style_axis(self):
        ax = self.ax
        ax.set_facecolor("#161b22")
        ax.set_title("Cp Distribution", color="#58a6ff", fontsize=11,
                      fontweight="bold", pad=8)
        ax.set_xlabel("x / L", color="#8b949e", fontsize=10)
        ax.set_ylabel("Cp", color="#8b949e", fontsize=10)
        ax.tick_params(colors="#484f58", labelsize=9)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#30363d")
        # Aerospace convention: invert Y-axis (suction = negative Cp = top)
        ax.invert_yaxis()

    def update_cp(self, x_norm, cp_values, nose_end=0.3, fin_start=0.75):
        """
        Plot Cp distribution with section highlighting.

        Parameters
        ----------
        x_norm : array - normalized position [0, 1]
        cp_values : array - Cp values
        nose_end : float - x/L where nose ends
        fin_start : float - x/L where fin region starts
        """
        self.ax.clear()
        self._style_axis()

        if len(x_norm) == 0:
            self.canvas.draw()
            return

        self._x_data = np.asarray(x_norm)
        self._y_data = np.asarray(cp_values)

        # Section shading
        self.ax.axvspan(0, nose_end, alpha=0.06, color="#58a6ff",
                        label="Nose")
        self.ax.axvspan(nose_end, fin_start, alpha=0.04, color="#7ee787",
                        label="Body")
        self.ax.axvspan(fin_start, 1.0, alpha=0.06, color="#f0883e",
                        label="Fins/Aft")

        # Cp = 0 reference line
        self.ax.axhline(y=0, color="#484f58", linestyle="--", linewidth=0.8,
                        alpha=0.6)

        # Main Cp curve
        self.ax.plot(self._x_data, self._y_data, color="#58a6ff",
                     linewidth=1.8, zorder=5)
        self.ax.fill_between(self._x_data, self._y_data, 0, alpha=0.08,
                             color="#58a6ff")

        # Stagnation point marker
        if len(self._y_data) > 0:
            max_idx = np.argmax(self._y_data)
            self.ax.plot(self._x_data[max_idx], self._y_data[max_idx],
                        "o", color="#f85149", markersize=6, zorder=6)

        self.ax.legend(facecolor="#161b22", edgecolor="#30363d",
                       labelcolor="#8b949e", fontsize=8, loc="lower right")
        self.ax.set_xlim(0, 1)

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_hover(self, event):
        """Show local Cp value on hover."""
        if event.inaxes != self.ax or self._x_data is None:
            if self._annotation is not None:
                self._annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        x = event.xdata
        idx = np.argmin(np.abs(self._x_data - x))
        cp_val = self._y_data[idx]
        x_val = self._x_data[idx]

        if self._annotation is None:
            self._annotation = self.ax.annotate(
                "", xy=(0, 0), xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d",
                          edgecolor="#30363d", alpha=0.95),
                color="#e6edf3", fontsize=9,
                fontfamily="monospace",
            )

        self._annotation.xy = (x_val, cp_val)
        self._annotation.set_text(f"x/L = {x_val:.3f}\nCp = {cp_val:.4f}")
        self._annotation.set_visible(True)
        self.canvas.draw_idle()
