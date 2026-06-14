"""
K2 AeroSim — Dark-themed Matplotlib widget for embedding plots in Qt.
"""
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class PlotWidget(QWidget):
    """Reusable dark-themed matplotlib plot widget."""

    def __init__(self, parent=None, title="", xlabel="", ylabel=""):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.patch.set_facecolor("#0d1117")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.cursor_line = None
        self._style_axis(title, xlabel, ylabel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _style_axis(self, title, xlabel, ylabel):
        ax = self.ax
        ax.set_facecolor("#161b22")
        ax.set_title(title, color="#58a6ff", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color="#8b949e", fontsize=10)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=10)
        ax.tick_params(colors="#484f58", labelsize=9)
        ax.spines["bottom"].set_color("#30363d")
        ax.spines["left"].set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color="#30363d")

    def clear(self):
        self.ax.clear()
        self.cursor_line = None
        self._style_axis(self.ax.get_title(), self.ax.get_xlabel(), self.ax.get_ylabel())

    def plot(self, x, y, color="#58a6ff", label=None, linewidth=1.5):
        self.ax.plot(x, y, color=color, label=label, linewidth=linewidth)
        if label:
            self.ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, x, y, title="", xlabel="", ylabel="", color="#58a6ff"):
        self.ax.clear()
        self._style_axis(title, xlabel, ylabel)
        self.ax.plot(x, y, color=color, linewidth=1.5)
        self.ax.fill_between(x, y, alpha=0.1, color=color)
        self.figure.tight_layout()
        self.canvas.draw()

    def multi_plot(self, datasets, title="", xlabel="", ylabel=""):
        """datasets: list of (x, y, color, label) tuples"""
        self.ax.clear()
        self._style_axis(title, xlabel, ylabel)
        for x, y, color, label in datasets:
            self.ax.plot(x, y, color=color, label=label, linewidth=1.5)
        if datasets:
            self.ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def set_cursor(self, x_val):
        if self.cursor_line is not None:
            self.cursor_line.remove()
            self.cursor_line = None
        if x_val is not None:
            self.cursor_line = self.ax.axvline(x=x_val, color="#ff7b72", linestyle="--", linewidth=1.2, alpha=0.8)
        self.canvas.draw()
