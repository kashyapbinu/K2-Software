from ui import theme
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
        self.figure.patch.set_facecolor(theme.BG)
        self.canvas = FigureCanvas(self.figure)
        # Let wheel events bubble to an enclosing QScrollArea instead of being
        # swallowed by the canvas — keeps scroll panels smooth over the plot.
        self.canvas.wheelEvent = lambda e: e.ignore()
        self.ax = self.figure.add_subplot(111)
        self.cursor_line = None
        self._style_axis(title, xlabel, ylabel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def _style_axis(self, title, xlabel, ylabel):
        ax = self.ax
        ax.set_facecolor(theme.PANEL)
        ax.set_title(title, color=theme.TEXT, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color=theme.TEXT_DIM, fontsize=10)
        ax.set_ylabel(ylabel, color=theme.TEXT_DIM, fontsize=10)
        ax.tick_params(colors=theme.LINE_STRONG, labelsize=9)
        ax.spines["bottom"].set_color(theme.LINE)
        ax.spines["left"].set_color(theme.LINE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, color=theme.LINE)

    def retheme(self):
        """Re-apply palette colours after a theme switch."""
        self.figure.patch.set_facecolor(theme.BG)
        self._style_axis(self.ax.get_title(), self.ax.get_xlabel(), self.ax.get_ylabel())
        self.canvas.draw_idle()

    def clear(self):
        self.ax.clear()
        self.cursor_line = None
        self._style_axis(self.ax.get_title(), self.ax.get_xlabel(), self.ax.get_ylabel())

    def plot(self, x, y, color=None, label=None, linewidth=1.5):
        color = color or theme.ACCENT
        self.ax.plot(x, y, color=color, label=label, linewidth=linewidth)
        if label:
            self.ax.legend(facecolor=theme.PANEL, edgecolor=theme.LINE, labelcolor=theme.TEXT, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def update_plot(self, x, y, title="", xlabel="", ylabel="", color=None,
                    linestyle="-", fill=True, note=""):
        """Draw a single series. `note` prints a caption under the title -
        use it to say where the data came from."""
        color = color or theme.ACCENT
        self.ax.clear()
        # ax.clear() detaches every artist, the scrub cursor included. Dropping
        # the reference here is what keeps set_cursor from later calling
        # .remove() on an artist that no longer belongs to an axes.
        self.cursor_line = None
        self._style_axis(title, xlabel, ylabel)
        self.ax.plot(x, y, color=color, linewidth=1.5, linestyle=linestyle)
        if fill:
            self.ax.fill_between(x, y, alpha=0.1, color=color)
        if note:
            self.ax.text(0.5, 1.005, note, transform=self.ax.transAxes,
                         ha="center", va="bottom", fontsize=8, color=theme.TEXT_DIM)
        self.figure.tight_layout()
        self.canvas.draw()

    def multi_plot(self, datasets, title="", xlabel="", ylabel=""):
        """datasets: list of (x, y, color, label) tuples"""
        self.ax.clear()
        self.cursor_line = None          # see update_plot
        self._style_axis(title, xlabel, ylabel)
        for x, y, color, label in datasets:
            self.ax.plot(x, y, color=color, label=label, linewidth=1.5)
        if datasets:
            self.ax.legend(facecolor=theme.PANEL, edgecolor=theme.LINE, labelcolor=theme.TEXT, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def set_cursor(self, x_val):
        """Move the scrub cursor to ``x_val``; None hides it.

        The removal is guarded because the cursor artist can be detached by
        anything that clears the axes, and a Matplotlib artist that is no
        longer attached raises rather than ignoring the call:

            matplotlib/artist.py, in remove
            NotImplementedError: cannot remove artist

        That is what a replot used to leave behind. It mattered out of
        proportion to its size: results_workspace._on_scrub sets the cursor on
        eight plots in a row and then fills the readout fields, so the first
        stale artist aborted the whole handler -- the other seven cursors never
        moved and the readouts never updated. Scrubbing back through time
        looked like "no data at this time" and logged 750 exceptions.

        Both fixes are kept deliberately. Clearing the reference at every
        ax.clear() is the correct one; this guard is what stops the next
        clear-site that forgets from breaking the whole panel again.
        """
        if self.cursor_line is not None:
            try:
                if self.cursor_line.axes is not None:
                    self.cursor_line.remove()
            except (NotImplementedError, ValueError, AttributeError):
                pass
            self.cursor_line = None
        if x_val is not None:
            self.cursor_line = self.ax.axvline(
                x=x_val, color=theme.ERR, linestyle="--", linewidth=1.2,
                alpha=0.8)
        self.canvas.draw()
