from ui import theme
"""
K2 AeroSim — Shared dark-themed widgets for the engine/motor design dialogs.
============================================================================
Reusable building blocks used by both the Liquid Engine Designer and the
Custom (solid) Motor Builder so they share one look & feel:

  * CollapsibleBox — toggle-able section (no QToolBox exists in the codebase).
  * MetricGrid     — two-column key/value grid, values updated by name.
  * MplCanvas      — dark matplotlib canvas with an optional zoom/pan toolbar.
"""

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QGridLayout, QLabel,
    QToolButton)
from PyQt6.QtCore import Qt

# Shared palette (matches ui/styles.py)
# Read through ui.theme at draw time - a module-level snapshot would freeze
# whichever palette happened to be active when this module was imported.


class CollapsibleBox(QWidget):
    """Lightweight dark-themed collapsible section."""

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        self.toggle = QToolButton(text=title, checkable=True, checked=expanded)
        self.toggle.setStyleSheet(
            f"QToolButton {{ background:{theme.PANEL}; color:{theme.TEXT_BRIGHT}; border:1px solid {theme.LINE};"
            " border-radius:6px; padding:6px 8px; font-weight:600; text-align:left; }"
            f"QToolButton:hover {{ background:{theme.RAISED}; }}")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.clicked.connect(self._on_toggle)

        self.content = QWidget()
        self.content.setVisible(expanded)
        self._content_layout = QVBoxLayout(self.content)
        self._content_layout.setContentsMargins(8, 4, 4, 8)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

    def _on_toggle(self, checked):
        self.content.setVisible(checked)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

    def add_widget(self, w):
        self._content_layout.addWidget(w)


class MetricGrid(QWidget):
    """Two-column key/value grid; values updated by name."""

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.labels = {}
        g = QGridLayout(self)
        g.setContentsMargins(2, 2, 2, 2)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(4)
        for i, name in enumerate(names):
            r, c = divmod(i, 2)
            k = QLabel(name + ":")
            k.setStyleSheet(f"color:{theme.TEXT_DIM};")
            v = QLabel("—")
            v.setStyleSheet(f"color:{theme.ACCENT}; font-weight:600;")
            g.addWidget(k, r, c * 2)
            g.addWidget(v, r, c * 2 + 1)
            self.labels[name] = v

    def set(self, name, text, color=None):
        color = color or theme.ACCENT
        if name in self.labels:
            self.labels[name].setText(text)
            self.labels[name].setStyleSheet(f"color:{color}; font-weight:600;")


class MplCanvas(QWidget):
    """Dark matplotlib canvas with a zoom/pan toolbar (interactive)."""

    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.patch.set_facecolor(theme.BG)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            tb = NavToolbar(self.canvas, self)
            tb.setStyleSheet(f"background:{theme.PANEL}; color:{theme.TEXT};")
            lay.addWidget(tb)
        lay.addWidget(self.canvas)

    def style_ax(self, title="", xlabel="", ylabel=""):
        ax = self.ax
        ax.set_facecolor(theme.PANEL)
        ax.set_title(title, color=theme.ACCENT, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(xlabel, color=theme.TEXT_DIM, fontsize=10)
        ax.set_ylabel(ylabel, color=theme.TEXT_DIM, fontsize=10)
        ax.tick_params(colors=theme.LINE_STRONG, labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color(theme.LINE)
        ax.grid(True, alpha=0.15, color=theme.LINE)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
