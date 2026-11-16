"""
K2 AeroSim — Readout tile
===========================
One quantity: a small caption over its value, with the unit set smaller.

The Simulation and Results tabs each had their own version: a caption label
in one grid row and a bordered value box in the next. In Results the grid's
frame stylesheet had no selector, so its border was drawn round every caption
as well; in Simulation the value boxes stretched to fill the tab. Both now
use this one widget.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from ui import theme


class Readout(QFrame):
    """Caption + value tile. ``setText("758 m")`` shows a large 758 and a small m."""

    def __init__(self, caption: str = "", compact: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("readout")
        self._compact = compact
        self._color = None
        self._text = "—"
        self.setFixedHeight(48 if compact else 58)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 5 if compact else 7, 10, 5 if compact else 7)
        lay.setSpacing(1 if compact else 2)
        self._caption = QLabel(caption)
        self._value = QLabel("—")
        self._value.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self._caption)
        lay.addWidget(self._value)
        self.retheme()

    def retheme(self):
        """Rebuild the styles from the active palette."""
        self.setStyleSheet(
            f"QFrame#readout {{ background-color: {theme.PANEL}; "
            f"border: 1px solid {theme.LINE}; border-radius: 3px; }}")
        self._caption.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 11px; background: transparent; border: none;")
        self._value.setStyleSheet(
            f"color: {self._color or theme.TEXT_BRIGHT}; font-family: {theme.MONO}; "
            f"font-size: {14 if self._compact else 18}px; font-weight: 600; "
            "background: transparent; border: none;")
        self.setText(self._text)

    def setValueColor(self, color: str = None):
        """Colour the value (a status colour); None returns it to plain text."""
        self._color = color
        self.retheme()

    def text(self) -> str:
        """The text last given to :meth:`setText`, unit included."""
        return self._text

    def setText(self, text: str):
        self._text = text
        number, sep, unit = text.rpartition(" ")
        # Only a short trailing token is a unit: "Main Descent" and
        # "12.3 m/s² (4.1 g)" are shown whole.
        if (not sep or not number.strip() or len(unit) > 5
                or unit[:1].isdigit() or unit.endswith(")")):
            number, unit = text, ""
        html = number
        if unit:
            html += (f' <span style="font-size:11px; font-weight:400; '
                     f'color:{theme.TEXT_DIM};">{unit}</span>')
        self._value.setText(html)
