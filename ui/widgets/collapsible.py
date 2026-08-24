"""
Collapsible section card.

A titled panel that folds away when its header is clicked, for stacking
several read-out groups in a narrow side panel without scrolling.

    sec = CollapsibleSection("Stability (static)")
    sec.set_content_layout(form_layout)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QFrame, QSizePolicy, QLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

from ui import theme

def _header_qss() -> str:
    return f"""
QToolButton {{
    background-color: {theme.PANEL};
    border: none;
    border-bottom: 1px solid {theme.LINE};
    color: {theme.TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 8px 10px;
    text-align: left;
}}
QToolButton:hover {{ color: {theme.TEXT_BRIGHT}; }}
QToolButton::indicator {{ width: 0px; }}
"""


class CollapsibleSection(QWidget):
    """Card with a clickable header that folds its body away."""

    toggled = pyqtSignal(bool)

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")
        self.setStyleSheet(
            f"#CollapsibleSection {{ background-color: {theme.PANEL};"
            f" border: 1px solid {theme.LINE}; border-radius: 6px; }}"
        )

        self._header = QToolButton()
        self._header.setText(title.upper())
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setStyleSheet(_header_qss())
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.clicked.connect(self._on_clicked)

        self._body = QFrame()
        self._body.setStyleSheet("background: transparent;")
        self._body.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header)
        outer.addWidget(self._body)

    # -- content ----------------------------------------------------------
    def set_content_layout(self, layout: QLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        self._body.setLayout(layout)

    def set_content_widget(self, widget: QWidget):
        lo = QVBoxLayout()
        lo.setContentsMargins(10, 8, 10, 10)
        lo.addWidget(widget)
        self.set_content_layout(lo)

    def body(self) -> QFrame:
        return self._body

    # -- state ------------------------------------------------------------
    def _on_clicked(self, checked: bool):
        self.set_expanded(checked)
        self.toggled.emit(checked)

    def set_expanded(self, expanded: bool):
        self._header.setChecked(expanded)
        self._header.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._body.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self._header.isChecked()
