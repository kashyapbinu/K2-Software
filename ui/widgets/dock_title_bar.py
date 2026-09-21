"""
Custom QDockWidget title bar with legible float/close buttons.

Qt's stock dock buttons are 10 px pixmaps that all but vanish on a dark
panel. This bar draws the caption plus proper 22 px icon buttons from the
shared icon set, and re-themes itself with the rest of the UI (``retheme``
is picked up by ``theme.restyle_widgets``).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QToolButton, QDockWidget

from ui import theme
from ui.icons import icon

_BTN = 22
_ICON = 13


class DockTitleBar(QWidget):
    def __init__(self, dock: QDockWidget, title: str = None):
        super().__init__(dock)
        self.dock = dock
        self.setObjectName("dockTitleBar")
        # Plain QWidget subclasses skip stylesheet backgrounds unless asked.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(30)

        lo = QHBoxLayout(self)
        lo.setContentsMargins(10, 0, 4, 0)
        lo.setSpacing(2)

        self.lbl = QLabel(title if title is not None else dock.windowTitle())
        self.lbl.setObjectName("dockTitle")
        lo.addWidget(self.lbl, 1)

        feats = dock.features()
        self.btn_float = self._button("Float / dock (double-click title also works)")
        self.btn_float.clicked.connect(self._toggle_float)
        self.btn_float.setVisible(bool(feats & QDockWidget.DockWidgetFeature.DockWidgetFloatable))
        lo.addWidget(self.btn_float)

        self.btn_close = self._button("Hide panel")
        self.btn_close.clicked.connect(dock.close)
        self.btn_close.setVisible(bool(feats & QDockWidget.DockWidgetFeature.DockWidgetClosable))
        lo.addWidget(self.btn_close)

        dock.topLevelChanged.connect(lambda _f: self._refresh_icons())
        dock.windowTitleChanged.connect(self.lbl.setText)
        self.retheme()

    @staticmethod
    def _button(tip: str) -> QToolButton:
        b = QToolButton()
        b.setObjectName("dockBtn")
        b.setAutoRaise(True)
        b.setFixedSize(_BTN, _BTN)
        b.setIconSize(QSize(_ICON, _ICON))
        b.setToolTip(tip)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return b

    def _toggle_float(self):
        self.dock.setFloating(not self.dock.isFloating())

    def _refresh_icons(self):
        c = theme.TEXT
        self.btn_float.setIcon(icon("dock_in" if self.dock.isFloating() else "dock_out", color=c))
        self.btn_float.setToolTip("Dock panel" if self.dock.isFloating() else "Float panel")
        self.btn_close.setIcon(icon("close", color=c))

    def retheme(self):
        self.setStyleSheet(f"""
            #dockTitleBar {{
                background-color: {theme.PANEL};
                border-top: 1px solid {theme.LINE};
                border-bottom: 1px solid {theme.LINE};
            }}
            #dockTitle {{ color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600;
                          letter-spacing: 0.4px; background: transparent; border: none; }}
            #dockBtn {{ background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 0; }}
            #dockBtn:hover {{ background-color: {theme.RAISED}; border-color: {theme.LINE_STRONG}; }}
            #dockBtn:pressed {{ background-color: {theme.SUNKEN}; }}
        """)
        self._refresh_icons()


def install(dock: QDockWidget, title: str = None) -> DockTitleBar:
    bar = DockTitleBar(dock, title)
    dock.setTitleBarWidget(bar)
    return bar
