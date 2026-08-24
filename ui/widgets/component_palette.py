"""
K2 AeroSim — Component Palette
Grid of buttons to add rocket components.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QScrollArea, QFrame, QToolButton)
from PyQt6.QtCore import Qt, pyqtSignal, QSize

from ui.icons import icon
from ui import theme
from core.components import (NoseCone, BodyTube, Transition, TrapezoidalFinSet,
    InnerTube, CenteringRing, Bulkhead, EngineBlock, Parachute,
    ShockCord, MassComponent, LaunchLug, RailButton, Stage, Nozzle)

logger = logging.getLogger("K2.Palette")

# (label, icon key, component class)
COMPONENT_DEFS = [
    ("Body Components", [
        ("Nose Cone", "comp_nosecone", NoseCone),
        ("Body Tube", "comp_bodytube", BodyTube),
        ("Transition", "comp_transition", Transition),
    ]),
    ("Fin Sets", [
        ("Trapezoidal", "comp_finset", TrapezoidalFinSet),
    ]),
    ("Propulsion", [
        ("Nozzle", "comp_nozzle", Nozzle),
    ]),
    ("Inner Components", [
        ("Inner Tube", "comp_innertube", InnerTube),
        ("Centering Ring", "comp_ring", CenteringRing),
        ("Bulkhead", "comp_bulkhead", Bulkhead),
        ("Engine Block", "comp_block", EngineBlock),
    ]),
    ("Recovery", [
        ("Parachute", "comp_parachute", Parachute),
        ("Shock Cord", "comp_cord", ShockCord),
    ]),
    ("Mass / Attach", [
        ("Mass Component", "comp_mass", MassComponent),
        ("Launch Lug", "comp_lug", LaunchLug),
        ("Rail Button", "comp_railbutton", RailButton),
    ]),
]

def card_qss() -> str:
    return f"""
QToolButton {{
    background-color: {theme.RAISED};
    border: 1px solid {theme.LINE};
    border-radius: 6px;
    color: {theme.TEXT};
    font-size: 11px;
    padding: 8px 4px;
}}
QToolButton:hover {{
    border-color: {theme.ACCENT};
    color: {theme.TEXT_BRIGHT};
}}
QToolButton:pressed {{ background-color: {theme.LINE}; }}
"""


class ComponentPalette(QWidget):
    add_component = pyqtSignal(object)  # emits component class

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def retheme(self):
        from PyQt6.QtWidgets import QToolButton
        for btn in self.findChildren(QToolButton):
            btn.setStyleSheet(card_qss())

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("ADD COMPONENT")
        header.setProperty("panelTitle", True)
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(268)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(8)

        for category_name, components in COMPONENT_DEFS:
            cat_label = QLabel(category_name)
            cat_label.setStyleSheet(
                f"color: {theme.TEXT_DIM}; font-size: 11px; padding: 4px 4px 2px 4px;")
            cl.addWidget(cat_label)

            row = QHBoxLayout()
            row.setSpacing(4)
            for label, icon_key, comp_class in components:
                btn = QToolButton()
                btn.setText(label)
                btn.setIcon(icon(icon_key, color=theme.TEXT_DIM))
                btn.setIconSize(QSize(20, 20))
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
                btn.setFixedSize(108, 68)
                btn.setStyleSheet(card_qss())
                btn.setToolTip(f"Add {label}")
                btn.clicked.connect(lambda checked, c=comp_class: self.add_component.emit(c))
                row.addWidget(btn)
            row.addStretch()
            cl.addLayout(row)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Add stage button
        stage_btn = QPushButton("  Add Stage (Booster)")
        stage_btn.setIcon(icon("comp_stage", color=theme.ACCENT))
        stage_btn.setProperty("primary", True)
        stage_btn.clicked.connect(lambda: self.add_component.emit(Stage))
        layout.addWidget(stage_btn)
