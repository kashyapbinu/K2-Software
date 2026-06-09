"""
K2 Aerospace — Component Palette
Grid of buttons to add rocket components.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from core.components import (NoseCone, BodyTube, Transition, TrapezoidalFinSet,
    InnerTube, CenteringRing, Bulkhead, EngineBlock, Parachute,
    ShockCord, MassComponent, LaunchLug, RailButton, Stage, Nozzle)

logger = logging.getLogger("K2.Palette")

COMPONENT_DEFS = [
    ("Body Components", [
        ("▲\nNose Cone", NoseCone),
        ("▬\nBody Tube", BodyTube),
        ("◇\nTransition", Transition),
    ]),
    ("Fin Sets", [
        ("✦\nTrapezoidal", TrapezoidalFinSet),
    ]),
    ("Propulsion", [
        ("🔥\nNozzle", Nozzle),
    ]),
    ("Inner Components", [
        ("◎\nInner Tube", InnerTube),
        ("◉\nCentering\nRing", CenteringRing),
        ("▣\nBulkhead", Bulkhead),
        ("▪\nEngine\nBlock", EngineBlock),
    ]),
    ("Recovery", [
        ("🪂\nParachute", Parachute),
        ("〰\nShock Cord", ShockCord),
    ]),
    ("Mass / Attach", [
        ("●\nMass\nComponent", MassComponent),
        ("▫\nLaunch\nLug", LaunchLug),
        ("▪\nRail\nButton", RailButton),
    ]),
]


class ComponentPalette(QWidget):
    add_component = pyqtSignal(object)  # emits component class

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("ADD NEW COMPONENT")
        header.setStyleSheet("color: #58a6ff; font-weight: 700; font-size: 11px; "
            "letter-spacing: 1px; padding: 4px 8px;")
        layout.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMaximumHeight(220)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(8)

        for category_name, components in COMPONENT_DEFS:
            cat_label = QLabel(category_name)
            cat_label.setStyleSheet("color: #8b949e; font-weight: 600; font-size: 10px; "
                "padding: 2px 4px;")
            cl.addWidget(cat_label)

            row = QHBoxLayout()
            row.setSpacing(4)
            for label, comp_class in components:
                btn = QPushButton(label)
                btn.setFixedSize(78, 58)
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #161b22; border: 1px solid #30363d;
                        border-radius: 6px; color: #c9d1d9; font-size: 9px;
                        padding: 4px; text-align: center;
                    }
                    QPushButton:hover {
                        background-color: #21262d; border-color: #58a6ff; color: #58a6ff;
                    }
                    QPushButton:pressed { background-color: #1f6feb; color: white; }
                """)
                btn.clicked.connect(lambda checked, c=comp_class: self.add_component.emit(c))
                row.addWidget(btn)
            row.addStretch()
            cl.addLayout(row)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Add stage button
        stage_btn = QPushButton("+ Add New Stage (Booster)")
        stage_btn.setStyleSheet("""
            QPushButton {
                background-color: #161b22; border: 1px solid #30363d;
                border-radius: 6px; color: #d29922; font-weight: 600;
                padding: 6px; font-size: 11px;
            }
            QPushButton:hover { border-color: #d29922; background-color: #21262d; }
        """)
        stage_btn.clicked.connect(lambda: self.add_component.emit(Stage))
        layout.addWidget(stage_btn)
