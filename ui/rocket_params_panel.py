"""
K2 AeroSim — Rocket Parameters Panel
=========================================
Left dock panel with editable fields for primary rocket configuration.
Changes are pushed live to the RocketStateEngine.
"""

import json
import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit,
    QDoubleSpinBox, QSpinBox, QComboBox, QGroupBox,
    QLabel, QScrollArea, QFrame, QHBoxLayout
)
from PyQt6.QtCore import Qt

logger = logging.getLogger("K2.RocketParams")


class RocketParamsPanel(QWidget):
    """
    Editable rocket parameter panel.
    All fields connect to the RocketStateEngine for live state updates.
    """
    
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._updating = False  # guard against feedback loops
        self._motors = self._load_motors()
        self._setup_ui()
        self._connect_signals()
        
        # Listen for external state changes (e.g., project load)
        self.engine.state_changed.connect(self._on_state_changed)
    
    def _load_motors(self) -> list:
        """Load motor database from JSON."""
        motors_path = Path(__file__).parent.parent / "data" / "motors.json"
        try:
            with open(motors_path, "r") as f:
                motors = json.load(f)
            logger.info(f"Loaded {len(motors)} motors from database")
            return motors
        except Exception as e:
            logger.error(f"Failed to load motors: {e}")
            return []
    
    def _setup_ui(self):
        """Build the parameter editing UI."""
        # Scroll area for when content exceeds panel height
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)
        
        # ═══ Identity Group ═══
        identity_group = QGroupBox("Identity")
        identity_layout = QFormLayout()
        identity_layout.setSpacing(8)
        
        self.name_edit = QLineEdit("Untitled Rocket")
        self.name_edit.setPlaceholderText("Enter rocket name...")
        identity_layout.addRow("Name:", self.name_edit)
        
        identity_group.setLayout(identity_layout)
        main_layout.addWidget(identity_group)
        
        # ═══ Geometry Group ═══
        geom_group = QGroupBox("Geometry")
        geom_layout = QFormLayout()
        geom_layout.setSpacing(8)
        
        self.length_spin = self._create_spin(0.1, 20.0, 1.0, " m", 3)
        geom_layout.addRow("Length:", self.length_spin)
        
        self.diameter_spin = self._create_spin(0.01, 2.0, 0.1, " m", 4)
        geom_layout.addRow("Diameter:", self.diameter_spin)
        
        self.fin_count_spin = QSpinBox()
        self.fin_count_spin.setRange(0, 8)
        self.fin_count_spin.setValue(4)
        self.fin_count_spin.setSuffix(" fins")
        geom_layout.addRow("Fins:", self.fin_count_spin)
        
        geom_group.setLayout(geom_layout)
        main_layout.addWidget(geom_group)
        
        # ═══ Mass Group ═══
        mass_group = QGroupBox("Mass Properties")
        mass_layout = QFormLayout()
        mass_layout.setSpacing(8)
        
        self.dry_mass_spin = self._create_spin(0.01, 500.0, 1.5, " kg", 3)
        mass_layout.addRow("Dry Mass:", self.dry_mass_spin)
        
        self.prop_mass_spin = self._create_spin(0.0, 100.0, 0.5, " kg", 4)
        mass_layout.addRow("Propellant:", self.prop_mass_spin)
        
        # Total mass display (read-only)
        self.total_mass_label = QLabel("2.000 kg")
        self.total_mass_label.setProperty("value", True)
        mass_layout.addRow("Total Mass:", self.total_mass_label)
        
        mass_group.setLayout(mass_layout)
        main_layout.addWidget(mass_group)
        
        # ═══ Motor Group ═══
        motor_group = QGroupBox("Motor Selection")
        motor_layout = QFormLayout()
        motor_layout.setSpacing(8)
        
        self.motor_combo = QComboBox()
        self.motor_combo.addItem("None (No Motor)")
        for motor in self._motors:
            label = f"{motor['designation']} — {motor['manufacturer']} ({motor['avg_thrust']:.0f}N avg)"
            self.motor_combo.addItem(label)
        motor_layout.addRow("Motor:", self.motor_combo)
        
        # Motor info display
        self.motor_info_label = QLabel("No motor selected")
        self.motor_info_label.setWordWrap(True)
        self.motor_info_label.setStyleSheet("color: #8b949e; font-size: 11px; padding: 4px;")
        motor_layout.addRow(self.motor_info_label)
        
        motor_group.setLayout(motor_layout)
        main_layout.addWidget(motor_group)
        
        main_layout.addStretch()
        
        scroll.setWidget(content)
        
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)
    
    def _create_spin(self, min_val, max_val, default, suffix, decimals):
        """Helper to create a configured QDoubleSpinBox."""
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSuffix(suffix)
        spin.setDecimals(decimals)
        spin.setSingleStep(10 ** (-decimals + 1))
        return spin
    
    def _connect_signals(self):
        """Connect all input widgets to state update handlers."""
        self.name_edit.textChanged.connect(self._push_state)
        self.length_spin.valueChanged.connect(self._push_state)
        self.diameter_spin.valueChanged.connect(self._push_state)
        self.fin_count_spin.valueChanged.connect(self._push_state)
        self.dry_mass_spin.valueChanged.connect(self._push_state)
        self.prop_mass_spin.valueChanged.connect(self._push_state)
        self.motor_combo.currentIndexChanged.connect(self._on_motor_changed)
    
    def _push_state(self):
        """Push all current field values to the RocketStateEngine."""
        if self._updating:
            return
        
        self.engine.update(
            name=self.name_edit.text(),
            length=self.length_spin.value(),
            diameter=self.diameter_spin.value(),
            fin_count=self.fin_count_spin.value(),
            dry_mass=self.dry_mass_spin.value(),
            propellant_mass=self.prop_mass_spin.value(),
        )
        
        # Update total mass display
        total = self.dry_mass_spin.value() + self.prop_mass_spin.value()
        self.total_mass_label.setText(f"{total:.3f} kg")
    
    def _on_motor_changed(self, index):
        """Handle motor selection change."""
        if self._updating:
            return
        
        if index == 0:
            # No motor
            self.engine.update(
                motor_designation="None",
                motor_avg_thrust=0.0,
                motor_max_thrust=0.0,
                motor_total_impulse=0.0,
                motor_burn_time=0.0,
                propellant_mass=0.0,
                propellant_mass_initial=0.0,
            )
            self.motor_info_label.setText("No motor selected")
            self.prop_mass_spin.setValue(0.0)
        else:
            motor = self._motors[index - 1]
            self.engine.update(
                motor_designation=motor["designation"],
                motor_avg_thrust=motor["avg_thrust"],
                motor_max_thrust=motor.get("max_thrust", motor["avg_thrust"] * 1.4),
                motor_total_impulse=motor["total_impulse"],
                motor_burn_time=motor["burn_time"],
                propellant_mass=motor["propellant_mass"],
                propellant_mass_initial=motor["propellant_mass"],
            )
            self.prop_mass_spin.setValue(motor["propellant_mass"])
            
            info = (
                f"Impulse: {motor['total_impulse']:.1f} N·s\n"
                f"Avg Thrust: {motor['avg_thrust']:.1f} N\n"
                f"Max Thrust: {motor['max_thrust']:.1f} N\n"
                f"Burn Time: {motor['burn_time']:.2f} s\n"
                f"Prop Mass: {motor['propellant_mass']*1000:.1f} g"
            )
            self.motor_info_label.setText(info)
    
    def _on_state_changed(self, state):
        """Update fields when state changes externally (e.g., project load)."""
        self._updating = True
        try:
            self.name_edit.setText(state.name)
            self.length_spin.setValue(state.length)
            self.diameter_spin.setValue(state.diameter)
            self.fin_count_spin.setValue(state.fin_count)
            self.dry_mass_spin.setValue(state.dry_mass)
            self.prop_mass_spin.setValue(state.propellant_mass)
            self.total_mass_label.setText(f"{state.total_mass():.3f} kg")
            
            # Update motor combo
            motor_idx = 0
            for i, motor in enumerate(self._motors):
                if motor["designation"] == state.motor_designation:
                    motor_idx = i + 1
                    break
            self.motor_combo.setCurrentIndex(motor_idx)
        finally:
            self._updating = False
