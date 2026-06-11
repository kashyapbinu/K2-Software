"""
K2 Aerospace — Component Editor
Dynamic property editor for the selected rocket component.
"""
import json, logging
from pathlib import Path
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit, QLabel, QScrollArea,
    QFrame, QPushButton, QCheckBox, QHBoxLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from ui.icons import icon
from core.components import (MATERIALS, NOSE_SHAPES, NOZZLE_TYPES, NoseCone, BodyTube,
    Transition, TrapezoidalFinSet, InnerTube, CenteringRing, Bulkhead,
    EngineBlock, Parachute, ShockCord, MassComponent, LaunchLug, RailButton, Stage, Nozzle)

logger = logging.getLogger("K2.CompEditor")


class ComponentEditor(QWidget):
    component_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._component = None
        self._updating = False
        self._presets = self._load_presets()
        self._setup_ui()

    def _load_presets(self):
        p = Path(__file__).parent.parent.parent / "data" / "component_presets.json"
        try:
            with open(p) as f:
                return json.load(f)
        except:
            return {}

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.header = QLabel("COMPONENT PROPERTIES")
        self.header.setStyleSheet("color: #58a6ff; font-weight: 700; font-size: 11px; "
            "letter-spacing: 1px; padding: 4px 8px;")
        layout.addWidget(self.header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.scroll)

        # Set initial empty state
        self.set_component(None)

    def set_component(self, component):
        self._component = component
        if component is None:
            empty_lbl = QLabel("Select a component to edit its properties")
            empty_lbl.setStyleSheet("color: #484f58; padding: 20px;")
            empty_lbl.setWordWrap(True)
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.scroll.setWidget(empty_lbl)
            return
        self._build_editor(component)

    def _build_editor(self, comp):
        self._updating = True
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Identity
        g1 = QGroupBox(comp.component_type)
        f1 = QFormLayout(); f1.setSpacing(6)

        name_edit = QLineEdit(comp.name)
        name_edit.textChanged.connect(lambda t: self._set_prop(comp, "name", t))
        f1.addRow("Name:", name_edit)

        if not isinstance(comp, Stage):
            mat_combo = QComboBox()
            for m in MATERIALS:
                mat_combo.addItem(m)
            mat_combo.setCurrentText(comp.material)
            mat_combo.currentTextChanged.connect(lambda t: self._set_prop(comp, "material", t))
            f1.addRow("Material:", mat_combo)

        g1.setLayout(f1)
        layout.addWidget(g1)

        # Presets button
        preset_key = self._get_preset_key(comp)
        if preset_key and preset_key in self._presets:
            preset_btn = QPushButton(icon("settings"), "Apply Preset...")
            preset_btn.setStyleSheet("""
                QPushButton { background: #161b22; border: 1px solid #30363d;
                    border-radius: 6px; padding: 8px; color: #58a6ff; font-weight: 600; }
                QPushButton:hover { border-color: #58a6ff; background: #21262d; }
            """)
            preset_btn.clicked.connect(lambda: self._show_presets(comp, preset_key))
            layout.addWidget(preset_btn)

        # Type-specific properties
        g2 = QGroupBox("Dimensions")
        f2 = QFormLayout(); f2.setSpacing(6)
        self._add_fields(f2, comp)
        g2.setLayout(f2)
        layout.addWidget(g2)

        # Mass info
        g3 = QGroupBox("Mass")
        f3 = QFormLayout(); f3.setSpacing(6)
        mass_lbl = QLabel(f"{comp.computed_mass()*1000:.2f} g")
        mass_lbl.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; font-weight: 600;")
        f3.addRow("Computed:", mass_lbl)
        self._mass_label = mass_lbl

        override = QCheckBox("Override mass")
        override.setChecked(comp.override_mass is not None)
        f3.addRow(override)

        if comp.override_mass is not None:
            ov_spin = self._spin(0, 10, comp.override_mass, " kg", 4)
            ov_spin.valueChanged.connect(lambda v: self._set_prop(comp, "override_mass", v))
            f3.addRow("Manual:", ov_spin)

        g3.setLayout(f3)
        layout.addWidget(g3)

        # Comment
        g4 = QGroupBox("Notes")
        f4 = QFormLayout()
        comment = QLineEdit(comp.comment)
        comment.setPlaceholderText("Optional notes...")
        comment.textChanged.connect(lambda t: self._set_prop(comp, "comment", t))
        f4.addRow(comment)
        g4.setLayout(f4)
        layout.addWidget(g4)

        layout.addStretch()
        self.scroll.setWidget(w)
        self._updating = False

    def _add_fields(self, form, comp):
        if isinstance(comp, NoseCone):
            self._combo(form, "Shape:", comp, "shape", NOSE_SHAPES)
            self._dim(form, "Length:", comp, "length", " m")
            self._dim(form, "Diameter:", comp, "diameter", " m")
            self._dim(form, "Wall:", comp, "wall_thickness", " m", 4, 0.0005, 0.05, 0.0005)
        elif isinstance(comp, BodyTube):
            self._dim(form, "Length:", comp, "length", " m", 3, 0.01, 5.0)
            self._dim(form, "Outer ⌀:", comp, "outer_diameter_val", " m", 4, 0.005, 1.0, 0.001)
            self._dim(form, "Inner ⌀:", comp, "inner_diameter", " m", 4, 0.005, 1.0, 0.001)
        elif isinstance(comp, Transition):
            self._combo(form, "Shape:", comp, "shape", ["Conical", "Ogive", "Elliptical"])
            self._dim(form, "Length:", comp, "length", " m")
            self._dim(form, "Fore ⌀:", comp, "fore_diameter", " m", 4, 0.005, 1.0, 0.001)
            self._dim(form, "Aft ⌀:", comp, "aft_diameter", " m", 4, 0.005, 1.0, 0.001)
            self._dim(form, "Wall:", comp, "wall_thickness", " m", 4, 0.0005, 0.05, 0.0005)
        elif isinstance(comp, TrapezoidalFinSet):
            s = QSpinBox(); s.setRange(1, 8); s.setValue(comp.fin_count)
            s.valueChanged.connect(lambda v: self._set_prop(comp, "fin_count", v))
            form.addRow("Fin Count:", s)
            self._dim(form, "Root Chord:", comp, "root_chord", " m")
            self._dim(form, "Tip Chord:", comp, "tip_chord", " m")
            self._dim(form, "Height:", comp, "height", " m")
            self._dim(form, "Sweep:", comp, "sweep_angle", "°", 1, 0, 80, 1)
            self._dim(form, "Thickness:", comp, "thickness", " m", 4, 0.001, 0.02, 0.0005)
        elif isinstance(comp, InnerTube):
            self._dim(form, "Length:", comp, "length", " m")
            self._dim(form, "Outer ⌀:", comp, "outer_diameter_val", " m", 4, 0.005, 0.5, 0.001)
            self._dim(form, "Inner ⌀:", comp, "inner_diameter", " m", 4, 0.005, 0.5, 0.001)
            cb = QCheckBox("Motor Mount")
            cb.setChecked(comp.is_motor_mount)
            cb.toggled.connect(lambda v: self._set_prop(comp, "is_motor_mount", v))
            form.addRow(cb)
        elif isinstance(comp, CenteringRing):
            self._dim(form, "Outer ⌀:", comp, "outer_diameter_val", " m", 4)
            self._dim(form, "Inner ⌀:", comp, "inner_diameter", " m", 4)
            self._dim(form, "Thickness:", comp, "thickness", " m", 4, 0.001, 0.02, 0.0005)
        elif isinstance(comp, (Bulkhead, EngineBlock)):
            self._dim(form, "Diameter:", comp, "diameter", " m", 4)
            self._dim(form, "Thickness:", comp, "thickness", " m", 4, 0.001, 0.02, 0.0005)
        elif isinstance(comp, Parachute):
            self._dim(form, "Diameter:", comp, "diameter", " m", 3, 0.1, 5.0, 0.05)
            self._dim(form, "CD:", comp, "cd", "", 2, 0.5, 3.0, 0.1)
            s = QSpinBox(); s.setRange(3, 24); s.setValue(comp.line_count)
            s.valueChanged.connect(lambda v: self._set_prop(comp, "line_count", v))
            form.addRow("Lines:", s)
            self._dim(form, "Line Len:", comp, "line_length", " m")
        elif isinstance(comp, ShockCord):
            self._dim(form, "Length:", comp, "length", " m", 2, 0.1, 5.0, 0.1)
        elif isinstance(comp, MassComponent):
            self._dim(form, "Mass:", comp, "mass", " kg", 4, 0.001, 10, 0.005)
            self._dim(form, "Length:", comp, "mass_length", " m")
        elif isinstance(comp, LaunchLug):
            self._dim(form, "Length:", comp, "length", " m")
            self._dim(form, "Outer ⌀:", comp, "outer_diameter_val", " m", 4)
            self._dim(form, "Inner ⌀:", comp, "inner_diameter", " m", 4)
        elif isinstance(comp, RailButton):
            self._dim(form, "Height:", comp, "height", " m", 4)
            self._dim(form, "Base ⌀:", comp, "base_diameter", " m", 4)
        elif isinstance(comp, Nozzle):
            self._combo(form, "Type:", comp, "nozzle_type", NOZZLE_TYPES)
            self._dim(form, "Length:", comp, "length", " m", 4, 0.01, 2.0, 0.005)
            self._dim(form, "Inlet ⌀:", comp, "inlet_diameter", " m", 4, 0.005, 1.0, 0.001)
            self._dim(form, "Throat ⌀:", comp, "throat_diameter", " m", 4, 0.005, 0.5, 0.001)
            self._dim(form, "Exit ⌀:", comp, "exit_diameter", " m", 4, 0.005, 1.0, 0.001)
            self._dim(form, "Half Angle:", comp, "half_angle", "°", 1, 5, 45, 1)
            self._dim(form, "Wall:", comp, "wall_thickness", " m", 4, 0.0005, 0.05, 0.0005)
            # Show expansion ratio (read-only)
            er_label = QLabel(f"{comp.expansion_ratio:.2f}")
            er_label.setStyleSheet("color: #e6edf3; font-family: 'Cascadia Code', monospace; font-weight: 600;")
            form.addRow("Expansion Ratio:", er_label)
            if comp.nozzle_type == "Full Propulsion":
                self._dim(form, "Chamber P:", comp, "design_chamber_pressure", " Pa", 0, 1e5, 5e7, 1e5)
                self._dim(form, "Exit P:", comp, "design_exit_pressure", " Pa", 0, 1e3, 1e6, 1e3)
        elif isinstance(comp, Stage):
            self._dim(form, "Sep Delay:", comp, "separation_delay", " s", 2, 0, 30, 0.5)
            self._combo(form, "Sep Event:", comp, "separation_event",
                        ["burnout", "apogee", "timer", "manual"])

    def _dim(self, form, label, comp, attr, suffix=" m", dec=3, mn=0.001, mx=5.0, step=0.01):
        s = self._spin(mn, mx, getattr(comp, attr), suffix, dec, step)
        s.valueChanged.connect(lambda v: self._set_prop(comp, attr, v))
        form.addRow(label, s)

    def _combo(self, form, label, comp, attr, options):
        c = QComboBox()
        for o in options:
            c.addItem(o)
        c.setCurrentText(str(getattr(comp, attr)))
        c.currentTextChanged.connect(lambda t: self._set_prop(comp, attr, t))
        form.addRow(label, c)

    def _spin(self, mn, mx, val, suffix, dec, step=0.01):
        s = QDoubleSpinBox()
        s.setRange(mn, mx); s.setValue(val); s.setSuffix(suffix)
        s.setDecimals(dec); s.setSingleStep(step)
        return s

    def _set_prop(self, comp, attr, value):
        if self._updating:
            return
        setattr(comp, attr, value)
        if hasattr(self, '_mass_label'):
            self._mass_label.setText(f"{comp.computed_mass()*1000:.2f} g")
        self.component_changed.emit()

    def _get_preset_key(self, comp):
        mapping = {
            NoseCone: "nose_cones", BodyTube: "body_tubes", Transition: "transitions",
            TrapezoidalFinSet: "fin_sets", InnerTube: "inner_tubes",
            CenteringRing: "centering_rings", Parachute: "parachutes",
        }
        return mapping.get(type(comp))

    def _show_presets(self, comp, key):
        from ui.widgets.preset_dialog import PresetDialog
        presets = self._presets.get(key, [])
        dlg = PresetDialog(presets, comp.component_type, self)
        if dlg.exec():
            preset = dlg.selected_preset()
            if preset:
                self._apply_preset(comp, preset)

    def _apply_preset(self, comp, preset):
        self._updating = True
        for k, v in preset.items():
            if k in ("manufacturer", "part", "name"):
                if k == "name":
                    comp.name = v
                continue
            if hasattr(comp, k):
                setattr(comp, k, v)
        self._build_editor(comp)
        self._updating = False
        self.component_changed.emit()
