"""
K2 Aerospace — Design Workspace (Component Builder)
OpenRocket-style component-based rocket designer with multi-stage support.
Includes real-time stability analysis (CP, CG, margin).
"""
import logging
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QLabel, QGroupBox, QFormLayout, QFrame)
from PyQt6.QtCore import Qt
from core.components import (RocketAssembly, Stage, NoseCone, BodyTube,
    Transition, TrapezoidalFinSet, InnerTube, CenteringRing, Bulkhead,
    EngineBlock, Parachute, ShockCord, MassComponent, LaunchLug, RailButton)
from ui.widgets.component_tree import ComponentTree
from ui.widgets.component_palette import ComponentPalette
from ui.widgets.component_editor import ComponentEditor
from ui.properties_panel import PropertiesPanel
from visualization.viewer_3d import Viewer3D

logger = logging.getLogger("K2.DesignWS")

_STAB_VAL_SS = (
    "color: #e6edf3; font-family: 'Cascadia Code', monospace; font-size: 13px; "
    "font-weight: 600; padding: 2px 4px; background-color: #161b22; border-radius: 4px;"
)

# Components that must be children of a BodyTube
INNER_TYPES = (InnerTube, CenteringRing, Bulkhead, EngineBlock)


class DesignWorkspace(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.assembly = RocketAssembly()

        # Store assembly on engine for other workspaces
        self.engine._assembly = self.assembly

        self._setup_ui()
        self._connect_signals()
        self._sync_to_engine()
        self.engine.state_changed.connect(self._refresh_stability)

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ═══ LEFT: Tree + Actions ═══
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(320)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(8)

        self.comp_tree = ComponentTree(self.assembly)
        ll.addWidget(self.comp_tree, 1)

        splitter.addWidget(left)

        # ═══ CENTER: Palette + 3D Viewer ═══
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)

        self.palette = ComponentPalette()
        cl.addWidget(self.palette)

        self.viewer = Viewer3D(self.engine, self)
        cl.addWidget(self.viewer, 1)

        splitter.addWidget(center)

        # ═══ RIGHT: Editor + Stability + Properties ═══
        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(380)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(4)

        self.editor = ComponentEditor()
        rl.addWidget(self.editor, 1)

        # ── Stability Analysis panel ──
        stab_grp = QGroupBox("Stability Analysis")
        stab_grp.setStyleSheet(
            "QGroupBox { font-weight: 700; font-size: 12px; color: #c9d1d9; "
            "border: 1px solid #30363d; border-radius: 6px; margin-top: 6px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
        )
        sf = QFormLayout()
        sf.setSpacing(4)
        sf.setContentsMargins(8, 4, 8, 8)

        self._stab_cp = QLabel("—")
        self._stab_cp.setStyleSheet(_STAB_VAL_SS)
        sf.addRow("CP:", self._stab_cp)

        self._stab_cg = QLabel("—")
        self._stab_cg.setStyleSheet(_STAB_VAL_SS)
        sf.addRow("CG:", self._stab_cg)

        self._stab_margin = QLabel("—")
        self._stab_margin.setStyleSheet(_STAB_VAL_SS)
        sf.addRow("Margin:", self._stab_margin)

        self._stab_status = QLabel("—")
        self._stab_status.setStyleSheet("font-weight: 600; font-size: 13px;")
        sf.addRow("Status:", self._stab_status)

        stab_grp.setLayout(sf)
        stab_grp.setMaximumHeight(140)
        rl.addWidget(stab_grp)

        self.properties = PropertiesPanel(self.engine)
        self.properties.setMaximumHeight(300)
        rl.addWidget(self.properties)

        splitter.addWidget(right)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 700, 300])

        layout.addWidget(splitter)

    def _connect_signals(self):
        self.palette.add_component.connect(self._on_add_component)
        self.comp_tree.component_selected.connect(self._on_component_selected)
        self.comp_tree.tree_changed.connect(self._on_tree_changed)
        self.editor.component_changed.connect(self._on_component_edited)

    def _on_add_component(self, comp_class):
        """Add a new component of the given class."""
        if comp_class == Stage:
            stage_num = len(self.assembly.stages) + 1
            self.assembly.add_stage(f"Booster {stage_num - 1}" if stage_num > 1 else "Sustainer")
            self.comp_tree.rebuild()
            self._sync_to_engine()
            self.engine.log_message.emit(f"Added new stage")
            return

        comp = comp_class()

        # Find the right parent
        selected = self.comp_tree.selected_component()

        if isinstance(comp, INNER_TYPES):
            # Inner components go inside a BodyTube
            parent = self._find_body_tube(selected)
            if parent is None:
                self.engine.log_message.emit(
                    f"⚠ {comp.component_type} must be placed inside a Body Tube")
                return
        else:
            # Body-level components go under the active stage
            parent = self._find_stage(selected)

        self.assembly.add_component(parent, comp)
        self.comp_tree.rebuild()
        self._sync_to_engine()
        self.engine.log_message.emit(f"Added {comp.component_type}: {comp.name}")

    def _find_stage(self, selected):
        """Walk up from selected to find the owning Stage."""
        if selected is None:
            return self.assembly.stages[0] if self.assembly.stages else None
        if isinstance(selected, Stage):
            return selected
        comp = selected
        while comp.parent:
            if isinstance(comp.parent, Stage):
                return comp.parent
            comp = comp.parent
        return self.assembly.stages[0] if self.assembly.stages else None

    def _find_body_tube(self, selected):
        """Find a BodyTube parent for inner components."""
        if isinstance(selected, BodyTube):
            return selected
        if selected and selected.parent and isinstance(selected.parent, BodyTube):
            return selected.parent
        # Find the last body tube in the active stage
        stage = self._find_stage(selected)
        if stage:
            for c in reversed(stage.children):
                if isinstance(c, BodyTube):
                    return c
        return None

    def _on_component_selected(self, component):
        self.editor.set_component(component)
        self.viewer.select_component(component)

    def _on_tree_changed(self):
        self._sync_to_engine()

    def _on_component_edited(self):
        self.comp_tree.rebuild()
        self._sync_to_engine()

    def _sync_to_engine(self):
        """Push assembly-computed values to the RocketStateEngine."""
        asm = self.assembly
        asm._recompute_positions()

        # Turn off generic length-based estimation since we have real components
        self.engine.auto_estimate_properties = False

        # Find first fin set position for the dynamic AeroModel
        fin_pos = 0.0
        for c in asm.all_components():
            from core.components import TrapezoidalFinSet
            if isinstance(c, TrapezoidalFinSet):
                fin_pos = c._position
                break

        # Find motor position (first inner tube marked as motor mount)
        motor_pos = asm.total_length() * 0.85
        for c in reversed(list(asm.all_components())):
            if getattr(c, 'is_motor_mount', False):
                motor_pos = c._position + c.component_length() / 2.0
                break

        self.engine.update(
            name=asm.name,
            length=asm.total_length(),
            diameter=asm.max_diameter(),
            fin_count=asm.fin_count(),
            fin_position=fin_pos,
            motor_position=motor_pos,
            dry_mass=asm.total_mass(),
            cg=asm.compute_cg(),
            dry_cg=asm.compute_cg(),
            cp=asm.compute_cp(),
        )

    def reset_camera(self):
        self.viewer.reset_camera()

    def get_assembly(self):
        return self.assembly

    def set_assembly(self, asm):
        # Block signals to prevent "signal storm" during import
        self.comp_tree.blockSignals(True)
        self.engine.blockSignals(True)
        try:
            self.assembly = asm
            self.engine._assembly = asm
            self.comp_tree.assembly = asm
            self.comp_tree.rebuild()
            
            # Clear editor to prevent ghost components from previous assemblies
            self.editor.set_component(None)
            self._sync_to_engine()
        finally:
            self.comp_tree.blockSignals(False)
            self.engine.blockSignals(False)
        # Re-emit state so properties panel, status bar, etc. get the update
        self.engine.state_changed.emit(self.engine.state)
        # Force one final 3D redraw
        self.viewer._rebuild(reset_cam=True)

    def _refresh_stability(self, state=None):
        """Update stability readouts from current engine state."""
        s = state or self.engine.state
        self._stab_cp.setText(f"{s.cp:.3f} m")
        self._stab_cg.setText(f"{s.cg:.3f} m")
        self._stab_margin.setText(f"{s.stability_margin:.2f} cal")

        if s.stability_margin < 0.5:
            self._stab_status.setText("⚠ UNSTABLE")
            self._stab_status.setStyleSheet("color: #f85149; font-weight: 600; font-size: 13px;")
        elif s.stability_margin < 1.0:
            self._stab_status.setText("⚡ MARGINAL")
            self._stab_status.setStyleSheet("color: #d29922; font-weight: 600; font-size: 13px;")
        elif s.stability_margin <= 2.5:
            self._stab_status.setText("✓ STABLE")
            self._stab_status.setStyleSheet("color: #7ee787; font-weight: 600; font-size: 13px;")
        else:
            self._stab_status.setText("⚠ OVERSTABLE")
            self._stab_status.setStyleSheet("color: #d29922; font-weight: 600; font-size: 13px;")

    def closeEvent(self, event):
        self.viewer.closeEvent(event)
        super().closeEvent(event)
