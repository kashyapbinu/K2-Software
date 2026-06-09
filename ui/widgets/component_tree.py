"""
K2 Aerospace — Component Tree Widget
Hierarchical tree view showing rocket structure.
"""
import logging
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem,
    QTreeWidgetItemIterator, QPushButton, QHBoxLayout, QGroupBox, QMenu, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QBrush, QFont
from core.components import RocketComponent, Stage

logger = logging.getLogger("K2.CompTree")

ICONS = {
    "Stage": "📦", "Nose Cone": "▲", "Body Tube": "▬", "Transition": "◇",
    "Trapezoidal Fins": "✦", "Inner Tube": "◎", "Centering Ring": "◉",
    "Bulkhead": "▣", "Engine Block": "▪", "Parachute": "🪂",
    "Shock Cord": "〰", "Mass Component": "●", "Launch Lug": "▫", "Rail Button": "▪",
    "Nozzle": "🔥",
}


class ComponentTree(QWidget):
    component_selected = pyqtSignal(object)  # emits RocketComponent or None
    tree_changed = pyqtSignal()  # emits when structure changes

    def __init__(self, assembly, parent=None):
        super().__init__(parent)
        self.assembly = assembly
        self._updating = False
        self._selected_component = None
        self._setup_ui()
        self.rebuild()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Tree header
        header = QLabel("ROCKET STRUCTURE")
        header.setStyleSheet("color: #58a6ff; font-weight: 700; font-size: 11px; "
            "letter-spacing: 1px; padding: 4px 8px;")
        layout.addWidget(header)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(20)
        self.tree.setAnimated(True)
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #0d1117; border: 1px solid #21262d; border-radius: 6px; }
            QTreeWidget::item { padding: 4px 6px; border-radius: 3px; }
            QTreeWidget::item:selected { background-color: #1f6feb; color: #ffffff; }
            QTreeWidget::item:hover:!selected { background-color: #161b22; }
        """)
        self.tree.currentItemChanged.connect(self._on_selection_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.tree, 1)

        # Action buttons
        btn_group = QGroupBox()
        btn_group.setStyleSheet("QGroupBox { border: none; }")
        bl = QHBoxLayout(btn_group)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)

        self.btn_up = QPushButton("▲")
        self.btn_up.setToolTip("Move Up")
        self.btn_up.setFixedSize(36, 30)
        self.btn_up.clicked.connect(self._move_up)
        bl.addWidget(self.btn_up)

        self.btn_down = QPushButton("▼")
        self.btn_down.setToolTip("Move Down")
        self.btn_down.setFixedSize(36, 30)
        self.btn_down.clicked.connect(self._move_down)
        bl.addWidget(self.btn_down)

        self.btn_dup = QPushButton("⧉")
        self.btn_dup.setToolTip("Duplicate")
        self.btn_dup.setFixedSize(36, 30)
        self.btn_dup.clicked.connect(self._duplicate)
        bl.addWidget(self.btn_dup)

        self.btn_del = QPushButton("✕")
        self.btn_del.setToolTip("Delete")
        self.btn_del.setFixedSize(36, 30)
        self.btn_del.setProperty("danger", True)
        self.btn_del.clicked.connect(self._delete)
        bl.addWidget(self.btn_del)

        bl.addStretch()
        layout.addWidget(btn_group)

    def rebuild(self):
        self._updating = True
        self.tree.clear()

        root = QTreeWidgetItem(self.tree, ["🚀 " + self.assembly.name])
        root.setData(0, Qt.ItemDataRole.UserRole, None)
        f = root.font(0)
        f.setBold(True)
        f.setPointSize(11)
        root.setFont(0, f)
        root.setForeground(0, QBrush(QColor("#e6edf3")))
        root.setExpanded(True)

        for stage in self.assembly.stages:
            stage_item = self._add_item(root, stage)
            stage_item.setExpanded(True)
            self._add_children(stage_item, stage)

        self.tree.expandAll()
        self._updating = False

    def _add_item(self, parent_item, component):
        icon = ICONS.get(component.component_type, "•")
        mass_str = f" ({component.computed_mass()*1000:.1f}g)" if component.computed_mass() > 0 else ""
        text = f"{icon} {component.name}{mass_str}"
        item = QTreeWidgetItem(parent_item, [text])
        item.setData(0, Qt.ItemDataRole.UserRole, component)
        item.setForeground(0, QBrush(QColor(component.color)))

        if isinstance(component, Stage):
            f = item.font(0)
            f.setBold(True)
            item.setFont(0, f)

        return item

    def _add_children(self, parent_item, parent_comp):
        for child in parent_comp.children:
            child_item = self._add_item(parent_item, child)
            if child.can_have_children:
                child_item.setExpanded(True)
                self._add_children(child_item, child)

    def _on_selection_changed(self, current, previous):
        if self._updating or not current:
            return
        comp = current.data(0, Qt.ItemDataRole.UserRole)
        self._selected_component = comp
        self.component_selected.emit(comp)

    def selected_component(self):
        return self._selected_component

    def select_component(self, component):
        """Programmatically select a component in the tree."""
        if not component:
            self.tree.clearSelection()
            return

        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, Qt.ItemDataRole.UserRole) == component:
                self._updating = True  # Prevent feedback loop
                self.tree.setCurrentItem(item)
                item.setSelected(True)
                # Ensure visibility
                self.tree.scrollToItem(item)
                self._updating = False
                break
            iterator += 1

    def _move_up(self):
        comp = self._selected_component
        if comp and not isinstance(comp, Stage):
            self.assembly.move_up(comp)
            self.rebuild()
            self.tree_changed.emit()

    def _move_down(self):
        comp = self._selected_component
        if comp and not isinstance(comp, Stage):
            self.assembly.move_down(comp)
            self.rebuild()
            self.tree_changed.emit()

    def _duplicate(self):
        comp = self._selected_component
        if comp and not isinstance(comp, Stage):
            self.assembly.duplicate_component(comp)
            self.rebuild()
            self.tree_changed.emit()

    def _delete(self):
        comp = self._selected_component
        if comp and not isinstance(comp, Stage):
            self.assembly.remove_component(comp)
            self._selected_component = None
            self.rebuild()
            self.component_selected.emit(None)
            self.tree_changed.emit()

    def _context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return
        comp = item.data(0, Qt.ItemDataRole.UserRole)
        if not comp or isinstance(comp, Stage):
            return

        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background: #161b22; border: 1px solid #30363d; } "
            "QMenu::item { padding: 6px 20px; } QMenu::item:selected { background: #1f6feb; }")
        menu.addAction("▲ Move Up", lambda: QTimer.singleShot(0, self._move_up))
        menu.addAction("▼ Move Down", lambda: QTimer.singleShot(0, self._move_down))
        menu.addSeparator()
        menu.addAction("📋 Duplicate", lambda: QTimer.singleShot(0, self._duplicate))
        menu.addAction("❌ Delete", lambda: QTimer.singleShot(0, self._delete))
        menu.exec(self.tree.viewport().mapToGlobal(pos))
