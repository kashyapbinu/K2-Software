"""
K2 Aerospace — Preset Picker Dialog
Searchable/filterable table of component presets.
"""
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QHeaderView)
from PyQt6.QtCore import Qt


class PresetDialog(QDialog):
    def __init__(self, presets, component_type, parent=None):
        super().__init__(parent)
        self.presets = presets
        self._selected = None
        self.setWindowTitle(f"Choose {component_type} Preset")
        self.setMinimumSize(700, 450)
        self.setStyleSheet("""
            QDialog { background: #0d1117; }
            QTableWidget { background: #0d1117; border: 1px solid #21262d; gridline-color: #21262d; }
            QTableWidget::item { padding: 4px 8px; }
            QTableWidget::item:selected { background: #1f6feb; }
            QHeaderView::section { background: #161b22; color: #58a6ff; border: 1px solid #21262d;
                padding: 6px; font-weight: 600; font-size: 11px; }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Filter
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Search by name, manufacturer, or part number...")
        self.filter_edit.textChanged.connect(self._filter)
        filt_row.addWidget(self.filter_edit)
        layout.addLayout(filt_row)

        # Table
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self.accept)
        layout.addWidget(self.table, 1)

        self._populate()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        ok = QPushButton("Apply"); ok.setProperty("primary", True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel); btn_row.addWidget(ok)
        layout.addLayout(btn_row)

    def _populate(self):
        if not self.presets:
            return
        keys = [k for k in self.presets[0].keys() if k != "material"]
        display_keys = keys[:6]
        self.table.setColumnCount(len(display_keys))
        self.table.setHorizontalHeaderLabels([k.replace("_", " ").title() for k in display_keys])
        self.table.setRowCount(len(self.presets))

        for r, preset in enumerate(self.presets):
            for c, key in enumerate(display_keys):
                val = preset.get(key, "")
                if isinstance(val, float):
                    val = f"{val:.4f}" if val < 1 else f"{val:.2f}"
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

    def _filter(self, text):
        text = text.lower()
        for r in range(self.table.rowCount()):
            match = False
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item and text in item.text().lower():
                    match = True; break
            self.table.setRowHidden(r, not match)

    def selected_preset(self):
        rows = self.table.selectionModel().selectedRows()
        if rows:
            return self.presets[rows[0].row()]
        return None
