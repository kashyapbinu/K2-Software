"""
K2 AeroSim — Rules panel
==========================
Checklist of a rule set (core.rules) against one flight or a Monte Carlo
run: value, limit, pass / fail / not evaluated, margin, and pass rate.

Limits are edited in the table and saved as the user's own rule set (app
settings). Built-in sets are never overwritten.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout,
                             QHeaderView, QInputDialog, QLabel, QMessageBox,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QVBoxLayout, QWidget)

from core import rules as R
from ui import settings, theme

logger = logging.getLogger("K2.RulesPanel")

SETTINGS_KEY = "rules/user_sets"
COLS = ["Rule", "Test", "Limit", "Upper", "Value", "Result", "Margin", "Pass rate"]
C_RULE, C_TEST, C_LIMIT, C_UPPER, C_VALUE, C_RESULT, C_MARGIN, C_RATE = range(len(COLS))


def _fmt(v, unit=""):
    if v is None:
        return "—"
    return f"{v:.3g} {unit}".strip() if abs(v) < 1000 else f"{v:.0f} {unit}".strip()


class RulesPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._summaries = []
        self._state = None
        self._filling = False
        self._build()
        self.reload_sets()

    # ── layout ──
    def _build(self):
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        top.addWidget(QLabel("Rule set:"))
        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self._on_set_chosen)
        top.addWidget(self.set_combo, 1)
        top.addWidget(QLabel("Target apogee:"))
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(0, 200000); self.target_spin.setDecimals(0)
        self.target_spin.setSuffix(" m"); self.target_spin.setSpecialValueText("none")
        self.target_spin.valueChanged.connect(lambda _: self.refresh())
        top.addWidget(self.target_spin)
        self.btn_save = QPushButton("Save As…")
        self.btn_save.clicked.connect(self._save_as)
        top.addWidget(self.btn_save)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._delete)
        top.addWidget(self.btn_delete)
        lay.addLayout(top)

        self.source_label = QLabel("")
        self.source_label.setWordWrap(True)
        lay.addWidget(self.source_label)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(C_RULE, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_cell_edited)
        lay.addWidget(self.table, 1)

        self.summary_label = QLabel("")
        lay.addWidget(self.summary_label)

    # ── rule sets ──
    def reload_sets(self, select: str = None):
        user = R.parse_user_rule_sets(settings.get(SETTINGS_KEY) or "")
        names = {s.name for s in user}
        self._sets = [s for s in R.builtin_rule_sets() if s.name not in names] + user
        current = select or self.set_combo.currentText()
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for s in self._sets:
            self.set_combo.addItem(s.name)
        self.set_combo.blockSignals(False)
        idx = self.set_combo.findText(current) if current else -1
        self.set_combo.setCurrentIndex(max(0, idx))
        self._on_set_chosen(self.set_combo.currentIndex())

    def current_set(self):
        i = self.set_combo.currentIndex()
        return self._sets[i] if 0 <= i < len(self._sets) else None

    def _on_set_chosen(self, _index):
        rs = self.current_set()
        self.btn_delete.setEnabled(bool(rs and not rs.builtin))
        if rs is None:
            self.source_label.setText("No rule sets.")
        elif rs.confirmed and rs.source:
            self.source_label.setText(f"Limits from: {rs.source}")
            self.source_label.setStyleSheet(f"color: {theme.TEXT_DIM};")
        else:
            src = rs.source or "no source given"
            self.source_label.setText(
                f"⚠ Limits not confirmed against a rulebook ({src}). "
                f"Check every number before relying on this check. {rs.notes}")
            self.source_label.setStyleSheet(f"color: {theme.WARN};")
        self.refresh()

    def _save_as(self):
        rs = self.current_set()
        if rs is None:
            return
        name, ok = QInputDialog.getText(self, "Save Rule Set", "Name:",
                                        text=rs.name if not rs.builtin else "")
        if not ok or not name.strip():
            return
        source, ok = QInputDialog.getText(
            self, "Save Rule Set",
            "Where the limits come from (rulebook, section, year):", text=rs.source)
        if not ok:
            return
        new = R.RuleSet(name=name.strip(), source=source.strip(), confirmed=bool(source.strip()),
                        notes=rs.notes, rules=[R.Rule(**vars(r)) for r in rs.rules])
        user = [s for s in R.parse_user_rule_sets(settings.get(SETTINGS_KEY) or "")
                if s.name != new.name] + [new]
        settings.set(SETTINGS_KEY, R.dump_user_rule_sets(user))
        self.reload_sets(select=new.name)

    def _delete(self):
        rs = self.current_set()
        if rs is None or rs.builtin:
            return
        user = [s for s in R.parse_user_rule_sets(settings.get(SETTINGS_KEY) or "")
                if s.name != rs.name]
        settings.set(SETTINGS_KEY, R.dump_user_rule_sets(user))
        self.reload_sets()

    # ── flights ──
    def set_flights(self, summaries, state=None):
        """One FlightSummary, or the list from a Monte Carlo run."""
        if summaries is None:
            summaries = []
        self._summaries = summaries if isinstance(summaries, list) else [summaries]
        self._many = isinstance(summaries, list) and len(summaries) > 1
        self._state = state
        self.refresh()

    def results(self):
        rs = self.current_set()
        if rs is None or not self._summaries:
            return []
        ctx = R.context_from_state(self._state, self.target_spin.value()) \
            if self._state is not None else R.Context(target_apogee=self.target_spin.value())
        flights = self._summaries if self._many else self._summaries[0]
        return R.evaluate_all(rs, flights, ctx)

    def refresh(self):
        rs = self.current_set()
        self._filling = True
        self.table.setRowCount(0)
        if rs is not None:
            results = {id(r.rule): r for r in self.results()}
            for row, rule in enumerate(rs.rules):
                self.table.insertRow(row)
                q = R.QUANTITIES.get(rule.quantity)
                unit = q.unit if q else ""
                res = results.get(id(rule))
                cells = [
                    rule.name + (f"  [{rule.citation}]" if rule.citation else ""),
                    rule.comparison,
                    "" if rule.limit is None else f"{rule.limit:g}",
                    "" if rule.limit_high is None else f"{rule.limit_high:g}",
                    _fmt(res.value, unit) if res else "—",
                    (res.status if res else "no flight"),
                    _fmt(res.margin, unit) if res else "—",
                    (f"{res.pass_rate * 100:.1f} % of {res.evaluated}"
                     if res and res.pass_rate is not None and self._many else ""),
                ]
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if col not in (C_LIMIT, C_UPPER):
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if col == C_RESULT and res:
                        color = {R.PASS: theme.OK, R.FAIL: theme.ERR}.get(res.status, theme.TEXT_DIM)
                        item.setForeground(QColor(color))
                    if col == C_RULE and res and res.note:
                        item.setToolTip(res.note)
                    self.table.setItem(row, col, item)
        self._filling = False
        res = self.results()
        if not self._summaries:
            self.summary_label.setText("No flight to check yet.")
        else:
            n_pass = sum(r.status == R.PASS for r in res)
            n_fail = sum(r.status == R.FAIL for r in res)
            n_na = len(res) - n_pass - n_fail
            self.summary_label.setText(f"{n_pass} pass, {n_fail} fail, {n_na} not evaluated")

    def _on_cell_edited(self, item):
        if self._filling or item.column() not in (C_LIMIT, C_UPPER):
            return
        rs = self.current_set()
        rule = rs.rules[item.row()]
        text = item.text().strip()
        try:
            value = float(text) if text else None
        except ValueError:
            QMessageBox.warning(self, "Rules", f"'{text}' is not a number.")
            self.refresh()
            return
        if item.column() == C_LIMIT:
            rule.limit = value
        else:
            rule.limit_high = value
        rs.confirmed = False       # edited numbers are no longer the cited ones
        self._on_set_chosen(self.set_combo.currentIndex())
