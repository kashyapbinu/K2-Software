"""
K2 AeroSim — Target Apogee dialog
===================================
Front end for core.apogee_tools: ballast to hit a target apogee, airbrake
area to hit it, and the apogee's sensitivity to launch-day unknowns.

Every answer comes from batch flights (about a second each), so the work
runs on a QThread and the dialog stays responsive.
"""
from __future__ import annotations

import logging

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit,
                             QPushButton, QVBoxLayout)

from core import apogee_tools
from core.batch_simulation import BatchSimConfig
from ui import theme

logger = logging.getLogger("K2.ApogeeTools")

FOOT = 0.3048


class _Worker(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str, object)       # kind, result
    failed = pyqtSignal(str)

    def __init__(self, kind, fn, parent=None):
        super().__init__(parent)
        self.kind, self.fn = kind, fn

    def run(self):
        try:
            self.done.emit(self.kind, self.fn(self.progress.emit))
        except Exception as exc:          # noqa: BLE001 — shown to the user
            logger.exception("apogee tool failed")
            self.failed.emit(str(exc))


class ApogeeToolsDialog(QDialog):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._worker = None
        self.last_result = None
        self.setWindowTitle("Target Apogee")
        self.setMinimumWidth(520)
        self._build()

    # ── layout ──
    def _build(self):
        s = self.engine.state
        lay = QVBoxLayout(self)

        g = QGroupBox("Target")
        f = QFormLayout(g)
        row = QHBoxLayout()
        self.target_spin = QDoubleSpinBox()
        self.target_spin.setRange(1, 200000); self.target_spin.setDecimals(0)
        self.target_spin.setValue(round(s.max_altitude * 0.95) if s.max_altitude > 0 else 3048)
        row.addWidget(self.target_spin, 1)
        self.unit_combo = QComboBox(); self.unit_combo.addItems(["m", "ft"])
        self.unit_combo.currentTextChanged.connect(self._on_unit)
        row.addWidget(self.unit_combo)
        f.addRow("Apogee above pad:", row)

        self.ballast_pos = QDoubleSpinBox()
        self.ballast_pos.setRange(0.0, max(0.01, s.length)); self.ballast_pos.setDecimals(3)
        self.ballast_pos.setSuffix(" m")
        self.ballast_pos.setValue(min(s.length, s.nose_length * 0.5) if s.nose_length else 0.0)
        self.ballast_pos.setToolTip("Where the ballast goes, measured from the nose tip.")
        f.addRow("Ballast at:", self.ballast_pos)

        self.min_stab = QDoubleSpinBox()
        self.min_stab.setRange(0.0, 5.0); self.min_stab.setDecimals(2)
        self.min_stab.setValue(1.0); self.min_stab.setSuffix(" cal")
        f.addRow("Min stability:", self.min_stab)
        lay.addWidget(g)

        btns = QHBoxLayout()
        self.btn_ballast = QPushButton("Solve Ballast")
        self.btn_ballast.clicked.connect(self.solve_ballast)
        self.btn_airbrake = QPushButton("Solve Airbrake")
        self.btn_airbrake.clicked.connect(self.solve_airbrake)
        self.btn_sens = QPushButton("Sensitivity")
        self.btn_sens.clicked.connect(self.sensitivity)
        for b in (self.btn_ballast, self.btn_airbrake, self.btn_sens):
            btns.addWidget(b)
        lay.addLayout(btns)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        lay.addWidget(self.status)
        self.output = QPlainTextEdit(); self.output.setReadOnly(True)
        self.output.setMinimumHeight(180)
        lay.addWidget(self.output)

        ab = QGroupBox("Airbrakes on the vehicle")
        af = QHBoxLayout(ab)
        self.airbrake_spin = QDoubleSpinBox()
        self.airbrake_spin.setRange(0.0, 5000.0); self.airbrake_spin.setDecimals(1)
        self.airbrake_spin.setSuffix(" cm²")
        self.airbrake_spin.setValue(s.airbrake_cd_area * 1e4)
        self.airbrake_spin.setToolTip("Cd × area, deployed from burnout to apogee. 0 = none.")
        af.addWidget(self.airbrake_spin, 1)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_airbrake)
        af.addWidget(apply_btn)
        lay.addWidget(ab)

    def _on_unit(self, unit):
        v = self.target_spin.value()
        self.target_spin.setValue(v / FOOT if unit == "ft" else v * FOOT)

    def target_m(self) -> float:
        v = self.target_spin.value()
        return v * FOOT if self.unit_combo.currentText() == "ft" else v

    def _cfg(self):
        """The rocket as the batch engine flies it; None, with the reason
        shown, for one it does not fly (a staged rocket)."""
        try:
            return BatchSimConfig.from_rocket_state(self.engine.state)
        except ValueError as exc:
            self.status.setText("Not flown")
            self.output.setPlainText(str(exc))
            return None

    # ── actions ──
    def _start(self, kind, fn):
        if self._worker is not None and self._worker.isRunning():
            return
        for b in (self.btn_ballast, self.btn_airbrake, self.btn_sens):
            b.setEnabled(False)
        self.status.setText("Flying…")
        self._worker = _Worker(kind, fn, self)
        self._worker.progress.connect(lambda n: self.status.setText(f"Flying… {n} flights"))
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def solve_ballast(self):
        cfg, tgt, pos, ms = self._cfg(), self.target_m(), self.ballast_pos.value(), self.min_stab.value()
        if cfg is None:
            return
        self._start("ballast", lambda p: apogee_tools.solve_ballast(cfg, tgt, pos, ms, p))

    def solve_airbrake(self):
        cfg, tgt = self._cfg(), self.target_m()
        if cfg is None:
            return
        self._start("airbrake", lambda p: apogee_tools.solve_airbrake(cfg, tgt, p))

    def sensitivity(self):
        cfg = self._cfg()
        if cfg is None:
            return
        self._start("sensitivity", lambda p: apogee_tools.apogee_sensitivity(cfg, p))

    def _finish(self):
        for b in (self.btn_ballast, self.btn_airbrake, self.btn_sens):
            b.setEnabled(True)

    def _on_failed(self, msg):
        self._finish()
        self.status.setText("Failed")
        self.output.setPlainText(msg)

    def _on_done(self, kind, result):
        self._finish()
        self.last_result = result
        self.status.setText("Done")
        self.output.setPlainText(format_result(kind, result))
        if kind == "airbrake" and result.reachable:
            self.airbrake_spin.setValue(result.value * 1e4)

    def _apply_airbrake(self):
        self.engine.update(airbrake_cd_area=self.airbrake_spin.value() / 1e4)


def format_result(kind, result) -> str:
    if kind == "sensitivity":
        lines = ["Apogee change per unit (central difference, batch flights):", ""]
        for x in result:
            lines.append(f"  {x.name:16s} {x.per_unit:+9.2f} m per {x.unit}"
                         f"   (±{x.step:.3g} {x.unit}: {x.apogee_minus:.0f} … {x.apogee_plus:.0f} m)")
        return "\n".join(lines)
    r = result
    lines = [r.message, "",
             f"Baseline apogee: {r.baseline_apogee:.0f} m", f"Flights: {r.runs}"]
    if r.reachable:
        lines.append(f"Apogee with it: {r.apogee:.0f} m (target {r.target:.0f} m)")
        if r.min_stability is not None:
            lines.append(f"Min stability: {r.min_stability:.2f} cal "
                         f"(was {r.baseline_min_stability:.2f})")
    lines += [f"⚠ {w}" for w in r.warnings]
    if kind == "ballast" and r.reachable:
        lines += ["", "Add it in the Design tab as a Mass Component at that position."]
    return "\n".join(lines)
