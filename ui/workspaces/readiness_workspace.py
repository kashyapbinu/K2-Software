"""
K2 AeroSim — Readiness Workspace
==================================
The design's path from drawing to report on one screen: which analyses have
been run, which of those still belong to the rocket as drawn, and what to do
next.

Every other tab keeps showing its last result after the design changes; this
one says when that result is out of date and what made it so. All of it is
read from ``engine.readiness`` (core.readiness) — the board holds no state of
its own.
"""
import logging

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from core import readiness as R
from ui import theme, units
from ui.widgets.readout import Readout

logger = logging.getLogger("K2.ReadinessWS")


def status_word(st: "R.StageStatus") -> str:
    """The one word a stage tile shows."""
    if st.running:
        return "Running"
    if st.status == R.NOT_RUN:
        return "Not Run"
    if st.status == R.STALE:
        return "Out of Date"
    if st.verdict is False:
        return "Fails"
    return "Current"


def status_color(st: "R.StageStatus") -> str:
    if st.running:
        return theme.ACCENT
    if st.status == R.NOT_RUN:
        return theme.TEXT_FAINT
    if st.status == R.STALE:
        return theme.WARN
    return theme.ERR if st.verdict is False else theme.OK


class StageTile(QFrame):
    """One analysis: a status bar over its name, verdict, result and reason."""

    open_requested = pyqtSignal(str)   # main-window tab name

    def __init__(self, stage: "R.Stage", parent=None):
        super().__init__(parent)
        self.setObjectName("stageTile")
        self.stage = stage
        self._status = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(120)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        self.lbl_title = QLabel(stage.title)
        self.lbl_status = QLabel()
        self.lbl_status.setIndent(0)   # a padded label is otherwise inset half a character
        self.lbl_headline = QLabel()
        self.lbl_reason = QLabel()
        for lbl in (self.lbl_headline, self.lbl_reason):
            lbl.setWordWrap(True)
        self.btn_open = QPushButton(f"Open {stage.tab}")
        self.btn_open.clicked.connect(lambda: self.open_requested.emit(stage.tab))
        lay.addWidget(self.lbl_title)
        lay.addWidget(self.lbl_status)
        lay.addWidget(self.lbl_headline)
        lay.addWidget(self.lbl_reason)
        lay.addStretch(1)
        lay.addWidget(self.btn_open)
        self.set_status(R.StageStatus(stage, R.NOT_RUN))

    def set_status(self, st: "R.StageStatus"):
        self._status = st
        self.retheme()

    def retheme(self):
        st = self._status
        color = status_color(st)
        bar = theme.LINE if (st.status == R.NOT_RUN and not st.running) else color
        self.setStyleSheet(
            f"QFrame#stageTile {{ background-color: {theme.PANEL}; "
            f"border: 1px solid {theme.LINE}; border-top: 3px solid {bar}; "
            f"border-radius: 3px; }}")
        plain = "background: transparent; border: none;"
        self.lbl_title.setStyleSheet(
            f"color: {theme.TEXT_BRIGHT}; font-size: 13px; font-weight: 600; {plain}")
        self.lbl_status.setText(status_word(st))
        self.lbl_status.setStyleSheet(theme.status_qss(color) + " border: none;")
        stale = st.status == R.STALE
        self.lbl_headline.setText(st.headline or
                                  ("Optional" if st.stage.optional and st.status == R.NOT_RUN else ""))
        self.lbl_headline.setStyleSheet(
            f"color: {theme.TEXT_DIM if stale or not st.headline else theme.TEXT}; "
            f"font-family: {theme.MONO}; font-size: 12px; {plain}")
        self.lbl_reason.setText(st.reason[:1].upper() + st.reason[1:] if stale else "")
        self.lbl_reason.setStyleSheet(f"color: {theme.WARN}; font-size: 11px; {plain}")
        self.lbl_reason.setVisible(stale)


class ReadinessWorkspace(QWidget):
    """Design → flight → dispersion → loads → rules → report, with what is
    current, what is out of date and why."""

    open_tab = pyqtSignal(str)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.tiles = {}
        self._setup_ui()
        self.engine.state_changed.connect(lambda _s: self.refresh())
        self.engine.readiness_changed.connect(self.refresh)
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        self.lbl_title = QLabel("Mission Readiness")
        self.lbl_title.setStyleSheet(theme.panel_title_qss())
        layout.addWidget(self.lbl_title)

        self.lbl_next = QLabel()
        self.lbl_next.setWordWrap(True)
        layout.addWidget(self.lbl_next)

        # ── The design everything below was (or was not) run on ──
        self.lbl_design = QLabel("Design")
        layout.addWidget(self.lbl_design)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.design_tiles = {}
        for col, (key, caption) in enumerate((
                ("name", "Rocket"), ("length", "Length"), ("diameter", "Diameter"),
                ("mass", "Liftoff Mass"), ("motor", "Motor"),
                ("stability", "Static Margin"))):
            tile = Readout(caption)
            self.design_tiles[key] = tile
            grid.addWidget(tile, 0, col)
        layout.addLayout(grid)

        # ── The analyses, in the order a design goes through them ──
        self.lbl_pipeline = QLabel("Analyses")
        layout.addWidget(self.lbl_pipeline)
        row = QHBoxLayout()
        row.setSpacing(0)
        for i, stage in enumerate(R.STAGES):
            if i:
                link = QFrame()
                link.setObjectName("stageLink")
                link.setFixedSize(14, 1)
                self._links = getattr(self, "_links", []) + [link]
                row.addWidget(link, 0, Qt.AlignmentFlag.AlignVCenter)
            tile = StageTile(stage)
            tile.open_requested.connect(self.open_tab)
            self.tiles[stage.key] = tile
            row.addWidget(tile, 1)
        layout.addLayout(row, 1)

        self.lbl_note = QLabel(
            "A result is out of date when the inputs it was computed from have "
            "changed since it was run. Results are kept for this session only.")
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)
        layout.addStretch(2)
        self.retheme()

    def retheme(self):
        section = (f"color: {theme.TEXT_DIM}; font-size: 11px; font-weight: 600; "
                   "background: transparent;")
        self.lbl_design.setStyleSheet(section)
        self.lbl_pipeline.setStyleSheet(section)
        self.lbl_note.setStyleSheet(
            f"color: {theme.TEXT_FAINT}; font-size: 11px; background: transparent;")
        for link in getattr(self, "_links", []):
            link.setStyleSheet(f"QFrame#stageLink {{ background-color: {theme.LINE}; border: none; }}")

    def reset_workspace(self):
        self.refresh()

    def refresh(self):
        s = self.engine.state
        problems = R.design_problems(s)
        self.design_tiles["name"].setText(s.name or "—")
        self.design_tiles["length"].setText(
            units.format_value(s.length, "length") if s.length > 0 else "—")
        self.design_tiles["diameter"].setText(
            units.format_value(s.diameter, "length") if s.diameter > 0 else "—")
        # The rocket on the pad, whatever the last flight burned. The margin
        # tile read the live state.stability_margin, which after a flight is
        # the burnt-out rocket's (6.78 cal for the canonical rocket, 5.30 on
        # the pad).
        mass = s.liftoff_mass()
        self.design_tiles["mass"].setText(units.format_value(mass, "mass") if mass > 0 else "—")
        self.design_tiles["motor"].setText(
            s.motor_designation if "No motor" not in problems else "—")
        drawn = s.length > 0 and s.diameter > 0
        margin = s.pad_stability_margin() if drawn else None
        self.design_tiles["stability"].setText(
            f"{margin:.2f} cal" if margin is not None else "—")
        self.design_tiles["stability"].setValueColor(
            theme.ERR if margin is not None and margin < 0 else None)

        for st in self.engine.readiness.all(s):
            self.tiles[st.stage.key].set_status(st)

        self.lbl_next.setText("Next: " + self.engine.readiness.next_step(s))
        self.lbl_next.setStyleSheet(
            f"color: {theme.TEXT_BRIGHT}; font-size: 13px; background: transparent;")
