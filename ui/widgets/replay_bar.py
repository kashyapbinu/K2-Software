"""
K2 AeroSim — Replay bar
=========================
Transport controls for the shared replay clock (ui.replay_clock): restart,
play / pause, speed, and a timeline of the whole flight with its events
marked on it. Click or drag on the timeline to move to an instant; click an
event's mark to land exactly on it.

The bar holds no flight state: it draws the clock's and tells the clock what
the user asked for, so any number of bars and views stay on the same instant.
"""
from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QWidget)

from ui import theme
from ui.icons import icon
from ui.replay_clock import SPEEDS


class FlightTimeline(QWidget):
    """The flight from first to last recorded instant, with event marks."""

    TRACK_Y = 8          # centre line of the track
    SNAP_PX = 6          # a click this close to an event mark lands on it

    def __init__(self, clock, parent=None):
        super().__init__(parent)
        self._clock = clock
        self.setMinimumWidth(220)
        self.setFixedHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        clock.time_changed.connect(lambda _t: self.update())
        clock.flight_changed.connect(self.update)

    # ── geometry ──
    def _span(self):
        return 6.0, max(7.0, self.width() - 6.0)

    def x_of(self, t: float) -> float:
        r = self._clock.replay
        x0, x1 = self._span()
        if r is None or r.duration <= 0:
            return x0
        return x0 + (x1 - x0) * (t - r.start) / r.duration

    def time_at(self, x: float) -> float:
        """The instant at pixel *x*: an event's own time when *x* is on its
        mark, otherwise the time under the pointer."""
        r = self._clock.replay
        if r is None:
            return 0.0
        for e in r.events:
            if abs(self.x_of(e.time) - x) <= self.SNAP_PX:
                return e.time
        x0, x1 = self._span()
        frac = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
        return r.start + frac * r.duration

    # ── input ──
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._clock.replay is not None:
            self._clock.pause()
            self._clock.seek(self.time_at(ev.position().x()))

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.LeftButton and self._clock.replay is not None:
            self._clock.seek(self.time_at(ev.position().x()))

    # ── drawing ──
    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        x0, x1 = self._span()
        y = float(self.TRACK_Y)
        r = self._clock.replay
        p.setPen(QPen(QColor(theme.LINE), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        if r is None:
            p.end()
            return
        xt = self.x_of(self._clock.time)
        p.setPen(QPen(QColor(theme.ACCENT), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawLine(QPointF(x0, y), QPointF(xt, y))

        font = QFont(self.font())
        font.setPixelSize(10)
        p.setFont(font)
        metrics = p.fontMetrics()
        last_right = -1e9
        for e in r.events:
            x = self.x_of(e.time)
            p.setPen(QPen(QColor(theme.TEXT_DIM), 1))
            p.drawLine(QPointF(x, y - 5), QPointF(x, y + 7))
            w = metrics.horizontalAdvance(e.label)
            left = min(max(x - w / 2, 0.0), self.width() - w)
            if left > last_right + 6:        # skip a label that would overlap the last
                p.drawText(QRectF(left, y + 9, w + 1, 14),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, e.label)
                last_right = left + w

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(theme.ACCENT))
        p.drawEllipse(QPointF(xt, y), 5.5, 5.5)
        p.end()


class ReplayBar(QWidget):
    """Restart, play / pause, speed, timeline and time read-out for a clock."""

    def __init__(self, clock, parent=None):
        super().__init__(parent)
        self._clock = clock
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.btn_restart = QPushButton(icon("reset"), "")
        self.btn_restart.setToolTip("Back to the start of the flight")
        self.btn_restart.clicked.connect(clock.restart)
        self.btn_play = QPushButton(icon("run"), "Play")
        self.btn_play.setMinimumWidth(82)
        self.btn_play.clicked.connect(clock.toggle)

        self.speed_combo = QComboBox()
        for s in SPEEDS:
            self.speed_combo.addItem(f"{s:g}×", s)
        self.speed_combo.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_combo.setToolTip("Playback speed relative to real time")
        self.speed_combo.currentIndexChanged.connect(
            lambda i: clock.set_speed(self.speed_combo.itemData(i)))

        self.timeline = FlightTimeline(clock)
        self.lbl_time = QLabel()
        self.lbl_time.setMinimumWidth(150)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        for w in (self.btn_restart, self.btn_play, self.speed_combo):
            lay.addWidget(w)
        lay.addWidget(self.timeline, 1)
        lay.addWidget(self.lbl_time)

        clock.time_changed.connect(self._show_time)
        clock.playing_changed.connect(self._show_playing)
        clock.flight_changed.connect(self._on_flight)
        self.retheme()
        self._on_flight()

    def retheme(self):
        self.lbl_time.setStyleSheet(
            f"color: {theme.TEXT}; font-family: {theme.MONO}; font-size: 11px; "
            "background: transparent;")
        self.timeline.update()

    def _on_flight(self):
        has = self._clock.replay is not None
        for w in (self.btn_restart, self.btn_play, self.speed_combo, self.timeline):
            w.setEnabled(has)
        self._show_time(self._clock.time)
        self._show_playing(self._clock.playing)

    def _show_playing(self, playing: bool):
        self.btn_play.setText("Pause" if playing else "Play")
        self.btn_play.setIcon(icon("pause" if playing else "run"))

    def _show_time(self, t: float):
        r = self._clock.replay
        if r is None:
            self.lbl_time.setText("No flight")
            return
        self.lbl_time.setText(f"T+{t:.2f} s / {r.end:.1f} s")
