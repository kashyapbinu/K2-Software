"""
AI assistant dock: streaming chat over the current rocket, with tool calls
that edit the design and run the sim. The LLM round-trip runs in a worker
thread; tool execution is marshalled back to the GUI thread by ToolBridge.

Transcript is a QTextBrowser; messages are rendered as HTML cards (tables —
the rich-text engine has no border-radius or flexbox) so they read as a chat
rather than a log.
"""

from __future__ import annotations

import html
import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QTextCursor, QKeyEvent, QTextBlockFormat
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser,
    QPlainTextEdit, QSizePolicy, QToolButton, QMenu, QFrame, QComboBox,
)

from ui import theme, settings
from ui.icons import icon
from ai.providers import build_provider, describe_auto_state, Provider, _setting, gemini_model_setting
from ai.assistant import AssistantSession
from ai.context import summarize_state, render_context
from ai.tools import ToolBridge, TOOL_SPECS
from ai import prompts

logger = logging.getLogger("K2.AI.Panel")

SUGGESTIONS = [
    ("Is my rocket stable?", "Is my rocket stable? Explain the margin and what drives it."),
    ("Which motor fits?", "Suggest 3 motors that fit this airframe and give a good thrust-to-weight. Explain the trade-offs."),
    ("Make it stable", "If the rocket is not in the 1–2 cal range, fix it with the smallest change (nose weight or fins), then report the new margin."),
    ("Explain last flight", None),   # routed to the debrief action
]

_TOOL_LABELS = {
    "get_rocket_state": "Reading design",
    "set_parameter": "Changing parameter",
    "add_mass": "Adding ballast",
    "search_motors": "Searching motors",
    "select_motor": "Installing motor",
    "run_simulation": "Running simulation",
    "get_flight_result": "Reading flight result",
}


class _Worker(QThread):
    text = pyqtSignal(str)
    tool_call = pyqtSignal(str, dict)
    tool_result = pyqtSignal(str, str)
    error = pyqtSignal(str)
    finished_turn = pyqtSignal()

    def __init__(self, session: AssistantSession, user_text: str, one_shot: bool = False):
        super().__init__()
        self.session = session
        self.user_text = user_text
        self.one_shot = one_shot

    def run(self):
        gen = (self.session.one_shot(self.user_text) if self.one_shot
               else self.session.run_turn(self.user_text))
        try:
            for evt, payload in gen:
                if evt == "text":
                    self.text.emit(payload)
                elif evt == "tool_call":
                    self.tool_call.emit(payload["name"], payload["args"])
                elif evt == "tool_result":
                    self.tool_result.emit(payload["name"], payload["result"])
                elif evt == "error":
                    self.error.emit(payload)
        except Exception as e:  # never let the thread die silently
            logger.exception("AI worker crashed")
            self.error.emit(f"{type(e).__name__}: {e}")
        self.finished_turn.emit()


class _Input(QPlainTextEdit):
    submit = pyqtSignal()

    def keyPressEvent(self, e: QKeyEvent):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                e.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.submit.emit()
            return
        super().keyPressEvent(e)


class AIPanel(QWidget):
    """Dock content. ``main_window`` supplies engine, sim_engine, workspaces."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.mw = main_window
        self.bridge = ToolBridge(main_window)
        self.provider: Provider | None = None
        self.session: AssistantSession | None = None
        self._worker: _Worker | None = None
        self._stream_buf = ""
        self._assistant_anchor = 0
        self._activity_text = "Thinking"
        self._dots = 0
        self._pulse = QTimer(self)
        self._pulse.setInterval(400)
        self._pulse.timeout.connect(self._tick_pulse)
        self._setup_ui()
        self.retheme()
        self.reload_provider()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        lo = QVBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(0)

        # ── Header ──
        self.header = QFrame()
        self.header.setObjectName("aiHeader")
        h = QHBoxLayout(self.header)
        h.setContentsMargins(12, 8, 8, 8)
        h.setSpacing(8)
        self.lbl_icon = QLabel()
        h.addWidget(self.lbl_icon)
        title_col = QVBoxLayout()
        title_col.setSpacing(0)
        self.lbl_title = QLabel("AI Assistant")
        self.lbl_title.setObjectName("aiTitle")
        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("aiStatus")
        title_col.addWidget(self.lbl_title)
        title_col.addWidget(self.lbl_status)
        h.addLayout(title_col, 1)

        # Backend picker: lets the user spend Gemini quota only when they
        # want to, and drop to the local model otherwise.
        self.cmb_backend = QComboBox()
        self.cmb_backend.setObjectName("aiBackend")
        self.cmb_backend.setToolTip("Which model answers")
        self.cmb_backend.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cmb_backend.setMaximumWidth(190)
        self.cmb_backend.currentIndexChanged.connect(self._on_backend_picked)
        h.addWidget(self.cmb_backend)

        self.btn_explain = QToolButton()
        self.btn_explain.setIcon(icon("debrief"))
        self.btn_explain.setToolTip("Debrief the last simulated flight")
        self.btn_explain.setAutoRaise(True)
        self.btn_explain.clicked.connect(self.explain_flight)
        h.addWidget(self.btn_explain)

        self.btn_more = QToolButton()
        self.btn_more.setIcon(icon("more"))
        self.btn_more.setAutoRaise(True)
        self.btn_more.setToolTip("More")
        self.btn_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.btn_more)
        menu.addAction(icon("add"), "New conversation", self.clear_conversation)
        menu.addAction(icon("refresh"), "Reconnect provider", self.reload_provider)
        menu.addSeparator()
        menu.addAction(icon("settings"), "AI settings…", self._open_settings)
        self.btn_more.setMenu(menu)
        h.addWidget(self.btn_more)
        lo.addWidget(self.header)

        # ── Transcript ──
        self.view = QTextBrowser()
        self.view.setObjectName("aiTranscript")
        self.view.setOpenExternalLinks(True)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.document().setDocumentMargin(12)
        lo.addWidget(self.view, 1)

        # ── Suggestion chips ──
        self.chips = QWidget()
        self.chips.setObjectName("aiChips")
        cl = QHBoxLayout(self.chips)
        cl.setContentsMargins(12, 0, 12, 8)
        cl.setSpacing(6)
        for label, prompt in SUGGESTIONS:
            b = QPushButton(label)
            b.setObjectName("aiChip")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if prompt is None:
                b.clicked.connect(self.explain_flight)
            else:
                b.clicked.connect(lambda _=False, p=prompt: self._send_text(p))
            cl.addWidget(b)
        cl.addStretch()
        lo.addWidget(self.chips)

        # ── Activity line ──
        self.lbl_activity = QLabel("")
        self.lbl_activity.setObjectName("aiActivity")
        self.lbl_activity.setContentsMargins(14, 0, 12, 4)
        self.lbl_activity.hide()
        lo.addWidget(self.lbl_activity)

        # ── Composer ──
        self.composer = QFrame()
        self.composer.setObjectName("aiComposer")
        c = QHBoxLayout(self.composer)
        c.setContentsMargins(10, 8, 8, 10)
        c.setSpacing(6)
        self.input = _Input()
        self.input.setObjectName("aiInput")
        self.input.setPlaceholderText("Ask about your rocket…   (Enter to send · Shift+Enter for newline)")
        self.input.setFixedHeight(58)
        self.input.setFrameShape(QFrame.Shape.NoFrame)
        self.input.submit.connect(self.send)
        c.addWidget(self.input, 1)
        self.btn_send = QToolButton()
        self.btn_send.setObjectName("aiSend")
        self.btn_send.setIconSize(QSize(16, 16))
        self.btn_send.setFixedSize(36, 36)
        self.btn_send.setToolTip("Send (Enter)")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.clicked.connect(self._on_send_clicked)
        c.addWidget(self.btn_send, 0, Qt.AlignmentFlag.AlignBottom)
        lo.addWidget(self.composer)

    def retheme(self):
        """Palette-dependent styles; called on construction and theme switch."""
        t = theme
        self.setStyleSheet(f"""
            #aiHeader {{ background: {t.PANEL}; border-bottom: 1px solid {t.LINE}; }}
            #aiTitle {{ color: {t.TEXT_BRIGHT}; font-weight: 600; font-size: 13px; }}
            #aiStatus {{ color: {t.TEXT_DIM}; font-size: 11px; }}
            #aiTranscript {{ background: {t.BG}; border: none; }}
            #aiChips {{ background: {t.BG}; }}
            #aiChip {{
                background: {t.RAISED}; color: {t.TEXT}; border: 1px solid {t.LINE};
                border-radius: 11px; padding: 3px 10px; font-size: 11px;
            }}
            #aiChip:hover {{ border-color: {t.ACCENT}; color: {t.ACCENT}; }}
            #aiBackend {{
                background: {t.FIELD_BG}; color: {t.TEXT}; border: 1px solid {t.LINE};
                border-radius: 6px; padding: 2px 6px; font-size: 11px;
            }}
            #aiBackend:hover {{ border-color: {t.ACCENT}; }}
            #aiActivity {{ color: {t.TEXT_DIM}; font-size: 11px; background: {t.BG}; }}
            #aiComposer {{ background: {t.PANEL}; border-top: 1px solid {t.LINE}; }}
            #aiInput {{
                background: {t.FIELD_BG}; color: {t.TEXT}; border: 1px solid {t.LINE};
                border-radius: 8px; padding: 6px 8px; font-size: 12px;
            }}
            #aiInput:focus {{ border-color: {t.ACCENT}; }}
            #aiSend {{ background: {t.ACCENT}; border: none; border-radius: 18px; }}
            #aiSend:hover {{ background: {t.ACCENT_HOVER}; }}
            #aiSend:disabled {{ background: {t.RAISED}; }}
        """)
        self.lbl_icon.setPixmap(icon("ai", color=t.ACCENT).pixmap(QSize(18, 18)))
        self._refresh_send_icon()

    def _refresh_send_icon(self):
        busy = self._busy()
        key = "stop" if busy else "send"
        color = theme.SELECTION_TEXT if (busy or self.btn_send.isEnabled()) else theme.TEXT_FAINT
        self.btn_send.setIcon(icon(key, color=color))
        self.btn_send.setToolTip("Stop" if busy else "Send (Enter)")

    # ------------------------------------------------------------ provider
    def _backend_choices(self) -> list[tuple[str, str]]:
        """(key, label) for the picker; configured backends only, Auto always."""
        out = [("auto", "Auto")]
        if _setting("ai/gemini_key"):
            out.append(("gemini", f"Gemini · {gemini_model_setting()}"))
        out.append(("ollama", f"Ollama · {_setting('ai/ollama_model', 'local')}"))
        if _setting("ai/custom_url"):
            out.append(("custom", f"Custom · {_setting('ai/custom_model', 'model')}"))
        return out

    def _refresh_backend_combo(self):
        current = str(_setting("ai/provider", "auto") or "auto").lower()
        self.cmb_backend.blockSignals(True)
        self.cmb_backend.clear()
        for key, label in self._backend_choices():
            self.cmb_backend.addItem(label, key)
        idx = self.cmb_backend.findData(current)
        self.cmb_backend.setCurrentIndex(max(idx, 0))
        self.cmb_backend.blockSignals(False)

    def _on_backend_picked(self, idx: int):
        key = self.cmb_backend.itemData(idx)
        if not key:
            return
        if self._busy():
            # Revert the picker: the provider cannot change mid-turn.
            self._refresh_backend_combo()
            return
        settings.set("ai/provider", key)
        self.reload_provider()

    def reload_provider(self):
        self._refresh_backend_combo()
        self.provider = build_provider()
        if self.provider is None:
            self.session = None
            self._set_status(False, "No AI backend configured")
            self._render_setup_card()
            self._set_enabled(False)
            return
        ok, why = self.provider.available()
        if not ok:
            self.session = None
            self._set_status(False, why)
            self._render_setup_card(why)
            self._set_enabled(False)
            return
        history = self.session.history if self.session else []
        self.session = AssistantSession(
            self.provider,
            context_fn=self._context,
            tool_executor=self.bridge.call,
            tool_specs=TOOL_SPECS,
        )
        self.session.history = history
        self._set_status(True, self.provider.label())
        self._set_enabled(True)
        if not history:
            self.view.clear()
            self._render_welcome()
        self.chips.setVisible(not history)

    def _set_status(self, ok: bool, text: str):
        dot = theme.OK if ok else theme.WARN
        self.lbl_status.setText(
            f"<span style='color:{dot}'>●</span> {html.escape(text)}")

    def _set_enabled(self, on: bool):
        self.input.setEnabled(on)
        self.btn_send.setEnabled(on)
        self.btn_explain.setEnabled(on)
        self.chips.setEnabled(on)
        self._refresh_send_icon()

    def _render_welcome(self):
        self.view.setHtml(
            f"<div style='color:{theme.TEXT_DIM}; font-size:12px; margin:6px 2px'>"
            f"I can read your design, explain stability and flight results, pick motors, "
            f"and make changes when you ask. Numbers come from K2's solvers, not guesses."
            f"</div>")

    def _render_setup_card(self, why: str = ""):
        rows = "".join(
            f"<tr><td style='padding:3px 6px'>{'✔' if ok else '✖'}</td>"
            f"<td style='padding:3px 6px'>{html.escape(name)}</td>"
            f"<td style='padding:3px 6px;color:{theme.TEXT_DIM}'>{html.escape(reason)}</td></tr>"
            for name, ok, reason in describe_auto_state())
        self.view.setHtml(
            f"<table width='100%' cellspacing='0' cellpadding='10' bgcolor='{theme.PANEL}'>"
            f"<tr><td>"
            f"<div style='font-weight:600; font-size:13px; color:{theme.TEXT_BRIGHT}'>Set up the assistant</div>"
            f"<div style='color:{theme.TEXT_DIM}; margin:4px 0 8px 0'>{html.escape(why)}</div>"
            f"<table cellspacing='0'>{rows}</table>"
            f"<div style='margin-top:10px; font-weight:600'>Free options</div>"
            f"<ol style='margin:4px 0 0 16px'>"
            f"<li><b>Gemini</b> (cloud, recommended) — free key at "
            f"<a href='https://aistudio.google.com/apikey' style='color:{theme.ACCENT}'>aistudio.google.com/apikey</a>, "
            f"paste it in <i>Settings → AI Assistant</i>.</li>"
            f"<li><b>Ollama</b> (local, offline) — install from "
            f"<a href='https://ollama.com/download' style='color:{theme.ACCENT}'>ollama.com</a>, then "
            f"<code>ollama pull qwen3:1.7b</code>.</li>"
            f"</ol>"
            f"<div style='color:{theme.TEXT_DIM}; margin-top:8px'>Then use ⋯ → Reconnect provider.</div>"
            f"</td></tr></table>")
        self.chips.hide()

    def _open_settings(self):
        fn = getattr(self.mw, "_on_settings", None)
        if fn:
            fn()
            self.reload_provider()

    # -------------------------------------------------------------- context
    def _context(self) -> str:
        asm = None
        ws = getattr(self.mw, "design_ws", None)
        if ws is not None and hasattr(ws, "get_assembly"):
            try:
                asm = ws.get_assembly()
            except Exception:
                asm = None
        return render_context(summarize_state(self.mw.engine.state, asm))

    # ----------------------------------------------------------------- chat
    def send(self):
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._send_text(text)

    def _send_text(self, text: str):
        if self.session is None or self._busy():
            return
        self._append_user(text)
        self._start(text, one_shot=False)

    def _on_send_clicked(self):
        if self._busy():
            self._cancel()
        else:
            self.send()

    def explain_flight(self):
        if self.session is None or self._busy():
            return
        s = self.mw.engine.state
        if not s.max_altitude:
            self._append_notice("Run a simulation first, then I can debrief the flight.")
            return
        notes = []
        fail = self.bridge._sim_failure
        if fail:
            notes.append(f"ABORTED — {fail['title']}: {fail['detail']}")
        notes.append(f"Final phase: {s.sim_phase}; parachute deployed: {s.parachute_deployed}")
        self._append_user("Explain the last flight.")
        self._start(prompts.EXPLAIN_FLIGHT.format(notes="\n".join(notes)), one_shot=True)

    def clear_conversation(self):
        if self._busy():
            return
        if self.session:
            self.session.clear()
        self.view.clear()
        self.reload_provider()

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _cancel(self):
        if self.session:
            self.session.cancel()
        self._set_activity("Stopping")

    def _start(self, text: str, one_shot: bool):
        self.chips.hide()
        self._stream_buf = ""
        self._open_assistant_card()
        self.btn_explain.setEnabled(False)
        self._set_activity("Thinking")
        self._pulse.start()
        w = _Worker(self.session, text, one_shot)
        w.text.connect(self._on_text)
        w.tool_call.connect(self._on_tool_call)
        w.tool_result.connect(self._on_tool_result)
        w.error.connect(self._on_error)
        w.finished_turn.connect(self._on_finished)
        # Drop our reference only once the QThread itself has exited; letting
        # the Python object die while run() is still unwinding destroys a
        # running QThread and crashes the process.
        w.finished.connect(self._on_worker_exited)
        self._worker = w
        w.start()
        self._refresh_send_icon()

    # ------------------------------------------------- worker → GUI thread
    def _on_text(self, delta: str):
        self._stream_buf += delta
        if self._activity_text != "Writing":
            self._set_activity("Writing")
        self._replace_streaming_block(self._stream_buf)

    def _on_tool_call(self, name: str, args: dict):
        self._flush_stream()
        pretty = ", ".join(f"{k}={v}" for k, v in (args or {}).items())
        label = _TOOL_LABELS.get(name, name)
        self._set_activity(label)
        self._append_html(
            f"<table cellspacing='0' cellpadding='4' bgcolor='{theme.SUNKEN}'>"
            f"<tr><td style='color:{theme.INFO}; font-size:11px'>⚙ {html.escape(label)}</td>"
            f"<td style='color:{theme.TEXT_FAINT}; font-size:11px; font-family:{theme.MONO}'>"
            f"{html.escape(name)}({html.escape(pretty)})</td></tr></table>", new_block=True)

    def _on_tool_result(self, name: str, result: str):
        short = result if len(result) < 200 else result[:197] + "…"
        self._append_html(
            f"<div style='color:{theme.TEXT_FAINT}; font-size:10px; font-family:{theme.MONO}; "
            f"margin:0 0 6px 8px'>{html.escape(short)}</div>", new_block=True)
        self._stream_buf = ""
        self._assistant_anchor = self._end_pos()

    def _on_error(self, msg: str):
        self._flush_stream()
        self._append_html(
            f"<table cellspacing='0' cellpadding='8' bgcolor='{theme.SUNKEN}' width='100%'>"
            f"<tr><td style='color:{theme.ERR}; font-size:11px'>⚠ {html.escape(msg)}</td></tr></table>",
            new_block=True)

    def _on_finished(self):
        self._flush_stream()
        self._pulse.stop()
        self.lbl_activity.hide()
        self.btn_explain.setEnabled(True)
        self._refresh_send_icon()
        self.input.setFocus()

    def _on_worker_exited(self):
        w = self.sender()
        if w is self._worker:
            self._worker = None
        if w is not None:
            w.deleteLater()
        self._refresh_send_icon()

    def shutdown(self, wait_ms: int = 3000):
        """Cancel any in-flight turn and wait for the worker thread; call on app close."""
        w = self._worker
        if w is None:
            return
        if self.session:
            self.session.cancel()
        if w.isRunning():
            w.wait(wait_ms)

    # ------------------------------------------------------------ activity
    def _set_activity(self, text: str):
        self._activity_text = text
        self.lbl_activity.setText(text + "…")
        self.lbl_activity.show()

    def _tick_pulse(self):
        self._dots = (self._dots + 1) % 4
        self.lbl_activity.setText(self._activity_text + "." * self._dots)

    # ----------------------------------------------------------- rendering
    def _end_pos(self) -> int:
        c = self.view.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        return c.position()

    def _append_html(self, h: str, new_block: bool = False):
        c = self.view.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        # insertHtml appends into the current block, so a card would run into
        # the tail of the previous message; start a fresh block.
        if new_block and not self.view.document().isEmpty():
            fmt = QTextBlockFormat()
            fmt.setTopMargin(10)
            c.insertBlock(fmt)
        c.insertHtml(h)
        self.view.setTextCursor(c)
        self.view.ensureCursorVisible()

    def _append_user(self, text: str):
        self._append_html(
            f"<table width='100%' cellspacing='0' cellpadding='0'><tr>"
            f"<td width='18%'></td>"
            f"<td bgcolor='{theme.RAISED}' style='padding:8px 10px'>"
            f"<div style='color:{theme.ACCENT}; font-size:10px; font-weight:600'>YOU</div>"
            f"<div style='color:{theme.TEXT}; white-space:pre-wrap'>{html.escape(text)}</div>"
            f"</td></tr></table>", new_block=True)

    def _append_notice(self, text: str):
        self._append_html(
            f"<div style='color:{theme.WARN}; font-size:11px'>{html.escape(text)}</div>",
            new_block=True)

    def _open_assistant_card(self):
        self._append_html(
            f"<div style='color:{theme.INFO}; font-size:10px; font-weight:600'>K2 ASSISTANT</div>",
            new_block=True)
        self._assistant_anchor = self._end_pos()

    def _replace_streaming_block(self, md: str):
        c = self.view.textCursor()
        c.setPosition(self._assistant_anchor)
        c.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        c.removeSelectedText()
        c.insertBlock()
        c.insertMarkdown(md)
        self.view.setTextCursor(c)
        self.view.ensureCursorVisible()

    def _flush_stream(self):
        if self._stream_buf:
            self._replace_streaming_block(self._stream_buf)
            self._stream_buf = ""
        self._assistant_anchor = self._end_pos()
