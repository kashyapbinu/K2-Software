"""
K2 AeroSim — Console Panel
===============================
Terminal-styled log output panel with custom logging handler.
Routes Python logging messages to the UI with timestamps and color coding.
"""

import logging
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QTextEdit, QHBoxLayout,
                             QPushButton, QLabel)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor, QColor

from ui import theme


class LogSignalEmitter(QObject):
    """Thread-safe signal emitter for log messages."""
    log_received = pyqtSignal(str, str)  # message, level


class QtLogHandler(logging.Handler):
    """
    Custom logging handler that routes messages to the console panel.
    Thread-safe via Qt signals.
    """
    
    def __init__(self):
        super().__init__()
        self.emitter = LogSignalEmitter()
    
    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_received.emit(msg, record.levelname)


class ConsolePanel(QWidget):
    """
    Bottom dock panel displaying timestamped log output.
    Styled as a terminal with color-coded log levels.
    """
    
    # Color map for log levels
    @property
    def LEVEL_COLORS(self):
        return {
            "DEBUG": theme.TEXT_FAINT,
            "INFO": theme.TEXT_DIM,
            "WARNING": theme.WARN,
            "ERROR": theme.ERR,
            "CRITICAL": theme.ERR,
        }

    def retheme(self):
        """Re-apply palette-dependent styles after a theme switch."""
        self.status_label.setStyleSheet(f"color: {theme.TEXT_DIM}; padding-left: 4px;")
        self._refresh_counters()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._setup_logging()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # ── Button bar ──
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet(f"color: {theme.TEXT_DIM}; padding-left: 4px;")
        btn_layout.addWidget(self.status_label)

        btn_layout.addStretch()

        self.warn_label = QLabel("0 warnings")
        self.warn_label.setStyleSheet(f"color: {theme.TEXT_FAINT}; padding-right: 14px;")
        btn_layout.addWidget(self.warn_label)

        self.err_label = QLabel("0 errors")
        self.err_label.setStyleSheet(f"color: {theme.TEXT_FAINT}; padding-right: 14px;")
        btn_layout.addWidget(self.err_label)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(70)
        self.clear_btn.clicked.connect(self._clear_console)
        btn_layout.addWidget(self.clear_btn)
        
        layout.addLayout(btn_layout)
        
        # ── Text output ──
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text_edit)
        
        self._line_count = 0
        self._warn_count = 0
        self._err_count = 0
    
    def _setup_logging(self):
        """Set up the custom log handler and connect signals."""
        self.log_handler = QtLogHandler()
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.log_handler.emitter.log_received.connect(self._append_log)
        
        # Attach to root K2 logger
        root_logger = logging.getLogger("K2")
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.DEBUG)
    
    def _append_log(self, message: str, level: str):
        """Append a color-coded log message to the console."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color = self.LEVEL_COLORS.get(level, theme.TEXT)
        level_tag = f"[{level:>8}]"
        
        html = (
            f'<span style="color:#45454d;">{timestamp}</span> '
            f'<span style="color:{color}; font-weight:600;">{level_tag}</span> '
            f'<span style="color:#d8d8dc;">{message}</span>'
        )
        
        self.text_edit.append(html)
        
        # Auto-scroll to bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        
        self._line_count += 1
        if level == "WARNING":
            self._warn_count += 1
        elif level in ("ERROR", "CRITICAL"):
            self._err_count += 1
        self._refresh_counters()
        self.status_label.setText(message if len(message) < 90 else message[:87] + "...")
    
    def _clear_console(self):
        """Clear all console output."""
        self.text_edit.clear()
        self._line_count = 0
        self._warn_count = 0
        self._err_count = 0
        self._refresh_counters()
        self.status_label.setText("Ready.")

    def _refresh_counters(self):
        self.warn_label.setText(f"{self._warn_count} warning" + ("" if self._warn_count == 1 else "s"))
        self.warn_label.setStyleSheet(
            f"color: {theme.WARN if self._warn_count else theme.TEXT_FAINT}; padding-right: 14px;")
        self.err_label.setText(f"{self._err_count} error" + ("" if self._err_count == 1 else "s"))
        self.err_label.setStyleSheet(
            f"color: {theme.ERR if self._err_count else theme.TEXT_FAINT}; padding-right: 14px;")
    
    def log(self, message: str, level: str = "INFO"):
        """Directly log a message to the console (convenience method)."""
        self._append_log(message, level)
