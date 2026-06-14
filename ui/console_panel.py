"""
K2 AeroSim — Console Panel
===============================
Terminal-styled log output panel with custom logging handler.
Routes Python logging messages to the UI with timestamps and color coding.
"""

import logging
from datetime import datetime
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QTextCursor, QColor


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
    LEVEL_COLORS = {
        "DEBUG": "#8b949e",
        "INFO": "#7ee787",
        "WARNING": "#d29922",
        "ERROR": "#f85149",
        "CRITICAL": "#ff7b72",
    }
    
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
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setFixedWidth(60)
        self.clear_btn.clicked.connect(self._clear_console)
        btn_layout.addWidget(self.clear_btn)
        
        btn_layout.addStretch()
        
        self.line_count_label = QPushButton("0 lines")
        self.line_count_label.setFlat(True)
        self.line_count_label.setEnabled(False)
        btn_layout.addWidget(self.line_count_label)
        
        layout.addLayout(btn_layout)
        
        # ── Text output ──
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text_edit)
        
        self._line_count = 0
    
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
        color = self.LEVEL_COLORS.get(level, "#c9d1d9")
        level_tag = f"[{level:>8}]"
        
        html = (
            f'<span style="color:#484f58;">{timestamp}</span> '
            f'<span style="color:{color}; font-weight:600;">{level_tag}</span> '
            f'<span style="color:#c9d1d9;">{message}</span>'
        )
        
        self.text_edit.append(html)
        
        # Auto-scroll to bottom
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
        
        self._line_count += 1
        self.line_count_label.setText(f"{self._line_count} lines")
    
    def _clear_console(self):
        """Clear all console output."""
        self.text_edit.clear()
        self._line_count = 0
        self.line_count_label.setText("0 lines")
    
    def log(self, message: str, level: str = "INFO"):
        """Directly log a message to the console (convenience method)."""
        self._append_log(message, level)
