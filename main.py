"""
K2 Aerospace — Rocket Simulation Platform
============================================
Integrated aerospace digital twin for high-power and experimental rockets.

Entry point: python main.py
"""

import sys
import os
import logging
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# ── Ensure project root is in path ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.main_window import MainWindow
from ui.styles import DARK_STYLESHEET


def setup_logging():
    import io
    from logging.handlers import RotatingFileHandler
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k2_crash.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(utf8_stdout),
            RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=2, encoding="utf-8"),
        ]
    )
    logging.getLogger("pyvista").setLevel(logging.WARNING)
    logging.getLogger("vtk").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


def main():
    """Launch the K2 Aerospace application."""
    setup_logging()
    logger = logging.getLogger("K2")
    logger.info("Starting K2 Aerospace...")

    # ── Global crash handlers ────────────────────────────────────────────────
    def _excepthook(exc_type, exc_value, exc_tb):
        import traceback
        msg = "UNHANDLED EXCEPTION:\n" + "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.critical(msg)
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    # Qt thread exceptions (PyQt6 re-raises on the main thread)
    try:
        from PyQt6.QtCore import qInstallMessageHandler, QtMsgType
        def _qt_msg(msg_type, context, message):
            if msg_type in (QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                logger.critical(f"Qt [{msg_type.name}] {message}")
            else:
                logger.debug(f"Qt [{msg_type.name}] {message}")
        qInstallMessageHandler(_qt_msg)
    except Exception:
        pass

    # High DPI support
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # QtWebEngine (Cinematic view) must be imported BEFORE the QApplication is
    # created, or Qt aborts with "Qt.AA_ShareOpenGLContexts must be set".
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
    except Exception:
        pass  # WebEngine optional — Cinematic tab shows an install hint instead

    app = QApplication(sys.argv)
    app.setApplicationName("K2 Aerospace")
    app.setOrganizationName("K2")
    app.setApplicationVersion("0.1.0")

    # Apply dark theme
    app.setStyleSheet(DARK_STYLESHEET)

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("K2 Aerospace is ready")
    sys.exit(app.exec())


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
