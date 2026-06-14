"""
K2 AeroSim — Dark Engineering Theme
=======================================
Professional QSS stylesheet inspired by aerospace engineering software.
Dark theme with cyan/teal accents for a technical, premium feel.
"""


DARK_STYLESHEET = """
/* ═══════════════════════════════════════════════════════════════════
   K2 AEROSPACE — DARK ENGINEERING THEME
   ═══════════════════════════════════════════════════════════════════ */

/* ── Global ─────────────────────────────────────────────────────── */
QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}

/* ── Main Window ────────────────────────────────────────────────── */
QMainWindow {
    background-color: #0d1117;
}

QMainWindow::separator {
    background-color: #21262d;
    width: 2px;
    height: 2px;
}

QMainWindow::separator:hover {
    background-color: #58a6ff;
}

/* ── Menu Bar ───────────────────────────────────────────────────── */
QMenuBar {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 2px;
}

QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #21262d;
}

QMenu {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #21262d;
    margin: 4px 8px;
}

/* ── Toolbar ────────────────────────────────────────────────────── */
QToolBar {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 4px 8px;
    spacing: 6px;
}

QToolBar::separator {
    width: 1px;
    background-color: #30363d;
    margin: 4px 6px;
}

QToolButton {
    background-color: transparent;
    color: #8b949e;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #21262d;
    color: #58a6ff;
    border-color: #30363d;
}

QToolButton:pressed {
    background-color: #1f6feb;
    color: #ffffff;
}

/* ── Dock Widgets ───────────────────────────────────────────────── */
QDockWidget {
    color: #c9d1d9;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a2332, stop:1 #161b22);
    border: 1px solid #21262d;
    border-radius: 4px;
    padding: 8px 12px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #58a6ff;
}

QDockWidget::close-button, QDockWidget::float-button {
    background: transparent;
    border: none;
    padding: 2px;
}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {
    background-color: #21262d;
    border-radius: 3px;
}

/* ── Scroll Areas & Frames ──────────────────────────────────────── */
QScrollArea {
    border: none;
    background-color: #0d1117;
}

QFrame {
    border: none;
}

/* ── Labels ─────────────────────────────────────────────────────── */
QLabel {
    color: #8b949e;
    background: transparent;
}

QLabel[heading="true"] {
    color: #58a6ff;
    font-weight: 700;
    font-size: 13px;
    padding: 8px 0px 4px 0px;
}

QLabel[unit="true"] {
    color: #484f58;
    font-size: 11px;
}

QLabel[value="true"] {
    color: #e6edf3;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 14px;
    font-weight: 600;
}

/* ── Input Fields ───────────────────────────────────────────────── */
QLineEdit {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
    font-size: 13px;
    selection-background-color: #1f6feb;
}

QLineEdit:focus {
    border-color: #58a6ff;
    background-color: #161b22;
}

QLineEdit:hover {
    border-color: #484f58;
}

/* ── Spin Boxes ─────────────────────────────────────────────────── */
QDoubleSpinBox, QSpinBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
    font-family: 'Cascadia Code', 'Consolas', monospace;
    font-size: 13px;
}

QDoubleSpinBox:focus, QSpinBox:focus {
    border-color: #58a6ff;
    background-color: #161b22;
}

QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {
    background-color: #21262d;
    border: none;
    width: 20px;
    border-radius: 3px;
}

QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {
    background-color: #30363d;
}

QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #8b949e;
    width: 0;
    height: 0;
}

QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8b949e;
    width: 0;
    height: 0;
}

/* ── Combo Box ──────────────────────────────────────────────────── */
QComboBox {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    color: #e6edf3;
    font-size: 13px;
    min-width: 120px;
}

QComboBox:focus {
    border-color: #58a6ff;
}

QComboBox:hover {
    border-color: #484f58;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
    background: transparent;
}

QComboBox::down-arrow {
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #8b949e;
    width: 0;
    height: 0;
}

QComboBox QAbstractItemView {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 4px;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
    outline: none;
}

/* ── Group Box ──────────────────────────────────────────────────── */
QGroupBox {
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 20px;
    background-color: #0d1117;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    color: #58a6ff;
    font-weight: 600;
    font-size: 12px;
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 4px;
    left: 12px;
}

/* ── Text Edit (Console) ────────────────────────────────────────── */
QTextEdit {
    background-color: #010409;
    color: #7ee787;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: #1f6feb;
}

/* ── Scroll Bars ────────────────────────────────────────────────── */
QScrollBar:vertical {
    background-color: #0d1117;
    width: 10px;
    border: none;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #30363d;
    min-height: 30px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background-color: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background-color: #0d1117;
    height: 10px;
    border: none;
    border-radius: 5px;
}

QScrollBar::handle:horizontal {
    background-color: #30363d;
    min-width: 30px;
    border-radius: 5px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #484f58;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Status Bar ─────────────────────────────────────────────────── */
QStatusBar {
    background-color: #161b22;
    border-top: 1px solid #21262d;
    color: #8b949e;
    font-size: 12px;
    padding: 2px 8px;
}

QStatusBar::item {
    border: none;
}

/* ── Tab Widget ─────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #21262d;
    border-radius: 6px;
    background-color: #0d1117;
}

QTabBar::tab {
    background-color: #161b22;
    border: 1px solid #21262d;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #8b949e;
}

QTabBar::tab:selected {
    background-color: #0d1117;
    color: #58a6ff;
    border-bottom-color: #0d1117;
}

QTabBar::tab:hover:!selected {
    background-color: #21262d;
    color: #c9d1d9;
}

/* ── Tab-bar scroll buttons (shown when tabs overflow) ──────────── */
QTabBar::scroller {
    width: 30px;
}
QTabBar QToolButton {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    margin: 2px 1px;
    width: 22px;
    color: #c9d1d9;
}
QTabBar QToolButton:hover {
    background-color: #1f6feb;
    border-color: #1f6feb;
}
QTabBar QToolButton:disabled {
    background-color: #161b22;
    border-color: #21262d;
}

/* ── Push Buttons ───────────────────────────────────────────────── */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #30363d;
    border-color: #484f58;
    color: #e6edf3;
}

QPushButton:pressed {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #1f6feb;
}

QPushButton[primary="true"] {
    background-color: #1f6feb;
    color: #ffffff;
    border-color: #1f6feb;
}

QPushButton[primary="true"]:hover {
    background-color: #388bfd;
}

QPushButton[danger="true"] {
    background-color: #da3633;
    color: #ffffff;
    border-color: #da3633;
}

/* ── Progress Bar ───────────────────────────────────────────────── */
QProgressBar {
    background-color: #21262d;
    border: 1px solid #30363d;
    border-radius: 4px;
    text-align: center;
    color: #c9d1d9;
    height: 8px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #58a6ff);
    border-radius: 3px;
}

/* ── Splitter ───────────────────────────────────────────────────── */
QSplitter::handle {
    background-color: #21262d;
}

QSplitter::handle:hover {
    background-color: #58a6ff;
}

/* ── Tooltips ───────────────────────────────────────────────────── */
QToolTip {
    background-color: #1c2128;
    color: #e6edf3;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}
"""
