"""
K2 AeroSim - application theme.

Single source of truth for colours, in two palettes. Widget code reads the
module-level constants, which point at whichever palette is active:

    from ui import theme
    label.setStyleSheet(f"color: {theme.ACCENT};")

Constants are rebound by :func:`set_mode`, so a widget built after a switch
picks up the new palette. Widgets already on screen keep whatever colours they
baked in at construction until they are rebuilt - the global stylesheet does
update live.
"""

DARK = {
    # surfaces
    "BG": "#0e0e10",          # window ground
    "PANEL": "#16161a",       # docks, panels, tab strip
    "RAISED": "#202024",      # buttons, headers, hover fills
    "SUNKEN": "#0a0a0b",      # console, viewport
    "LINE": "#2e2e34",        # hairlines, control borders
    "LINE_STRONG": "#45454d",
    # text
    "TEXT_BRIGHT": "#f0f0f2",
    "TEXT": "#d8d8dc",
    "TEXT_DIM": "#8a8a92",
    "TEXT_FAINT": "#5a5a62",
    # accent
    "ACCENT": "#e8843a",
    "ACCENT_HOVER": "#f09550",
    "ACCENT_DEEP": "#c96a26",
    # semantic
    "OK": "#5fb87a",
    "WARN": "#d9a441",
    "ERR": "#e05c55",
    "ERR_DEEP": "#c0453f",
    "INFO": "#6fa8c8",
    # roles that differ in kind (not just shade) between the palettes
    "FIELD_BG": "#202024",       # input wells
    "SEL_BG": "#202024",         # list / tree selection fill
    "BTN_BG": "#202024",         # default button fill
    "BTN_HOVER": "#2e2e34",
    "PRIMARY_BG": "transparent",  # dark: outlined primary
    "PRIMARY_FG": "#e8843a",
    "PRIMARY_HOVER_BG": "rgba(232, 132, 58, 0.14)",
    "PRIMARY_HOVER_FG": "#f09550",
    "SUCCESS_HOVER_BG": "rgba(95, 184, 122, 0.14)",
    "DANGER_HOVER_BG": "rgba(224, 92, 85, 0.14)",
    # misc
    "SELECTION_TEXT": "#ffffff",
    "MONO": "'Cascadia Mono', 'Consolas', monospace",
}

LIGHT = {
    "BG": "#ffffff",
    "PANEL": "#ffffff",
    "RAISED": "#f3f4f6",
    "SUNKEN": "#fafbfc",
    "LINE": "#e5e7eb",
    "LINE_STRONG": "#cbd0d8",
    "TEXT_BRIGHT": "#111827",
    "TEXT": "#1f2937",
    "TEXT_DIM": "#6b7280",
    "TEXT_FAINT": "#9ca3af",
    "ACCENT": "#2563eb",
    "ACCENT_HOVER": "#1d4ed8",
    "ACCENT_DEEP": "#1e40af",
    "OK": "#15803d",
    "WARN": "#b45309",
    "ERR": "#b91c1c",
    "ERR_DEEP": "#991b1b",
    "INFO": "#0369a1",
    "FIELD_BG": "#ffffff",
    "SEL_BG": "#e8effc",
    "BTN_BG": "#ffffff",
    "BTN_HOVER": "#f3f4f6",
    "PRIMARY_BG": "#2563eb",      # light: filled primary
    "PRIMARY_FG": "#ffffff",
    "PRIMARY_HOVER_BG": "#1d4ed8",
    "PRIMARY_HOVER_FG": "#ffffff",
    "SUCCESS_HOVER_BG": "#eaf6ee",
    "DANGER_HOVER_BG": "#fdeceb",
    "SELECTION_TEXT": "#ffffff",
    "MONO": "'Cascadia Mono', 'Consolas', monospace",
}

PALETTES = {"dark": DARK, "light": LIGHT}
DEFAULT_MODE = "dark"

_mode = DEFAULT_MODE


def set_mode(mode: str) -> str:
    """Make ``mode`` ('dark' or 'light') the active palette. Returns the mode."""
    global _mode
    if mode not in PALETTES:
        mode = DEFAULT_MODE
    _mode = mode
    globals().update(PALETTES[mode])
    return mode


def mode() -> str:
    return _mode


def palette() -> dict:
    return PALETTES[_mode]


def stylesheet(for_mode: str = None) -> str:
    """Full application stylesheet for the active (or given) palette."""
    return _QSS_TEMPLATE.format(**PALETTES.get(for_mode or _mode, DARK))


_QSS_TEMPLATE = """
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}}

QMainWindow {{ background-color: {BG}; }}
QMainWindow::separator {{ background-color: {LINE}; width: 1px; height: 1px; }}
QMainWindow::separator:hover {{ background-color: {LINE_STRONG}; }}

/* ---- menu bar ---- */
QMenuBar {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border-bottom: 1px solid {LINE};
    padding: 2px 4px;
}}
QMenuBar::item {{ padding: 5px 11px; border-radius: 4px; background: transparent; }}
QMenuBar::item:selected {{ background-color: {RAISED}; color: {TEXT_BRIGHT}; }}
QMenuBar::item:pressed {{ background-color: {RAISED}; color: {ACCENT}; }}

QMenu {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 26px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {RAISED}; color: {TEXT_BRIGHT}; }}
QMenu::item:disabled {{ color: {TEXT_FAINT}; }}
QMenu::separator {{ height: 1px; background-color: {LINE}; margin: 4px 8px; }}

/* ---- tab strip: active marked by an orange underline ---- */
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {LINE};
    background-color: {BG};
}}
QTabBar {{ background-color: {PANEL}; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 14px;
    margin: 0px;
    color: {TEXT_DIM};
    font-size: 13px;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT_BRIGHT}; }}
QTabBar::scroller {{ width: 28px; }}
QTabBar QToolButton {{
    background-color: {RAISED};
    border: 1px solid {LINE};
    border-radius: 4px;
    margin: 3px 1px;
    width: 20px;
    color: {TEXT_DIM};
}}
QTabBar QToolButton:hover {{ color: {ACCENT}; border-color: {LINE_STRONG}; }}

/* ---- toolbars ---- */
QToolBar {{
    background-color: {PANEL};
    border: none;
    padding: 2px 6px;
    spacing: 2px;
}}
QToolBar::separator {{ width: 1px; background-color: {LINE}; margin: 6px 6px; }}
QToolButton {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 8px;
}}
QToolButton:hover {{ background-color: {RAISED}; color: {TEXT_BRIGHT}; }}
QToolButton:pressed, QToolButton:checked {{ background-color: {RAISED}; color: {ACCENT}; }}
QToolButton:disabled {{ color: {TEXT_FAINT}; }}

/* ---- docks: plain caption, no gradient ---- */
QDockWidget {{ color: {TEXT_DIM}; font-size: 11px; }}
QDockWidget::title {{
    background-color: {PANEL};
    border-top: 1px solid {LINE};
    border-bottom: 1px solid {LINE};
    padding: 6px 10px;
    text-align: left;
    color: {TEXT_DIM};
}}
QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent; border: none; padding: 2px;
}}
QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {RAISED}; border-radius: 3px;
}}

QScrollArea {{ border: none; background-color: transparent; }}
QFrame {{ border: none; }}

/* ---- labels ---- */
QLabel {{ color: {TEXT}; background: transparent; }}
QLabel[panelTitle="true"] {{
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 10px 12px 8px 12px;
    border-bottom: 1px solid {LINE};
}}
QLabel[heading="true"] {{
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 10px 0px 4px 0px;
}}
QLabel[unit="true"] {{ color: {TEXT_FAINT}; font-size: 11px; }}
QLabel[value="true"] {{
    color: {TEXT_BRIGHT};
    font-family: {MONO};
    font-size: 13px;
}}
QLabel[metric="true"] {{
    color: {ACCENT};
    font-size: 17px;
    font-weight: 600;
}}

/* ---- inputs ---- */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox, QDateEdit, QTimeEdit {{
    background-color: {FIELD_BG};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 5px 8px;
    color: {TEXT_BRIGHT};
    selection-background-color: {ACCENT_DEEP};
    selection-color: #ffffff;
}}
QDoubleSpinBox, QSpinBox {{ font-family: {MONO}; }}
QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{
    border-color: {LINE_STRONG};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
    background-color: {PANEL}; color: {TEXT_FAINT}; border-color: {LINE};
}}

QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background-color: transparent;
    border-left: 1px solid {LINE};
    width: 17px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: {LINE};
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    image: none; width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {TEXT_DIM};
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {TEXT_DIM};
}}

QComboBox {{ min-width: 110px; }}
QComboBox::drop-down {{ border: none; width: 20px; background: transparent; }}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
    padding: 4px;
    selection-background-color: {SEL_BG};
    selection-color: {ACCENT};
    outline: none;
}}

/* ---- cards / group boxes ---- */
QGroupBox {{
    border: 1px solid {LINE};
    border-radius: 6px;
    margin-top: 10px;
    padding: 16px 10px 10px 10px;
    background-color: {PANEL};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0px 6px;
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    background-color: {PANEL};
}}

/* ---- text / console ---- */
QTextEdit, QPlainTextEdit {{
    background-color: {SUNKEN};
    color: {TEXT};
    border: 1px solid {LINE};
    border-radius: 6px;
    font-family: {MONO};
    font-size: 12px;
    padding: 6px 8px;
    selection-background-color: {ACCENT_DEEP};
}}

/* ---- item views ---- */
QTreeView, QTreeWidget, QTableView, QTableWidget, QListView, QListWidget {{
    background-color: {PANEL};
    alternate-background-color: {BG};
    border: 1px solid {LINE};
    border-radius: 6px;
    outline: none;
    selection-background-color: {SEL_BG};
    selection-color: {ACCENT};
}}
QTreeView::item, QTableView::item, QListView::item {{
    padding: 5px 4px;
    border: none;
}}
QTreeView::item:hover, QListView::item:hover {{ background-color: {RAISED}; }}
QTreeView::item:selected, QTableView::item:selected, QListView::item:selected {{
    background-color: {SEL_BG};
    color: {ACCENT};
}}
QHeaderView::section {{
    background-color: {PANEL};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {LINE};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 12px; border: none; margin: 0; }}
QScrollBar::handle:vertical {{
    background-color: {LINE}; min-height: 30px;
    border-radius: 3px; margin: 2px 4px;
}}
QScrollBar::handle:vertical:hover {{ background-color: {LINE_STRONG}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; border: none; margin: 0; }}
QScrollBar::handle:horizontal {{
    background-color: {LINE}; min-width: 30px;
    border-radius: 3px; margin: 4px 2px;
}}
QScrollBar::handle:horizontal:hover {{ background-color: {LINE_STRONG}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- status bar ---- */
QStatusBar {{
    background-color: {PANEL};
    border-top: 1px solid {LINE};
    color: {TEXT_DIM};
    font-size: 11px;
    padding: 2px 8px;
}}
QStatusBar::item {{ border: none; }}

/* ---- buttons ---- */
QPushButton {{
    background-color: {BTN_BG};
    color: {TEXT};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: {BTN_HOVER}; color: {TEXT_BRIGHT}; border-color: {LINE_STRONG}; }}
QPushButton:pressed {{ background-color: {PANEL}; }}
QPushButton:disabled {{ background-color: {PANEL}; color: {TEXT_FAINT}; border-color: {LINE}; }}
QPushButton[primary="true"] {{
    background-color: {PRIMARY_BG};
    border: 1px solid {ACCENT};
    color: {PRIMARY_FG};
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{
    background-color: {PRIMARY_HOVER_BG};
    border-color: {ACCENT_HOVER};
    color: {PRIMARY_HOVER_FG};
}}
QPushButton[success="true"] {{
    background-color: transparent;
    border: 1px solid {OK};
    color: {OK};
    font-weight: 600;
}}
QPushButton[success="true"]:hover {{ background-color: {SUCCESS_HOVER_BG}; }}
QPushButton[danger="true"] {{
    background-color: transparent;
    border: 1px solid {ERR};
    color: {ERR};
}}
QPushButton[danger="true"]:hover {{ background-color: {DANGER_HOVER_BG}; }}
QPushButton:flat {{ background: transparent; border: none; color: {TEXT_DIM}; }}

QCheckBox, QRadioButton {{ color: {TEXT}; spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {LINE_STRONG};
    background-color: {RAISED};
    border-radius: 3px;
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACCENT}; border-color: {ACCENT};
}}

/* ---- progress / sliders ---- */
QProgressBar {{
    background-color: {RAISED};
    border: none;
    border-radius: 3px;
    text-align: center;
    color: {TEXT_DIM};
    font-size: 11px;
    height: 6px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}

QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {TEXT_DIM}; width: 12px; margin: -5px 0; border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DEEP}; border-radius: 2px; }}

QSplitter::handle {{ background-color: {LINE}; }}
QSplitter::handle:hover {{ background-color: {LINE_STRONG}; }}

QToolTip {{
    background-color: {RAISED};
    color: {TEXT_BRIGHT};
    border: 1px solid {LINE_STRONG};
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 12px;
}}
"""


# Bind the constants (BG, TEXT, ACCENT, ...) for the default palette.
set_mode(DEFAULT_MODE)


def value_qss(color: str = None, size: int = 13, weight: int = 600) -> str:
    """Inline style for a numeric read-out, in the active palette."""
    return (f"color:{color or TEXT_BRIGHT}; font-family:{MONO}; "
            f"font-size:{size}px; font-weight:{weight}; "
            f"padding:2px 0px; background:transparent;")


def apply_matplotlib_theme():
    """Point matplotlib's defaults at the application palette.

    Call once at startup; per-plot colour arguments still win where a chart
    needs to say something specific.
    """
    try:
        import matplotlib as mpl
    except Exception:
        return
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "figure.edgecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": LINE,
        "axes.labelcolor": TEXT_DIM,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "normal",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=[
            ACCENT, INFO, OK, WARN, "#a98bd0", ERR, "#7fbfb0", TEXT_DIM,
        ]),
        "grid.color": LINE,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "text.color": TEXT,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": PANEL,
        "legend.edgecolor": LINE,
        "legend.labelcolor": TEXT,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "font.size": 10,
    })
