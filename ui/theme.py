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

import re

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
_prev_mode = DEFAULT_MODE


def set_mode(mode: str) -> str:
    """Make ``mode`` ('dark' or 'light') the active palette. Returns the mode."""
    global _mode, _prev_mode
    if mode not in PALETTES:
        mode = DEFAULT_MODE
    if mode != _mode:
        _prev_mode = _mode
    _mode = mode
    globals().update(PALETTES[mode])
    return mode


def previous_mode() -> str:
    """The palette in force before the last change - what widgets may still hold."""
    return _prev_mode


def mode() -> str:
    return _mode


def palette() -> dict:
    return PALETTES[_mode]


def stylesheet(for_mode: str = None) -> str:
    """Full application stylesheet for the active (or given) palette."""
    pal = dict(PALETTES.get(for_mode or _mode, DARK))
    pal.update(_arrow_images(pal))
    return _QSS_TEMPLATE.format(**pal)


_CHEVRON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="14" viewBox="0 0 10 14">'
                '<path d="{d}" fill="none" stroke="{c}" stroke-width="2.2" '
                'stroke-linecap="round" stroke-linejoin="round"/></svg>')
_CHEVRON_PATHS = {"left": "M7 2 L2.5 7 L7 12", "right": "M3 2 L7.5 7 L3 12"}


def _arrow_images(pal: dict) -> dict:
    """Write chevron SVGs for this palette and return their QSS ``url()`` keys.

    QSS cannot draw a triangle reliably on tab-bar scroll buttons (the border
    trick renders as a filled square there), and Qt's stock arrow pixmaps
    are too small to read, so real images are generated per colour.
    """
    from pathlib import Path
    import hashlib
    out = {}
    try:
        from core.paths import user_data_dir
        d = Path(user_data_dir("ui_cache"))
    except Exception:
        import tempfile
        d = Path(tempfile.gettempdir()) / "k2_ui_cache"
        d.mkdir(parents=True, exist_ok=True)
    for role, color in (("ARROW", pal["TEXT"]), ("ARROW_DIM", pal["TEXT_FAINT"]),
                        ("ARROW_HOT", pal["ACCENT"])):
        for side, path in _CHEVRON_PATHS.items():
            svg = _CHEVRON_SVG.format(d=path, c=color)
            f = d / f"chevron_{side}_{hashlib.md5(svg.encode()).hexdigest()[:8]}.svg"
            try:
                if not f.exists():
                    f.write_text(svg, encoding="utf-8")
                out[f"{role}_{side.upper()}"] = f'url("{f.as_posix()}")'
            except Exception:
                out[f"{role}_{side.upper()}"] = "none"
    return out


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
/* Tab-strip overflow scrollers: draw explicit chevron triangles. The stock
   arrow pixmaps are ~6 px and unreadable on a dark panel, leaving what looks
   like two blank pills. */
QTabBar::scroller {{ width: 56px; }}
QTabBar QToolButton {{
    background-color: {RAISED};
    border: 1px solid {LINE};
    border-radius: 4px;
    margin: 4px 2px;
    min-width: 22px;
    color: {TEXT};
}}
QTabBar QToolButton:hover {{ background-color: {LINE}; border-color: {ACCENT}; }}
QTabBar QToolButton:disabled {{ background-color: transparent; border-color: {LINE}; }}
QTabBar QToolButton::left-arrow {{ image: {ARROW_LEFT}; width: 10px; height: 14px; }}
QTabBar QToolButton::right-arrow {{ image: {ARROW_RIGHT}; width: 10px; height: 14px; }}
QTabBar QToolButton::left-arrow:hover {{ image: {ARROW_HOT_LEFT}; }}
QTabBar QToolButton::right-arrow:hover {{ image: {ARROW_HOT_RIGHT}; }}
QTabBar QToolButton::left-arrow:disabled {{ image: {ARROW_DIM_LEFT}; }}
QTabBar QToolButton::right-arrow:disabled {{ image: {ARROW_DIM_RIGHT}; }}

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


# ---------------------------------------------------------------------------
# Live re-theming
#
# The global stylesheet repaints Qt chrome the moment it is re-applied, but
# anything that painted itself from the palette at construction - matplotlib
# figures, VTK viewports, per-widget stylesheets - keeps the old colours. These
# helpers walk a widget tree and bring those surfaces along, so switching the
# theme does not need a restart.
# ---------------------------------------------------------------------------

def _restyle_axes(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_DIM, which="both")
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(TEXT)
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_color(LINE)
    leg = ax.get_legend()
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor(PANEL)
        frame.set_edgecolor(LINE)
        for txt in leg.get_texts():
            txt.set_color(TEXT)
    for txt in ax.texts:
        txt.set_color(TEXT)


def restyle_matplotlib(root) -> int:
    """Repaint every embedded matplotlib canvas under `root`. Returns the count.

    Plotted data keeps its own colours - only the furniture (background, axes,
    ticks, labels, grid, legend) is re-coloured.
    """
    try:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    except Exception:
        return 0
    n = 0
    for canvas in root.findChildren(FigureCanvasQTAgg):
        fig = getattr(canvas, "figure", None)
        if fig is None:
            continue
        fig.patch.set_facecolor(BG)
        for ax in fig.get_axes():
            _restyle_axes(ax)
        for txt in fig.texts:
            txt.set_color(TEXT)
        try:
            canvas.draw_idle()
        except Exception:
            pass
        n += 1
    return n


def restyle_viewports(root) -> int:
    """Repaint the background of every embedded VTK/pyvista view under `root`."""
    from PyQt6.QtWidgets import QWidget
    seen, n = set(), 0
    for w in [root] + root.findChildren(QWidget):
        for attr in ("plotter", "_plotter"):
            pl = getattr(w, attr, None)
            if pl is None or id(pl) in seen:
                continue
            seen.add(id(pl))
            try:
                pl.set_background(BG, top=PANEL)
                n += 1
            except Exception:
                pass
    return n


def restyle_widgets(root) -> int:
    """Give widgets that own palette-dependent styles a chance to rebuild them.

    A widget opts in by defining ``retheme()``.
    """
    from PyQt6.QtWidgets import QWidget
    n = 0
    for w in [root] + root.findChildren(QWidget):
        fn = getattr(w, "retheme", None)
        if callable(fn):
            try:
                fn()
                n += 1
            except Exception:
                pass
    return n


# Which palette entry a colour most likely means, given the CSS property it
# was written against. Ordered: first match wins, so shared values resolve to
# the most plausible role (white as a background is a surface, white as text
# is on-accent ink and should stay white).
_SURFACE_KEYS = ("PANEL", "BG", "RAISED", "SUNKEN", "FIELD_BG", "BTN_BG",
                 "SEL_BG", "PRIMARY_BG", "LINE", "LINE_STRONG")
_INK_KEYS = ("SELECTION_TEXT", "PRIMARY_FG", "TEXT_BRIGHT", "TEXT", "TEXT_DIM",
             "TEXT_FAINT", "ACCENT", "ACCENT_HOVER", "ACCENT_DEEP", "OK", "WARN",
             "ERR", "ERR_DEEP", "INFO", "LINE_STRONG", "LINE")

_DECL_RE = re.compile(r"([-a-zA-Z]+)\s*:\s*([^;{}]*)", re.S)
_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}")


def _translate_stylesheet(qss: str, old: dict, new: dict) -> str:
    """Rewrite palette colours in `qss` from the `old` palette to the `new` one.

    Colours that are not palette entries (component colours, chart series) are
    left alone.
    """
    def fix_decl(m):
        prop, value = m.group(1), m.group(2)
        if "#" not in value:
            return m.group(0)
        keys = _SURFACE_KEYS if "background" in prop.lower() else _INK_KEYS

        def fix_hex(hm):
            hexv = hm.group(0).lower()
            for k in keys:
                if str(old.get(k, "")).lower() == hexv:
                    return new.get(k, hm.group(0))
            return hm.group(0)

        return f"{prop}:{_HEX_RE.sub(fix_hex, value)}"

    return _DECL_RE.sub(fix_decl, qss)


def restyle_stylesheets(root, from_mode: str = None) -> int:
    """Re-colour inline widget stylesheets left over from the previous palette.

    Most widgets build their stylesheet once, at construction, so a theme switch
    would otherwise leave them painted in the palette that was active back then.
    """
    from PyQt6.QtWidgets import QWidget
    old = PALETTES.get(from_mode or _prev_mode)
    new = PALETTES[_mode]
    if old is None or old is new:
        return 0
    n = 0
    for w in [root] + root.findChildren(QWidget):
        qss = w.styleSheet()
        if not qss or "#" not in qss:
            continue
        fixed = _translate_stylesheet(qss, old, new)
        if fixed != qss:
            w.setStyleSheet(fixed)
            n += 1
    return n


def restyle_all(root) -> dict:
    """Apply every live re-theming pass to a widget tree."""
    apply_matplotlib_theme()
    return {
        "stylesheets": restyle_stylesheets(root),
        "widgets": restyle_widgets(root),
        "figures": restyle_matplotlib(root),
        "viewports": restyle_viewports(root),
    }
