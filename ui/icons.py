"""
K2 AeroSim — Icon helper
==========================
Central registry mapping semantic keys to scalable FontAwesome icons
(via qtawesome). Replaces the old emoji glyphs in tabs, toolbar actions
and buttons.

Usage:
    from ui.icons import icon
    action.setIcon(icon("save"))
    btn.setIcon(icon("run", color="#5fb87a"))

Degrades gracefully: if qtawesome is missing the calls return a null
QIcon so the UI still runs (just without icons).
"""

import os
# qtawesome routes through QtPy; pin it to the binding the app uses so we
# don't accidentally load a second Qt binding alongside PyQt6.
os.environ.setdefault("QT_API", "pyqt6")

import logging
from PyQt6.QtGui import QIcon

from ui import theme

logger = logging.getLogger("K2.Icons")

try:
    import qtawesome as qta
    _HAS_QTA = True
except Exception as e:  # pragma: no cover - optional dependency
    qta = None
    _HAS_QTA = False
    logger.warning("qtawesome unavailable (%s) — UI icons disabled", e)

# Fallback tint; the live default follows the active palette.
DEFAULT_COLOR = "#d8d8dc"


def _default_color() -> str:
    """Icon tint for the palette in force right now."""
    try:
        from ui import theme
        return theme.TEXT
    except Exception:
        return DEFAULT_COLOR

# Semantic key -> FontAwesome spec. Colours can be overridden per-call.
_MAP = {
    # ── Main workspace tabs ──
    "design":        "fa5s.drafting-compass",
    "propulsion":    "fa5s.fire",
    "cfd":           "fa5s.water",
    "structures":    "fa5s.building",
    "dynamics":      "fa5s.wind",
    "avionics":      "fa5s.satellite-dish",
    "simulation":    "fa5s.rocket",
    "mission":       "fa5s.globe-americas",
    "results":       "fa5s.chart-bar",
    "montecarlo":    "fa5s.dice",
    "optimization":  "fa5s.bullseye",

    # ── Toolbar / file ops ──
    "new":           "fa5s.file",
    "open":          "fa5s.folder-open",
    "save":          "fa5s.save",
    "save_as":       "fa5s.clone",
    "import":        "fa5s.file-import",
    "run":           "fa5s.play",
    "pause":         "fa5s.pause",
    "stop":          "fa5s.stop",
    "reset":         "fa5s.undo",
    "refresh":       "fa5s.sync-alt",
    "reset_view":    "fa5s.expand",
    "settings":      "fa5s.cog",
    "ai":            "fa5s.robot",
    "send":          "fa5s.paper-plane",
    "debrief":       "fa5s.clipboard-list",
    "more":          "fa5s.ellipsis-h",
    "close":         "fa5s.times",
    "dock_out":      "fa5s.external-link-alt",   # float a dock panel
    "dock_in":       "fa5s.window-restore",      # re-dock a floating panel
    "update":        "fa5s.cloud-download-alt",

    # ── Generic actions ──
    "add":           "fa5s.plus",
    "export":        "fa5s.download",
    "duplicate":     "fa5s.clone",
    "delete":        "fa5s.trash-alt",
    "move_up":       "fa5s.arrow-up",
    "move_down":     "fa5s.arrow-down",
    "search":        "fa5s.search",
    "report":        "fa5s.file-pdf",
    "screenshot":    "fa5s.camera",
    "browse":        "fa5s.folder-open",
    "probe":         "fa5s.crosshairs",
    "inject":        "fa5s.sign-in-alt",
    "map_fem":       "fa5s.project-diagram",

    # ── Rocket components (design palette + tree) ──
    "comp_nosecone":   "fa5s.location-arrow",
    "comp_bodytube":   "fa5s.grip-lines-vertical",
    "comp_transition": "fa5s.compress-alt",
    "comp_finset":     "fa5s.fighter-jet",
    "comp_nozzle":     "fa5s.filter",
    "comp_innertube":  "fa5s.circle-notch",
    "comp_ring":       "fa5s.dot-circle",
    "comp_bulkhead":   "fa5s.stop-circle",
    "comp_block":      "fa5s.square",
    "comp_parachute":  "fa5s.parachute-box",
    "comp_cord":       "fa5s.grip-lines",
    "comp_mass":       "fa5s.weight-hanging",
    "comp_lug":        "fa5s.minus-square",
    "comp_railbutton": "fa5s.circle",
    "comp_stage":      "fa5s.layer-group",

    # ── Analysis / structures ──
    "static":        "fa5s.wrench",
    "modal":         "fa5s.music",
    "thermal":       "fa5s.thermometer-half",
    "stress3d":      "fa5s.cube",
    "stress_profile":"fa5s.chart-area",
    "temperature":   "fa5s.thermometer-half",
    "deformation":   "fa5s.ruler-combined",
    "fin":           "fa5s.fighter-jet",
    "recovery":      "fa5s.parachute-box",
    "buckling":      "fa5s.compress-arrows-alt",
    "loadpath":      "fa5s.project-diagram",
    "failuremap":    "fa5s.map",
    "mass":          "fa5s.balance-scale",
    "flutter":       "fa5s.wind",
    "vibration":     "fa5s.wave-square",
    "aeroelastic":   "fa5s.feather-alt",

    # ── Optimization solutions ──
    "apogee":        "fa5s.trophy",
    "reliability":   "fa5s.shield-alt",
    "balanced":      "fa5s.bolt",

    # ── Status ──
    "ok":            "fa5s.check-circle",
    "error":         "fa5s.times-circle",
    "warn":          "fa5s.exclamation-triangle",
}


def icon(key, color=None):
    """Return a QIcon for a semantic key, or a null QIcon if unavailable."""
    if not _HAS_QTA:
        return QIcon()
    spec = _MAP.get(key)
    if spec is None:
        logger.debug("no icon mapped for key %r", key)
        return QIcon()
    try:
        return qta.icon(spec, color=color or _default_color())
    except Exception as e:  # pragma: no cover
        logger.debug("icon(%r) failed: %s", key, e)
        return QIcon()


def available():
    """True if the icon backend (qtawesome) loaded successfully."""
    return _HAS_QTA
