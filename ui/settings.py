"""
Persisted application preferences.

Thin typed wrapper over ``QSettings`` (registry on Windows, ini elsewhere) so
the rest of the UI never touches raw keys or string-typed values.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QSettings

logger = logging.getLogger("K2.Settings")

ORG = "K2 AeroSim"
APP = "K2 AeroSim"

# key -> default
DEFAULTS = {
    "appearance/theme": "dark",
    "console/log_level": "INFO",
    "startup/check_updates": True,
    "projects/default_dir": "",       # empty -> core.project_manager default
    "sim/confirm_on_exit": True,
}

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def _store() -> QSettings:
    return QSettings(ORG, APP)


def get(key: str):
    default = DEFAULTS.get(key)
    value = _store().value(key, default)
    if isinstance(default, bool):
        # QSettings round-trips bools as "true"/"false" strings on Windows.
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    return value


def set(key: str, value) -> None:
    _store().setValue(key, value)


def theme_mode() -> str:
    mode = str(get("appearance/theme") or "dark").lower()
    return mode if mode in ("dark", "light") else "dark"


def project_dir() -> Path:
    """Configured project directory, or the built-in default."""
    raw = str(get("projects/default_dir") or "").strip()
    if raw:
        p = Path(raw)
        try:
            p.mkdir(parents=True, exist_ok=True)
            return p
        except OSError:
            logger.warning("Configured project dir unusable: %s", p)
    from core.project_manager import get_default_project_dir
    return get_default_project_dir()


def apply_log_level() -> str:
    """Push the stored console log level onto the K2 logger tree."""
    level = str(get("console/log_level") or "INFO").upper()
    if level not in LOG_LEVELS:
        level = "INFO"
    logging.getLogger("K2").setLevel(getattr(logging, level))
    return level
