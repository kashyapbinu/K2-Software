"""
K2 Aerospace — Project Manager
================================
Handles saving and loading rocket projects as JSON files.
Supports forward-compatible deserialization (unknown keys are ignored).
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from core.rocket_state import RocketState

logger = logging.getLogger("K2.ProjectManager")


# Project file metadata version for future migration support
PROJECT_FORMAT_VERSION = "1.0.0"


def save_project(state: RocketState, filepath: str) -> bool:
    """
    Save the current rocket state to a JSON project file.
    
    Args:
        state: The RocketState to serialize.
        filepath: Destination .k2proj file path.
        
    Returns:
        True if save was successful, False otherwise.
    """
    try:
        project_data = {
            "format_version": PROJECT_FORMAT_VERSION,
            "application": "K2 Aerospace",
            "saved_at": datetime.now().isoformat(),
            "rocket_state": state.to_dict()
        }
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Project saved: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save project: {e}")
        return False


def load_project(filepath: str) -> RocketState | None:
    """
    Load a rocket state from a JSON project file.
    
    Args:
        filepath: Path to the .k2proj file.
        
    Returns:
        RocketState if successful, None otherwise.
    """
    try:
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Project file not found: {filepath}")
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            project_data = json.load(f)
        
        # Version check
        version = project_data.get("format_version", "unknown")
        if version != PROJECT_FORMAT_VERSION:
            logger.warning(f"Project format version mismatch: {version} (expected {PROJECT_FORMAT_VERSION})")
        
        rocket_data = project_data.get("rocket_state", {})
        state = RocketState.from_dict(rocket_data)
        
        logger.info(f"Project loaded: {filepath} (rocket: {state.name})")
        return state
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in project file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load project: {e}")
        return None


def get_default_project_dir() -> Path:
    """Return the default directory for saving projects."""
    docs = Path.home() / "Documents" / "K2 Aerospace Projects"
    docs.mkdir(parents=True, exist_ok=True)
    return docs
