"""
K2 AeroSim — Project Manager
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


def save_project(state: RocketState, filepath: str, assembly=None) -> bool:
    """
    Save the current rocket project to a JSON ``.k2`` file.

    Saves the full rocket state *and* the component assembly (the design tree),
    so a reopened project restores the actual rocket — not just scalar numbers.
    Analysis results (flight, CFD, FEM, optimization) are not saved; they are
    recomputed from the design.

    Args:
        state: The RocketState to serialize.
        filepath: Destination ``.k2`` file path.
        assembly: The RocketAssembly (component tree), if available.

    Returns:
        True if save was successful, False otherwise.
    """
    try:
        project_data = {
            "format_version": PROJECT_FORMAT_VERSION,
            "application": "K2 AeroSim",
            "saved_at": datetime.now().isoformat(),
            "rocket_state": state.to_dict(),
            "assembly": assembly.to_dict() if assembly is not None else None,
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


def load_project(filepath: str):
    """
    Load a rocket project from a ``.k2`` (or legacy ``.k2proj``) file.

    Returns:
        ``(RocketState, RocketAssembly | None)`` on success, or
        ``(None, None)`` on failure. The assembly is the rebuilt component tree
        when the file contains one (older files without it return None).
    """
    try:
        filepath = Path(filepath)

        if not filepath.exists():
            logger.error(f"Project file not found: {filepath}")
            return None, None

        with open(filepath, "r", encoding="utf-8") as f:
            project_data = json.load(f)

        # Version check
        version = project_data.get("format_version", "unknown")
        if version != PROJECT_FORMAT_VERSION:
            logger.warning(f"Project format version mismatch: {version} (expected {PROJECT_FORMAT_VERSION})")

        rocket_data = project_data.get("rocket_state", {})
        state = RocketState.from_dict(rocket_data)

        assembly = None
        asm_data = project_data.get("assembly")
        if asm_data:
            try:
                from core.components import RocketAssembly
                assembly = RocketAssembly.from_dict(asm_data)
            except Exception as e:
                logger.warning(f"Could not rebuild assembly from project: {e}")

        logger.info(f"Project loaded: {filepath} (rocket: {state.name}, "
                    f"assembly: {'yes' if assembly else 'no'})")
        return state, assembly

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in project file: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to load project: {e}")
        return None, None


def get_default_project_dir() -> Path:
    """Return the default directory for saving projects."""
    docs = Path.home() / "Documents" / "K2 AeroSim Projects"
    docs.mkdir(parents=True, exist_ok=True)
    return docs
