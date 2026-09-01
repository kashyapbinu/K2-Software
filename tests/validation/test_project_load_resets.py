"""
Opening a project must not leave the previous rocket's results on screen.

MainWindow ships a `_reset_all_workspaces()` that calls `reset_workspace()` on
every tab that defines one — the machinery is complete, and every workspace
implements its half. It was simply never called: neither Open (.k2/.json) nor
Import OpenRocket invoked it, so loading a second rocket kept the first one's
CFD coefficients, stress contours, flight traces, Monte Carlo scatter and
optimization history on display, silently re-attributed to the file just
opened.

These tests read the wiring off the source rather than driving a live
QApplication: constructing MainWindow needs an OpenGL context, and the thing
worth pinning is that the load paths call the reset at all.
"""
import ast
from pathlib import Path

import pytest

import ui.main_window as main_window

SRC = Path(main_window.__file__).with_suffix(".py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _method(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in main_window.py")


def _calls_self(node, method_name):
    """True if *node* contains a self.<method_name>(...) call."""
    for sub in ast.walk(node):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == method_name
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id == "self"):
            return True
    return False


def test_reset_helper_still_exists():
    assert hasattr(main_window.MainWindow, "_reset_all_workspaces")


def test_open_project_resets_workspaces():
    assert _calls_self(_method("_on_open"), "_reset_all_workspaces"), (
        "Open leaves the previous rocket's results on screen"
    )


def test_ork_import_resets_workspaces():
    assert _calls_self(_method("_on_import_ork_path"), "_reset_all_workspaces"), (
        "Import OpenRocket leaves the previous rocket's results on screen"
    )


def test_reset_runs_before_the_new_state_is_applied():
    """Order matters: reset first, then load.

    set_state / set_assembly repopulate workspaces through signals. Resetting
    afterwards would blank the values that belong to the NEW rocket.
    """
    node = _method("_on_open")
    reset_line = apply_line = None
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == "_reset_all_workspaces" and reset_line is None:
                reset_line = sub.lineno
            if sub.func.attr == "set_state" and apply_line is None:
                apply_line = sub.lineno
    assert reset_line is not None and apply_line is not None
    assert reset_line < apply_line, (
        "workspaces must be reset BEFORE the new state repopulates them"
    )


def test_reset_covers_every_workspace_that_defines_one():
    """The helper iterates a hardcoded name tuple — keep it in step with the tabs.

    A workspace that grows a reset_workspace() but never gets added to this
    tuple is stale-state bug shaped exactly like the original.
    """
    node = _method("_reset_all_workspaces")
    listed = {c.value for c in ast.walk(node) if isinstance(c, ast.Constant)
              and isinstance(c.value, str) and c.value.endswith("_ws")}

    setup = _method("_setup_tabs")
    assigned = set()
    for sub in ast.walk(setup):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) \
                and sub.value.id == "self" and sub.attr.endswith("_ws"):
            assigned.add(sub.attr)

    # design_ws is deliberately excluded: it is repopulated by set_assembly and
    # defines no reset_workspace.
    missing = assigned - listed - {"design_ws"}
    assert not missing, (
        f"workspace(s) never reset on project load: {sorted(missing)}"
    )
