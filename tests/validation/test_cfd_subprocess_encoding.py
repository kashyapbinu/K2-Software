"""
Mesh-subprocess log encoding gates.

The mesh runs in a child Python process whose stdout the parent reads as
UTF-8. On Windows the child defaults to the locale codec (cp1252), and the
mesh logger emits both '±' (radial extent) and '→' (revolve summary):

  * '→' is not in cp1252 at all, so logging raised UnicodeEncodeError while
    emitting. The record was DROPPED and a handler traceback went to stderr —
    which the workspace merges into the same pipe, so every mesh build spat
    "[Gmsh] Traceback ... UnicodeEncodeError" into the CFD console and looked
    like a meshing failure when the mesh had in fact built fine.
  * '±' does encode in cp1252, to a byte that is not valid UTF-8, so the
    parent's errors="replace" turned it into a mojibake marker.

These tests run a real subprocess, so they prove the encoding end to end
rather than pattern-matching the source.
"""
import os
import subprocess
import sys

import pytest

from ui.workspaces.cfd_workspace import _utf8_env

# The exact characters seen in cfd/meshing.py log lines.
ARROW = "→"      # not encodable in cp1252 -> record dropped
PLUSMINUS = "±"  # encodable in cp1252 -> mojibake in a UTF-8 parent

CHILD = (
    "import sys, logging\n"
    "try:\n"
    "    sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
    "except Exception:\n"
    "    pass\n"
    "logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s',"
    " stream=sys.stdout)\n"
    "logging.getLogger('K2.CFD.Meshing').info("
    "'radial=\\u00b125.20 m \\u2192 1 solid(s)')\n"
    "print('MESH_OK')\n"
)

CHILD_NO_RECONFIGURE = (
    "import sys, logging\n"
    "logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s',"
    " stream=sys.stdout)\n"
    "logging.getLogger('K2.CFD.Meshing').info("
    "'radial=\\u00b125.20 m \\u2192 1 solid(s)')\n"
    "print('MESH_OK')\n"
)


def _run(tmp_path, source, env):
    """Run a child the way the workspace does: UTF-8 pipe, stderr merged in."""
    script = tmp_path / "child.py"
    script.write_text(source, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=env,
    )


def _locale_env():
    """A child environment with any UTF-8 override stripped out.

    Reproduces a stock Windows session, where the failure actually happens.
    """
    env = os.environ.copy()
    env.pop("PYTHONIOENCODING", None)
    env.pop("PYTHONUTF8", None)
    return env


def test_utf8_env_sets_the_child_pipe_encoding():
    env = _utf8_env()
    assert env["PYTHONIOENCODING"].startswith("utf-8")
    # Must be the real environment plus the override, not a bare dict —
    # the child still needs PATH to find its DLLs.
    assert len(env) > 1


def test_env_override_keeps_unicode_log_lines(tmp_path):
    """PYTHONIOENCODING alone is enough, even with no reconfigure in-script."""
    env = _locale_env()
    env.update(_utf8_env())
    p = _run(tmp_path, CHILD_NO_RECONFIGURE, env)

    assert "MESH_OK" in p.stdout
    assert ARROW in p.stdout, "the revolve-summary arrow must survive"
    assert PLUSMINUS in p.stdout, "the radial-extent sign must survive"
    assert "UnicodeEncodeError" not in p.stderr


def test_in_script_reconfigure_keeps_unicode_log_lines(tmp_path):
    """The in-script guard alone is also enough (frozen-build belt braces)."""
    p = _run(tmp_path, CHILD, _locale_env())

    assert "MESH_OK" in p.stdout
    assert ARROW in p.stdout
    assert PLUSMINUS in p.stdout
    assert "UnicodeEncodeError" not in p.stderr


@pytest.mark.skipif(sys.platform != "win32",
                    reason="only Windows defaults stdout to a non-UTF-8 codec")
def test_the_unguarded_child_really_does_lose_the_line(tmp_path):
    """Pin the original failure, so the fix cannot be quietly reverted.

    If this ever starts passing without a guard, Python changed its default
    and the belt braces above can be reconsidered — but not before.
    """
    p = _run(tmp_path, CHILD_NO_RECONFIGURE, _locale_env())

    assert "MESH_OK" in p.stdout, "the child still completes..."
    assert ARROW not in p.stdout, "...but the log record is dropped"
    assert "UnicodeEncodeError" in p.stderr, (
        "and the handler traceback is what floods the CFD console"
    )


def test_both_launchers_are_guarded():
    """There are two copies of the mesh launcher (solve + sweep).

    The sweep one was an exact copy of the bug; keep them in step.
    """
    from pathlib import Path
    from ui.workspaces import cfd_workspace

    text = Path(cfd_workspace.__file__).with_suffix(".py").read_text(encoding="utf-8")
    assert text.count("sys.stdout.reconfigure(encoding='utf-8'") == 2, (
        "both generated mesh scripts must reconfigure stdout"
    )
    assert text.count("env=_utf8_env()") == 2, (
        "both mesh subprocesses must get the UTF-8 environment"
    )
