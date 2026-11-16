"""
What gets imported at launch, and by every helper process.

main.py imported the whole UI at module level, so each optimiser worker and
each `--run-script` helper (CFD meshing) loaded PyQt, pyvista and matplotlib
before doing any work: about 2 s warm, per process. The window's own import
also pulled scipy.stats, scipy.spatial and httpx for features most sessions
never touch.

Each check runs in a fresh interpreter: sys.modules in the pytest process
already holds everything.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _loaded_after(statement: str, candidates) -> set:
    """Top-level packages among *candidates* in sys.modules after running
    *statement* in a new interpreter at the project root."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); {statement}; "
            f"tops = {{m.split('.')[0] for m in sys.modules}}; "
            f"print('LOADED', *sorted(tops & set({list(candidates)!r})))")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    line = next(l for l in proc.stdout.splitlines() if l.startswith("LOADED"))
    return set(line.split()[1:])


def test_a_worker_process_does_not_load_the_ui():
    loaded = _loaded_after("import main", ("PyQt6", "ui", "pyvista", "vtkmodules",
                                           "matplotlib", "scipy"))
    assert loaded == set()


def test_the_window_imports_without_scipy_or_the_http_client():
    loaded = _loaded_after("import PyQt6.QtWebEngineWidgets; import ui.main_window",
                           ("scipy", "httpx", "rich", "sklearn", "reportlab"))
    assert loaded == set()


def test_the_cfd_smoothing_still_finds_scipy_when_asked():
    from cfd.post_processing import _ckdtree
    assert _ckdtree() is not None
