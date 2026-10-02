"""
A 3D viewport must not draw while its tab is hidden.

pyvistaqt starts a 5 Hz redraw timer on every QtInteractor and never checks
whether the widget is visible. K2 has seven viewports and shows at most one,
so the other six redrew behind hidden tabs: 31 hidden frames a second during
a flight, measured on the real application. With software OpenGL that starved
the simulation timer.

ui.widgets.viewport_gate suspends a hidden viewport. Two things it must not
break are pinned here as well:

* a change made while hidden has to appear when the view is shown again;
* ``Plotter.screenshot`` copies the last drawn frame rather than drawing one,
  and the PDF reports screenshot views whose tab is not selected, so a
  suspended view has to draw one real frame before it is captured.

Runs in a subprocess with a stand-in interactor: widgets need a GUI
QApplication, and a real QtInteractor needs OpenGL, which the offscreen
platform does not provide. The stand-in reproduces the three pyvistaqt
behaviours the gate relies on (the auto-update timer, ``suppress_rendering``,
and screenshot-returns-last-frame).
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {root!r})
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QLabel, QTabWidget, QVBoxLayout, QWidget
    app = QApplication([])

    class FakeInteractor(QWidget):
        '''What pyvistaqt.QtInteractor does, minus OpenGL.'''
        def __init__(self, parent=None, auto_update=5.0, **kw):
            super().__init__(parent)
            self.suppress_rendering = False
            self.scene = "empty"            # what add_mesh & co. have built
            self.frame = None               # what is in the window
            self.draws = 0
            self.render_timer = QTimer(parent)
            if auto_update:
                self.render_timer.timeout.connect(self.render)
                self.render_timer.start(int(1000 / auto_update))
        @property
        def interactor(self):
            return self
        def render(self):
            if not self.suppress_rendering:
                self.draws += 1
                self.frame = self.scene
        def screenshot(self, *a, **k):
            return self.frame               # copies the window, never draws
        def __getattr__(self, name):        # set_background, add_axes, ...
            if name.startswith("_"):
                raise AttributeError(name)
            return lambda *a, **k: None

    import pyvistaqt
    pyvistaqt.QtInteractor = FakeInteractor
    from ui.widgets.viewport_gate import gate_hidden_rendering

    def pump(seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents(); time.sleep(0.005)

    def tabs(view):
        '''view on an inner tab of an outer tab, as in the Structures workspace.'''
        page = QWidget(); QVBoxLayout(page).addWidget(view)
        inner = QTabWidget(); inner.addTab(page, "3D"); inner.addTab(QLabel("plots"), "plots")
        outer = QTabWidget(); outer.addTab(inner, "ws"); outer.addTab(QLabel("other"), "other")
        outer.resize(400, 300); outer.show(); pump(0.1)
        return outer, inner

    # Control: without the gate a hidden view keeps drawing.
    plain = FakeInteractor()
    outer, inner = tabs(plain)
    outer.setCurrentIndex(1); pump(0.1)
    plain.draws = 0; pump(0.7)
    assert not plain.isVisible() and plain.draws >= 2, plain.draws

    view = FakeInteractor()
    outer, inner = tabs(view)
    gate = gate_hidden_rendering(view)
    assert gate is not None and not gate.hidden
    view.draws = 0; pump(0.7)
    assert view.draws >= 2, "a visible view must keep its auto-update"

    for hide, show in ((lambda: outer.setCurrentIndex(1), lambda: outer.setCurrentIndex(0)),
                       (lambda: inner.setCurrentIndex(1), lambda: inner.setCurrentIndex(0))):
        hide(); pump(0.1)
        assert gate.hidden
        # The auto-update timer keeps ticking; its ticks must draw nothing.
        assert view.render_timer.isActive()
        view.draws = 0
        view.scene = "changed while hidden"
        view.render()                       # what add_mesh() does
        pump(0.7)
        assert view.draws == 0, f"hidden view drew {{view.draws}} frames"

        # A report screenshots the hidden view: it must get the current scene.
        assert view.screenshot() == "changed while hidden"
        assert view.draws == 1 and view.suppress_rendering
        pump(0.5)
        assert view.draws == 1

        view.scene = "changed again"
        show(); pump(0.05)
        assert not gate.hidden
        assert view.frame == "changed again", "change made while hidden never appeared"
        view.draws = 0; pump(0.7)
        assert view.draws >= 2

    # A view built hidden (every tab but the first) starts suspended.
    late = FakeInteractor()
    outer2, _ = tabs(QLabel("front"))
    page = QWidget(); QVBoxLayout(page).addWidget(late); outer2.addTab(page, "late")
    g2 = gate_hidden_rendering(late)
    late.draws = 0; pump(0.5)
    assert g2.hidden and late.draws == 0
    outer2.setCurrentIndex(2); pump(0.3)
    assert not g2.hidden and late.draws >= 1

    # auto_update=False (the CFD view): explicit renders only, and the gate
    # must not start a timer the view never had.
    manual = FakeInteractor(auto_update=False)
    outer3, _ = tabs(manual)
    g3 = gate_hidden_rendering(manual)
    outer3.setCurrentIndex(1); pump(0.1)
    manual.render(); assert manual.draws == 0
    outer3.setCurrentIndex(0); pump(0.3)
    assert not manual.render_timer.isActive()
    assert manual.draws == 1, "one catch-up frame on show, and no timer"

    # Tearing a hidden, gated view down must not need a live timer: stopping
    # the timer from the hide path crashed the interpreter at exit.
    orphan = FakeInteractor()
    orphan.render_timer.setParent(None)
    outer5, _ = tabs(orphan)
    gate_hidden_rendering(orphan)

    # Not an interactor (viewer fell back to a label): no gate, no exception.
    assert gate_hidden_rendering(object()) is None

    # The mode-shape animation only runs while its viewer is on screen.
    from ui.widgets.mode_shape_viewer import ModeShapeViewer
    viewer = ModeShapeViewer()
    assert not viewer.timer.isActive(), "animating before it was ever shown"
    outer4, _ = tabs(viewer)
    assert viewer.timer.isActive()
    outer4.setCurrentIndex(1); pump(0.1)
    assert not viewer.timer.isActive(), "animating behind a hidden tab"
    outer4.setCurrentIndex(0); pump(0.1)
    assert viewer.timer.isActive()
    viewer.btn_play.click(); pump(0.05)             # user pauses
    assert not viewer.timer.isActive()
    outer4.setCurrentIndex(1); pump(0.1); outer4.setCurrentIndex(0); pump(0.1)
    assert not viewer.timer.isActive(), "a tab switch un-paused the animation"
    print("OK")
""")


def _run_script():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", _SCRIPT.format(root=str(ROOT))],
                          env=env, capture_output=True, text=True, timeout=180)


def test_hidden_viewports_do_not_draw_and_reports_still_get_a_current_frame():
    proc = _run_script()
    assert "OK" in proc.stdout, proc.stderr[-3000:]
    assert proc.returncode == 0, f"assertions passed, then exit {proc.returncode:#x}"


def test_gated_views_survive_interpreter_shutdown():
    """The exit crash was intermittent (about three runs in four), so one
    clean exit proves little: repeat it."""
    codes = [_run_script().returncode for _ in range(6)]
    assert codes == [0] * 6, [hex(c & 0xFFFFFFFF) for c in codes]


def test_every_viewport_is_gated():
    """A new QtInteractor that is never gated redraws behind its tab again."""
    import re

    offenders = []
    for top in ("ui", "visualization"):
        for path in (ROOT / top).rglob("*.py"):
            src = path.read_text(encoding="utf-8", errors="ignore")
            built = len(re.findall(r"=\s*QtInteractor\(", src))
            gated = len(re.findall(r"^\s*gate_hidden_rendering\(", src, flags=re.M))
            if built != gated:
                offenders.append(f"{path.relative_to(ROOT)}: {built} built, {gated} gated")
    assert not offenders, offenders


def test_quitting_does_not_close_the_design_viewer_by_hand():
    """MainWindow.closeEvent used to call design_ws.closeEvent(), which closes
    the Design viewer's plotter while it may be the tab on screen. With the AI
    panel closed that killed the process on every quit from the Design tab,
    after the close handler had already returned. No test can drive a real
    OpenGL window here, so the wiring is pinned instead."""
    import ast

    tree = ast.parse((ROOT / "ui" / "main_window.py").read_text(encoding="utf-8"))
    close = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "closeEvent")
    calls = {ast.unparse(c.func) for c in ast.walk(close) if isinstance(c, ast.Call)}
    assert "self.design_ws.closeEvent" not in calls
    assert "self.mission_viz_ws.shutdown" in calls     # its timers still stop
