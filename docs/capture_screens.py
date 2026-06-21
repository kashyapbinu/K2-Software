"""Capture a screenshot of every K2 AeroSim workspace tab into docs/shots/.

Launches the real app on-screen and uses QScreen.grabWindow so VTK 3D views and
the QtWebEngine cinematic view are captured as real composited pixels (widget
.grab() returns black for those). Run:  python docs/capture_screens.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

# WebEngine must import before QApplication (Cinematic tab).
try:
    import PyQt6.QtWebEngineWidgets  # noqa: F401
except Exception:
    pass

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QEventLoop, Qt
from PyQt6.QtGui import QGuiApplication

from ui.main_window import MainWindow
from ui.styles import DARK_STYLESHEET

OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)


def _pump(ms):
    """Spin the event loop for ms so renders/threads settle."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def build_sample_rocket(win):
    """Populate Design with a real airframe + motor so 3D views and stability
    readouts have content (a fresh app boots with an empty 'Untitled Rocket')."""
    from core.components import (RocketAssembly, NoseCone, BodyTube,
                                 TrapezoidalFinSet)
    asm = RocketAssembly()
    asm.name = "K2 Demo — 'Skylark'"
    stage = asm.stages[0]

    nose = NoseCone()
    nose.shape = "Ogive"
    nose.length = 0.22
    asm.add_component(stage, nose)

    body = BodyTube()
    body.length = 0.70
    body.is_motor_mount = True
    asm.add_component(stage, body)

    fins = TrapezoidalFinSet()
    fins.fin_count = 4
    fins.root_chord = 0.12
    fins.tip_chord = 0.05
    fins.height = 0.06
    fins.sweep_angle = 35.0
    asm.add_component(body, fins)

    win.design_ws.set_assembly(asm)

    # Pick a realistic mid-power motor (H/I class) instead of the tiny first
    # entry, so every downstream tab has a believable flight.
    try:
        import re
        combo = win.propulsion_ws.motor_combo
        pick = 1
        for i in range(1, combo.count()):
            if re.search(r"(?<![A-Za-z])[HI]\d", combo.itemText(i)):
                pick = i
                break
        if combo.count() > 1:
            combo.setCurrentIndex(pick)
    except Exception as e:
        print("motor select skipped:", e)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)

    win = MainWindow()
    win.resize(1600, 1000)
    win.show()
    win.raise_()
    win.activateWindow()
    _pump(1500)

    build_sample_rocket(win)
    _pump(1200)

    screen = QGuiApplication.primaryScreen()
    tabs = win.tab_widget
    n = tabs.count()
    cinematic = getattr(win, "cinematic_ws", None)
    mission = getattr(win, "mission_viz_ws", None)
    for i in range(n):
        tabs.setCurrentIndex(i)
        name = tabs.tabText(i)
        w = tabs.currentWidget()
        # Heavy 3D (VTK) / web (QtWebEngine) tabs render in a native child
        # window that can lag a tab switch — the screen still shows the previous
        # tab. Wait longer for those and pump events the whole time so the new
        # page actually composites before grabbing.
        heavy = w in (cinematic, mission)
        settle = 6000 if heavy else 2800
        steps = settle // 200
        for _ in range(steps):
            app.processEvents()
            if w is not None:
                w.repaint()
            _pump(200)
        win.raise_()
        win.activateWindow()
        app.processEvents()
        _pump(500)
        pm = screen.grabWindow(int(win.winId()))
        slug = f"{i:02d}_" + "".join(c if c.isalnum() else "_" for c in name).lower()
        path = OUT / f"{slug}.png"
        pm.save(str(path), "PNG")
        print("saved", path, pm.width(), "x", pm.height())

    win.close()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
