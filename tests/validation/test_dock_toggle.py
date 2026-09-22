"""
The console and AI docks must survive being tabified together.

Both were wired as action.toggled -> dock.setVisible plus
dock.visibilityChanged -> action.setChecked. visibilityChanged(False) also
fires when a dock is merely the background tab of a tabified group, so the
round trip closed it outright: drag the AI panel onto the console, click
the Console tab, and the AI panel was gone, with "hidden" persisted to
settings. ui.widgets.dock_title_bar.bind_toggle_action tracks Qt's
toggleViewAction (open/closed) instead.

Runs in a subprocess: widgets need a GUI QApplication, and the headless sim
tests put a bare QCoreApplication in this process.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QApplication, QDockWidget, QLabel, QMainWindow, QTabBar
    app = QApplication([])
    from ui.widgets.dock_title_bar import bind_toggle_action

    def window():
        mw = QMainWindow(); mw.setCentralWidget(QLabel("central")); mw.resize(900, 600)
        return mw

    def dock(mw, name, area):
        d = QDockWidget(name, mw); d.setObjectName(name); d.setWidget(QLabel(name))
        mw.addDockWidget(area, d)
        a = QAction(name, mw); a.setCheckable(True)
        return d, a

    mw = window()
    opened = []
    con, con_act = dock(mw, "Console", Qt.DockWidgetArea.BottomDockWidgetArea)
    ai, ai_act = dock(mw, "AI", Qt.DockWidgetArea.RightDockWidgetArea)
    bind_toggle_action(con, con_act)
    bind_toggle_action(ai, ai_act, on_open_changed=opened.append)
    mw.show(); app.processEvents(); opened.clear()

    mw.tabifyDockWidget(con, ai); app.processEvents()
    bar = next(t for t in mw.findChildren(QTabBar) if t.count() == 2)
    for i in range(bar.count()):                  # click through both tabs
        bar.setCurrentIndex(i); app.processEvents()
    assert not con.isHidden() and not ai.isHidden(), "a tab switch closed a dock"
    assert con_act.isChecked() and ai_act.isChecked(), "a tab switch unchecked an action"
    assert opened == [], f"tab switch reported open/close: {{opened}}"

    ai_act.trigger(); app.processEvents()         # user closes the AI panel
    assert ai.isHidden() and not ai_act.isChecked()
    assert opened == [False], opened
    ai_act.trigger(); app.processEvents()         # ...and reopens it
    assert not ai.isHidden() and ai_act.isChecked()

    # Hidden from settings before the window is first shown: one click opens it.
    mw2 = window()
    late, late_act = dock(mw2, "Late", Qt.DockWidgetArea.RightDockWidgetArea)
    late.setVisible(False)
    bind_toggle_action(late, late_act)
    mw2.show(); app.processEvents()
    assert late.isHidden() and not late_act.isChecked()
    late_act.trigger(); app.processEvents()
    assert not late.isHidden() and late_act.isChecked()
    print("OK")
""")


def test_docks_survive_tabbing_and_toggle_with_one_click():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=str(ROOT))],
                          env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
