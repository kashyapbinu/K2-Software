"""
One look for the whole app, set in one place.

The interface had drifted into several styles at once, each small on its own:

* spin-box steppers and combo-box drop-downs drew their arrows with the QSS
  zero-size-box border trick, which Qt paints as small filled squares;
* every group box was a bordered, filled card, so side panels read as boxes
  inside boxes;
* the Simulation and Results tabs each built their own caption-plus-value
  tiles, and the Results frame's stylesheet (no selector) drew its border
  round every caption too;
* captions such as "Atmosphere & Flow" lost the ampersand, because Qt reads
  a single "&" as a keyboard-mnemonic marker;
* verdicts were emoji badges ("✅ Good") and panel titles were a mix of
  centred capitals and left-aligned title case;
* inner tab bars carried their own stylesheets and looked different from
  the main one.

These tests pin the rule that fixes each of them.
"""
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UI_FILES = sorted(
    p for sub in ("ui", "ui/widgets", "ui/workspaces", "ui/dialogs")
    for p in (ROOT / sub).glob("*.py")
    # dead modules (CLAUDE.md section 2) and the palette definition itself
    if p.name not in ("aero_workspace.py", "rocket_params_panel.py", "styles.py", "theme.py")
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_stepper_and_dropdown_arrows_are_images_not_border_triangles(mode):
    from ui import theme
    qss = theme.stylesheet(mode)
    for sub in ("QDoubleSpinBox::up-arrow", "QSpinBox::down-arrow", "QComboBox::down-arrow"):
        block = qss[qss.index(sub):]
        block = block[:block.index("}")]
        assert "image: url(" in block, f"{sub} has no arrow image in the {mode} theme"
        assert "solid transparent" not in block, f"{sub} still uses the border trick"


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_a_checked_checkbox_shows_a_tick(mode):
    from ui import theme
    qss = theme.stylesheet(mode)
    block = qss[qss.index("QCheckBox::indicator:checked {"):]
    assert "image: url(" in block[:block.index("}")]


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_a_group_box_is_a_ruled_section_not_a_card(mode):
    from ui import theme
    qss = theme.stylesheet(mode)
    block = qss[qss.index("QGroupBox {"):]
    block = block[:block.index("}")]
    assert "border: none;" in block and "border-top: 1px solid" in block
    assert "background-color: transparent" in block


@pytest.mark.parametrize("mode", ["dark", "light"])
def test_every_stylesheet_placeholder_is_filled(mode):
    from ui import theme
    qss = theme.stylesheet(mode)
    assert not re.search(r"\{[A-Z_]+\}", qss)
    assert "url(\"none\")" not in qss and "image: none" not in qss


def test_both_palettes_define_the_same_roles():
    from ui import theme
    assert set(theme.DARK) == set(theme.LIGHT)


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_a_caption_with_an_ampersand_escapes_it(path):
    """ "Solver & Flow" is drawn as "Solver  Flow"; it has to be "Solver && Flow"."""
    pattern = re.compile(
        r'(QGroupBox|QCheckBox|QRadioButton|QPushButton|addTab)\([^)\n]*"[^"\n]*[^&]&[^&"\n][^"\n]*"')
    bad = [m.group(0) for m in pattern.finditer(_source(path))]
    assert not bad, bad


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_inner_tab_bars_do_not_restyle_themselves(path):
    assert "QTabBar::tab" not in _source(path), (
        f"{path.name} styles its own tab bar; the application stylesheet does that")


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_no_emoji_verdicts_or_pictographs_in_the_interface(path):
    # Colour emoji only. The plain warning sign (U+26A0 without the emoji
    # selector) is a text glyph and stays as the prefix of a warning line.
    found = re.findall("[\U0001F300-\U0001FAFF✅❌️]", _source(path))
    assert not found, f"{path.name}: {sorted(set(found))}"


@pytest.mark.parametrize("path", UI_FILES, ids=lambda p: p.name)
def test_the_old_blue_and_purple_palette_is_not_hard_coded(path):
    old = re.findall(r"#(?:0d1117|161b22|21262d|30363d|58a6ff|1f6feb|bc8cff|d2a8ff|a371f7)",
                     _source(path), flags=re.I)
    assert not old, f"{path.name}: {sorted(set(old))}"


_READOUT_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication, QLabel
    app = QApplication([])
    from ui import theme
    from ui.widgets.readout import Readout

    r = Readout("Altitude")
    assert r.text() == "\\u2014"
    labels = r.findChildren(QLabel)
    assert [l.text() for l in labels][0] == "Altitude"
    value = labels[1]

    def shown():
        import re
        return re.sub("<[^>]+>", "", value.text())

    r.setText("3,048 m")
    assert r.text() == "3,048 m", r.text()          # what callers and tests read back
    assert shown() == "3,048 m" and "font-size:11px" in value.text(), value.text()

    r.setText("0.809")                               # no unit: nothing set small
    assert value.text() == "0.809", value.text()
    r.setText("Main Descent")                        # two words, not value + unit
    assert value.text() == "Main Descent", value.text()
    r.setText("412.0 m/s\\u00b2 (42.0 g)")           # a bracketed tail is not a unit
    assert "<span" not in value.text(), value.text()

    # the border belongs to the tile only, not to the labels inside it
    assert "QFrame#readout" in r.styleSheet()
    assert all("border: none" in l.styleSheet() for l in labels)

    theme.set_mode("light"); r.retheme()
    assert theme.LIGHT["PANEL"] in r.styleSheet()
    assert r.text() == "412.0 m/s\\u00b2 (42.0 g)"
    print("OK")
""")

_SIM_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication
    app = QApplication([])
    from core.rocket_state import RocketStateEngine
    from core.simulation_engine import SimulationEngine
    from ui.workspaces.simulation_workspace import SimulationWorkspace
    from ui.widgets.readout import Readout

    engine = RocketStateEngine()
    ws = SimulationWorkspace(engine, SimulationEngine(engine))
    ws.resize(1400, 800); ws.show(); app.processEvents()

    # transport buttons are a compact bar, not four tab-wide slabs
    for b in (ws.btn_run, ws.btn_pause, ws.btn_stop, ws.btn_reset):
        assert b.height() <= 32 and b.width() <= 160, (b.text(), b.width(), b.height())
    assert ws.btn_run.text() == "Run"

    # every live readout is the shared tile and none grows to fill the tab
    assert len(ws.readouts) == 8
    for name, tile in ws.readouts.items():
        assert isinstance(tile, Readout), name
        assert tile.height() <= 60, (name, tile.height())

    # the wind block is as tall as the page showing, no blank band under it
    avg_h = ws.wind_stack.sizeHint().height()
    ws.wind_mode_combo.setCurrentIndex(1); app.processEvents()
    assert ws.wind_stack.sizeHint().height() > avg_h
    ws.wind_mode_combo.setCurrentIndex(0); app.processEvents()
    assert ws.wind_stack.sizeHint().height() == avg_h

    # the live trace follows the recorded flight, and clears on reset
    hist = ws.sim_engine.history
    for i in range(5):
        hist.record(time=0.1 * i, altitude=10.0 * i, velocity=100.0 * i)
    ws._update_trace()
    xs, ys = ws.trace._alt.get_data()
    assert list(ys) == [0.0, 10.0, 20.0, 30.0, 40.0], list(ys)
    lo, hi = ws.trace.ax2.get_ylim()
    assert hi >= 400.0, ("velocity axis clipped", lo, hi)
    ws._reset_ui()
    assert len(ws.trace._alt.get_data()[0]) == 0

    ws._on_pause.__func__                             # still present
    ws.btn_run.setText("x"); ws._reset_ui()
    assert ws.btn_run.text() == "Run"
    print("OK")
""")


def _run(script: str):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    out = subprocess.run([sys.executable, "-c", script.format(root=str(ROOT))],
                         capture_output=True, text=True, timeout=180, env=env, cwd=str(ROOT))
    assert out.returncode == 0 and "OK" in out.stdout, out.stdout + out.stderr


def test_readout_tile_keeps_its_text_and_sets_only_a_unit_small():
    _run(_READOUT_SCRIPT)


def test_simulation_tab_has_a_compact_transport_bar_and_a_live_trace():
    _run(_SIM_SCRIPT)
