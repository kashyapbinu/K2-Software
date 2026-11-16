"""
AI assistant: the numbers it is given, and how a tool-heavy turn ends.

1. Liftoff mass and thrust-to-weight left out the motor casing
   (motor_dry_mass), which on a high-power motor weighs about as much as the
   propellant, so the model was told T/W was 14% higher than it is for an
   L1090W on a 6 kg airframe.
2. When the model used up its tool rounds, the "limit reached, answer with
   what you have" results were recorded but never sent back, so the turn
   ended without an answer.
3. Gemini is the only backend (2026-11-15). The backend choice was stored,
   and a stored "ollama" with no model pulled left the panel dead beside a
   saved, working Gemini key. A Gemini model that accepted a request and then
   sat on it (51 s measured) was waited on for up to 120 s instead of handing
   over to the next model. The setup card told the user to go to Settings; it
   now takes the key, checks it and stores it only if Gemini accepts it.
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import ai.providers as providers
from ai.assistant import MAX_TOOL_ROUNDS, AssistantSession
from ai.context import liftoff_mass, summarize_state
from ai.providers import (Chunk, GeminiProvider, OpenAICompatProvider, Provider,
                          ProviderError, ToolCall)
from core.rocket_state import RocketState

ROOT = Path(__file__).resolve().parents[2]


def _l1090_rocket():
    # L1090W: 2.432 kg loaded, 1.4 kg propellant (motors.json).
    return RocketState(dry_mass=6.0, motor_dry_mass=1.032,
                       propellant_mass=1.4, propellant_mass_initial=1.4,
                       motor_designation="L1090W", motor_avg_thrust=1090.0)


def test_liftoff_mass_includes_the_motor_casing():
    s = _l1090_rocket()
    assert liftoff_mass(s) == pytest.approx(8.432)
    ctx = summarize_state(s)
    assert ctx["mass_and_stability"]["liftoff_mass_kg"] == pytest.approx(8.432)
    assert ctx["motor"]["thrust_to_weight"] == pytest.approx(
        1090.0 / (8.432 * 9.80665), abs=0.01)


def test_the_model_is_told_the_peak_dynamic_pressure_not_the_one_after_landing():
    from ai.context import last_flight_summary
    from core.flight_summary import FlightSummary
    landed = RocketState(sim_phase="Landed", max_altitude=1498.9, dynamic_pressure=0.0)
    flight = FlightSummary(source="interactive", final_phase="Landed", apogee=1498.9,
                           max_velocity=158.4, max_mach=0.47, max_acceleration=66.8,
                           max_q=14250.0, flight_time=174.1, rail_exit_velocity=21.3)
    told = last_flight_summary(landed, flight)
    assert told["max_dynamic_pressure_Pa"] == pytest.approx(14250.0)
    assert told["rail_exit_velocity_m_s"] == pytest.approx(21.3)
    assert summarize_state(landed, flight=flight)["last_flight"]["max_dynamic_pressure_Pa"]         == pytest.approx(14250.0)
    # No flight summary to read it from: the number is absent, not zero.
    assert "max_dynamic_pressure_Pa" not in last_flight_summary(landed)


def test_the_model_is_told_the_rockets_own_drag_coefficient():
    """"cd0" was RocketState.cd: 0.45 for every rocket, whatever its nose and
    fins, until it flew, and after that the drag coefficient of the last
    instant of the flight (1.57 under the canopy for the canonical rocket).
    It is the aero model's zero-lift value at Mach 0.3."""
    from physics.aerodynamics import AeroModel
    from validation.cases.rocket_canonical import canonical_state
    s = canonical_state()
    told = summarize_state(s)["mass_and_stability"]["cd0"]
    v = 0.3 * 340.294
    expected = AeroModel.from_state(s).compute(
        0.0, 0.3, 0.5 * 1.225 * v * v, 0.0, v, s.liftoff_cg())["cd"]
    assert told == pytest.approx(expected, abs=2e-3)
    s.cd = 1.57
    assert summarize_state(s)["mass_and_stability"]["cd0"] == told
    # No airframe drawn: no drag coefficient.
    assert summarize_state(_l1090_rocket())["mass_and_stability"]["cd0"] is None


def test_the_model_is_told_the_peak_acceleration_the_results_tab_shows():
    """state.max_acceleration is the peak of d|v|/dt, about 1 g short of what
    the airframe feels on the way up. The Results tab and the report give the
    sensed peak (FlightSummary.max_acceleration); the assistant was told the
    other one: 185.9 m/s² against 195.8 for the same flight."""
    from ai.context import last_flight_summary
    from core.flight_summary import FlightSummary
    landed = RocketState(sim_phase="Landed", max_altitude=2564.2, max_acceleration=185.9)
    flight = FlightSummary(source="interactive", final_phase="Landed", apogee=2564.2,
                           max_velocity=329.2, max_mach=0.97, max_acceleration=195.8,
                           max_q=63808.0, flight_time=210.5)
    told = last_flight_summary(landed, flight)
    assert told["max_acceleration_m_s2"] == pytest.approx(195.8)
    assert told["max_acceleration_g"] == pytest.approx(195.8 / 9.80665, abs=0.01)


class _ToolHappy(Provider):
    """Calls a tool every round; answers only once told the budget is spent
    (or never, if ``stubborn``)."""

    def __init__(self, stubborn=False):
        self.stubborn = stubborn
        self.requests = 0

    def chat(self, messages, tools=None, **kw):
        self.requests += 1
        refused = any(m.get("role") == "tool" and "limit reached" in (m.get("content") or "")
                      for m in messages)
        if refused and not self.stubborn:
            yield Chunk(text="Best answer from what I have.")
            yield Chunk(done=True)
            return
        yield Chunk(done=True, tool_calls=[
            ToolCall(id=f"call_{self.requests}", name="get_rocket_state", arguments={})])


def _session(provider, executed):
    return AssistantSession(
        provider, context_fn=lambda: "",
        tool_executor=lambda name, args: executed.append(name) or json.dumps({}),
        tool_specs=[{"type": "function", "function": {"name": "get_rocket_state"}}])


def test_tool_limit_still_ends_with_an_answer():
    executed = []
    session = _session(_ToolHappy(), executed)
    events = list(session.run_turn("Make it stable."))
    text = "".join(p for e, p in events if e == "text")
    assert text == "Best answer from what I have."
    assert len(executed) == MAX_TOOL_ROUNDS
    assert session.history[-1] == {"role": "assistant", "content": text}
    assert not any(e == "error" for e, _ in events)
    assert events[-1] == ("done", None)


def test_endless_tool_calls_end_with_a_notice():
    executed = []
    events = list(_session(_ToolHappy(stubborn=True), executed).run_turn("Go."))
    assert len(executed) == MAX_TOOL_ROUNDS
    assert any(e == "error" for e, _ in events)
    assert events[-1] == ("done", None)


def test_a_stored_ollama_choice_still_gives_the_gemini_assistant(monkeypatch):
    stored = {"ai/provider": "ollama", "ai/gemini_key": "AIza-test",
              "ai/gemini_model": "gemini-3.6-flash"}
    monkeypatch.setattr(providers, "_setting",
                        lambda key, default="": stored.get(key, default))
    p = providers.build_provider()
    assert isinstance(p, GeminiProvider)
    assert p.available() == (True, "")
    assert p.api_key == "AIza-test" and p.model == "gemini-3.6-flash"


def test_no_key_is_reported_as_the_reason_not_as_no_provider(monkeypatch):
    monkeypatch.setattr(providers, "_setting", lambda key, default="": default)
    ok, why = providers.build_provider().available()
    assert not ok and "key" in why.lower()


def test_a_gemini_model_that_never_answers_hands_over_to_the_next(monkeypatch):
    asked = []

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        asked.append(self.model)
        if len(asked) == 1:
            raise ProviderError("Gemini timed out: The read operation timed out")
        yield Chunk(text="Stable, 1.6 cal.")
        yield Chunk(done=True)

    monkeypatch.setattr(OpenAICompatProvider, "chat", chat)
    g = GeminiProvider("AIza-test")
    assert g.timeout == 30.0, "waits for a silent model no longer than 30 s"
    text = "".join(c.text for c in g.chat([{"role": "user", "content": "Stable?"}]))
    assert text == "Stable, 1.6 cal."
    assert asked == ["gemini-3.6-flash", GeminiProvider.FALLBACK_MODELS[0]]
    assert g.model == "gemini-3.6-flash", "the configured model is kept for the next turn"


def test_a_rejected_request_is_not_retried_on_other_models(monkeypatch):
    asked = []

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        asked.append(self.model)
        raise ProviderError("HTTP 400: API key not valid")
        yield  # pragma: no cover

    monkeypatch.setattr(OpenAICompatProvider, "chat", chat)
    with pytest.raises(ProviderError):
        list(GeminiProvider("bad").chat([{"role": "user", "content": "Hi"}]))
    assert asked == ["gemini-3.6-flash"]


@pytest.mark.parametrize("raw, says, hides", [
    ('HTTP 503: [{ "error": { "code": 503, "status": "UNAVAILABLE" } } ]', "overloaded", "503"),
    ("Gemini timed out: The read operation timed out", "overloaded", "read operation"),
    ('HTTP 429: [{ "error": { "code": 429, "message": "You exceeded your current quota" } } ]',
     "allowance", "429"),
])
def test_when_no_gemini_model_answers_the_chat_says_why_in_plain_words(
        monkeypatch, raw, says, hides):
    asked = []

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        asked.append(self.model)
        raise ProviderError(raw)
        yield  # pragma: no cover

    monkeypatch.setattr(OpenAICompatProvider, "chat", chat)
    with pytest.raises(ProviderError) as err:
        list(GeminiProvider("AIza-test").chat([{"role": "user", "content": "Hi"}]))
    assert len(asked) == 1 + len(GeminiProvider.FALLBACK_MODELS)
    assert says in str(err.value) and hides not in str(err.value)
    assert "{" not in str(err.value)


_PANEL_SCRIPT = textwrap.dedent("""
    import os, sys, tempfile
    sys.path.insert(0, {root!r})
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtCore import QSettings
    app = QApplication([])
    import ui.settings as st
    ini = os.path.join(tempfile.mkdtemp(), "k2.ini")
    st._store = lambda: QSettings(ini, QSettings.Format.IniFormat)
    # What an install that had picked the local model holds.
    st.set("ai/provider", "ollama")
    st.set("ai/ollama_model", "qwen3:1.7b")

    import ui.widgets.ai_panel as ap
    answers = {{"AIza-good": (True, "Key accepted."),
               "AIza-bad": (False, "Gemini rejected this key. Copy it again from AI Studio.")}}
    ap.check_gemini_key = lambda key, timeout=10.0: answers[key]

    def settle(panel):
        panel._key_check.wait(5000)
        app.processEvents()

    mw = QWidget()
    panel = ap.AIPanel(mw)

    # No key: the key card stands where the transcript would, nothing to type into.
    assert not panel.setup.isHidden() and panel.view.isHidden()
    assert not panel.input.isEnabled() and not panel.btn_send.isEnabled()
    assert not hasattr(panel, "cmb_backend"), "no backend picker"
    assert "aistudio.google.com/apikey" in panel.btn_get_key.toolTip()

    # Nothing pasted: says so, asks nobody.
    panel._save_key()
    assert panel._key_check is None
    assert "Paste" in panel.lbl_setup_result.text()

    # A key Gemini rejects is not stored and the reason is shown.
    panel.edit_key.setText("  AIza-bad ")
    panel._save_key(); settle(panel)
    assert not st.get("ai/gemini_key")
    assert "rejected" in panel.lbl_setup_result.text()
    assert not panel.setup.isHidden() and panel.btn_save_key.isEnabled()

    # An accepted key is stored and the chat opens, with no trip to Settings.
    panel.edit_key.setText("AIza-good")
    panel._save_key(); settle(panel)
    assert st.get("ai/gemini_key") == "AIza-good"
    assert panel.setup.isHidden() and not panel.view.isHidden()
    assert panel.input.isEnabled() and panel.session is not None
    assert panel.provider.name == "Gemini"
    assert panel.edit_key.text() == ""

    # Settings: one Gemini group, no backend chooser, and OK still saves.
    from ui.dialogs.settings_dialog import SettingsDialog
    dlg = SettingsDialog()
    for gone in ("cmb_ai", "edit_ollama_url", "edit_ollama_model", "edit_custom_url"):
        assert not hasattr(dlg, gone), gone
    assert dlg.edit_gemini_key.text() == "AIza-good"
    dlg.edit_gemini_model.setText("gemini-3.5-flash")
    dlg._restore_defaults()
    assert dlg.edit_gemini_model.text() == st.DEFAULTS["ai/gemini_model"]
    dlg._on_accept()
    assert st.get("ai/gemini_key") == "AIza-good"

    # Removing the key brings the card back.
    st.set("ai/gemini_key", "")
    panel.reload_provider()
    assert not panel.setup.isHidden() and not panel.input.isEnabled()
    panel.shutdown()
    print("OK")
""")


def test_the_panel_takes_a_gemini_key_and_opens_the_chat():
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_API="pyqt6")
    proc = subprocess.run([sys.executable, "-c", _PANEL_SCRIPT.format(root=str(ROOT))],
                          env=env, capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0 and "OK" in proc.stdout,         (proc.stdout + proc.stderr)[-3000:]
