"""
AI assistant: the numbers it is given, and how a tool-heavy turn ends.

1. Liftoff mass and thrust-to-weight left out the motor casing
   (motor_dry_mass), which on a high-power motor weighs about as much as the
   propellant, so the model was told T/W was 14% higher than it is for an
   L1090W on a 6 kg airframe.
2. When the model used up its tool rounds, the "limit reached, answer with
   what you have" results were recorded but never sent back, so the turn
   ended without an answer.
"""
import json

import pytest

from ai.assistant import MAX_TOOL_ROUNDS, AssistantSession
from ai.context import liftoff_mass, summarize_state
from ai.providers import Chunk, Provider, ToolCall
from core.rocket_state import RocketState


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
