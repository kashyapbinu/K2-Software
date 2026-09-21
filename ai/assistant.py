"""
Chat session: history, context injection, streaming, and the tool loop.

``run_turn`` is a generator of (event, payload) tuples so the UI can render
progressively from a worker thread:
    ("text", delta) · ("tool_call", {name, args}) · ("tool_result", {name, result})
    ("error", message) · ("done", None)
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Iterator, Optional

from ai import prompts
from ai.providers import Provider, ProviderError, Chunk

logger = logging.getLogger("K2.AI.Session")

MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 40


class AssistantSession:
    def __init__(self, provider: Provider,
                 context_fn: Callable[[], str],
                 tool_executor: Optional[Callable[[str, dict], str]] = None,
                 tool_specs: Optional[list[dict]] = None):
        self.provider = provider
        self.context_fn = context_fn
        self.tool_executor = tool_executor
        self.tool_specs = tool_specs or []
        self.history: list[dict] = []
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def clear(self):
        self.history.clear()

    def _messages(self) -> list[dict]:
        ctx = ""
        try:
            ctx = self.context_fn() or ""
        except Exception as e:
            logger.warning("context_fn failed: %s", e)
        system = prompts.SYSTEM + ("\n\n" + ctx if ctx else "")
        # Trim old turns but never split a tool_call/tool pair.
        hist = self.history[-MAX_HISTORY_MESSAGES:]
        while hist and hist[0]["role"] == "tool":
            hist = hist[1:]
        return [{"role": "system", "content": system}] + hist

    def run_turn(self, user_text: str) -> Iterator[tuple[str, object]]:
        self._cancel = False
        self.history.append({"role": "user", "content": user_text})
        tools = self.tool_specs if (self.tool_executor and self.provider.supports_tools) else None

        for _round in range(MAX_TOOL_ROUNDS + 1):
            text_parts: list[str] = []
            final: Chunk | None = None
            try:
                for chunk in self.provider.chat(self._messages(), tools=tools):
                    if self._cancel:
                        break
                    if chunk.text:
                        text_parts.append(chunk.text)
                        yield ("text", chunk.text)
                    if chunk.done:
                        final = chunk
            except ProviderError as e:
                yield ("error", str(e))
                # Keep the user turn so a retry works; drop nothing else.
                yield ("done", None)
                return
            except Exception as e:
                logger.exception("Provider failure")
                yield ("error", f"{type(e).__name__}: {e}")
                yield ("done", None)
                return

            text = "".join(text_parts)
            calls = final.tool_calls if final else []
            if self._cancel or not calls:
                self.history.append({"role": "assistant", "content": text})
                yield ("done", None)
                return

            # Record the assistant turn with its tool calls (OpenAI shape).
            self.history.append({
                "role": "assistant", "content": text or None,
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.name,
                                             "arguments": json.dumps(c.arguments)},
                                **({"extra_content": c.extra} if c.extra else {})}
                               for c in calls]})
            for c in calls:
                yield ("tool_call", {"name": c.name, "args": c.arguments})
                if _round >= MAX_TOOL_ROUNDS:
                    result = json.dumps({"error": "tool round limit reached; answer with what you have"})
                else:
                    try:
                        result = self.tool_executor(c.name, c.arguments)
                    except Exception as e:
                        result = json.dumps({"error": f"{type(e).__name__}: {e}"})
                self.history.append({"role": "tool", "tool_call_id": c.id,
                                     "name": c.name, "content": result})
                yield ("tool_result", {"name": c.name, "result": result})
        yield ("done", None)

    def one_shot(self, instruction: str) -> Iterator[tuple[str, object]]:
        """Single response outside the chat history (e.g. flight debrief)."""
        saved = self.history
        self.history = []
        try:
            yield from self.run_turn(instruction)
            reply = next((m["content"] for m in reversed(self.history)
                          if m["role"] == "assistant" and m.get("content")), "")
        finally:
            self.history = saved
        # Keep the debrief visible to follow-up questions.
        if reply:
            self.history.append({"role": "user", "content": "(Flight debrief requested.)"})
            self.history.append({"role": "assistant", "content": reply})
