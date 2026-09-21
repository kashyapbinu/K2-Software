"""
LLM providers behind one streaming interface.

Every provider yields ``Chunk`` objects from ``chat()``: text deltas as they
arrive, then a final chunk carrying any complete tool calls. The OpenAI
chat-completions wire format is spoken by Gemini (AI Studio), Ollama, Groq,
OpenRouter and most others, so one HTTP client covers all of them -- any such
endpoint is reachable through the "custom" provider.

Selection order for ``auto``: Gemini if a key is configured, else a running
Ollama, else nothing (the panel shows a setup card).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import httpx

logger = logging.getLogger("K2.AI.Providers")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
_GEMINI_RETIRED = {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"}
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen3:1.7b"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    # Provider-specific opaque data that must be echoed back with the call
    # (Gemini 3.x: extra_content.google.thought_signature, else 400).
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False
    finish_reason: str = ""


class ProviderError(RuntimeError):
    pass


class Provider:
    """Base: subclasses implement ``chat`` and ``available``."""

    name = "base"
    model = ""
    supports_tools = True

    def available(self) -> tuple[bool, str]:
        """(ok, human reason). Cheap; called on panel refresh."""
        return True, ""

    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None,
             temperature: float = 0.3, max_tokens: int = 4096) -> Iterator[Chunk]:
        raise NotImplementedError

    def label(self) -> str:
        return f"{self.name} · {self.model}"


# ---------------------------------------------------------------------------
# OpenAI-compatible (Gemini, Ollama, Groq, OpenRouter, ...)
# ---------------------------------------------------------------------------
class OpenAICompatProvider(Provider):
    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str, model: str,
                 name: str = None, timeout: float = 120.0,
                 supports_tools: bool = True, extra_body: dict = None):
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key = api_key or "none"
        self.model = model
        if name:
            self.name = name
        self.timeout = timeout
        self.supports_tools = supports_tools
        self.extra_body = extra_body or {}

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def available(self) -> tuple[bool, str]:
        # Keyless local servers (llama.cpp, vLLM, LM Studio) are valid; only
        # an unset base URL makes the provider unusable.
        if not self.base_url.strip("/"):
            return False, "No base URL configured"
        return True, ""

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and self.supports_tools:
            body["tools"] = tools
        body.update(self.extra_body)

        url = self.base_url + "chat/completions"
        # Streamed tool calls arrive as fragments keyed by index; assemble.
        pending: dict[int, dict] = {}
        finish = ""
        try:
            with httpx.Client(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
                with client.stream("POST", url, headers=self._headers(), json=body) as r:
                    if r.status_code >= 400:
                        r.read()
                        raise ProviderError(_http_error_text(r))
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            evt = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = evt.get("choices") or []
                        if not choices:
                            continue
                        ch = choices[0]
                        delta = ch.get("delta") or {}
                        if ch.get("finish_reason"):
                            finish = ch["finish_reason"]
                        text = delta.get("content")
                        if text:
                            yield Chunk(text=text)
                        for tc in delta.get("tool_calls") or []:
                            idx = tc.get("index", len(pending))
                            slot = pending.setdefault(idx, {"id": "", "name": "", "args": "", "extra": {}})
                            if tc.get("id"):
                                slot["id"] = tc["id"]
                            if tc.get("extra_content"):
                                slot["extra"] = tc["extra_content"]
                            fn = tc.get("function") or {}
                            if fn.get("name"):
                                slot["name"] += fn["name"]
                            if fn.get("arguments"):
                                slot["args"] += fn["arguments"]
        except httpx.ConnectError as e:
            raise ProviderError(f"Cannot reach {self.name} at {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} timed out: {e}") from e

        calls = []
        for i in sorted(pending):
            p = pending[i]
            if not p["name"]:
                continue
            try:
                args = json.loads(p["args"]) if p["args"].strip() else {}
            except json.JSONDecodeError:
                logger.warning("Unparseable tool args from %s: %r", self.name, p["args"])
                args = {"_raw": p["args"]}
            calls.append(ToolCall(id=p["id"] or f"call_{i}", name=p["name"], arguments=args,
                                  extra=p.get("extra") or {}))
        yield Chunk(done=True, tool_calls=calls, finish_reason=finish)


def _http_error_text(r: httpx.Response) -> str:
    try:
        j = r.json()
        msg = j.get("error", {}).get("message") or j.get("message") or r.text
    except Exception:
        msg = r.text
    return f"HTTP {r.status_code}: {str(msg)[:400]}"


class GeminiProvider(OpenAICompatProvider):
    name = "Gemini"

    def __init__(self, api_key: str, model: str = GEMINI_DEFAULT_MODEL):
        super().__init__(GEMINI_BASE_URL, api_key, model or GEMINI_DEFAULT_MODEL)

    # Free-tier flash models return 503 "high demand" / 429 quota in bursts;
    # step through siblings before giving up, but only if nothing streamed yet.
    FALLBACK_MODELS = ("gemini-3.8-flash", "gemini-flash-latest", "gemini-3.5-flash")
    _TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand")

    def available(self):
        if not self.api_key or self.api_key == "none":
            return False, "No Gemini API key — add one in Settings → AI Assistant"
        return True, ""

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        chain = [self.model] + [m for m in self.FALLBACK_MODELS if m != self.model]
        primary = self.model
        try:
            for i, model in enumerate(chain):
                self.model = model
                started = False
                try:
                    for chunk in super().chat(messages, tools, temperature, max_tokens):
                        started = True
                        yield chunk
                    return
                except ProviderError as e:
                    transient = any(t in str(e) for t in self._TRANSIENT)
                    if started or not transient or i == len(chain) - 1:
                        raise
                    logger.warning("Gemini %s unavailable (%s); trying %s", model, str(e)[:80], chain[i + 1])
        finally:
            self.model = primary


class OllamaProvider(OpenAICompatProvider):
    name = "Ollama"

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, host: str = OLLAMA_BASE_URL):
        self.host = host.rstrip("/")
        super().__init__(self.host + "/v1", "ollama", model or OLLAMA_DEFAULT_MODEL,
                         timeout=300.0)

    def installed_models(self) -> list[str]:
        try:
            r = httpx.get(self.host + "/api/tags", timeout=2.0)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def chat(self, messages, tools=None, temperature=0.3, max_tokens=4096):
        # Qwen3's "thinking" mode burns minutes on a 4 GB GPU; the soft switch
        # in the system prompt turns it off without touching the native API.
        if "qwen3" in self.model.lower() and messages and messages[0].get("role") == "system":
            messages = [dict(messages[0], content=messages[0]["content"] + "\n/no_think")] + messages[1:]
        yield from super().chat(messages, tools, temperature, max_tokens)

    def running(self) -> bool:
        try:
            return httpx.get(self.host + "/api/tags", timeout=1.5).status_code == 200
        except Exception:
            return False

    def available(self):
        if not self.running():
            return False, f"Ollama not running at {self.host}"
        models = self.installed_models()
        if not models:
            return False, f"Ollama has no models — run: ollama pull {self.model}"
        if not any(m == self.model or m.split(":")[0] == self.model.split(":")[0]
                   for m in models):
            return False, f"Model '{self.model}' not pulled — run: ollama pull {self.model}"
        return True, ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def _setting(key: str, default=""):
    try:
        from ui import settings
        v = settings.get(key)
        return v if v not in (None, "") else default
    except Exception:
        return default


def gemini_model_setting() -> str:
    """Stored Gemini model, with retired names moved forward. Google drops old
    flash models for new keys (404 "no longer available to new users")."""
    model = _setting("ai/gemini_model", GEMINI_DEFAULT_MODEL)
    return GEMINI_DEFAULT_MODEL if model in _GEMINI_RETIRED else model


def build_provider(kind: str = None) -> Provider | None:
    """Construct the configured provider. ``kind`` overrides the setting."""
    kind = (kind or _setting("ai/provider", "auto") or "auto").lower()

    def gemini():
        return GeminiProvider(_setting("ai/gemini_key"), gemini_model_setting())

    def ollama():
        return OllamaProvider(_setting("ai/ollama_model", OLLAMA_DEFAULT_MODEL),
                              _setting("ai/ollama_url", OLLAMA_BASE_URL))

    if kind == "gemini":
        return gemini()
    if kind == "ollama":
        return ollama()
    if kind == "custom":
        return OpenAICompatProvider(_setting("ai/custom_url", "http://localhost:8080/v1"),
                                    _setting("ai/custom_key", "none"),
                                    _setting("ai/custom_model", "default"), name="Custom")
    # auto: first available wins
    for p in (gemini(), ollama()):
        ok, _ = p.available()
        if ok:
            return p
    return None


def describe_auto_state() -> list[tuple[str, bool, str]]:
    """Per-backend availability for the setup card: [(name, ok, reason)]."""
    out = []
    for p in (GeminiProvider(_setting("ai/gemini_key"), gemini_model_setting()),
              OllamaProvider(_setting("ai/ollama_model", OLLAMA_DEFAULT_MODEL),
                             _setting("ai/ollama_url", OLLAMA_BASE_URL))):
        ok, why = p.available()
        out.append((p.label(), ok, why))
    return out
