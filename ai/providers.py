"""
LLM providers behind one streaming interface.

Every provider yields ``Chunk`` objects from ``chat()``: text deltas as they
arrive, then a final chunk carrying any complete tool calls. Gemini (Google
AI Studio) is the one backend the app offers; it is reached over its
OpenAI-compatible endpoint, so ``OpenAICompatProvider`` holds the wire format
and ``GeminiProvider`` the Gemini-specific behaviour.

There used to be four choices (Auto, Gemini, a local Ollama model, a custom
endpoint). A stored choice of "ollama" with no model pulled left the panel
dead even though a working Gemini key was saved, and the local model was too
slow and too weak to recommend. ``build_provider`` now always returns Gemini
and ignores a stored ``ai/provider``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

# httpx is imported inside the functions that talk to a provider. At module
# level it (and the rich/click CLI it drags in) cost about 1 s of every
# launch, for an assistant most sessions never open.

logger = logging.getLogger("K2.AI.Providers")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_DEFAULT_MODEL = "gemini-3.6-flash"
_GEMINI_RETIRED = {"gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"}
GEMINI_KEY_URL = "https://aistudio.google.com/apikey"


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
# OpenAI-compatible wire format
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

        import httpx

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

    # Seconds to wait for the model to start answering before trying the
    # next one. Measured 2026-11-15 on the free tier: the same request to the
    # default model took 3 s once and 51 s the next time, with the user
    # watching "Thinking" throughout (the old limit was 120 s).
    RESPONSE_TIMEOUT_S = 30.0

    def __init__(self, api_key: str, model: str = GEMINI_DEFAULT_MODEL):
        super().__init__(GEMINI_BASE_URL, api_key, model or GEMINI_DEFAULT_MODEL,
                         timeout=self.RESPONSE_TIMEOUT_S)

    # Free-tier flash models return 503 "high demand" / 429 quota in bursts,
    # or accept the request and sit on it; step through siblings before
    # giving up, but only if nothing streamed yet. 3.5-flash is first because
    # it answered while the other two were returning 503 (2026-11-15).
    FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest")
    _TRANSIENT = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand",
                  "timed out")

    _QUOTA = ("429", "RESOURCE_EXHAUSTED", "quota")

    def available(self):
        if not self.api_key or self.api_key == "none":
            return False, "No Gemini API key"
        return True, ""

    @classmethod
    def _plain_reason(cls, error: str, tried: int) -> str:
        """The sentence the chat shows when every model failed. It used to
        show the provider's raw reply (a JSON blob, or "The read operation
        timed out"), which does not say whether to wait or to fix something."""
        if any(t in error for t in cls._QUOTA):
            return ("The free Gemini allowance for this key is used up for now. "
                    "Wait a few minutes and ask again; a paid key has higher limits.")
        return (f"Gemini is overloaded right now ({tried} models tried, none answered). "
                "Nothing is wrong with the key or the rocket; ask again in a minute.")

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
                    if started or not transient:
                        raise
                    if i == len(chain) - 1:
                        logger.warning("Gemini: no model answered; last error %s", str(e)[:200])
                        raise ProviderError(self._plain_reason(str(e), len(chain))) from e
                    logger.warning("Gemini %s unavailable (%s); trying %s", model, str(e)[:80], chain[i + 1])
        finally:
            self.model = primary


def check_gemini_key(api_key: str, timeout: float = 10.0) -> tuple[bool, str]:
    """Ask Gemini whether it accepts ``api_key``: (ok, sentence for the user).

    Lists the models, which costs no quota and sends nothing about the
    rocket. Blocks for up to ``timeout`` seconds, so call it off the GUI
    thread.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Paste a key first."
    import httpx
    try:
        r = httpx.get(GEMINI_BASE_URL + "models", timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as e:
        return False, f"Could not reach Gemini ({type(e).__name__}). Check the internet connection."
    if r.status_code == 200:
        return True, "Key accepted."
    if r.status_code in (400, 401, 403):
        return False, "Gemini rejected this key. Copy it again from AI Studio."
    return False, _http_error_text(r)


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


def build_provider() -> Provider:
    """The Gemini provider with the stored key and model. A missing key
    shows up in ``available()``, not as ``None``."""
    return GeminiProvider(_setting("ai/gemini_key"), gemini_model_setting())
