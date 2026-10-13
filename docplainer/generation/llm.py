"""Chat completion over OpenAI-compatible HTTP APIs.

Groq, Gemini and OpenRouter all expose /chat/completions, so one small client covers every
free provider, and a FallbackChain moves to the next one when a free quota is exhausted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx

from docplainer.config import Settings, get_settings

Messages = list[dict[str, str]]


class LLMError(RuntimeError):
    """Provider failure. `retryable` means another provider may succeed."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class OpenAICompatClient:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 45.0
    max_retries: int = 2
    extra_headers: dict[str, str] | None = None

    def _request(self, messages: Messages, temperature: float, max_tokens: int, stream: bool):
        return {
            "url": f"{self.base_url.rstrip('/')}/chat/completions",
            "headers": {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **(self.extra_headers or {}),
            },
            "json": {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            },
        }

    @staticmethod
    def _check(resp: httpx.Response, provider: str) -> None:
        if resp.status_code == 200:
            return
        try:
            body = resp.text[:200]
        except httpx.ResponseNotRead:
            body = ""
        retryable = resp.status_code in (402, 403, 408, 409, 425, 429) or resp.status_code >= 500
        if resp.status_code in (401, 403):
            raise LLMError(f"{provider}: API key rejected ({resp.status_code}).", retryable=True)
        raise LLMError(f"{provider}: HTTP {resp.status_code} {body}", retryable=retryable)

    def complete(self, messages: Messages, temperature: float = 0.2, max_tokens: int = 800) -> str:
        req = self._request(messages, temperature, max_tokens, stream=False)
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(timeout=self.timeout, **req)
            except httpx.HTTPError as exc:
                raise LLMError(f"{self.name}: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503) and attempt < self.max_retries:
                time.sleep(min(float(resp.headers.get("retry-after", 2**attempt)), 8.0))
                continue
            self._check(resp, self.name)
            return resp.json()["choices"][0]["message"]["content"] or ""
        raise LLMError(f"{self.name}: retries exhausted")

    def stream(
        self, messages: Messages, temperature: float = 0.2, max_tokens: int = 800
    ) -> Iterator[str]:
        req = self._request(messages, temperature, max_tokens, stream=True)
        try:
            with httpx.stream("POST", timeout=self.timeout, **req) as resp:
                if resp.status_code != 200:
                    resp.read()
                    self._check(resp, self.name)
                for line in resp.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name}: {exc}") from exc


class FallbackChain:
    """Try providers in order. A provider that fails before emitting anything is skipped."""

    def __init__(self, clients: list[OpenAICompatClient]):
        if not clients:
            raise ValueError("No LLM provider configured.")
        self.clients = clients
        self.last_used: str = ""

    def complete(self, messages: Messages, **kw) -> str:
        errors: list[str] = []
        for client in self.clients:
            try:
                text = client.complete(messages, **kw)
                self.last_used = client.name
                return text
            except LLMError as exc:
                errors.append(str(exc))
                if not exc.retryable:
                    break
        raise LLMError("All providers failed: " + " | ".join(errors))

    def stream(self, messages: Messages, **kw) -> Iterator[str]:
        errors: list[str] = []
        for client in self.clients:
            started = False
            try:
                for piece in client.stream(messages, **kw):
                    started = True
                    self.last_used = client.name
                    yield piece
                return
            except LLMError as exc:
                if started:  # cannot switch providers mid-answer
                    raise
                errors.append(str(exc))
        raise LLMError("All providers failed: " + " | ".join(errors))


def build_chain(keys: dict[str, str | None], settings: Settings | None = None) -> FallbackChain:
    """keys: provider name -> API key (missing/empty providers are skipped)."""
    cfg = settings or get_settings()
    t = cfg.llm_timeout_s
    presets = [
        ("groq", "https://api.groq.com/openai/v1", cfg.groq_model, None),
        (
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            cfg.gemini_model,
            None,
        ),
        ("nvidia", "https://integrate.api.nvidia.com/v1", cfg.nvidia_model, None),
        (
            "openrouter",
            "https://openrouter.ai/api/v1",
            cfg.openrouter_model,
            {"X-Title": "DocPlainer"},
        ),
    ]
    clients = [
        OpenAICompatClient(name, url, keys[name], model, timeout=t, extra_headers=headers)
        for name, url, model, headers in presets
        if keys.get(name)
    ]
    return FallbackChain(clients)
