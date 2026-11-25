"""Provider-independent LLM interface + the OpenRouter implementation.

Configuration comes from environment variables, loaded from the project's
git-ignored ``.env`` file:

    SEPHRA_LLM_PROVIDER=openrouter
    OPENROUTER_API_KEY=...            (never logged or printed)
    OPENROUTER_MODEL=baidu/ernie-5.5-flash-preview:free

Retrieval code never imports this module. Only the query parser and the
answer generator use an ``LLMClient``, so the provider can be swapped by
writing one new class.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from sephra import config

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DAILY_LIMIT_COOLDOWN_S = 600   # after a daily-limit 429, skip calls for 10 minutes


class LLMError(RuntimeError):
    """The LLM call failed (network, HTTP error, empty or invalid reply)."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited us (common on free tiers)."""


class LLMNotConfiguredError(LLMError):
    """A provider is selected but its settings are incomplete."""


@dataclass
class LLMReply:
    text: str
    model: str
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 schema_name: str = "output", temperature: float = 0.2,
                 max_tokens: int = 2000, reasoning_effort: str | None = None) -> LLMReply:
        ...


class OpenRouterClient:
    """Minimal OpenRouter chat-completions client (OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str, timeout_s: float = 90.0,
                 max_retries: int = 2, transport=None) -> None:
        if not api_key:
            raise LLMNotConfiguredError("OPENROUTER_API_KEY is empty")
        import httpx

        self.model = model
        self._max_retries = max_retries
        self._blocked_until = 0.0
        self._http = httpx.Client(
            timeout=timeout_s, transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "X-Title": "Sephra"})

    def __repr__(self) -> str:  # never expose the key
        return f"OpenRouterClient(model={self.model!r})"

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 schema_name: str = "output", temperature: float = 0.2,
                 max_tokens: int = 2000, reasoning_effort: str | None = None) -> LLMReply:
        import httpx

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema_name, "strict": True, "schema": json_schema}}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        if time.monotonic() < self._blocked_until:
            raise LLMRateLimitError("Daily free-tier limit reached earlier; not retrying for now.")

        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._http.post(OPENROUTER_URL, json=body)
            except httpx.HTTPError as exc:            # network problem / timeout
                last_error = LLMError(f"Network error calling OpenRouter: {type(exc).__name__}")
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), time.perf_counter() - start)
                detail = _error_detail(response)
                if response.status_code == 429:
                    if "per-day" in detail or "per day" in detail:
                        # Retrying a DAILY limit cannot succeed: fail fast and
                        # skip calls for a while (circuit breaker).
                        self._blocked_until = time.monotonic() + DAILY_LIMIT_COOLDOWN_S
                        raise LLMRateLimitError(f"OpenRouter daily free-tier limit reached: {detail}")
                    last_error = LLMRateLimitError(f"OpenRouter rate limit reached: {detail}")
                elif response.status_code >= 500:
                    last_error = LLMError(f"OpenRouter server error {response.status_code}: {detail}")
                else:                                  # 400/401/402/404: retrying will not help
                    raise LLMError(f"OpenRouter error {response.status_code}: {detail}")
            if attempt < self._max_retries:
                time.sleep(2 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _parse(self, payload: dict, latency: float) -> LLMReply:
        if "error" in payload:                         # some errors arrive with HTTP 200
            raise LLMError(f"OpenRouter error: {payload['error'].get('message', payload['error'])}")
        try:
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected OpenRouter response format") from exc
        if not text.strip():
            reason = choice.get("finish_reason")
            raise LLMError(f"Model returned no text (finish_reason={reason}); "
                           "max_tokens may be too small for its reasoning")
        return LLMReply(text=text.strip(), model=payload.get("model", self.model),
                        latency_s=latency, usage=payload.get("usage") or {})


def _error_detail(response) -> str:
    try:
        data = response.json()
        return str(data.get("error", {}).get("message", data))[:300]
    except ValueError:
        return response.text[:300]


def get_llm_client(load_env_file: bool = True) -> LLMClient | None:
    """Return a configured client, or None when no provider is selected.

    None means "retrieval-only mode": callers must not pretend an answer
    was generated.
    """
    if load_env_file:
        from dotenv import load_dotenv

        load_dotenv(config.PROJECT_ROOT / ".env", override=False)
    provider = os.getenv("SEPHRA_LLM_PROVIDER", "").strip().lower()
    if not provider:
        return None
    if provider != "openrouter":
        raise LLMNotConfiguredError(f"Unsupported SEPHRA_LLM_PROVIDER={provider!r} (supported: openrouter)")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    if not api_key or not model:
        raise LLMNotConfiguredError("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in .env")
    logger.info("LLM provider: OpenRouter, model %s", model)
    return OpenRouterClient(api_key=api_key, model=model)
