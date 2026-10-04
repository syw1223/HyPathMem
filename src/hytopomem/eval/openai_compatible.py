from __future__ import annotations

import json
import contextlib
import http.client
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY = 1.5
DEFAULT_MAX_DELAY = 20.0


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    usage: dict[str, Any]
    elapsed_seconds: float


@dataclass
class OpenAICompatibleChatClient:
    api_key: str
    base_url: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "E_MEM_API_KEY",
        base_url_env: str = "E_MEM_BASE_URL",
        default_base_url: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> "OpenAICompatibleChatClient":
        api_key = os.getenv(api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing API key environment variable: {api_key_env}")
        base_url = os.getenv(base_url_env, default_base_url or "").strip()
        if not base_url:
            raise RuntimeError(f"missing base URL environment variable: {base_url_env}")
        return cls(
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_attempts=env_int("HYTOPOMEM_OPENAI_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
            base_delay=env_float("HYTOPOMEM_OPENAI_BASE_DELAY", DEFAULT_BASE_DELAY),
            max_delay=env_float("HYTOPOMEM_OPENAI_MAX_DELAY", DEFAULT_MAX_DELAY),
        )

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage | dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        return self.chat_completion_with_metadata(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        ).content

    def chat_completion_with_metadata(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessage | dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message_payload(message) for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        started = time.perf_counter()
        response = self._post_json(self.chat_completions_url(), payload)
        elapsed = time.perf_counter() - started
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"chat completion response has no choices: {response}")
        message = choices[0].get("message") or {}
        usage = response.get("usage") or {}
        return ChatCompletionResult(
            content=str(message.get("content") or ""),
            usage=usage if isinstance(usage, dict) else {},
            elapsed_seconds=elapsed,
        )

    def chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if base.endswith("/chat"):
            return f"{base}/completions"
        return f"{base}/chat/completions"

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    try:
                        return json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"OpenAI-compatible endpoint returned non-JSON body: {body[:240]!r}"
                        ) from exc
            except Exception as exc:  # noqa: BLE001
                display_error: Exception = exc
                if isinstance(exc, urllib.error.HTTPError):
                    with contextlib.suppress(Exception):
                        body = exc.read().decode("utf-8")
                        display_error = RuntimeError(
                            f"OpenAI-compatible endpoint HTTP {exc.code}: {body[:240]!r}"
                        )
                last_error = display_error
                if not is_retryable(exc) or attempt >= self.max_attempts:
                    raise display_error from exc
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                delay *= 1.0 + random.uniform(0.0, 0.25)
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("unreachable OpenAI-compatible retry state")


def message_payload(message: ChatMessage | dict[str, str]) -> dict[str, str]:
    if isinstance(message, ChatMessage):
        return {"role": message.role, "content": message.content}
    return {"role": str(message["role"]), "content": str(message["content"])}


def is_retryable(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    if isinstance(exc, (TimeoutError, urllib.error.URLError, http.client.RemoteDisconnected)):
        return True
    return False


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except ValueError:
        return default
