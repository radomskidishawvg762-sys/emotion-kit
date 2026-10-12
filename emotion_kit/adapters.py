"""OpenAI-compatible chat client and an offline mock.

The mock lets the whole pipeline run with no API key and no network.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

# Detector prompts include this marker. The mock uses it to return JSON.
CLASSIFY_MARKER = "EMOTION_KIT_CLASSIFY"

Message = Mapping[str, str]


# One try plus this many retries. Only timeouts and 429/502/503/504 are retried.
_MAX_RETRIES = 2
_RETRYABLE_HTTP = frozenset({429, 502, 503, 504})
_BACKOFF_SECONDS = 0.2
_DEFAULT_TIMEOUT = 90.0


class LLMError(RuntimeError):
    """The chat endpoint failed or returned a payload we cannot read."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class LLMResult:
    content: str
    model: str
    raw: dict[str, Any] | None = None
    usage: Mapping[str, Any] | None = None


class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> LLMResult: ...


class OpenAICompatibleClient:
    """POST ``{base_url}/chat/completions``.

    ``base_url`` should already include the ``/v1`` prefix, for example
    ``https://api.openai.com/v1``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required")
        if not api_key.strip():
            raise ValueError("api_key is required")
        if not model.strip():
            raise ValueError("model is required")
        self.base_url = base_url.strip().rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = _resolve_timeout(timeout)

    @property
    def chat_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> LLMResult:
        chosen = model or self.model
        payload = {
            "model": chosen,
            "messages": [_plain_message(message) for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        last_error: LLMError | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._post(body, model=chosen)
            except LLMError as exc:
                last_error = exc
                if attempt >= _MAX_RETRIES or not exc.retryable:
                    raise
                time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _post(self, body: bytes, *, model: str) -> LLMResult:
        request = urllib.request.Request(
            self.chat_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(
                f"LLM HTTP {exc.code}: {detail}",
                retryable=exc.code in _RETRYABLE_HTTP,
            ) from exc
        except urllib.error.URLError as exc:
            raise _error_from_cause(exc.reason) from exc
        except TimeoutError as exc:
            raise _error_from_cause(exc) from exc
        except OSError as exc:
            raise _error_from_cause(exc) from exc

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LLMError("LLM response was not JSON") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM response JSON was not an object")
        try:
            message = parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"LLM response missing choices[0].message: {parsed!r}"[:400]) from exc
        if not isinstance(message, dict):
            raise LLMError("LLM message payload was not an object")
        content = _message_text(message)
        if not content.strip():
            raise LLMError("LLM response content was empty")
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            usage = None
        return LLMResult(content=content, model=model, raw=parsed, usage=usage)


class MockClient:
    """Deterministic stand-in for tests and the keyless demo.

    Pass ``script`` to force successive completions. Otherwise classification
    prompts return JSON, deliberation prompts return a short note, and the
    final turn returns a reply that echoes the active strategy label.
    """

    def __init__(self, script: Sequence[str] | None = None, model: str = "mock-emotion") -> None:
        self.script: list[str] = list(script or [])
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> LLMResult:
        chosen = model or self.model
        plain = [_plain_message(message) for message in messages]
        self.calls.append(
            {
                "messages": plain,
                "model": chosen,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.script:
            content = self.script.pop(0)
        else:
            content = _mock_content(plain)
        return LLMResult(content=content, model=chosen, raw=None)


def _plain_message(message: Message) -> dict[str, str]:
    role = str(message.get("role", "user"))
    content = message.get("content", "")
    return {"role": role, "content": str(content)}


def _resolve_timeout(explicit: float | None) -> float:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError("timeout must be > 0")
        return float(explicit)
    raw = os.environ.get("EMOTION_KIT_TIMEOUT")
    if raw is None or not raw.strip():
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"EMOTION_KIT_TIMEOUT must be a number, got {raw.strip()!r}") from exc
    if value <= 0:
        raise ValueError("EMOTION_KIT_TIMEOUT must be > 0")
    return value


def _error_from_cause(reason: object) -> LLMError:
    if isinstance(reason, TimeoutError):
        return LLMError(f"LLM timeout: {reason}", retryable=True)
    if isinstance(reason, ConnectionResetError):
        return LLMError(f"LLM connection reset: {reason}")
    if isinstance(reason, OSError):
        return LLMError(f"LLM connection failed: {reason}")
    return LLMError(f"LLM connection failed: {reason}")


def _message_text(message: Mapping[str, Any]) -> str:
    # ``reasoning_content`` is model scratch space, not the user-visible reply.
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


_MOCK_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anxious", ("焦虑", "担心", "紧张", "来不及", "怎么办", "anxious", "worried")),
    ("frustrated", ("受挫", "失败", "搞不定", "崩溃", "frustrated", "stuck")),
    ("excited", ("兴奋", "太棒", "终于", "激动", "excited", "awesome")),
    ("calm", ("平静", "没事", "还好", "calm", "fine")),
)


def _mock_content(messages: Sequence[Mapping[str, str]]) -> str:
    blob = "\n".join(message.get("content", "") for message in messages)
    user = _last_user(messages)
    if CLASSIFY_MARKER in blob:
        emotion, confidence, evidence = _mock_classify(_latest_for_classifier(user))
        return json.dumps(
            {"emotion": emotion, "confidence": confidence, "evidence": evidence},
            ensure_ascii=False,
        )
    if "内部思考草稿" in blob:
        return "先给一个最小可执行步骤，再对照召回记忆里的事实。"
    label_match = re.search(
        r"当前编排状态：[^\n（]*（(calm|anxious|frustrated|excited)）",
        blob,
    )
    label = label_match.group(1) if label_match else "calm"
    preview = user.replace("\n", " ").strip()
    if len(preview) > 80:
        preview = preview[:80] + "…"
    return (
        f"（模拟回复 / {label}）按当前策略处理：「{preview}」。"
        "下一步收成一个可执行动作，并遵守工具策略。"
    )


def _mock_classify(text: str) -> tuple[str, float, list[str]]:
    folded = text.casefold()
    for emotion, hints in _MOCK_HINTS:
        hit = next((hint for hint in hints if hint.casefold() in folded), None)
        if hit is not None:
            return emotion, 0.93, [hit]
    return "calm", 0.4, []


def _last_user(messages: Sequence[Mapping[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _latest_for_classifier(user_prompt: str) -> str:
    marker = "Latest user message:\n"
    if marker in user_prompt:
        return user_prompt.rsplit(marker, 1)[-1].strip()
    return user_prompt
