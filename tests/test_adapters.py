from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from emotion_kit.adapters import LLMError, MockClient, OpenAICompatibleClient


def test_mock_classification_ignores_transcript_history() -> None:
    client = MockClient()
    content = client.complete(
        [
            {"role": "system", "content": "EMOTION_KIT_CLASSIFY"},
            {
                "role": "user",
                "content": "Recent transcript:\nuser: 我很焦虑\n\nLatest user message:\n嗯，也许吧。",
            },
        ]
    ).content
    payload = json.loads(content)
    assert payload["emotion"] == "calm"
    assert payload["confidence"] < 0.45


def test_openai_client_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "pong"}}], "model": "gpt-test"}
            ).encode()

    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["body"] = json.loads(request.data.decode())  # type: ignore[union-attr]
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1/",
        api_key="sk-test",
        model="gpt-main",
    )
    result = client.complete(
        [{"role": "user", "content": "ping"}],
        model="gpt-detect",
        temperature=0.2,
        max_tokens=32,
    )
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["model"] == "gpt-detect"
    assert body["max_tokens"] == 32
    assert result.content == "pong"
    assert result.model == "gpt-detect"


def test_openai_client_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"bad key"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    with pytest.raises(LLMError, match="401"):
        client.complete([{"role": "user", "content": "ping"}])
    assert calls["n"] == 1


def test_timeout_becomes_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
        calls["n"] += 1
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr("emotion_kit.adapters.time.sleep", lambda _delay: None)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    with pytest.raises(LLMError, match="timeout") as caught:
        client.complete([{"role": "user", "content": "ping"}])
    assert calls["n"] == 3
    assert caught.value.retryable is True


@pytest.mark.parametrize(
    ("error", "match"),
    [
        (TimeoutError("timed out"), "timeout"),
        (ConnectionResetError("reset by peer"), "connection reset"),
        (OSError("network down"), "connection failed"),
    ],
)
def test_socket_errors_become_llm_error(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    match: str,
) -> None:
    calls = {"n": 0}

    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
        calls["n"] += 1
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr("emotion_kit.adapters.time.sleep", lambda _delay: None)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    with pytest.raises(LLMError, match=match):
        client.complete([{"role": "user", "content": "ping"}])
    assert calls["n"] == (3 if isinstance(error, TimeoutError) else 1)


def test_retryable_http_status_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [{"message": {"content": "pong"}}],
                    "usage": {"total_tokens": 11},
                }
            ).encode()

    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "unavailable",
                hdrs=None,  # type: ignore[arg-type]
                fp=io.BytesIO(b"busy"),
            )
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr("emotion_kit.adapters.time.sleep", lambda _delay: None)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    result = client.complete([{"role": "user", "content": "ping"}])
    assert calls["n"] == 3
    assert result.content == "pong"
    assert result.usage == {"total_tokens": 11}


def test_empty_content_is_llm_error_and_ignores_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "reasoning_content": "hidden chain of thought",
                                }
                            }
                        ]
                    }
                ).encode()

        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    client = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    with pytest.raises(LLMError, match="empty"):
        client.complete([{"role": "user", "content": "ping"}])


def test_timeout_env_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMOTION_KIT_TIMEOUT", "12")
    from_env = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
    )
    assert from_env.timeout == 12
    explicit = OpenAICompatibleClient(
        base_url="https://example.invalid/v1",
        api_key="sk-test",
        model="gpt-main",
        timeout=5,
    )
    assert explicit.timeout == 5
    with pytest.raises(ValueError, match="EMOTION_KIT_TIMEOUT"):
        monkeypatch.setenv("EMOTION_KIT_TIMEOUT", "nope")
        OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="sk-test",
            model="gpt-main",
        )


def test_openai_client_requires_credentials() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="https://example.invalid/v1", api_key="", model="m")
