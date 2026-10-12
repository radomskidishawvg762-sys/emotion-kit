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
    def _urlopen(request: urllib.request.Request, timeout: float = 0) -> object:
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


def test_openai_client_requires_credentials() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="https://example.invalid/v1", api_key="", model="m")
