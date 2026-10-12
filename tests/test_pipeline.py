from __future__ import annotations

import pytest

from emotion_kit.adapters import LLMError, LLMResult, MockClient
from emotion_kit.memory import MemoryStore
from emotion_kit.pipeline import Orchestrator, build_orchestrator
from emotion_kit.state import DebounceConfig, Detection, StickyState
from emotion_kit.strategy import load_orchestration

SCRIPTED_TURNS = (
    "我有点焦虑，明天发布会来不及了，怎么办？",
    "嗯，也许吧。",
    "试了三次都失败，真的很受挫，搞不定。",
    "等等，通过了！太棒了，终于可以演示，好兴奋！",
    "好，我平静下来了，按清单收尾就行。",
)


@pytest.mark.parametrize("mode", ["auto", "llm"])
def test_scripted_mock_path_switches_and_holds(mode: str) -> None:
    orchestrator = build_orchestrator(mode=mode, force_mock=True)
    assert isinstance(orchestrator.client, MockClient)
    expected = (
        ("anxious", True, "immediate_high_confidence", "mem-release-checklist"),
        ("anxious", False, "below_min_confidence", "mem-release-checklist"),
        ("frustrated", True, "immediate_high_confidence", "mem-env-failure"),
        ("excited", True, "immediate_high_confidence", "mem-demo-win"),
        ("calm", True, "immediate_high_confidence", "mem-calm-close"),
    )
    for text, (emotion, switched, reason, memory_id) in zip(SCRIPTED_TURNS, expected, strict=True):
        result = orchestrator.turn(text)
        assert result.state.emotion == emotion
        assert result.state.switched is switched
        assert result.state.reason == reason
        if mode == "llm" and switched:
            assert result.detection.source == "llm"
            assert result.detection.confidence >= 0.90
        assert result.memories
        assert result.memories[0].memory.id == memory_id
        assert emotion in result.reply
        assert result.messages[0]["role"] == "system"
        system = str(result.messages[0]["content"])
        assert result.strategy.system_prompt_delta.strip() in system
        assert memory_id in system
        assert "max_tool_calls" in system

    anxious = orchestrator.strategies["anxious"]
    assert anxious.thinking_depth == 2
    assert len(orchestrator.history) == 10


def test_thinking_depth_equals_main_model_calls() -> None:
    orchestrator = build_orchestrator(mode="rule", force_mock=True)
    assert isinstance(orchestrator.client, MockClient)
    result = orchestrator.turn("我有点焦虑，明天发布会来不及了，怎么办？")
    assert result.strategy.thinking_depth == 2
    assert len(result.deliberation_notes) == 1
    assert len(orchestrator.client.calls) == 2
    assert orchestrator.client.calls[0]["max_tokens"] == 240
    assert orchestrator.client.calls[-1]["max_tokens"] == 800
    final_system = orchestrator.client.calls[-1]["messages"][0]["content"]
    assert result.deliberation_notes[0] in final_system
    assert "send_message" in final_system
    assert not result.strategy.tool_policy.allows("delete")
    calm = orchestrator.turn("我现在很平静，没事了")
    assert calm.strategy.thinking_depth == 1
    assert calm.deliberation_notes == ()
    assert len(orchestrator.client.calls) == 3


def test_pipeline_debounce_holds_until_the_second_qualified_turn() -> None:
    config = load_orchestration()
    client = MockClient()
    detections = [
        Detection("frustrated", 0.80, ("失败",), "rule"),
        Detection("frustrated", 0.80, ("失败",), "rule"),
    ]

    class ScriptDetector:
        def detect(self, text: str, history=None) -> Detection:  # type: ignore[no-untyped-def]
            return detections.pop(0)

    orchestrator = Orchestrator(
        detector=ScriptDetector(),  # type: ignore[arg-type]
        state=StickyState(
            DebounceConfig(
                min_confidence=0.45,
                switch_confidence=0.60,
                immediate_confidence=0.99,
                consecutive_turns=2,
            )
        ),
        strategies=config.states,
        memory=MemoryStore(weights=config.weights, top_k=config.top_k),
        client=client,
        base_system_prompt=config.base_system_prompt,
        main_model="mock-emotion",
    )
    first = orchestrator.turn("still calm")
    assert first.state.emotion == "calm"
    assert first.state.reason == "pending_debounce"
    assert len(client.calls) == 1
    second = orchestrator.turn("now frustrated")
    assert second.state.switched
    assert second.state.emotion == "frustrated"
    assert second.strategy.thinking_depth == 2
    assert len(client.calls) == 3


def test_deliberation_notes_are_passed_into_later_drafts() -> None:
    client = MockClient(script=["第一稿要点", "第二稿要点", "最终回复"])
    orchestrator = build_orchestrator(
        mode="rule",
        client=client,
        overrides={"states": {"anxious": {"thinking_depth": 3}}},
    )
    result = orchestrator.turn("我有点焦虑，明天发布会来不及了，怎么办？")
    assert result.deliberation_notes == ("第一稿要点", "第二稿要点")
    assert result.reply == "最终回复"
    second_system = client.calls[1]["messages"][0]["content"]
    assert "第一稿要点" in second_system
    assert "第二稿要点" not in second_system
    final_system = client.calls[2]["messages"][0]["content"]
    assert "第一稿要点" in final_system
    assert "第二稿要点" in final_system


def test_failed_turn_leaves_state_and_history_unchanged() -> None:
    class Flaky:
        def __init__(self) -> None:
            self.calls = 0
            self.fail_final = True

        def complete(
            self,
            messages: object,
            *,
            model: str | None = None,
            temperature: float = 0.3,
            max_tokens: int = 800,
        ) -> LLMResult:
            self.calls += 1
            if self.fail_final and max_tokens > 240:
                self.fail_final = False
                raise LLMError("upstream timeout")
            if max_tokens <= 240:
                return LLMResult(f"note-{self.calls}", model or "mock")
            return LLMResult("（模拟回复 / anxious）收成一步。", model or "mock")

    client = Flaky()
    orchestrator = build_orchestrator(mode="rule", client=client)
    with pytest.raises(LLMError, match="upstream timeout"):
        orchestrator.turn("我有点焦虑，明天发布会来不及了，怎么办？")
    assert client.calls == 2
    assert orchestrator.state.emotion == "calm"
    assert orchestrator.state.confidence == pytest.approx(0.5)
    assert orchestrator.state.turns_in_state == 0
    assert orchestrator.state.pending_emotion is None
    assert orchestrator.state.pending_count == 0
    assert orchestrator.history == []

    result = orchestrator.turn("我有点焦虑，明天发布会来不及了，怎么办？")
    assert result.state.emotion == "anxious"
    assert result.state.switched
    assert result.state.reason == "immediate_high_confidence"
    assert [item["role"] for item in orchestrator.history] == ["user", "assistant"]


def test_detect_client_model_is_used_when_detect_model_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMOTION_KIT_DETECT_MODEL", raising=False)
    monkeypatch.delenv("EMOTION_KIT_MAIN_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    detect = MockClient(model="classifier-small")
    main = MockClient(model="chat-large")
    orchestrator = build_orchestrator(client=main, detect_client=detect, mode="llm")
    assert orchestrator.detect_model == "classifier-small"
    assert orchestrator.detector.model == "classifier-small"
    orchestrator.turn("你好")
    assert detect.calls[0]["model"] == "classifier-small"

    explicit = build_orchestrator(
        client=main,
        detect_client=MockClient(model="classifier-small"),
        detect_model="from-arg",
        mode="llm",
    )
    assert explicit.detect_model == "from-arg"


def test_turn_result_exposes_usage() -> None:
    class UsageClient:
        def complete(
            self,
            messages: object,
            *,
            model: str | None = None,
            temperature: float = 0.3,
            max_tokens: int = 800,
        ) -> LLMResult:
            return LLMResult("好的。", model or "mock", usage={"total_tokens": 9})

    orchestrator = build_orchestrator(mode="rule", client=UsageClient())
    result = orchestrator.turn("我现在很平静")
    assert result.usage == {"total_tokens": 9}


def test_passed_history_is_not_stored() -> None:
    orchestrator = build_orchestrator(mode="rule", force_mock=True)
    result = orchestrator.turn(
        "我现在很平静",
        history=({"role": "user", "content": "之前"}, {"role": "assistant", "content": "好"}),
    )
    assert orchestrator.history == []
    roles = [message["role"] for message in result.messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_reset_returns_to_calm() -> None:
    orchestrator = build_orchestrator(mode="rule", force_mock=True)
    orchestrator.turn("太棒了，终于成功，好兴奋")
    assert orchestrator.state.emotion == "excited"
    orchestrator.reset()
    assert orchestrator.state.emotion == "calm"
    assert orchestrator.history == []
