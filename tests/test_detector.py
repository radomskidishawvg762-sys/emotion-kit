from __future__ import annotations

from dataclasses import replace

import pytest

from emotion_kit.adapters import MockClient
from emotion_kit.detector import Detector, detection_from_json, load_rules
from emotion_kit.strategy import load_orchestration


@pytest.fixture
def detector() -> Detector:
    config = load_orchestration()
    return Detector(load_rules(), settings=config.detector)


def test_rule_detects_each_clear_state(detector: Detector) -> None:
    samples = {
        "我现在很平静，没事了": "calm",
        "我有点焦虑，明天发布会来不及了，怎么办？": "anxious",
        "试了三次都失败，真的很受挫，搞不定。": "frustrated",
        "太棒了，终于可以演示，好兴奋！": "excited",
    }
    for text, emotion in samples.items():
        detection = detector.detect_rule(text)
        assert detection.emotion == emotion
        assert detection.source == "rule"
        assert detection.confidence >= 0.90
        assert detection.evidence


def test_english_keywords(detector: Detector) -> None:
    assert detector.detect_rule("I feel ANXIOUS about the deadline").emotion == "anxious"
    assert detector.detect_rule("I am fine, thanks").emotion == "calm"
    assert detector.detect_rule("This is awesome").emotion == "excited"
    assert detector.detect_rule("I am frustrated and stuck").emotion == "frustrated"


def test_empty_and_unknown_text_stay_low_confidence(detector: Detector) -> None:
    detection = detector.detect_rule("嗯，也许吧。")
    assert detection.emotion == "calm"
    assert detection.confidence == pytest.approx(0.34)
    assert detection.evidence == ()


def test_contested_text_is_not_confident(detector: Detector) -> None:
    detection = detector.detect_rule("我很焦虑，但是又很兴奋")
    assert detection.confidence < 0.62
    assert detection.evidence


def test_auto_without_client_uses_rules(detector: Detector) -> None:
    detection = detector.detect("我有点焦虑，担心来不及")
    assert detection.source == "rule"
    assert detection.emotion == "anxious"


def test_auto_keeps_a_confident_rule_without_calling_llm(detector: Detector) -> None:
    client = MockClient()
    detector.llm = client
    detection = detector.detect("我有点焦虑，担心来不及")
    assert detection.source == "rule"
    assert client.calls == []


def test_auto_asks_llm_when_rule_confidence_is_low(detector: Detector) -> None:
    client = MockClient(
        script=['{"emotion": "frustrated", "confidence": 0.81, "evidence": ["失败"]}']
    )
    detector.llm = client
    detection = detector.detect("今天吃什么")
    assert detection.source == "llm"
    assert detection.emotion == "frustrated"
    assert detection.confidence == pytest.approx(0.81)
    assert len(client.calls) == 1


def test_auto_falls_back_to_rule_when_llm_returns_garbage(detector: Detector) -> None:
    detector.llm = MockClient(script=["not json at all"])
    detection = detector.detect("嗯")
    assert detection.source == "rule"
    assert detection.emotion == "calm"


def test_llm_mode_parses_fenced_json() -> None:
    config = load_orchestration()
    client = MockClient(
        script=['```json\n{"emotion": "excited", "confidence": "0.8", "evidence": ["太棒"]}\n```']
    )
    detector = Detector(
        load_rules(),
        settings=replace(config.detector, mode="llm"),
        llm=client,
        model="detector-model",
    )
    detection = detector.detect("anything")
    assert detection.emotion == "excited"
    assert detection.source == "llm"
    assert detection.confidence == pytest.approx(0.8)
    assert client.calls[0]["model"] == "detector-model"
    assert client.calls[0]["temperature"] == 0.0


def test_llm_mode_requires_a_client_and_rejects_bad_json() -> None:
    config = load_orchestration()
    detector = Detector(load_rules(), settings=replace(config.detector, mode="llm"))
    with pytest.raises(RuntimeError):
        detector.detect("hello")

    detector.llm = MockClient(script=["nope"])
    with pytest.raises(ValueError):
        detector.detect("hello")


def test_detection_from_json_rejects_unknown_label() -> None:
    with pytest.raises(ValueError):
        detection_from_json('{"emotion": "sad", "confidence": 0.9, "evidence": []}')
