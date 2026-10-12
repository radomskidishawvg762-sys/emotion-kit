from __future__ import annotations

import pytest

from emotion_kit.state import DebounceConfig, Detection, StickyState


def _hit(emotion: str, confidence: float) -> Detection:
    return Detection(emotion, confidence, evidence=(emotion,), source="rule")  # type: ignore[arg-type]


def test_immediate_high_confidence_switches() -> None:
    state = StickyState()
    snapshot = state.update(_hit("anxious", 0.93))
    assert snapshot.switched
    assert snapshot.previous == "calm"
    assert snapshot.emotion == "anxious"
    assert snapshot.reason == "immediate_high_confidence"
    assert snapshot.turns_in_state == 1
    assert snapshot.pending_emotion is None


def test_moderate_confidence_needs_two_turns() -> None:
    state = StickyState(
        DebounceConfig(
            min_confidence=0.45,
            switch_confidence=0.60,
            immediate_confidence=0.99,
            consecutive_turns=2,
        )
    )
    first = state.update(_hit("frustrated", 0.80))
    assert first.emotion == "calm"
    assert first.reason == "pending_debounce"
    assert first.pending_emotion == "frustrated"
    assert first.pending_count == 1

    second = state.update(_hit("frustrated", 0.80))
    assert second.switched
    assert second.emotion == "frustrated"
    assert second.reason == "consecutive_debounce"


def test_low_confidence_noise_does_not_clear_pending() -> None:
    state = StickyState(
        DebounceConfig(
            min_confidence=0.45,
            switch_confidence=0.60,
            immediate_confidence=0.99,
            consecutive_turns=2,
        )
    )
    state.update(_hit("frustrated", 0.80))
    noise = state.update(_hit("excited", 0.20))
    assert noise.emotion == "calm"
    assert noise.reason == "below_min_confidence"
    assert state.pending_emotion == "frustrated"
    assert state.pending_count == 1
    switched = state.update(_hit("frustrated", 0.80))
    assert switched.emotion == "frustrated"
    assert switched.reason == "consecutive_debounce"


def test_below_switch_confidence_does_not_advance_pending() -> None:
    state = StickyState()
    snapshot = state.update(_hit("excited", 0.50))
    assert snapshot.emotion == "calm"
    assert snapshot.reason == "below_switch_confidence"
    assert snapshot.pending_count == 0


def test_a_different_qualified_label_resets_the_streak() -> None:
    state = StickyState(
        DebounceConfig(
            min_confidence=0.45,
            switch_confidence=0.60,
            immediate_confidence=0.99,
            consecutive_turns=2,
        )
    )
    state.update(_hit("frustrated", 0.80))
    snapshot = state.update(_hit("excited", 0.80))
    assert snapshot.emotion == "calm"
    assert snapshot.pending_emotion == "excited"
    assert snapshot.pending_count == 1


def test_same_label_reinforces_and_clears_pending() -> None:
    state = StickyState(
        DebounceConfig(
            min_confidence=0.45,
            switch_confidence=0.60,
            immediate_confidence=0.99,
            consecutive_turns=2,
        )
    )
    state.update(_hit("frustrated", 0.80))
    snapshot = state.update(_hit("calm", 0.70))
    assert snapshot.reason == "reinforce"
    assert snapshot.emotion == "calm"
    assert snapshot.pending_emotion is None
    assert snapshot.confidence == pytest.approx(0.7 * 0.5 + 0.3 * 0.70)


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError):
        DebounceConfig(
            min_confidence=0.8,
            switch_confidence=0.2,
            immediate_confidence=0.9,
            consecutive_turns=2,
        )
    with pytest.raises(ValueError):
        DebounceConfig(consecutive_turns=0)
