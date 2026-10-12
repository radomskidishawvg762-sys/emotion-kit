"""Sticky strategy state with a confidence debounce.

Labels choose a reply strategy. They do not measure subjective feeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

EMOTIONS: tuple[str, ...] = ("calm", "anxious", "frustrated", "excited")
Emotion = Literal["calm", "anxious", "frustrated", "excited"]
DetectionSource = Literal["rule", "llm"]


def parse_emotion(value: object) -> Emotion:
    if isinstance(value, str) and value in EMOTIONS:
        return cast(Emotion, value)
    names = ", ".join(EMOTIONS)
    raise ValueError(f"unknown emotion {value!r}; expected one of {names}")


@dataclass(frozen=True)
class Detection:
    """One classification of the latest user message."""

    emotion: Emotion
    confidence: float
    evidence: tuple[str, ...] = ()
    source: DetectionSource = "rule"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        parse_emotion(self.emotion)


@dataclass(frozen=True)
class DebounceConfig:
    """Thresholds that keep the active strategy from thrashing.

    A different label is ignored under ``min_confidence``.
    At or above ``immediate_confidence`` it replaces the current label at once.
    Between ``switch_confidence`` and ``immediate_confidence`` it must appear on
    ``consecutive_turns`` qualified turns in a row.
    """

    min_confidence: float = 0.45
    switch_confidence: float = 0.62
    immediate_confidence: float = 0.90
    consecutive_turns: int = 2

    def __post_init__(self) -> None:
        if self.consecutive_turns < 1:
            raise ValueError("consecutive_turns must be >= 1")
        ordered = (
            0.0
            <= self.min_confidence
            <= self.switch_confidence
            <= self.immediate_confidence
            <= 1.0
        )
        if not ordered:
            raise ValueError(
                "confidence thresholds must satisfy 0 <= min <= switch <= immediate <= 1"
            )


@dataclass(frozen=True)
class StateSnapshot:
    emotion: Emotion
    confidence: float
    turns_in_state: int
    pending_emotion: Emotion | None
    pending_count: int
    switched: bool
    previous: Emotion
    reason: str


class StickyState:
    """Hold the active strategy label until a new one clears the debounce."""

    def __init__(
        self,
        debounce: DebounceConfig | None = None,
        initial: Emotion = "calm",
    ) -> None:
        self.debounce = debounce or DebounceConfig()
        self.emotion: Emotion = parse_emotion(initial)
        self.confidence: float = 0.5
        self.turns_in_state: int = 0
        self.pending_emotion: Emotion | None = None
        self.pending_count: int = 0

    def propose(self, detection: Detection) -> StateSnapshot:
        """Return the next snapshot without mutating this state."""

        trial = StickyState(self.debounce, self.emotion)
        trial.confidence = self.confidence
        trial.turns_in_state = self.turns_in_state
        trial.pending_emotion = self.pending_emotion
        trial.pending_count = self.pending_count
        return trial.update(detection)

    def commit(self, snapshot: StateSnapshot) -> None:
        """Apply a snapshot from :meth:`propose` after the turn succeeds."""

        self.emotion = snapshot.emotion
        self.confidence = snapshot.confidence
        self.turns_in_state = snapshot.turns_in_state
        self.pending_emotion = snapshot.pending_emotion
        self.pending_count = snapshot.pending_count

    def update(self, detection: Detection) -> StateSnapshot:
        previous = self.emotion
        if detection.confidence < self.debounce.min_confidence:
            return self._snapshot(previous, switched=False, reason="below_min_confidence")

        if detection.emotion == self.emotion:
            if detection.confidence < self.debounce.switch_confidence:
                self.turns_in_state += 1
                return self._snapshot(previous, switched=False, reason="below_switch_confidence")
            self.pending_emotion = None
            self.pending_count = 0
            self.turns_in_state += 1
            self.confidence = min(1.0, (0.7 * self.confidence) + (0.3 * detection.confidence))
            return self._snapshot(previous, switched=False, reason="reinforce")

        if detection.confidence >= self.debounce.immediate_confidence:
            self._switch(detection)
            return self._snapshot(previous, switched=True, reason="immediate_high_confidence")

        if detection.confidence < self.debounce.switch_confidence:
            self.turns_in_state += 1
            return self._snapshot(previous, switched=False, reason="below_switch_confidence")

        if self.pending_emotion == detection.emotion:
            self.pending_count += 1
        else:
            self.pending_emotion = detection.emotion
            self.pending_count = 1

        if self.pending_count >= self.debounce.consecutive_turns:
            self._switch(detection)
            return self._snapshot(previous, switched=True, reason="consecutive_debounce")

        self.turns_in_state += 1
        return self._snapshot(previous, switched=False, reason="pending_debounce")

    def reset(self, initial: Emotion = "calm") -> None:
        self.emotion = parse_emotion(initial)
        self.confidence = 0.5
        self.turns_in_state = 0
        self.pending_emotion = None
        self.pending_count = 0

    def _switch(self, detection: Detection) -> None:
        self.emotion = detection.emotion
        self.confidence = detection.confidence
        self.turns_in_state = 1
        self.pending_emotion = None
        self.pending_count = 0

    def _snapshot(self, previous: Emotion, *, switched: bool, reason: str) -> StateSnapshot:
        return StateSnapshot(
            emotion=self.emotion,
            confidence=self.confidence,
            turns_in_state=self.turns_in_state,
            pending_emotion=self.pending_emotion,
            pending_count=self.pending_count,
            switched=switched,
            previous=previous,
            reason=reason,
        )
