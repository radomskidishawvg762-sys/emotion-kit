"""Emotion orchestration middleware.

Strategy labels (calm, anxious, frustrated, excited) select reply behavior,
thinking depth, and tool policy for an OpenAI-compatible chat model. They are
a behavioral simulation, not a measurement of subjective emotion.
"""

from emotion_kit.adapters import (
    CLASSIFY_MARKER,
    LLMError,
    LLMResult,
    MockClient,
    OpenAICompatibleClient,
)
from emotion_kit.detector import Detector
from emotion_kit.memory import MemoryEntry, MemoryStore, RecallWeights, ScoredMemory
from emotion_kit.pipeline import Orchestrator, TurnResult, build_orchestrator
from emotion_kit.state import EMOTIONS, DebounceConfig, Detection, StickyState
from emotion_kit.strategy import Strategy, ToolPolicy

__version__ = "0.1.0"

__all__ = [
    "CLASSIFY_MARKER",
    "EMOTIONS",
    "DebounceConfig",
    "Detection",
    "Detector",
    "LLMError",
    "LLMResult",
    "MemoryEntry",
    "MemoryStore",
    "MockClient",
    "OpenAICompatibleClient",
    "Orchestrator",
    "RecallWeights",
    "ScoredMemory",
    "StickyState",
    "Strategy",
    "ToolPolicy",
    "TurnResult",
    "build_orchestrator",
    "__version__",
]
