"""Small memory store with per-state weighted recall."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from emotion_kit.state import EMOTIONS, Emotion, parse_emotion

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RecallWeights:
    tag_overlap: float = 0.34
    affinity: float = 0.33
    recency: float = 0.33

    def __post_init__(self) -> None:
        for name in ("tag_overlap", "affinity", "recency"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} weight must be >= 0")


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    text: str
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    emotion_affinity: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("memory id is required")
        if not self.text.strip():
            raise ValueError(f"memory {self.id} text is required")


@dataclass(frozen=True)
class ScoredMemory:
    memory: MemoryEntry
    score: float
    components: Mapping[str, float]


def tokenize(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _TOKEN.finditer(text)}


def tag_overlap(query: str, entry: MemoryEntry) -> float:
    """Score in ``[0, 1]``. A tag counts only when the whole tag is a substring."""

    query_tokens = tokenize(query)
    folded = query.casefold()
    if entry.tags:
        hits = 0
        for tag in entry.tags:
            tag_folded = tag.casefold()
            if tag_folded and tag_folded in folded:
                hits += 1
        tag_score = hits / len(entry.tags)
    else:
        tag_score = 0.0
    text_tokens = tokenize(entry.text)
    if text_tokens and query_tokens:
        text_score = len(text_tokens & query_tokens) / len(text_tokens)
    else:
        text_score = 0.0
    if entry.tags:
        return min(1.0, (0.75 * tag_score) + (0.25 * text_score))
    return min(1.0, text_score)


def affinity_score(entry: MemoryEntry, emotion: Emotion) -> float:
    if not entry.emotion_affinity:
        return 0.0
    raw = entry.emotion_affinity.get(emotion, 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def recency_score(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return (index + 1) / total


class MemoryStore:
    """In-memory entries recalled with a per-state weight table."""

    def __init__(
        self,
        entries: Sequence[MemoryEntry] | None = None,
        weights: Mapping[Emotion, RecallWeights] | None = None,
        top_k: int = 3,
    ) -> None:
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        self.top_k = top_k
        self._entries: list[MemoryEntry] = []
        self.weights: dict[Emotion, RecallWeights] = dict(weights or _default_weights())
        missing = [name for name in EMOTIONS if name not in self.weights]
        if missing:
            raise ValueError(f"memory weights missing emotions: {', '.join(missing)}")
        for entry in entries or ():
            self.add(entry)

    def add(self, entry: MemoryEntry) -> None:
        if any(existing.id == entry.id for existing in self._entries):
            raise ValueError(f"duplicate memory id {entry.id!r}")
        self._entries.append(entry)

    def __len__(self) -> int:
        return len(self._entries)

    def recall(
        self,
        emotion: Emotion,
        query: str,
        top_k: int | None = None,
    ) -> list[ScoredMemory]:
        limit = self.top_k if top_k is None else top_k
        if limit < 0:
            raise ValueError("top_k must be >= 0")
        if limit == 0 or not self._entries:
            return []
        weights = self.weights[parse_emotion(emotion)]
        total = len(self._entries)
        scored: list[ScoredMemory] = []
        for index, entry in enumerate(self._entries):
            components = {
                "tag_overlap": tag_overlap(query, entry),
                "affinity": affinity_score(entry, emotion),
                "recency": recency_score(index, total),
            }
            score = (
                weights.tag_overlap * components["tag_overlap"]
                + weights.affinity * components["affinity"]
                + weights.recency * components["recency"]
            )
            scored.append(ScoredMemory(memory=entry, score=score, components=components))
        scored.sort(key=lambda item: (-item.score, -item.components["recency"], item.memory.id))
        return scored[:limit]


def load_memories(directory: str | Path | None = None) -> tuple[MemoryEntry, ...]:
    """Load ``memories.yaml`` or ``memories.json`` from a config directory."""

    from emotion_kit.strategy import find_config_file, load_mapping, resolve_config_dir

    path = find_config_file(resolve_config_dir(directory), "memories")
    raw = load_mapping(path)
    items = raw.get("memories") or []
    if not isinstance(items, list):
        raise ValueError(f"{path} memories must be a list")
    return tuple(memory_from_dict(item) for item in items)


def memory_from_dict(data: Mapping[str, Any]) -> MemoryEntry:
    if not isinstance(data, Mapping):
        raise ValueError("memory entry must be a mapping")
    raw_tags = data.get("tags") or ()
    if isinstance(raw_tags, str):
        tags = (raw_tags,)
    else:
        tags = tuple(str(tag) for tag in raw_tags)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError(f"memory {data.get('id')!r} metadata must be a mapping")
    affinity = data.get("emotion_affinity")
    parsed_affinity: dict[str, float] | None
    if affinity is None:
        parsed_affinity = None
    elif isinstance(affinity, Mapping):
        parsed_affinity = {str(key): float(value) for key, value in affinity.items()}
    else:
        raise ValueError(f"memory {data.get('id')!r} emotion_affinity must be a mapping")
    return MemoryEntry(
        id=str(data.get("id", "")),
        text=str(data.get("text", "")),
        tags=tags,
        metadata=dict(metadata),
        emotion_affinity=parsed_affinity,
    )


def _default_weights() -> dict[Emotion, RecallWeights]:
    return {parse_emotion(name): RecallWeights() for name in EMOTIONS}
