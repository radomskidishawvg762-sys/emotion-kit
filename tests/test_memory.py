from __future__ import annotations

import pytest

from emotion_kit.memory import MemoryEntry, MemoryStore, RecallWeights, load_memories, tag_overlap
from emotion_kit.state import EMOTIONS, parse_emotion


def _weights(tag: float = 0.0, affinity: float = 1.0, recency: float = 0.0) -> dict:
    table = RecallWeights(tag_overlap=tag, affinity=affinity, recency=recency)
    return {parse_emotion(name): table for name in EMOTIONS}


def test_affinity_weight_ranks_the_matching_state_first() -> None:
    store = MemoryStore(
        entries=[
            MemoryEntry(
                "calm-note",
                "shared topic",
                tags=("topic",),
                emotion_affinity={"calm": 0.9, "anxious": 0.1},
            ),
            MemoryEntry(
                "anxious-note",
                "shared topic",
                tags=("topic",),
                emotion_affinity={"calm": 0.1, "anxious": 0.9},
            ),
        ],
        weights=_weights(),
        top_k=2,
    )
    anxious = [item.memory.id for item in store.recall("anxious", "topic")]
    calm = [item.memory.id for item in store.recall("calm", "topic")]
    assert anxious == ["anxious-note", "calm-note"]
    assert calm == ["calm-note", "anxious-note"]


def test_top_k_and_empty_store() -> None:
    store = MemoryStore(
        entries=[
            MemoryEntry("one", "alpha", emotion_affinity={"calm": 0.2}),
            MemoryEntry("two", "beta", emotion_affinity={"calm": 0.8}),
        ],
        weights=_weights(),
        top_k=1,
    )
    assert [item.memory.id for item in store.recall("calm", "")] == ["two"]
    assert MemoryStore(weights=_weights()).recall("calm", "anything") == []


def test_duplicate_id_is_rejected() -> None:
    store = MemoryStore(weights=_weights())
    store.add(MemoryEntry("same", "first"))
    with pytest.raises(ValueError, match="duplicate"):
        store.add(MemoryEntry("same", "second"))


def test_shipped_memories_load() -> None:
    entries = load_memories()
    assert {entry.id for entry in entries} == {
        "mem-release-checklist",
        "mem-env-failure",
        "mem-demo-win",
        "mem-calm-close",
    }
    checklist = next(entry for entry in entries if entry.id == "mem-release-checklist")
    assert checklist.emotion_affinity is not None
    assert checklist.emotion_affinity["anxious"] > checklist.emotion_affinity["calm"]
    assert checklist.metadata["source"] == "demo"


def test_chinese_tag_matches_whole_tag_only() -> None:
    entry = MemoryEntry("demo", "plain text", tags=("发布会", "清单"))
    assert tag_overlap("会", entry) == 0
    assert tag_overlap("单", entry) == 0
    assert tag_overlap("发布会快到了", entry) > 0


def test_missing_weight_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        MemoryStore(weights={"calm": RecallWeights()})
