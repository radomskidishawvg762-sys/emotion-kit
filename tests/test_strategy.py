from __future__ import annotations

import json
from pathlib import Path

import pytest

from emotion_kit.state import EMOTIONS
from emotion_kit.strategy import (
    ToolPolicy,
    find_config_file,
    load_mapping,
    load_orchestration,
    orchestration_from_dict,
    resolve_config_dir,
)


def test_shipped_packs_cover_all_four_states() -> None:
    config = load_orchestration()
    assert set(config.states) == set(EMOTIONS)
    depths = {name: config.states[name].thinking_depth for name in EMOTIONS}
    assert depths == {"calm": 1, "anxious": 3, "frustrated": 2, "excited": 2}
    anxious = config.states["anxious"]
    assert "焦虑" in anxious.system_prompt_delta
    assert anxious.tool_policy.max_tool_calls == 2
    assert anxious.tool_policy.allows("search")
    assert not anxious.tool_policy.allows("send_message")
    assert not anxious.tool_policy.allows("code_exec")
    assert "thinking_depth=3" in anxious.render_for_prompt()
    assert set(config.weights) == set(EMOTIONS)
    assert config.base_system_prompt


def test_override_replaces_one_field_and_keeps_the_rest() -> None:
    config = load_orchestration(overrides={"states": {"calm": {"thinking_depth": 4}}})
    assert config.states["calm"].thinking_depth == 4
    assert config.states["anxious"].thinking_depth == 3
    assert "平静" in config.states["calm"].system_prompt_delta


def test_json_config_directory_loads(tmp_path: Path) -> None:
    source = resolve_config_dir()
    raw = load_mapping(find_config_file(source, "orchestration"))
    (tmp_path / "orchestration.json").write_text(
        json.dumps(raw, ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = load_orchestration(tmp_path)
    assert loaded.states["frustrated"].label_zh == "受挫"
    assert loaded.detector.mode == "auto"


def test_missing_state_is_rejected() -> None:
    with pytest.raises(ValueError, match="calm"):
        orchestration_from_dict({"base_system_prompt": "be precise", "states": {}})


def test_empty_allow_means_everything_except_deny() -> None:
    policy = ToolPolicy(allow=(), deny=("delete",), max_tool_calls=1)
    assert policy.allows("search")
    assert not policy.allows("delete")


def test_packaged_data_matches_config_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    packaged = root / "emotion_kit" / "data"
    for name in ("orchestration.yaml", "rules.yaml", "memories.yaml"):
        assert (root / "config" / name).read_text(encoding="utf-8") == (
            packaged / name
        ).read_text(encoding="utf-8")
