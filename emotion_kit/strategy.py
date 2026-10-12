"""Load strategy packs and related orchestration settings.

Shipped defaults live in ``config/``. Pass another directory, or a partial
mapping, to override them. JSON is accepted anywhere YAML is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from emotion_kit.memory import RecallWeights
from emotion_kit.state import EMOTIONS, DebounceConfig, Emotion, parse_emotion

_MODES = ("rule", "llm", "auto")
_STATE_KEYS = frozenset(
    {"label_zh", "description", "system_prompt_delta", "thinking_depth", "tool_policy"}
)
_TOOL_POLICY_KEYS = frozenset({"allow", "deny", "max_tool_calls"})


@dataclass(frozen=True)
class ToolPolicy:
    """Host-enforced tool gate. The prompt repeats it; the host should too.

    An empty ``allow`` list means every tool is allowed except those in ``deny``.
    """

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()
    max_tool_calls: int = 3

    def __post_init__(self) -> None:
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be >= 0")

    def allows(self, name: str) -> bool:
        if name in self.deny:
            return False
        if self.allow and name not in self.allow:
            return False
        return True


@dataclass(frozen=True)
class Strategy:
    emotion: Emotion
    label_zh: str
    description: str
    system_prompt_delta: str
    thinking_depth: int
    tool_policy: ToolPolicy

    def __post_init__(self) -> None:
        if self.thinking_depth < 1:
            raise ValueError(f"{self.emotion} thinking_depth must be >= 1")
        if not self.system_prompt_delta.strip():
            raise ValueError(f"{self.emotion} system_prompt_delta is required")

    def render_for_prompt(self) -> str:
        allow = ", ".join(self.tool_policy.allow) if self.tool_policy.allow else "(all except deny)"
        deny = ", ".join(self.tool_policy.deny) if self.tool_policy.deny else "(none)"
        return (
            f"{self.system_prompt_delta.strip()}\n\n"
            f"思考深度 thinking_depth={self.thinking_depth}。\n"
            f"工具策略 tool_policy: allow=[{allow}]; deny=[{deny}]; "
            f"max_tool_calls={self.tool_policy.max_tool_calls}。\n"
            "不要请求被拒绝的工具。达到 max_tool_calls 后停止工具调用。"
        )


@dataclass(frozen=True)
class DetectorSettings:
    mode: str
    rule_accept_confidence: float
    default_emotion: Emotion
    default_confidence: float

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"detector.mode must be one of {', '.join(_MODES)}")
        if not 0.0 <= self.rule_accept_confidence <= 1.0:
            raise ValueError("rule_accept_confidence must be in [0, 1]")
        if not 0.0 <= self.default_confidence <= 1.0:
            raise ValueError("default_confidence must be in [0, 1]")


@dataclass(frozen=True)
class OrchestrationConfig:
    base_system_prompt: str
    debounce: DebounceConfig
    detector: DetectorSettings
    top_k: int
    weights: Mapping[Emotion, RecallWeights]
    states: Mapping[Emotion, Strategy]


def resolve_config_dir(explicit: str | Path | None = None) -> Path:
    """Find a directory that contains ``orchestration.yaml`` or ``.json``."""

    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"config directory not found: {path}")
        find_config_file(path, "orchestration")
        return path

    env = os.environ.get("EMOTION_KIT_CONFIG_DIR")
    if env:
        path = Path(env).expanduser()
        if not path.is_dir() or not _has_stem(path, "orchestration"):
            raise FileNotFoundError(f"EMOTION_KIT_CONFIG_DIR is not a config directory: {path}")
        return path

    cwd = Path.cwd() / "config"
    if cwd.is_dir() and _has_stem(cwd, "orchestration"):
        return cwd

    repo = Path(__file__).resolve().parents[1] / "config"
    if repo.is_dir() and _has_stem(repo, "orchestration"):
        return repo

    packaged = Path(__file__).resolve().parent / "data"
    if packaged.is_dir() and _has_stem(packaged, "orchestration"):
        return packaged

    raise FileNotFoundError(
        "orchestration config not found. Set EMOTION_KIT_CONFIG_DIR or pass a config directory."
    )


def find_config_file(directory: Path, stem: str) -> Path:
    for name in (f"{stem}.yaml", f"{stem}.yml", f"{stem}.json"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {stem}.yaml or {stem}.json in {directory}")


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def load_orchestration(
    directory: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> OrchestrationConfig:
    path = find_config_file(resolve_config_dir(directory), "orchestration")
    raw = load_mapping(path)
    if overrides:
        raw = deep_merge(raw, overrides)
    return orchestration_from_dict(raw)


def orchestration_from_dict(raw: Mapping[str, Any]) -> OrchestrationConfig:
    base = raw.get("base_system_prompt")
    if not isinstance(base, str) or not base.strip():
        raise ValueError("base_system_prompt is required")

    debounce_raw = _mapping(raw.get("debounce") or {}, "debounce")
    debounce = DebounceConfig(
        min_confidence=float(debounce_raw.get("min_confidence", 0.45)),
        switch_confidence=float(debounce_raw.get("switch_confidence", 0.62)),
        immediate_confidence=float(debounce_raw.get("immediate_confidence", 0.90)),
        consecutive_turns=int(debounce_raw.get("consecutive_turns", 2)),
    )

    detector_raw = _mapping(raw.get("detector") or {}, "detector")
    default_emotion = parse_emotion(detector_raw.get("default_emotion", "calm"))
    detector = DetectorSettings(
        mode=str(detector_raw.get("mode", "auto")),
        rule_accept_confidence=float(detector_raw.get("rule_accept_confidence", 0.72)),
        default_emotion=default_emotion,
        default_confidence=float(detector_raw.get("default_confidence", 0.34)),
    )

    memory_raw = _mapping(raw.get("memory") or {}, "memory")
    top_k = int(memory_raw.get("top_k", 3))
    if top_k < 0:
        raise ValueError("memory.top_k must be >= 0")
    weights = _parse_weights(_mapping(memory_raw.get("weights") or {}, "memory.weights"))
    states = _parse_states(_mapping(raw.get("states") or {}, "states"))
    return OrchestrationConfig(
        base_system_prompt=base.strip(),
        debounce=debounce,
        detector=detector,
        top_k=top_k,
        weights=weights,
        states=states,
    )


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _parse_states(raw: Mapping[str, Any]) -> dict[Emotion, Strategy]:
    states: dict[Emotion, Strategy] = {}
    for name in EMOTIONS:
        if name not in raw:
            raise ValueError(f"strategy pack missing state {name!r}")
        item = _mapping(raw[name], f"states.{name}")
        _reject_unknown(item, _STATE_KEYS, f"states.{name}")
        policy_raw = _mapping(item.get("tool_policy") or {}, f"states.{name}.tool_policy")
        _reject_unknown(policy_raw, _TOOL_POLICY_KEYS, f"states.{name}.tool_policy")
        policy = ToolPolicy(
            allow=_str_tuple(policy_raw.get("allow") or ()),
            deny=_str_tuple(policy_raw.get("deny") or ()),
            max_tool_calls=int(policy_raw.get("max_tool_calls", 3)),
        )
        states[parse_emotion(name)] = Strategy(
            emotion=parse_emotion(name),
            label_zh=str(item.get("label_zh") or name),
            description=str(item.get("description") or ""),
            system_prompt_delta=str(item.get("system_prompt_delta") or ""),
            thinking_depth=int(item.get("thinking_depth", 1)),
            tool_policy=policy,
        )
    extras = sorted(set(raw) - set(EMOTIONS))
    if extras:
        raise ValueError(
            "unknown strategy states "
            f"{', '.join(extras)}; v1 supports {', '.join(EMOTIONS)}"
        )
    return states


def _parse_weights(raw: Mapping[str, Any]) -> dict[Emotion, RecallWeights]:
    weights: dict[Emotion, RecallWeights] = {}
    for name in EMOTIONS:
        if name not in raw:
            raise ValueError(f"memory.weights missing state {name!r}")
        item = _mapping(raw[name], f"memory.weights.{name}")
        weights[parse_emotion(name)] = RecallWeights(
            tag_overlap=float(item.get("tag_overlap", 0.0)),
            affinity=float(item.get("affinity", 0.0)),
            recency=float(item.get("recency", 0.0)),
        )
    return weights


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"{label} has unknown fields: {', '.join(extra)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _has_stem(directory: Path, stem: str) -> bool:
    return any((directory / f"{stem}{suffix}").is_file() for suffix in (".yaml", ".yml", ".json"))
