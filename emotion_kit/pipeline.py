"""Five-step orchestration in front of an OpenAI-compatible chat model.

1. Detect a strategy label.
2. Update the sticky state, with a confidence debounce.
3. Select that state's strategy pack.
4. Recall memories with the state's weight table.
5. Call the main model with the adapted system prompt, thinking depth, and tool policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from emotion_kit.adapters import LLMClient, MockClient, OpenAICompatibleClient
from emotion_kit.detector import Detector, load_rules
from emotion_kit.memory import MemoryStore, ScoredMemory, load_memories
from emotion_kit.state import Detection, Emotion, StateSnapshot, StickyState
from emotion_kit.strategy import OrchestrationConfig, Strategy, load_orchestration

ChatDict = dict[str, str]


@dataclass(frozen=True)
class TurnResult:
    user_message: str
    reply: str
    detection: Detection
    state: StateSnapshot
    strategy: Strategy
    memories: tuple[ScoredMemory, ...]
    deliberation_notes: tuple[str, ...]
    messages: tuple[Mapping[str, str], ...]
    model: str
    usage: Mapping[str, Any] | None = None


class Orchestrator:
    """Run one user message through the middleware and return the model reply."""

    def __init__(
        self,
        *,
        detector: Detector,
        state: StickyState,
        strategies: Mapping[Emotion, Strategy],
        memory: MemoryStore,
        client: LLMClient,
        base_system_prompt: str,
        main_model: str,
        adapter_name: str = "custom",
        base_url: str | None = None,
        detect_model: str | None = None,
        history_limit: int = 8,
        temperature: float = 0.4,
        note_max_tokens: int = 240,
        reply_max_tokens: int = 800,
    ) -> None:
        missing = [name for name in ("calm", "anxious", "frustrated", "excited") if name not in strategies]
        if missing:
            raise ValueError(f"strategies missing: {', '.join(missing)}")
        if history_limit < 0:
            raise ValueError("history_limit must be >= 0")
        self.detector = detector
        self.state = state
        self.strategies = dict(strategies)
        self.memory = memory
        self.client = client
        self.base_system_prompt = base_system_prompt.strip()
        self.main_model = main_model
        self.detect_model = detect_model or main_model
        self.adapter_name = adapter_name
        self.base_url = base_url
        self.history_limit = history_limit
        self.temperature = temperature
        self.note_max_tokens = note_max_tokens
        self.reply_max_tokens = reply_max_tokens
        self.history: list[ChatDict] = []

    def turn(
        self,
        user_message: str,
        *,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> TurnResult:
        """Orchestrate one turn.

        When ``history`` is omitted, the orchestrator uses and extends its own
        transcript. A passed history is read-only context for this call.
        """

        text = user_message.strip()
        if not text:
            raise ValueError("user_message is empty")
        if history is None:
            active = self.history
            store_history = True
        else:
            active = [_as_chat(message) for message in history]
            store_history = False

        detection = self.detector.detect(text, active)
        # State and history stay put until every main-model call returns.
        snapshot = self.state.propose(detection)
        strategy = self.strategies[snapshot.emotion]
        recalled = tuple(self.memory.recall(snapshot.emotion, text))
        notes: list[str] = []
        depth = strategy.thinking_depth
        if depth > 1:
            for round_index in range(1, depth):
                draft_messages = self._build_messages(
                    strategy,
                    recalled,
                    active,
                    text,
                    depth=depth,
                    notes=notes,
                    deliberation_round=round_index,
                )
                draft = self.client.complete(
                    draft_messages,
                    model=self.main_model,
                    temperature=self.temperature,
                    max_tokens=self.note_max_tokens,
                )
                notes.append(draft.content.strip())

        final_messages = self._build_messages(
            strategy,
            recalled,
            active,
            text,
            depth=depth,
            notes=notes,
            deliberation_round=None,
        )
        final = self.client.complete(
            final_messages,
            model=self.main_model,
            temperature=self.temperature,
            max_tokens=self.reply_max_tokens,
        )
        result = TurnResult(
            user_message=text,
            reply=final.content.strip(),
            detection=detection,
            state=snapshot,
            strategy=strategy,
            memories=recalled,
            deliberation_notes=tuple(notes),
            messages=tuple(final_messages),
            model=final.model,
            usage=final.usage,
        )
        self.state.commit(snapshot)
        if store_history:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": result.reply})
        return result

    def reset(self) -> None:
        self.history.clear()
        self.state.reset()

    def _build_messages(
        self,
        strategy: Strategy,
        memories: Sequence[ScoredMemory],
        history: Sequence[Mapping[str, str]],
        user_message: str,
        *,
        depth: int,
        notes: Sequence[str],
        deliberation_round: int | None,
    ) -> list[ChatDict]:
        messages: list[ChatDict] = [
            {
                "role": "system",
                "content": self._system_prompt(
                    strategy,
                    memories,
                    depth=depth,
                    notes=notes,
                    deliberation_round=deliberation_round,
                ),
            }
        ]
        if self.history_limit:
            for item in list(history)[-self.history_limit :]:
                messages.append(_as_chat(item))
        messages.append({"role": "user", "content": user_message})
        return messages

    def _system_prompt(
        self,
        strategy: Strategy,
        memories: Sequence[ScoredMemory],
        *,
        depth: int,
        notes: Sequence[str],
        deliberation_round: int | None,
    ) -> str:
        parts = [
            self.base_system_prompt,
            "",
            strategy.render_for_prompt(),
            "",
            _memory_block(memories),
        ]
        if deliberation_round is not None:
            if notes:
                parts.append(
                    "\n此前的内部草稿（吸收进这一轮思考，不要原样输出给用户）：\n"
                    + _render_notes(notes)
                )
            parts.append(
                "\n这是内部思考草稿第 "
                f"{deliberation_round}/{depth} 轮，不是给用户的最终回复。"
                "只用一两句话写下下一步该怎么组织回答。不要问候用户。"
            )
        elif notes:
            parts.append(
                "\n思考深度为 "
                f"{depth}。以下是内部草稿，吸收进最终回复，不要原样输出草稿标题：\n"
                + _render_notes(notes)
            )
            parts.append("用最终回复面对用户。不要声称你测量到了对方的真实情绪。")
        else:
            parts.append(f"\n思考深度为 {depth}。直接给出最终回复。")
            parts.append("不要声称你测量到了对方的真实情绪。")
        return "\n".join(parts)


def build_orchestrator(
    *,
    config_dir: str | Path | None = None,
    mode: str | None = None,
    force_mock: bool = False,
    client: LLMClient | None = None,
    detect_client: LLMClient | None = None,
    overrides: Mapping[str, Any] | None = None,
    main_model: str | None = None,
    detect_model: str | None = None,
) -> Orchestrator:
    """Build an orchestrator from config, environment, and an optional client.

    Without ``OPENAI_API_KEY`` (or with ``force_mock=True``) the main and
    detector clients are :class:`MockClient`. Detection and chat may still use
    different model names when a real client is configured.
    """

    config = load_orchestration(config_dir, overrides=overrides)
    chosen_mode = mode or _env("EMOTION_KIT_MODE")
    if chosen_mode:
        config = replace(config, detector=replace(config.detector, mode=chosen_mode))

    env_main = main_model or _env("EMOTION_KIT_MAIN_MODEL") or _env("OPENAI_MODEL")
    explicit_detect = detect_model or _env("EMOTION_KIT_DETECT_MODEL")
    api_key = _env("OPENAI_API_KEY")
    base_url = _env("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    adapter_name = "custom"
    resolved_base: str | None = None
    if client is None:
        resolved_main = env_main or "gpt-4o-mini"
        if force_mock or not api_key:
            client = MockClient()
            adapter_name = "mock"
        else:
            client = OpenAICompatibleClient(
                base_url=base_url,
                api_key=api_key,
                model=resolved_main,
            )
            adapter_name = "openai-compatible"
            resolved_base = base_url
    elif isinstance(client, MockClient):
        adapter_name = "mock"
        resolved_main = env_main or "gpt-4o-mini"
    elif isinstance(client, OpenAICompatibleClient):
        adapter_name = "openai-compatible"
        resolved_base = client.base_url
        resolved_main = env_main or client.model
    else:
        resolved_main = env_main or "gpt-4o-mini"
    if explicit_detect:
        resolved_detect = explicit_detect
    else:
        resolved_detect = _client_model(detect_client) or resolved_main

    detector_llm = detect_client if detect_client is not None else client
    directory = config_dir
    detector = Detector(
        load_rules(directory),
        settings=config.detector,
        llm=detector_llm,
        model=resolved_detect,
    )
    memory = MemoryStore(
        entries=load_memories(directory),
        weights=config.weights,
        top_k=config.top_k,
    )
    return Orchestrator(
        detector=detector,
        state=StickyState(config.debounce),
        strategies=config.states,
        memory=memory,
        client=client,
        base_system_prompt=config.base_system_prompt,
        main_model=resolved_main,
        detect_model=resolved_detect,
        adapter_name=adapter_name,
        base_url=resolved_base,
    )


def _render_notes(notes: Sequence[str]) -> str:
    return "\n".join(f"{index}. {note}" for index, note in enumerate(notes, start=1))


def _memory_block(memories: Sequence[ScoredMemory]) -> str:
    if not memories:
        return "召回记忆：无。"
    lines = ["召回记忆（按当前状态加权，只作背景，不要编造未列出的事实）："]
    for item in memories:
        lines.append(f"- [{item.memory.id}] (score={item.score:.2f}) {item.memory.text}")
    return "\n".join(lines)


def _as_chat(message: Mapping[str, str]) -> ChatDict:
    return {
        "role": str(message.get("role", "user")),
        "content": str(message.get("content", "")),
    }


def _client_model(client: LLMClient | None) -> str | None:
    if client is None:
        return None
    model = getattr(client, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


# Re-exported for callers that want the loaded config type alongside the builder.
__all__ = ["Orchestrator", "TurnResult", "build_orchestrator", "OrchestrationConfig"]
