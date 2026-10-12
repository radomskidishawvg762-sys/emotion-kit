"""Scripted or interactive demo of the emotion orchestration pipeline.

Default mode runs five turns on the mock adapter when ``OPENAI_API_KEY`` is
unset, then exits. Labels printed here are strategy switches, not a reading
of subjective emotion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion_kit import TurnResult, build_orchestrator
from emotion_kit.adapters import LLMError
from emotion_kit.pipeline import Orchestrator

SCRIPTED_TURNS: tuple[str, ...] = (
    "我有点焦虑，明天发布会来不及了，怎么办？",
    "嗯，也许吧。",
    "试了三次都失败，真的很受挫，搞不定。",
    "等等，通过了！太棒了，终于可以演示，好兴奋！",
    "好，我平静下来了，按清单收尾就行。",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="情绪编排中间层演示。状态标签是策略开关，不是主观情绪测量。",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="强制使用 Mock 适配器，即使设置了 OPENAI_API_KEY",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="从标准输入读对话；默认跑一段脚本对话后退出",
    )
    parser.add_argument("--mode", choices=("rule", "llm", "auto"), default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置目录，内含 orchestration、rules、memories",
    )
    parser.add_argument("--verbose", action="store_true", help="打印思考草稿和系统提示")
    args = parser.parse_args(argv)

    try:
        orchestrator = build_orchestrator(
            config_dir=args.config,
            mode=args.mode,
            force_mock=args.mock,
        )
    except (OSError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(_banner(orchestrator, forced_mock=args.mock))
    print()
    try:
        if args.interactive:
            return _interactive(orchestrator, verbose=args.verbose)
        _scripted(orchestrator, verbose=args.verbose)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _banner(orchestrator: Orchestrator, *, forced_mock: bool) -> str:
    if orchestrator.adapter_name == "mock":
        if forced_mock:
            adapter_line = "adapter: mock（已强制使用 Mock 适配器，不访问网络）"
        else:
            adapter_line = "adapter: mock（未设置 OPENAI_API_KEY，使用 Mock 跑完整条流水线）"
    else:
        adapter_line = f"adapter: {orchestrator.adapter_name}"
        if orchestrator.base_url:
            adapter_line += f"  base_url: {orchestrator.base_url}"
    mode = orchestrator.detector.mode
    return "\n".join(
        [
            "情绪编排中间层 demo",
            "标签只选择回复策略、思考深度和工具权限，不表示测到了真实情绪。",
            adapter_line,
            f"mode: {mode}",
            f"main_model: {orchestrator.main_model}",
            f"detect_model: {orchestrator.detect_model}",
        ]
    )


def _scripted(orchestrator: Orchestrator, *, verbose: bool) -> None:
    for index, text in enumerate(SCRIPTED_TURNS, start=1):
        result = orchestrator.turn(text)
        print(format_turn(index, result, verbose=verbose))
        print()


def _interactive(orchestrator: Orchestrator, *, verbose: bool) -> int:
    print("输入一句对话，/quit 结束。")
    index = 1
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0
        if not text:
            continue
        if text in {"/quit", "/exit", "quit", "exit"}:
            return 0
        result = orchestrator.turn(text)
        print(format_turn(index, result, verbose=verbose))
        print()
        index += 1


def format_turn(index: int, result: TurnResult, *, verbose: bool) -> str:
    detection = result.detection
    state = result.state
    evidence = "、".join(detection.evidence) if detection.evidence else "—"
    route = f"{state.previous} -> {state.emotion}" if state.switched else state.emotion
    lines = [
        f"=== 第 {index} 轮 ===",
        f"user: {result.user_message}",
        (
            f"detect: {detection.emotion} ({detection.confidence:.2f}, {detection.source})"
            f" evidence: {evidence}"
        ),
        f"state: {route}  reason={state.reason}  turns={state.turns_in_state}",
        (
            f"strategy: {result.strategy.emotion} / {result.strategy.label_zh}"
            f"  thinking_depth={result.strategy.thinking_depth}"
            f"  max_tool_calls={result.strategy.tool_policy.max_tool_calls}"
        ),
        (
            f"tools: allow={list(result.strategy.tool_policy.allow)}"
            f" deny={list(result.strategy.tool_policy.deny)}"
        ),
    ]
    if result.memories:
        lines.append("memories:")
        for item in result.memories:
            lines.append(f"  - {item.memory.id} ({item.score:.2f}) {item.memory.text}")
    else:
        lines.append("memories: —")
    if verbose and result.deliberation_notes:
        lines.append("notes:")
        for note in result.deliberation_notes:
            lines.append(f"  - {note}")
    lines.append(f"reply: {result.reply}")
    if verbose and result.messages:
        lines.append("system:")
        lines.append(str(result.messages[0].get("content", "")))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
