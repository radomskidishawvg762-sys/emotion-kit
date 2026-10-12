from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_mock_demo_completes() -> None:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("EMOTION_KIT_MODE", None)
    env.pop("EMOTION_KIT_CONFIG_DIR", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "cli_demo.py")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    output = completed.stdout
    assert "adapter: mock" in output
    assert "第 5 轮" in output
    assert "anxious" in output
    assert "frustrated" in output
    assert "excited" in output
    assert "calm" in output
    assert "below_min_confidence" in output
    assert "mem-release-checklist" in output
    assert "模拟回复" in output
    assert "OPENAI_API_KEY" not in completed.stderr


def test_interactive_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.cli_demo import _interactive
    from emotion_kit import build_orchestrator

    def _raise(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise)
    orchestrator = build_orchestrator(mode="rule", force_mock=True)
    assert _interactive(orchestrator, verbose=False) == 0
