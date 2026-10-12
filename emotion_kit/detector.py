"""Rule, LLM, and auto strategy-label detectors.

The label is a routing key for reply strategy. It is not a reading of how a
person feels.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from emotion_kit.adapters import CLASSIFY_MARKER, LLMClient, LLMError
from emotion_kit.state import EMOTIONS, Detection, Emotion, parse_emotion
from emotion_kit.strategy import DetectorSettings, find_config_file, load_mapping, resolve_config_dir

# Smooths the rule score so a single clear category is high and a tie is not.
RULE_SMOOTHING = 0.08

_CLASSIFY_SYSTEM = f"""{CLASSIFY_MARKER}
Classify the latest user message into exactly one behavioral strategy label.
This is not a diagnosis of subjective emotion and not a measurement of feeling.
Labels: calm, anxious, frustrated, excited.
Return a single JSON object and nothing else:
{{"emotion": "calm|anxious|frustrated|excited", "confidence": 0.0, "evidence": ["short span from the text"]}}
confidence is how clear the strategy label is, from 0 to 1.
"""


@dataclass(frozen=True)
class RulePattern:
    emotion: Emotion
    pattern: re.Pattern[str]
    raw: str
    weight: float


class Detector:
    """Classify the latest user message.

    Modes
    -----
    ``rule``
        Keyword and regex scores only.
    ``llm``
        One JSON classification call. Requires a client.
    ``auto``
        Use the rule result when it is confident. Otherwise call the LLM if a
        client is configured, and fall back to the rule result if that call fails.
        With no client, ``auto`` is the rule detector.
    """

    def __init__(
        self,
        rules: Sequence[RulePattern],
        *,
        settings: DetectorSettings | None = None,
        llm: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        self.rules = tuple(rules)
        self.settings = settings or DetectorSettings(
            mode="auto",
            rule_accept_confidence=0.72,
            default_emotion="calm",
            default_confidence=0.34,
        )
        self.llm = llm
        self.model = model

    @property
    def mode(self) -> str:
        return self.settings.mode

    def detect(
        self,
        text: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> Detection:
        cleaned = text.strip()
        if self.mode == "rule":
            return self.detect_rule(cleaned)
        if self.mode == "llm":
            return self.detect_llm(cleaned, history)
        rule = self.detect_rule(cleaned)
        if self.llm is None:
            return rule
        if rule.evidence and rule.confidence >= self.settings.rule_accept_confidence:
            return rule
        try:
            return self.detect_llm(cleaned, history)
        except (LLMError, ValueError, KeyError, TypeError):
            return rule

    def detect_rule(self, text: str) -> Detection:
        scores = {parse_emotion(name): 0.0 for name in EMOTIONS}
        evidence: dict[Emotion, list[str]] = {parse_emotion(name): [] for name in EMOTIONS}
        for rule in self.rules:
            matches = list(rule.pattern.finditer(text))
            if not matches:
                continue
            scores[rule.emotion] += rule.weight
            for match in matches:
                span = match.group(0).strip()
                if span and span not in evidence[rule.emotion]:
                    evidence[rule.emotion].append(span)
        best = max(scores, key=lambda name: (scores[name], name))
        best_score = scores[best]
        if best_score <= 0:
            return Detection(
                emotion=self.settings.default_emotion,
                confidence=self.settings.default_confidence,
                evidence=(),
                source="rule",
            )
        second = max(score for name, score in scores.items() if name != best)
        confidence = best_score / (best_score + second + RULE_SMOOTHING)
        confidence = min(0.99, max(0.0, confidence))
        return Detection(
            emotion=best,
            confidence=confidence,
            evidence=tuple(evidence[best][:6]),
            source="rule",
        )

    def detect_llm(
        self,
        text: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> Detection:
        if self.llm is None:
            raise RuntimeError("llm detection requires a client")
        transcript = _format_history(history)
        messages = [
            {"role": "system", "content": _CLASSIFY_SYSTEM},
            {
                "role": "user",
                "content": f"Recent transcript:\n{transcript}\n\nLatest user message:\n{text}",
            },
        ]
        result = self.llm.complete(
            messages,
            model=self.model,
            temperature=0.0,
            max_tokens=300,
        )
        return detection_from_json(result.content)


def detection_from_json(content: str) -> Detection:
    payload = _extract_json(content)
    emotion = parse_emotion(payload.get("emotion"))
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("classifier confidence is not a number") from exc
    confidence = min(1.0, max(0.0, confidence))
    raw_evidence = payload.get("evidence") or ()
    if isinstance(raw_evidence, str):
        evidence = (raw_evidence,)
    elif isinstance(raw_evidence, Sequence):
        evidence = tuple(str(item) for item in raw_evidence if str(item).strip())
    else:
        raise ValueError("classifier evidence must be a list or string")
    return Detection(emotion=emotion, confidence=confidence, evidence=evidence[:6], source="llm")


def load_rules(directory: str | Path | None = None) -> tuple[RulePattern, ...]:
    path = find_config_file(resolve_config_dir(directory), "rules")
    return rules_from_dict(load_mapping(path))


def rules_from_dict(raw: Mapping[str, Any]) -> tuple[RulePattern, ...]:
    emotions = raw.get("emotions")
    if not isinstance(emotions, Mapping):
        raise ValueError("rules.emotions must be a mapping")
    compiled: list[RulePattern] = []
    for name in EMOTIONS:
        if name not in emotions:
            raise ValueError(f"rules missing emotion {name!r}")
        entries = emotions[name]
        if not isinstance(entries, Sequence) or isinstance(entries, str):
            raise ValueError(f"rules.{name} must be a list")
        for item in entries:
            if isinstance(item, str):
                pattern, weight = item, 1.0
            elif isinstance(item, Mapping):
                pattern = str(item.get("pattern") or "")
                weight = float(item.get("weight", 1.0))
            else:
                raise ValueError(f"rules.{name} entries must be strings or mappings")
            if not pattern:
                raise ValueError(f"rules.{name} pattern is empty")
            if weight < 0:
                raise ValueError(f"rules.{name} weight must be >= 0")
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid rule pattern for {name}: {pattern!r}") from exc
            compiled.append(
                RulePattern(
                    emotion=parse_emotion(name),
                    pattern=regex,
                    raw=pattern,
                    weight=weight,
                )
            )
    return tuple(compiled)


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("classifier did not return JSON") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("classifier JSON could not be parsed") from exc
    if not isinstance(payload, dict):
        raise ValueError("classifier JSON must be an object")
    return payload


def _format_history(history: Sequence[Mapping[str, str]] | None, limit: int = 6) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for message in list(history)[-limit:]:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(none)"
