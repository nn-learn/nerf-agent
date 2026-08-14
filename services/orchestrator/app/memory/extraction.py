import hashlib
import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.memory.models import (
    MemoryAspect,
    MemoryCandidate,
    MemoryKind,
    MemoryMessage,
    MemoryWindow,
    MessageRole,
)

_PROMPT_INJECTION_RE = re.compile(
    r"忽略.{0,8}(指令|规则|系统)|system\s*prompt|调用.{0,6}工具|执行.{0,6}命令",
    re.IGNORECASE,
)
_SAFETY_RE = re.compile(r"自杀|自残|伤害自己|结束生命|不想活|活不下去")
_DIRECT_IDENTIFIER_RE = re.compile(
    r"身份证|住址|家庭地址|1[3-9]\d{9}|\d{17}[\dXx]"
)
_SENSITIVE_RE = re.compile(
    r"诊断|确诊|抑郁症|焦虑症|双相|精神分裂|服药|药物|身份证|住址|家庭地址|"
    r"1[3-9]\d{9}|\d{17}[\dXx]"
)
_EXPLICIT_MEMORY_RE = re.compile(
    r"(?:请|帮我)?记住[：,:，\s]*(?P<claim>.+)|以后(?:请)?记得[：,:，\s]*(?P<claim2>.+)"
)
_BOUNDARY_RE = re.compile(r"我(?:不喜欢|不想|不要|希望不要|讨厌)(?P<claim>.+)")
_PREFERENCE_RE = re.compile(
    r"我(?:现在)?(?:更喜欢|比较喜欢|喜欢|偏好|希望)(?P<claim>.+)"
)
_GOAL_RE = re.compile(r"我(?:现在)?(?:的目标是|计划|打算|想要)(?P<claim>.+)")
_COPING_RE = re.compile(
    r"(?P<claim>.+?)(?:对我有帮助|对我有效|能让我平静|能缓解我的焦虑)"
)


class MemoryExtractor(Protocol):
    def extract_batch(
        self,
        windows: Sequence[MemoryWindow],
    ) -> list[list[MemoryCandidate]]:
        raise NotImplementedError


class ExtractionTelemetry(BaseModel):
    request_count: int = Field(default=0, ge=0)
    cache_hit_count: int = Field(default=0, ge=0)
    prompt_eval_count: int = Field(default=0, ge=0)
    eval_count: int = Field(default=0, ge=0)
    total_duration_ns: int = Field(default=0, ge=0)
    load_duration_ns: int = Field(default=0, ge=0)
    invalid_candidate_count: int = Field(default=0, ge=0)
    fallback_candidate_count: int = Field(default=0, ge=0)

    def delta(self, previous: "ExtractionTelemetry") -> "ExtractionTelemetry":
        return ExtractionTelemetry(
            **{
                name: max(0, int(getattr(self, name)) - int(getattr(previous, name)))
                for name in type(self).model_fields
            }
        )


@runtime_checkable
class InstrumentedMemoryExtractor(Protocol):
    def telemetry(self) -> ExtractionTelemetry:
        raise NotImplementedError


def memory_kind_for_text(text: str) -> MemoryKind:
    return MemoryKind.SAFETY if _SAFETY_RE.search(text) else MemoryKind.SEMANTIC


def contains_sensitive_memory_content(text: str) -> bool:
    return bool(_SENSITIVE_RE.search(text))


def memory_integrity_flags(text: str) -> list[str]:
    flags: list[str] = []
    if _PROMPT_INJECTION_RE.search(text):
        flags.append("prompt_injection")
    if _DIRECT_IDENTIFIER_RE.search(text):
        flags.append("direct_identifier")
    return flags


def infer_memory_aspect(claim: str) -> MemoryAspect:
    if re.search(r"不喜欢|不要|避免|别再", claim):
        return MemoryAspect.BOUNDARY
    if re.search(r"喜欢|偏好|希望|简短|详细", claim):
        return MemoryAspect.PREFERENCE
    if re.search(r"目标|计划|完成|坚持", claim):
        return MemoryAspect.GOAL
    if re.search(r"有帮助|有效|缓解|平静|呼吸|数数", claim):
        return MemoryAspect.COPING_STRATEGY
    return MemoryAspect.FACT


def normalized_memory_text(aspect: MemoryAspect, claim: str) -> str:
    cleaned = claim.strip("。！？!?，,：: ")
    prefixes = {
        MemoryAspect.FACT: "用户明确表示：",
        MemoryAspect.PREFERENCE: "用户偏好：",
        MemoryAspect.GOAL: "用户目标：",
        MemoryAspect.COPING_STRATEGY: "对用户有效的支持方式：",
        MemoryAspect.BOUNDARY: "用户不希望：",
    }
    return f"{prefixes[aspect]}{cleaned}"


def memory_subject_key(aspect: MemoryAspect, claim: str) -> str:
    shared_topics = (
        (r"简短|详细|回复|回答", "communication.response_style"),
        (r"称呼|叫我", "communication.form_of_address"),
        (r"语气|温柔|直接", "communication.tone"),
        (r"呼吸", "coping.breathing"),
        (r"数数", "coping.counting"),
        (r"冥想", "coping.meditation"),
        (r"睡眠|入睡|早睡", "wellbeing.sleep"),
    )
    for pattern, key in shared_topics:
        if re.search(pattern, claim):
            return key
    digest = hashlib.sha256(claim.encode("utf-8")).hexdigest()[:12]
    return f"{aspect.value.lower()}.{digest}"


class RuleBasedMemoryExtractor:
    """A conservative no-model baseline; later extractors must preserve this contract."""

    def extract_batch(
        self,
        windows: Sequence[MemoryWindow],
    ) -> list[list[MemoryCandidate]]:
        return [self.extract(window) for window in windows]

    def extract(self, window: MemoryWindow) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for message in window.messages:
            if message.role is not MessageRole.USER:
                continue
            candidate = self._extract_message(window, message)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _extract_message(
        self,
        window: MemoryWindow,
        message: MemoryMessage,
    ) -> MemoryCandidate | None:
        text = " ".join(message.text.split()).strip("。！？!?，, ")
        if not text:
            return None

        aspect: MemoryAspect
        confidence: float
        claim = text
        kind = MemoryKind.SEMANTIC

        if _SAFETY_RE.search(text):
            aspect = MemoryAspect.FACT
            confidence = 1.0
            kind = MemoryKind.SAFETY
        elif match := _EXPLICIT_MEMORY_RE.search(text):
            claim = (match.group("claim") or match.group("claim2")).strip()
            aspect = infer_memory_aspect(claim)
            confidence = 0.98
        elif match := _BOUNDARY_RE.search(text):
            claim = match.group("claim").strip()
            aspect = MemoryAspect.BOUNDARY
            confidence = 0.94
        elif match := _PREFERENCE_RE.search(text):
            claim = match.group("claim").strip()
            aspect = MemoryAspect.PREFERENCE
            confidence = 0.92
        elif match := _GOAL_RE.search(text):
            claim = match.group("claim").strip()
            aspect = MemoryAspect.GOAL
            confidence = 0.9
        elif match := _COPING_RE.search(text):
            claim = self._resolve_coping_reference(
                match.group("claim").strip(),
                window,
            )
            aspect = MemoryAspect.COPING_STRATEGY
            confidence = 0.86
        else:
            return None

        normalized = normalized_memory_text(aspect, claim)
        flags = memory_integrity_flags(text)
        return MemoryCandidate(
            source="transcript",
            contains_sensitive_content=contains_sensitive_memory_content(text),
            text=normalized,
            kind=kind,
            source_turn_id=message.turn_id,
            aspect=aspect,
            subject_key=memory_subject_key(aspect, claim),
            confidence=confidence,
            source_message_ids=[message.message_id],
            source_window_id=window.window_id,
            valid_from_ms=message.timestamp_ms,
            integrity_flags=flags,
        )

    @staticmethod
    def _infer_aspect(claim: str) -> MemoryAspect:
        return infer_memory_aspect(claim)

    @staticmethod
    def _normalized_text(aspect: MemoryAspect, claim: str) -> str:
        return normalized_memory_text(aspect, claim)

    @staticmethod
    def _subject_key(aspect: MemoryAspect, claim: str) -> str:
        return memory_subject_key(aspect, claim)

    @staticmethod
    def _resolve_coping_reference(claim: str, window: MemoryWindow) -> str:
        if claim not in {"这", "这个", "这种方法", "刚才这个方法"}:
            return claim
        for context in reversed([*window.context_messages, *window.messages]):
            for keyword in ("呼吸练习", "数数练习", "正念", "冥想", "散步"):
                if keyword in context.text:
                    return keyword
        return "当前讨论的方法"
