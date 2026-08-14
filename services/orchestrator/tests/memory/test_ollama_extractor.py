import json
from pathlib import Path

import httpx
import pytest

from app.memory.models import (
    MemoryAspect,
    MemoryKind,
    MemoryMessage,
    MemoryWindow,
    MessageRole,
)
from app.memory.ollama_extractor import (
    MEMORY_EXTRACTION_POLICY,
    OllamaMemoryExtractor,
    OllamaMemoryExtractorError,
)
from app.memory.pipeline import MemoryPipeline
from app.memory.repository import MemoryRepository


def message(
    message_id: str,
    role: MessageRole,
    text: str,
    *,
    sequence: int,
) -> MemoryMessage:
    return MemoryMessage(
        message_id=message_id,
        turn_id=f"turn_{sequence}",
        role=role,
        text=text,
        sequence=sequence,
        timestamp_ms=1_000 + sequence,
    )


def window() -> MemoryWindow:
    return MemoryWindow(
        window_id="window_000000",
        context_messages=[
            message("context_1", MessageRole.ASSISTANT, "我们刚才聊到回答方式。", sequence=0)
        ],
        messages=[
            message("user_1", MessageRole.USER, "我更喜欢简短回答。", sequence=1),
            message("assistant_1", MessageRole.ASSISTANT, "好的。", sequence=2),
        ],
        estimated_tokens=20,
        context_tokens=8,
    )


def ollama_payload(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "model": "qwen3.6:latest",
        "message": {
            "role": "assistant",
            "content": json.dumps({"candidates": candidates}, ensure_ascii=False),
        },
        "done": True,
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 300,
        "eval_count": 80,
    }


def valid_candidate() -> dict[str, object]:
    return {
        "window_id": "window_000000",
        "source_message_ids": ["user_1"],
        "evidence_quote": "简短回答",
        "aspect": "PREFERENCE",
        "claim": "用户更喜欢简短回答",
        "confidence": 0.96,
    }


def test_ollama_extractor_uses_structured_batch_and_grounded_source() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=ollama_payload([valid_candidate()]))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    candidate = extractor.extract_batch([window()])[0][0]

    assert captured["think"] is False
    assert captured["stream"] is False
    assert isinstance(captured["format"], dict)
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["content"] == MEMORY_EXTRACTION_POLICY
    request_context = json.loads(messages[1]["content"])
    assert request_context["windows"][0]["context_messages"][0]["context_only"] is True
    assert candidate.aspect is MemoryAspect.PREFERENCE
    assert candidate.kind is MemoryKind.SEMANTIC
    assert candidate.subject_key == "communication.response_style"
    assert candidate.source_message_ids == ["user_1"]
    assert candidate.text == "用户偏好：用户更喜欢简短回答"
    assert extractor.telemetry().prompt_eval_count == 300


def test_context_assistant_and_fabricated_quotes_cannot_become_sources() -> None:
    invalid_candidates = [
        {
            **valid_candidate(),
            "source_message_ids": ["context_1"],
        },
        {
            **valid_candidate(),
            "source_message_ids": ["assistant_1"],
        },
        {
            **valid_candidate(),
            "evidence_quote": "用户从未说过的内容",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=ollama_payload(invalid_candidates))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        use_rule_fallback=False,
    )

    assert extractor.extract_batch([window()]) == [[]]
    assert extractor.telemetry().invalid_candidate_count == 3


def test_identical_batch_hits_memory_only_cache_without_second_model_call() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        _ = request
        calls += 1
        return httpx.Response(200, json=ollama_payload([valid_candidate()]))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    first = extractor.extract_batch([window()])
    second = extractor.extract_batch([window()])

    assert first == second
    assert first is not second
    assert calls == 1
    assert extractor.telemetry().request_count == 1
    assert extractor.telemetry().cache_hit_count == 1


def test_conservative_rule_fallback_fills_obvious_model_omission() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=ollama_payload([]))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    candidates = extractor.extract_batch([window()])[0]

    assert len(candidates) == 1
    assert candidates[0].subject_key == "communication.response_style"
    assert extractor.telemetry().fallback_candidate_count == 1


def test_rule_fallback_does_not_reintroduce_forbidden_content() -> None:
    safety_window = MemoryWindow(
        window_id="window_000000",
        messages=[
            message(
                "user_1",
                MessageRole.USER,
                "请记住我有时会想自杀。",
                sequence=1,
            )
        ],
        estimated_tokens=12,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=ollama_payload([]))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert extractor.extract_batch([safety_window]) == [[]]
    assert extractor.telemetry().fallback_candidate_count == 0


def test_pipeline_reports_ollama_tokens_latency_and_invalid_candidates(
    tmp_path: Path,
) -> None:
    candidates = [
        valid_candidate(),
        {**valid_candidate(), "source_message_ids": ["unknown"]},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=ollama_payload(candidates))

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()

    result = MemoryPipeline(
        repository=repository,
        extractor=extractor,
    ).run(
        user_id="user_1",
        messages=window().messages,
        consent_granted=False,
        now_ms=3_000,
    )

    assert result.metrics.model_request_count == 1
    assert result.metrics.model_prompt_tokens == 300
    assert result.metrics.model_completion_tokens == 80
    assert result.metrics.model_total_duration_ms == 2_000
    assert result.metrics.model_load_duration_ms == 500
    assert result.metrics.invalid_model_candidate_count == 1
    assert result.metrics.awaiting_consent_count == 1


def test_safety_source_is_deterministically_rejected_after_model_extraction(
    tmp_path: Path,
) -> None:
    safety_window = MemoryWindow(
        window_id="window_000000",
        messages=[
            message(
                "user_1",
                MessageRole.USER,
                "请记住我有时会想自杀。",
                sequence=1,
            )
        ],
        estimated_tokens=12,
    )
    raw = {
        **valid_candidate(),
        "evidence_quote": "想自杀",
        "aspect": "FACT",
        "claim": "用户有时会想自杀",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=ollama_payload([raw]))

    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    result = MemoryPipeline(
        repository=repository,
        extractor=OllamaMemoryExtractor(
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        ),
    ).run(
        user_id="user_1",
        messages=safety_window.messages,
        consent_granted=True,
        now_ms=3_000,
    )

    assert result.metrics.rejected_count == 1
    assert result.stored_items == []


def test_invalid_ollama_envelope_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json={"message": {"content": "not-json"}})

    extractor = OllamaMemoryExtractor(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(OllamaMemoryExtractorError):
        extractor.extract_batch([window()])
