import json
from pathlib import Path

import httpx

from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.models import MemoryAspect, MemoryCandidate, MemoryItem, MemoryKind
from app.memory.ollama_normalizer import (
    CLAIM_NORMALIZATION_POLICY,
    OllamaMemoryClaimNormalizer,
)
from app.memory.repository import MemoryRepository


def _candidate(text: str, *, turn_id: str, subject_key: str) -> MemoryCandidate:
    return MemoryCandidate(
        source="transcript",
        contains_sensitive_content=False,
        text=text,
        kind=MemoryKind.SEMANTIC,
        source_turn_id=turn_id,
        aspect=MemoryAspect.PREFERENCE,
        subject_key=subject_key,
        confidence=0.9,
    )


def _items(
    tmp_path: Path,
    claims: list[tuple[str, str]],
) -> tuple[MemoryRepository, list[MemoryItem]]:
    repository = MemoryRepository(tmp_path / "memory.sqlite3")
    repository.initialize()
    items = [
        repository.add_active(
            user_id="user_1",
            candidate=_candidate(
                text,
                turn_id=f"turn_{index}",
                subject_key=subject_key,
            ),
            now_ms=1_000 + index,
        )
        for index, (subject_key, text) in enumerate(claims)
    ]
    return repository, items


def _payload(clusters: list[list[str]]) -> dict[str, object]:
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "clusters": [
                        {"member_memory_ids": members} for members in clusters
                    ]
                }
            ),
        },
        "prompt_eval_count": 80,
        "eval_count": 20,
        "total_duration": 1_000,
        "load_duration": 100,
    }


def test_model_can_only_group_opaque_ids_and_never_write_profile_text(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("coping.music", "听轻音乐能让我慢慢平静"),
            ("coping.music", "焦虑时播放舒缓音乐对我有帮助"),
        ],
    )
    items = raw_items
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json=_payload([[item.memory_id for item in items]]),
        )

    normalizer = OllamaMemoryClaimNormalizer(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    normalized = normalizer.normalize_many(items)

    assert len({claim.signature for claim in normalized.values()}) == 1
    assert {claim.statement for claim in normalized.values()} == {
        "听轻音乐能让我慢慢平静"
    }
    request_messages = captured["messages"]
    assert isinstance(request_messages, list)
    assert request_messages[0]["content"] == CLAIM_NORMALIZATION_POLICY
    assert captured["think"] is False
    assert captured["stream"] is False
    assert normalizer.telemetry().model_grouped_claim_count == 2


def test_known_high_risk_semantics_remain_rule_locked_without_model_call(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("communication.response_style", "用户偏好：回答时简短"),
            ("communication.response_style", "用户偏好：先给简洁结论"),
        ],
    )
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        _ = request
        calls += 1
        return httpx.Response(500)

    normalizer = OllamaMemoryClaimNormalizer(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    normalized = normalizer.normalize_many(
        raw_items
    )

    assert calls == 0
    assert {claim.signature for claim in normalized.values()} == {
        "response_style.concise"
    }


def test_fabricated_omitted_or_duplicate_ids_fail_back_to_exact_matching(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("coping.music", "听轻音乐能让我平静"),
            ("coping.music", "舒缓音乐对我有帮助"),
        ],
    )
    items = raw_items

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json=_payload([[items[0].memory_id, "memory_fabricated"]]),
        )

    normalizer = OllamaMemoryClaimNormalizer(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    normalized = normalizer.normalize_many(items)

    assert len({claim.signature for claim in normalized.values()}) == 2
    telemetry = normalizer.telemetry()
    assert telemetry.failure_count == 1
    assert telemetry.invalid_response_count == 1
    assert telemetry.fallback_claim_count == 2


def test_polarity_and_subject_boundaries_reject_unsafe_model_merges(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("coping.music", "舒缓音乐对我有帮助"),
            ("coping.music", "我不希望再听舒缓音乐"),
            ("coping.walking", "散步能让我平静"),
            ("coping.walking", "慢走对我有帮助"),
        ],
    )
    items = raw_items

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            200,
            json=_payload([[item.memory_id for item in items]]),
        )

    normalizer = OllamaMemoryClaimNormalizer(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    normalized = normalizer.normalize_many(items)

    assert len({claim.signature for claim in normalized.values()}) == 4
    assert normalizer.telemetry().invalid_response_count == 1


def test_http_failure_is_cached_only_as_no_result_and_next_rebuild_can_retry(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("coping.music", "听轻音乐能让我平静"),
            ("coping.music", "舒缓音乐对我有帮助"),
        ],
    )
    items = raw_items
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        _ = request
        calls += 1
        if calls == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=_payload([[item.memory_id for item in items]]))

    normalizer = OllamaMemoryClaimNormalizer(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )

    first = normalizer.normalize_many(items)
    second = normalizer.normalize_many(items)
    third = normalizer.normalize_many(items)

    assert len({claim.signature for claim in first.values()}) == 2
    assert len({claim.signature for claim in second.values()}) == 1
    assert second == third
    assert calls == 2
    assert normalizer.telemetry().cache_hit_count == 1


def test_semantic_grouping_still_requires_two_sessions_and_user_confirmation(
    tmp_path: Path,
) -> None:
    repository, raw_items = _items(
        tmp_path,
        [
            ("coping.music", "听轻音乐能让我慢慢平静"),
            ("coping.music", "焦虑时舒缓音乐对我有帮助"),
        ],
    )
    items = raw_items

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=_payload([[item.memory_id for item in items]]))

    profile_repository = MemoryProfileRepository(repository.database_path)
    profile_repository.record_observation(items[0], session_id="session_1")
    profile_repository.record_observation(items[1], session_id="session_2")
    consolidator = MemoryConsolidator(
        profile_repository,
        normalizer=OllamaMemoryClaimNormalizer(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    )

    proposed = consolidator.rebuild_user("user_1")

    assert len(proposed) == 1
    assert proposed[0].distinct_session_count == 2
    assert proposed[0].statement in {
        "听轻音乐能让我慢慢平静",
        "焦虑时舒缓音乐对我有帮助",
    }


def test_normalizer_version_change_stales_old_active_profile(
    tmp_path: Path,
) -> None:
    repository, items = _items(
        tmp_path,
        [
            ("coping.music", "听轻音乐能让我慢慢平静"),
            ("coping.music", "焦虑时舒缓音乐对我有帮助"),
        ],
    )
    profile_repository = MemoryProfileRepository(repository.database_path)
    profile_repository.record_observation(items[0], session_id="session_1")
    profile_repository.record_observation(items[0], session_id="session_2")
    baseline = MemoryConsolidator(profile_repository).rebuild_user("user_1")
    old_active = profile_repository.confirm_profile(
        baseline[0].profile_id,
        user_id="user_1",
    )
    profile_repository.record_observation(items[1], session_id="session_3")

    def handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json=_payload([[item.memory_id for item in items]]))

    changed = MemoryConsolidator(
        profile_repository,
        normalizer=OllamaMemoryClaimNormalizer(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        ),
    ).rebuild_user("user_1")

    assert len(changed) == 1
    assert changed[0].profile_id != old_active.profile_id
    assert profile_repository.get_profile(
        old_active.profile_id,
        user_id="user_1",
    ).state.value == "STALE"
    assert profile_repository.list_active_profiles("user_1") == []
