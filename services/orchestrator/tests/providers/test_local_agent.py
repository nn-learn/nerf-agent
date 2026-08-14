import json

import httpx
import pytest

from app.contracts.session import CancellationRegistry
from app.providers.local_agent import LocalAgentProvider
from app.providers.ollama_text import (
    OllamaMetrics,
    OllamaTextProvider,
    OllamaTextResult,
)
from app.rag.retriever import EvidenceBundle, EvidenceItem
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def risk(level: RiskLevel = RiskLevel.GREEN) -> RiskAssessment:
    return RiskAssessment(level=level, confidence=0.7)


def normal_response(
    *,
    evidence_ids: list[str] | None = None,
) -> AgentResponse:
    return AgentResponse(
        spoken_text="听起来你最近承受了不少压力。我们可以先梳理最困扰的一件事。",
        display_text="我们可以先梳理最困扰的一件事。",
        support_mode="listen",
        risk_level=RiskLevel.GREEN,
        evidence_ids=evidence_ids or [],
        visual_observation_ids=[],
        action_proposals=[],
        memory_candidates=[],
        avatar_style="warm",
    )


def reviewed_item(chunk_id: str = "chunk_1") -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id="stress-1",
        document_version="1.0.0",
        title="压力支持",
        text="先识别当前最困扰的一件事。",
        score=0.9,
    )


class RecordingRetriever:
    def __init__(self, bundle: EvidenceBundle) -> None:
        self.bundle = bundle
        self.call: tuple[str, RiskLevel, int] | None = None

    async def retrieve(
        self,
        query: str,
        *,
        risk_level: RiskLevel,
        k: int,
    ) -> EvidenceBundle:
        self.call = (query, risk_level, k)
        return self.bundle


class RecordingTextProvider(OllamaTextProvider):
    def __init__(self, result: OllamaTextResult) -> None:
        self.result = result
        self.call: tuple[dict[str, object], str, str, RiskLevel] | None = None

    async def respond(
        self,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
        risk_level: RiskLevel,
    ) -> OllamaTextResult:
        self.call = (context, turn_id, cancel_token, risk_level)
        return self.result


def ollama_payload(response: AgentResponse) -> dict[str, object]:
    return {
        "model": "qwen3.6:latest",
        "message": {
            "role": "assistant",
            "content": response.model_dump_json(),
        },
        "done": True,
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 120,
        "eval_count": 48,
    }


@pytest.mark.asyncio
async def test_load_context_serializes_only_reviewed_evidence() -> None:
    """Catches raw retriever objects or unreviewed memory leaking into model context."""
    bundle = EvidenceBundle(
        query="最近压力很大",
        items=[reviewed_item()],
        has_sufficient_evidence=True,
    )
    retriever = RecordingRetriever(bundle)
    text = RecordingTextProvider(
        OllamaTextResult(
            response=normal_response(),
            metrics=OllamaMetrics(
                total_duration_ns=0,
                load_duration_ns=0,
                prompt_eval_count=0,
                eval_count=0,
            ),
        )
    )
    provider = LocalAgentProvider(text_provider=text, retriever=retriever)

    context = await provider.load_context(
        "最近压力很大",
        "用户看起来有些疲惫",
        risk(),
        reviewed_evidence_required=True,
    )

    assert retriever.call == ("最近压力很大", RiskLevel.GREEN, 3)
    assert context == {
        "transcript": "最近压力很大",
        "visual_summary": "用户看起来有些疲惫",
        "reviewed_evidence": [
            {
                "chunk_id": "chunk_1",
                "document_id": "stress-1",
                "document_version": "1.0.0",
                "title": "压力支持",
                "text": "先识别当前最困扰的一件事。",
                "score": 0.9,
            }
        ],
        "has_sufficient_evidence": True,
        "working_memory": [],
        "long_term_memory": [],
    }


@pytest.mark.asyncio
async def test_load_context_hides_chunks_from_insufficient_bundle() -> None:
    """Catches insufficient retrieval candidates becoming citeable model evidence."""
    retriever = RecordingRetriever(
        EvidenceBundle(
            query="最近压力很大",
            items=[reviewed_item()],
            has_sufficient_evidence=False,
        )
    )
    text = RecordingTextProvider(
        OllamaTextResult(
            response=normal_response(),
            metrics=OllamaMetrics(
                total_duration_ns=0,
                load_duration_ns=0,
                prompt_eval_count=0,
                eval_count=0,
            ),
        )
    )
    provider = LocalAgentProvider(text_provider=text, retriever=retriever)

    context = await provider.load_context(
        "最近压力很大",
        "",
        risk(),
        reviewed_evidence_required=True,
    )

    assert context["reviewed_evidence"] == []
    assert context["has_sufficient_evidence"] is False


@pytest.mark.asyncio
async def test_load_context_skips_rag_when_control_plane_does_not_require_it() -> None:
    retriever = RecordingRetriever(
        EvidenceBundle(
            query="最近压力很大",
            items=[reviewed_item()],
            has_sufficient_evidence=True,
        )
    )
    provider = LocalAgentProvider(
        text_provider=RecordingTextProvider(
            OllamaTextResult(
                response=normal_response(),
                metrics=OllamaMetrics(
                    total_duration_ns=0,
                    load_duration_ns=0,
                    prompt_eval_count=0,
                    eval_count=0,
                ),
            )
        ),
        retriever=retriever,
    )

    context = await provider.load_context(
        "最近压力很大",
        "",
        risk(),
        reviewed_evidence_required=False,
    )

    assert retriever.call is None
    assert context["reviewed_evidence"] == []


@pytest.mark.asyncio
async def test_plan_reply_forwards_turn_cancellation_and_returns_metrics() -> None:
    """Catches the graph losing cancellation identity or Ollama timing metrics."""
    response = normal_response()
    text = RecordingTextProvider(
        OllamaTextResult(
            response=response,
            metrics=OllamaMetrics(
                total_duration_ns=2_000_000_000,
                load_duration_ns=500_000_000,
                prompt_eval_count=120,
                eval_count=48,
            ),
        )
    )
    retriever = RecordingRetriever(
        EvidenceBundle(
            query="最近压力很大",
            items=[],
            has_sufficient_evidence=False,
        )
    )
    provider = LocalAgentProvider(text_provider=text, retriever=retriever)
    context: dict[str, object] = {
        "reviewed_evidence": [],
        "has_sufficient_evidence": False,
    }

    plan = await provider.plan_reply(
        "最近压力很大",
        risk(),
        context,
        turn_id="turn_1",
        cancel_token="ct_1",
    )

    assert text.call == (
        context,
        "turn_1",
        "ct_1",
        RiskLevel.GREEN,
    )
    assert plan.response == response
    assert plan.provider_metrics == {
        "total_duration_ns": 2_000_000_000,
        "load_duration_ns": 500_000_000,
        "prompt_eval_count": 120,
        "eval_count": 48,
    }


@pytest.mark.asyncio
async def test_plan_crisis_uses_reviewed_handoff_without_calling_ollama() -> None:
    """Catches crisis handling delegating safety wording to the generative model."""
    text = RecordingTextProvider(
        OllamaTextResult(
            response=normal_response(),
            metrics=OllamaMetrics(
                total_duration_ns=0,
                load_duration_ns=0,
                prompt_eval_count=0,
                eval_count=0,
            ),
        )
    )
    provider = LocalAgentProvider(
        text_provider=text,
        retriever=RecordingRetriever(
            EvidenceBundle(
                query="我现在正在伤害自己",
                items=[],
                has_sufficient_evidence=False,
            )
        ),
    )

    plan = await provider.plan_crisis(
        "我现在正在伤害自己",
        risk(RiskLevel.EMERGENCY),
    )

    assert text.call is None
    assert plan.response.model_dump() == {
        "spoken_text": (
            "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
            "并尽快联系身边可信赖的人或当地紧急服务。"
        ),
        "display_text": (
            "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
            "并尽快联系身边可信赖的人或当地紧急服务。"
        ),
        "support_mode": "handoff",
        "risk_level": RiskLevel.EMERGENCY,
        "evidence_ids": [],
        "memory_ids": [],
        "visual_observation_ids": [],
        "action_proposals": [
            {
                "tool": "request_clinician_handoff",
                "status": "PROPOSED",
            }
        ],
        "memory_candidates": [],
        "avatar_style": "handoff_calm",
    }
    assert plan.provider_metrics == {"provider": "deterministic_crisis"}


@pytest.mark.asyncio
async def test_plan_reply_preserves_model_citations_for_host_evidence_gate() -> None:
    """The provider remains a worker; graph policy owns citation authorization."""
    registry = CancellationRegistry()
    cancel_token = await registry.issue("turn_1")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=ollama_payload(normal_response(evidence_ids=["chunk_2"])),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = OllamaTextProvider(registry=registry, client=client)
        provider = LocalAgentProvider(
            text_provider=text,
            retriever=RecordingRetriever(
                EvidenceBundle(
                    query="最近压力很大",
                    items=[reviewed_item("chunk_1")],
                    has_sufficient_evidence=True,
                )
            ),
        )
        context = await provider.load_context(
            "最近压力很大",
            "",
            risk(),
            reviewed_evidence_required=True,
        )

        plan = await provider.plan_reply(
            "最近压力很大",
            risk(),
            context,
            turn_id="turn_1",
            cancel_token=cancel_token,
        )

    assert plan.response.evidence_ids == ["chunk_2"]


@pytest.mark.asyncio
async def test_plan_reply_allows_no_citations_when_no_evidence_exists() -> None:
    """Catches evidence-free supportive responses being rejected without citations."""
    registry = CancellationRegistry()
    cancel_token = await registry.issue("turn_1")

    def handler(request: httpx.Request) -> httpx.Response:
        request_context = json.loads(json.loads(request.content)["messages"][1]["content"])
        assert request_context["context"]["reviewed_evidence"] == []
        return httpx.Response(200, json=ollama_payload(normal_response()))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        text = OllamaTextProvider(registry=registry, client=client)
        provider = LocalAgentProvider(
            text_provider=text,
            retriever=RecordingRetriever(
                EvidenceBundle(
                    query="最近压力很大",
                    items=[],
                    has_sufficient_evidence=False,
                )
            ),
        )
        context = await provider.load_context(
            "最近压力很大",
            "",
            risk(),
            reviewed_evidence_required=True,
        )

        plan = await provider.plan_reply(
            "最近压力很大",
            risk(),
            context,
            turn_id="turn_1",
            cancel_token=cancel_token,
        )

    assert plan.response.evidence_ids == []
    assert plan.response.support_mode == "listen"
