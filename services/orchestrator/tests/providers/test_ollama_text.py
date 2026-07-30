import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.contracts.session import CancellationRegistry
from app.providers.ollama_text import OllamaTextProvider
from app.providers.qwen_text import TextProviderCancelled, TextProviderResponseError
from app.safety.models import AgentResponse, RiskLevel


def valid_agent_response(
    *,
    risk_level: str = "GREEN",
    evidence_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "spoken_text": "听起来你最近承受了不少压力。我们可以先梳理最困扰的一件事。",
        "display_text": "我们可以先梳理最困扰的一件事。",
        "support_mode": "listen",
        "risk_level": risk_level,
        "evidence_ids": evidence_ids or [],
        "visual_observation_ids": [],
        "action_proposals": [],
        "memory_candidates": [],
        "avatar_style": "warm",
    }


def ollama_chat_payload(
    *,
    agent_response: dict[str, object] | None = None,
    content: str | None = None,
) -> dict[str, object]:
    response_content = content
    if response_content is None:
        response_content = json.dumps(
            agent_response or valid_agent_response(),
            ensure_ascii=False,
        )
    return {
        "model": "qwen3.6:latest",
        "message": {
            "role": "assistant",
            "content": response_content,
        },
        "done": True,
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 120,
        "eval_count": 48,
    }


async def current_turn(
    registry: CancellationRegistry,
    turn_id: str = "turn_1",
) -> tuple[str, str]:
    return turn_id, await registry.issue(turn_id)


@pytest.mark.asyncio
async def test_respond_sends_native_chat_contract_and_returns_metrics() -> None:
    """Catches incompatible Ollama request options or discarded timing metrics."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=ollama_chat_payload())

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        result = await provider.respond(
            {
                "transcript": "最近压力很大",
                "reviewed_evidence": [],
                "has_sufficient_evidence": False,
            },
            turn_id=turn_id,
            cancel_token=token,
            risk_level=RiskLevel.GREEN,
        )

    assert captured["model"] == "qwen3.6:latest"
    assert captured["stream"] is False
    assert captured["think"] is False
    assert captured["format"] == AgentResponse.model_json_schema()
    assert captured["keep_alive"] == "30m"
    assert captured["options"] == {
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 320,
    }
    assert result.response.risk_level is RiskLevel.GREEN
    assert result.metrics.model_dump() == {
        "total_duration_ns": 2_000_000_000,
        "load_duration_ns": 500_000_000,
        "prompt_eval_count": 120,
        "eval_count": 48,
    }


@pytest.mark.asyncio
async def test_respond_rejects_cancelled_turn_before_request() -> None:
    """Catches stale turns that still consume a local generation slot."""
    entered_transport = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal entered_transport
        entered_transport = True
        return httpx.Response(200, json=ollama_chat_payload())

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    await registry.cancel(token)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(TextProviderCancelled):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )

    assert entered_transport is False


@pytest.mark.asyncio
async def test_respond_rejects_cancelled_turn_after_response() -> None:
    """Catches a completed stale response being published after barge-in."""
    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)

    async def handler(request: httpx.Request) -> httpx.Response:
        await registry.cancel(token)
        return httpx.Response(200, json=ollama_chat_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(TextProviderCancelled):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_rejects_model_changed_deterministic_risk() -> None:
    """Catches a model overriding the deterministic safety classifier."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=ollama_chat_payload(
                agent_response=valid_agent_response(risk_level="AMBER"),
            ),
        )

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(
            TextProviderResponseError,
            match="changed the deterministic risk level",
        ):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_rejects_evidence_outside_reviewed_bundle() -> None:
    """Catches citations invented outside the reviewed turn evidence."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=ollama_chat_payload(
                agent_response=valid_agent_response(evidence_ids=["chunk_unknown"]),
            ),
        )

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(
            TextProviderResponseError,
            match="outside the reviewed turn bundle",
        ):
            await provider.respond(
                {
                    "reviewed_evidence": [{"chunk_id": "chunk_1"}],
                    "has_sufficient_evidence": True,
                },
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_rejects_citations_from_insufficient_bundle() -> None:
    """Catches citations when retrieval did not meet the evidence threshold."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=ollama_chat_payload(
                agent_response=valid_agent_response(evidence_ids=["chunk_1"]),
            ),
        )

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(
            TextProviderResponseError,
            match="insufficient evidence bundle",
        ):
            await provider.respond(
                {
                    "reviewed_evidence": [{"chunk_id": "chunk_1"}],
                    "has_sufficient_evidence": False,
                },
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_maps_malformed_json_to_provider_error() -> None:
    """Catches invalid model JSON escaping the shared provider error vocabulary."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ollama_chat_payload(content="not-json"))

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(TextProviderResponseError):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_maps_invalid_metrics_to_provider_error() -> None:
    """Catches malformed Ollama metrics leaking Pydantic-specific exceptions."""
    payload = ollama_chat_payload()
    payload["eval_count"] = -1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(TextProviderResponseError):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_respond_maps_timeout_to_provider_error() -> None:
    """Catches local transport timeouts leaking httpx-specific exceptions."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("local model timed out", request=request)

    registry = CancellationRegistry()
    turn_id, token = await current_turn(registry)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        with pytest.raises(TextProviderResponseError):
            await provider.respond(
                {"reviewed_evidence": []},
                turn_id=turn_id,
                cancel_token=token,
                risk_level=RiskLevel.GREEN,
            )


@pytest.mark.asyncio
async def test_readiness_reports_exact_model_missing_without_response_body() -> None:
    """Catches fuzzy model matches and diagnostic leakage from readiness."""
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.11.4"})
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3.6:latest-q4",
                        "model": "qwen3.6:latest-q4",
                        "size": 23_000_000_000,
                    }
                ]
            },
        )

    registry = CancellationRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        readiness = await provider.readiness()

    assert requested_paths == ["/api/version", "/api/tags"]
    assert readiness.ready is False
    assert readiness.version == "0.11.4"
    assert readiness.model == "qwen3.6:latest"
    assert readiness.reason == "model_not_found"
    assert "qwen3.6:latest-q4" not in repr(readiness)


@pytest.mark.asyncio
async def test_prewarm_sends_exact_generate_contract() -> None:
    """Catches prewarm accidentally generating content or unloading the model."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    registry = CancellationRegistry()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)
        await provider.prewarm()

    assert captured == {
        "model": "qwen3.6:latest",
        "prompt": "",
        "stream": False,
        "keep_alive": "30m",
    }


@pytest.mark.asyncio
async def test_respond_serializes_generations_across_turns() -> None:
    """Catches concurrent local generations that can exhaust model memory."""
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    request_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            first_entered.set()
            await release_first.wait()
        return httpx.Response(200, json=ollama_chat_payload())

    registry = CancellationRegistry()
    first_turn, first_token = await current_turn(registry, "turn_1")
    second_turn, second_token = await current_turn(registry, "turn_2")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = OllamaTextProvider(registry=registry, client=client)

        def respond(turn_id: str, token: str) -> Callable[[], object]:
            async def call() -> object:
                return await provider.respond(
                    {"reviewed_evidence": []},
                    turn_id=turn_id,
                    cancel_token=token,
                    risk_level=RiskLevel.GREEN,
                )

            return call

        first_task = asyncio.create_task(respond(first_turn, first_token)())
        await first_entered.wait()
        second_task = asyncio.create_task(respond(second_turn, second_token)())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert request_count == 1

        release_first.set()
        await asyncio.gather(first_task, second_task)

    assert request_count == 2
