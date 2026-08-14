from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.graph.build import GraphDependencies, build_graph
from app.providers.protocols import AgentPlan
from app.safety.models import AgentResponse, RiskAssessment, RiskLevel


def agent_response(risk_level: RiskLevel, *, support_mode: str) -> AgentResponse:
    return AgentResponse(
        spoken_text="我听见你正在承受压力，我们可以慢慢说。",
        display_text="我听见你正在承受压力，我们可以慢慢说。",
        support_mode=support_mode,
        risk_level=risk_level,
        evidence_ids=[],
        visual_observation_ids=[],
        action_proposals=[],
        memory_candidates=[],
        avatar_style="handoff_calm" if support_mode == "handoff" else "warm",
    )


class RecordingAgentProvider:
    def __init__(self) -> None:
        self.context_call: tuple[str, str, RiskLevel] | None = None
        self.reply_call: tuple[str, str, RiskLevel] | None = None
        self.crisis_call: tuple[str, RiskLevel] | None = None

    async def load_context(
        self,
        transcript: str,
        visual_summary: str,
        risk: RiskAssessment,
        *,
        reviewed_evidence_required: bool,
    ) -> dict[str, object]:
        _ = reviewed_evidence_required
        self.context_call = (transcript, visual_summary, risk.level)
        return {
            "reviewed_evidence": [],
            "has_sufficient_evidence": False,
        }

    async def plan_reply(
        self,
        transcript: str,
        risk: RiskAssessment,
        context: dict[str, object],
        *,
        turn_id: str,
        cancel_token: str,
    ) -> AgentPlan:
        _ = transcript, context
        self.reply_call = (turn_id, cancel_token, risk.level)
        return AgentPlan(
            response=agent_response(risk.level, support_mode="listen"),
            provider_metrics={"provider": "recording_normal"},
        )

    async def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentPlan:
        self.crisis_call = (transcript, risk.level)
        return AgentPlan(
            response=agent_response(risk.level, support_mode="handoff"),
            provider_metrics={"provider": "deterministic_crisis"},
        )


@pytest.mark.asyncio
async def test_green_turn_fetches_context_before_reply() -> None:
    """Catches normal turns that bypass governed context or cancellation metadata."""
    provider = RecordingAgentProvider()
    graph = build_graph(
        GraphDependencies(
            agent_provider=provider,
            output_guard=GraphDependencies.for_mock().output_guard,
        )
    )

    result = await graph.ainvoke(
        {
            "transcript": "最近压力很大",
            "visual_summary": "",
            "turn_id": "turn_1",
            "cancel_token": "ct_1",
            "visited": [],
        }
    )

    assert result["risk"].level is RiskLevel.GREEN
    assert result["visited"] == [
        "normalize_input",
        "partial_risk",
        "final_risk",
        "intent_policy",
        "context_fetch",
        "decision_gate",
        "reply_planner",
        "output_guard",
        "publish_response",
        "propose_memory",
    ]
    assert provider.context_call == ("最近压力很大", "", RiskLevel.GREEN)
    assert provider.reply_call == ("turn_1", "ct_1", RiskLevel.GREEN)
    assert result["response"].support_mode == "listen"
    assert result["provider_metrics"] == {"provider": "recording_normal"}


@pytest.mark.asyncio
async def test_emergency_turn_bypasses_normal_context_and_uses_crisis_policy() -> None:
    """Catches emergency turns that call normal retrieval or Ollama planning."""
    provider = RecordingAgentProvider()
    graph = build_graph(
        GraphDependencies(
            agent_provider=provider,
            output_guard=GraphDependencies.for_mock().output_guard,
        )
    )

    result = await graph.ainvoke(
        {
            "transcript": "我现在正在伤害自己",
            "visual_summary": "",
            "turn_id": "turn_1",
            "cancel_token": "ct_1",
            "visited": [],
        }
    )

    assert result["risk"].level is RiskLevel.EMERGENCY
    assert "crisis_policy" in result["visited"]
    assert "intent_policy" in result["visited"]
    assert "context_fetch" not in result["visited"]
    assert "reply_planner" not in result["visited"]
    assert provider.context_call is None
    assert provider.reply_call is None
    assert provider.crisis_call == ("我现在正在伤害自己", RiskLevel.EMERGENCY)
    assert result["response"].support_mode == "handoff"
    assert result["provider_metrics"] == {"provider": "deterministic_crisis"}
    assert result["agent_trace"].reason_codes[0] == "RISK_OVERRIDE"


@pytest.mark.asyncio
async def test_memory_question_without_confirmed_memory_uses_deterministic_gate() -> None:
    provider = RecordingAgentProvider()

    async def empty_memory(
        session_id: str,
        turn_id: str,
        query: str,
    ) -> list[dict[str, object]]:
        _ = session_id, turn_id, query
        return []

    graph = build_graph(
        GraphDependencies(
            agent_provider=provider,
            output_guard=GraphDependencies.for_mock().output_guard,
            memory_context_loader=empty_memory,
        )
    )

    result = await graph.ainvoke(
        {
            "session_id": "session_1",
            "transcript": "你还记得我之前说过什么吗？",
            "visual_summary": "",
            "turn_id": "turn_1",
            "cancel_token": "ct_1",
            "visited": [],
        }
    )

    assert "evidence_fallback" in result["visited"]
    assert "reply_planner" not in result["visited"]
    assert provider.reply_call is None
    assert result["provider_metrics"] == {
        "provider": "deterministic_evidence_gate"
    }
    assert "不会猜测" in result["response"].spoken_text


@pytest.mark.asyncio
async def test_graph_state_round_trips_through_sqlite_checkpoint(
    tmp_path: Path,
) -> None:
    """Catches graph state that cannot be resumed or audited by session thread ID."""
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "session_1"}}

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(GraphDependencies.for_mock(), checkpointer=checkpointer)
        await graph.ainvoke(
            {
                "transcript": "最近工作压力有点大",
                "visual_summary": "",
                "turn_id": "turn_1",
                "cancel_token": "ct_1",
                "visited": [],
            },
            config,
        )
        snapshot = await graph.aget_state(config)

    assert snapshot.values["risk"].level is RiskLevel.GREEN
    assert snapshot.values["response"].support_mode == "listen"
    assert snapshot.values["provider_metrics"] == {}
