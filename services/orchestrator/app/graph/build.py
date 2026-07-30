from dataclasses import dataclass
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph.state import AgentState
from app.providers.mock import MockAgentProvider
from app.providers.protocols import AgentProvider
from app.safety.models import RiskLevel
from app.safety.output_guard import OutputGuard
from app.safety.rules import assess_risk


@dataclass(frozen=True)
class GraphDependencies:
    agent_provider: AgentProvider
    output_guard: OutputGuard

    @classmethod
    def for_mock(cls) -> "GraphDependencies":
        return cls(agent_provider=MockAgentProvider(), output_guard=OutputGuard())


def build_graph(
    dependencies: GraphDependencies,
    *,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> Any:
    builder = StateGraph(AgentState)

    def normalize_input(state: AgentState) -> dict[str, object]:
        return {"transcript": state["transcript"].strip(), "visited": ["normalize_input"]}

    def partial_risk(state: AgentState) -> dict[str, object]:
        _ = state
        return {"visited": ["partial_risk"]}

    def final_risk(state: AgentState) -> dict[str, object]:
        return {
            "risk": assess_risk(state["transcript"], state.get("visual_summary", "")),
            "visited": ["final_risk"],
        }

    async def context_fetch(state: AgentState) -> dict[str, object]:
        return {
            "context": await dependencies.agent_provider.load_context(
                state["transcript"],
                state.get("visual_summary", ""),
                state["risk"],
            ),
            "visited": ["context_fetch"],
        }

    async def reply_planner(state: AgentState) -> dict[str, object]:
        plan = await dependencies.agent_provider.plan_reply(
            transcript=state["transcript"],
            risk=state["risk"],
            context=state["context"],
            turn_id=state["turn_id"],
            cancel_token=state["cancel_token"],
        )
        return {
            "candidate_response": plan.response,
            "provider_metrics": plan.provider_metrics,
            "visited": ["reply_planner"],
        }

    async def crisis_policy(state: AgentState) -> dict[str, object]:
        plan = await dependencies.agent_provider.plan_crisis(
            transcript=state["transcript"],
            risk=state["risk"],
        )
        return {
            "candidate_response": plan.response,
            "provider_metrics": plan.provider_metrics,
            "visited": ["crisis_policy"],
        }

    def route_after_risk(
        state: AgentState,
    ) -> Literal["crisis_policy", "context_fetch"]:
        if state["risk"].level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            return "crisis_policy"
        return "context_fetch"

    def output_guard(state: AgentState) -> dict[str, object]:
        return {
            "response": dependencies.output_guard.validate(state["candidate_response"]),
            "visited": ["output_guard"],
        }

    def publish_response(state: AgentState) -> dict[str, object]:
        _ = state
        return {"visited": ["publish_response"]}

    def propose_memory(state: AgentState) -> dict[str, object]:
        _ = state
        return {"visited": ["propose_memory"]}

    builder.add_node("normalize_input", normalize_input)
    builder.add_node("partial_risk", partial_risk)
    builder.add_node("final_risk", final_risk)
    builder.add_node("context_fetch", context_fetch)
    builder.add_node("reply_planner", reply_planner)
    builder.add_node("crisis_policy", crisis_policy)
    builder.add_node("output_guard", output_guard)
    builder.add_node("publish_response", publish_response)
    builder.add_node("propose_memory", propose_memory)
    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "partial_risk")
    builder.add_edge("partial_risk", "final_risk")
    builder.add_conditional_edges("final_risk", route_after_risk)
    builder.add_edge("context_fetch", "reply_planner")
    builder.add_edge("reply_planner", "output_guard")
    builder.add_edge("crisis_policy", "output_guard")
    builder.add_edge("output_guard", "publish_response")
    builder.add_edge("publish_response", "propose_memory")
    builder.add_edge("propose_memory", END)
    return builder.compile(checkpointer=checkpointer)
