from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent.actions import CapabilityProposalGate
from app.agent.avatar import AvatarPolicy
from app.agent.control import AgentControlPlane
from app.agent.evidence import EvidenceOrchestrator
from app.agent.models import EvidenceRequirement, MemoryAccessMode, ResponseStrategy
from app.graph.state import AgentState
from app.providers.mock import MockAgentProvider
from app.providers.protocols import AgentProvider
from app.safety.models import AgentResponse
from app.safety.output_guard import OutputGuard
from app.safety.rules import assess_risk

MemoryContextLoader = Callable[
    [str, str, str], Awaitable[list[dict[str, object]]]
]


@dataclass(frozen=True)
class GraphDependencies:
    agent_provider: AgentProvider
    output_guard: OutputGuard
    memory_context_loader: MemoryContextLoader | None = None
    control_plane: AgentControlPlane = field(default_factory=AgentControlPlane)
    capability_gate: CapabilityProposalGate = field(
        default_factory=CapabilityProposalGate
    )
    evidence_orchestrator: EvidenceOrchestrator = field(
        default_factory=EvidenceOrchestrator
    )
    avatar_policy: AvatarPolicy = field(default_factory=AvatarPolicy)

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

    def intent_policy(state: AgentState) -> dict[str, object]:
        intent = dependencies.control_plane.classify(state["transcript"])
        directive = dependencies.control_plane.draft(
            state["transcript"],
            state["risk"],
        )
        result: dict[str, object] = {
            "intent": intent,
            "agent_directive": directive,
            "visited": ["intent_policy"],
        }
        if directive.response_strategy is ResponseStrategy.DETERMINISTIC_CRISIS:
            _, trace = dependencies.control_plane.finalize(directive, {})
            result["agent_trace"] = trace
        return result

    async def context_fetch(state: AgentState) -> dict[str, object]:
        context = await dependencies.agent_provider.load_context(
            state["transcript"],
            state.get("visual_summary", ""),
            state["risk"],
            reviewed_evidence_required=(
                state["agent_directive"].evidence_requirement
                is EvidenceRequirement.REVIEWED_KNOWLEDGE
            ),
        )
        loader = dependencies.memory_context_loader
        session_id = state.get("session_id")
        if (
            loader is not None
            and session_id is not None
            and state["agent_directive"].memory_access
            is not MemoryAccessMode.FORBIDDEN
        ):
            try:
                context["long_term_memory"] = await loader(
                    session_id,
                    state["turn_id"],
                    state["transcript"],
                )
                context["memory_retrieval_degraded"] = False
            except Exception:
                # Personalization is optional context. Its failure must never
                # prevent a support turn or weaken deterministic risk routing.
                context["long_term_memory"] = []
                context["memory_retrieval_degraded"] = True
        else:
            context["long_term_memory"] = []
            context["memory_retrieval_degraded"] = False
        return {
            "context": context,
            "visited": ["context_fetch"],
        }

    def evidence_prepare(state: AgentState) -> dict[str, object]:
        context, audit = dependencies.evidence_orchestrator.prepare(
            state["context"],
            state["agent_directive"],
        )
        return {
            "context": context,
            "evidence_context_audit": audit,
            "visited": ["evidence_prepare"],
        }

    def decision_gate(state: AgentState) -> dict[str, object]:
        directive, trace = dependencies.control_plane.finalize(
            state["agent_directive"],
            state["context"],
        )
        context = dict(state["context"])
        context["agent_control"] = directive.model_dump(mode="json")
        return {
            "agent_directive": directive,
            "agent_trace": trace,
            "context": context,
            "visited": ["decision_gate"],
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

    def route_after_intent(
        state: AgentState,
    ) -> Literal["crisis_policy", "context_fetch"]:
        if (
            state["agent_directive"].response_strategy
            is ResponseStrategy.DETERMINISTIC_CRISIS
        ):
            return "crisis_policy"
        return "context_fetch"

    def route_after_decision(
        state: AgentState,
    ) -> Literal["evidence_fallback", "reply_planner"]:
        if (
            state["agent_directive"].response_strategy
            is ResponseStrategy.DETERMINISTIC_ABSTAIN
        ):
            return "evidence_fallback"
        return "reply_planner"

    def evidence_fallback(state: AgentState) -> dict[str, object]:
        if (
            state["agent_directive"].evidence_requirement
            is EvidenceRequirement.CONFIRMED_MEMORY
        ):
            text = "我没有找到与你这个问题相关的已确认记忆，所以不会猜测。"
            support_mode: Literal["listen", "educate"] = "listen"
        else:
            text = "我目前没有足够的经审核资料来可靠回答这个问题，所以不会猜测。"
            support_mode = "educate"
        return {
            "candidate_response": AgentResponse(
                spoken_text=text,
                display_text=text,
                support_mode=support_mode,
                risk_level=state["risk"].level,
                evidence_ids=[],
                visual_observation_ids=[],
                action_proposals=[],
                memory_candidates=[],
                avatar_style="neutral_listening",
            ),
            "provider_metrics": {"provider": "deterministic_evidence_gate"},
            "visited": ["evidence_fallback"],
        }

    def output_guard(state: AgentState) -> dict[str, object]:
        controlled = dependencies.control_plane.validate_response(
            state["candidate_response"],
            state["agent_directive"],
        )
        return {
            "response": dependencies.output_guard.validate(controlled),
            "visited": ["output_guard"],
        }

    def capability_gate(state: AgentState) -> dict[str, object]:
        response, audit = dependencies.capability_gate.apply(
            state["response"],
            state["agent_directive"],
        )
        return {
            "response": response,
            "capability_audit": audit,
            "visited": ["capability_gate"],
        }

    def evidence_response_gate(state: AgentState) -> dict[str, object]:
        response, audit = dependencies.evidence_orchestrator.authorize_response(
            state["response"],
            state.get("context", {}),
            state["agent_directive"],
        )
        return {
            "response": response,
            "evidence_response_audit": audit,
            "visited": ["evidence_response_gate"],
        }

    def avatar_policy(state: AgentState) -> dict[str, object]:
        plan = dependencies.avatar_policy.plan(
            state["response"],
            state["agent_directive"],
        )
        return {
            "response": state["response"].model_copy(
                update={"avatar_style": plan.style}
            ),
            "avatar_plan": plan,
            "visited": ["avatar_policy"],
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
    builder.add_node("intent_policy", intent_policy)
    builder.add_node("context_fetch", context_fetch)
    builder.add_node("evidence_prepare", evidence_prepare)
    builder.add_node("decision_gate", decision_gate)
    builder.add_node("evidence_fallback", evidence_fallback)
    builder.add_node("reply_planner", reply_planner)
    builder.add_node("crisis_policy", crisis_policy)
    builder.add_node("output_guard", output_guard)
    builder.add_node("evidence_response_gate", evidence_response_gate)
    builder.add_node("capability_gate", capability_gate)
    builder.add_node("avatar_policy", avatar_policy)
    builder.add_node("publish_response", publish_response)
    builder.add_node("propose_memory", propose_memory)
    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "partial_risk")
    builder.add_edge("partial_risk", "final_risk")
    builder.add_edge("final_risk", "intent_policy")
    builder.add_conditional_edges("intent_policy", route_after_intent)
    builder.add_edge("context_fetch", "evidence_prepare")
    builder.add_edge("evidence_prepare", "decision_gate")
    builder.add_conditional_edges("decision_gate", route_after_decision)
    builder.add_edge("reply_planner", "output_guard")
    builder.add_edge("evidence_fallback", "output_guard")
    builder.add_edge("crisis_policy", "output_guard")
    builder.add_edge("output_guard", "evidence_response_gate")
    builder.add_edge("evidence_response_gate", "capability_gate")
    builder.add_edge("capability_gate", "avatar_policy")
    builder.add_edge("avatar_policy", "publish_response")
    builder.add_edge("publish_response", "propose_memory")
    builder.add_edge("propose_memory", END)
    return builder.compile(checkpointer=checkpointer)
