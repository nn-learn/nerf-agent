from app.providers.protocols import AgentPlan
from app.safety.models import AgentResponse, RiskAssessment


class MockAgentProvider:
    async def load_context(
        self,
        transcript: str,
        visual_summary: str,
        risk: RiskAssessment,
        *,
        reviewed_evidence_required: bool,
    ) -> dict[str, object]:
        _ = reviewed_evidence_required
        return {
            "working_memory": [],
            "long_term_memory": [],
            "reviewed_evidence": [],
            "has_sufficient_evidence": False,
            "visual_summary": visual_summary,
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
        return AgentPlan(
            response=AgentResponse(
                spoken_text="我听见你正在承受压力，我们可以慢慢说。",
                display_text="我听见你正在承受压力，我们可以慢慢说。",
                support_mode="listen",
                risk_level=risk.level,
                evidence_ids=[],
                visual_observation_ids=[],
                action_proposals=[],
                memory_candidates=[],
                avatar_style="warm",
            )
        )

    async def plan_crisis(
        self,
        transcript: str,
        risk: RiskAssessment,
    ) -> AgentPlan:
        return AgentPlan(
            response=AgentResponse(
                spoken_text=(
                    "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
                    "并尽快联系身边可信赖的人或当地紧急服务。"
                ),
                display_text=(
                    "我很重视你刚才说的情况。请先远离可能伤害你的物品，"
                    "并尽快联系身边可信赖的人或当地紧急服务。"
                ),
                support_mode="handoff",
                risk_level=risk.level,
                evidence_ids=[],
                visual_observation_ids=[],
                action_proposals=[
                    {
                        "tool": "request_clinician_handoff",
                        "status": "PROPOSED",
                    }
                ],
                memory_candidates=[],
                avatar_style="handoff_calm",
            )
        )

