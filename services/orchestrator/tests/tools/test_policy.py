from app.safety.models import RiskLevel
from app.tools.gateway import ToolGateway
from app.tools.models import ToolState


def test_emergency_contact_tool_cannot_execute_without_approval() -> None:
    """Catches a high-impact contact draft bypassing explicit approval."""
    gateway = ToolGateway.with_builtin_demo_tools()
    proposal = gateway.propose(
        name="draft_emergency_contact_message",
        arguments={"message": "演示文本"},
        risk_level=RiskLevel.RED,
    )

    result = gateway.execute(proposal.tool_call_id)

    assert result.state is ToolState.AWAITING_APPROVAL
    assert result.output is None


def test_approved_contact_tool_creates_draft_without_sending() -> None:
    """Catches the V1 demo tool accidentally acquiring an external send side effect."""
    gateway = ToolGateway.with_builtin_demo_tools()
    proposal = gateway.propose(
        name="draft_emergency_contact_message",
        arguments={"message": "请联系我"},
        risk_level=RiskLevel.RED,
    )
    gateway.approve(proposal.tool_call_id)

    result = gateway.execute(proposal.tool_call_id)

    assert result.state is ToolState.SUCCEEDED
    assert result.output == {
        "draft": "请联系我",
        "sent": False,
    }


def test_idempotency_key_returns_one_tool_result() -> None:
    """Catches duplicate delivery executing the same tool action twice."""
    gateway = ToolGateway.with_builtin_demo_tools()
    first = gateway.propose(
        name="start_breathing_exercise",
        arguments={"duration_seconds": 60},
        risk_level=RiskLevel.GREEN,
        idempotency_key="idem_1",
    )
    second = gateway.propose(
        name="start_breathing_exercise",
        arguments={"duration_seconds": 60},
        risk_level=RiskLevel.GREEN,
        idempotency_key="idem_1",
    )

    assert first.tool_call_id == second.tool_call_id
    assert gateway.execute(first.tool_call_id) == gateway.execute(second.tool_call_id)
