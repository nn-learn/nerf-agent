import pytest

from app.safety.models import RiskLevel
from app.tools.gateway import ToolGateway, ToolPolicyError
from app.tools.models import (
    ApprovalRole,
    CapabilityDefinition,
    CapabilityImpact,
    CapabilityOrigin,
    ToolState,
)
from app.tools.policy import CapabilityRegistrationError, CapabilityRegistry


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


def test_idempotency_key_is_bound_to_turn_tool_and_arguments() -> None:
    gateway = ToolGateway.with_builtin_demo_tools()
    gateway.propose(
        name="start_breathing_exercise",
        arguments={"duration_seconds": 60},
        risk_level=RiskLevel.GREEN,
        turn_id="turn_1",
        idempotency_key="idem_bound",
    )

    with pytest.raises(ToolPolicyError, match="bound"):
        gateway.propose(
            name="start_breathing_exercise",
            arguments={"duration_seconds": 90},
            risk_level=RiskLevel.GREEN,
            turn_id="turn_1",
            idempotency_key="idem_bound",
        )


def test_gateway_enforces_control_plane_allowlist_and_turn_budget() -> None:
    gateway = ToolGateway.with_builtin_demo_tools()

    with pytest.raises(ToolPolicyError, match="not allowed"):
        gateway.propose(
            name="delete_user_memory",
            arguments={"memory_id": "memory_1"},
            risk_level=RiskLevel.GREEN,
            turn_id="turn_1",
            directive_allowlist={"start_breathing_exercise"},
        )

    with pytest.raises(ToolPolicyError, match="budget"):
        gateway.propose(
            name="start_breathing_exercise",
            arguments={"duration_seconds": 60},
            risk_level=RiskLevel.GREEN,
            turn_id="turn_2",
            directive_allowlist={"start_breathing_exercise"},
            max_tool_calls=0,
        )


def test_untrusted_or_unapproved_high_impact_mcp_capability_cannot_register() -> None:
    registry = CapabilityRegistry(definitions=[], trusted_mcp_servers={"trusted.mcp"})
    base = CapabilityDefinition(
        name="mcp.send_message",
        origin=CapabilityOrigin.MCP,
        source_id="untrusted.mcp",
        trusted=True,
        impact=CapabilityImpact.EXTERNAL_COMMUNICATION,
        approval_role=ApprovalRole.USER,
        allowed_risk_levels=[RiskLevel.GREEN],
    )

    with pytest.raises(CapabilityRegistrationError, match="not host-trusted"):
        registry.register(base)
    with pytest.raises(CapabilityRegistrationError, match="approval"):
        registry.register(
            base.model_copy(
                update={
                    "source_id": "trusted.mcp",
                    "approval_role": ApprovalRole.NONE,
                }
            )
        )
