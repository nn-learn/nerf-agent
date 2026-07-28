from app.safety.models import RiskLevel


class ToolPolicy:
    _known_tools = {
        "start_breathing_exercise",
        "show_reviewed_resource",
        "request_clinician_handoff",
        "draft_emergency_contact_message",
        "export_user_memories",
        "delete_user_memory",
    }
    _approval_required = {
        "draft_emergency_contact_message",
        "export_user_memories",
        "delete_user_memory",
    }

    def is_known(self, name: str) -> bool:
        return name in self._known_tools

    def requires_approval(self, name: str) -> bool:
        return name in self._approval_required

    def is_allowed(self, name: str, risk_level: RiskLevel) -> bool:
        if risk_level in {RiskLevel.RED, RiskLevel.EMERGENCY}:
            return name in {
                "request_clinician_handoff",
                "draft_emergency_contact_message",
                "show_reviewed_resource",
            }
        return self.is_known(name)
