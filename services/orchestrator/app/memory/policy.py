from app.memory.models import MemoryCandidate, MemoryDecision, MemoryKind


class MemoryPolicy:
    def evaluate(
        self,
        candidate: MemoryCandidate,
        *,
        consent_granted: bool,
    ) -> MemoryDecision:
        if candidate.kind is MemoryKind.SAFETY:
            return MemoryDecision.REJECT
        if candidate.source in {"mcp", "external", "rag"}:
            return MemoryDecision.REJECT
        if candidate.source == "visual" and candidate.contains_sensitive_content:
            return MemoryDecision.REJECT
        if {"prompt_injection", "direct_identifier"} & set(
            candidate.integrity_flags
        ):
            return MemoryDecision.REJECT
        if candidate.contains_sensitive_content and not candidate.user_confirmed:
            return MemoryDecision.REQUEST_CONSENT
        if not consent_granted:
            return MemoryDecision.REQUEST_CONSENT
        return MemoryDecision.ACCEPT
