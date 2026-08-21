from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class MemoryKind(StrEnum):
    WORKING = "WORKING"
    SEMANTIC = "SEMANTIC"
    EPISODIC = "EPISODIC"
    SAFETY = "SAFETY"


class MemoryAspect(StrEnum):
    """The user-facing meaning of a memory, independent of its storage tier."""

    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    COPING_STRATEGY = "COPING_STRATEGY"
    BOUNDARY = "BOUNDARY"


class MemorySourceType(StrEnum):
    """Auditable origin of a claim, independent of its transport name."""

    USER_STATEMENT = "USER_STATEMENT"
    USER_EDIT = "USER_EDIT"
    MODEL_EXTRACTION = "MODEL_EXTRACTION"
    EXTERNAL_TOOL = "EXTERNAL_TOOL"
    VISUAL_OBSERVATION = "VISUAL_OBSERVATION"
    LEGACY = "LEGACY"


class MemorySensitivity(StrEnum):
    GENERAL = "GENERAL"
    PERSONAL = "PERSONAL"
    HEALTH_SENSITIVE = "HEALTH_SENSITIVE"
    CRISIS_SENSITIVE = "CRISIS_SENSITIVE"


class MemoryAllowedUse(StrEnum):
    PERSONALIZATION = "PERSONALIZATION"
    RESPONSE_CONTEXT = "RESPONSE_CONTEXT"
    SAFETY_SUPPORT = "SAFETY_SUPPORT"
    AGGREGATE_ANALYTICS = "AGGREGATE_ANALYTICS"


class MessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"


class MemoryDecision(StrEnum):
    ACCEPT = "ACCEPT"
    REQUEST_CONSENT = "REQUEST_CONSENT"
    REJECT = "REJECT"


class MemoryState(StrEnum):
    CANDIDATE = "CANDIDATE"
    AWAITING_CONSENT = "AWAITING_CONSENT"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"
    QUARANTINED = "QUARANTINED"


class MemoryProfileState(StrEnum):
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    ACTIVE = "ACTIVE"
    STALE = "STALE"
    HISTORICAL = "HISTORICAL"
    REJECTED = "REJECTED"


class MemoryConflictState(StrEnum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class MemoryEvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"


class MemoryChangeState(StrEnum):
    OPEN = "OPEN"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"


class MemoryMessage(BaseModel):
    """A normalized message read from the event log."""

    message_id: str = Field(min_length=1)
    turn_id: str = Field(min_length=1)
    role: MessageRole
    text: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    timestamp_ms: int = Field(ge=0)


class MemoryWindow(BaseModel):
    """A semantically bounded extraction unit plus a small read-only context tail."""

    window_id: str = Field(min_length=1)
    messages: list[MemoryMessage] = Field(min_length=1)
    context_messages: list[MemoryMessage] = Field(default_factory=list)
    estimated_tokens: int = Field(ge=1)
    context_tokens: int = Field(default=0, ge=0)


class MemoryCandidate(BaseModel):
    source: str = Field(min_length=1)
    contains_sensitive_content: bool
    text: str = Field(min_length=1)
    kind: MemoryKind
    source_turn_id: str = Field(min_length=1)
    aspect: MemoryAspect = MemoryAspect.FACT
    subject_key: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
    source_message_ids: list[str] = Field(default_factory=list)
    source_window_id: str | None = None
    purpose_scope: str = "personalization"
    valid_from_ms: int | None = Field(default=None, ge=0)
    expires_at_ms: int | None = Field(default=None, ge=0)
    user_confirmed: bool = False
    integrity_flags: list[str] = Field(default_factory=list)
    user_edited: bool = False
    source_type: MemorySourceType = MemorySourceType.MODEL_EXTRACTION
    sensitivity: MemorySensitivity = MemorySensitivity.GENERAL
    allowed_uses: list[MemoryAllowedUse] = Field(
        default_factory=lambda: [
            MemoryAllowedUse.PERSONALIZATION,
            MemoryAllowedUse.RESPONSE_CONTEXT,
        ]
    )
    observed_at_ms: int | None = Field(default=None, ge=0)
    valid_to_ms: int | None = Field(default=None, ge=0)
    derived_from_memory_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ledger_contract(self) -> "MemoryCandidate":
        if (
            self.valid_from_ms is not None
            and self.valid_to_ms is not None
            and self.valid_to_ms <= self.valid_from_ms
        ):
            raise ValueError("memory valid_to_ms must be after valid_from_ms")
        if len(set(self.allowed_uses)) != len(self.allowed_uses):
            raise ValueError("memory allowed_uses must be unique")
        if not self.allowed_uses:
            raise ValueError("memory must allow at least one governed use")
        if any(not memory_id.strip() for memory_id in self.derived_from_memory_ids):
            raise ValueError("derived memory ids must not be blank")
        if len(set(self.derived_from_memory_ids)) != len(self.derived_from_memory_ids):
            raise ValueError("derived memory ids must be unique")
        return self


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str
    candidate: MemoryCandidate
    state: MemoryState
    created_at_ms: int = Field(default=0, ge=0)
    updated_at_ms: int = Field(default=0, ge=0)
    supersedes_memory_id: str | None = None


class MemoryObservation(BaseModel):
    observation_id: str
    memory_id: str
    user_id: str
    session_id: str
    turn_id: str
    valid_at_ms: int = Field(ge=0)
    observed_at_ms: int = Field(ge=0)


class MemoryProfileEvidence(BaseModel):
    observation_id: str
    memory_id: str
    session_id: str
    turn_id: str
    text: str
    relation: MemoryEvidenceRelation
    valid_at_ms: int = Field(ge=0)
    observed_at_ms: int = Field(ge=0)


class MemoryProfile(BaseModel):
    profile_id: str
    user_id: str
    subject_key: str
    aspect: MemoryAspect
    statement: str
    state: MemoryProfileState
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    supporting_evidence_count: int = Field(ge=0)
    conflicting_evidence_count: int = Field(ge=0)
    distinct_session_count: int = Field(ge=0)
    contains_sensitive_content: bool
    purpose_scope: str = "personalization"
    valid_from_ms: int | None = Field(default=None, ge=0)
    valid_to_ms: int | None = Field(default=None, ge=0)
    expires_at_ms: int | None = Field(default=None, ge=0)
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    profile_version: str
    signature_hash: str
    evidence_digest: str
    user_edited: bool = False
    evidence: list[MemoryProfileEvidence] = Field(default_factory=list)


class MemoryConflict(BaseModel):
    conflict_id: str
    user_id: str
    subject_key: str
    state: MemoryConflictState
    evidence_digest: str
    selected_profile_id: str | None = None
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    options: list[MemoryProfile] = Field(default_factory=list)


class MemoryChange(BaseModel):
    change_id: str
    user_id: str
    subject_key: str
    state: MemoryChangeState
    previous_profile_id: str
    proposed_profile_id: str
    effective_at_ms: int = Field(ge=0)
    observed_at_ms: int = Field(ge=0)
    evidence_digest: str
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)
    previous_profile: MemoryProfile
    proposed_profile: MemoryProfile


class MemoryEpisodeSummary(BaseModel):
    """A bounded retrieval index over governed memories from one episode."""

    summary_id: str
    user_id: str
    session_id: str
    text: str
    purpose_scope: str = "personalization"
    member_memory_ids: list[str] = Field(min_length=1)
    started_at_ms: int = Field(ge=0)
    ended_at_ms: int = Field(ge=0)
    contains_sensitive_content: bool
    source_digest: str
    summary_version: str
    created_at_ms: int = Field(ge=0)
    updated_at_ms: int = Field(ge=0)


class MemoryRecall(BaseModel):
    memory_id: str
    session_id: str
    turn_id: str
    score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    reason_codes: list[str]
    used_at_ms: int = Field(ge=0)


class MemoryDeletionReceipt(BaseModel):
    """Content-free proof that a root memory and its derivatives were purged."""

    deletion_id: str
    user_id: str
    root_memory_id_digest: str
    deleted_memory_count: int = Field(ge=1)
    deleted_derived_count: int = Field(ge=0)
    completed_at_ms: int = Field(ge=0)
    verification_digest: str
