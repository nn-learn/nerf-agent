import asyncio
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Path, status
from pydantic import BaseModel, Field

from app.events.store import EventStore
from app.memory.consolidation import MemoryConsolidator, MemoryProfileRepository
from app.memory.episodes import MemoryEpisodeRepository
from app.memory.extraction import (
    contains_sensitive_memory_content,
    infer_memory_aspect,
    memory_integrity_flags,
    memory_kind_for_text,
    memory_subject_key,
)
from app.memory.identity import MemorySubjectStore
from app.memory.models import (
    MemoryAllowedUse,
    MemoryAspect,
    MemoryChange,
    MemoryChangeState,
    MemoryConflict,
    MemoryConflictState,
    MemoryDeletionReceipt,
    MemoryEvidenceRelation,
    MemoryItem,
    MemoryProfile,
    MemoryProfileState,
    MemorySensitivity,
    MemorySourceType,
    MemoryState,
)
from app.memory.repository import MemoryRepository
from app.memory.shadow import (
    MemoryShadowRepository,
    MemoryShadowRunner,
    ShadowAggregateReport,
    ShadowRuntimeStatus,
)
from app.memory.worker import MemoryIngestionWorker
from app.realtime.session import SessionManager, SessionNotFoundError, SessionStatus
from app.security.auth import require_session_access

MemoryId = Annotated[str, Path(pattern=r"^memory_[a-f0-9]{32}$")]
ProfileId = Annotated[str, Path(pattern=r"^profile_[a-f0-9]{32}$")]
ConflictId = Annotated[str, Path(pattern=r"^conflict_[a-f0-9]{32}$")]
ChangeId = Annotated[str, Path(pattern=r"^change_[a-f0-9]{32}$")]


class MemoryView(BaseModel):
    memory_id: str
    text: str
    aspect: MemoryAspect
    state: MemoryState
    contains_sensitive_content: bool
    confidence: float = Field(ge=0, le=1)
    purpose_scope: str
    created_at_ms: int
    updated_at_ms: int
    source_turn_id: str
    expires_at_ms: int | None = None
    user_edited: bool = False
    source_type: MemorySourceType
    sensitivity: MemorySensitivity
    allowed_uses: list[MemoryAllowedUse]
    observed_at_ms: int | None = None
    valid_from_ms: int | None = None
    valid_to_ms: int | None = None
    derived_from_memory_ids: list[str]


class MemoryRecallView(BaseModel):
    used_at_ms: int
    score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    reason_codes: list[str]
    turn_id: str


class MemoryUpdateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    retention: Literal["7_days", "30_days", "90_days", "forever"] = "forever"


class MemoryControlRequest(BaseModel):
    action: Literal["pause", "resume"]


class MemoryUsesUpdateRequest(BaseModel):
    allowed_uses: list[MemoryAllowedUse] = Field(min_length=1)


class MemoryExportView(BaseModel):
    schema_version: str = "memory-user-export-1.0"
    exported_at_ms: int
    memories: list[MemoryView]


class MemoryDeletionReceiptView(BaseModel):
    deletion_id: str
    root_memory_id_digest: str
    deleted_memory_count: int
    deleted_derived_count: int
    completed_at_ms: int
    verification_digest: str


class MemoryDecisionRequest(BaseModel):
    decision: Literal["confirm", "reject"]


class MemoryIngestionView(BaseModel):
    state: Literal["idle", "queued", "processing", "complete", "failed", "pending"]
    retryable: bool = False


class MemoryResearchConsentUpdate(BaseModel):
    granted: bool
    acknowledged_policy_version: str = Field(min_length=1, max_length=100)


class MemoryResearchConsentView(BaseModel):
    enabled: bool
    granted: bool
    policy_version: str
    strategy_version: str
    updated_at_ms: int | None = None
    retained_fields: list[str]
    excluded_fields: list[str]


class MemoryShadowReportView(BaseModel):
    enabled: bool
    consent_granted: bool
    aggregate: ShadowAggregateReport
    runtime: ShadowRuntimeStatus | None = None


class MemoryProfileDecisionRequest(BaseModel):
    decision: Literal["confirm", "reject"]
    statement: str | None = Field(default=None, min_length=1, max_length=500)


class MemoryConflictDecisionRequest(BaseModel):
    decision: Literal["select", "dismiss"]
    profile_id: str | None = Field(
        default=None,
        pattern=r"^profile_[a-f0-9]{32}$",
    )


class MemoryChangeDecisionRequest(BaseModel):
    decision: Literal["apply", "reject"]


class MemoryProfileEvidenceView(BaseModel):
    memory_id: str
    text: str
    relation: MemoryEvidenceRelation
    valid_at_ms: int
    observed_at_ms: int


class MemoryProfileView(BaseModel):
    profile_id: str
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
    valid_from_ms: int | None = None
    valid_to_ms: int | None = None
    expires_at_ms: int | None = None
    created_at_ms: int
    updated_at_ms: int
    user_edited: bool
    evidence: list[MemoryProfileEvidenceView]


class MemoryConflictView(BaseModel):
    conflict_id: str
    subject_key: str
    state: MemoryConflictState
    selected_profile_id: str | None = None
    created_at_ms: int
    updated_at_ms: int
    options: list[MemoryProfileView]


class MemoryChangeView(BaseModel):
    change_id: str
    subject_key: str
    state: MemoryChangeState
    effective_at_ms: int
    observed_at_ms: int
    created_at_ms: int
    updated_at_ms: int
    previous_profile: MemoryProfileView
    proposed_profile: MemoryProfileView


def create_memory_router(
    *,
    manager: SessionManager,
    repository: MemoryRepository,
    subjects: MemorySubjectStore,
    store: EventStore,
    worker: MemoryIngestionWorker,
    profiles: MemoryProfileRepository,
    consolidator: MemoryConsolidator,
    episodes: MemoryEpisodeRepository,
    shadow_repository: MemoryShadowRepository,
    shadow_runner: MemoryShadowRunner | None,
    shadow_policy_version: str,
    shadow_strategy_version: str,
) -> APIRouter:
    router = APIRouter(prefix="/api/sessions", tags=["memory"])

    async def current_user(session_id: str, authorization: str | None) -> str:
        try:
            await require_session_access(
                manager,
                session_id=session_id,
                authorization=authorization,
            )
            return subjects.user_for_session(session_id)
        except SessionNotFoundError as error:
            raise HTTPException(status_code=404, detail="session not found") from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory subject not found") from error

    @router.get("/{session_id}/memories", response_model=list[MemoryView])
    async def list_memories(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemoryView]:
        user_id = await current_user(session_id, authorization)
        return [_view(item) for item in repository.list_visible(user_id)]

    @router.get(
        "/{session_id}/memories/export",
        response_model=MemoryExportView,
    )
    async def export_memories(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> MemoryExportView:
        user_id = await current_user(session_id, authorization)
        return MemoryExportView(
            exported_at_ms=int(time.time() * 1000),
            memories=[_view(item) for item in repository.list_visible(user_id)],
        )

    @router.get(
        "/{session_id}/memory-profiles",
        response_model=list[MemoryProfileView],
    )
    async def list_memory_profiles(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemoryProfileView]:
        user_id = await current_user(session_id, authorization)
        change_proposals = profiles.open_change_proposal_ids(user_id)
        return [
            _profile_view(profile)
            for profile in profiles.list_profiles(
                user_id,
                states={
                    MemoryProfileState.AWAITING_CONFIRMATION,
                    MemoryProfileState.ACTIVE,
                },
            )
            if profile.profile_id not in change_proposals
        ]

    @router.post(
        "/{session_id}/memory-profiles/rebuild",
        response_model=list[MemoryProfileView],
    )
    async def rebuild_memory_profiles(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemoryProfileView]:
        user_id = await current_user(session_id, authorization)
        await asyncio.to_thread(consolidator.rebuild_user, user_id)
        change_proposals = profiles.open_change_proposal_ids(user_id)
        return [
            _profile_view(profile)
            for profile in profiles.list_profiles(
                user_id,
                states={
                    MemoryProfileState.AWAITING_CONFIRMATION,
                    MemoryProfileState.ACTIVE,
                },
            )
            if profile.profile_id not in change_proposals
        ]

    @router.post(
        "/{session_id}/memory-profiles/{profile_id}/decision",
        response_model=MemoryProfileView | None,
    )
    async def decide_memory_profile(
        session_id: str,
        profile_id: ProfileId,
        request: MemoryProfileDecisionRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryProfileView | None:
        user_id = await current_user(session_id, authorization)
        if request.statement is not None:
            normalized = " ".join(request.statement.split()).strip()
            if memory_integrity_flags(normalized):
                raise HTTPException(
                    status_code=422,
                    detail="profile edit contains forbidden instructions or identifiers",
                )
            if memory_kind_for_text(normalized).value == "SAFETY":
                raise HTTPException(
                    status_code=422,
                    detail="safety-critical content cannot be a memory profile",
                )
        try:
            if request.decision == "confirm":
                profile = profiles.confirm_profile(
                    profile_id,
                    user_id=user_id,
                    statement=request.statement,
                )
                event_type = "memory.profile.confirmed"
            else:
                profiles.reject_profile(profile_id, user_id=user_id)
                profile = None
                event_type = "memory.profile.rejected"
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory profile not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={"profile_id": profile_id},
        )
        return _profile_view(profile) if profile is not None else None

    @router.get(
        "/{session_id}/memory-conflicts",
        response_model=list[MemoryConflictView],
    )
    async def list_memory_conflicts(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemoryConflictView]:
        user_id = await current_user(session_id, authorization)
        return [
            _conflict_view(conflict)
            for conflict in profiles.list_conflicts(
                user_id,
                state=MemoryConflictState.OPEN,
            )
        ]

    @router.post(
        "/{session_id}/memory-conflicts/{conflict_id}/decision",
        response_model=MemoryConflictView,
    )
    async def decide_memory_conflict(
        session_id: str,
        conflict_id: ConflictId,
        request: MemoryConflictDecisionRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryConflictView:
        user_id = await current_user(session_id, authorization)
        try:
            conflict = profiles.get_conflict(conflict_id, user_id=user_id)
            if conflict.state is not MemoryConflictState.OPEN:
                raise ValueError("memory conflict is already closed")
            if request.decision == "select":
                if request.profile_id is None:
                    raise ValueError("profile_id is required when selecting")
                if request.profile_id not in {
                    option.profile_id for option in conflict.options
                }:
                    raise ValueError("selected profile is not a conflict option")
                profiles.confirm_profile(request.profile_id, user_id=user_id)
                event_type = "memory.conflict.resolved"
            else:
                profiles.dismiss_conflict(conflict_id, user_id=user_id)
                event_type = "memory.conflict.dismissed"
            resolved = profiles.get_conflict(conflict_id, user_id=user_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory conflict not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={
                "conflict_id": conflict_id,
                **(
                    {"profile_id": request.profile_id}
                    if request.profile_id is not None
                    else {}
                ),
            },
        )
        return _conflict_view(resolved)

    @router.get(
        "/{session_id}/memory-changes",
        response_model=list[MemoryChangeView],
    )
    async def list_memory_changes(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> list[MemoryChangeView]:
        user_id = await current_user(session_id, authorization)
        return [
            _change_view(change)
            for change in profiles.list_changes(
                user_id,
                state=MemoryChangeState.OPEN,
            )
        ]

    @router.post(
        "/{session_id}/memory-changes/{change_id}/decision",
        response_model=MemoryChangeView,
    )
    async def decide_memory_change(
        session_id: str,
        change_id: ChangeId,
        request: MemoryChangeDecisionRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryChangeView:
        user_id = await current_user(session_id, authorization)
        try:
            if request.decision == "apply":
                changed = profiles.apply_change(change_id, user_id=user_id)
                event_type = "memory.change.applied"
            else:
                changed = profiles.reject_change(change_id, user_id=user_id)
                event_type = "memory.change.rejected"
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory change not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={"change_id": change_id},
        )
        return _change_view(changed)

    @router.get(
        "/{session_id}/memories/status",
        response_model=MemoryIngestionView,
    )
    async def memory_status(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> MemoryIngestionView:
        await current_user(session_id, authorization)
        descriptor = await manager.get_session(session_id)
        if descriptor.status is SessionStatus.ACTIVE:
            return MemoryIngestionView(state="idle")
        worker_state = worker.session_state(session_id)
        if worker_state in {"queued", "processing", "failed"}:
            return MemoryIngestionView(
                state=worker_state,
                retryable=worker_state == "failed",
            )
        if subjects.ingestion_complete(session_id):
            return MemoryIngestionView(state="complete")
        return MemoryIngestionView(state="pending", retryable=True)

    @router.get(
        "/{session_id}/memories/research-consent",
        response_model=MemoryResearchConsentView,
    )
    async def research_consent(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> MemoryResearchConsentView:
        user_id = await current_user(session_id, authorization)
        consent = shadow_repository.consent_for_user(user_id)
        granted = bool(
            consent is not None
            and consent.granted
            and consent.policy_version == shadow_policy_version
        )
        return _research_consent_view(
            enabled=shadow_runner is not None,
            granted=granted,
            policy_version=shadow_policy_version,
            strategy_version=shadow_strategy_version,
            updated_at_ms=consent.updated_at_ms if consent is not None else None,
        )

    @router.put(
        "/{session_id}/memories/research-consent",
        response_model=MemoryResearchConsentView,
    )
    async def update_research_consent(
        session_id: str,
        request: MemoryResearchConsentUpdate,
        authorization: str | None = Header(default=None),
    ) -> MemoryResearchConsentView:
        user_id = await current_user(session_id, authorization)
        if request.granted and shadow_runner is None:
            raise HTTPException(
                status_code=409,
                detail="memory shadow evaluation is disabled",
            )
        if (
            request.granted
            and request.acknowledged_policy_version != shadow_policy_version
        ):
            raise HTTPException(
                status_code=409,
                detail="research policy version changed; review it before consenting",
            )
        if not request.granted and shadow_runner is not None:
            shadow_runner.cancel_user(user_id)
        consent = shadow_repository.set_consent(
            user_id=user_id,
            granted=request.granted,
            policy_version=shadow_policy_version,
        )
        if request.granted and shadow_runner is not None:
            shadow_runner.start_prewarm()
        await store.append_payload(
            session_id=session_id,
            event_type=(
                "memory.shadow_consent.granted"
                if request.granted
                else "memory.shadow_consent.revoked"
            ),
            payload={"policy_version": shadow_policy_version},
        )
        return _research_consent_view(
            enabled=shadow_runner is not None,
            granted=consent.granted,
            policy_version=shadow_policy_version,
            strategy_version=shadow_strategy_version,
            updated_at_ms=consent.updated_at_ms,
        )

    @router.get(
        "/{session_id}/memories/shadow-report",
        response_model=MemoryShadowReportView,
    )
    async def shadow_report(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> MemoryShadowReportView:
        user_id = await current_user(session_id, authorization)
        return MemoryShadowReportView(
            enabled=shadow_runner is not None,
            consent_granted=shadow_repository.is_granted(
                user_id,
                policy_version=shadow_policy_version,
            ),
            aggregate=shadow_repository.aggregate(user_id),
            runtime=shadow_runner.status() if shadow_runner is not None else None,
        )

    @router.post(
        "/{session_id}/memories/retry",
        response_model=MemoryIngestionView,
    )
    async def retry_memory_ingestion(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> MemoryIngestionView:
        await current_user(session_id, authorization)
        descriptor = await manager.get_session(session_id)
        if descriptor.status is not SessionStatus.ENDED:
            raise HTTPException(
                status_code=409,
                detail="memory extraction starts after session end",
            )
        await worker.enqueue(session_id)
        state = worker.session_state(session_id)
        return MemoryIngestionView(
            state=state if state in {"queued", "processing"} else "pending",
            retryable=state is None,
        )

    @router.post(
        "/{session_id}/memories/{memory_id}/decision",
        response_model=MemoryView,
    )
    async def decide_memory(
        session_id: str,
        memory_id: MemoryId,
        request: MemoryDecisionRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryView:
        user_id = await current_user(session_id, authorization)
        try:
            if request.decision == "confirm":
                item = repository.activate(memory_id, user_id=user_id)
                profiles.record_observation(
                    item,
                    session_id=session_id,
                    resolve_source_session=True,
                )
                await asyncio.to_thread(consolidator.rebuild_user, user_id)
                await asyncio.to_thread(
                    episodes.rebuild_for_memory,
                    user_id=user_id,
                    memory_id=item.memory_id,
                    memory_repository=repository,
                )
                event_type = "memory.confirmed"
            else:
                item = repository.get(memory_id, user_id=user_id)
                if item.state is not MemoryState.AWAITING_CONSENT:
                    raise ValueError("only awaiting-consent memory can be rejected")
                if not repository.purge(memory_id, user_id=user_id):
                    raise KeyError(memory_id)
                item = item.model_copy(
                    update={"state": MemoryState.REJECTED}
                )
                event_type = "memory.rejected"
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await store.append_payload(
            session_id=session_id,
            event_type=event_type,
            payload={"memory_id": memory_id},
        )
        return _view(item)

    @router.put(
        "/{session_id}/memories/{memory_id}",
        response_model=MemoryView,
    )
    async def update_memory(
        session_id: str,
        memory_id: MemoryId,
        request: MemoryUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryView:
        user_id = await current_user(session_id, authorization)
        text = " ".join(request.text.split()).strip()
        flags = memory_integrity_flags(text)
        if flags:
            raise HTTPException(
                status_code=422,
                detail="memory edit contains forbidden instructions or identifiers",
            )
        if memory_kind_for_text(text).value == "SAFETY":
            raise HTTPException(
                status_code=422,
                detail="safety-critical content cannot be personalization memory",
            )
        now_ms = int(time.time() * 1000)
        expiry_by_retention = {
            "7_days": now_ms + 7 * 24 * 60 * 60 * 1000,
            "30_days": now_ms + 30 * 24 * 60 * 60 * 1000,
            "90_days": now_ms + 90 * 24 * 60 * 60 * 1000,
            "forever": None,
        }
        aspect = infer_memory_aspect(text)
        try:
            item = repository.update_user_memory(
                memory_id,
                user_id=user_id,
                text=text,
                expires_at_ms=expiry_by_retention[request.retention],
                contains_sensitive_content=contains_sensitive_memory_content(text),
                aspect=aspect,
                subject_key=memory_subject_key(aspect, text),
                now_ms=now_ms,
            )
            if item.state is MemoryState.ACTIVE:
                profiles.record_observation(
                    item,
                    session_id=session_id,
                    now_ms=now_ms,
                )
                await asyncio.to_thread(
                    consolidator.rebuild_user,
                    user_id,
                    now_ms=now_ms,
                )
                await asyncio.to_thread(
                    episodes.rebuild_for_memory,
                    user_id=user_id,
                    memory_id=item.memory_id,
                    memory_repository=repository,
                    now_ms=now_ms,
                )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await store.append_payload(
            session_id=session_id,
            event_type="memory.updated",
            payload={
                "memory_id": memory_id,
                "retention": request.retention,
            },
        )
        return _view(item)

    @router.post(
        "/{session_id}/memories/{memory_id}/control",
        response_model=MemoryView,
    )
    async def control_memory(
        session_id: str,
        memory_id: MemoryId,
        request: MemoryControlRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryView:
        user_id = await current_user(session_id, authorization)
        try:
            items = repository.set_paused(
                memory_id,
                user_id=user_id,
                paused=request.action == "pause",
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        item = next(item for item in items if item.memory_id == memory_id)
        await asyncio.to_thread(consolidator.rebuild_user, user_id)
        if request.action == "resume":
            for resumed_item in items:
                if resumed_item.state is MemoryState.ACTIVE:
                    await asyncio.to_thread(
                        episodes.rebuild_for_memory,
                        user_id=user_id,
                        memory_id=resumed_item.memory_id,
                        memory_repository=repository,
                    )
        await store.append_payload(
            session_id=session_id,
            event_type=f"memory.{request.action}d",
            payload={"memory_id": memory_id},
        )
        return _view(item)

    @router.put(
        "/{session_id}/memories/{memory_id}/uses",
        response_model=MemoryView,
    )
    async def update_memory_uses(
        session_id: str,
        memory_id: MemoryId,
        request: MemoryUsesUpdateRequest,
        authorization: str | None = Header(default=None),
    ) -> MemoryView:
        user_id = await current_user(session_id, authorization)
        try:
            item = repository.update_allowed_uses(
                memory_id,
                user_id=user_id,
                allowed_uses=request.allowed_uses,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory not found") from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        await asyncio.to_thread(consolidator.rebuild_user, user_id)
        if MemoryAllowedUse.PERSONALIZATION in request.allowed_uses:
            await asyncio.to_thread(
                episodes.rebuild_for_memory,
                user_id=user_id,
                memory_id=item.memory_id,
                memory_repository=repository,
            )
        await store.append_payload(
            session_id=session_id,
            event_type="memory.uses_updated",
            payload={
                "memory_id": memory_id,
                "allowed_uses": [allowed.value for allowed in request.allowed_uses],
            },
        )
        return _view(item)

    @router.post(
        "/{session_id}/memories/{memory_id}/forget",
        response_model=MemoryDeletionReceiptView,
    )
    async def forget_memory(
        session_id: str,
        memory_id: MemoryId,
        authorization: str | None = Header(default=None),
    ) -> MemoryDeletionReceiptView:
        user_id = await current_user(session_id, authorization)
        receipt = repository.purge_with_receipt(memory_id, user_id=user_id)
        if receipt is None:
            raise HTTPException(status_code=404, detail="memory not found")
        await asyncio.to_thread(consolidator.rebuild_user, user_id)
        await store.append_payload(
            session_id=session_id,
            event_type="memory.forgotten",
            payload={
                "deletion_id": receipt.deletion_id,
                "deleted_memory_count": receipt.deleted_memory_count,
                "deleted_derived_count": receipt.deleted_derived_count,
            },
        )
        return _deletion_receipt_view(receipt)

    @router.get(
        "/{session_id}/memories/{memory_id}/latest-recall",
        response_model=MemoryRecallView | None,
    )
    async def latest_memory_recall(
        session_id: str,
        memory_id: MemoryId,
        authorization: str | None = Header(default=None),
    ) -> MemoryRecallView | None:
        user_id = await current_user(session_id, authorization)
        try:
            repository.get(memory_id, user_id=user_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="memory not found") from error
        recall = repository.latest_recall(memory_id, user_id=user_id)
        if recall is None:
            return None
        return MemoryRecallView(
            used_at_ms=recall.used_at_ms,
            score=recall.score,
            relevance_score=recall.relevance_score,
            reason_codes=recall.reason_codes,
            turn_id=recall.turn_id,
        )

    @router.delete(
        "/{session_id}/memories/{memory_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_memory(
        session_id: str,
        memory_id: MemoryId,
        authorization: str | None = Header(default=None),
    ) -> None:
        user_id = await current_user(session_id, authorization)
        if not repository.purge(memory_id, user_id=user_id):
            raise HTTPException(status_code=404, detail="memory not found")
        await asyncio.to_thread(consolidator.rebuild_user, user_id)
        await store.append_payload(
            session_id=session_id,
            event_type="memory.deleted",
            payload={"memory_id": memory_id},
        )

    return router


def _view(item: MemoryItem) -> MemoryView:
    return MemoryView(
        memory_id=item.memory_id,
        text=item.candidate.text,
        aspect=item.candidate.aspect,
        state=item.state,
        contains_sensitive_content=item.candidate.contains_sensitive_content,
        confidence=item.candidate.confidence,
        purpose_scope=item.candidate.purpose_scope,
        created_at_ms=item.created_at_ms,
        updated_at_ms=item.updated_at_ms,
        source_turn_id=item.candidate.source_turn_id,
        expires_at_ms=item.candidate.expires_at_ms,
        user_edited=item.candidate.user_edited,
        source_type=item.candidate.source_type,
        sensitivity=item.candidate.sensitivity,
        allowed_uses=item.candidate.allowed_uses,
        observed_at_ms=item.candidate.observed_at_ms,
        valid_from_ms=item.candidate.valid_from_ms,
        valid_to_ms=item.candidate.valid_to_ms,
        derived_from_memory_ids=item.candidate.derived_from_memory_ids,
    )


def _deletion_receipt_view(
    receipt: MemoryDeletionReceipt,
) -> MemoryDeletionReceiptView:
    return MemoryDeletionReceiptView(
        deletion_id=receipt.deletion_id,
        root_memory_id_digest=receipt.root_memory_id_digest,
        deleted_memory_count=receipt.deleted_memory_count,
        deleted_derived_count=receipt.deleted_derived_count,
        completed_at_ms=receipt.completed_at_ms,
        verification_digest=receipt.verification_digest,
    )


def _profile_view(profile: MemoryProfile) -> MemoryProfileView:
    return MemoryProfileView(
        profile_id=profile.profile_id,
        subject_key=profile.subject_key,
        aspect=profile.aspect,
        statement=profile.statement,
        state=profile.state,
        confidence=profile.confidence,
        evidence_count=profile.evidence_count,
        supporting_evidence_count=profile.supporting_evidence_count,
        conflicting_evidence_count=profile.conflicting_evidence_count,
        distinct_session_count=profile.distinct_session_count,
        contains_sensitive_content=profile.contains_sensitive_content,
        valid_from_ms=profile.valid_from_ms,
        valid_to_ms=profile.valid_to_ms,
        expires_at_ms=profile.expires_at_ms,
        created_at_ms=profile.created_at_ms,
        updated_at_ms=profile.updated_at_ms,
        user_edited=profile.user_edited,
        evidence=[
            MemoryProfileEvidenceView(
                memory_id=evidence.memory_id,
                text=evidence.text,
                relation=evidence.relation,
                valid_at_ms=evidence.valid_at_ms,
                observed_at_ms=evidence.observed_at_ms,
            )
            for evidence in profile.evidence
        ],
    )


def _conflict_view(conflict: MemoryConflict) -> MemoryConflictView:
    return MemoryConflictView(
        conflict_id=conflict.conflict_id,
        subject_key=conflict.subject_key,
        state=conflict.state,
        selected_profile_id=conflict.selected_profile_id,
        created_at_ms=conflict.created_at_ms,
        updated_at_ms=conflict.updated_at_ms,
        options=[_profile_view(profile) for profile in conflict.options],
    )


def _change_view(change: MemoryChange) -> MemoryChangeView:
    return MemoryChangeView(
        change_id=change.change_id,
        subject_key=change.subject_key,
        state=change.state,
        effective_at_ms=change.effective_at_ms,
        observed_at_ms=change.observed_at_ms,
        created_at_ms=change.created_at_ms,
        updated_at_ms=change.updated_at_ms,
        previous_profile=_profile_view(change.previous_profile),
        proposed_profile=_profile_view(change.proposed_profile),
    )


def _research_consent_view(
    *,
    enabled: bool,
    granted: bool,
    policy_version: str,
    strategy_version: str,
    updated_at_ms: int | None,
) -> MemoryResearchConsentView:
    return MemoryResearchConsentView(
        enabled=enabled,
        granted=granted,
        policy_version=policy_version,
        strategy_version=strategy_version,
        updated_at_ms=updated_at_ms,
        retained_fields=[
            "session/turn/memory pseudonymous identifiers",
            "baseline and shadow ranks/scores",
            "latency, overlap, status, and version",
        ],
        excluded_fields=["query text", "memory text", "audio/video", "embedding"],
    )
