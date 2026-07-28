import asyncio
import sqlite3
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel

from app.events.store import EventStore


class HandoffState(StrEnum):
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    CLOSED = "CLOSED"


class HandoffRecord(BaseModel):
    handoff_id: str
    session_id: str
    risk_assessment_id: str
    idempotency_key: str
    state: HandoffState
    accepted_by: str | None = None


class HandoffPermissionError(PermissionError):
    pass


class HandoffNotFoundError(LookupError):
    pass


class HandoffService:
    """Persisted, auditable demo handoff state machine.

    V1 only simulates the clinician boundary. It never claims that a real clinician
    has been contacted or that emergency services were notified.
    """

    def __init__(self, event_store: EventStore) -> None:
        self._event_store = event_store
        self._lock = asyncio.Lock()

    async def request(
        self,
        *,
        session_id: str,
        risk_assessment_id: str,
        idempotency_key: str,
    ) -> HandoffRecord:
        async with self._lock:
            existing = self._find_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
            record = HandoffRecord(
                handoff_id=f"handoff_{uuid4().hex}",
                session_id=session_id,
                risk_assessment_id=risk_assessment_id,
                idempotency_key=idempotency_key,
                state=HandoffState.REQUESTED,
            )
            with sqlite3.connect(self._event_store.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO handoffs(
                        handoff_id, session_id, risk_assessment_id,
                        idempotency_key, state, accepted_by
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.handoff_id,
                        record.session_id,
                        record.risk_assessment_id,
                        record.idempotency_key,
                        record.state.value,
                        record.accepted_by,
                    ),
                )

        await self._event_store.append_payload(
            session_id=record.session_id,
            event_type="handoff.requested",
            payload={
                "handoff_id": record.handoff_id,
                "risk_assessment_id": record.risk_assessment_id,
                "state": record.state.value,
                "simulated": True,
            },
        )
        return record

    async def accept(
        self,
        handoff_id: str,
        *,
        actor_role: str,
    ) -> HandoffRecord:
        if actor_role != "CLINICIAN_DEMO":
            raise HandoffPermissionError(
                "handoff acceptance requires CLINICIAN_DEMO role"
            )

        async with self._lock:
            record = self._find_by_id(handoff_id)
            if record is None:
                raise HandoffNotFoundError(handoff_id)
            if record.state is HandoffState.ACCEPTED:
                return record
            accepted = record.model_copy(
                update={
                    "state": HandoffState.ACCEPTED,
                    "accepted_by": actor_role,
                }
            )
            with sqlite3.connect(self._event_store.database_path) as connection:
                connection.execute(
                    """
                    UPDATE handoffs
                    SET state = ?, accepted_by = ?
                    WHERE handoff_id = ?
                    """,
                    (
                        accepted.state.value,
                        accepted.accepted_by,
                        accepted.handoff_id,
                    ),
                )

        await self._event_store.append_payload(
            session_id=accepted.session_id,
            event_type="handoff.accepted",
            payload={
                "handoff_id": accepted.handoff_id,
                "accepted_by": actor_role,
                "state": accepted.state.value,
                "simulated": True,
            },
        )
        return accepted

    def _find_by_idempotency_key(self, idempotency_key: str) -> HandoffRecord | None:
        return self._find_one(
            "SELECT * FROM handoffs WHERE idempotency_key = ?",
            idempotency_key,
        )

    def _find_by_id(self, handoff_id: str) -> HandoffRecord | None:
        return self._find_one(
            "SELECT * FROM handoffs WHERE handoff_id = ?",
            handoff_id,
        )

    def _find_one(self, query: str, value: str) -> HandoffRecord | None:
        with sqlite3.connect(self._event_store.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(query, (value,)).fetchone()
        if row is None:
            return None
        return HandoffRecord(
            handoff_id=row["handoff_id"],
            session_id=row["session_id"],
            risk_assessment_id=row["risk_assessment_id"],
            idempotency_key=row["idempotency_key"],
            state=HandoffState(row["state"]),
            accepted_by=row["accepted_by"],
        )
