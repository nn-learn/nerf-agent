from dataclasses import dataclass
from typing import Protocol

from app.agent.avatar import AvatarResponsePlan
from app.realtime.models import TurnHandle


class AvatarUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AvatarRenderResult:
    frame_ids: list[str]


class AvatarClient(Protocol):
    async def start_session(self, session_id: str) -> None: ...

    async def render(
        self,
        turn: TurnHandle,
        *,
        plan: AvatarResponsePlan,
        audio_chunk_count: int,
    ) -> AvatarRenderResult: ...

    async def cancel_turn(self, turn: TurnHandle) -> None: ...

    async def end_session(self, session_id: str) -> None: ...


class MockAvatarClient:
    """CPU-only stand-in for the RAD-NeRF gRPC worker."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self._sessions: set[str] = set()
        self.cancelled_turns: list[str] = []

    async def start_session(self, session_id: str) -> None:
        self._sessions.add(session_id)

    async def render(
        self,
        turn: TurnHandle,
        *,
        plan: AvatarResponsePlan,
        audio_chunk_count: int,
    ) -> AvatarRenderResult:
        _ = (plan, audio_chunk_count)
        if not self.available:
            raise AvatarUnavailable("mock Avatar worker is unavailable")
        if turn.session_id not in self._sessions:
            raise AvatarUnavailable("Avatar session has not been started")
        return AvatarRenderResult(frame_ids=[f"frame_{turn.turn_id}_0"])

    async def cancel_turn(self, turn: TurnHandle) -> None:
        self.cancelled_turns.append(turn.turn_id)

    async def end_session(self, session_id: str) -> None:
        self._sessions.discard(session_id)
