from typing import Protocol

from app.realtime.models import PcmChunk, TurnHandle


class TurnObserver(Protocol):
    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        raise NotImplementedError

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        raise NotImplementedError


class NullTurnObserver:
    async def on_event(
        self,
        turn: TurnHandle,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        _ = (turn, event_type, payload)

    async def on_audio(
        self,
        turn: TurnHandle,
        chunk: PcmChunk,
    ) -> None:
        _ = (turn, chunk)
