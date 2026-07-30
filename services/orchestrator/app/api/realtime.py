from collections.abc import Callable

from fastapi import APIRouter, WebSocket

from app.providers.faster_whisper import FasterWhisperProvider
from app.realtime.session import SessionManager
from app.realtime.vad import WebRtcVad
from app.realtime.websocket import handle_realtime_websocket
from app.settings import Settings


def _default_vad_factory() -> WebRtcVad:
    return WebRtcVad()


def create_realtime_router(
    *,
    settings: Settings,
    manager: SessionManager,
    transcriber: FasterWhisperProvider | None,
    vad_factory: Callable[[], WebRtcVad] = _default_vad_factory,
) -> APIRouter:
    router = APIRouter(tags=["realtime"])

    @router.websocket("/api/sessions/{session_id}/realtime")
    async def realtime(websocket: WebSocket, session_id: str) -> None:
        await handle_realtime_websocket(
            websocket,
            session_id=session_id,
            settings=settings,
            manager=manager,
            transcriber=transcriber,
            vad_factory=vad_factory,
        )

    return router
