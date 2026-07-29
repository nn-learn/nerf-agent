from datetime import timedelta
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from livekit import api
from pydantic import BaseModel, Field

from app.settings import Settings


class LiveKitTokenRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,80}$")
    participant_name: str = Field(min_length=1, max_length=80)


class LiveKitTokenResponse(BaseModel):
    server_url: str
    room_name: str
    participant_token: str


def create_livekit_router(settings: Settings) -> APIRouter:
    router = APIRouter(prefix="/api/livekit", tags=["livekit"])

    @router.post("/token", response_model=LiveKitTokenResponse)
    async def create_token(request: LiveKitTokenRequest) -> LiveKitTokenResponse:
        if not (
            settings.livekit_url
            and settings.livekit_api_key
            and settings.livekit_api_secret
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LiveKit is not configured",
            )

        room_name = f"psyavatar_{request.session_id}"
        identity = f"user_{uuid4().hex}"
        token = (
            api.AccessToken(
                settings.livekit_api_key,
                settings.livekit_api_secret.get_secret_value(),
            )
            .with_identity(identity)
            .with_name(request.participant_name)
            .with_ttl(timedelta(minutes=10))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_publish_data=True,
                    can_publish_sources=["camera", "microphone"],
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )
        return LiveKitTokenResponse(
            server_url=settings.livekit_url,
            room_name=room_name,
            participant_token=token,
        )

    return router
