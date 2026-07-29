from fastapi import HTTPException, status

from app.realtime.session import SessionManager


def parse_bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="session bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid authorization scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def require_session_access(
    manager: SessionManager,
    *,
    session_id: str,
    authorization: str | None,
) -> None:
    token = parse_bearer_token(authorization)
    if not await manager.has_access(session_id, token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="session access denied",
        )
