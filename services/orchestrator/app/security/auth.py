from urllib.parse import urlsplit

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


def websocket_origin_allowed(
    origin: str | None,
    *,
    configured_origin: str,
) -> bool:
    if origin is None:
        return False
    try:
        actual = urlsplit(origin)
        configured = urlsplit(configured_origin)
        actual_port = actual.port
        configured_port = configured.port
    except ValueError:
        return False
    return (
        actual.scheme.lower() == configured.scheme.lower()
        and (
            actual.hostname == configured.hostname
            or (
                _is_loopback_host(actual.hostname)
                and _is_loopback_host(configured.hostname)
            )
        )
        and actual_port == configured_port
        and actual.path.rstrip("/") == configured.path.rstrip("/")
        and actual.query == configured.query
        and actual.fragment == configured.fragment
    )


def websocket_transport_allowed(
    *,
    scheme: str,
    request_host: str | None,
    configured_origin: str,
) -> bool:
    if scheme.lower() == "wss":
        return True
    if scheme.lower() != "ws":
        return False
    origin_host = urlsplit(configured_origin).hostname
    return _is_loopback_host(request_host) and _is_loopback_host(origin_host)


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    return host.lower().strip("[]") in {"localhost", "127.0.0.1", "::1"}
