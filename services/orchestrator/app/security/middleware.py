import asyncio
import time
from collections import defaultdict, deque

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityHeadersMiddleware:
    _headers = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"no-referrer"),
        (
            b"permissions-policy",
            b"camera=(self), microphone=(self), geolocation=()",
        ),
        (
            b"content-security-policy",
            b"default-src 'none'; frame-ancestors 'none'",
        ),
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {name.lower() for name, _ in headers}
                headers.extend(
                    (name, value)
                    for name, value in self._headers
                    if name not in existing
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class InMemoryRateLimitMiddleware:
    """Small single-process limiter for the local demo deployment."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        requests_per_minute: int,
    ) -> None:
        self.app = app
        self._limit = requests_per_minute
        self._requests: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") not in {"POST", "PUT", "PATCH", "DELETE"}
        ):
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        client_host = str(client[0]) if client else "unknown"
        now = time.monotonic()
        async with self._lock:
            timestamps = self._requests[client_host]
            while timestamps and now - timestamps[0] >= 60:
                timestamps.popleft()
            if len(timestamps) >= self._limit:
                response = JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": "60"},
                )
                await response(scope, receive, send)
                return
            timestamps.append(now)
        await self.app(scope, receive, send)
