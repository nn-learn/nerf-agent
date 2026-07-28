import asyncio
from uuid import uuid4


class CancellationRegistry:
    def __init__(self) -> None:
        self._current: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def issue(self, turn_id: str) -> str:
        token = f"ct_{uuid4().hex}"
        async with self._lock:
            self._current[turn_id] = token
        return token

    async def cancel(self, token: str) -> None:
        async with self._lock:
            cancelled_turns = [
                turn_id
                for turn_id, current_token in self._current.items()
                if current_token == token
            ]
            for turn_id in cancelled_turns:
                del self._current[turn_id]

    async def is_current(self, turn_id: str, token: str) -> bool:
        async with self._lock:
            return self._current.get(turn_id) == token
