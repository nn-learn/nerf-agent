import asyncio
from collections.abc import AsyncIterator, Callable

from app.contracts.session import CancellationRegistry
from app.realtime.models import PcmChunk

PcmSource = Callable[[str, str], AsyncIterator[bytes]]


async def _edge_tts_pcm_source(
    text: str,
    voice: str,
) -> AsyncIterator[bytes]:
    import edge_tts  # type: ignore[import-not-found]

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("FFmpeg pipes were not created")
    stdin = process.stdin
    stdout = process.stdout

    async def feed_mp3() -> None:
        communicator = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate="+0%",
            volume="+0%",
            pitch="+0Hz",
        )
        async for message in communicator.stream():
            if message.get("type") != "audio":
                continue
            data = message.get("data")
            if isinstance(data, bytes):
                stdin.write(data)
                await stdin.drain()
        stdin.close()
        await stdin.wait_closed()

    feeder = asyncio.create_task(feed_mp3())
    try:
        while chunk := await stdout.read(4_096):
            yield chunk
        await feeder
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError("FFmpeg audio conversion failed")
    finally:
        if not feeder.done():
            feeder.cancel()
        if process.returncode is None:
            process.kill()
            await process.wait()


class PcmFramer:
    BYTES_PER_CHUNK = 640

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sequence = 0
        self._base_pts_ms: int | None = None

    def feed(self, data: bytes, *, start_pts_ms: int) -> list[PcmChunk]:
        if self._base_pts_ms is None:
            self._base_pts_ms = start_pts_ms
        self._buffer.extend(data)
        chunks: list[PcmChunk] = []
        while len(self._buffer) >= self.BYTES_PER_CHUNK:
            raw = bytes(self._buffer[: self.BYTES_PER_CHUNK])
            del self._buffer[: self.BYTES_PER_CHUNK]
            chunks.append(
                PcmChunk(
                    sequence=self._sequence,
                    pts_ms=self._base_pts_ms + self._sequence * 20,
                    pcm_s16le=raw,
                )
            )
            self._sequence += 1
        return chunks

    def flush(self) -> list[PcmChunk]:
        if not self._buffer or self._base_pts_ms is None:
            return []
        padding = self.BYTES_PER_CHUNK - len(self._buffer)
        return self.feed(
            bytes(padding),
            start_pts_ms=self._base_pts_ms,
        )


class EdgeTtsProvider:
    def __init__(
        self,
        *,
        registry: CancellationRegistry,
        voice: str = "zh-CN-XiaoxiaoNeural",
        pcm_source: PcmSource = _edge_tts_pcm_source,
    ) -> None:
        self._registry = registry
        self._voice = voice
        self._pcm_source = pcm_source

    async def synthesize(
        self,
        text: str,
        *,
        turn_id: str,
        cancel_token: str,
        start_pts_ms: int,
    ) -> AsyncIterator[PcmChunk]:
        framer = PcmFramer()
        async for pcm_bytes in self._pcm_source(text, self._voice):
            if not await self._registry.is_current(turn_id, cancel_token):
                return
            for chunk in framer.feed(
                pcm_bytes,
                start_pts_ms=start_pts_ms,
            ):
                if not await self._registry.is_current(
                    turn_id,
                    cancel_token,
                ):
                    return
                yield chunk
        if await self._registry.is_current(turn_id, cancel_token):
            for chunk in framer.flush():
                yield chunk
