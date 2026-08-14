import json
import tempfile
import time
from pathlib import Path

from app.memory.extraction import RuleBasedMemoryExtractor
from app.memory.models import MemoryMessage, MessageRole
from app.memory.pipeline import MemoryPipeline
from app.memory.repository import MemoryRepository


def build_messages(count: int) -> list[MemoryMessage]:
    messages: list[MemoryMessage] = []
    for index in range(count):
        role = MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT
        if role is MessageRole.USER and index % 20 == 0:
            text = f"请记住我在第{index // 20}周完成了一次散步"
        elif role is MessageRole.USER:
            text = f"这是第{index}条关于工作压力和睡眠的普通消息"
        else:
            text = "我听到了，我们继续梳理这件事。"
        messages.append(
            MemoryMessage(
                message_id=f"message_{index}",
                turn_id=f"turn_{index // 2}",
                role=role,
                text=text,
                sequence=index,
                timestamp_ms=1_700_000_000_000 + index * 1_000,
            )
        )
    return messages


def run(count: int = 10_000) -> dict[str, int | float]:
    with tempfile.TemporaryDirectory(prefix="psyavatar-memory-") as directory:
        repository = MemoryRepository(Path(directory) / "memory.sqlite3")
        repository.initialize()
        messages = build_messages(count)
        started = time.perf_counter()
        result = MemoryPipeline(
            repository=repository,
            extractor=RuleBasedMemoryExtractor(),
        ).run(
            user_id="benchmark_user",
            messages=messages,
            consent_granted=True,
            now_ms=1_700_100_000_000,
        )
        elapsed = time.perf_counter() - started
        return {
            "messages": count,
            "windows": result.metrics.window_count,
            "extraction_batches": result.metrics.extraction_batch_count,
            "candidates": result.metrics.candidate_count,
            "active_rows": len(
                repository.list_active(
                    "benchmark_user",
                    as_of_ms=1_700_100_000_001,
                )
            ),
            "context_token_ratio": round(result.metrics.context_token_ratio, 4),
            "elapsed_ms": round(elapsed * 1_000, 2),
            "messages_per_second": round(count / elapsed, 2),
        }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
