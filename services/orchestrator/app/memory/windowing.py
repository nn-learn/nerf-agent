import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from app.memory.models import MemoryMessage, MemoryWindow, MessageRole

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
_SENTENCE_RE = re.compile(r".+?(?:[。！？!?]|$)", re.DOTALL)
_TOPIC_BOUNDARY_MARKERS = (
    "换个话题",
    "说点别的",
    "另外一件事",
    "还有一件事",
    "回到刚才",
    "至于另一个",
)


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic upper-ish estimate suitable for CPU-side scheduling."""
    cjk_count = len(_CJK_RE.findall(text))
    non_cjk = _CJK_RE.sub(" ", text)
    other_count = sum(
        max(1, math.ceil(len(token) / 4))
        for token in _LATIN_TOKEN_RE.findall(non_cjk)
    )
    return max(1, cjk_count + other_count)


@dataclass(frozen=True, slots=True)
class AdaptiveWindowConfig:
    target_tokens: int = 512
    max_tokens: int = 768
    min_tokens: int = 96
    max_messages: int = 32
    context_tail_tokens: int = 96
    time_gap_ms: int = 30 * 60 * 1000

    def __post_init__(self) -> None:
        if self.min_tokens <= 0:
            raise ValueError("min_tokens must be positive")
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("expected min_tokens <= target_tokens <= max_tokens")
        if self.max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if self.context_tail_tokens < 0:
            raise ValueError("context_tail_tokens cannot be negative")


@dataclass(frozen=True, slots=True)
class WindowBatch:
    windows: tuple[MemoryWindow, ...]
    estimated_prompt_tokens: int


class AdaptiveMessageWindower:
    """Linear-time event-aware windowing with bounded non-extractable carry-over."""

    def __init__(self, config: AdaptiveWindowConfig | None = None) -> None:
        self.config = config or AdaptiveWindowConfig()

    def partition(self, messages: Sequence[MemoryMessage]) -> list[MemoryWindow]:
        if not messages:
            return []
        ordered = [
            part
            for message in sorted(
                messages,
                key=lambda item: (item.sequence, item.timestamp_ms),
            )
            for part in self._split_oversized(message)
        ]
        windows: list[MemoryWindow] = []
        current: list[MemoryMessage] = []
        current_tokens = 0

        for message in ordered:
            message_tokens = estimate_tokens(message.text)
            if current and self._should_flush(
                current=current,
                current_tokens=current_tokens,
                incoming=message,
                incoming_tokens=message_tokens,
            ):
                windows.append(self._make_window(windows, current, current_tokens))
                current = []
                current_tokens = 0
            current.append(message)
            current_tokens += message_tokens

        if current:
            windows.append(self._make_window(windows, current, current_tokens))
        return windows

    def pack_batches(
        self,
        windows: Sequence[MemoryWindow],
        *,
        max_prompt_tokens: int = 4096,
        max_windows: int = 8,
    ) -> list[WindowBatch]:
        """Pack independent windows for a future batched local-model extractor."""
        if max_prompt_tokens <= 0 or max_windows <= 0:
            raise ValueError("batch limits must be positive")
        batches: list[WindowBatch] = []
        current: list[MemoryWindow] = []
        current_tokens = 0
        for window in windows:
            tokens = window.estimated_tokens + window.context_tokens
            if current and (
                len(current) >= max_windows
                or current_tokens + tokens > max_prompt_tokens
            ):
                batches.append(WindowBatch(tuple(current), current_tokens))
                current = []
                current_tokens = 0
            current.append(window)
            current_tokens += tokens
        if current:
            batches.append(WindowBatch(tuple(current), current_tokens))
        return batches

    def _should_flush(
        self,
        *,
        current: Sequence[MemoryMessage],
        current_tokens: int,
        incoming: MemoryMessage,
        incoming_tokens: int,
    ) -> bool:
        config = self.config
        if len(current) >= config.max_messages:
            return True
        if current_tokens + incoming_tokens > config.max_tokens:
            return True
        if incoming.timestamp_ms - current[-1].timestamp_ms > config.time_gap_ms:
            return True
        if (
            current_tokens >= config.min_tokens
            and any(marker in incoming.text for marker in _TOPIC_BOUNDARY_MARKERS)
        ):
            return True
        return (
            current_tokens >= config.target_tokens
            and incoming.role is MessageRole.USER
            and current[-1].role is MessageRole.ASSISTANT
        )

    def _make_window(
        self,
        existing: Sequence[MemoryWindow],
        messages: list[MemoryMessage],
        estimated_tokens: int,
    ) -> MemoryWindow:
        context = self._tail_context(existing[-1].messages if existing else [])
        return MemoryWindow(
            window_id=f"window_{len(existing):06d}",
            messages=list(messages),
            context_messages=context,
            estimated_tokens=estimated_tokens,
            context_tokens=sum(estimate_tokens(item.text) for item in context),
        )

    def _tail_context(
        self,
        previous_messages: Iterable[MemoryMessage],
    ) -> list[MemoryMessage]:
        budget = self.config.context_tail_tokens
        if budget == 0:
            return []
        selected: list[MemoryMessage] = []
        used = 0
        for message in reversed(list(previous_messages)):
            tokens = estimate_tokens(message.text)
            if selected and used + tokens > budget:
                break
            if tokens > budget:
                continue
            selected.append(message)
            used += tokens
        selected.reverse()
        return selected

    def _split_oversized(self, message: MemoryMessage) -> list[MemoryMessage]:
        budget = self.config.max_tokens
        if estimate_tokens(message.text) <= budget:
            return [message]
        chunks: list[str] = []
        current = ""
        for sentence in _SENTENCE_RE.findall(message.text):
            if not sentence:
                continue
            if estimate_tokens(sentence) > budget:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_text_to_budget(sentence, budget))
            elif current and estimate_tokens(current + sentence) > budget:
                chunks.append(current)
                current = sentence
            else:
                current += sentence
        if current:
            chunks.append(current)
        return [message.model_copy(update={"text": chunk}) for chunk in chunks]

    @staticmethod
    def _split_text_to_budget(text: str, budget: int) -> list[str]:
        chunks: list[str] = []
        remaining = text
        while remaining:
            low = 1
            high = len(remaining)
            best = 1
            while low <= high:
                middle = (low + high) // 2
                if estimate_tokens(remaining[:middle]) <= budget:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            chunks.append(remaining[:best])
            remaining = remaining[best:]
        return chunks
