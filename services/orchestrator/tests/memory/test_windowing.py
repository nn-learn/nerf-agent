from app.memory.models import MemoryMessage, MessageRole
from app.memory.windowing import (
    AdaptiveMessageWindower,
    AdaptiveWindowConfig,
)


def message(
    sequence: int,
    role: MessageRole,
    text: str,
    *,
    timestamp_ms: int | None = None,
) -> MemoryMessage:
    return MemoryMessage(
        message_id=f"message_{sequence}",
        turn_id=f"turn_{sequence // 2}",
        role=role,
        text=text,
        sequence=sequence,
        timestamp_ms=timestamp_ms if timestamp_ms is not None else sequence * 1_000,
    )


def test_topic_marker_creates_event_boundary_without_losing_messages() -> None:
    windower = AdaptiveMessageWindower(
        AdaptiveWindowConfig(
            min_tokens=8,
            target_tokens=100,
            max_tokens=150,
            context_tail_tokens=20,
        )
    )
    messages = [
        message(1, MessageRole.USER, "最近工作压力很大，每天都睡不好。"),
        message(2, MessageRole.ASSISTANT, "我们可以先聊聊最近最累的时刻。"),
        message(3, MessageRole.USER, "换个话题，我更喜欢你简短回答。"),
        message(4, MessageRole.ASSISTANT, "好的。"),
    ]

    windows = windower.partition(messages)

    assert [[item.sequence for item in window.messages] for window in windows] == [
        [1, 2],
        [3, 4],
    ]
    assert windows[1].context_messages
    assert all(
        context.message_id not in {item.message_id for item in windows[1].messages}
        for context in windows[1].context_messages
    )


def test_hard_message_limit_keeps_windowing_bounded_and_linear_in_shape() -> None:
    windower = AdaptiveMessageWindower(
        AdaptiveWindowConfig(
            min_tokens=20,
            target_tokens=40,
            max_tokens=60,
            max_messages=4,
            context_tail_tokens=8,
        )
    )
    messages = [
        message(
            index,
            MessageRole.USER if index % 2 else MessageRole.ASSISTANT,
            f"第{index}条普通对话",
        )
        for index in range(1, 101)
    ]

    windows = windower.partition(messages)

    assert sum(len(window.messages) for window in windows) == 100
    assert max(len(window.messages) for window in windows) <= 4
    assert sum(window.context_tokens for window in windows) < sum(
        window.estimated_tokens for window in windows
    )


def test_windows_are_packed_without_exceeding_batch_budget_when_individually_fit() -> None:
    windower = AdaptiveMessageWindower(
        AdaptiveWindowConfig(
            min_tokens=4,
            target_tokens=8,
            max_tokens=16,
            max_messages=2,
            context_tail_tokens=0,
        )
    )
    windows = windower.partition(
        [message(index, MessageRole.USER, "我喜欢简短回答") for index in range(20)]
    )

    batches = windower.pack_batches(windows, max_prompt_tokens=40, max_windows=3)

    assert sum(len(batch.windows) for batch in batches) == len(windows)
    assert all(batch.estimated_prompt_tokens <= 40 for batch in batches)
    assert all(len(batch.windows) <= 3 for batch in batches)


def test_single_oversized_message_is_sentence_split_under_hard_token_limit() -> None:
    windower = AdaptiveMessageWindower(
        AdaptiveWindowConfig(
            min_tokens=8,
            target_tokens=12,
            max_tokens=16,
            context_tail_tokens=0,
        )
    )

    windows = windower.partition(
        [message(1, MessageRole.USER, "这是一段很长的内容。" * 20)]
    )

    assert len(windows) > 1
    assert all(window.estimated_tokens <= 16 for window in windows)
    assert "".join(item.text for window in windows for item in window.messages) == (
        "这是一段很长的内容。" * 20
    )
