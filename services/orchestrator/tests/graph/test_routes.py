from langgraph.checkpoint.sqlite import SqliteSaver

from app.graph.build import GraphDependencies, build_graph
from app.safety.models import RiskLevel


def test_green_turn_fetches_context_before_reply() -> None:
    """Catches normal turns that bypass governed context or output validation."""
    graph = build_graph(GraphDependencies.for_mock())

    result = graph.invoke(
        {
            "transcript": "我最近工作压力有点大",
            "visual_summary": "",
            "visited": [],
        }
    )

    assert result["risk"].level is RiskLevel.GREEN
    assert result["visited"] == [
        "normalize_input",
        "partial_risk",
        "final_risk",
        "context_fetch",
        "reply_planner",
        "output_guard",
        "publish_response",
        "propose_memory",
    ]
    assert result["response"].support_mode == "listen"


def test_emergency_turn_bypasses_normal_context_and_uses_crisis_policy() -> None:
    """Catches emergency turns that continue through normal RAG and memory paths."""
    graph = build_graph(GraphDependencies.for_mock())

    result = graph.invoke(
        {
            "transcript": "我现在正在伤害自己",
            "visual_summary": "",
            "visited": [],
        }
    )

    assert result["risk"].level is RiskLevel.EMERGENCY
    assert "crisis_policy" in result["visited"]
    assert "context_fetch" not in result["visited"]
    assert "reply_planner" not in result["visited"]
    assert result["response"].support_mode == "handoff"


def test_graph_state_round_trips_through_sqlite_checkpoint(tmp_path) -> None:
    """Catches graph state that cannot be resumed or audited by session thread ID."""
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "session_1"}}

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(GraphDependencies.for_mock(), checkpointer=checkpointer)
        graph.invoke(
            {
                "transcript": "我最近工作压力有点大",
                "visual_summary": "",
                "visited": [],
            },
            config,
        )
        snapshot = graph.get_state(config)

    assert snapshot.values["risk"].level is RiskLevel.GREEN
    assert snapshot.values["response"].support_mode == "listen"
