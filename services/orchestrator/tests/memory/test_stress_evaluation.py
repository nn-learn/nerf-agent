from app.memory.stress_evaluation import MemoryStressEvaluator


def test_v241_runs_50_queries_without_confusing_stress_with_independent_gold() -> None:
    # Keep correctness regression independent from shared-CI scheduler jitter;
    # the CLI's default 100 ms gate remains the performance acceptance run.
    report = MemoryStressEvaluator(latency_gate_ms=500.0).evaluate()

    assert report.user_count == 10
    assert report.query_count == 50
    assert report.answerable_recall_rate == 1.0
    assert report.correct_abstention_rate == 1.0
    assert report.forbidden_retrieval_rate == 0.0
    assert report.cross_user_retrieval_rate == 0.0
    assert report.summary_context_leakage_rate == 0.0
    assert report.engineering_stress_passed is True
    assert report.independently_annotated_query_count == 0
    assert report.independent_graph_gate_ready is False


def test_v241_user_count_scales_queries_at_five_per_user() -> None:
    report = MemoryStressEvaluator(user_count=2, latency_gate_ms=500.0).evaluate()

    assert report.query_count == 10
    assert {row.user_id for row in report.query_results} == {
        "stress_user_00",
        "stress_user_01",
    }
