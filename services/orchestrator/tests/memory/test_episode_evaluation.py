from pathlib import Path

from app.memory.episode_evaluation import MemoryEpisodeEvaluator, load_episode_cases


def test_v23_episode_fixture_improves_same_episode_recall_without_enabling_graph() -> None:
    fixture = Path(__file__).parents[4] / "evals" / "memory_v23_cases.jsonl"

    report = MemoryEpisodeEvaluator().evaluate(load_episode_cases(fixture))

    assert report.case_count == 4
    assert report.episode_multihop_recall_at_k > report.flat_multihop_recall_at_k
    assert report.episode_precision_at_k >= 0.8
    assert report.summary_context_leakage_rate == 0.0
    assert not report.graph_gate_ready
    assert not report.graph_activation_recommended
    assert report.failed_case_ids == []
