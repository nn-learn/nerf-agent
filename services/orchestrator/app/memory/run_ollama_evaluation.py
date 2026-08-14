import json
import time
from pathlib import Path

from app.memory.evaluation import MemoryEvaluator, load_cases
from app.memory.ollama_extractor import OllamaMemoryExtractor
from app.settings import Settings

if __name__ == "__main__":
    settings = Settings()
    cases_path = Path(__file__).parents[4] / "evals" / "memory_cases.jsonl"
    extractor = OllamaMemoryExtractor(
        model=settings.text_model,
        base_url=settings.ollama_base_url,
        keep_alive=settings.ollama_keep_alive,
        timeout_seconds=settings.ollama_timeout_seconds,
        num_ctx=settings.memory_ollama_num_ctx,
        num_predict=settings.memory_ollama_num_predict,
        cache_entries=settings.memory_extraction_cache_entries,
        use_rule_fallback=settings.memory_rule_fallback,
    )
    started = time.perf_counter()
    try:
        report = MemoryEvaluator(extractor=extractor).evaluate(load_cases(cases_path))
        elapsed_ms = (time.perf_counter() - started) * 1_000
        print(
            json.dumps(
                {
                    "model": settings.text_model,
                    "report": report.model_dump(),
                    "telemetry": extractor.telemetry().model_dump(),
                    "wall_elapsed_ms": round(elapsed_ms, 2),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        extractor.close()
