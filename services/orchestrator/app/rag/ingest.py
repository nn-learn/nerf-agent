import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.rag.models import KnowledgeDocument, chunk_document


def load_manifest(manifest_path: Path) -> list[KnowledgeDocument]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("documents"), list):
        raise ValueError("knowledge manifest must contain a documents list")

    documents: list[KnowledgeDocument] = []
    for item in raw["documents"]:
        if not isinstance(item, dict):
            raise ValueError("each knowledge manifest item must be an object")
        metadata: dict[str, Any] = dict(item)
        source = metadata.pop("source", None)
        if not isinstance(source, str) or not source:
            raise ValueError("each knowledge document requires a source path")
        source_path = manifest_path.parent / source
        metadata["body"] = source_path.read_text(encoding="utf-8").strip()
        documents.append(KnowledgeDocument.model_validate(metadata))
    return documents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()
    documents = load_manifest(arguments.manifest)
    summary = {
        "documents": len(documents),
        "chunks": sum(len(chunk_document(document)) for document in documents),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

