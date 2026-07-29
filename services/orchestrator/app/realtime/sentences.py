import re


def split_spoken_sentences(text: str) -> list[str]:
    """Split on spoken clause boundaries without touching URLs or decimals."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.findall(r".+?(?:[。！？!?；;]+|$)", normalized)
    return [part.strip() for part in parts if part.strip()]
