from typing import Iterable, List, Dict, Any


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, 0)
    return chunks


def chunk_text_items(
    items: Iterable[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 100
) -> List[Dict[str, Any]]:
    chunked: List[Dict[str, Any]] = []
    for item in items:
        text = item.get("text", "")
        metadata = item.get("metadata", {}).copy()
        for idx, chunk in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap)):
            entry = {"text": chunk, "metadata": {**metadata, "chunk_index": idx}}
            chunked.append(entry)
    return chunked
