from typing import Iterable, List, Dict, Any


def _snap_to_word_boundary(text: str, pos: int, search_start: int) -> int:
    """Move *pos* backwards to the nearest whitespace boundary within [search_start, pos].
    If no whitespace is found, returns the original *pos* unchanged (avoids empty chunks)."""
    if pos >= len(text) or (pos > 0 and text[pos - 1:pos + 1].strip() == ""):
        # already on a boundary or at end
        return pos
    boundary = text.rfind(" ", search_start, pos)
    if boundary > search_start:
        return boundary + 1  # start of the next word
    # Try newline as boundary too
    boundary = text.rfind("\n", search_start, pos)
    if boundary > search_start:
        return boundary + 1
    return pos  # no boundary found; keep original to avoid empty chunk


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
        # Snap chunk end to a word boundary to avoid splitting mid-word/sentence
        if end < length:
            end = _snap_to_word_boundary(text, end, start)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(end - overlap, 0)
        # Snap overlap start to a word boundary too
        if start > 0 and start < length:
            next_space = text.find(" ", start)
            if 0 < next_space < end:
                start = next_space + 1
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
