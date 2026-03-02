from src.ingestion.chunking import chunk_text, chunk_text_items


def test_chunk_text_with_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_items_preserves_metadata_and_chunk_index():
    items = [{"text": "hello world", "metadata": {"source": "x.pdf"}}]
    out = chunk_text_items(items, chunk_size=5, overlap=0)
    assert len(out) == 3
    assert out[0]["metadata"]["source"] == "x.pdf"
    assert out[0]["metadata"]["chunk_index"] == 0
    assert out[1]["metadata"]["chunk_index"] == 1


def test_chunk_text_respects_word_boundaries():
    """Issue 2: chunks should not split mid-word when a space is available."""
    text = "The company net-zero commitment is important for sustainability goals"
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    for chunk in chunks:
        # Each chunk must start and end on a word boundary (no trailing partial words)
        # The chunk should not end with a hyphenated split like "net-ze"
        assert not chunk.endswith("-"), f"Chunk ends with hyphen: {chunk!r}"


def test_chunk_text_no_word_boundary_fallback():
    """When there are no spaces, chunking should still work (fall back to hard cut)."""
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    # All characters should be covered
    assert "".join(chunks) == text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_shorter_than_chunk_size():
    assert chunk_text("hi", chunk_size=100, overlap=0) == ["hi"]
