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
