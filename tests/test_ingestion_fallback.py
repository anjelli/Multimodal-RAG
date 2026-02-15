import types
import sys

from src.ingestion.pipeline import IngestionPipeline


def test_ingestion_falls_back_when_unstructured_fails(monkeypatch, tmp_path):
    unstructured = types.ModuleType("unstructured")
    partition = types.ModuleType("unstructured.partition")
    pdf = types.ModuleType("unstructured.partition.pdf")

    def broken_partition_pdf(**kwargs):
        raise RuntimeError("boom")

    pdf.partition_pdf = broken_partition_pdf

    monkeypatch.setitem(sys.modules, "unstructured", unstructured)
    monkeypatch.setitem(sys.modules, "unstructured.partition", partition)
    monkeypatch.setitem(sys.modules, "unstructured.partition.pdf", pdf)

    called = {"fallback": False}

    def fake_fallback(self):
        called["fallback"] = True
        self.raw_elements = []

    monkeypatch.setattr(IngestionPipeline, "_load_data_with_pdfplumber", fake_fallback)

    p = IngestionPipeline("dummy.pdf", extracted_dir=str(tmp_path))
    p.load_data()

    assert called["fallback"] is True
