from typing import Any, Dict, List, Optional
import logging
import json
from pathlib import Path

import pandas as pd


class IngestionPipeline:
    def __init__(self, source: str, extracted_dir: Optional[str] = None, extract_images: bool = True):
        self.source = source
        self.raw_elements = None
        self.extracted_dir = extracted_dir or "extracted_data"
        self.extract_images = extract_images
        self.processed_data = {
            "Header": [],
            "Footer": [],
            "Title": [],
            "NarrativeText": [],
            "Text": [],
            "ListItem": [],
            "Image": [],
            "Table": [],
        }

    def load_data(self):
        logging.info("Loading PDF and extracting elements from %s", self.source)
        try:
            from unstructured.partition.pdf import partition_pdf

            kwargs = dict(
                filename=self.source,
                strategy="hi_res",
                extract_images_in_pdf=self.extract_images,
                infer_table_structure=True,
                extract_image_block_types=["Image", "Table"],
                extract_image_block_to_payload=False,
                extract_image_block_output_dir=self.extracted_dir,
            )

            self.raw_elements = partition_pdf(**kwargs)
            if not self.raw_elements:
                raise RuntimeError("unstructured returned no elements")
        except Exception as exc:
            logging.warning(
                "Unstructured PDF extraction failed (%s). Falling back to pdfplumber/PyMuPDF extraction.",
                exc,
            )
            self._load_data_with_pdfplumber()

    def _load_data_with_pdfplumber(self):
        """Fallback: extract text, tables, and images using pdfplumber (no system dependencies)."""
        import os
        from src.ingestion.pdfplumber_extract import (
            extract_with_pdfplumber,
            extract_images_with_pdfplumber,
            extract_images_with_pymupdf,
            save_tables_as_csvs,
        )

        # Issue 12: respect MMRAG_MAX_PAGES to avoid OOM on large PDFs
        max_pages_env = os.environ.get("MMRAG_MAX_PAGES")
        max_pages = int(max_pages_env) if max_pages_env and max_pages_env.isdigit() else None

        # Extract with pdfplumber
        extracted = extract_with_pdfplumber(self.source, max_pages=max_pages)
        text_blocks = extracted.get("text_blocks", [])
        tables = extracted.get("tables", [])

        # Save tables to CSVs
        if tables:
            csv_paths = save_tables_as_csvs(tables, "processed_data", self.source)
            for csv_path in csv_paths:
                if isinstance(csv_path, dict):
                    self.processed_data["Table"].append(csv_path)
                else:
                    self.processed_data["Table"].append({"csv_path": csv_path})

        # Convert text blocks into elements (mimic unstructured output)
        class TextElement:
            def __init__(self, text):
                self.text = text
            def __str__(self):
                return self.text

        class ImageElement:
            def __init__(self, path):
                self.filename = path
            def __str__(self):
                return f"Image({self.filename})"

        self.raw_elements = []
        for block in text_blocks:
            text = block.get("text", "")
            page_num = block.get("page")
            # Preserve paragraph structure: split on double newlines first;
            # fall back to single newlines only if no paragraphs found.
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            if not paragraphs:
                paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            for para in paragraphs:
                # Skip very short fragments (likely headers/footers/noise)
                if len(para) < 10:
                    continue
                elem = TextElement(para)
                if page_num is not None:
                    elem.page = page_num
                self.raw_elements.append(elem)

        if self.extract_images:
            image_paths = extract_images_with_pdfplumber(self.source, self.extracted_dir)
            if not image_paths:
                try:
                    image_paths = extract_images_with_pymupdf(self.source, self.extracted_dir)
                except Exception as exc:
                    logging.warning("PyMuPDF image extraction unavailable: %s", exc)
            for path_info in image_paths:
                image_path = path_info.get("path") if isinstance(path_info, dict) else path_info
                if image_path:
                    self.raw_elements.append(ImageElement(image_path))

    def _save_table_df(self, df: pd.DataFrame, prefix: str, idx: int) -> str:
        out_dir = Path("processed_data")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{Path(self.source).stem}_table_{prefix}_{idx}.csv"
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def process_data(self):
        if not self.raw_elements:
            logging.warning("No raw elements to process; call load_data() first")
            return

        for i, element in enumerate(self.raw_elements):
            t = str(type(element))
            try:
                if "Header" in t:
                    self.processed_data["Header"].append(str(element))
                elif "Footer" in t:
                    self.processed_data["Footer"].append(str(element))
                elif "Title" in t:
                    self.processed_data["Title"].append(str(element))
                elif "NarrativeText" in t:
                    self.processed_data["NarrativeText"].append(str(element))
                elif "Text" in t:
                    text_str = str(element)
                    # Skip very short text fragments (noise/garbage)
                    if len(text_str.strip()) < 10:
                        continue
                    page = getattr(element, "page", None)
                    if page is not None:
                        self.processed_data["Text"].append({"text": text_str, "metadata": {"page": page}})
                    else:
                        self.processed_data["Text"].append(text_str)
                elif "ListItem" in t:
                    self.processed_data["ListItem"].append(str(element))
                elif "Image" in t:
                    # element may include a filename saved by unstructured; try to capture it
                    try:
                        path = (
                            getattr(element, "filename", None)
                            or getattr(element, "source", None)
                            or getattr(getattr(element, "metadata", None), "image_path", None)
                        )
                        if path:
                            self.processed_data["Image"].append({"path": path, "raw": str(element)})
                        else:
                            self.processed_data["Image"].append({"raw": str(element)})
                    except Exception:
                        self.processed_data["Image"].append({"raw": str(element)})
                elif "Table" in t:
                    # Try to convert to pandas DataFrame when possible
                    df = None
                    # common Unstructured table API: element.to_pandas() or element.to_dataframe()
                    to_pd = getattr(element, "to_pandas", None) or getattr(element, "to_dataframe", None)
                    if callable(to_pd):
                        try:
                            df = to_pd()
                        except Exception:
                            df = None

                    # fallback: try rows property
                    if df is None:
                        rows = getattr(element, "rows", None)
                        if rows:
                            try:
                                df = pd.DataFrame(rows)
                            except Exception:
                                df = None

                    if df is not None and isinstance(df, pd.DataFrame):
                        csv_path = self._save_table_df(df, "auto", i)
                        self.processed_data["Table"].append({"csv_path": csv_path, "shape": df.shape})
                    else:
                        # last resort: store text representation
                        self.processed_data["Table"].append({"raw": str(element)})
            except Exception as e:
                logging.exception("Error processing element %s: %s", i, e)

    def get_processed_data(self) -> Dict[str, List[Any]]:
        # optionally persist a JSON summary of processed_data
        out = Path("processed_data")
        out.mkdir(parents=True, exist_ok=True)
        summary_path = out / f"{Path(self.source).stem}_summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("Failed to write processed summary")
        return self.processed_data
