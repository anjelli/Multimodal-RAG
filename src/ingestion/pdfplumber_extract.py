"""
Fallback PDF extraction using pdfplumber (pure Python, no system deps).
Extracts text and tables from PDFs without requiring Poppler.
"""
import logging
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path


def extract_with_pdfplumber(pdf_path: str, max_pages: int = None) -> Dict[str, List[Any]]:
    """
    Extract text and tables from PDF using pdfplumber.
    Returns dict with 'text_blocks' and 'tables' keys.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")

    elements = {"text_blocks": [], "tables": []}

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        if max_pages:
            num_pages = min(num_pages, max_pages)

        for page_num in range(num_pages):
            page = pdf.pages[page_num]
            logging.info(f"Extracting page {page_num + 1}/{num_pages}...")

            # Extract text
            text = page.extract_text()
            if text:
                elements["text_blocks"].append({"page": page_num + 1, "text": text})

            # Extract tables
            try:
                tables = page.extract_tables()
                if tables:
                    for table_idx, table in enumerate(tables):
                        # Convert table rows to DataFrame
                        try:
                            df = pd.DataFrame(table)
                            elements["tables"].append(
                                {"page": page_num + 1, "table_idx": table_idx, "df": df}
                            )
                        except Exception as e:
                            logging.warning(f"Failed to convert table to DF: {e}")
            except Exception as e:
                logging.warning(f"Failed to extract tables from page {page_num + 1}: {e}")

    return elements


def extract_images_with_pdfplumber(
    pdf_path: str, output_dir: str, max_pages: int = None
) -> List[Dict[str, Any]]:
    """
    Extract embedded images from a PDF using pdfplumber and save them to output_dir.
    Returns a list of saved image paths.
    """
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        if max_pages:
            num_pages = min(num_pages, max_pages)

        for page_num in range(num_pages):
            page = pdf.pages[page_num]
            images = page.images or []
            for img_idx, img in enumerate(images):
                obj_id = img.get("object_id") or img.get("xref") or img.get("name")
                if obj_id is None:
                    continue
                try:
                    extracted = page.extract_image(obj_id)
                except Exception as exc:
                    logging.warning("Failed to extract image on page %s (%s): %s", page_num + 1, obj_id, exc)
                    continue

                img_bytes = extracted.get("image")
                ext = extracted.get("ext") or "png"
                if not img_bytes:
                    continue
                page_number = page_num + 1
                filename = f"{Path(pdf_path).stem}_page{page_number}_img{img_idx + 1}.{ext}"
                out_path = out_dir / filename
                try:
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    saved.append({"path": str(out_path), "page": page_number})
                except Exception as exc:
                    logging.warning("Failed to save image %s: %s", out_path, exc)

    return saved


def save_tables_as_csvs(tables: List[Dict], output_dir: str, pdf_name: str) -> List[Dict[str, Any]]:
    """
    Save extracted tables as CSV files. Returns list of CSV paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Dict[str, Any]] = []

    for i, table_info in enumerate(tables):
        df = table_info.get("df")
        page = table_info.get("page")
        table_idx = table_info.get("table_idx")

        if df is not None:
            csv_path = out_dir / f"{Path(pdf_name).stem}_page{page}_table{table_idx}.csv"
            try:
                df.to_csv(csv_path, index=False)
                csv_paths.append(
                    {"csv_path": str(csv_path), "page": page, "table_idx": table_idx}
                )
                logging.info(f"Saved table to {csv_path}")
            except Exception as e:
                logging.exception(f"Failed to save table CSV: {e}")

    return csv_paths
