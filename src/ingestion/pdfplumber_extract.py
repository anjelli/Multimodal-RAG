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


def save_tables_as_csvs(tables: List[Dict], output_dir: str, pdf_name: str) -> List[str]:
    """
    Save extracted tables as CSV files. Returns list of CSV paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = []

    for i, table_info in enumerate(tables):
        df = table_info.get("df")
        page = table_info.get("page")
        table_idx = table_info.get("table_idx")

        if df is not None:
            csv_path = out_dir / f"{Path(pdf_name).stem}_page{page}_table{table_idx}.csv"
            try:
                df.to_csv(csv_path, index=False)
                csv_paths.append(str(csv_path))
                logging.info(f"Saved table to {csv_path}")
            except Exception as e:
                logging.exception(f"Failed to save table CSV: {e}")

    return csv_paths
