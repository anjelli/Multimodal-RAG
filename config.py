import os
from pathlib import Path


class Config:
    """Centralized config for the project. Reads env vars and provides defaults."""
    # Read the standard OPENAI_API_KEY environment variable (do NOT hard-code keys)
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    DATA_DIR: Path = Path(os.environ.get("MMRAG_DATA_DIR", "data")).absolute()
    EXTRACTED_DIR: Path = Path(os.environ.get("MMRAG_EXTRACTED_DIR", "extracted_data")).absolute()
    PROCESSED_DIR: Path = Path(os.environ.get("MMRAG_PROCESSED_DIR", "processed_data")).absolute()
    CHROMA_PERSIST_DIR: Path = Path(os.environ.get("MMRAG_CHROMA_DIR", "chroma_db")).absolute()

    @classmethod
    def ensure_dirs(cls):
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        cls.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
