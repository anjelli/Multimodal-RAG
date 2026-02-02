import os
import runpy
from pathlib import Path

bin_dir = r"C:\Users\AJIT SINGH\Desktop\EY\Multimodal_RAG\poppler-25.12.0\Library\bin"
if Path(bin_dir).exists():
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + bin_dir
    print("Added Poppler to PATH:", bin_dir)
else:
    print("Poppler bin not found:", bin_dir)

os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", ".")

print("Starting ingestion (images enabled)...")
runpy.run_path("src/main.py", run_name="__main__")
