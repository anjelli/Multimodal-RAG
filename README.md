# Multimodal RAG Project

This project implements a multimodal Retrieval-Augmented Generation (RAG) architecture. It is designed to handle various data types and facilitate efficient data ingestion and retrieval processes.

## Project Structure

```
multimodal-rag-project
├── src
│   ├── ingestion
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── retriever
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── llm_output
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── main.py
│   └── image_processing
│       └── pipeline.py
├── requirements.txt
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

If you see `ModuleNotFoundError: No module named 'pandas'` while running the pipeline, make sure the dependencies are installed (the `requirements.txt` file includes `pandas`) or install it explicitly:

```bash
pip install pandas
```

Usage (CLI):

```
python -m src.main --source path/to/report.pdf --persist-dir chroma_db --extracted-dir extracted_data
```

Environment variables:
- `OPENAI_API_KEY` — required for embedding/LLM calls when used.
- `MMRAG_DATA_DIR`, `MMRAG_EXTRACTED_DIR`, `MMRAG_PROCESSED_DIR`, `MMRAG_CHROMA_DIR` — optional directory overrides.

OS prerequisites
- Poppler (required by `pdf2image` / `unstructured` to convert PDFs to images):
	- macOS: `brew install poppler`
	- Debian/Ubuntu: `sudo apt-get install poppler-utils`
	- Windows: download a build from https://github.com/oschwartz10612/poppler-windows/releases and add the `Library\bin` folder to your `PATH` (or install via Chocolatey: `choco install poppler`).

	If you cannot install Poppler, you can skip image extraction by setting the environment variable `MMRAG_SKIP_IMAGES=1` before running the integration, or by passing `--no-images` to `src/main.py`.


## Usage and Output Generation

1. **Ingestion Pipeline**: Use the `IngestionPipeline` class from `src/ingestion/pipeline.py` to load, process, and save data.
   

2. **Retriever Pipeline**: Use the `RetrieverPipeline` class from `src/retriever/pipeline.py` to retrieve and format results from the stored data.


3. **LLM Output Generation**: Display the original charts/tables/images alongside the LLM's output.Below is the output


**Get both the LLM answer and the source images and texts that the model used**


![get both the LLM answer and the source images and texts that the model used!](asset/new_result.png)


**source image**
![source image](asset/result_image.png)


**Get both the LLM answer and the source images and texts that the model used**
![get both the LLM answer and the source images and texts that the model used!](asset/result2.png)


**source image**
![source image](asset/result2_image.png)
