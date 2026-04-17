# RareDisease RAG (English)

A lightweight Retrieval-Augmented Generation (RAG) project for rare disease Q&A based on medical guideline PDFs.

The pipeline:
1. Read PDF files
2. Build parent/child chunks
3. Store embeddings in ChromaDB
4. Retrieve relevant content
5. Generate answers with an LLM (via vLLM-compatible API)

## Project Structure

```text
RareDisease_rag_eng/
├── config.py          # paths, model config, prompts
├── indexer.py         # build vector index from PDFs
├── retrieval.py       # hybrid retrieval + fusion
├── generation.py      # context building + answer generation
├── main.py            # CLI chat entry
├── eva.py             # optional evaluation script
├── groundtruth.json   # sample eval data
├── nur pdf/           # input PDF folder
└── chroma_db_med/     # generated ChromaDB data
```

## Requirements

- Python 3.10+
- Main dependencies in `requirements.txt`
- A running vLLM/OpenAI-compatible endpoint for generation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.py` before running:

- `INPUT_DIR`: folder containing PDF files
- `DB_PATH`: ChromaDB storage path
- `COLLECTION_NAME`: Chroma collection name
- `EMBED_MODEL_PATH`: embedding model path or HF model id
- `VLLM_API_BASE`: vLLM API base URL
- `VLLM_MODEL_NAME`: served model name

## How to Run

### 1) Build the index (required first step)

```bash
python indexer.py
```

This reads PDFs from `INPUT_DIR` and writes vector data to `DB_PATH`.

### 2) Run interactive CLI

```bash
python main.py
```


## Docker Notes

If you run with Docker, make sure:
- Your local PDF directory is mounted into the container (for example to `/data/input`)
- `INPUT_DIR` in runtime environment points to that mounted path
- `DB_PATH` points to a mounted persistent directory (for example `/data/chroma_db_med`)

You still need to run indexing at least once before query service can work.

## Quick Troubleshooting

- No retrieval results: check if `indexer.py` has been run successfully.
- Import warnings in IDE: usually environment/interpreter mismatch.
- Slow first run: model download/cache may take time.
