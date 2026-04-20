# Rare Disease RAG Pipline

This project is a RAG pipeline for the rare disease domain. It combines a local document knowledge base (e.g. medical PDF literature), a vector retrieval pipeline built on ChromaDB, and **Qwen2-7B-Instruct** served through **vLLM** for high-performance inference.

The system is fully containerized with Docker Compose and is designed for reproducible deployment, local development, and further extension.

---

> **Note:** The first startup may trigger a large model download from Hugging Face. Use a stable internet connection and ensure sufficient disk space.

---

## Project structure

The project follows a flat service-oriented layout that separates deployment configuration from core RAG logic:

```text
RareDisease_rag_eng/
├── .env.example             # Environment variable template
├── docker-compose.prod.yml  # Optional production config (pulls pre-built image)
├── docker-compose.yml       # Default development config (builds from local source)
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── main.py                  # Main application entry point
├── config.py                # Global configuration
├── generation.py            # RAG generation module
├── indexer.py               # Document parsing and vector indexing
├── retrieval.py             # Retrieval and similarity search
├── .gitignore               # Git ignore rules
└── .dockerignore            # Docker build ignore rules
```

---

## Deployment

### Option 1: Build from source (recommended)

This is the recommended way to run the project for development, evaluation, or code review. The running container reflects your current local source code.

#### 1. Prepare environment variables

Clone the repository, move into the project root, and create the environment file:

```bash
cp .env.example .env
```

Edit `.env` if needed to adjust ports, model settings, or other configuration values.

#### 2. Build and start the services

```bash
docker-compose up -d --build
```

Docker builds the image locally from the `Dockerfile` and starts the required services.

#### 3. Check startup logs

To monitor the LLM startup process:

```bash
docker-compose logs -f vllm
```

On the first launch, vLLM may download model weights from Hugging Face and cache them under `./model_cache`. Duration depends on network speed.

Once the logs show that the server is running, the system is ready.

#### 4. Rebuild after code changes

If you modify the source code and want the container to reflect the latest changes:

```bash
docker-compose up -d --build
```

### Option 2: Pre-built image (optional)

a pre-built image is available on Docker Hub, you can start the system without building locally.

#### 1. Prepare environment variables

```bash
cp .env.example .env
```

#### 2. Start services with the production compose file

```bash
docker-compose -f docker-compose.prod.yml up -d
```

This path is for quick runtime setup

#### 3. Check startup logs

```bash
docker-compose -f docker-compose.prod.yml logs -f vllm
```

As with the source-build setup, the first run may still download model weights into `./model_cache`.

