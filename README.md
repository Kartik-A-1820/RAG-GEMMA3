# Local Hybrid GraphRAG with Gemma-3

Local-first Retrieval-Augmented Generation platform for document question answering and summarization. The project is designed for a practical 4 GB VRAM-class laptop, avoids paid APIs, and keeps the original FastAPI + Streamlit workflow while upgrading retrieval into a hybrid stack.

## What This Demonstrates

- Local LLM serving with Gemma-3 through Hugging Face Transformers
- Modular document ingestion for PDF, TXT, DOCX, and CSV
- Dense vector retrieval with Chroma and SentenceTransformers
- Sparse lexical retrieval with an in-repo BM25 implementation
- Reciprocal-rank fusion across dense and sparse rankings
- Pluggable reranker interface with a no-op default and optional local CrossEncoder
- Resource-aware configuration for 4-bit loading, CPU offload, small chunks, and bounded context
- Evaluation scaffolding for retrieval metrics
- Docker-friendly local deployment

## Architecture

```text
Documents
   |
   v
Modular ingestion
   |-- parsing: PDF / TXT / DOCX / CSV
   |-- hashing and duplicate detection
   |-- chunking with practical local defaults
   |
   v
Dense index: Chroma + all-MiniLM-L6-v2
Sparse index: BM25 rebuilt from local Chroma contents
   |
   v
Reciprocal-rank fusion
   |
   v
Optional local reranker
   |
   v
Grounded Gemma-3 generation
   |
   v
Answer + source references + retrieval diagnostics
```

The next portfolio milestone is a lightweight knowledge-graph layer for entity and relation extraction, graph traversal candidates, and graph-aware answer citations. This branch lays the retrieval and evaluation foundation for that work.

If you already have documents in an older local Chroma directory, reingest them after switching to this branch. New chunks include stable `chunk_id` metadata so dense and BM25 results can fuse cleanly.

## Project Structure

```text
RAG-GEMMA3/
├── backend/
│   ├── main.py                  # FastAPI app, local hybrid retrieval pipeline
│   ├── main_4bit.py             # 4-bit default entrypoint
│   ├── config.py                # Environment-driven local runtime settings
│   ├── ingestion.py             # File loading, hashing, chunking
│   ├── llm.py                   # Lazy local Gemma generator
│   ├── retrieval/
│   │   ├── bm25.py              # Sparse retrieval
│   │   ├── fusion.py            # Reciprocal-rank fusion
│   │   ├── hybrid.py            # Dense + sparse orchestration
│   │   └── rerankers.py         # Pluggable reranker interface
│   └── evaluation/
│       └── metrics.py           # Recall, precision, MRR scaffolding
├── frontend/
│   └── app.py                   # Streamlit UI
├── config/
│   └── local.example.env        # Local-only runtime defaults
├── tests/
│   └── test_retrieval.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run_project.py
```

## Local Runtime

Copy the example config if you want to override defaults:

```bash
cp config/local.example.env .env
```

Recommended 4 GB VRAM-class defaults:

```env
RAG_MODEL_ID=google/gemma-3-1b-it
RAG_LOAD_IN_4BIT=true
RAG_DEVICE=auto
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=180
RAG_DENSE_K=8
RAG_SPARSE_K=8
RAG_FUSED_K=5
RAG_MAX_CONTEXT_CHARS=6000
RAG_MAX_NEW_TOKENS=512
RAG_ENABLE_RERANKER=false
```

Keep the reranker disabled on constrained hardware unless retrieval quality needs the extra pass. To try a CPU reranker locally:

```env
RAG_ENABLE_RERANKER=true
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-cuda-cu130.txt
uvicorn backend.main_4bit:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py
```

Or use the launcher:

```bash
python run_project.py
```

Docker:

```bash
docker compose up --build
```

The Docker default uses CPU mode. For GPU acceleration, install the matching PyTorch build on the host or customize the image for your CUDA stack.

For the tested local Windows laptop setup with NVIDIA GTX 1650 Ti, driver `610.62`, and CUDA UMD `13.3`, the project venv uses `torch==2.14.0+cu130` from the official PyTorch CUDA 13.0 wheel index. Verify CUDA with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## API

- `GET /health` returns local runtime configuration.
- `POST /ingest` ingests PDF, TXT, DOCX, or CSV into the local Chroma index and BM25 layer.
- `POST /query/local` retrieves with dense + BM25 + RRF and answers using local Gemma.
- `POST /summarize_pdf` keeps the original summarization workflow with lazy local generation.

## Evaluation

The initial retrieval metrics live in `backend/evaluation/metrics.py` and are covered by tests. You can evaluate saved retrieval results from a small local JSONL benchmark:

```json
{"query": "What is the refund window?", "relevant_ids": ["chunk-a"], "retrieved_ids": ["chunk-b", "chunk-a"]}
```

```bash
python -m backend.evaluation.run_retrieval_eval eval_results.jsonl --k 5
```

Next, compare dense-only, BM25-only, hybrid RRF, and reranked hybrid retrieval with Recall@K, Precision@K, and MRR.

## No Paid APIs

This project does not require OpenAI, Anthropic, Cohere, Pinecone, Weaviate Cloud, or other paid/cloud APIs. Models, embeddings, vector storage, lexical retrieval, reranking, and evaluation are intended to run locally.

## Hardware Notes

- Use Gemma-3 1B or another small instruction model first.
- Prefer 4-bit loading on compatible NVIDIA GPUs.
- Keep chunks and top-k modest to avoid oversized prompts.
- Leave `RAG_ENABLE_RERANKER=false` until the base flow is working.
- Expect first model and embedding downloads to need internet unless the artifacts already exist in the Hugging Face cache.
- Gemma model repositories may require Hugging Face login and accepted model terms. For smoke tests on locked-down laptops, set `RAG_MODEL_ID` to another local Hugging Face causal LM directory after downloading it.
