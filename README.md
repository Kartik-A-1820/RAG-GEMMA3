# Local Hybrid GraphRAG with Gemma-3

Local-first Hybrid GraphRAG platform for document question answering and summarization. It is designed for laptop-class hardware, avoids paid APIs, and combines vector search, sparse lexical retrieval, Kuzu graph storage, optional local LLM-based entity/relation extraction, fusion, reranking hooks, and evaluation monitoring.

## What This Demonstrates

- Local LLM serving with Gemma-3 or another Hugging Face causal LM
- Modular document ingestion for PDF, TXT, DOCX, and CSV
- Dense vector retrieval with Chroma and SentenceTransformers
- Sparse lexical retrieval with in-repo BM25
- Kuzu embedded graph database for local entity, chunk, and relationship storage
- Optional local LLM graph extraction for typed entities and relationships
- Safe rule-based fallback extraction when the local model cannot return clean JSON
- Graph retrieval fused with dense and sparse candidates
- Reciprocal-rank fusion across dense, BM25, and graph rankings
- Pluggable reranker interface with a no-op default and optional local CrossEncoder
- Live runtime monitoring for ingest/query latency, retrieval counts, graph size, and errors
- Streamlit dashboard for querying, GraphRAG visualization, and live monitoring
- Offline retrieval evaluation with Recall@K, Precision@K, HitRate@K, NDCG@K, and MRR
- Resource-aware configuration for 4-bit loading, CPU offload, small chunks, and bounded context
- Docker-friendly local deployment

## Architecture

```text
Documents
   |
   v
Modular ingestion
   |-- parsing: PDF / TXT / DOCX / CSV
   |-- hashing and duplicate detection
   |-- local chunking
   |
   v
Dense index: Chroma + SentenceTransformers
Sparse index: BM25
Graph index: Kuzu entities + typed relationships
   |-- extraction: local LLM JSON triples or rule fallback
   |-- storage: Entity, Chunk, MENTIONED_IN, RELATED_TO
   |
   v
Dense candidates + BM25 candidates + graph candidates
   |
   v
Reciprocal-rank fusion
   |
   v
Optional local reranker
   |
   v
Grounded local LLM generation
   |
   v
Answer + source references + retrieval diagnostics + monitor metrics
```

This is intentionally a practical local GraphRAG architecture rather than a cloud service architecture. The graph layer uses Kuzu, an embedded local graph database, so it avoids the operational overhead of running a separate Neo4j server while still storing first-class graph nodes and relationships. The app also writes a compact JSON graph snapshot for fast reloads, tests, and UI visualization.

For constrained machines, `RAG_GRAPH_EXTRACTION_MODE=rules` is the fastest option. For the stronger portfolio demo, set `RAG_GRAPH_EXTRACTION_MODE=llm` and use a small local instruction model to extract JSON entities and relations. If the model fails to produce valid JSON for a chunk, the system records a fallback extraction and still builds the graph instead of failing ingestion.

If you already have documents in an older local Chroma directory, reingest them after switching to this version. New chunks include stable `chunk_id` metadata so dense, BM25, and graph results can fuse cleanly.

## Project Structure

```text
RAG-GEMMA3/
├── backend/
│   ├── main.py                  # FastAPI app and local GraphRAG pipeline
│   ├── main_4bit.py             # 4-bit default entrypoint
│   ├── config.py                # Environment-driven local runtime settings
│   ├── ingestion.py             # File loading, hashing, chunking
│   ├── llm.py                   # Lazy local causal LM generator
│   ├── retrieval/
│   │   ├── bm25.py              # Sparse retrieval
│   │   ├── fusion.py            # Reciprocal-rank fusion
│   │   ├── graph.py             # Kuzu graph store + local graph extraction/retrieval
│   │   ├── hybrid.py            # Dense + BM25 + graph orchestration
│   │   └── rerankers.py         # Pluggable reranker interface
│   └── evaluation/
│       ├── metrics.py           # Retrieval quality metrics
│       ├── monitoring.py        # Runtime metrics monitor
│       └── run_retrieval_eval.py
├── frontend/
│   └── app.py                   # Streamlit GraphRAG dashboard
├── config/
│   └── local.example.env        # Local-only runtime defaults
├── tests/
│   └── test_retrieval.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-cuda-cu130.txt
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
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_CHROMA_DIR=./data/chroma
RAG_GRAPH_PATH=./data/graph_index.json
RAG_GRAPH_DB_PATH=./data/kuzu_graph
RAG_GRAPH_BACKEND=kuzu
RAG_GRAPH_EXTRACTION_MODE=rules
RAG_GRAPH_EXTRACTOR_MODEL=Qwen/Qwen2.5-1.5B-Instruct
RAG_GRAPH_EXTRACTION_MAX_NEW_TOKENS=512
RAG_GRAPH_EXTRACTION_MAX_CHUNK_CHARS=2200
RAG_LOAD_IN_4BIT=true
RAG_DEVICE=auto
RAG_CHUNK_SIZE=1200
RAG_CHUNK_OVERLAP=180
RAG_DENSE_K=8
RAG_SPARSE_K=8
RAG_GRAPH_K=8
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

To enable local LLM knowledge graph extraction:

```env
RAG_GRAPH_EXTRACTION_MODE=llm
RAG_GRAPH_EXTRACTOR_MODEL=Qwen/Qwen2.5-1.5B-Instruct
```

For this graph-extraction flow, Qwen is usually a better first choice than Gemma on a small laptop because the Qwen2.5 model cards emphasize stronger structured output and JSON behavior. Gemma-3 1B remains a good small local answer-generation target, but its Hugging Face repository can require login and accepted model terms.

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

- `GET /health` returns local runtime configuration and graph index stats.
- `GET /metrics` returns live ingest/query latency, retrieval counts, graph stats, recent queries, and recent errors.
- `POST /metrics/reset` clears in-memory runtime metrics.
- `GET /graph` returns top Kuzu-backed graph nodes and typed relations for UI visualization.
- `POST /ingest` ingests PDF, TXT, DOCX, or CSV into Chroma, BM25, and the local graph index.
- `POST /query/local` retrieves with dense + BM25 + graph + RRF and answers using the local model.
- `POST /summarize_pdf` keeps the summarization workflow with lazy local generation.

## Evaluation And Monitoring

The evaluation module supports:

- Recall@K
- Precision@K
- HitRate@K
- NDCG@K
- Mean Reciprocal Rank
- Live latency and retrieval-count monitoring
- Recent error tracking
- Graph index size tracking
- LLM extraction and fallback extraction counters

Evaluate saved retrieval results from a local JSONL benchmark:

```json
{"query": "What is the refund window?", "relevant_ids": ["chunk-a"], "retrieved_ids": ["chunk-b", "chunk-a"]}
```

```bash
python -m backend.evaluation.run_retrieval_eval eval_results.jsonl --k 5
```

Use `/metrics` during demos to show the system is not just answering questions, but also measuring retrieval behavior and operational health.

## Streamlit Dashboard

The Streamlit UI includes:

- Ask tab with document ingestion, Hybrid GraphRAG query execution, retrieval counts, answers, and source references
- GraphRAG tab with Kuzu-backed knowledge-graph visualization, typed entities, strongest relations, and extraction counters
- Monitor tab with live ingest/query metrics, p95 latency, retrieval averages, graph size, recent queries, and recent errors
- Summarize tab for local document summarization

## No Paid APIs

This project does not require OpenAI, Anthropic, Cohere, Pinecone, Weaviate Cloud, or other paid/cloud APIs. Models, embeddings, vector storage, lexical retrieval, graph retrieval, reranking, monitoring, and evaluation are intended to run locally.

## Hardware Notes

- Use Gemma-3 1B, Qwen2.5 0.5B/1.5B Instruct, or another small instruction model first.
- Prefer Qwen2.5 1.5B Instruct for local graph extraction when disk and memory allow; use Qwen2.5 0.5B Instruct or the rule fallback on tighter machines.
- Prefer 4-bit loading on compatible NVIDIA GPUs.
- Keep chunks and top-k modest to avoid oversized prompts.
- Leave `RAG_ENABLE_RERANKER=false` until the base flow is working.
- Expect first model and embedding downloads to need internet unless the artifacts already exist in the Hugging Face cache.
- Gemma model repositories may require Hugging Face login and accepted model terms. For smoke tests on locked-down laptops, set `RAG_MODEL_ID` to another local Hugging Face causal LM directory after downloading it.
