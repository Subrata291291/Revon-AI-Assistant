# RevOn Policy AI Chatbot — Backend

An AI-powered policy retrieval and question-answering backend built with **FastAPI**, **Pinecone**, **Gemini Embeddings**, and **Groq LLM**.

---

## Architecture

```
Client / WordPress Proxy
        │
        ▼
  FastAPI (main.py)
        │
        ├─── /api/v1/chat        ← Public policy chat (server-to-server via REVON_CHAT_TOKEN)
        ├─── /api/v1/documents   ← Admin document management (REVON_API_KEY)
        └─── /api/v1/health      ← Health check
        │
        ▼
  retrieval_service.py           ← Orchestrates embedding + vector search + LLM generation
        │
        ├─── embedding_service.py  ← Gemini gemini-embedding-2 (768-dim)
        ├─── vector_store.py       ← Pinecone index (revon-policy-index, revon-policies ns)
        └─── groq_service.py       ← Groq LLM (llama-3.3-70b-versatile)
        │
  ingestion_service.py           ← PDF ingestion pipeline
        │
        ├─── pdf_processor.py      ← PyPDF text extraction
        └─── chunker.py            ← Custom sentence-aware chunker (800 chars / 120 overlap)
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in all required values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (for `gemini-embedding-2`) |
| `GROQ_API_KEY` | Groq API key (for LLM generation) |
| `PINECONE_API_KEY` | Pinecone API key |
| `PINECONE_INDEX_NAME` | Pinecone index name (default: `revon-policy-index`) |
| `PINECONE_NAMESPACE` | Pinecone namespace (default: `revon-policies`) |
| `REVON_API_KEY` | Admin API key for document management endpoints |
| `REVON_CHAT_TOKEN` | Server-to-server token for the public chat proxy |
| `CHAT_RATE_LIMIT_PER_MINUTE` | Per-visitor chat rate limit (default: `10`) |

### 4. Run the application

```bash
uvicorn main:app --reload
```

---

## Key Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Health check |
| `POST` | `/api/v1/chat` | `X-RevOn-Chat-Token` | Public policy chat (WordPress proxy) |
| `POST` | `/api/v1/documents/ingest` | `X-RevOn-Api-Key` | Ingest a PDF document |
| `GET` | `/api/v1/documents` | `X-RevOn-Api-Key` | List ingested documents |
| `DELETE` | `/api/v1/documents/{doc_id}` | `X-RevOn-Api-Key` | Delete document and vectors |
| `GET` | `/docs` | None | Interactive Swagger UI |

---

## Running Tests

```bash
pytest
```

All 56 tests in the test suite pass without requiring live API keys (mocks are used).

---

## Project Structure

```
backend/
├── main.py                    # FastAPI application entrypoint
├── retrieval_service.py       # RAG orchestration (retrieve + generate)
├── embedding_service.py       # Gemini embedding client
├── vector_store.py            # Pinecone vector store client
├── groq_service.py            # Groq LLM client
├── ingestion_service.py       # PDF ingestion pipeline
├── pdf_processor.py           # PDF text extraction (PyPDF)
├── chunker.py                 # Custom sentence-aware text chunker
├── create_pinecone_index.py   # One-time index creation script
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── test_main_api.py           # FastAPI endpoint tests
├── test_ingestion_service.py  # Ingestion pipeline tests
├── test_vector_store.py       # Vector store tests
├── test_chunking.py           # Chunker unit tests
└── evaluation/                # RAG evaluation audit reports and gold dataset v1.0
```

---

## Embedding Model & Vector Index

| Property | Value |
|---|---|
| Embedding model | `gemini-embedding-2` |
| Embedding dimension | `768` |
| Pinecone index | `revon-policy-index` |
| Namespace | `revon-policies` |
| Vector ID format | `{document_id}-page-{page}-chunk-{chunk}` |

---

## Evaluation

The `evaluation/` directory contains a complete RAG audit trail:

- **Retrieval evaluation reports** — live baseline, diversity audit, text-level chunk audit
- **Gold Dataset v1.0** — 5-query graded evaluation dataset with canonical source passages
- **Metric baseline** — Hit Rate@3, Precision@3, Recall@3, Evidence-MRR@3, nDCG@3
- **Annotation policy** — Broad Policy with explicit Grade 0/1/2 definitions

See [`evaluation/gold_dataset_v1_freeze_record.md`](evaluation/gold_dataset_v1_freeze_record.md) for the full freeze record.
