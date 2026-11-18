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

## Gemini Daily Embedding Quota Protection (local estimate)

The embedding pipeline includes configurable RPM pacing, bounded retry/backoff (`RetryInfo` aware), incremental/resumable ingestion, and **persistent local daily request-budget protection** (counts every embedding API request attempt, including retries). **Local accounting is an estimate only; Google quota/usage is authoritative.** No automatic reset is performed based on assumed quota reset time.

**Key env vars:** `GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED`, `GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET`, `GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID` (operator-controlled reset/window rotation), `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH`, `GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE` (explicit recovery only).

**Required for production accounting (durability):** do **not** rely on ephemeral/temp storage. Set `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH` to a **durable persistent volume/path** that survives restarts and deployments/redeploys for the same environment. Local filesystem persistence does **not** guarantee survival across deployments/hosts unless the same durable volume is retained and remounted.

**Fail-closed state handling:** if an existing state file is corrupted or unreadable, the service refuses to issue further Gemini embedding requests (no silent counter reset). Operator must repair the state file or perform explicit controlled recovery (`GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE=reset`) after verifying accounting impact.

**Concurrency:** current filesystem budget store uses a process-local lock and is **not safe for multiple processes/instances** sharing a budget. Distributed/multi-worker production requires a shared transactional store (e.g. PostgreSQL/Supabase atomic reservation design). See `backend/docs/gemini_embedding_daily_quota_hardening.md` for persistence, recovery, and concurrency design/evaluation. **Production readiness for distributed deployments is not claimed without verified shared-storage/concurrency guarantees.**

---

## Evaluation

The `evaluation/` directory contains a complete RAG audit trail:

- **Retrieval evaluation reports** — live baseline, diversity audit, text-level chunk audit
- **Gold Dataset v1.0** — 5-query graded evaluation dataset with canonical source passages
- **Metric baseline** — Hit Rate@3, Precision@3, Recall@3, Evidence-MRR@3, nDCG@3
- **Annotation policy** — Broad Policy with explicit Grade 0/1/2 definitions

See [`evaluation/gold_dataset_v1_freeze_record.md`](evaluation/gold_dataset_v1_freeze_record.md) for the full freeze record.
