# RevOn LangSmith RAG Evaluation (Additive, Evaluation-Only)

This toolkit evaluates the **existing RevOn RAG workflow** (no isolated-prompt substitute) for:
1. **Retrieval Relevance** (independent of answer correctness)
2. **Faithfulness / Groundedness** (evidence = retrieved context only)
3. **Answer Correctness** (separate abstention handling)

> **Safety:** additive-only, no production behavior changes, no deploy, no live experiment executed by default. Run live experiments only after separate explicit approval and only against approved, non-sensitive/synthetic corpora.

## 0. Mandatory SDK verification (before implementation/changes)

Verify the installed LangSmith SDK and actual APIs/signatures in the target environment:

```bash
cd backend
python -m pip show langsmith
python -c "import langsmith; print(langsmith.__version__)"
python -c "import inspect; from langsmith.evaluation import evaluate; print(inspect.signature(evaluate))"
```

Confirmed during implementation: **`langsmith==0.12.6`**. Evaluators use simplified signatures supported by `_normalize_evaluator_func` in 0.12.6: `(inputs, outputs, reference_outputs)` returning `dict`/`EvaluationResult`-compatible objects. Runner uses `langsmith.evaluation.evaluate(...)` with `max_concurrency=1`. If SDK differs, adapt via the compatibility layer and do **not** invent unsupported APIs.

## 1. Prerequisites

- Python env with `backend/requirements.txt` installed (includes `langsmith`).
- **Dedicated LangSmith project:** `revon-policy-chatbot-eval` (recommended/default in scripts). Do not mix with production tracing project.
- Env (opt-in): `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true` (for traced runs), `LANGSMITH_PROJECT=revon-policy-chatbot-eval`.
- **Real workflow runs** require existing configured credentials/services: `PINECONE_*`, embedding provider (Gemini), and LLM provider keys as currently used by the app, plus an indexed corpus matching the dataset.
- **No secrets are logged or committed.**

## 2. Dataset

Source of truth: `evaluation/langsmith_eval/datasets/revon_qa_eval_v1.json` (versioned, synthetic/redacted).

- **≥15** carefully annotated examples covering: `direct_factual`, `technical_skills`, `employment_history`, `multi_chunk`, `missing_info`, `hallucination_prevention`.
- Each example includes: `inputs.question`, `outputs.reference_answer`, `outputs.expected_key_facts`, `outputs.expected_supporting_information`, `outputs.expected_relevant_chunks` (stable `evidence_id` + precise `text_must_contain`/hints; not filename/page alone), and `outputs.expected_abstention` (`should_abstain`, `acceptable_responses`, `rationale`).
- **Corpus/index snapshot** is documented at dataset level and **must be recorded for every real experiment** (documents/SHA256, index, namespace, vector count, embedding model/dimension, chunking/top_k, timestamp, app commit). Real repeatability depends on matching corpus/index.

### Validate

```bash
cd backend
python -m evaluation.langsmith_eval.dataset_manager validate --file evaluation/langsmith_eval/datasets/revon_qa_eval_v1.json
```

### Sync to LangSmith (idempotent, redacted before upload)

```bash
cd backend
python -m evaluation.langsmith_eval.dataset_manager sync \
  --file evaluation/langsmith_eval/datasets/revon_qa_eval_v1.json \
  --dataset revon-rag-qa-eval \
  --project revon-policy-chatbot-eval
```

## 3. Real RAG target and capture

- **Target:** `evaluation.langsmith_eval.target.langsmith_target` invokes existing `retrieval_service.answer_conversational_question(...)` (real retrieval + real generation).
- Captures (returned to evaluators, redacted): original `question`, `final_answer`, all `retrieved_chunks` (`source_index`, `evidence_id`, `filename`, `page_number`, `score`, `text`), `retrieved_sources`, `retrieval_scores`, `retrieved_context`, `total_retrieved_matches`, `cited_sources`, `intent`.
- **Isolation:** unique `conversation_id` per example, `client_history=None`, dedicated evaluation-only `ConversationManager` (module reference swapped and restored; **never modifies/clears/deletes production users' conversation history**).
- **Retrieval capture:** context-local sink with safe hook install/restore around `retrieval_service.retrieve_context`. **Runner enforces `max_concurrency=1`**.

## 4. Evaluators

| Metric | What it measures | Key outputs |
|---|---|---|
| `retrieval_relevance` | Evidence/fact coverage and chunk relevance **independent of answer correctness/abstention correctness**. Adds `precision@k/recall@k/f1@k` when ground-truth chunk annotations are available (stable `evidence_id` + precise text match; **not filename/page alone**). Missing-info: irrelevant retrieved chunks score low even if abstention is correct. | `score`, `pass`, `evidence_coverage`, `retrieval_precision/recall/f1` (nullable), explanation |
| `faithfulness_groundedness` | Claims supported by `retrieved_context` **only** (reference answer never used). Identifies `unsupported_claims` and `fabricated_citations`. Real eval: **isolated LLM-as-judge** (not `llm_router`). Unit tests: deterministic rule-based/mocked. | `score`, `pass`, `unsupported_claims`, `fabricated_citations`, explanation |
| `answer_correctness` | Semantic/key-fact correctness vs `reference_answer` + `expected_key_facts`; **separate abstention branch** when `expected_abstention.should_abstain=true` (not exact-string-only). | `score`, `pass`, `abstention_expected`, `abstention_correct`, explanation |

**Judge (isolated, optional):** configure env vars (no secrets in metadata/logs): `EVAL_JUDGE_ENABLED=true`, `EVAL_JUDGE_PROVIDER`, `EVAL_JUDGE_BASE_URL`, `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_API_KEY`, optional `EVAL_JUDGE_TEMPERATURE`, `EVAL_JUDGE_TIMEOUT_SECONDS`. Judge config (sans secrets) is recorded in experiment metadata.

## 5. Run experiment (only after separate explicit approval)

Dry run (no live evaluation):

```bash
cd backend
python -m evaluation.langsmith_eval.run_experiment --dry-run --project revon-policy-chatbot-eval
```

Real experiment (requires approval, credentials, indexed corpus, and documented corpus/index snapshot):

```bash
cd backend
set LANGSMITH_API_KEY=...
set LANGSMITH_TRACING=true
set LANGSMITH_PROJECT=revon-policy-chatbot-eval
python -m evaluation.langsmith_eval.run_experiment \
  --dataset revon-rag-qa-eval \
  --experiment-prefix revon-rag-qa-v1 \
  --project revon-policy-chatbot-eval \
  --max-concurrency 1 \
  [--sync --dataset-file evaluation/langsmith_eval/datasets/revon_qa_eval_v1.json] \
  [--judge]
```

Local sanitized summaries are written to `evaluation/langsmith_eval/results/` (gitignored).

## 6. Privacy and security

- **PII/secret redaction** applied to dataset sync payloads, target-returned outputs, local artifacts, and logs (emails, phone numbers, common personal identifiers, secret-like values). Redaction preserves evaluation-critical facts (technical skills, company names, project names, employment dates).
- **No unnecessary PII** in datasets (synthetic/redacted). **Never log secrets.**
- **Limitation:** underlying production `@trace_*` decorators record inputs/outputs as implemented in production code; our toolkit cannot redact inside those prod trace payloads without production changes. Therefore, enable tracing/evaluation only for approved non-sensitive/synthetic corpora and in the dedicated evaluation project.

## 7. Deterministic tests (no live external calls)

```bash
cd backend
python -m unittest test_langsmith_eval_dataset test_langsmith_eval_target test_langsmith_eval_evaluators
```

## 8. Limitations

- Real experiments depend on live embedding/Pinecone/LLM services and the indexed corpus; results vary with corpus/index/model state (record snapshot metadata).
- Chunk-level precision/recall are reported only when stable, sufficiently precise ground-truth annotations exist; filename/page alone is insufficient and not claimed as exact.
- LLM-as-judge scoring can be nondeterministic; deterministic tests cover rule-based logic and mocked judge parsing.
- `gold_dataset_v1` remains a separate historical retrieval benchmark and is unchanged.
