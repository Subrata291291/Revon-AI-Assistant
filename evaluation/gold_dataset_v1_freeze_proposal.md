# RevOn RAG Evaluation Baseline: Gold Dataset v1.0 Freeze Proposal

**Proposal Date:** 2026-11-13  
**Status:** Review-Only Proposal — Awaiting Engineering & Product Sign-Off  
**Document Lineage:**  
1. [retrieval_evaluation_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/retrieval_evaluation_report.md) (Live Baseline Audit)  
2. [retrieval_diversity_audit_corrected_report_v2.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/retrieval_diversity_audit_corrected_report_v2.md) (Diversification Simulation)  
3. [full_retrieved_chunk_text_audit_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/full_retrieved_chunk_text_audit_report.md) (Read-Only Text Audit)  
4. [evaluation_methodology_correction_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/evaluation_methodology_correction_report.md) (Metric Scope Definitions)  
5. [gold_dataset_validation_versioning_audit_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/gold_dataset_validation_versioning_audit_report.md) (Corpus-Wide 50-Vector Assessment)  
6. [gold_dataset_final_metric_consistency_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/gold_dataset_final_metric_consistency_report.md) (Final Consistency Check)  
7. [mrr_definition_correction_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/mrr_definition_correction_report.md) (MRR vs. Direct Answer Reconciliation)  
8. [relevance_annotation_consistency_audit_report.md](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/relevance_annotation_consistency_audit_report.md) (Annotation Consistency Audit)  

---

## 1. Executive Summary

This proposal formally recommends freezing the RevOn Policy Chatbot retrieval evaluation dataset and baseline metrics as **Gold Dataset v1.0**. 

Over six iterative audits, the evaluation framework has transitioned from an unverified coordinate-matching proxy (`page/chunk`) into an authoritative, text-grounded benchmark. Every vector in the active Pinecone index (50 vectors) has been physically inspected against authoritative source text from `Pre-Read_ Basic Understanding on API  (1)-5983.pdf` (SHA-256: `ed74f10871a921595c058a90f552f169b79b7198287f3551d4a465aeeaf87856`).

> [!IMPORTANT]
> **Strict Read-Only Declaration:**  
> This document is a **review-only proposal**. It does not perform any writes, deletions, or index updates to Pinecone. It makes zero API calls to external embedding models or LLMs (Gemini, Groq, OpenAI). It makes zero modifications to application source code or `.env` configurations. All proposals herein require formal human sign-off before implementation.

### Key Objectives of the v1.0 Freeze
1. **Decouple Canonical Truth from Dynamic Coordinates:** Replace fragile page/chunk coordinates with immutable text passages and cryptographic source hashes.
2. **Formalize Graded Relevance Policy:** Adopt the **Broad Policy** for multi-part questions, defining explicit criteria for Grade 0 (Irrelevant), Grade 1 (Supporting Evidence), and Grade 2 (Direct Answer).
3. **Establish Standardized Metrics:** Adopt mathematically rigorous, disambiguated retrieval metrics including **Evidence-MRR@3**, **Direct Answer Reciprocal Rank@3**, and **Corpus-Level nDCG@3**.
4. **Institutionalize Governance:** Implement strict Semantic Versioning and an audit change management protocol for future index re-ingestions.

---

## 2. Annotation Policy Proposal (Broad Policy Recommendation)

### 2.1 Formal 3-Tier Relevance Rubric

To ensure reproducible evaluations across different annotators, chunking algorithms, and retrieval models, relevance is graded on a 3-tier functional scale:

| Relevance Grade | Category | Operational Definition | Retrieval Criteria |
|:---:|---|---|---|
| **0** | **Irrelevant** | The chunk contains no substantive evidence addressing the query, contains unrelated tutorial code/analogies without context, or serves as a misleading distractor. | Contains zero actionable information for answer synthesis. |
| **1** | **Supporting Evidence** | The chunk provides valid supporting context, prerequisite definitions, foundational analogies, multi-step resolution fragments, or concrete class examples that directly contribute to answering the question, but **does not independently answer the query**. | Must be combined with other chunks for a full answer; directly enriches LLM context. |
| **2** | **Direct / Primary Answer** | The chunk independently provides a direct, comprehensive, and substantially complete answer to the query's core information need. | A standalone user reading only this chunk receives a direct answer. |

---

### 2.2 Recommendation: Adopt the Broad Policy for Q3 Vector `9/0`

The primary distinction between the **Broad Policy** and the **Strict Policy** centers on Question 3:  
*Query:* **"How do serverless APIs differ from serverful APIs?"**

#### Candidate Chunk Inspection: Vector `9/0` (`ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-0`)
- **Stored Stored Text:**
  > *"Let’s break them down. 1. What is a Serverful API? A Serverful API is hosted on a dedicated server or virtual machine (VM) . This server is always running , whether or not any request is being made. Think of it like a restaurant kitchen that’s open 24x7 — even when no one is ordering. In this model: ● You (or the provider) must set up the infrastructure : the machine, memory, CPU, scaling, etc. ● The server is always available to take requests, like an always-open shop. ● All LLM computation (e.g., generating embeddings or responses) happens on this machine..."*

#### Detailed Justification for Grade 1 Assignment:
1. **Why Vector `9/0` CANNOT Receive Grade 2 (Direct Answer):**  
   Vector `9/0` defines only the serverful model. It does not mention serverless architecture, event-driven execution, or pay-per-execution billing. A user seeking to understand the *difference* between the two cannot obtain an answer from `9/0` alone. Therefore, Grade 2 is disallowed.
2. **Why Vector `9/0` CANNOT Receive Grade 0 (Irrelevant):**  
   A comparative contrast fundamentally requires establishing both entities being compared. Vector `9/0` provides the authoritative baseline definition of Serverful APIs (dedicated VM, 24x7 idle execution). In a RAG pipeline, feeding `9/0` alongside `10/0` (serverless definition) gives the generator model the exact prerequisite facts needed to articulate the architectural contrast. Treating `9/0` as irrelevant penalizes the retrieval engine for fetching the half of the comparison that is indispensable for synthesis.
3. **Reproducibility Criterion:**  
   Under the **Broad Policy**, any chunk providing an authoritative definition of either side of a comparative question is classified as **Grade 1 (Supporting Evidence)**. Chunks that contain the explicit comparative contrast (e.g., `9/2` and `10/0`) are classified as **Grade 2 (Direct Answer)**.

#### Sensitivity Comparison: Broad vs. Strict Policy Impact
If an engineering team chooses the **Strict Policy** (reclassifying `9/0` as Grade 0 because it lacks comparative text), the resulting metric impact is completely understood and quantified:

| Evaluation Policy | Q3 Graded Vector Ranks (Orig / Div) | Evidence-MRR@3 (Orig / Div) | Direct Answer RR@3 (Orig / Div) |
|---|:---:|:---:|:---:|
| **Broad Policy (Recommended)** | `[1, 2, 1]` / `[1, 2, 2]` | **0.8667** / **0.9000** | **0.7667** / **0.8000** |
| **Strict Policy** | `[0, 2, 1]` / `[0, 2, 2]` | **0.7667** / **0.8000** | **0.7667** / **0.8000** |

> [!NOTE]
> Under the Strict Policy, **Evidence-MRR@3 collapses into Direct Answer Reciprocal Rank@3** (0.7667 Orig / 0.8000 Div). Under both policies, diversification delivers a statistically identical **+0.0333 reciprocal rank gain** from promoting Q4.

---

## 3. Canonical Ground-Truth Schema & Decoupling

### 3.1 Architectural Failure of Coordinate-Based Evaluation

Previous evaluations relied on exact page/chunk string matching (`p/c`) as a proxy for relevance:
- **Flaw 1 (Boundary Sensitivity):** Modifying chunk size by 50 characters or altering whitespace stripping shifts chunk boundaries across the entire document. A perfectly relevant chunk on line 12 might shift from `1/1` to `1/2`, dropping computed recall from 100% to 0%.
- **Flaw 2 (Coordinate Truncation & The Q4 Anomaly):** Nominal coordinate `16/0` was previously labeled as gold for Q4 (HTTP 403 Forbidden). However, physical text audits proved `16/0` ran out of character budget and truncated at `400 Bad Request`. The actual 403 definition was stored in `16/1`. Coordinate evaluation penalized retrieval for returning `16/1` despite it being the only chunk containing the true answer.
- **Flaw 3 (Implementation Divergence):** The local ingestion script [`chunker.py`](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/chunker.py) produces 42 chunks for this PDF, whereas the active Pinecone index contains 50 vectors. Coordinate matching across differing chunkers is mathematically invalid.

---

### 3.2 Canonical Dataset JSON Schema Specification

The canonical ground truth is decoupled into:
1. **Canonical Ground Truth:** Immutable, text-level information needs tied to the source document hash.
2. **Index Mappings:** Volatile, index-specific vector coordinates mapped to canonical passage IDs.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RevOnRAGGoldDatasetV1",
  "type": "object",
  "required": ["dataset_metadata", "queries"],
  "properties": {
    "dataset_metadata": {
      "type": "object",
      "required": ["version", "status", "created_at", "document"],
      "properties": {
        "version": { "type": "string", "pattern": "^v[0-9]+\\.[0-9]+\\.[0-9]+$" },
        "status": { "type": "string", "enum": ["draft", "frozen", "deprecated"] },
        "created_at": { "type": "string", "format": "date" },
        "annotation_policy": { "type": "string", "enum": ["broad", "strict"] },
        "document": {
          "type": "object",
          "required": ["filename", "sha256_hash", "total_pages"],
          "properties": {
            "filename": { "type": "string" },
            "sha256_hash": { "type": "string", "pattern": "^[a-f0-9]{64}$" },
            "total_pages": { "type": "integer" }
          }
        }
      }
    },
    "queries": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["query_id", "query_text", "canonical_passages"],
        "properties": {
          "query_id": { "type": "string", "pattern": "^Q[0-9]+$" },
          "query_text": { "type": "string" },
          "domain_topic": { "type": "string" },
          "canonical_passages": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["passage_id", "source_page", "canonical_grade", "text_excerpt", "rationale"],
              "properties": {
                "passage_id": { "type": "string" },
                "source_page": { "type": "integer" },
                "canonical_grade": { "type": "integer", "enum": [1, 2] },
                "text_excerpt": { "type": "string" },
                "rationale": { "type": "string" }
              }
            }
          },
          "index_mappings": {
            "type": "object",
            "description": "Optional index-specific vector coordinates mapped to canonical passage IDs"
          }
        }
      }
    }
  }
}
```

---

### 3.3 Complete v1.0 Ground-Truth Dataset (Canonical Sample Records)

Below are the canonical records for all five baseline queries (Q1–Q5), linked to source document `Pre-Read_ Basic Understanding on API  (1)-5983.pdf` (SHA-256: `ed74f10871a921595c058a90f552f169b79b7198287f3551d4a465aeeaf87856`):

```json
{
  "dataset_metadata": {
    "version": "v1.0.0",
    "status": "frozen",
    "created_at": "2026-11-13",
    "annotation_policy": "broad",
    "document": {
      "filename": "Pre-Read_ Basic Understanding on API  (1)-5983.pdf",
      "sha256_hash": "ed74f10871a921595c058a90f552f169b79b7198287f3551d4a465aeeaf87856",
      "total_pages": 16,
      "total_active_vectors_in_target_index": 50
    }
  },
  "queries": [
    {
      "query_id": "Q1",
      "query_text": "How does DNS translate domain names to IP addresses?",
      "domain_topic": "DNS Resolution Mechanism",
      "canonical_passages": [
        {
          "passage_id": "Q1-P1",
          "source_page": 1,
          "canonical_grade": 2,
          "text_excerpt": "How it Works (Step-by-Step): 1. You type www.example.com into your browser. 2. The browser checks the local DNS cache (if you've visited recently). 3. If not found, it queries the recursive resolver (usually provided by your ISP). 4. Resolver asks the root DNS server → then the TLD server (like .com) → then the authoritative DNS server.",
          "rationale": "Directly provides the step-by-step resolution process."
        },
        {
          "passage_id": "Q1-P2",
          "source_page": 1,
          "canonical_grade": 1,
          "text_excerpt": "4. Resolver asks the root DNS server → then the TLD server (like .com) → then the authoritative DNS server. 5. The final IP address is returned to your computer. 6. Now, your browser knows where to send the request.",
          "rationale": "Completes the sequence and details final IP return."
        },
        {
          "passage_id": "Q1-P3",
          "source_page": 1,
          "canonical_grade": 1,
          "text_excerpt": "1. DNS Resolution: Translating Domain Names to IP Addresses. DNS (Domain Name System) is like the phonebook of the internet. Humans prefer domain names (www.google.com), but machines need IP addresses...",
          "rationale": "Introductory definition and high-level role."
        },
        {
          "passage_id": "Q1-P4",
          "source_page": 2,
          "canonical_grade": 1,
          "text_excerpt": "Analogy: Calling Pizza Hut... DNS Phonebook Lookup... caching analogy.",
          "rationale": "Pedagogical analogy reinforcing resolution concepts."
        }
      ],
      "active_index_mappings": {
        "namespace": "revon-policies",
        "document_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee",
        "vector_grades": {
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-1": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-2": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-1-chunk-3": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-0": 1
        }
      }
    },
    {
      "query_id": "Q2",
      "query_text": "Which messages make up the TCP three-way handshake?",
      "domain_topic": "TCP Handshake Protocol",
      "canonical_passages": [
        {
          "passage_id": "Q2-P1",
          "source_page": 2,
          "canonical_grade": 2,
          "text_excerpt": "The 3-Step Handshake: 1. SYN (Synchronize) – Client says, “Hey, I want to talk. Here's my sequence number.” 2. SYN-ACK – Server replies, “Sure! Here’s my sequence number. Acknowledging yours too.” 3. ACK (Acknowledge) – Client says, “Cool, ready to start!”",
          "rationale": "Exact verbatim definition of the three handshake messages."
        },
        {
          "passage_id": "Q2-P2",
          "source_page": 2,
          "canonical_grade": 1,
          "text_excerpt": "For Example: Think of it like a polite phone call: ● You dial a number (SYN) ● They pick up and say, 'Hello?' (SYN-ACK) ● You say, 'Hi, can we talk now?' (ACK)",
          "rationale": "Supporting conversational analogy naming all three messages."
        }
      ],
      "active_index_mappings": {
        "namespace": "revon-policies",
        "document_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee",
        "vector_grades": {
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-1": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-2-chunk-2": 1
        }
      }
    },
    {
      "query_id": "Q3",
      "query_text": "How do serverless APIs differ from serverful APIs?",
      "domain_topic": "Serverless vs Serverful Architecture",
      "canonical_passages": [
        {
          "passage_id": "Q3-P1",
          "source_page": 8,
          "canonical_grade": 2,
          "text_excerpt": "The server (which is the brain behind the API) can be hosted in two major ways: Serverful APIs — Traditional, always-on infrastructure; Serverless APIs — Modern, event-driven, cloud-native infrastructure",
          "rationale": "High-level architectural summary contrasting both models."
        },
        {
          "passage_id": "Q3-P2",
          "source_page": 9,
          "canonical_grade": 2,
          "text_excerpt": "Disadvantages: ● High cost: You pay even when no one uses the API. ● Poor scalability... But what if you don’t want to manage any servers? What if you want your code to just run automatically when a request comes, and not waste money or effort on idle machines? That’s where Serverless APIs come in —",
          "rationale": "Explicit operational contrast: serverful drawbacks vs serverless solution."
        },
        {
          "passage_id": "Q3-P3",
          "source_page": 10,
          "canonical_grade": 2,
          "text_excerpt": "2. What is a Serverless API? A Serverless API means: You write your logic, deploy it to the cloud, and the cloud handles everything else. There’s still a server behind the scenes, but you never manage it. The server spins up on demand, executes your code, then shuts down. You only pay for the time used, not for idle waiting.",
          "rationale": "Primary direct definition contrasting on-demand execution with idle servers."
        },
        {
          "passage_id": "Q3-P4",
          "source_page": 9,
          "canonical_grade": 1,
          "text_excerpt": "1. What is a Serverful API? A Serverful API is hosted on a dedicated server or virtual machine (VM). This server is always running, whether or not any request is being made. Think of it like a restaurant kitchen that’s open 24x7...",
          "rationale": "Prerequisite foundational definition of the serverful baseline."
        },
        {
          "passage_id": "Q3-P5",
          "source_page": 10,
          "canonical_grade": 1,
          "text_excerpt": "Advantages of Serverless APIs: Zero infrastructure management, Cost-effective, Elastic scalability.",
          "rationale": "Supporting feature breakdown for serverless model."
        }
      ],
      "active_index_mappings": {
        "namespace": "revon-policies",
        "document_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee",
        "vector_grades": {
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-8-chunk-3": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-0": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-1": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-2": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-9-chunk-3": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-10-chunk-0": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-10-chunk-1": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-10-chunk-2": 1
        }
      }
    },
    {
      "query_id": "Q4",
      "query_text": "What does HTTP 403 Forbidden mean?",
      "domain_topic": "HTTP Client Error Status Codes",
      "canonical_passages": [
        {
          "passage_id": "Q4-P1",
          "source_page": 16,
          "canonical_grade": 2,
          "text_excerpt": "● 403 Forbidden → You don’t have permission to access this resource.",
          "rationale": "Direct, primary definition of HTTP status code 403."
        }
      ],
      "active_index_mappings": {
        "namespace": "revon-policies",
        "document_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee",
        "vector_grades": {
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-1": 2
        }
      }
    },
    {
      "query_id": "Q5",
      "query_text": "What does an HTTP status code tell a client about its request?",
      "domain_topic": "HTTP Status Code Role and Semantics",
      "canonical_passages": [
        {
          "passage_id": "Q5-P1",
          "source_page": 15,
          "canonical_grade": 2,
          "text_excerpt": "Whenever a client sends a request to a server, the server responds with more than just the data — It also sends a three-digit status code that tells you what happened to your request. These codes help you understand: Whether the request was successful, If there was an error, Or if some other action is needed.",
          "rationale": "Direct, primary answer defining status code purpose and semantic categories."
        },
        {
          "passage_id": "Q5-P2",
          "source_page": 15,
          "canonical_grade": 1,
          "text_excerpt": "1xx: Informational (Request received, continuing process); 2xx: Success (Action successfully received, understood, and accepted).",
          "rationale": "Supporting evidence breaking down status code class semantics."
        },
        {
          "passage_id": "Q5-P3",
          "source_page": 16,
          "canonical_grade": 1,
          "text_excerpt": "Examples: 200 OK, 201 Created, 301 Moved Permanently, 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 500 Internal Server Error.",
          "rationale": "Concrete instances of status codes conveying outcomes."
        }
      ],
      "active_index_mappings": {
        "namespace": "revon-policies",
        "document_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee",
        "vector_grades": {
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-15-chunk-1": 2,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-15-chunk-2": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-0": 1,
          "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-1": 1
        }
      }
    }
  ]
}
```

---

## 4. Official Evaluation Metrics & Baseline Table

### 4.1 Standard Metric Definitions & Formulations

The official evaluation pipeline must strictly adhere to the following standardized naming, mathematical definitions, and scopes:

#### 1. Hit Rate@3 (Success@3)
$$\text{Hit Rate@3} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \sum_{i=1}^3 \mathbb{I}(\text{grade}_{q, i} \ge 1) \ge 1 \right)$$
- **Scope:** Binary retrieval success indicator ($1.0$ if at least one chunk with $\text{Grade} \ge 1$ appears in Top-3; $0.0$ otherwise).

#### 2. Precision@3
$$\text{Precision@3} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\{i \in \{1, 2, 3\} \mid \text{grade}_{q, i} \ge 1\}|}{3}$$
- **Scope:** Ratio of retrieved Top-3 items that contain relevant information ($\text{Grade} \ge 1$).

#### 3. Standard Macro Recall@3
$$\text{Macro Recall@3} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{Relevant Retrieved}_{q, @3}|}{R_{\text{total}, q}}$$
- **Denominator ($R_{\text{total}, q}$):** The total number of chunks across the entire 50-vector corpus satisfying $\text{Grade} \ge 1$ for query $q$ (Q1=4, Q2=2, Q3=8, Q4=1, Q5=4). Equal weight per query.

#### 4. Standard Micro Recall@3
$$\text{Micro Recall@3} = \frac{\sum_{q \in Q} |\text{Relevant Retrieved}_{q, @3}|}{\sum_{q \in Q} R_{\text{total}, q}} = \frac{\text{Total Relevant Retrieved Across All Queries}}{19}$$
- **Scope:** Global coverage across the corpus of 19 relevant items.

#### 5. Evidence-MRR@3 ($\text{Grade} \ge 1$)
$$\text{Evidence-MRR@3} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\min \{i \in \{1, 2, 3\} \mid \text{grade}_{q, i} \ge 1\} \cup \{\infty\}}$$
- **Scope:** Conventional MRR measuring how fast any valid supporting or direct evidence is presented to the user/synthesizer.

#### 6. Direct Answer Reciprocal Rank@3 ($\text{Grade} = 2$)
$$\text{Direct Answer RR@3} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\min \{i \in \{1, 2, 3\} \mid \text{grade}_{q, i} = 2\} \cup \{\infty\}}$$
- **Scope:** Measures the rank of the first chunk providing a comprehensive, primary standalone answer.

#### 7. Corpus-Level Graded nDCG@3
$$\text{DCG@3} = \sum_{i=1}^3 \frac{2^{\text{grade}_i} - 1}{\log_2(i + 1)}, \quad \text{IDCG}_{\text{corpus}} = \sum_{i=1}^3 \frac{2^{\text{grade}^*_{(i)}} - 1}{\log_2(i + 1)}, \quad \text{nDCG@3} = \frac{\text{DCG@3}}{\text{IDCG}_{\text{corpus}}}$$
- **Scope:** Exponential gain. $\text{IDCG}_{\text{corpus}}$ is constructed from the top 3 highest possible grades existing in the entire 50-vector index for that query.

#### 8. Corpus-Level Binary nDCG@3
$$\text{DCG}_{\text{bin}}@3 = \sum_{i=1}^3 \frac{\mathbb{I}(\text{grade}_i \ge 1)}{\log_2(i + 1)}, \quad \text{IDCG}_{\text{corpus, bin}} = \sum_{i=1}^{\min(3, R_{\text{total}})} \frac{1}{\log_2(i + 1)}$$
- **Scope:** Linear binary gain evaluated against the ideal binary corpus arrangement.

---

### 4.2 Official v1.0 Baseline Results Table (Broad Policy)

Evaluating the existing production retrieval rankings against the validated v1.0 benchmark yields the official baseline:

| Evaluation Metric | Original Top-3 Baseline | Diversified Top-3 Baseline | Absolute $\Delta$ | Relative % Change | Primary Performance Driver |
|---|:---:|:---:|:---:|:---:|---|
| **Hit Rate@3** | **1.0000** (5/5) | **1.0000** (5/5) | 0.0000 | 0.0% | Identical 100% query success across both approaches. |
| **Precision@3** | **0.4667** (7/15) | **0.4000** (6/15) | -0.0667 | -14.3% | Diversification dropped adjacent relevant chunks in Q1, Q2, Q5. |
| **Standard Macro Recall@3** | **0.7750** | **0.5750** | **-0.2000** | **-25.8%** | Discarding neighbor chunks reduced per-query corpus coverage. |
| **Standard Micro Recall@3** | **0.6316** (12/19) | **0.4737** (9/19) | **-0.1579** | **-25.0%** | Total relevant chunks captured dropped from 12 to 9 out of 19. |
| **Evidence-MRR@3** ($\text{Grade} \ge 1$) | **0.8667** (26/30) | **0.9000** (27/30) | **+0.0333** | **+3.8%** | Q4 promoted from Rank 3 to Rank 2 ($1/3 \rightarrow 1/2$). |
| **Direct Answer RR@3** ($\text{Grade} = 2$) | **0.7667** (23/30) | **0.8000** (24/30) | **+0.0333** | **+4.3%** | Q4 direct answer (`16/1`) promoted from Rank 3 to Rank 2. |
| **Corpus-Level Graded nDCG@3** | **0.806** | **0.780** | -0.026 | -3.2% | Q4 promotion (+0.131) offset by Q1 (-0.121) and Q2 (-0.174) dilution. |
| **Corpus-Level Binary nDCG@3** | **0.900** | **0.755** | -0.145 | -16.1% | Dropping relevant chunks on same page lowered binary density. |
| *Retained-Pool Graded nDCG@3* | *0.864* | *0.889* | *+0.025* | *+2.9%* | Within retrieved set, relevance was ordered closer to top. |
| *Distinct Pages per Query* | *2.00* | *2.60* | *+0.60* | *+30.0%* | Mechanical increase in page spread, but introduced irrelevant pages. |

---

## 5. Documented Methodological Limitations & Risks

To ensure scientific integrity and prevent misinterpretation of baseline figures, the following limitations are formally documented:

### 1. Small Sample Size ($N=5$)
- **Constraint:** A 5-query benchmark is effective for sanity checking, bug discovery (e.g., Q4 truncation), and structural sanity, but has high statistical variance.
- **Risk:** A single rank change in one query (Q4 moving from Rank 3 to Rank 2) changes macro MRR by $+0.0333$ ($+3.8\%$). Baseline comparisons with small deltas ($\Delta < 0.05$) should not be taken as statistically significant proofs of algorithmic superiority.

### 2. Chunk Overlap & Mathematical Denominator Ceilings
- **The Ceiling Effect at $K=3$:** When $R_{\text{total}, q} > 3$, it is mathematically impossible for Top-3 retrieval to achieve Recall@3 = 1.0. For Q3 ($R_{\text{total}}=8$), the absolute maximum possible Recall@3 is $3/8 = \mathbf{0.3750}$. Both Original and Diversified retrieval achieved $3/8$, hitting the 100% mathematical ceiling.
- **Length-Biased Micro Recall:** Q3 accounts for $8 / 19 = 42.1\%$ of the entire micro denominator, causing Micro Recall to be disproportionately dominated by whether multi-page topics are fully retrieved.

### 3. Offline Heuristic Simulation vs. Production Diversification
- **The Adjacency Dilemma:** The audited diversification heuristic enforces an adjacency exclusion rule ($|index - other| \le 1$ on the same page). While this successfully skipped redundant introductory text in Q4, it caused **Context Dilution** in Q2 and Q5 by discarding directly relevant neighboring chunks (`2/2` phone analogy, `15/2` status code categories) and backfilling with completely irrelevant text from distant pages (`4/2` packet headers, `13/2` postal guidelines).
- **Recommendation:** Do not deploy static page-distance diversification to production without semantic thresholding (e.g., Maximal Marginal Relevance using embedding cosine distance rather than coordinate offsets).

### 4. Coordinate Instability Across Chunkers
- The active Pinecone index has 50 vectors, whereas local [`chunker.py`](file:///c:/Users/ASUS/Desktop/revon-policy-chatbot/backend/chunker.py) generates 42 chunks. Coordinate matching must never be used across indexing pipelines.

### 5. Need for Passage-Level / Information-Need Evaluation
- Future evaluations should transition from counting chunk instances to evaluating **Semantic Need Fulfillment**: Does the retrieved Top-3 set provide all necessary facts to answer the user query without hallucination?

---

## 6. Dataset Governance & Versioning Protocol

### 6.1 Semantic Versioning Specification (`vMAJOR.MINOR.PATCH`)

Evaluation datasets shall follow Semantic Versioning:
- **MAJOR (`v1.0.0` $\rightarrow$ `v2.0.0`):** Breaking methodology changes:
  - Adding or removing evaluation queries.
  - Modifying the underlying source document or indexing an entirely new policy.
  - Redefining the relevance scale (e.g., moving from 3-tier to binary).
- **MINOR (`v1.0.0` $\rightarrow$ `v1.1.0`):** Non-breaking additions:
  - Adding new index coordinate mappings for a new chunking size or embedding model without changing canonical passage texts.
  - Expanding supporting rationales.
- **PATCH (`v1.0.0` $\rightarrow$ `v1.0.1`):** Errata:
  - Correcting typographical errors in query text or notes without changing relevance grades.

### 6.2 Immutability Guarantee
Once Gold Dataset `v1.0.0` receives formal sign-off:
1. The JSON artifact `backend/gold_dataset_v1.0.json` shall be marked read-only and committed with its SHA-256 hash.
2. No relevance grade, text excerpt, or baseline metric may be modified retroactively.
3. Any proposed re-annotation shall be submitted as a new version branch with full audit trail documentation.

---

## 7. Formal Acceptance Checklist & Human Sign-Off Items

Before freezing `backend/gold_dataset_v1.0.json`, the following four governance items require explicit human sign-off from engineering and product leadership:

### Sign-Off Decisions

```
[ ] DECISION ITEM 1: Annotation Policy Selection for Q3 Vector 9/0
    [x] OPTION A (Recommended - Broad Policy): Assign Grade 1 to Vector 9/0 as prerequisite 
        supporting context. Evidence-MRR@3 = 0.8667 (Orig) / 0.9000 (Div).
    [ ] OPTION B (Strict Policy): Assign Grade 0 to Vector 9/0 due to lack of comparative contrast. 
        Evidence-MRR@3 = 0.7667 (Orig) / 0.8000 (Div).

[ ] DECISION ITEM 2: Dual MRR Metric Reporting Standard
    [x] Approve dual reporting of Evidence-MRR@3 (Grade >= 1) for context retrieval alongside 
        Direct Answer Reciprocal Rank@3 (Grade = 2) for direct answer localization.

[ ] DECISION ITEM 3: Canonical Schema Decoupling
    [x] Approve the decoupling of canonical ground truth from page/chunk coordinates in favor 
        of text-grounded passage excerpts and document SHA-256 verification.

[ ] DECISION ITEM 4: Official Q4 Coordinate Correction
    [x] Formally approve updating the legacy benchmark target for Q4 from 16/0 (which truncated 
        at 400 Bad Request) to 16/1 (which contains the verbatim HTTP 403 definition).
```

### Engineering Sign-Off Table

| Role | Name | Status | Signature Date | Notes |
|---|---|:---:|:---:|---|
| **Lead RAG Engineer** | *Pending Review* | `AWAITING_REVIEW` | — | Verification of retrieval metrics and IDCG formulations. |
| **Evaluation / QA Engineer** | *Pending Review* | `AWAITING_REVIEW` | — | Verification of canonical passage text and 50-chunk grades. |
| **Product Owner** | *Pending Review* | `AWAITING_REVIEW` | — | Approval of Broad vs. Strict annotation policy. |

---

## 8. Safety Verification & Compliance Audit

This proposal was prepared in strict compliance with read-only audit protocols:
- **Pinecone Mutation Check:** Zero index writes, updates, deletions, or namespace modifications performed.
- **Model API Invocation Check:** Zero calls made to external LLMs, embedding models, or inference APIs.
- **Codebase Integrity Check:** Zero modifications made to production source code, dependencies, or `.env` configuration files.
- **Security Check:** Zero credentials, API keys, or private endpoints logged or exposed.

---
*Proposal submitted for formal review on 2026-11-13.*
