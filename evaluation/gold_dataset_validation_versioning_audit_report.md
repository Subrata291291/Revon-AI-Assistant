# RAG Gold Dataset Validation & Versioning Audit Report

**Audit Date:** 2026-11-13  
**Status:** Read-Only Audit & Schema Recommendation  
**Target Document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`  
**Logical Document ID:** `20dcdd7c-a603-4a3f-b384-46081d625a73`  
**Namespace:** `revon-policies`  
**Target Document Filename:** `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`  
**Total Active Indexed Vectors:** 50  

---

## 1. Executive Summary

This report delivers a comprehensive, read-only audit of the RAG evaluation gold dataset across questions Q1–Q5. Following earlier findings that coordinate-based matching falsely penalized relevant retrievals, this audit directly inspects the **source PDF** (`Pre-Read_ Basic Understanding on API  (1)-5983.pdf`), audits the complete stored text of all **50 active Pinecone vectors**, and designs a robust, versioned evaluation dataset schema.

### Core Audit Discoveries

1. **Absence of a Formal Standalone Gold Dataset File:**  
   The project has never maintained a machine-readable gold dataset (e.g., `gold_dataset.json`). Instead, gold coordinates (`1/1`, `2/1`, `8/3`, `16/0`, `15/1`) existed only as inline text literals across markdown evaluation reports.
2. **Exhaustive 50-Vector Corpus Relevance Mapping:**  
   Every one of the 50 active Pinecone vectors was retrieved and evaluated against all five evaluation queries. Multi-chunk answer distributions were uncovered across the corpus:
   - **Q1 (DNS):** 4 relevant chunks (`1/1`, `1/2`, `1/3`, `2/0`). Primary answer is in `1/2` (Step-by-step resolution).
   - **Q2 (TCP Handshake):** 2 relevant chunks (`2/1`, `2/2`). Primary answer is in `2/1` (3-step handshake).
   - **Q3 (Serverless APIs):** 8 relevant chunks (`8/3`, `9/0`, `9/1`, `9/2`, `9/3`, `10/0`, `10/1`, `10/2`). Contrast spans across pages 8, 9, and 10.
   - **Q4 (HTTP 403):** Exactly 1 relevant chunk (`16/1`). Nominal coordinate `16/0` contains 0 evidence for 403.
   - **Q5 (HTTP Status Codes):** 4 relevant chunks (`15/1`, `15/2`, `16/0`, `16/1`). Overarching definition is in `15/1`.
3. **Physical Verification of Q4 Conflict:**  
   Source PDF extraction confirmed that `● 403 Forbidden → You don’t have permission to access this resource.` is on Page 16. In Pinecone, vector `16/0` ran out of character budget due to uncollapsed whitespace during historical ingestion and terminated at `400 Bad Request`. Vector `16/1` contains the full 403 definition. The verified active-index gold coordinate is **`16/1`**.
4. **Coordinate Fragility:**  
   Page/chunk coordinates (`p/c`) are transient implementation artifacts that shift whenever text normalization, chunk size, overlap, or PDF parsers change. A text-grounded, versioned dataset schema is proposed to decouple ground truth from indexing implementations.

---

## 2. Existing Dataset Inventory

A filesystem search confirmed that no structured dataset files (JSON, CSV, YAML) currently define evaluation labels. The existing evaluation definitions are embedded within previous reports:

| Asset / Report | Path | Role in Evaluation | Existing Gold Labels |
|---|---|---|---|
| Live RAG Retrieval Evaluation | `backend/retrieval_evaluation_report.md` | Establishes initial live retrieval results and inline gold coordinates. | Q1: `1/1`, Q2: `2/1`, Q3: `8/3`, Q4: `16/0`, Q5: `15/1` |
| Retrieval Diversity Audit v2 | `backend/retrieval_diversity_audit_corrected_report_v2.md` | Defines the candidate adjacency deduplication heuristic. | Same 5 coordinates |
| Gold Label Alignment Audit | `backend/gold_label_alignment_audit_report.md` | Verified coordinate existence; checked character similarity against local chunks. | Recommended no coordinate changes |
| Content-Aware Retrieval Evaluation | `backend/content_aware_retrieval_evaluation_report.md` | Audited coordinate proxy limitations based on short excerpts. | Same 5 coordinates |
| Full Retrieved Chunk Text Audit | `backend/full_retrieved_chunk_text_audit_report.md` | Fetched 19 Pinecone vectors; established text-level relevance. | Proved `16/0` lacks 403, `16/1` contains 403 |
| Evaluation Methodology Correction | `backend/evaluation_methodology_correction_report.md` | Formalized Hit Rate@3, nDCG scopes, and context dilution analysis. | Recommended `16/1` for active index |
| Source PDF Document | `...\Pre-Read_ Basic Understanding on API  (1)-5983.pdf` | Authoritative source text (16 pages, 32,849 raw characters). | Ground truth content source |
| Local Chunker Implementation | `backend/chunker.py` | Generates 42 chunks locally (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=120`). | Diverges from 50 Pinecone vectors |
| Pinecone Vector Store | `backend/vector_store.py` | Interfaces with index `revon-policy-index` via REST API. | Vector ID: `{doc_id}-page-{p}-chunk-{c}` |

---

## 3. Gold Passage Text Validation

Each gold passage was located in the source PDF and matched against the 50 active Pinecone vectors.

```
====================================================================================================
Q1: How does DNS translate domain names to IP addresses?
====================================================================================================
Source PDF: Page 1, lines 34–48
Authoritative Source Text:
"1. DNS Resolution: Translating Domain Names to IP Addresses
DNS (Domain Name System) is like the phonebook of the internet. Humans prefer domain names
(www.google.com), but machines need IP addresses (142.250.190.36) to route data. Why is it needed?
You can’t expect users to remember the IPs of every website. DNS allows us to use easy-to-remember
domain names. How it Works (Step-by-Step):
1. You type www.example.com into your browser.
2. The browser checks the local DNS cache (if you've visited recently).
3. If not found, it queries the recursive resolver (usually provided by your ISP).
4. Resolver asks the root DNS server → then the TLD server (like .com) → then the authoritative DNS server.
5. The final IP address is returned to your computer.
6. Now, your browser knows where to send the request."
----------------------------------------------------------------------------------------------------
Pinecone Distribution:
- Vector 1/1: Contains intro and cuts off at "Humans prefer" (Grade 1 - Partial).
- Vector 1/2: Contains phonebook analogy and Steps 1–4 (Grade 2 - Complete Mechanism).
- Vector 1/3: Contains Steps 4–6 and final IP return (Grade 1 - Supporting Sequence).
- Vector 2/0: Contains Pizza Hut phonebook/cache analogy (Grade 1 - Supporting Analogy).

====================================================================================================
Q2: Which messages make up the TCP three-way handshake?
====================================================================================================
Source PDF: Page 2, lines 12–28
Authoritative Source Text:
"2. TCP Handshake: Establishing a Reliable Connection
TCP (Transmission Control Protocol) ensures that data is sent reliably and in order. Before any
actual data transfer, a connection must be established using the three-way handshake.
The 3-Step Handshake:
1. SYN (Synchronize) – Client says, “Hey, I want to talk. Here's my sequence number.”
2. SYN-ACK – Server replies, “Sure! Here’s my sequence number. Acknowledging yours too.”
3. ACK (Acknowledge) – Client says, “Cool, ready to start!”
Only after this process can data transfer begin.
For Example: Think of it like a polite phone call:
● You dial a number (SYN)
● They pick up and say, "Hello?" (SYN-ACK)
● You say, “Hi, can we talk now?” (ACK)
● Now the conversation starts."
----------------------------------------------------------------------------------------------------
Pinecone Distribution:
- Vector 2/1: Exact verbatim match for 3-Step Handshake: SYN, SYN-ACK, ACK (Grade 2 - Complete).
- Vector 2/2: Contains phone call analogy naming SYN, SYN-ACK, ACK (Grade 1 - Supporting Evidence).

====================================================================================================
Q3: How do serverless APIs differ from serverful APIs?
====================================================================================================
Source PDF: Pages 8, 9, 10
Authoritative Source Text:
Page 8: "The server (which is the brain behind the API) can be hosted in two major ways:
Serverful APIs — Traditional, always-on infrastructure
Serverless APIs — Modern, event-driven, cloud-native infrastructure"
Page 9: "1. What is a Serverful API? A Serverful API is hosted on a dedicated server or virtual
machine (VM). This server is always running, whether or not any request is being made... Disadvantages:
High cost: You pay even when no one uses the API. Poor scalability... Higher responsibility...
But what if you don’t want to manage any servers? What if you want your code to just run automatically
when a request comes, and not waste money or effort on idle machines? That’s where Serverless APIs come in —"
Page 10: "2. What is a Serverless API? A Serverless API means: You write your logic, deploy it to the
cloud, and the cloud handles everything else. There’s still a server behind the scenes, but you never
manage it. The server spins up on demand, executes your code, then shuts down. You only pay for the time
used, not for idle waiting. Advantages: Zero infrastructure management... Cost-effective... Elastic scalability..."
----------------------------------------------------------------------------------------------------
Pinecone Distribution:
- Vector 8/3: High-level definitions of both models (Grade 2 - Complete Summary).
- Vector 9/0: Detailed definition of Serverful API (Grade 1 - Supporting Definition).
- Vector 9/1: Serverful architecture and advantages (Grade 1 - Supporting Evidence).
- Vector 9/2: Serverful drawbacks directly contrasted with Serverless (Grade 2 - Complete Contrast).
- Vector 9/3: Transition sentence on idle machines vs serverless (Grade 1 - Supporting Fragment).
- Vector 10/0: Definition of Serverless API and on-demand vs idle contrast (Grade 2 - Complete Definition).
- Vector 10/1: Advantages of Serverless APIs (Grade 1 - Supporting Evidence).
- Vector 10/2: Infrastructure summary and trade-off choice (Grade 1 - Supporting Summary).

====================================================================================================
Q4: What does HTTP 403 Forbidden mean?
====================================================================================================
Source PDF: Page 16, line 14
Authoritative Source Text:
"● 403 Forbidden → You don’t have permission to access this resource."
----------------------------------------------------------------------------------------------------
Pinecone Distribution:
- Vector 16/0: Terminated at "400 Bad Request" (Grade 0 - Irrelevant / Missing Evidence).
- Vector 16/1: Contains "● 403 Forbidden → You don’t have permission to access this resource." (Grade 2 - Complete).

====================================================================================================
Q5: What does an HTTP status code tell a client about its request?
====================================================================================================
Source PDF: Page 15, lines 16–25
Authoritative Source Text:
"What is an HTTP Status Code?
Whenever a client (like your browser or Python script) sends a request to a server, the server responds
with more than just the data — It also sends a three-digit status code that tells you what happened
to your request.
These codes help you understand:
● Whether the request was successful
● If there was an error
● Or if some other action is needed"
----------------------------------------------------------------------------------------------------
Pinecone Distribution:
- Vector 15/1: Verbatim definition of status codes and 3 categories of information (Grade 2 - Complete).
- Vector 15/2: 1xx and 2xx class explanations (Grade 1 - Supporting Evidence).
- Vector 16/0: Examples of 200, 201, 301, 302, 400 conveying status (Grade 1 - Supporting Evidence).
- Vector 16/1: Examples of 401, 403, 404, 500, 503 conveying status (Grade 1 - Supporting Evidence).
```

---

## 4. Q1–Q5 Corpus-Wide Relevance Evidence Table

Every vector across the 50-chunk Pinecone index containing relevant evidence is cataloged below:

| Question | Vector ID | Coord | Grade | Evidence from Stored Vector Text |
|---|---|:---:|:---:|---|
| **Q1** | `...-page-1-chunk-1` | `1/1` | 1 | Introduces DNS resolution and phonebook metaphor; cuts off at "Humans prefer". |
| **Q1** | `...-page-1-chunk-2` | `1/2` | **2** | Explains why DNS is needed and details Steps 1–4 of translation (cache $\rightarrow$ resolver $\rightarrow$ root server). |
| **Q1** | `...-page-1-chunk-3` | `1/3` | 1 | Details Steps 4–6: authoritative DNS lookup and final IP return to computer. |
| **Q1** | `...-page-2-chunk-0` | `2/0` | 1 | Supporting real-world analogy: Pizza Hut directory lookup compared to DNS cache/resolver. |
| **Q2** | `...-page-2-chunk-1` | `2/1` | **2** | Verbatim complete answer: lists `1. SYN`, `2. SYN-ACK`, and `3. ACK` with sequence number definitions. |
| **Q2** | `...-page-2-chunk-2` | `2/2` | 1 | Supporting conversational analogy: Phone call sequence explicitly naming `SYN`, `SYN-ACK`, and `ACK`. |
| **Q3** | `...-page-8-chunk-3` | `8/3` | **2** | Explicit structural contrast: Serverful (traditional, always-on) vs. Serverless (event-driven, cloud-native). |
| **Q3** | `...-page-9-chunk-0` | `9/0` | 1 | Defines Serverful API: dedicated server/VM, always running 24x7, manual infra setup. |
| **Q3** | `...-page-9-chunk-1` | `9/1` | 1 | Serverful implementation details: hosting local LLMs, full hardware control, warm latency. |
| **Q3** | `...-page-9-chunk-2` | `9/2` | **2** | Explicit comparative contrast: Serverful disadvantages (idle cost, manual scaling) vs. Serverless on-demand code execution. |
| **Q3** | `...-page-9-chunk-3` | `9/3` | 1 | Supporting fragment: contrasts wasted effort on idle machines with modern Serverless LLM backbones. |
| **Q3** | `...-page-10-chunk-0` | `10/0` | **2** | Defines Serverless API: unmanaged server, spins up on demand, executes, shuts down; pay only for time used. |
| **Q3** | `...-page-10-chunk-1` | `10/1` | 1 | Supporting advantages: zero infra management, pay-per-use, elastic scalability, cold-start trade-off. |
| **Q3** | `...-page-10-chunk-2` | `10/2` | 1 | Supporting summary: comparing serverful vs. serverless trade-offs based on use case and cost. |
| **Q4** | `...-page-16-chunk-1` | `16/1` | **2** | Verbatim authoritative definition: `● 403 Forbidden → You don’t have permission to access this resource.` |
| **Q5** | `...-page-15-chunk-1` | `15/1` | **2** | Verbatim complete answer: 3-digit status code telling client what happened to request (success, error, action). |
| **Q5** | `...-page-15-chunk-2` | `15/2` | 1 | Supporting range evidence: explains how 1xx (processing) and 2xx (success) communicate request status. |
| **Q5** | `...-page-16-chunk-0` | `16/0` | 1 | Supporting examples: 200 (successful), 201 (created), 3xx (redirection), 4xx (client error). |
| **Q5** | `...-page-16-chunk-1` | `16/1` | 1 | Supporting examples: 401 (credentials needed), 403 (forbidden), 404 (not found), 5xx (server error). |

*All other 31 vectors in the 50-chunk index are verified Grade 0 (Irrelevant) for all five questions.*

---

## 5. Q4 Gold Coordinate Conflict: Fact vs. Hypothesis

A rigorous audit demands distinguishing physically verified facts from historical hypotheses:

```
+--------------------------------------------------------------------------------------------------+
|                                     VERIFIED FACTS                                               |
+--------------------------------------------------------------------------------------------------+
| 1. Source PDF Page 16 contains:                                                                  |
|    "● 403 Forbidden → You don’t have permission to access this resource."                       |
| 2. Pinecone Vector 16/0 (doc ab56ef89-c787-470c-b796-7d4b9fcdb2ee) has 793 raw characters:      |
|    Begins at "● 200 OK" and ends at "● 400 Bad Request → The request was malformed or missing". |
|    Substrings "403" and "Forbidden" are ABSENT (Checked exact boolean: False).                  |
| 3. Pinecone Vector 16/1 (doc ab56ef89-c787-470c-b796-7d4b9fcdb2ee) has 788 raw characters:      |
|    Contains the exact definition: "● 403 Forbidden → You don’t have permission to access...".   |
| 4. Current backend/chunker.py produces 42 chunks for the same PDF.                               |
|    Local Chunk 16/0 has 748 characters and DOES contain the 403 Forbidden definition.           |
+--------------------------------------------------------------------------------------------------+
|                                  HISTORICAL HYPOTHESIS                                           |
+--------------------------------------------------------------------------------------------------+
| Why did early reports assign gold coordinate 16/0?                                               |
| - The evaluator who authored the initial gold coordinate was viewing either:                     |
|   a) The current local chunker output (where 16/0 fits 403), or                                 |
|   b) A preliminary segmentation build that had collapsed whitespace prior to chunking.          |
| - When vectors were actually ingested into Pinecone, raw text with uncollapsed whitespace was   |
|   used. The character budget was consumed early, pushing 403 into vector 16/1.                  |
| - Later, gold_label_alignment_audit_report.md evaluated coordinate 16/0 using string similarity  |
|   against local chunk 16/0 (scoring 0.8827), erroneously inferring that stored vector 16/0      |
|   contained the gold passage without verifying the substring.                                    |
+--------------------------------------------------------------------------------------------------+
```

### Recommendation
For all evaluations querying the **active Pinecone index**, the verified gold coordinate for Q4 is **`16/1`**.

---

## 6. Coordinate Stability Risks

Using page/chunk coordinates (e.g., `16/0` or `1/1`) as permanent ground truth labels introduces severe structural fragility:

```
                                  VULNERABILITY SURFACE
  
   Source PDF Text -------------> Chunking Pipeline -------------> Vector Store Coordinates
  ("403 Forbidden")                 (Parameters)                     (p16/c0 vs p16/c1)
                                         |
                       +-----------------+-----------------+
                       | Changes in:                       |
                       | - Normalization (whitespace/tabs) |
                       | - Chunk size (800 -> 1000)        |
                       | - Chunk overlap (120 -> 150)      |
                       | - PDF parser (pypdf -> PyMuPDF)   |
                       | - Document revisions              |
                       +-----------------------------------+
```

### Fragility Drivers

1. **Whitespace Normalization Drift:**  
   Raw PDF text contains formatting spaces, linebreaks, and non-breaking spaces. Normalizing `\s+` to ` ` compresses text by 15–30%. A chunking window of 800 characters will span entirely different sentence boundaries depending on whether normalization occurs before or after windowing.
2. **Chunker Parameter Sensitivity:**  
   Altering `CHUNK_SIZE` from 800 to 1,000 or `CHUNK_OVERLAP` from 120 to 150 immediately shifts every coordinate downstream of page 1.
3. **Sentence Boundary Heuristics:**  
   If the boundary detector changes how it handles abbreviations (`e.g.`, `i.e.`) or bullet lists (`●`), chunk start/end offsets drift, renumbering chunk indices.
4. **Document Versioning:**  
   Inserting a single introductory paragraph on page 1 shifts every subsequent page's chunk indices, rendering coordinate labels invalid.

**Conclusion:** Ground truth must be anchored to **semantic text passages and document identifiers**, never to dynamic indexing coordinates.

---

## 7. Proposed Gold Dataset Schema

To eliminate coordinate fragility, the evaluation dataset must be formalized as a versioned, machine-readable JSON schema. Coordinates are retained strictly as optional, index-specific lookup caches.

### 7.1 JSON Schema Definition (`rag_gold_dataset_schema.json`)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RAGGoldDataset",
  "type": "object",
  "required": [
    "dataset_version",
    "schema_version",
    "document_lineage",
    "evaluation_questions"
  ],
  "properties": {
    "dataset_version": { "type": "string" },
    "schema_version": { "type": "string" },
    "created_at": { "type": "string", "format": "date-time" },
    "document_lineage": {
      "type": "object",
      "required": ["logical_document_id", "filename", "document_hash"],
      "properties": {
        "logical_document_id": { "type": "string" },
        "filename": { "type": "string" },
        "document_hash": { "type": "string" }
      }
    },
    "evaluation_questions": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "question_id",
          "query_text",
          "canonical_gold_passages",
          "relevance_judgments"
        ],
        "properties": {
          "question_id": { "type": "string" },
          "query_text": { "type": "string" },
          "canonical_gold_passages": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["passage_id", "source_page", "exact_text"],
              "properties": {
                "passage_id": { "type": "string" },
                "source_page": { "type": "integer" },
                "exact_text": { "type": "string" }
              }
            }
          },
          "relevance_judgments": {
            "type": "array",
            "items": {
              "type": "object",
              "required": ["target_index_version", "judged_chunks"],
              "properties": {
                "target_index_version": { "type": "string" },
                "judged_chunks": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "required": ["vector_id", "page_number", "chunk_index", "grade", "rationale"],
                    "properties": {
                      "vector_id": { "type": "string" },
                      "page_number": { "type": "integer" },
                      "chunk_index": { "type": "integer" },
                      "grade": { "type": "integer", "enum": [0, 1, 2] },
                      "rationale": { "type": "string" }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
```

### 7.2 Concrete Sample Record (Q4 — HTTP 403 Forbidden)

```json
{
  "question_id": "Q4",
  "query_text": "What does HTTP 403 Forbidden mean?",
  "canonical_gold_passages": [
    {
      "passage_id": "Q4-P1",
      "source_page": 16,
      "exact_text": "● 403 Forbidden → You don’t have permission to access this resource."
    }
  ],
  "relevance_judgments": [
    {
      "target_index_version": "revon-policies-v1-pinecone-50chunks",
      "judged_chunks": [
        {
          "vector_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-1",
          "page_number": 16,
          "chunk_index": 1,
          "grade": 2,
          "rationale": "Contains exact verbatim canonical gold passage Q4-P1."
        },
        {
          "vector_id": "ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-0",
          "page_number": 16,
          "chunk_index": 0,
          "grade": 0,
          "rationale": "Terminated at 400 Bad Request; zero mention of 403 Forbidden."
        }
      ]
    }
  ]
}
```

---

## 8. Dataset Versioning Strategy

To maintain evaluation integrity across code and index iterations, the gold dataset must follow Semantic Versioning (`MAJOR.MINOR.PATCH`):

```
MAJOR Version (v2.0.0):
- Triggered by changes to Evaluation Questions, Question Semantics, or Canonical Source Documents.
- Breaking change: Historical evaluation runs are not directly comparable.

MINOR Version (v1.1.0):
- Triggered by adding new Target Index Mappings (e.g., adding judgments for a new 42-chunk index),
  or expanding relevance judgments across unjudged chunks.
- Non-breaking: Preserves existing question definitions and existing index judgments.

PATCH Version (v1.0.1):
- Triggered by typographical corrections to rationale text, metadata field updates, or
  correcting documented coordinate references (e.g., updating Q4 index reference from 16/0 to 16/1).
```

### Change Review & Governance Workflow

1. **Immutable Historical Tags:** Every released dataset file (e.g., `gold_dataset_v1.0.0.json`) is read-only and immutable.
2. **Pull Request Protocol:** Any change to questions or grades requires an evaluation PR with:
   - Diff of proposed label changes.
   - Evidence citations from the source document.
   - Dual-reviewer sign-off (IR engineer and domain expert).
3. **Audit Trail:** An append-only `CHANGELOG.md` inside `backend/evaluation/` logging the timestamp, author, justification, and version bump.

---

## 9. Future Evaluation Methodology & Metrics

With an exhaustive corpus mapping ($R_{\text{total}}$ known per question), the evaluation system can compute both standard and pool-level metrics:

### Metric Definitions & Formulations

| Metric | Formula | Required Denominator | Practical Meaning for This Corpus |
|---|---|---|---|
| **Hit Rate@3** | $\mathbb{I}\left( \sum_{i=1}^3 rel_i \ge 1 \right)$ | N/A (binary indicator) | Did the user receive at least one answer in Top-3? (100% for both rankings). |
| **Precision@3** | $\frac{1}{3} \sum_{i=1}^3 rel_i$ | 3 (fixed rank cutoff) | Fraction of Top-3 slots containing verified answer evidence. |
| **Standard Recall@3** | $\frac{1}{R_{\text{total}}} \sum_{i=1}^3 rel_i$ | **$R_{\text{total}}$ (Total relevant chunks in corpus)** | Fraction of *all* corpus answers captured. ($R_{\text{total}}$: Q1=4, Q2=2, Q3=8, Q4=1, Q5=4). |
| **MRR@3** | $\frac{1}{\min \{ i \le 3 \mid rel_i = 1 \}}$ | First relevant rank | Speed/rank of first useful answer (0.767 Orig vs 0.800 Div). |
| **Retained-Pool nDCG@3** | $\frac{\text{DCG}@3}{\text{IDCG}_{\text{retrieved}}@3}$ | $\text{IDCG}$ of the 3 retrieved chunks | How well ordered are the 3 chunks that were retrieved? |
| **Corpus-Level nDCG@3** | $\frac{\text{DCG}@3}{\text{IDCG}_{\text{corpus}}@3}$ | $\text{IDCG}$ of top 3 grades in full corpus | How close is Top-3 to the absolute best 3 chunks in the 50-chunk document? |

### Denominator Demonstration: Standard Recall@3 vs. Hit Rate@3

Now that $R_{\text{total}}$ is known across all 50 vectors, we can calculate true standard Recall@3:

| Question | Total Relevant in Corpus ($R_{\text{total}}$) | Original Top-3 Relevant Retrieved | Original Standard Recall@3 | Diversified Top-3 Relevant Retrieved | Diversified Standard Recall@3 | Hit Rate@3 (Both) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1 (DNS)** | 4 (`1/1`, `1/2`, `1/3`, `2/0`) | 3 (`1/2`, `1/3`, `2/0`) | 3/4 = **0.750** | 2 (`1/2`, `2/0`) | 2/4 = **0.500** | **1.000** |
| **Q2 (TCP)** | 2 (`2/1`, `2/2`) | 2 (`2/1`, `2/2`) | 2/2 = **1.000** | 1 (`2/1`) | 1/2 = **0.500** | **1.000** |
| **Q3 (Serverless)** | 8 (`8/3`, `9/0-3`, `10/0-2`) | 3 (`9/0`, `10/0`, `10/1`) | 3/8 = **0.375** | 3 (`9/0`, `10/0`, `9/2`) | 3/8 = **0.375** | **1.000** |
| **Q4 (HTTP 403)** | 1 (`16/1`) | 1 (`16/1`) | 1/1 = **1.000** | 1 (`16/1`) | 1/1 = **1.000** | **1.000** |
| **Q5 (Status Codes)**| 4 (`15/1`, `15/2`, `16/0`, `16/1`)| 3 (`15/1`, `15/2`, `16/0`)| 3/4 = **0.750** | 2 (`15/1`, `16/0`) | 2/4 = **0.500** | **1.000** |
| **Macro Average** | — | — | **0.775** | — | **0.575** | **1.000** |

*Notice the clarity this brings:* Hit Rate@3 is 1.000 for both rankings. Standard Recall@3 shows that Original retrieval retrieved 77.5% of all relevant chunks across the document, while Diversification retrieved 57.5% because it intentionally skipped adjacent duplicates (`2/2`, `15/2`, `1/3`) to provide cross-page diversity.

---

## 10. Unresolved Issues and Manual Review Requirements

1. **Granularity of Q3 Relevance:**  
   Question Q3 asks *"How do serverless APIs differ from serverful APIs?"*  
   Vectors `9/0` and `9/1` describe *only* serverful APIs. In this audit, they are marked Grade 1 because they provide the baseline for comparison. However, if a strict standard requires that a chunk must mention *both* sides of the comparison to be relevant, `9/0` and `9/1` could be reclassified as Grade 0. A formal product policy decision is required.
2. **Chunk `1/3` Sentence Fragment:**  
   Chunk `1/3` begins mid-sentence: `(usually provided by your ISP). 4. Resolver asks...`. While it contains technical resolution steps (Grade 1), returning sentence fragments to users harms readability. A policy on whether sentence-truncated chunks should be penalized in evaluation must be decided.
3. **Index Re-Ingestion Synchronization:**  
   The active Pinecone index contains 50 vectors, while the local repository code produces 42 chunks. When the index is re-ingested under the modern chunker, all 50 coordinates will shift to 42 new coordinates. The proposed versioned schema must be deployed before re-ingestion so historical evaluations remain valid.

---

## 11. Final Recommendations

1. **Deploy the Proposed Versioned JSON Dataset:**  
   Create `backend/evaluation/gold_dataset_v1.0.0.json` conforming to the schema in Section 7.
2. **Correct Active Index Reference for Q4:**  
   Officially register vector `16/1` as the active-index target for Q4 while preserving documentation of the `16/0` whitespace drift.
3. **Decouple Evaluation Harness from Coordinates:**  
   Refactor evaluation scripts to match retrieved text against `canonical_gold_passages` (using exact substring or semantic token overlap) rather than matching string coordinates.
4. **Standardize Reporting Nomenclature:**  
   Adopt **Hit Rate@K** for single-hit success, and report **Standard Recall@K** only with documented corpus denominators ($R_{\text{total}}$).

---

## 12. Safety Verification Checklist

All tasks in this validation and versioning audit strictly adhered to the read-only mandate:

- **No Pinecone write operations:** 100% read-only GET `/vectors/fetch` and `/vectors/list`. Zero upserts, updates, deletes, or index operations performed.
- **Zero API calls:** Zero calls to Gemini, Groq, OpenAI, or embedding/LLM APIs.
- **Zero source code modifications:** Application files (`main.py`, `chunker.py`, `retrieval_service.py`) were untouched.
- **Zero dataset or configuration modifications:** Existing evaluation markdown reports and `.env` were preserved without edits.
- **Zero dependency changes:** No packages installed or modified.
- **Credentials protected:** All API keys, environment secrets, and connection strings remained strictly confidential.

---
*Report completed on 2026-11-13.*
