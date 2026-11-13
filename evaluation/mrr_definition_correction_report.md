# Final MRR Definition and Calculation Audit Report

**Audit Date:** 2026-11-13  
**Status:** Strictly Read-Only Metric Audit  
**Target Document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`  
**Namespace:** `revon-policies`  
**Reference Report:** `backend/gold_dataset_final_metric_consistency_report.md`  

---

## 1. Executive Summary

This audit resolves the definitional ambiguity surrounding Mean Reciprocal Rank (MRR) across previous evaluation reports. Specifically, it clarifies the distinction between:
1. **Conventional MRR@3:** Evaluates the reciprocal rank of the first retrieved chunk containing *any verified answer evidence* ($\text{Grade} \ge 1$, including supporting analogies and partial steps).
2. **Direct Answer Reciprocal Rank@3:** Evaluates the reciprocal rank of the first retrieved chunk containing a *complete, standalone answer* ($\text{Grade} = 2$).

### Core Findings & Verification

1. **Origin of Previously Reported Values (0.7667 vs. 0.8000):**  
   The previously reported values of **0.7667** (Original) and **0.8000** (Diversified) in earlier audit stages originated from **Conventional MRR@3** under the initial conservative labeling policy where non-gold coordinate chunks (`1/2` for Q1, `10/0` for Q3, and `16/1` for Q4) were conservatively capped at Grade 1.
2. **Convergence under Validated Text Labels:**  
   When the full 50-chunk validation established that chunks `1/2` (Q1), `10/0` (Q3), and `16/1` (Q4) contain the primary direct answers (Grade 2):
   - **Direct Answer Reciprocal Rank@3** evaluates to **0.7667** for Original Top-3 and **0.8000** for Diversified Top-3.
   - **Conventional MRR@3** evaluates to **0.8667** for Original Top-3 and **0.9000** for Diversified Top-3 (because for Q3, Rank 1 chunk `9/0` provides Grade 1 supporting context).
3. **Diversification Ranking Lift Verified:**  
   Under both metric definitions, diversification produces a net Macro Reciprocal Rank improvement of **+0.0333**. This gain is driven entirely by **Q4 (HTTP 403)**, where skipping redundant neighbor `15/2` promoted the primary answer chunk `16/1` from **Rank 3 to Rank 2** (increasing RR from $1/3$ to $1/2$).

---

## 2. Existing Relevance Policy & MRR Formula Review

### 2.1 Relevance Grading Scale

The evaluation framework defines three discrete relevance tiers:
- **Grade 0 (Irrelevant):** The chunk contains no substantive evidence addressing the query.
- **Grade 1 (Relevant Supporting Evidence):** The chunk provides valid supporting evidence, components of a multi-step answer, or concrete analogies, but lacks the standalone primary answer.
- **Grade 2 (Direct / Complete Answer):** The chunk directly answers the question or contains the complete technical mechanism required by the query.

### 2.2 Mathematical Formulations

#### Metric A: Conventional MRR@K ($\text{Grade} \ge 1$)
In standard Information Retrieval, any document meeting a minimum relevance threshold is considered relevant:
$$rel_i = \mathbb{I}(g_i \ge 1)$$
$$\text{RR@K}_{\text{conv}} = \begin{cases} \frac{1}{\min \{ i \le K \mid rel_i = 1 \}} & \text{if } \sum_{i=1}^K rel_i \ge 1 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Conventional MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR@K}_{\text{conv}, q}$$

#### Metric B: Direct Answer Reciprocal Rank@K ($\text{Grade} = 2$)
In question-answering and RAG applications, the user's primary information need is satisfied only when the direct answer is retrieved:
$$direct_i = \mathbb{I}(g_i = 2)$$
$$\text{RR@K}_{\text{direct}} = \begin{cases} \frac{1}{\min \{ i \le K \mid direct_i = 1 \}} & \text{if } \sum_{i=1}^K direct_i \ge 1 \\ 0 & \text{otherwise} \end{cases}$$
$$\text{Direct Answer Reciprocal Rank@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR@K}_{\text{direct}, q}$$

---

## 3. Per-Question Candidate Rankings & Grade Distribution

The table below lists the retrieved Top-3 candidates for both rankings, their text-validated relevance grades, and their exact evidence roles:

### 3.1 Original Top-3 (Live Pinecone Similarity Ranking)

| Query | Rank | Coordinate | Vector ID | Score | Grade | Content / Evidence Role |
|---|:---:|:---:|---|:---:|:---:|---|
| **Q1 (DNS)** | 1 | `1/2` | `...-page-1-chunk-2` | 0.8208 | **2** | **Direct Answer:** Step-by-step DNS translation mechanism (cache $\rightarrow$ resolver $\rightarrow$ root server). |
| | 2 | `1/3` | `...-page-1-chunk-3` | 0.7795 | 1 | **Supporting:** Steps 4–6 completing the resolution and IP return. |
| | 3 | `2/0` | `...-page-2-chunk-0` | 0.7247 | 1 | **Supporting:** Real-world Pizza Hut directory and cache analogy. |
| **Q2 (TCP)** | 1 | `2/1` | `...-page-2-chunk-1` | 0.7770 | **2** | **Direct Answer:** Formal 3-step handshake definition (`SYN`, `SYN-ACK`, `ACK`). |
| | 2 | `2/2` | `...-page-2-chunk-2` | 0.7470 | 1 | **Supporting:** Conversational phone call analogy explicitly naming the 3 messages. |
| | 3 | `6/0` | `...-page-6-chunk-0` | 0.7052 | 0 | **Irrelevant:** 4-step FIN-ACK connection teardown. |
| **Q3 (Serverless)** | 1 | `9/0` | `...-page-9-chunk-0` | 0.8146 | 1 | **Supporting:** Detailed definition of Serverful APIs (dedicated server/VM, 24x7). |
| | 2 | `10/0` | `...-page-10-chunk-0` | 0.7864 | **2** | **Direct Answer:** Defines Serverless API (unmanaged, on-demand spin-up, pay-per-use). |
| | 3 | `10/1` | `...-page-10-chunk-1` | 0.7735 | 1 | **Supporting:** Advantages and operational trade-offs of Serverless APIs. |
| **Q4 (HTTP 403)** | 1 | `15/1` | `...-page-15-chunk-1` | 0.7336 | 0 | **Irrelevant:** General definition of status codes; no 403 mention. |
| | 2 | `15/2` | `...-page-15-chunk-2` | 0.7239 | 0 | **Irrelevant:** 1xx Informational and 2xx Success status codes. |
| | 3 | `16/1` | `...-page-16-chunk-1` | 0.7145 | **2** | **Direct Answer:** Verbatim definition: `● 403 Forbidden → You don’t have permission to access this resource.` |
| **Q5 (Status Codes)** | 1 | `15/1` | `...-page-15-chunk-1` | 0.8050 | **2** | **Direct Answer:** Verbatim definition of status codes communicating request outcomes. |
| | 2 | `15/2` | `...-page-15-chunk-2` | 0.7722 | 1 | **Supporting:** 1xx (processing) and 2xx (success) class meanings. |
| | 3 | `16/0` | `...-page-16-chunk-0` | 0.7458 | 1 | **Supporting:** Specific code examples (200, 201, 301, 302, 400). |

---

### 3.2 Diversified Top-3 (Adjacency Deduplication Heuristic)

| Query | Rank | Coordinate | Vector ID | Score | Grade | Content / Evidence Role |
|---|:---:|:---:|---|:---:|:---:|---|
| **Q1 (DNS)** | 1 | `1/2` | `...-page-1-chunk-2` | 0.8208 | **2** | **Direct Answer:** Step-by-step DNS translation mechanism. |
| | 2 | `2/0` | `...-page-2-chunk-0` | 0.7247 | 1 | **Supporting:** Pizza Hut directory analogy. |
| | 3 | `1/0` | `...-page-1-chunk-0` | 0.7043 | 0 | **Irrelevant:** API introductory hook; stops before DNS. |
| **Q2 (TCP)** | 1 | `2/1` | `...-page-2-chunk-1` | 0.7770 | **2** | **Direct Answer:** Formal 3-step handshake definition (`SYN`, `SYN-ACK`, `ACK`). |
| | 2 | `6/0` | `...-page-6-chunk-0` | 0.7052 | 0 | **Irrelevant:** 4-step FIN-ACK connection teardown. |
| | 3 | `4/2` | `...-page-4-chunk-2` | 0.6984 | 0 | **Irrelevant:** Packet sequence numbers and general ACKs. |
| **Q3 (Serverless)** | 1 | `9/0` | `...-page-9-chunk-0` | 0.8146 | 1 | **Supporting:** Detailed definition of Serverful APIs. |
| | 2 | `10/0` | `...-page-10-chunk-0` | 0.7864 | **2** | **Direct Answer:** Defines Serverless API (on-demand vs. idle). |
| | 3 | `9/2` | `...-page-9-chunk-2` | 0.7644 | **2** | **Direct Answer:** Explicit comparative contrast between Serverful and Serverless. |
| **Q4 (HTTP 403)** | 1 | `15/1` | `...-page-15-chunk-1` | 0.7336 | 0 | **Irrelevant:** General definition of status codes. |
| | 2 | `16/1` | `...-page-16-chunk-1` | 0.7145 | **2** | **Direct Answer:** `● 403 Forbidden → You don’t have permission...` (Promoted from Rank 3). |
| | 3 | `13/2` | `...-page-13-chunk-2` | 0.6903 | 0 | **Irrelevant:** HTTP protocol postal guidelines. |
| **Q5 (Status Codes)** | 1 | `15/1` | `...-page-15-chunk-1` | 0.8050 | **2** | **Direct Answer:** Status codes communicating request outcomes. |
| | 2 | `16/0` | `...-page-16-chunk-0` | 0.7458 | 1 | **Supporting:** Specific code examples (200, 3xx, 400). |
| | 3 | `13/2` | `...-page-13-chunk-2` | 0.7439 | 0 | **Irrelevant:** HTTP protocol postal guidelines. |

---

## 4. Conventional MRR@3 Results ($\text{Grade} \ge 1$)

Under Conventional MRR@3, a query is satisfied by the first chunk meeting $\text{Grade} \ge 1$:

| Query ID | Topic | Original Top-3 Grades | Original First Rel Rank | Original RR@3 | Diversified Top-3 Grades | Diversified First Rel Rank | Diversified RR@3 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | DNS Translation | `[2, 1, 1]` | Rank 1 | **1.0000** | `[2, 1, 0]` | Rank 1 | **1.0000** |
| **Q2** | TCP Handshake | `[2, 1, 0]` | Rank 1 | **1.0000** | `[2, 0, 0]` | Rank 1 | **1.0000** |
| **Q3** | Serverless APIs | `[1, 2, 1]` | Rank 1 | **1.0000** | `[1, 2, 2]` | Rank 1 | **1.0000** |
| **Q4** | HTTP 403 Forbidden | `[0, 0, 2]` | Rank 3 | **0.3333** | `[0, 2, 0]` | Rank 2 | **0.5000** |
| **Q5** | Status Codes | `[2, 1, 1]` | Rank 1 | **1.0000** | `[2, 1, 0]` | Rank 1 | **1.0000** |
| **Macro Average** | — | — | — | **0.8667** | — | — | **0.9000** |

$$\text{Macro MRR@3}_{\text{conv}} (\text{Original}) = \frac{1.0 + 1.0 + 1.0 + 0.3333 + 1.0}{5} = \frac{4.3333}{5} = \mathbf{0.8667}$$
$$\text{Macro MRR@3}_{\text{conv}} (\text{Diversified}) = \frac{1.0 + 1.0 + 1.0 + 0.5000 + 1.0}{5} = \frac{4.5000}{5} = \mathbf{0.9000}$$

---

## 5. Direct Answer Reciprocal Rank@3 Results ($\text{Grade} = 2$)

Under Direct Answer Reciprocal Rank@3, reciprocal rank is determined strictly by the first chunk containing the complete/direct answer ($\text{Grade} = 2$):

| Query ID | Topic | Original Top-3 Grades | Original First Direct Rank | Original Direct RR@3 | Diversified Top-3 Grades | Diversified First Direct Rank | Diversified Direct RR@3 |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | DNS Translation | `[2, 1, 1]` | Rank 1 | **1.0000** | `[2, 1, 0]` | Rank 1 | **1.0000** |
| **Q2** | TCP Handshake | `[2, 1, 0]` | Rank 1 | **1.0000** | `[2, 0, 0]` | Rank 1 | **1.0000** |
| **Q3** | Serverless APIs | `[1, 2, 1]` | Rank 2 | **0.5000** | `[1, 2, 2]` | Rank 2 | **0.5000** |
| **Q4** | HTTP 403 Forbidden | `[0, 0, 2]` | Rank 3 | **0.3333** | `[0, 2, 0]` | Rank 2 | **0.5000** |
| **Q5** | Status Codes | `[2, 1, 1]` | Rank 1 | **1.0000** | `[2, 1, 0]` | Rank 1 | **1.0000** |
| **Macro Average** | — | — | — | **0.7667** | — | — | **0.8000** |

$$\text{Macro Direct Answer RR@3} (\text{Original}) = \frac{1.0 + 1.0 + 0.5 + 0.3333 + 1.0}{5} = \frac{3.8333}{5} = \mathbf{0.7667}$$
$$\text{Macro Direct Answer RR@3} (\text{Diversified}) = \frac{1.0 + 1.0 + 0.5 + 0.5000 + 1.0}{5} = \frac{4.0000}{5} = \mathbf{0.8000}$$

---

## 6. Verification & Explanation of Previously Reported Values

### 6.1 What Did 0.7667 and 0.8000 Represent?

The previously reported values:
- **Original: 0.7667**
- **Diversified: 0.8000**

represent **both** of the following depending on the evaluation stage:

1. **In Initial Audit Reports (`full_retrieved_chunk_text_audit_report.md`):**  
   These values were calculated as **Conventional MRR@3 ($\text{Grade} \ge 1$)** under an initial conservative grading policy:
   - Chunk `1/2` was marked Grade 1 (conservative, because it did not equal nominal coordinate `1/1`).
   - Chunk `10/0` was marked Grade 1 (conservative, because it did not equal nominal coordinate `8/3`).
   - Chunk `16/1` was marked Grade 1 (conservative, because it did not equal nominal coordinate `16/0`).
   - Under that conservative label set, the first Grade $\ge 1$ chunks were at Ranks 1, 1, 2, 3, 1 for Original (Macro = 0.7667) and Ranks 1, 1, 2, 2, 1 for Diversified (Macro = 0.8000).
2. **In Final Validated Ground Truth (`gold_dataset_final_metric_consistency_report.md`):**  
   Once text validation confirmed that `1/2` (DNS steps), `10/0` (Serverless definition), and `16/1` (HTTP 403 definition) are **primary direct answers (Grade 2)**:
   - These exact numerical values (**0.7667** and **0.8000**) become the authoritative **Direct Answer Reciprocal Rank@3 ($\text{Grade} = 2$)**.
   - Meanwhile, Conventional MRR@3 ($\text{Grade} \ge 1$) rises to **0.8667** (Original) and **0.9000** (Diversified) because Rank 1 in Q3 (`9/0`) provides Grade 1 supporting context.

### 6.2 Side-by-Side Comparison: Original vs. Diversified

| Metric Formulation | Target Grade Threshold | Original Top-3 | Diversified Top-3 | Absolute Gain | Relative Gain | Key Mechanism |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Conventional MRR@3** | $\text{Grade} \ge 1$ | **0.8667** (26/30) | **0.9000** (27/30) | **+0.0333** | **+3.8%** | Q4 promoted from Rank 3 to Rank 2 ($1/3 \rightarrow 1/2$). |
| **Direct Answer RR@3** | $\text{Grade} = 2$ | **0.7667** (23/30) | **0.8000** (24/30) | **+0.0333** | **+4.3%** | Q4 direct answer promoted from Rank 3 to Rank 2 ($1/3 \rightarrow 1/2$). |

**Takeaway:** Regardless of whether evaluation measures time-to-first-evidence (Conventional) or time-to-direct-answer (Direct Answer), diversification consistently yields a **+0.0333 Reciprocal Rank lift**, driven solely by the promotion of chunk `16/1` in Q4.

---

## 7. Final Metric Naming Recommendations

To eliminate confusion between binary evidence retrieval and primary answer discovery in future RAG audits, the following dual-metric standard must be adopted:

```
+--------------------------------------------------------------------------------------------------+
|                                RECOMMENDED METRIC NOMENCLATURE                                   |
+--------------------------------------------------------------------------------------------------+
| 1. Conventional MRR@K (or Evidence-MRR@K):                                                       |
|    - Threshold: Grade >= 1 (Any verified answer or supporting evidence).                         |
|    - Measures: Retrieval efficiency in surfacing relevant context anywhere in Top-K.             |
|    - Use Case: Multi-document synthesis, complex reasoning, and context-augmented generation.   |
+--------------------------------------------------------------------------------------------------+
| 2. Direct Answer Reciprocal Rank@K (or Direct-MRR@K):                                            |
|    - Threshold: Grade == 2 (Strict complete/primary answer passage).                             |
|    - Measures: Retrieval precision in surfacing the standalone answer at the highest rank.       |
|    - Use Case: Short-answer factoid Q&A, extractive QA, and zero-shot citation accuracy.         |
+--------------------------------------------------------------------------------------------------+
```

---

## 8. Safety Verification Checklist

All analyses and calculations in this audit were performed in strict compliance with the read-only mandate:

- **Zero Pinecone API calls:** No queries, fetches, upserts, updates, or deletes were executed.
- **Zero Embedding / LLM API calls:** No calls made to Gemini, Groq, OpenAI, or other model APIs.
- **Zero file or dataset modifications:** Existing reports, production code (`main.py`, `retrieval_service.py`), `.env`, and dependencies were preserved without change.
- **Zero deployments or git operations:** Working tree remains clean.
- **Confidentiality:** All credentials and secrets remain protected.

---
*Report completed on 2026-11-13.*
