# RAG Evaluation Methodology Correction Report

**Audit Date:** 2026-11-13  
**Status:** Read-Only Methodological Audit & Correction  
**Target Document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`  
**Namespace:** `revon-policies`  

---

## 1. Executive Summary

Following the completion of the *Full Retrieved Chunk Text Audit*, this report establishes formal methodological corrections for the RAG retrieval evaluation framework. 

Previous evaluation iterations identified critical discrepancies when comparing coordinate-based proxy metrics with verified text-level evidence. However, those findings highlighted several methodological weaknesses in how metrics were defined, normalized, and labeled:

1. **Recall@3 vs. Hit Rate@3:** The previous reporting defined "Recall@3" as a binary indicator of whether at least one relevant chunk was retrieved in the Top-3. This metric is formally **Hit Rate@3** (or **Success@3**). Reporting it as standard Recall@3 is methodologically invalid because the total number of relevant chunks across the 50-chunk corpus ($R_{\text{total}}$) has not been exhaustively judged.
2. **nDCG Normalization & Evaluation Scope:** The previously reported nDCG scores were normalized against the sorted subset of the retrieved Top-3 items (**Top-3 Retained-Pool nDCG**). While valid as a measure of intra-pool ranking efficiency, it must not be conflated with **Corpus-Level nDCG**, which requires knowledge of the absolute best $K$ chunks across the entire corpus.
3. **Q4 Gold Label Conflict Investigation:** Physical extraction from the source PDF and inspection of the stored Pinecone vectors resolved the conflict between nominal gold label `16/0` and actual text evidence in `16/1`. Chunking boundary drift caused by uncollapsed raw whitespace during historical ingestion pushed the definition of HTTP 403 Forbidden out of vector `16/0` and into vector `16/1`.
4. **Diversification Safety & Context Dilution:** Using Q2 chunk `2/2` as a primary case study, the positional adjacency deduplication heuristic (`same page and |index - other| <= 1`) was found to eliminate relevant supporting evidence in exchange for lower-ranked, irrelevant chunks (`6/0` and `4/2`), demonstrating the risk of context dilution.
5. **No Production Code or Dataset Alteration:** In accordance with strict safety rules, these findings establish the evaluation standards and corrections without modifying application code, `.env` files, or the existing gold dataset.

---

## 2. Issue 1: Recall@3 vs. Hit Rate@3

### 2.1 The Methodological Discrepancy

In `backend/full_retrieved_chunk_text_audit_report.md` (and prior reports), the metric labeled "Recall@3" was calculated as:
$$\text{Recall@3}_{\text{reported}} = \begin{cases} 1.0 & \text{if } \sum_{i=1}^3 rel_i \ge 1 \\ 0.0 & \text{otherwise} \end{cases}$$

This formula computes the probability that the Top-3 results contain at least one relevant passage. In Information Retrieval (IR) literature, this metric is formally defined as **Hit Rate@K** (also known as **Success@K**):
$$\text{Hit Rate@K} = \mathbb{I}\left( \sum_{i=1}^K rel_i \ge 1 \right)$$

### 2.2 Standard Recall@K Definition & Denominator Requirements

Standard Recall at rank $K$ measures the fraction of *all existing relevant items in the corpus* that are successfully retrieved in the top $K$ positions:
$$\text{Recall@K} = \frac{|\text{Relevant Items in Top-K}|}{R_{\text{total}}}$$
where:
$$R_{\text{total}} = |\text{Rel}_{\text{corpus}}(q)| = \sum_{c \in \text{Corpus}} \mathbb{I}(c \text{ is relevant to } q)$$

To report standard Recall@K legitimately, the evaluation must satisfy one of two conditions:
1. **Known Exhaustive Denominator ($R_{\text{total}}$):** Every chunk in the document collection (all 50 indexed vectors) must be annotated as relevant or irrelevant for each query $q$.
2. **Strict Single-Passage Ground Truth Assumption:** An explicit guarantee that exactly one chunk in the entire universe can answer the query ($R_{\text{total}} = 1$).

### 2.3 Why the Single-Passage Assumption Fails for This Corpus

The text-level audit demonstrated that technical answers in this corpus naturally span multiple adjacent chunks due to chunking window boundaries and overlap:
- **Q1 (DNS):** Chunk `1/1` contains the conceptual introduction; Chunk `1/2` contains steps 1–4 of resolution; Chunk `1/3` contains steps 4–6 and final IP return.
- **Q2 (TCP Handshake):** Chunk `2/1` contains the formal 3-step handshake definition; Chunk `2/2` contains the concrete phone-call analogy explicitly listing SYN, SYN-ACK, and ACK.
- **Q3 (Serverless APIs):** Chunks `9/0`, `9/2`, `10/0`, and `10/1` each contain substantive portions of the comparative explanation.

Because $R_{\text{total}} > 1$ for several questions and the unretrieved 45 vectors have not been exhaustively judged, calculating standard Recall@3 with a fixed denominator of 1 is invalid.

### 2.4 Corrective Action

1. The metric must be formally renamed **Hit Rate@3** (or **Success@3**) across all evaluation reporting.
2. Standard **Recall@3** must be withheld until a complete relevance inventory of all 50 vectors is conducted to establish true $R_{\text{total}}$ per question.

---

## 3. Issue 2: nDCG Normalization & Evaluation Scope

### 3.1 Gain Formula & Discounted Cumulative Gain (DCG)

Discounted Cumulative Gain evaluates ranking quality by discounting the value of relevant documents appearing at lower positions.

- **Exponential Gain (Standard IR formulation):**
  $$\text{gain}_{\text{exp}}(g_i) = 2^{g_i} - 1$$
- **Linear Gain:**
  $$\text{gain}_{\text{lin}}(g_i) = g_i$$
- **Discounted Cumulative Gain at rank $K$:**
  $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{gain}(g_i)}{\log_2(i + 1)}$$

For binary relevance ($g_i \in \{0, 1\}$), $2^1 - 1 = 1$ and $2^0 - 1 = 0$, so exponential gain and linear gain are mathematically identical:
$$\text{DCG}_{\text{bin}}@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}$$

### 3.2 Ideal DCG (IDCG) Construction & Scope Distinctions

Normalized DCG is defined as:
$$\text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

The value of $\text{IDCG}@K$ depends entirely on the reference pool used to construct the ideal ranking:

```
+-----------------------------------------------------------------------------------------+
|                                  Evaluation Scope                                       |
+-----------------------------------------------------------------------------------------+
| Scope A: Top-3 Retained-Pool nDCG                                                       |
| - Pool: Only the 3 chunks selected for that ranking.                                    |
| - IDCG Construction: Sort the 3 retrieved grades descending.                            |
| - What it measures: Did the ranker order the 3 retrieved items optimally among          |
|   themselves? (Intra-pool ranking efficiency).                                          |
| - Limitation: If only 1 relevant item was retrieved (e.g., [1, 0, 0]), IDCG is based on |
|   [1, 0, 0], yielding nDCG = 1.000, even if the corpus contained better candidates.     |
+-----------------------------------------------------------------------------------------+
| Scope B: Corpus-Level nDCG                                                              |
| - Pool: All 50 indexed vectors in the corpus.                                           |
| - IDCG Construction: Take the top 3 highest relevance grades across the entire corpus.  |
| - What it measures: How close is the retrieved Top-3 to the absolute best 3 chunks in   |
|   the entire document?                                                                  |
| - Requirement: Requires complete relevance judgments for all 50 vectors.               |
+-----------------------------------------------------------------------------------------+
```

### 3.3 Scope Analysis of Previously Reported Results

In `full_retrieved_chunk_text_audit_report.md`, the reported nDCG values were constructed using **Scope A (Top-3 Retained-Pool nDCG)**:
- For Q1, retrieved grades were `[1, 0, 0]`. Sorting them gives `[1, 0, 0]`, so $\text{DCG} = \text{IDCG} = 1.0$, yielding $\text{nDCG} = 1.000$.
- For Q2 Original, retrieved grades were `[2, 1, 0]`. Sorting gives `[2, 1, 0]`, yielding $\text{nDCG} = 1.000$.
- For Q4 Original, retrieved grades were `[0, 0, 1]`. Sorting gives `[1, 0, 0]`, yielding $\text{nDCG} = 0.500$. In Diversified, grades were `[0, 1, 0]`, yielding $\text{nDCG} = 0.631$.

While Scope A accurately reflects the internal ordering quality of the returned list, labeling it without qualification allows readers to misinterpret it as Corpus-Level nDCG.

### 3.4 Corrective Action

1. All future nDCG reporting must explicitly specify the normalization scope as **Top-3 Retained-Pool nDCG** ($\text{nDCG}_{\text{pool}}@3$).
2. Corpus-Level nDCG ($\text{nDCG}_{\text{corpus}}@3$) must not be reported until all 50 vectors are labeled.
3. Both binary and graded scales must be preserved, labeled as:
   - `nDCG@3 (Binary, Retained-Pool)`
   - `nDCG@3 (Graded, Retained-Pool)`

---

## 4. Issue 3: Q4 Gold Label Conflict Investigation

### 4.1 The Conflict

A direct contradiction existed across previous audit artifacts:
- **`backend/gold_label_alignment_audit_report.md`:** Recorded `16/0` as the verified gold coordinate for Q4 (*"Stored text contains gold passage: Yes... Similarity = 0.8827"*).
- **`backend/full_retrieved_chunk_text_audit_report.md`:** Found that Pinecone vector `16/0` ends at `400 Bad Request` and does not contain HTTP 403 Forbidden, whereas retrieved vector `16/1` contains the complete definition (`403 Forbidden → You don’t have permission to access this resource`).

### 4.2 Source Evidence Verification

To resolve this discrepancy conclusively, three physical artifacts were inspected:
1. **Source PDF:** `Pre-Read_ Basic Understanding on API  (1)-5983.pdf`, Page 16.
2. **Active Pinecone Stored Vectors:** `16/0` and `16/1` (Document ID `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`).
3. **Current Local Chunker Implementation:** `backend/chunker.py`.

#### Source PDF Page 16 Text
Extraction of Page 16 via `pypdf.PdfReader` confirms that HTTP 403 Forbidden is present on Page 16:
```text
● 400 Bad Request → The request was malformed or missing required data.
● 401 Unauthorized → You need to provide a valid API key or credentials.
● 403 Forbidden → You don’t have permission to access this resource.
● 404 Not Found → The requested resource doesn’t exist.
```

#### Pinecone Stored Vector `16/0` (`...-page-16-chunk-0`)
- **Raw Length:** 793 characters
- **Content:** Begins at `● 200 OK` and concludes with:
  > `● 400 Bad Request → The request was malformed or missing`
- **Contains "403" or "Forbidden":** **False**

#### Pinecone Stored Vector `16/1` (`...-page-16-chunk-1`)
- **Raw Length:** 788 characters
- **Content:** Begins with `bad input, unauthorized access). ● 400 Bad Request...` and contains:
  > `● 403 Forbidden → You don’t have permission to access this resource.`
- **Contains "403" or "Forbidden":** **True**

### 4.3 Root Cause of the Discrepancy

The root cause is **chunking boundary drift caused by whitespace normalization differences**:

```
Source PDF Page 16 Text
|
+---> Active Pinecone Index (Historical Build, 50 Vectors)
|     - Extracted raw text retained redundant newline/space formatting.
|     - Raw characters filled the 800-character budget early.
|     - Vector 16/0 reached limit at "400 Bad Request".
|     - "403 Forbidden" was pushed into Vector 16/1.
|
+---> Current Local Chunker (backend/chunker.py, 42 Chunks)
      - Runs normalize_text() before segmenting.
      - Collapsed whitespace allows more semantic tokens per chunk.
      - Local Chunk 16/0 (748 chars) fits 200 OK through 403 Forbidden.
      - "403 Forbidden" appears in Local Chunk 16/0.
```

When `gold_label_alignment_audit_report.md` evaluated gold alignment, it compared Pinecone vector `16/0` against *current local chunk `16/0`* using `difflib.SequenceMatcher`, obtaining a 0.8827 similarity ratio. Because the score was high, the report incorrectly assumed the stored chunk contained the gold passage without verifying the exact substring `"403 Forbidden"`.

### 4.4 Recommendations for Q4 Gold Labeling

1. **For the Active Pinecone Index (50 Vectors):**
   - The correct gold coordinate is **`16/1`** (`ab56ef89-c787-470c-b796-7d4b9fcdb2ee-page-16-chunk-1`).
   - The authoritative gold text passage is:
     > `● 403 Forbidden → You don’t have permission to access this resource.`
2. **Dataset Modification Policy:**
   - In accordance with strict read-only safety rules, the legacy gold dataset must not be modified automatically.
   - Evaluation documentation must record `16/1` as the verified active index gold coordinate, while documenting `16/0` as a legacy artifact of chunker boundary drift.

---

## 5. Issue 4: Diversification Safety & Context Dilution Trade-off

### 5.1 The Diversification Heuristic

The candidate diversification algorithm applies the following deterministic rule per question:
> *Walk the ranked candidate list in order. A candidate is skipped if it is on the same page AND its chunk index differs by at most 1 from an already selected candidate. Stop when three candidates are selected.*

### 5.2 Case Study: Question 2 (TCP Handshake)

To evaluate the safety of this heuristic, examine the candidate trajectory for Q2:

| Rank | Coordinate | Cosine Score | Relevance Status | Action | Stored Content Summary |
|:---:|:---:|:---:|:---:|:---:|---|
| **1** | `2/1` | 0.7770 | Relevant (Grade 2) | **Selected** | Formal 3-step handshake definition (`SYN`, `SYN-ACK`, `ACK`). |
| **2** | `2/2` | 0.7470 | Relevant (Grade 1) | **Skipped** | Phone call analogy explicitly listing `SYN`, `SYN-ACK`, and `ACK`. |
| **3** | `6/0` | 0.7052 | Irrelevant (Grade 0) | **Selected** | Connection termination (4-Step FIN-ACK process). |
| **4** | `4/2` | 0.6984 | Irrelevant (Grade 0) | **Selected** | Packet headers, sequence numbers, and transmission ACKs. |
| **5** | `2/3` | 0.6907 | Irrelevant (Grade 0) | *Unused* | Data transfer packetization. |

### 5.3 Analysis of Skipped Chunk `2/2`

Chunk `2/2` contains the following text:
> *"process can data transfer begin. For Example: Think of it like a polite phone call: ● You dial a number (SYN) ● They pick up and say, 'Hello?' (SYN-ACK) ● You say, “Hi, can we talk now?” (ACK) ● Now the conversation starts. Great! So now both your browser and the server are ready to talk. The connection has been established. Now comes the actual conversation— the data transfer..."*

- **Content Relevance:** Directly relevant supporting evidence. It provides the conversational analogy specifically explaining the three messages of the handshake.
- **Similarity Score:** 0.7470 (Rank 2 across the entire corpus).
- **Impact of Removal:**
  - The heuristic removed a highly relevant chunk (`2/2`, score 0.7470) solely because $|2 - 1| \le 1$ on Page 2.
  - To fill the Top-3 quota, the ranker selected `6/0` (score 0.7052, FIN-ACK teardown) and `4/2` (score 0.6984, packet headers).
  - Neither replacement chunk addresses the three-way handshake.

### 5.4 The Dual Impact of Adjacency Deduplication

The audit demonstrates that the coordinate-based adjacency heuristic produces contrasting effects:

```
+-----------------------------------------------------------------------------------------+
|                               Heuristic Impact Analysis                                 |
+-----------------------------------------------------------------------------------------+
| Positive Effect: Noise Removal & Rank Promotion (Q4 Example)                            |
| - In Q4, Rank 1 (15/1) and Rank 2 (15/2) were unhelpful category headers.              |
| - Skipping 15/2 promoted the true answer chunk 16/1 from Rank 3 to Rank 2.             |
| - Result: MRR increased from 0.333 to 0.500; nDCG increased from 0.500 to 0.631.        |
+-----------------------------------------------------------------------------------------+
| Negative Effect: Context Dilution & Quality Degradation (Q2 Example)                    |
| - In Q2, Rank 2 (2/2) was strong supporting evidence for the answer in Rank 1 (2/1).   |
| - Skipping 2/2 forced the selection of irrelevant out-of-domain chunks (6/0, 4/2).     |
| - Result: Downstream context quality was diluted; Precision@3 dropped from 0.667       |
|   to 0.333 without increasing Recall or MRR.                                           |
+-----------------------------------------------------------------------------------------+
```

### 5.5 Conclusion on Heuristic Safety

Coordinate-based adjacency deduplication is a purely positional heuristic. It cannot distinguish between:
1. **Redundant duplication** (text that merely repeats already captured concepts without adding value), and
2. **Complementary supporting evidence** (examples, analogies, or second-half definitions that aid downstream generation).

While safe for preserving top-ranked primary answers, the heuristic introduces **context dilution risks**. Future retrieval optimization must not rely on coordinate distance alone.

---

## 6. Recommended Corrected Evaluation Methodology

The following methodology is formally recommended for all future RAG retrieval audits:

### 6.1 Metric Nomenclature & Formulas

| Metric Name | Mathematical Definition | Scope & Conditions |
|---|---|---|
| **Hit Rate@K** | $\mathbb{I}\left( \sum_{i=1}^K rel_i \ge 1 \right)$ | Binary indicator of whether at least one relevant chunk is present in Top-K. |
| **Precision@K** | $\frac{1}{K} \sum_{i=1}^K rel_i$ | Fraction of retrieved Top-K chunks containing verified answer evidence ($rel_i \in \{0, 1\}$). |
| **MRR@K** | $\frac{1}{\min \{ i \le K \mid rel_i = 1 \}}$ (or 0) | Reciprocal rank of the first relevant chunk in Top-K. |
| **nDCG@K (Binary, Retained-Pool)** | $\frac{\text{DCG}_{\text{bin}}@K}{\text{IDCG}_{\text{bin}}@K}$ | Evaluates ranking efficiency of the retrieved Top-K using binary relevance ($rel_i \in \{0, 1\}$). $\text{IDCG}$ derived from retrieved items sorted. |
| **nDCG@K (Graded, Retained-Pool)** | $\frac{\text{DCG}_{\text{exp}}@K}{\text{IDCG}_{\text{exp}}@K}$ | Evaluates ranking efficiency using graded relevance ($g_i \in \{0, 1, 2\}$) with exponential gain ($2^{g_i} - 1$). $\text{IDCG}$ derived from retrieved items sorted. |
| **Standard Recall@K** | $\frac{\sum_{i=1}^K rel_i}{R_{\text{total}}}$ | **Do not report** until full corpus relevance inventory ($R_{\text{total}}$) is established. |

### 6.2 Standardized Relevance Grading Scale

Relevance judgments must be made on verified chunk text, not coordinates:
- **Grade 2 (Complete Answer):** The chunk contains the complete, standalone answer to the question.
- **Grade 1 (Substantive Supporting Evidence):** The chunk contains partial gold passage overlap, essential steps/components of a multi-part answer, or direct supporting examples.
- **Grade 0 (Insufficient Evidence):** The chunk lacks substantive evidence to answer the query, even if topically related.

### 6.3 Corrected Baseline Performance Summary

Applying this corrected methodology to the live retrieval evaluation data produces the following authoritative baseline:

| Evaluation Metric | Coordinate Proxy (Deprecated) | Original Top-3 (Corrected Method) | Diversified Top-3 (Corrected Method) | Net Diversification Effect |
|---|:---:|:---:|:---:|:---:|
| **Hit Rate@3** | 0.400 | **1.000** | **1.000** | 0.000 (100% success) |
| **Precision@3** | 0.133 | **0.467** | **0.400** | -0.067 (redundancy removed) |
| **MRR@3** | 0.400 | **0.767** | **0.800** | **+0.033 (+4.3% ranking lift)** |
| **nDCG@3 (Binary, Retained-Pool)** | *N/A* | **0.839** | **0.865** | **+0.026 (+3.1%)** |
| **nDCG@3 (Graded, Retained-Pool)** | *N/A* | **0.839** | **0.865** | **+0.026 (+3.1%)** |
| **Standard Recall@3** | *Invalid* | *Withheld* | *Withheld* | *Pending corpus labeling* |

---

## 7. Required Future Dataset & Evaluation Improvements

To evolve the RAG evaluation framework into a production-grade benchmark, the following enhancements are required:

1. **Exhaustive Corpus Relevance Labeling ($R_{\text{total}}$):**
   - Annotate all 50 vectors in the active document for all five evaluation queries.
   - Establishing true $R_{\text{total}}$ will enable standard Recall@K, Mean Average Precision (MAP), and true Corpus-Level nDCG.
2. **Multi-Chunk Ground Truth Passages:**
   - Replace single-point coordinate gold labels with passage spans or multi-coordinate sets (e.g., Q1 Gold: `[1/1, 1/2]`; Q3 Gold: `[9/0, 9/2, 10/0, 10/1]`).
3. **Index-Synchronized Chunk Metadata:**
   - Embed `chunker_version` and `normalization_hash` in Pinecone vector metadata during ingestion to prevent drift between local chunkers and stored vectors.
4. **Semantic Redundancy Filtering (Post-Retrieval):**
   - Replace the positional coordinate heuristic (`same page, |diff| <= 1`) with text-level redundancy filtering:
     - **Character/Token Jaccard Overlap:** Drop candidates only if token overlap with an already selected chunk exceeds a high threshold (e.g., $> 60\%$).
     - **Maximal Marginal Relevance (MMR):** Balance query relevance with embedding diversity to prevent discarding valuable supporting examples like Q2 `2/2`.

---

## 8. Safety Verification Checklist

All work conducted during this methodology audit strictly adhered to the safety rules:

- **Read-only vector store operations:** Verified. Only read-only `/vectors/fetch` was performed. Zero vectors were created, updated, deleted, or ingested.
- **Zero Embedding / LLM API calls:** Verified. No calls were made to Gemini, Groq, OpenAI, or other model APIs.
- **Zero source code or configuration changes:** Verified. `chunker.py`, `retrieval_service.py`, `main.py`, `.env`, and production files were left completely unmodified.
- **Zero dependency changes:** Verified. No packages were installed or upgraded.
- **No deployment or git push:** Verified.
- **Gold dataset preservation:** Verified. Existing benchmark definitions were analyzed without automated alteration.
- **No exposure of credentials:** Verified. All API keys, tokens, and hosts remain strictly protected.

---
*Report completed on 2026-11-13.*
