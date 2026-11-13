# RAG Gold Dataset Final Metric Consistency Report

**Audit Date:** 2026-11-13  
**Status:** Read-Only Metric Consistency & Final Audit  
**Target Document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`  
**Logical Document ID:** `20dcdd7c-a603-4a3f-b384-46081d625a73`  
**Namespace:** `revon-policies`  
**Target Document:** `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf` (16 Pages, 50 Active Vectors)  

---

## 1. Executive Summary

This report performs a comprehensive consistency review of the evaluation metrics, relevance judgments, and diversification findings established during the *Gold Dataset Validation & Versioning Audit*.

### Core Audit Conclusions

1. **Complete 50-Vector Corpus Assessment:** All 50 active Pinecone vectors were systematically inspected across all 16 pages of the target document. The corpus contains 19 relevant chunks across the five evaluation queries: Q1 ($R_{\text{total}}=4$), Q2 ($R_{\text{total}}=2$), Q3 ($R_{\text{total}}=8$), Q4 ($R_{\text{total}}=1$), and Q5 ($R_{\text{total}}=4$).
2. **Standard Recall vs. Hit Rate Resolution:** Under the verified corpus denominators ($R_{\text{total}}$), **Hit Rate@3 is 1.000** for both Original and Diversified retrieval. However, **Standard Macro Recall@3** is **0.775** for Original Top-3 and **0.575** for Diversified Top-3, while **Micro Recall@3** is **0.632** (12/19) for Original and **0.474** (9/19) for Diversified.
3. **The Diversification Trade-Off Decoupled:** While diversification increases page diversity from **2.00 to 2.60 distinct pages per query**, in 3 out of 5 queries (Q2, Q4, Q5), the third page added was irrelevant filler (Pages 4 and 13). However, diversification successfully promoted the true answer chunk in Q4 (`16/1`) from Rank 3 to Rank 2, improving MRR from 0.767 to 0.800.
4. **nDCG Normalization Clarification:** When normalized against the retrieved set (**Retained-Pool nDCG**), diversification appears to improve nDCG (0.864 to 0.889 graded) due to Q4 promotion. However, when normalized against the absolute best top-3 chunks in the corpus (**Corpus-Level nDCG**), diversification decreases binary nDCG from **0.900 to 0.755** because it intentionally discards relevant adjacent chunks in exchange for irrelevant out-of-domain chunks.
5. **v1.0 Baseline Readiness:** The dataset is **READY for a v1.0 baseline**, provided it is formalized using canonical text passages rather than brittle coordinate matching.

---

## 2. Check 1: 50-Vector Corpus Assessment Verification

All 50 vectors in the active Pinecone index (namespace `revon-policies`) were extracted and topic-mapped against the source PDF pages.

| Page | Active Vectors | Primary Page Topic | Relevant to Questions |
|:---:|:---:|---|---|
| **1** | `1/0, 1/1, 1/2, 1/3` (4) | API Intro, Internet Protocol Stack, DNS Resolution | **Q1** (`1/1`, `1/2`, `1/3`) |
| **2** | `2/0, 2/1, 2/2, 2/3` (4) | DNS Cache Analogy, TCP 3-Way Handshake, Packet Intro | **Q1** (`2/0`), **Q2** (`2/1`, `2/2`) |
| **3** | `3/0, 3/1, 3/2` (3) | Packet Structure, IP Address Definition | *None* (Grade 0) |
| **4** | `4/0, 4/1, 4/2` (3) | MAC Address, Sequence Numbers, Packet ACKs | *None* (Grade 0) |
| **5** | `5/0, 5/1, 5/2` (3) | Packet Reassembly, Webpage Loading Example | *None* (Grade 0) |
| **6** | `6/0, 6/1, 6/2` (3) | 4-Step FIN-ACK Connection Teardown, Client-Server Intro | *None* (Grade 0) |
| **7** | `7/0, 7/1` (2) | Client-Server Analogy (Waiter), What is an API | *None* (Grade 0) |
| **8** | `8/0, 8/1, 8/2, 8/3` (4) | API Role, Waiter Mapping, Server Hosting Intro | **Q3** (`8/3`) |
| **9** | `9/0, 9/1, 9/2, 9/3` (4) | Serverful API Definition, Local LLM Setup, Disadvantages | **Q3** (`9/0`, `9/1`, `9/2`, `9/3`) |
| **10** | `10/0, 10/1, 10/2, 10/3` (4) | Serverless API Definition, Advantages, Trade-off Summary | **Q3** (`10/0`, `10/1`, `10/2`) |
| **11** | `11/0, 11/1, 11/2` (3) | JSON Data Format, Key-Value Structure | *None* (Grade 0) |
| **12** | `12/0, 12/1` (2) | Browser Request Walkthrough, jsonplaceholder Example | *None* (Grade 0) |
| **13** | `13/0, 13/1, 13/2` (3) | Mock API Architecture, HTTP Protocol Postal Analogy | *None* (Grade 0) |
| **14** | `14/0, 14/1` (2) | HTTP Verbs & Methods (GET, POST, PUT, DELETE) | *None* (Grade 0) |
| **15** | `15/0, 15/1, 15/2` (3) | PUT/DELETE Details, What is an HTTP Status Code, 1xx/2xx | **Q5** (`15/1`, `15/2`) |
| **16** | `16/0, 16/1, 16/2` (3) | 2xx, 3xx, 4xx (403 Forbidden), 5xx, Live Session Wrap-up | **Q4** (`16/1`), **Q5** (`16/0`, `16/1`) |
| **Total** | **50 Vectors** | **16 Pages** | **19 Relevant Instances** |

**Verification Result:** Confirmed. Zero active vectors were omitted. All 50 stored vector texts have been audited.

---

## 3. Check 2: Relevance Grading & Binary Conversion Policy

### 3.1 Grading Standards

Relevance is evaluated under a 3-tier graded scheme:
- **Grade 2 (Direct / Complete Answer):** The chunk contains the primary answer or complete technical mechanism required by the query.
- **Grade 1 (Substantive Supporting Evidence):** The chunk contains a valid step of a multi-part answer, an illustrative real-world analogy, or range examples directly reinforcing the answer.
- **Grade 0 (Irrelevant / Insufficient):** The chunk lacks substantive answer evidence.

### 3.2 Binary Mapping Policy
For standard binary metrics (Precision, Recall, Hit Rate, Binary nDCG):
$$rel_i = \begin{cases} 1 & \text{if Grade } \ge 1 \\ 0 & \text{if Grade } = 0 \end{cases}$$

**Verification Result:** Confirmed. All standard Recall, Hit Rate, and binary calculations consistently treat grades $\ge 1$ as relevant ($rel=1$).

---

## 4. Checks 3, 4 & 5: Comprehensive Metric Recalculations

### 4.1 Metric Distinctions & Mathematical Formulations

1. **Per-Question Recall@K:**
   $$\text{Recall@K}_q = \frac{\sum_{i=1}^K rel_{q, i}}{R_{\text{total}, q}}$$
   Measures the fraction of all relevant chunks for question $q$ captured in the top $K$.
2. **Macro Recall@K:**
   $$\text{Macro Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \text{Recall@K}_q$$
   Unweighted arithmetic average of per-question recall. Equal weight per query.
3. **Micro Recall@K:**
   $$\text{Micro Recall@K} = \frac{\sum_{q \in Q} |\text{Relevant Retrieved}_q|}{\sum_{q \in Q} R_{\text{total}, q}} = \frac{\text{Total Relevant Retrieved Across Corpus}}{\text{Total Relevant Chunks Across Corpus}}$$
   Aggregates across all relevant instances. Queries with larger candidate pools (e.g., Q3 with 8 chunks) exert greater influence.
4. **Hit Rate@K (Success@K):**
   $$\text{Hit Rate@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \sum_{i=1}^K rel_{q, i} \ge 1 \right)$$
   Measures the percentage of queries where the user found *at least one* useful answer in Top-$K$.

---

### 4.2 Comprehensive Recalculated Metric Table

Both Original and Diversified rankings are evaluated against the **identical corpus ground truth** ($R_{\text{total}}$):

| Query ID | Query Topic | Corpus Relevant ($R_{\text{total}}$) | Original Top-3 Retrieved | Original Rel Ret | Original Recall@3 | Diversified Top-3 Retrieved | Diversified Rel Ret | Diversified Recall@3 | Hit Rate@3 (Both) |
|---|---|:---:|---|:---:|:---:|---|:---:|:---:|:---:|
| **Q1** | DNS Translation | 4 | `1/2, 1/3, 2/0` | 3 | **0.7500** | `1/2, 2/0, 1/0` | 2 | **0.5000** | **1.0000** |
| **Q2** | TCP Handshake | 2 | `2/1, 2/2, 6/0` | 2 | **1.0000** | `2/1, 6/0, 4/2` | 1 | **0.5000** | **1.0000** |
| **Q3** | Serverless APIs | 8 | `9/0, 10/0, 10/1` | 3 | **0.3750** | `9/0, 10/0, 9/2` | 3 | **0.3750** | **1.0000** |
| **Q4** | HTTP 403 Forbidden | 1 | `15/1, 15/2, 16/1` | 1 | **1.0000** | `15/1, 16/1, 13/2` | 1 | **1.0000** | **1.0000** |
| **Q5** | Status Codes | 4 | `15/1, 15/2, 16/0` | 3 | **0.7500** | `15/1, 16/0, 13/2` | 2 | **0.5000** | **1.0000** |
| **Summary** | **Macro Average** | **19 Total** | — | **12 / 19** | **0.7750** | — | **9 / 19** | **0.5750** | **1.0000** |
| **Summary** | **Micro Average** | **19 Total** | — | **12 / 19** | **0.6316** | — | **9 / 19** | **0.4737** | **1.0000** |

---

## 5. Check 6: Overlap Inflation & Annotation Policy Analysis

### 5.1 Chunk Boundary Overlap Mechanism

In the active index, adjacent chunks on the same page share a literal text overlap of **79 to 94 normalized characters** (120 raw characters).

```
[--- Chunk 1/1 ---]
...DNS (Domain Name System) is like the phonebook of the internet. Humans prefer
                  [--- Chunk 1/2 ---]
                  Addresses DNS (Domain Name System) is like the phonebook of the internet. Humans prefer...
```

### 5.2 Risks of Denominator Inflation

When adjacent chunks overlap across a continuous discussion, treating every chunk as an independent relevant document introduces two critical distortions:

1. **Length-Biased Micro Recall:**
   - In Q3 (Serverless APIs), the topic spans 3 pages, resulting in 8 relevant chunks.
   - In Q4 (HTTP 403), the definition is concise, resulting in 1 relevant chunk.
   - Because Q3 accounts for $8 / 19 = 42.1\%$ of the entire micro denominator, Micro Recall is heavily skewed toward multi-page topics.
2. **Mathematical Recall Ceiling at $K=3$:**
   - If a query has $R_{\text{total}} > 3$, it is mathematically impossible for Top-3 retrieval to achieve Recall@3 = 1.0.
   - For Q3 ($R_{\text{total}}=8$), the theoretical maximum Recall@3 is $3 / 8 = \mathbf{0.3750}$. Both Original and Diversified retrieval achieved 3 relevant chunks, thus hitting the **100% mathematical ceiling** for $K=3$.
   - For Q1 and Q5 ($R_{\text{total}}=4$), the theoretical maximum Recall@3 is $3 / 4 = \mathbf{0.7500}$. Original retrieval achieved the absolute maximum possible score.

### 5.3 Recommended Annotation Policy

To prevent artificial denominator inflation, future evaluation standards must distinguish:
- **Chunk-Level Recall (Instance Exhaustiveness):** Uses all chunks meeting Grade $\ge 1$ as the denominator ($R_{\text{total}}$). Used to measure index redundancy capture.
- **Passage-Level / Information-Need Recall (Hit Rate / Semantic Recall):** Evaluates whether the set of retrieved chunks collectively covers all semantic facets of the gold answer, capped at 1.0 once the information need is fulfilled.

---

## 6. Check 7: Diversification & Cross-Page Coverage Audit

Previous reports noted that diversification "improves cross-page coverage." This claim was audited against the actual page numbers of the selected Top-3 chunks:

### 6.1 Page Distribution Audit

| Query | Original Top-3 Coords | Original Pages | Distinct Pages | Diversified Top-3 Coords | Diversified Pages | Distinct Pages | Net Page Change | Semantic Relevance of Added Page |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|---|
| **Q1** | `1/2, 1/3, 2/0` | `1, 1, 2` | 2 | `1/2, 2/0, 1/0` | `1, 2, 1` | 2 | 0 | None (both have two p1, one p2) |
| **Q2** | `2/1, 2/2, 6/0` | `2, 2, 6` | 2 | `2/1, 6/0, 4/2` | `2, 6, 4` | **3** | **+1** | **Negative:** Added Page 4 (`4/2`, packet headers) is irrelevant |
| **Q3** | `9/0, 10/0, 10/1`| `9, 10, 10`| 2 | `9/0, 10/0, 9/2` | `9, 10, 9` | 2 | 0 | None (both have p9 and p10) |
| **Q4** | `15/1, 15/2, 16/1`| `15, 15, 16`| 2 | `15/1, 16/1, 13/2`| `15, 16, 13`| **3** | **+1** | **Mixed:** Promoted Page 16 (`16/1`), but Page 13 is irrelevant |
| **Q5** | `15/1, 15/2, 16/0`| `15, 15, 16`| 2 | `15/1, 16/0, 13/2`| `15, 16, 13`| **3** | **+1** | **Negative:** Added Page 13 (`13/2`, postal analogy) is irrelevant |
| **Average**| — | — | **2.00** | — | — | **2.60** | **+0.60** | **Page diversity increased; semantic precision decreased** |

### 6.2 Findings on Diversification

1. **Page Diversity Metric:** Distinct pages increased from an average of **2.00 to 2.60 pages per query**.
2. **Context Dilution Reality:** In Q2 and Q5, the heuristic discarded a relevant neighboring chunk on the same page (`2/2` and `15/2`) and backfilled the Top-3 quota with an irrelevant chunk from a distant page (`4/2` and `13/2`).
3. **The Single True Win:** Diversification's primary success was **Q4**, where skipping non-answering neighbor `15/2` allowed the true answer on Page 16 (`16/1`) to move up from Rank 3 to Rank 2, improving Reciprocal Rank from 0.333 to 0.500.

**Conclusion:** The claim of "improved cross-page coverage" is technically true in terms of page count, but misleading if interpreted as improved answer quality. In practice, diversification traded relevant adjacent explanations for irrelevant distant filler in 2 out of 5 queries.

---

## 7. Check 8: nDCG Gain Formulations & Normalization Scope

### 7.1 Mathematical Formulations

- **Exponential Gain:** $\text{gain}_{\text{exp}}(g_i) = 2^{g_i} - 1$
- **Linear Gain:** $\text{gain}_{\text{lin}}(g_i) = g_i$
- **Discount Factor:** $\frac{1}{\log_2(i + 1)}$ for rank $i \in \{1, 2, 3\}$

### 7.2 Retained-Pool vs. Corpus-Level nDCG@3

To resolve all ambiguity, both normalization scopes are calculated below:

```
Retained-Pool nDCG: IDCG is the sorted DCG of the 3 retrieved items.
Corpus-Level nDCG:  IDCG is the DCG of the top 3 highest grades across all 50 vectors.
```

| Query ID | Original Graded | Original Retained-Pool nDCG | Original Corpus-Level nDCG | Diversified Graded | Diversified Retained-Pool nDCG | Diversified Corpus-Level nDCG |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | `[2, 1, 1]` | 1.000 | **1.000** | `[2, 1, 0]` | 1.000 | **0.879** |
| **Q2** | `[2, 1, 0]` | 1.000 | **1.000** | `[2, 0, 0]` | 1.000 | **0.826** |
| **Q3** | `[1, 2, 1]` | 0.821 | **0.531** | `[1, 2, 2]` | 0.815 | **0.687** |
| **Q4** | `[0, 0, 2]` | 0.500 | **0.500** | `[0, 2, 0]` | 0.631 | **0.631** |
| **Q5** | `[2, 1, 1]` | 1.000 | **1.000** | `[2, 1, 0]` | 1.000 | **0.879** |
| **Macro Average (Graded)** | — | **0.864** | **0.806** | — | **0.889** | **0.780** |
| **Macro Average (Binary)** | — | **0.900** | **0.900** | — | **0.926** | **0.755** |

### 7.3 Scope Insights

- **Retained-Pool nDCG:** Diversification improves graded nDCG from **0.864 to 0.889 (+0.025)** because it elevated Q4's Grade 2 answer from rank 3 to rank 2.
- **Corpus-Level nDCG:** Diversification decreases graded nDCG from **0.806 to 0.780 (-0.026)** and binary nDCG from **0.900 to 0.755 (-0.145)** because discarding relevant adjacent chunks (`2/2`, `1/3`, `15/2`) dropped the total relevance captured relative to the corpus optimum.

---

## 8. Inconsistencies & Anomalies Identified

1. **Q4 Label Contradiction:** Early reports marked `16/0` as gold; text audit proved `16/0` contains 0 evidence for 403. Resolved: `16/1` is the verified active-index gold coordinate.
2. **Prior Recall Labeling:** Previous reports labeled Hit Rate as "Recall@3". Resolved: Renamed to Hit Rate@3, and true Recall@3 calculated with corpus denominators.
3. **Local vs. Index Segmentation Mismatch:** `backend/chunker.py` generates 42 chunks for this PDF, while Pinecone holds 50 vectors. In local chunking, `16/0` contains 403; in Pinecone, `16/1` contains 403.
4. **Sentence Truncation in Q1 `1/3`:** Vector `1/3` starts mid-sentence with `(usually provided by your ISP)`. While technically containing resolution steps, returning broken syntax to an LLM is suboptimal.

---

## 9. Remaining Methodological Limitations

1. **Small Sample Size ($N=5$):** A 5-query benchmark is effective for sanity checking and structural validation, but lacks statistical power for fine-grained ranking comparisons.
2. **Binary vs. Graded Boundary for Analogies:** Analogies (e.g., Pizza Hut for DNS in `2/0`, or phone call for TCP in `2/2`) are graded as 1 (Supporting). If a stricter evaluation requires only technical explanations, these would become 0.
3. **Static Snapshot:** These metrics evaluate the static 50-vector snapshot of document `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`. Any index re-ingestion will require re-mapping.

---

## 10. Baseline v1.0 Readiness Assessment & Recommendation

### 10.1 Readiness Verdict: QUALIFIED YES

The evaluation dataset and methodology are **READY for adoption as the v1.0 Evaluation Baseline**, subject to the following binding constraints:

1. **Adopt Canonical Text Ground Truth:** The baseline must store exact gold passage strings (as structured in Section 7 of the Versioning Audit report), not naked coordinates.
2. **Register `16/1` for Active Pinecone Index:** The active index mapping for Q4 must point to `16/1`.
3. **Report Both Hit Rate@3 and Standard Recall@3:**
   - Report **Hit Rate@3 = 1.000** (user success rate).
   - Report **Standard Recall@3 = 0.775 (Orig) / 0.575 (Div)** with explicit corpus denominators ($R_{\text{total}}$).
4. **Specify nDCG Normalization Scope:** All future reports must label nDCG as either `Retained-Pool nDCG` or `Corpus-Level nDCG`.

---

## 11. Safety Verification Checklist

All checks in this consistency review strictly followed the read-only mandate:

- **No Pinecone write operations:** Verified.
- **Zero API calls:** Verified. No calls made to Gemini, Groq, OpenAI, or embedding/LLM services.
- **Zero code or dataset modifications:** Verified. Production files and markdown reports were unchanged.
- **Zero configuration or dependency changes:** Verified.
- **Zero deployments or git pushes:** Verified.
- **Confidentiality:** All credentials, keys, and hosts remained protected.

---
*Report completed on 2026-11-13.*
