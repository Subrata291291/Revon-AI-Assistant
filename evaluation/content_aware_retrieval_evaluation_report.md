# Content‑Aware Retrieval Evaluation Report

**Report timestamp:** 2026‑11‑13 16:58 UTC+05:30

---

## 1. Executive Summary
- Performed a read‑only, content‑aware audit of the existing five‑question RAG retrieval evaluation.
- Compared the original Top‑3 results (ranks 1‑3) with the corrected diversified Top‑3 selections (Section 3).
- Because the full stored chunk text is not available in any local artifact, relevance judgments rely on the **Gold‑match** column in `retrieval_evaluation_report.md` (which indicates an exact page‑/chunk‑coordinate match) and on the excerpt content.
- Binary relevance (complete gold passage present) is achieved for **Q2** and **Q5** only; all other questions have no relevant chunk in the top‑3 of either ranking.
- Consequently, the macro‑averaged metrics are identical for the original and diversified selections:
  - **Precision@3:** 0.133
  - **Recall@3:** 0.400
  - **MRR@3:** 0.400
- The lack of full chunk text prevents a finer‑grained assessment (partial passage, literal overlap, or graded relevance).

---

## 2. Source Artifacts
| Artifact | Path | Role |
|---|---|---|
| Live RAG evaluation report | `backend/retrieval_evaluation_report.md` | Provides original Top‑5 results, gold‑match flags, and excerpts. |
| Corrected diversity audit (Version 2) | `backend/retrieval_diversity_audit_corrected_report_v2.md` | Supplies the heuristic‑based diversified Top‑3 selections. |
| Gold‑label alignment audit | `backend/gold_label_alignment_audit_report.md` | Confirms that each gold coordinate exists in the indexed vectors (used only for context). |
| Pinecone chunk audit report | `backend/pinecone_chunk_audit_report.md` | Shows that full chunk text for the 25 retrieved vectors is **not** stored locally. |

---

## 3. Evaluation Methodology
1. **Gold passage definition** – The gold passage is defined by the page/chunk coordinate listed in the original evaluation (e.g., Q1 → p1/c1). The `Gold match` column in the evaluation report marks a chunk as an exact coordinate match.
2. **Relevance criteria** – A retrieved chunk is considered **relevant** (binary = 1) **only** when it is an exact gold‑match, i.e., the chunk’s page/chunk coordinate equals the gold coordinate. This is a strict interpretation required by the existing evaluation pipeline.
3. **Partial or related content** – Excerpts that discuss the same topic but are not gold‑matches are recorded as *related but insufficient*; they are **not** counted as relevant for the binary metrics.
4. **Metrics** – Calculated per‑question for the first three results (original Top‑3 and diversified Top‑3):
   - **Precision@3** = (relevant chunks in top 3) / 3
   - **Recall@3** = (relevant chunks in top 3) / (total gold chunks = 1)
   - **Reciprocal Rank (RR)** = 1 / (rank of first relevant chunk) or 0 if none in top 3.
   - **Macro‑averages** are the mean of the per‑question values.
5. **Limitations** – The full text of the retrieved vectors is not present in any local artifact; therefore we cannot verify whether a chunk contains a *partial* gold passage or compute literal text overlap.

---

## 4. Question‑by‑Question Relevance Evidence
| Question | Gold coordinate | Original Top‑3 (ranks 1‑3) – relevance | Diversified Top‑3 – relevance |
|---|---|---|---|
| **Q1 – DNS** | p1/c1 | Rank 1 (1/2) – related but **not** gold‑match; Rank 2 (1/3) – related; Rank 3 (2/0) – unrelated → **0** relevant | Rank 1 (1/2) – related; Rank 2 (2/0) – unrelated; Rank 3 (1/0) – related → **0** relevant |
| **Q2 – TCP handshake** | p2/c1 | Rank 1 (2/1) – **gold‑match** → relevant; Rank 2 (2/2) – unrelated; Rank 3 (6/0) – unrelated → **1** relevant | Rank 1 (2/1) – **gold‑match**; Rank 2 (6/0) – unrelated; Rank 3 (4/2) – unrelated → **1** relevant |
| **Q3 – Serverless APIs** | p8/c3 | Ranks 1‑3 are 9/0, 10/0, 10/1 – none match gold coordinate → **0** relevant | Diversified ranks 1 (9/0), 2 (10/0), 3 (9/2) – none match gold coordinate → **0** relevant |
| **Q4 – HTTP 403** | p16/c0 | Ranks 1‑3 are 15/1, 15/2, 16/1 – none are gold‑match → **0** relevant | Diversified ranks 1 (15/1), 2 (16/1), 3 (13/2) – none are gold‑match → **0** relevant |
| **Q5 – HTTP status codes** | p15/c1 | Rank 1 (15/1) – **gold‑match**; Rank 2 (15/2) – unrelated; Rank 3 (16/0) – unrelated → **1** relevant | Rank 1 (15/1) – **gold‑match**; Rank 2 (16/0) – unrelated; Rank 3 (13/2) – unrelated → **1** relevant |

---

## 5. Metric Results
### 5.1 Original Top‑3
| Question | Precision@3 | Recall@3 | Reciprocal Rank |
|---|---|---|---|
| Q1 | 0.000 | 0.000 | 0.000 |
| Q2 | 0.333 | 1.000 | 1.000 |
| Q3 | 0.000 | 0.000 | 0.000 |
| Q4 | 0.000 | 0.000 | 0.000 |
| Q5 | 0.333 | 1.000 | 1.000 |
| **Macro average** | **0.133** | **0.400** | **0.400** |

### 5.2 Diversified Top‑3 (corrected heuristic)
| Question | Precision@3 | Recall@3 | Reciprocal Rank |
|---|---|---|---|
| Q1 | 0.000 | 0.000 | 0.000 |
| Q2 | 0.333 | 1.000 | 1.000 |
| Q3 | 0.000 | 0.000 | 0.000 |
| Q4 | 0.000 | 0.000 | 0.000 |
| Q5 | 0.333 | 1.000 | 1.000 |
| **Macro average** | **0.133** | **0.400** | **0.400** |

---

## 6. Comparison & Observations
- The diversified selection **does not improve** the binary relevance metrics because the heuristic only removes adjacent‑page chunks; the gold‑matching chunks for Q2 and Q5 were already in the original top‑3.
- For Q1, Q3, and Q4 the gold chunk remains outside the top‑3 in both rankings, so metrics are unchanged.
- The lack of full chunk text prevents assessing whether any of the non‑gold‑match chunks contain a *partial* gold passage (which could affect graded‑relevance metrics such as nDCG). Consequently, the **Literal text overlap** analysis is marked as *unavailable*.

---

## 7. Limitations
1. **Missing full chunk text** – Neither `retrieval_evaluation_report.md` nor any other local artifact stores the complete text of the 25 retrieved vectors. All relevance judgments are therefore based solely on the gold‑match flag and short excerpts.
2. **Binary relevance only** – Without full text we cannot assign graded relevance scores; nDCG@3 is therefore omitted.
3. **Assumption of exact‑match relevance** – The evaluation treats a gold‑match as a complete gold passage; any partial containment is not counted.
4. **Static dataset** – The analysis reflects the snapshot captured on 2026‑11‑13; any later index updates are outside the scope of this report.

---

## 8. Safety Verification
- **No Pinecone write operations** (upsert, update, delete) were performed.
- **No embedding or LLM API calls** were made.
- **No source‑code, configuration, or `.env` files** were modified.
- **No deployment or GitHub push** occurred.
- The only artifact created is this report file `backend/content_aware_retrieval_evaluation_report.md`.

---

*End of report.*