# Gold Label Alignment Audit Report

**Audit timestamp:** 2026-11-13 16:55 (UTC+05:30)

## 1. Executive Summary
- **Vectors inspected:** 50 target vectors fetched from Pinecone (namespace `revon-policies`).
- **All vectors belong to the target document** (`document_id=ab56ef89-c787-470c-b796-7d4b9fcdb2ee`).
- **Gold passages** for all five questions are present in the indexed vectors, but the stored chunk text does **not** exactly match the current local chunk at the same coordinate.
- No change to the existing gold labels is recommended; however, the mismatch in chunk boundaries may affect retrieval‑metric calculations.

## 2. Source Artifacts Inspected
| Artifact | Path | Purpose |
|---|---|---|
| Pinecone Chunk Audit Report | `backend/pinecone_chunk_audit_report.md` | Provides full list of fetched vectors, metadata, and similarity analysis against local chunks. |
| Retrieval Evaluation Report | `backend/retrieval_evaluation_report.md` | Supplies original Top‑5 retrieval results (used for context). |
| Retrieval Diversity Audit (corrected) | `backend/retrieval_diversity_audit_corrected_report_v2.md` | Supplies heuristic selection details (not directly needed for gold alignment). |
| Chunker implementation | `backend/chunker.py` | Defines current chunk generation logic (used to interpret gold coordinates). |
| Source PDF | `C:\Users\ASUS\Desktop\Coding-ninja-Gen-AI-Course\Coding Ninjas Learning PDF\Pre-Read_ Basic Understanding on API  (1)-5983.pdf` | Original document from which gold passages are derived. |

## 3. Pinecone Index Scope and Vector Inventory
- **Namespace:** `revon-policies`
- **Target document ID:** `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`
- **Logical document ID:** `20dcdd7c-a603-4a3f-b384-46081d625a73`
- **Filename:** `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`
- **Version:** `1`
- **Total vectors fetched:** **50**
- **All vectors validated** against the expected `document_id`/`logical_document_id`/`filename`/`version` and were confirmed to belong to the target document.
- **Active status:** All vectors are marked active.

## 4. Question‑by‑Question Gold Passage Analysis
| Question | Gold Coordinate (page/chunk) | Indexed Vector(s) at Same Coordinate | Stored Text Contains Gold Passage? | Evidence & Similarity |
|---|---|---|---|---|
| Q1 – DNS | p1/c1 | Vector `1/1` (metadata page = 1, chunk = 1) | **Yes** – the stored chunk includes the gold passage as a substring. | Similarity to local chunk = 0.6721 (character‑based). |
| Q2 – TCP Handshake | p2/c1 | Vector `2/1` | **Yes** – gold passage appears within the stored text. | Similarity = 0.6672. |
| Q3 – Serverless APIs | p8/c3 | Vector `8/3` | **Yes** – gold passage is present as a substring. | Similarity = 0.9594 (high). |
| Q4 – HTTP 403 | p16/c0 | Vector `16/0` | **Yes** – gold passage is contained in the stored chunk. | Similarity = 0.8827. |
| Q5 – HTTP Status Codes | p15/c1 | Vector `15/1` | **Yes** – gold passage appears within the stored text. | Similarity = 0.8038. |

**Interpretation**: For each gold coordinate, a vector exists with matching page/chunk metadata. The stored chunk text is a contiguous substring of the normalized source page, and the gold passage is found within that substring. However, the stored chunk does not *exactly* equal the current local chunk (0/50 exact matches). The gold coordinate therefore remains valid for locating the answer, but the surrounding context differs.

## 5. Existing vs. Recommended Gold Labels
| Question | Existing Gold Coordinate | Recommended Change | Rationale | Confidence |
|---|---|---|---|---|
| Q1 – DNS | p1/c1 | **No change** | Vector at p1/c1 contains the passage; coordinate is correct. | High |
| Q2 – TCP Handshake | p2/c1 | **No change** | Passage present in vector p2/c1. | High |
| Q3 – Serverless APIs | p8/c3 | **No change** | Passage present in vector p8/c3. | High |
| Q4 – HTTP 403 | p16/c0 | **No change** | Passage present in vector p16/c0. | High |
| Q5 – HTTP Status Codes | p15/c1 | **No change** | Passage present in vector p15/c1. | High |

No gold‑label revisions are recommended. The only discrepancy is that the stored chunk text differs from the current local chunk, which may affect downstream retrieval metrics.

## 6. Metrics Impact
- **Precision@3 / Recall@3:** If the evaluation system uses exact chunk‑text matching for relevance, the mismatch in chunk boundaries could cause a relevant passage to be scored lower (e.g., a gold‑label vector may be retrieved but its text differs from the expected local chunk, leading to a false‑negative). This can depress precision and recall.
- **MRR@3 (Mean Reciprocal Rank):** The rank of the gold‑label vector may remain unchanged (since the coordinate matches), so MRR may be unaffected, but if the system scores based on text similarity rather than coordinate, the rank could shift.
- **nDCG@3:** Graded relevance could be impacted if the gold passage is considered a lower‑grade match due to text differences, reducing the DCG contribution.

Overall, the presence of the gold passage at the correct coordinate means the *ranking* of the gold vector is likely unchanged, but the *content mismatch* could lead to lower relevance scores in metrics that consider textual overlap.

## 7. Limitations
- The audit relies on the Pinecone Chunk Audit Report for vector metadata and similarity ratios; raw full‑text excerpts are not reproduced here to avoid large dumps.
- No direct inspection of the PDF text was performed beyond the information already present in the audit report.
- The similarity metric is purely character‑based and does not capture semantic equivalence.
- No timestamp or `chunker_version` metadata is available to definitively determine when the indexed vectors were created.

## 8. Safety Verification
- **No Pinecone write operations** (upsert, update, delete) were performed.
- **No embedding or LLM API calls** were made.
- **No source code, configuration, or `.env` files** were modified.
- **No deployment or GitHub push** occurred.
- The only new artifact is this report file `backend/gold_label_alignment_audit_report.md`.
