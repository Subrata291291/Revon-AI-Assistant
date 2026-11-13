# Retrieval Diversity & Redundancy Audit – Corrected Simulation

**Audit timestamp:** 2026-11-13 15:51 UTC+05:30

## 1. Scope & Source Artifacts

| Artifact | Path | Use in this audit |
|---|---|---|
| Live RAG evaluation report | `backend/retrieval_evaluation_report.md` | Provides the original Top‑5 ranked results (page/chunk, score, excerpt) for the five test questions. |
| Pinecone chunk audit report | `backend/pinecone_chunk_audit_report.md` | Confirms that the 50 indexed vectors belong to the same PDF and that adjacent indexed chunks overlap, but does **not** contain the full text of the 25 vectors retrieved in the live evaluation. |
| Gold‑label list (unchanged) | – | Q1 p1/c1, Q2 p2/c1, Q3 p8/c3, Q4 p16/c0, Q5 p15/c1. |

Only these local, read‑only artifacts are consulted. No external API calls, embeddings, or Pinecone queries are performed.

## 2. Original Top‑5 Retrieval Summary

The tables below are reproduced verbatim from the live evaluation report. They show the ranked page/chunk pairs, similarity scores and the short excerpt that was logged (the full chunk text is **not** available).

### Q1 – DNS
| Rank | Page/Chunk | Score | Excerpt |
|---:|---|---:|---|
| 1 | 1/2 | 0.8208 | “Addresses DNS (Domain Name System) is like the phonebook of the internet…” |
| 2 | 1/3 | 0.7795 | “…Resolver asks the root DNS server…” |
| 3 | 2/0 | 0.7247 | “Imagine you want to call Pizza Hut …” |
| 4 | 1/1 | 0.7225 | “…Internet Protocol Stack working in harmony…” |
| 5 | 1/0 | 0.7043 | “BASIC UNDERSTANDING OF APIs…” |

### Q2 – TCP handshake
| Rank | Page/Chunk | Score | Excerpt |
|---:|---|---:|---|
| 1 | 2/1 | 0.7770 | “TCP Handshake. 2. TCP Handshake: Establishing a Reliable Connection…” |
| 2 | 2/2 | 0.7470 | “…process can data transfer begin. For Example…” |
| 3 | 6/0 | 0.7052 | “The 4‑Step FIN‑ACK Process…” |
| 4 | 4/2 | 0.6984 | “…sequence number, source/destination…” |
| 5 | 2/3 | 0.6907 | “…split it into parts and send them one by one…” |

### Q3 – Serverless APIs
| Rank | Page/Chunk | Score | Excerpt |
|---:|---|---:|---|
| 1 | 9/0 | 0.8146 | “Let’s break them down. 1. What is a Serverful API?…” |
| 2 | 10/0 | 0.7864 | “2. What is a Serverless API? A Serverless API means…” |
| 3 | 10/1 | 0.7735 | “…Spins up LLM inference engines…” |
| 4 | 9/1 | 0.7667 | “…generating embeddings or responses happens on this machine…” |
| 5 | 9/2 | 0.7644 | “…latency because the server is always warm…” |

### Q4 – HTTP 403
| Rank | Page/Chunk | Score | Excerpt |
|---:|---|---:|---|
| 1 | 15/1 | 0.7336 | “Did it fail? Did we make a mistake?… HTTP Status Codes…” |
| 2 | 15/2 | 0.7239 | “…Types of HTTP Status Codes…” |
| 3 | 16/1 | 0.7145 | “…400 Bad Request…” |
| 4 | 16/0 | 0.7089 | “200 OK… The request was successful…” |
| 5 | 13/2 | 0.6903 | “…protocol that ensures the client’s request is structured properly…” |

### Q5 – HTTP status codes
| Rank | Page/Chunk | Score | Excerpt |
|---:|---|---:|---|
| 1 | 15/1 | 0.8050 | “Did it fail? Did we make a mistake?… HTTP Status Codes…” |
| 2 | 15/2 | 0.7722 | “…Types of HTTP Status Codes…” |
| 3 | 16/0 | 0.7458 | “200 OK… The request was successful…” |
| 4 | 13/2 | 0.7439 | “…protocol that ensures the client’s request is structured properly…” |
| 5 | 15/0 | 0.7095 | “3. PUT… Update Something That Already Exists…” |

## 3. Corrected Diversified Top‑3 Selection

**Heuristic rule (exactly as stated in the task):**
> *A candidate is skipped only when it is on the same page **and** its chunk index differs by at most 1 from an already selected candidate.*

The rule is applied **independently per question**, walking the original rank list in order and stopping once three non‑redundant candidates have been accepted.

### Q1 – DNS (original ranks 1‑5)
1. Rank 1 → 1/2 – selected.
2. Rank 2 → 1/3 – same page, index diff = 1 → **skipped**.
3. Rank 3 → 2/0 – different page → selected.
4. Rank 4 → 1/1 – same page as 1/2, diff = 1 → **skipped**.
5. Rank 5 → 1/0 – same page, diff = 2 (not ≤ 1) → selected.

**Diversified Top‑3:** ranks 1, 3, 5 → (1/2, 2/0, 1/0).

### Q2 – TCP handshake
1. Rank 1 → 2/1 – selected.
2. Rank 2 → 2/2 – same page, diff = 1 → **skipped**.
3. Rank 3 → 6/0 – different page → selected.
4. Rank 4 → 4/2 – different page → selected (now three).

**Diversified Top‑3:** ranks 1, 3, 4 → (2/1, 6/0, 4/2).

### Q3 – Serverless APIs
1. Rank 1 → 9/0 – selected.
2. Rank 2 → 10/0 – different page → selected.
3. Rank 3 → 10/1 – same page as 10/0, diff = 1 → **skipped**.
4. Rank 4 → 9/1 – same page as 9/0, diff = 1 → **skipped**.
5. Rank 5 → 9/2 – same page as 9/0, diff = 2 → selected.

**Diversified Top‑3:** ranks 1, 2, 5 → (9/0, 10/0, 9/2).

### Q4 – HTTP 403
1. Rank 1 → 15/1 – selected.
2. Rank 2 → 15/2 – same page, diff = 1 → **skipped**.
3. Rank 3 → 16/1 – different page → selected.
4. Rank 4 → 16/0 – same page as 16/1, diff = 1 → **skipped**.
5. Rank 5 → 13/2 – different page → selected.

**Diversified Top‑3:** ranks 1, 3, 5 → (15/1, 16/1, 13/2).

### Q5 – HTTP status codes
1. Rank 1 → 15/1 – selected.
2. Rank 2 → 15/2 – same page, diff = 1 → **skipped**.
3. Rank 3 → 16/0 – different page → selected.
4. Rank 4 → 13/2 – different page → selected (now three).
5. Rank 5 → 15/0 – same page as 15/1, diff = 1 → **skipped** (but we already have three).

**Diversified Top‑3:** ranks 1, 3, 4 → (15/1, 16/0, 13/2).

## 4. Adjacency Metrics (Three Distinct Definitions)

### 4.1 Adjacent‑in‑rank same‑page pairs
Count of consecutive ranks where the page number is identical.
| Question | Adjacent‑in‑rank same‑page pairs |
|---|---|
| Q1 | 3 (1/2‑1/3, 1/3‑1/1, 1/1‑1/0) |
| Q2 | 1 (2/1‑2/2) |
| Q3 | 2 (9/0‑9/1, 9/1‑9/2) |
| Q4 | 2 (15/1‑15/2, 16/1‑16/0) |
| Q5 | 2 (15/1‑15/2, 16/0‑15/0) |

### 4.2 Consecutive page‑chunk coordinate pairs (|chunk‑index − other| = 1 on the same page)
We examine all unordered pairs among the five results and count those that satisfy the coordinate condition.
| Question | Consecutive‑coordinate pairs |
|---|---|
| Q1 | 4 (1/0‑1/1, 1/1‑1/2, 1/2‑1/3, 1/3‑1/0 is *not* consecutive) |
| Q2 | 2 (2/1‑2/2, 2/2‑2/3) |
| Q3 | 3 (9/0‑9/1, 9/1‑9/2, 10/0‑10/1) |
| Q4 | 2 (15/1‑15/2, 16/0‑16/1) |
| Q5 | 2 (15/1‑15/2, 15/0‑15/1) |

### 4.3 Literal text overlap between retrieved chunks
The Pinecone chunk audit confirms that **every adjacent indexed chunk in the full index overlaps by 79‑94 characters after normalization**. However, the live evaluation report only supplies short excerpts; the complete stored text for the 25 retrieved vectors is **not** present in any local artifact. Therefore we **cannot compute** exact literal overlap for these specific results. We note this limitation.

## 5. Gold‑Passage Coverage
We keep the original gold labels unchanged.
| Question | Gold page/chunk | Gold‑coordinate present in original Top‑5? | Gold‑passage text found in any excerpt? |
|---|---|---|---|
| Q1 – DNS | 1/1 | Yes (rank 4) | **Yes** – rank 1 excerpt contains the exact phrase “is like the phonebook of the internet”. |
| Q2 – TCP handshake | 2/1 | Yes (rank 1) | **Yes** – rank 1 excerpt explicitly mentions the TCP handshake. |
| Q3 – Serverless APIs | 8/3 | No (absent from Top‑5) | **Uncertain** – none of the five excerpts mention the exact “Serverful … Serverless” contrast found in the gold passage. |
| Q4 – HTTP 403 | 16/0 | Yes (rank 4) | **Uncertain** – excerpt for rank 4 talks about “200 OK” rather than the 403 description; no excerpt contains the phrase “You don’t have permission…”. |
| Q5 – HTTP status codes | 15/1 | Yes (rank 1) | **Yes** – rank 1 excerpt discusses HTTP status codes, matching the gold topic. |

**Note:** “Gold‑passage text found” is based solely on the short excerpts; the full chunk text could contain the passage even when the excerpt does not.

## 6. Corrected Diversified Top‑3 Summary
| Question | Diversified Top‑3 (rank → page/chunk) | Gold‑coordinate present? |
|---|---|---|
| Q1 – DNS | 1 → 1/2, 3 → 2/0, 5 → 1/0 | **No** (gold = 1/1, skipped as redundant) |
| Q2 – TCP handshake | 1 → 2/1, 3 → 6/0, 4 → 4/2 | **Yes** (gold = 2/1, kept) |
| Q3 – Serverless APIs | 1 → 9/0, 2 → 10/0, 5 → 9/2 | **No** (gold = 8/3, absent) |
| Q4 – HTTP 403 | 1 → 15/1, 3 → 16/1, 5 → 13/2 | **No** (gold = 16/0, skipped as redundant) |
| Q5 – HTTP status codes | 1 → 15/1, 3 → 16/0, 4 → 13/2 | **Yes** (gold = 15/1, kept) |

## 7. Limitations & Caveats
* The live report only provides truncated excerpts; full chunk text is unavailable, so literal overlap calculations are impossible for the specific retrieved vectors.
* The redundancy heuristic is **purely positional** (same page + |chunk‑index − other| ≤ 1). It does not consider semantic similarity or actual character overlap, which may be lower or higher than the positional expectation.
* Gold‑passage presence is judged from excerpts; a passage could be present in a chunk whose excerpt was omitted.
* The corrected diversified selection differs from the earlier report for Q4 and Q5 because rank 4 (16/0) is redundant with rank 3 (16/1) under the exact rule.

## 8. Recommendations (for future work only)
1. **Store full chunk text** alongside retrieval logs so that precise overlap and redundancy analyses can be performed without reconstructing from excerpts.
2. **Separate deduplication logic** from the heuristic used here; a production system should compute actual character overlap (e.g., Jaccard or Levenshtein) before discarding candidates.
3. **Synchronise gold‑label coordinates** with the active index version (e.g., embed a `chunker_version` metadata field) to avoid mismatches when the chunking strategy changes.
4. **Consider expanding `top_k`** and applying a post‑retrieval deduplication filter so that after removing overlapping neighbours the LLM still receives three distinct contexts.

---

## 9. Safety Verification
* No embedding APIs were called.
* No LLM (Gemini, Groq, etc.) calls were made.
* No Pinecone queries, listings, or fetches were performed (the earlier read‑only audit is already completed).
* No vectors were created, updated, or deleted.
* No document ingestion or re‑ingestion occurred.
* No code, configuration, or `.env` files were modified.
* No deployment or GitHub push was executed.
* All work was performed locally using existing markdown artifacts.
