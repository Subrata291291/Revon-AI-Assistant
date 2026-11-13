# Live RAG Retrieval Evaluation

**Run date:** 2026-11-13  
**Mode:** Read-only query evaluation; no LLM completion calls.

## Pre-flight configuration

| Setting | Verified value | Evidence |
|---|---|---|
| Embedding model | `gemini-embedding-2` | `embedding_service.py` |
| Query embedding dimension | 768 | Embedding service and vector-store validation; all five returned vectors had 768 dimensions |
| Configured index | `revon-policy-index` | Host label matched the configured index name |
| Namespace | `revon-policies` | `vector_store.py` |
| Similarity metric | `cosine` | Existing index configuration in `create_pinecone_index.py` |
| Active-vector filter | `is_active == true` | Normal query behavior; explicitly enabled for every query |
| Query operation | Read-only Pinecone `/query` | Existing `query_vectors` implementation |

Credentials and the Pinecone host/connection string are intentionally omitted.
The metric above is the repository's configured metric; this run did not make
a separate Pinecone index-description request.

## Evaluation labels and method

The five questions and their supplied page/chunk gold labels were used without
changing the labels. Relevance is an exact page-number and chunk-index match
within the intended document. A result is counted only once for a duplicate
document/page/chunk identity.

The returned filename is `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`;
the supplied local PDF is `Pre-Read_ Basic Understanding on API  (1)-5983.pdf`.
The filename strings differ, but their alphanumeric-normalized names match.
Every returned match had the same document ID, logical document ID, filename,
and version, which supports identifying these results as the intended PDF.

For each question, Gemini generated a query embedding and Pinecone returned
five matches in its original ranking order. No client-side reranking was
performed. Excerpts below are truncated to avoid reproducing full document
chunks.

## Live top-five results

All rows below had `active=true`; none of the requested metadata fields were
missing. Each row returned document ID
`ab56ef89-c787-470c-b796-7d4b9fcdb2ee`, logical document ID
`20dcdd7c-a603-4a3f-b384-46081d625a73`, version `1`, and filename
`Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`.

### Q1 — DNS

Question: How does DNS translate domain names to IP addresses?  
Gold: page 1, chunk 1.

| Rank | Page/chunk | Score | Gold match | Excerpt |
|---:|---|---:|:---:|---|
| 1 | 1/2 | 0.820759 | No | Addresses DNS (Domain Name System) is like the phonebook of the internet… |
| 2 | 1/3 | 0.779504 | No | …Resolver asks the root DNS server… |
| 3 | 2/0 | 0.724725 | No | Imagine you want to call Pizza Hut but don’t know the number… |
| 4 | 1/1 | 0.722509 | Yes | …Internet Protocol Stack working in harmony… |
| 5 | 1/0 | 0.704289 | No | BASIC UNDERSTANDING OF APIs… |

The relevant DNS passage appears in rank 1's neighboring chunk excerpt, but the
strict gold chunk (page 1/chunk 1) is rank 4 and therefore is not counted in
P@3 or Recall@3.

### Q2 — TCP handshake

Question: Which messages make up the TCP three-way handshake?  
Gold: page 2, chunk 1.

| Rank | Page/chunk | Score | Gold match | Excerpt |
|---:|---|---:|:---:|---|
| 1 | 2/1 | 0.777012 | Yes | TCP Handshake. 2. TCP Handshake: Establishing a Reliable Connection… |
| 2 | 2/2 | 0.747026 | No | …process can data transfer begin. For Example… |
| 3 | 6/0 | 0.705172 | No | The 4-Step FIN-ACK Process… |
| 4 | 4/2 | 0.698388 | No | …sequence number, source/destination… |
| 5 | 2/3 | 0.690702 | No | …split it into parts and send them one by one… |

### Q3 — Serverless APIs

Question: How do serverless APIs differ from serverful APIs?  
Gold: page 8, chunk 3.

| Rank | Page/chunk | Score | Gold match | Excerpt |
|---:|---|---:|:---:|---|
| 1 | 9/0 | 0.814555 | No | Let’s break them down. 1. What is a Serverful API?… |
| 2 | 10/0 | 0.786444 | No | 2. What is a Serverless API? A Serverless API means… |
| 3 | 10/1 | 0.773531 | No | …Spins up LLM inference engines… |
| 4 | 9/1 | 0.766661 | No | …generating embeddings or responses happens on this machine… |
| 5 | 9/2 | 0.764442 | No | …latency because the server is always warm… |

The retrieved chunks are topically relevant to the distinction, but the
supplied exact gold location (page 8/chunk 3) is absent from the top five. This
evaluation cannot establish whether that chunk exists elsewhere in the active
index.

### Q4 — HTTP 403

Question: What does HTTP 403 Forbidden mean?  
Gold: page 16, chunk 0.

| Rank | Page/chunk | Score | Gold match | Excerpt |
|---:|---|---:|:---:|---|
| 1 | 15/1 | 0.733647 | No | Did it fail? Did we make a mistake?… HTTP Status Codes… |
| 2 | 15/2 | 0.723875 | No | …Types of HTTP Status Codes… |
| 3 | 16/1 | 0.714509 | No | …400 Bad Request… |
| 4 | 16/0 | 0.708881 | Yes | 200 OK… The request was successful… |
| 5 | 13/2 | 0.690340 | No | …protocol that ensures the client’s request is structured properly… |

### Q5 — HTTP status codes

Question: What does an HTTP status code tell a client about its request?  
Gold: page 15, chunk 1.

| Rank | Page/chunk | Score | Gold match | Excerpt |
|---:|---|---:|:---:|---|
| 1 | 15/1 | 0.804955 | Yes | Did it fail? Did we make a mistake?… HTTP Status Codes… |
| 2 | 15/2 | 0.772188 | No | …Types of HTTP Status Codes… |
| 3 | 16/0 | 0.745803 | No | 200 OK… The request was successful… |
| 4 | 13/2 | 0.743901 | No | …protocol that ensures the client’s request is structured properly… |
| 5 | 15/0 | 0.709531 | No | 3. PUT… Update Something That Already Exists… |

## Metrics

Metrics use the first three results, as requested. Precision@3 uses a fixed
denominator of three. Each question has one gold chunk, so Recall@3 is 1 when
that unique gold chunk appears in the first three, otherwise 0. Reciprocal
rank is zero when the gold chunk is below rank 3 or absent from the top five.

| Question | Gold found in top 5 | Gold rank | First result relevant | Precision@3 | Recall@3 | Reciprocal rank |
|---|:---:|---:|:---:|---:|---:|---:|
| Q1 DNS | Yes | 4 | No | 0.000 | 0.000 | 0.000 |
| Q2 TCP handshake | Yes | 1 | Yes | 0.333 | 1.000 | 1.000 |
| Q3 Serverless APIs | No | — | No | 0.000 | 0.000 | 0.000 |
| Q4 HTTP 403 | Yes | 4 | No | 0.000 | 0.000 | 0.000 |
| Q5 HTTP status codes | Yes | 1 | Yes | 0.333 | 1.000 | 1.000 |
| **Macro average** |  |  |  | **0.133** | **0.400** | **0.400** |

## Retrieval observations

- All five embedding and Pinecone query pairs succeeded; all query embeddings
  had 768 dimensions.
- All 25 results were active, came from the same document/version, and included
  all requested metadata fields.
- The exact gold chunk appeared in the top three for two questions (Q2 and Q5),
  ranked 4th for Q1 and Q4, and did not appear in the top five for Q3.
- Each question's top three included neighboring chunk indexes on the same
  page. This is consistent with overlap redundancy and can spend top-K slots
  on repeated context.
- Q1's top result excerpt contains the expected DNS passage but belongs to
  chunk 2 rather than the specified gold chunk 1. Under the required exact
  page/chunk metric it is not counted as relevant.
- Q3's top results describe serverful/serverless APIs on pages 9 and 10, but
  the supplied page 8/chunk 3 label was not returned. This may indicate a
  ranking miss or an index/chunk-label alignment issue; a top-five query alone
  cannot distinguish them. The supplied label was not changed.
- Q1 and Q4 gold chunks rank below the top three with scores below the first
  results. Q3 returns scores from 0.764 to 0.815 without the labeled chunk.
  These scores are relative to each query; no relevance threshold was applied
  or assumed.

## Recommendations (not implemented)

1. Verify the indexed chunk inventory and chunk-version alignment for the
   active document, especially page 8/chunk 3, using an approved read-only
   metadata inspection. Do not infer index absence from top-five results.
2. Expand the manually judged query set and distinguish exact chunk labels
   from passages whose content is duplicated into overlapping neighbors.
3. Measure top-K redundancy on a larger live evaluation before considering a
   diversity or overlap-aware retrieval strategy.
4. Calibrate any future relevance threshold on judged live queries; the current
   five-question sample does not justify a threshold.

## Safety confirmation

- No Pinecone upsert was performed.
- No Pinecone update was performed.
- No Pinecone delete was performed.
- No document ingestion or re-ingestion was performed.
- No vectors or metadata were modified.
- No `.env`, credentials, or production configuration were changed.
- No deployment or GitHub push occurred.
- Only Gemini query embeddings and read-only Pinecone vector queries were
  performed.
