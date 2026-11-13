# Gold Dataset v1.0.0 — Freeze Record

## Status
**Frozen evaluation snapshot prepared.** Canonical source passages, approved Broad Policy relevance judgments, index-coordinate mappings, and historical Top-3 metric snapshots are packaged as v1.0.0.

**Reproducibility limitation:** Verbatim historical query prompts were not present in the available audit artifacts. `query_text` is null; this package cannot replay the exact original query embedding requests. This limitation is explicitly recorded and must not be silently filled with reconstructed prompts.

## Source integrity
- Source: `Pre-Read_ Basic Understanding on API  (1)-5983.pdf`
- Pages: 16
- Algorithm: SHA-256
- Expected: `ed74f10871a921595c058a90f552f169b79b7198287f3551d4a465aeeaf87856`
- Calculated: `ed74f10871a921595c058a90f552f169b79b7198287f3551d4a465aeeaf87856`
- Result: **MATCH**

## Approved decisions
1. Broad Policy: Q3 vector `9/0` = Grade 1.
2. Report Evidence-MRR@3 (Grade >= 1) and Direct Answer Reciprocal Rank@3 (Grade = 2).
3. Canonical ground truth is source passage text anchored to the source PDF hash; coordinates are optional index mappings.
4. Q4 active-index coordinate is `16/1`, not `16/0`.

## Baseline
See `gold_dataset_v1_metric_baseline.json`. Values are preserved from the completed metric consistency audit, not recalculated from a live retrieval run.

## Snapshot provenance
- Document ID: `ab56ef89-c787-470c-b796-7d4b9fcdb2ee`
- Logical document ID: `20dcdd7c-a603-4a3f-b384-46081d625a73`
- Pinecone namespace: `revon-policies`
- Active vector count in audit: 50
- Query count: 5
- Relevant chunk instances: 19

## Limitations
- Five-query evaluation sample only.
- Chunk-level relevance can be affected by overlapping chunks.
- Diversification is an offline positional heuristic simulation, not proof of production improvement.
- Coordinates are index-version-specific and may shift after re-ingestion.
- Exact original query strings are not preserved in available audit artifacts.

## Safety boundary
Prepared from the supplied source PDF and completed audit artifacts. No Pinecone request, embedding/LLM API call, vector write, ingestion, application-code change, `.env` change, deployment, or GitHub push was performed.
