# Read-Only Pinecone Chunk Audit

## Scope and method

- Audit timestamp: 2026-11-13 15:41:43 +05:30
- Target namespace: `revon-policies`
- Index: the existing index selected by the application's configured vector store; no alternate index was selected.
- Target document: `document_id=ab56ef89-c787-470c-b796-7d4b9fcdb2ee`, `logical_document_id=20dcdd7c-a603-4a3f-b384-46081d625a73`, filename `Pre-Read_-Basic-Understanding-on-API-1-5983.pdf`, version `1`.
- Pinecone SDK: 10.3.0.
- Read-only methods: `Index.list(prefix=..., limit=..., namespace=...)` and `Index.fetch(ids=..., namespace=...)`. Their signatures and listing prefix behavior were inspected before use. The server-side ID prefix was the known document UUID followed by `-page-`; every returned ID was validated against that document's deterministic page/chunk ID pattern, and only those IDs were fetched. No Pinecone query was needed.
- Local baseline: supplied PDF, current `pdf_processor.py` extraction and `chunker.py` normalization/chunk generation; 16 pages and 42 chunks.
- Text equality uses the application's current `normalize_text` for normalized comparisons. Candidate similarity is `difflib.SequenceMatcher` ratio (`autojunk=False`) between normalized stored text and each normalized local chunk. Similarity is character-based, not a relevance judgment.

## Inventory and identity

Found and fetched **50** target vectors. All vectors had complete text and the requested identity, version, page/chunk, filename, and active-status metadata. `document_id`, `logical_document_id`, filename, and version were consistent across the target set; every vector was active and version 1. No duplicate full chunk texts were found.

| Page | Indexed vectors | Current local chunks |
|---:|---:|---:|
| 1 | 4 | 3 |
| 2 | 4 | 3 |
| 3 | 3 | 2 |
| 4 | 3 | 3 |
| 5 | 3 | 3 |
| 6 | 3 | 2 |
| 7 | 2 | 1 |
| 8 | 4 | 4 |
| 9 | 4 | 3 |
| 10 | 4 | 3 |
| 11 | 3 | 3 |
| 12 | 2 | 2 |
| 13 | 3 | 3 |
| 14 | 2 | 2 |
| 15 | 3 | 2 |
| 16 | 3 | 3 |
| **Total** | **50** | **42** |

All 50 vector IDs had a page/chunk coordinate that could be parsed. **42/50** coordinates exist in the current local page/chunk ranges; **8/50** indexed coordinates have no current local chunk at that coordinate: p1/c3, p2/c3, p3/c2, p6/c2, p7/c1, p9/c3, p10/c3, p15/c2. Coordinate existence does not imply content equality: **0/50** stored texts equal the local chunk at the same coordinate.

## Exact text comparison

| Measure | Result |
|---|---:|
| Stored target vectors | 50 |
| Current local chunks | 42 |
| Exact whole-text matches | 0/50 (0%) |
| Exact whole-text matches after current normalization | 0/50 (0%) |
| Different-text stored chunks | 50/50 |
| Missing stored text | 0 |
| Duplicate full text | 0 |
| Stored text found as an exact contiguous substring of its normalized source page | 50/50 |
| Indexed coordinates present in local coordinate ranges | 42/50 |
| Indexed coordinates absent from local coordinate ranges | 8/50 |
| Same-coordinate text matches | 0/50 |

Each stored chunk is present as a contiguous substring of its corresponding normalized source page. Thus the comparison found no inserted or internally altered text relative to the source-page text after current normalization. However, the stored texts are not equal as standalone chunks to any of the current local chunks. Thirty stored chunks cross a current local chunk boundary when mapped onto the local page text.

Raw stored text lengths range from **148 to 800** characters (median **791.5**); after current normalization they range from **110 to 609** (median **579**). These are observed stored metadata text lengths, not reconstructed lengths.

## Per-vector best local candidate

`Stored p/c` is the Pinecone metadata coordinate. `Best local p/c` is the local chunk with the greatest normalized character similarity. The score is a string similarity ratio and must not be interpreted as semantic relevance.

| Stored p/c | Raw / normalized chars | Best local p/c | Similarity |
|---|---:|---|---:|
| 1/0 | 793 / 591 | 1/0 | 0.8743 |
| 1/1 | 791 / 591 | 1/1 | 0.6721 |
| 1/2 | 790 / 598 | 1/2 | 0.5878 |
| 1/3 | 355 / 261 | 1/2 | 0.6850 |
| 2/0 | 790 / 590 | 2/0 | 0.8657 |
| 2/1 | 797 / 609 | 2/1 | 0.6672 |
| 2/2 | 791 / 571 | 2/1 | 0.5760 |
| 2/3 | 414 / 310 | 2/2 | 0.7311 |
| 3/0 | 797 / 581 | 3/0 | 0.9043 |
| 3/1 | 792 / 586 | 3/1 | 0.8184 |
| 3/2 | 298 / 216 | 3/1 | 0.5100 |
| 4/0 | 797 / 589 | 4/0 | 0.9139 |
| 4/1 | 794 / 580 | 4/1 | 0.7394 |
| 4/2 | 533 / 401 | 4/1 | 0.6302 |
| 5/0 | 794 / 592 | 5/0 | 0.8567 |
| 5/1 | 791 / 591 | 5/1 | 0.6199 |
| 5/2 | 731 / 549 | 5/1 | 0.6706 |
| 6/0 | 796 / 458 | 6/0 | 0.7287 |
| 6/1 | 794 / 604 | 6/0 | 0.6101 |
| 6/2 | 721 / 539 | 6/1 | 0.8370 |
| 7/0 | 798 / 586 | 7/0 | 0.8839 |
| 7/1 | 326 / 238 | 7/0 | 0.4867 |
| 8/0 | 795 / 573 | 8/0 | 0.9106 |
| 8/1 | 799 / 575 | 8/1 | 0.8476 |
| 8/2 | 797 / 575 | 8/2 | 0.8719 |
| 8/3 | 666 / 500 | 8/3 | 0.9594 |
| 9/0 | 793 / 585 | 9/0 | 0.8565 |
| 9/1 | 790 / 592 | 9/1 | 0.6319 |
| 9/2 | 794 / 590 | 9/1 | 0.6284 |
| 9/3 | 191 / 147 | 9/2 | 0.5939 |
| 10/0 | 800 / 596 | 10/0 | 0.8876 |
| 10/1 | 796 / 592 | 10/1 | 0.7038 |
| 10/2 | 792 / 596 | 10/2 | 0.7306 |
| 10/3 | 148 / 110 | 10/2 | 0.4453 |
| 11/0 | 799 / 599 | 11/0 | 0.8868 |
| 11/1 | 796 / 596 | 11/1 | 0.7713 |
| 11/2 | 578 / 420 | 11/2 | 0.8119 |
| 12/0 | 784 / 576 | 12/0 | 0.8834 |
| 12/1 | 613 / 454 | 12/1 | 0.8477 |
| 13/0 | 792 / 594 | 13/0 | 0.9369 |
| 13/1 | 798 / 604 | 13/1 | 0.7983 |
| 13/2 | 709 / 529 | 13/2 | 0.7626 |
| 14/0 | 800 / 578 | 14/0 | 0.8975 |
| 14/1 | 728 / 536 | 14/1 | 0.9005 |
| 15/0 | 794 / 570 | 15/0 | 0.9484 |
| 15/1 | 788 / 574 | 15/1 | 0.8038 |
| 15/2 | 438 / 330 | 15/1 | 0.5882 |
| 16/0 | 793 / 591 | 16/0 | 0.8827 |
| 16/1 | 788 / 586 | 16/1 | 0.8442 |
| 16/2 | 750 / 554 | 16/2 | 0.9454 |

Best-candidate ratios range from **0.4453 to 0.9594**, with mean **0.7689** and median **0.8119**. The best candidates are on the same page as their stored chunks. High ratios indicate character resemblance only; exact matching and source-substring checks above are the stronger evidence.

## Adjacent indexed chunk overlap

There are 34 consecutive same-page chunk pairs among the 50 stored vectors. **All 34/34** have a literal suffix/prefix text overlap:

- Raw stored text overlap: **105–117 characters**.
- After current whitespace normalization: **79–94 characters**.

This confirms actual repeated text between adjacent indexed chunks, not merely nearby page/chunk IDs. The normalized overlap lengths differ because the stored metadata text contains whitespace that current normalization collapses. No full duplicate chunks were found.

## Representative source passages

The matching source text includes passages such as “DNS (Domain Name System) is like the phonebook of the internet,” the TCP SYN/SYN-ACK/ACK handshake, and the page 8 contrast between traditional always-on serverful APIs and event-driven serverless APIs. For HTTP 403, the local PDF uses a curly apostrophe in “You don’t have permission to access this resource”; this is a typography difference from the straight apostrophe in the supplied gold wording, not evidence of a different passage.

The five previously supplied gold locations remain present in the current local chunks: Q1 p1/c1, Q2 p2/c1, Q3 p8/c3, Q4 p16/c0, and Q5 p15/c1. Q2's passage also crosses/appears in an adjacent local chunk, as does Q4's passage across p16/c0 and p16/c1. These are local baseline findings; no Pinecone vectors or gold labels were changed.

## Freshness assessment and conclusion

Evidence that the indexed segmentation differs from the current local output:

1. Pinecone has 50 vectors while the current local output has 42 chunks.
2. None of the 50 stored texts matches a current chunk, even after current normalization.
3. Eight stored page/chunk coordinates do not exist in the current local coordinate ranges.
4. Thirty stored chunks cross current local chunk boundaries.
5. All 34 adjacent indexed chunk pairs have measurable literal overlap.

At the same time, all 50 normalized stored texts are exact source-page substrings, and the target identity is consistent. This supports a different chunk boundary/layout for the same source content rather than unrelated text. The current report of an earlier 40-chunk build does **not** establish that Pinecone contains that exact build: the audit found 50 indexed vectors. Document version `1` is not a chunker-version identifier. No ingestion timestamp or chunker-version metadata was available to establish exactly when or with which implementation these vectors were created.

**Conclusion:** the active indexed chunks do not match the current 42-chunk local output. The evidence is consistent with older or otherwise different segmentation of the same PDF, but the exact chunker version and index freshness cannot be established from the available metadata.

## Safety verification

- No embeddings were generated.
- No Pinecone vectors were created, updated, or deleted.
- No document was ingested or re-ingested.
- No application code or configuration was changed for this audit.
- No deployment or GitHub push occurred.
- Only target-prefix listing and fetch-by-validated-target-ID were used; no unrelated vectors were fetched.
- Complete fetched texts were compared locally and were not saved as a raw dump. This report contains only audit metrics and limited representative excerpts.
