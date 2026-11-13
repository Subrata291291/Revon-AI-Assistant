# RevOn Gold Dataset v1.0.0 — Annotation Policy

## Status
Approved policy: **Broad Policy**. Q3 vector `9/0` is Grade 1.

## Relevance grades
- **Grade 0 — Irrelevant / insufficient:** The chunk does not provide substantive evidence for the query.
- **Grade 1 — Supporting evidence:** The chunk contributes meaningful partial evidence, a prerequisite definition, a relevant example/analogy, or a useful part of a multi-part answer.
- **Grade 2 — Direct answer:** The chunk independently provides the primary answer or essential mechanism requested.

## Binding Q3 decision
Vector `9/0` defines the Serverful API baseline (dedicated server/VM and always-running operation). For a comparative Serverful-vs-Serverless question, this is meaningful prerequisite context, but it does not independently define Serverless. Therefore it is Grade 1.

## Other direct-answer examples
- Q1 `1/2`: Grade 2 — step-by-step DNS resolution.
- Q2 `2/1`: Grade 2 — SYN, SYN-ACK, ACK handshake.
- Q3 `10/0` and `9/2`: Grade 2 — operational Serverless definition and direct comparison.
- Q4 `16/1`: Grade 2 — verbatim HTTP 403 definition.
- Q5 `15/1`: Grade 2 — definition and purpose of HTTP status codes.

## Coordinate policy
Coordinates are optional mappings into a specific index snapshot. They are not canonical ground-truth identifiers. Canonical passage text and source-document SHA-256 are authoritative.

## Change governance
Any relevance-grade change requires rationale, reviewer, timestamp, and annotation revision. Never silently overwrite a frozen annotation.
