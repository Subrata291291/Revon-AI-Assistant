# Gemini Embedding Daily Quota Protection — Production Hardening

> **Status:** local implementation and unit tests completed; **not production-ready** until persistence, corruption recovery operations, and distributed concurrency guarantees are verified in the target deployment.

## Scope
- Model: `gemini-embedding-2`
- Embedding dimension: `768`
- Existing Pinecone index/namespace, tenant isolation, active-document filtering, deterministic vector IDs, idempotent upserts, and incremental/resumable ingestion are preserved.
- Local daily accounting is an **estimate only**; **Google quota/usage is authoritative**. Do not assume Google quota reset time.

## 1) Persistent state durability (required for production accounting)

### Current implementation (filesystem-backed local estimate)
- Config: `GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET`, `GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED`, `GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID`, `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH`.
- Default state path: `tempdir/revon-gemini-embedding-daily-quota.json` (implementation fallback). **This is not durable production storage** in many environments.
- State is read/written as JSON with atomic replace for normal updates; counts every Gemini embedding API request attempt (including retries) and stops before the next request would exceed the configured local budget.

### Required production configuration
- **Do not rely on ephemeral temporary storage** (container layer, `/tmp`, pod filesystem without a volume, serverless scratch, or any path lost on restart/redeploy/scale event).
- Set `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH` to a **durable persistent volume/path** mounted into the backend runtime (e.g. attached disk/PVC/network volume) with correct read/write permissions for the app user.
- Ensure the path **survives process restarts, container restarts, and deployments**. **Local filesystem persistence does not guarantee survival across deployments/hosts** unless the same durable volume is explicitly retained and remounted.
- Back up and monitor the state file as operational data (it is part of quota-safety controls, but still only a local estimate).
- For multi-instance/distributed deployments, a shared filesystem path alone is **not sufficient** for correctness (see concurrency). Use a shared transactional store.

### Durability limitations (filesystem backend)
- Safe for single-process/single-instance use on a truly durable mounted volume (subject to filesystem semantics).
- Not a guarantee against disk loss, volume re-provisioning, image redeploys without volume reattach, or node migration.
- No distributed transaction/coordination; cannot safely enforce a global budget across workers/instances.

### Evaluation: Supabase/PostgreSQL-backed implementation (recommended for distributed production)
A PostgreSQL-backed store (including Supabase-hosted Postgres) is the recommended evolution for distributed deployments because it provides shared state and transactional atomic reservation.

**Suggested design:**
- Table (example): `gemini_embedding_daily_budget(window_id text primary key, used_requests bigint not null check (used_requests >= 0), budget_snapshot bigint null, updated_at timestamptz not null default now(), note text null)`.
- **Atomic reservation per request attempt** (before issuing Gemini request, including retries):
  - Ensure row exists for configured window (idempotent init, do not auto-reset by time).
  - `UPDATE gemini_embedding_daily_budget SET used_requests = used_requests + 1, updated_at = now() WHERE window_id = $1 AND used_requests < $2 RETURNING used_requests;`
  - If no row returned -> local budget exhausted (fail closed for this attempt, non-retryable budget error). If returned -> attempt is reserved/accounted.
- **Operator-controlled window reset:** explicit admin operation that creates/activates a new `window_id` (or rotates configured window) rather than time-based reset; optionally archive prior windows. Never assume Google reset time.
- **Fail-closed behavior:** on DB connectivity/transaction errors, do not issue embedding requests (fail closed) and surface clear operational error; require explicit recovery/verification.
- **Supabase considerations:** can be accessed via direct Postgres connection string (preferred for server-side transactional control/connection pooling, e.g. PgBouncer/transaction mode compatibility reviewed) or via Supabase APIs/RPC; atomic reservation must execute server-side in a single statement/transaction. Manage credentials/least privilege, TLS, migrations, and observability. Evaluate latency impact on embedding request path and retry behavior.
- **Migration path:** introduce a storage abstraction (`filesystem` vs `postgres`) selected by config (e.g. `GEMINI_EMBEDDING_DAILY_BUDGET_STORE`), keep filesystem for controlled single-instance/dev, require Postgres for distributed prod.

**Not implemented in this change:** no PostgreSQL/Supabase store, schema migration, or DB dependency added; this is an evaluation/design.

## 2) Fail-closed corruption handling (filesystem backend)

### Behavior
- If the configured state file **does not exist**: initialize a fresh local state for the configured window (normal first-run).
- If an **existing** state file is **corrupted, invalid, or unreadable**: **fail closed**. The system raises a clear non-retryable `EmbeddingDailyBudgetStateError` and **does not issue another Gemini embedding request** and **does not silently reset `used_requests` to zero**.
- Corrupt/invalid includes malformed/partial JSON (including interrupted-write symptoms leaving truncated target content), wrong schema/types, or negative/ambiguous counters. Unreadable includes permission/IO errors on an existing file.
- Interrupted writes: normal writes use temp-file + atomic replace; a leftover temp file alone is not treated as authoritative state. A partially written/corrupt target state file is treated as corrupt and fails closed.

### Explicit operator recovery (required before requests resume)
- Inspect and repair the state file at `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH` if an accurate local count can be restored, **or**
- Perform an explicit controlled recovery by setting `GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE=reset` for a recovery run **after verifying local accounting impact**. This explicit action reinitializes the local counter for the configured window (records recovery metadata) and allows requests to proceed under the configured budget/window. It is **not** automatic.
- After recovery, unset/remove recovery mode and verify state/budget/window configuration.

### Operational error messaging
Errors identify the state path, explain fail-closed rationale (avoid unsafe quota-accounting reset; local estimate vs Google authoritative), and provide recovery instructions.

## 3) Multi-worker concurrency

### Current guarantee (as implemented)
- **Single-process only (best-effort):** in-process `threading` lock + file read-modify-write with atomic replace.
- **Not safe across multiple processes/instances/containers/workers:** concurrent workers can read the same `used_requests` and both write increments, allowing the configured local budget to be exceeded. Atomic file replace does not provide cross-process atomic read-modify-write reservation.
- **Not safe across deployments/hosts without shared durable transactional state:** separate filesystems/instances have independent counters.

### Requirement for distributed production
- Use a **shared transactional storage** backend with **atomic reservation** per embedding API request attempt (including retries) so concurrent workers cannot exceed the configured local budget for the same window.

### Proposed PostgreSQL atomic reservation design
- **Schema:** one row per `window_id` tracking `used_requests` (and metadata). Initialize row explicitly for active window (operator/config-managed window id; no time-based reset).
- **Reserve-before-request (per attempt, including retries):**
  1. Begin transaction (or single autocommitted atomic statement).
  2. `UPDATE gemini_embedding_daily_budget SET used_requests = used_requests + 1, updated_at = now() WHERE window_id = $1 AND used_requests < $2 RETURNING used_requests;`
  3. Commit. If a row is returned, the attempt is safely reserved -> proceed with RPM pacing and Gemini API call. If no row is returned, budget is exhausted -> raise clear non-retryable daily-budget error **without** issuing the Gemini request.
- **Properties:** atomic check-and-increment prevents concurrent workers from exceeding budget; each retry attempt performs its own reservation (preserving retry accounting). No Gemini request is issued unless reserved.
- **Failure handling:** if reservation transaction fails/DB unavailable, fail closed (no request issued) with operational error. Handle idempotency carefully: do not "refund" a committed reservation on later Gemini/network failure, because the provider attempt may have been consumed; this matches current conservative local-estimate behavior.
- **Compatibility:** preserves incremental ingestion/resume (Pinecone deterministic IDs/idempotent upserts remain the source for chunk progress; budget store only governs request attempts).

### Verification status
- Filesystem single-process behavior covered by local unit tests (including fail-closed corruption/recovery cases added in this change).
- **Distributed/multi-process concurrency guarantees are not verified** (no shared transactional backend implemented/tested). **Do not claim production readiness for multi-worker/distributed deployments.**

## Configuration reference (env)
- `GEMINI_EMBEDDING_DAILY_BUDGET_ENABLED` (default `true`)
- `GEMINI_EMBEDDING_DAILY_REQUEST_BUDGET` (default `1000`, local estimate)
- `GEMINI_EMBEDDING_DAILY_BUDGET_WINDOW_ID` (default `default`, operator-controlled)
- `GEMINI_EMBEDDING_DAILY_BUDGET_STATE_PATH` (default temp path; **set to durable persistent volume in production**)
- `GEMINI_EMBEDDING_DAILY_BUDGET_RECOVERY_MODE` (`reset` only for explicit controlled recovery of corrupted/unreadable existing state)
- Existing: `GEMINI_EMBEDDING_RPM`, `GEMINI_EMBEDDING_BATCH_SIZE`, retry/backoff settings.