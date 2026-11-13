import os
import httpx
from dotenv import load_dotenv

load_dotenv()


# ==========================================
# 1. Pinecone Configuration
# ==========================================

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")

DIMENSION = 768
NAMESPACE = "revon-policies"
BATCH_SIZE = 100
LIST_PAGE_SIZE = 100


def get_headers():
    if not API_KEY:
        raise ValueError("PINECONE_API_KEY is missing")

    return {
        "Api-Key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }


def get_base_url():
    if not INDEX_HOST:
        raise ValueError("PINECONE_INDEX_HOST is missing")

    host = INDEX_HOST.replace("https://", "").rstrip("/")
    return f"https://{host}"


# ==========================================
# 2. Upsert Vectors
# ==========================================

def upsert_vectors(
    embedded_chunks: list[dict],
    document_id: str,
    logical_document_id: str,
    version: int,
    filename: str,
    request_id: str | None = None,
    expected_vectors: int | None = None
):
    """
    Store one policy version in Pinecone.

    All new vectors are initially inactive.

    Vector IDs are deterministic:
        {document_id}-page-{page_number}-chunk-{chunk_index}

    Therefore, retrying the same operation with the same document_id
    overwrites the same vector IDs instead of creating duplicates.

    request_id and expected_vectors are optional for backward
    compatibility. The durable operation coordinator should supply
    both when the new upload workflow is implemented.
    """

    if not embedded_chunks:
        raise ValueError("No embedded chunks provided")

    if not document_id or not logical_document_id:
        raise ValueError("Document IDs are required")

    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError("Version must be an integer")

    if version < 1:
        raise ValueError("Version must be at least 1")

    if request_id is not None and (
        not isinstance(request_id, str) or not request_id.strip()
    ):
        raise ValueError("request_id must be a non-empty string")

    if expected_vectors is not None:
        if (
            not isinstance(expected_vectors, int)
            or isinstance(expected_vectors, bool)
            or expected_vectors < 1
        ):
            raise ValueError("expected_vectors must be a positive integer")

        if expected_vectors != len(embedded_chunks):
            raise ValueError(
                "Expected vector count does not match embedded chunk count."
            )

    base_url = get_base_url()
    headers = get_headers()

    total_upserted = 0

    with httpx.Client(timeout=60.0) as client:
        for start in range(0, len(embedded_chunks), BATCH_SIZE):
            batch = embedded_chunks[start:start + BATCH_SIZE]
            vectors = []

            for chunk in batch:
                embedding = chunk["embedding"]

                if len(embedding) != DIMENSION:
                    raise ValueError(
                        f"Expected {DIMENSION} dimensions, "
                        f"received {len(embedding)}"
                    )

                vector_id = (
                    f"{document_id}-"
                    f"page-{chunk['page_number']}-"
                    f"chunk-{chunk['chunk_index']}"
                )

                metadata = {
                    "document_id": document_id,
                    "logical_document_id": logical_document_id,
                    "version": version,
                    "is_active": False,
                    "filename": filename,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"]
                }

                # Keep metadata backward-compatible for existing records.
                if request_id is not None:
                    metadata["request_id"] = request_id

                if expected_vectors is not None:
                    metadata["expected_vectors"] = expected_vectors

                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })

            payload = {
                "vectors": vectors,
                "namespace": NAMESPACE
            }

            response = client.post(
                f"{base_url}/vectors/upsert",
                headers=headers,
                json=payload
            )

            response.raise_for_status()

            result = response.json()
            batch_count = result.get("upsertedCount")

            if batch_count != len(batch):
                raise RuntimeError(
                    "Pinecone upsert count mismatch. "
                    f"Expected {len(batch)}, "
                    f"received {batch_count}."
                )

            total_upserted += batch_count

    if expected_vectors is not None and total_upserted != expected_vectors:
        raise RuntimeError(
            "Total Pinecone upsert count mismatch. "
            f"Expected {expected_vectors}, received {total_upserted}."
        )

    return {
        "status": "success",
        "document_id": document_id,
        "logical_document_id": logical_document_id,
        "version": version,
        "request_id": request_id,
        "expected_vectors": expected_vectors,
        "total_upserted": total_upserted
    }


# ==========================================
# 3. List Document Vector IDs
# ==========================================

def list_document_vector_ids(document_id: str) -> list[str]:
    """
    List every vector ID belonging to one document.

    Uses Pinecone serverless prefix-based pagination.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    base_url = get_base_url()
    headers = get_headers()

    vector_ids = []
    pagination_token = None
    expected_prefix = f"{document_id}-"

    with httpx.Client(timeout=30.0) as client:
        while True:
            params = {
                "namespace": NAMESPACE,
                "prefix": expected_prefix,
                "limit": LIST_PAGE_SIZE
            }

            if pagination_token:
                params["paginationToken"] = pagination_token

            response = client.get(
                f"{base_url}/vectors/list",
                headers=headers,
                params=params
            )

            response.raise_for_status()
            result = response.json()

            for vector in result.get("vectors", []):
                vector_id = vector.get("id")

                if not isinstance(vector_id, str):
                    raise RuntimeError(
                        "Pinecone returned a vector without a valid ID."
                    )

                # Defensive check: never accept a prefix collision.
                if vector_id.startswith(expected_prefix):
                    vector_ids.append(vector_id)

            pagination_token = (
                result.get("pagination", {}).get("next")
            )

            if not pagination_token:
                break

    return vector_ids


def verify_document_vector_count(
    document_id: str,
    expected_vectors: int
) -> dict:
    """
    Compare Pinecone's currently listed vector IDs with an expected count.

    Note:
    Pinecone listing may be eventually consistent immediately after upsert.
    A mismatch should be treated as a verification-pending state and retried
    by the operation coordinator, not as proof that the upload failed.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    if (
        not isinstance(expected_vectors, int)
        or isinstance(expected_vectors, bool)
        or expected_vectors < 1
    ):
        raise ValueError("expected_vectors must be a positive integer")

    vector_ids = list_document_vector_ids(document_id)
    actual_vectors = len(vector_ids)

    return {
        "document_id": document_id,
        "expected_vectors": expected_vectors,
        "actual_vectors": actual_vectors,
        "verified": actual_vectors == expected_vectors,
        "vector_ids": vector_ids
    }



# ==========================================
# 3A. Fetch and Verify Document Metadata
# ==========================================

def get_document_vector_metadata(document_id: str) -> dict[str, dict]:
    """
    Fetch metadata for every vector belonging to a document.

    The document ID prefix is used to discover vector IDs. Metadata is then
    fetched in batches so callers can verify document ownership before
    activation.

    This function verifies the vectors currently visible to Pinecone.
    It does not create a durable operation record.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    vector_ids = list_document_vector_ids(document_id)
    if not vector_ids:
        return {}

    base_url = get_base_url()
    headers = get_headers()
    fetched = {}

    with httpx.Client(timeout=30.0) as client:
        for start in range(0, len(vector_ids), 1000):
            batch_ids = vector_ids[start:start + 1000]

            response = client.get(
                f"{base_url}/vectors/fetch",
                headers=headers,
                params=[
                    *[("ids", vector_id) for vector_id in batch_ids],
                    ("namespace", NAMESPACE)
                ]
            )

            response.raise_for_status()

            if not response.content:
                raise RuntimeError(
                    "Pinecone fetch returned an empty response."
                )

            result = response.json()

            vectors = result.get("vectors", {})
            if not isinstance(vectors, dict):
                raise RuntimeError("Unexpected Pinecone fetch response.")

            for vector_id, vector_data in vectors.items():
                if not isinstance(vector_data, dict):
                    raise RuntimeError(
                        f"Invalid Pinecone vector payload for {vector_id}."
                    )

                metadata = vector_data.get("metadata")
                if not isinstance(metadata, dict):
                    raise RuntimeError(
                        f"Missing metadata for Pinecone vector {vector_id}."
                    )

                fetched[vector_id] = metadata

    return fetched


def verify_document_ownership(
    document_id: str,
    logical_document_id: str,
    version: int | None = None,
    request_id: str | None = None,
    expected_vectors: int | None = None
) -> dict:
    """
    Verify all discovered vectors belong to the requested document lineage.

    A missing vector or mismatched metadata causes verification to fail.
    """

    if not document_id or not logical_document_id:
        raise ValueError("Document IDs are required")

    metadata_by_id = get_document_vector_metadata(document_id)

    if not metadata_by_id:
        raise ValueError(
            f"No Pinecone vectors found for document: {document_id}"
        )

    mismatches = []

    for vector_id, metadata in metadata_by_id.items():
        if metadata.get("document_id") != document_id:
            mismatches.append(f"{vector_id}: document_id mismatch")

        if metadata.get("logical_document_id") != logical_document_id:
            mismatches.append(f"{vector_id}: logical_document_id mismatch")

        if version is not None and metadata.get("version") != version:
            mismatches.append(f"{vector_id}: version mismatch")

        if request_id is not None and metadata.get("request_id") != request_id:
            mismatches.append(f"{vector_id}: request_id mismatch")

    actual_vectors = len(metadata_by_id)

    if expected_vectors is not None and actual_vectors != expected_vectors:
        mismatches.append(
            f"vector count mismatch: expected {expected_vectors}, "
            f"found {actual_vectors}"
        )

    if mismatches:
        raise ValueError(
            "Pinecone document verification failed: " + "; ".join(mismatches)
        )

    return {
        "document_id": document_id,
        "logical_document_id": logical_document_id,
        "version": version,
        "request_id": request_id,
        "actual_vectors": actual_vectors,
        "verified": True
    }

def verify_document_active_state(
    document_id: str,
    expected_active: bool,
    expected_vectors: int | None = None
) -> dict:
    """
    Verify that every vector belonging to a document has
    the expected is_active state.

    This checks the current metadata visible in Pinecone.
    It does not provide an atomic snapshot or durable
    activation operation record.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    if not isinstance(expected_active, bool):
        raise ValueError("expected_active must be a boolean")

    metadata_by_id = get_document_vector_metadata(document_id)

    if not metadata_by_id:
        raise ValueError(
            f"No Pinecone vectors found for document: {document_id}"
        )

    actual_vectors = len(metadata_by_id)

    if (
        expected_vectors is not None
        and actual_vectors != expected_vectors
    ):
        raise ValueError(
            f"Vector count mismatch: expected {expected_vectors}, "
            f"found {actual_vectors}"
        )

    mismatches = []

    for vector_id, metadata in metadata_by_id.items():
        actual_state = metadata.get("is_active")

        if actual_state is not expected_active:
            mismatches.append(
                f"{vector_id}: expected is_active={expected_active}, "
                f"found {actual_state}"
            )

    if mismatches:
        raise ValueError(
            "Pinecone active state mismatch: " + "; ".join(mismatches)
        )

    return {
        "document_id": document_id,
        "expected_active": expected_active,
        "actual_vectors": actual_vectors,
        "verified": True
    }

# ==========================================
# 4. Update One Vector
# ==========================================

def _update_vector_active(
    client: httpx.Client,
    base_url: str,
    headers: dict,
    vector_id: str,
    is_active: bool
):
    """
    Update the active status of one Pinecone vector.
    """

    payload = {
        "id": vector_id,
        "namespace": NAMESPACE,
        "setMetadata": {
            "is_active": is_active
        }
    }

    response = client.post(
        f"{base_url}/vectors/update",
        headers=headers,
        json=payload
    )

    response.raise_for_status()


# ==========================================
# 5. Set Document Active Status
# ==========================================

def set_document_active(
    document_id: str,
    is_active: bool,
    rollback_to: bool | None = None
):
    """
    Update is_active metadata for all vectors belonging to one document.

    If an update fails, attempt to restore all vectors whose update
    was attempted.

    rollback_to:
        Expected previous state of the document.

        Examples:
            Activate new version:
                is_active=True, rollback_to=False

            Deactivate old version:
                is_active=False, rollback_to=True

    This is a best-effort compensating rollback, not an atomic transaction.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    if not isinstance(is_active, bool):
        raise ValueError("is_active must be a boolean")

    if rollback_to is None:
        rollback_to = not is_active

    if not isinstance(rollback_to, bool):
        raise ValueError("rollback_to must be a boolean")

    if rollback_to == is_active:
        raise ValueError("rollback_to must differ from is_active")

    vector_ids = list_document_vector_ids(document_id)

    if not vector_ids:
        raise ValueError(
            f"No Pinecone vectors found for document: {document_id}"
        )

    base_url = get_base_url()
    headers = get_headers()

    attempted_ids = []

    with httpx.Client(timeout=30.0) as client:
        try:
            for vector_id in vector_ids:
                # Record before request because a timeout may happen after
                # Pinecone has processed the update.
                attempted_ids.append(vector_id)

                _update_vector_active(
                    client=client,
                    base_url=base_url,
                    headers=headers,
                    vector_id=vector_id,
                    is_active=is_active
                )

        except Exception as original_error:
            rollback_success = []
            rollback_failures = []

            # Attempt every rollback independently.
            for vector_id in attempted_ids:
                try:
                    _update_vector_active(
                        client=client,
                        base_url=base_url,
                        headers=headers,
                        vector_id=vector_id,
                        is_active=rollback_to
                    )

                    rollback_success.append(vector_id)

                except Exception as rollback_error:
                    rollback_failures.append({
                        "vector_id": vector_id,
                        "error": str(rollback_error)
                    })

            error_message = (
                f"Document activation update failed for {document_id}. "
                f"Attempted: {len(attempted_ids)}. "
                f"Rollback succeeded: {len(rollback_success)}. "
                f"Rollback failed: {len(rollback_failures)}. "
                f"Original error: {original_error}"
            )

            if rollback_failures:
                error_message += (
                    " Manual reconciliation required. "
                    f"Failed rollback details: {rollback_failures}"
                )

            raise RuntimeError(error_message) from original_error

    return {
        "status": "success",
        "document_id": document_id,
        "is_active": is_active,
        "total_updated": len(vector_ids)
    }


# ==========================================
# 6. Query Vectors
# ==========================================

def query_vectors(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str = None,
    logical_document_id: str = None,
    active_only: bool = True
):
    """
    Search policy chunks in Pinecone.

    By default, only active vectors are returned.
    """

    if len(query_embedding) != DIMENSION:
        raise ValueError(
            f"Expected {DIMENSION} dimensions, "
            f"received {len(query_embedding)}"
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    base_url = get_base_url()
    headers = get_headers()

    payload = {
        "vector": query_embedding,
        "topK": top_k,
        "includeMetadata": True,
        "namespace": NAMESPACE
    }

    metadata_filter = {}

    if active_only:
        metadata_filter["is_active"] = {"$eq": True}

    if document_id:
        metadata_filter["document_id"] = {"$eq": document_id}

    if logical_document_id:
        metadata_filter["logical_document_id"] = {
            "$eq": logical_document_id
        }

    if metadata_filter:
        payload["filter"] = metadata_filter

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{base_url}/query",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        result = response.json()

    return result.get("matches", [])


# ==========================================
# 7. Delete Document Vectors
# ==========================================

def delete_vectors(document_id: str):
    """
    Delete all vectors belonging to one document.
    """

    if not document_id:
        raise ValueError("Document ID is required")

    base_url = get_base_url()
    headers = get_headers()

    payload = {
        "filter": {
            "document_id": {
                "$eq": document_id
            }
        },
        "namespace": NAMESPACE
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{base_url}/vectors/delete",
            headers=headers,
            json=payload
        )

        response.raise_for_status()

    return {
        "status": "success",
        "document_id": document_id,
        "message": "Document vectors deleted successfully"
    }


# ==========================================
# 8. Verify Document Absence (Read-only)
# ==========================================

def verify_document_absence(document_id: str) -> dict:
    """
    Verify that no Pinecone vectors remain for a document.

    This function is read-only. It does not delete vectors and does not
    establish which request caused their absence. WordPress must only use
    this result to reconcile a previously persisted delete operation.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID is required")

    document_id = document_id.strip()
    vector_ids = list_document_vector_ids(document_id)
    remaining_vectors = len(vector_ids)
    verified_absence = remaining_vectors == 0

    return {
        "document_id": document_id,
        "remaining_vectors": remaining_vectors,
        "verified_absence": verified_absence,
        "verified": verified_absence,
        "vector_ids": vector_ids
    }
