import os
import secrets
import logging
import time
import threading
from collections import defaultdict, deque
from typing import Deque

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
    Depends,
    Query,
    Request,
)
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from ingestion_service import ingest_pdf
from vector_store import (
    set_document_active,
    verify_document_vector_count,
    verify_document_ownership,
    verify_document_active_state,
    get_document_vector_metadata,
    delete_vectors,
    list_document_vector_ids,
    verify_document_absence,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("revon-policy-chatbot")


# Public chat is accessed through the WordPress PHP proxy. The separate chat
# token is never sent to browser JavaScript and is not the admin document token.
CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "10"))
CHAT_RATE_WINDOW_SECONDS = 60
_chat_requests: dict[str, Deque[float]] = defaultdict(deque)
_chat_rate_lock = threading.Lock()


class PublicChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def verify_chat_token(
    x_revon_chat_token: str | None = Header(
        default=None,
        alias="X-RevOn-Chat-Token",
    ),
):
    """Authenticate the server-to-server WordPress chat proxy."""
    expected_token = os.getenv("REVON_CHAT_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Public chat service is not configured.",
        )

    if (
        not x_revon_chat_token
        or not secrets.compare_digest(x_revon_chat_token, expected_token)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing chat credential.",
        )


def enforce_chat_rate_limit(visitor_key: str) -> None:
    """Best-effort per-process limiter; enforce edge limits at WordPress/CDN too."""
    now = time.monotonic()
    cutoff = now - CHAT_RATE_WINDOW_SECONDS

    with _chat_rate_lock:
        # Opportunistically remove expired entries to keep memory bounded.
        for key in list(_chat_requests):
            queue = _chat_requests[key]
            while queue and queue[0] <= cutoff:
                queue.popleft()
            if not queue:
                _chat_requests.pop(key, None)

        queue = _chat_requests[visitor_key]
        while queue and queue[0] <= cutoff:
            queue.popleft()

        if len(queue) >= CHAT_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Too many chat requests. Please try again shortly.",
                headers={"Retry-After": str(CHAT_RATE_WINDOW_SECONDS)},
            )

        queue.append(now)

app = FastAPI(
    title="RevOn Policy AI Chatbot",
    description="AI-powered policy chatbot for RevOn",
    version="1.4.2",
)


def verify_admin_token(
    x_revon_admin_token: str | None = Header(
        default=None,
        alias="X-RevOn-Admin-Token",
    )
):
    expected_token = os.getenv("REVON_ADMIN_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Document service is not configured.",
        )

    if (
        not x_revon_admin_token
        or not secrets.compare_digest(x_revon_admin_token, expected_token)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token.",
        )


def validate_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required.",
        )
    return value.strip()


@app.get("/health")
def health_check():
    return {
        "status": "success",
        "message": "RevOn Policy AI Backend is running!",
    }


@app.get("/")
def home():
    return {"message": "Welcome to RevOn Policy AI Chatbot"}


@app.post("/api/v1/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    upload_type: str = Form(...),
    logical_document_id: str = Form(...),
    document_id: str = Form(...),
    request_id: str = Form(...),
    version: int = Form(...),
    _: None = Depends(verify_admin_token),
):
    """Ingest one policy version as inactive Pinecone vectors.

    WordPress is the source of truth for operation IDs and version numbers.
    Retrying the same operation must reuse request_id, document_id, and
    logical_document_id. This API does not itself maintain a durable ledger.
    """

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed.",
        )

    if upload_type not in ("new_policy", "new_version"):
        raise HTTPException(status_code=400, detail="Invalid upload_type.")

    logical_document_id = validate_identifier(
        logical_document_id, "logical_document_id"
    )
    document_id = validate_identifier(document_id, "document_id")
    request_id = validate_identifier(request_id, "request_id")

    if version < 1:
        raise HTTPException(
            status_code=400,
            detail="Version must be a positive integer.",
        )

    if upload_type == "new_policy" and version != 1:
        raise HTTPException(
            status_code=400,
            detail="A new policy must start at version 1.",
        )

    if upload_type == "new_version" and version < 2:
        raise HTTPException(
            status_code=400,
            detail="A new version must be at least version 2.",
        )

    file_bytes = await file.read(10 * 1024 * 1024 + 1)

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF size must not exceed 10 MB.",
        )

    try:
        return ingest_pdf(
            file_bytes=file_bytes,
            filename=file.filename,
            logical_document_id=logical_document_id,
            version=version,
            document_id=document_id,
            request_id=request_id,
        )

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        logger.exception(
            "Document ingestion failed. request_id=%s document_id=%s",
            request_id,
            document_id,
        )
        # Preserve vectors and the operation record for WordPress reconciliation.
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Document ingestion did not complete. "
                    "Check operation status before retrying or deleting."
                ),
                "request_id": request_id,
                "document_id": document_id,
                "reconciliation_required": True,
            },
        )


@app.get("/api/v1/documents/{document_id}/status")
def get_document_status(
    document_id: str,
    expected_vectors: int = Query(..., ge=1),
    request_id: str | None = Query(default=None),
    logical_document_id: str | None = Query(default=None),
    _: None = Depends(verify_admin_token),
):
    """Inspect Pinecone vector count and, when requested, vector ownership.

    This endpoint reads Pinecone state; it is not a durable operation ledger.
    WordPress must persist operation status in its MySQL table.
    """

    document_id = validate_identifier(document_id, "document_id")

    if request_id is not None:
        request_id = validate_identifier(request_id, "request_id")

    try:
        result = verify_document_vector_count(
            document_id=document_id,
            expected_vectors=expected_vectors,
        )

        if result["verified"] and (request_id or logical_document_id):
            if not logical_document_id:
                raise HTTPException(
                    status_code=400,
                    detail="logical_document_id is required with request_id.",
                )

            ownership = verify_document_ownership(
                document_id=document_id,
                logical_document_id=logical_document_id,
                request_id=request_id,
                expected_vectors=expected_vectors,
            )
            result["ownership_verified"] = ownership["verified"]

        result["status"] = (
            "indexed" if result["verified"] else "verification_pending"
        )
        result["request_id"] = request_id
        return result

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        logger.exception("Document status check failed for %s", document_id)
        raise HTTPException(
            status_code=502,
            detail="Unable to verify document state in Pinecone.",
        )


@app.post("/api/v1/documents/delete")
def delete_document(
    document_id: str = Form(...),
    logical_document_id: str = Form(...),
    version: int = Form(...),
    expected_vectors: int = Form(...),
    request_id: str = Form(...),
    _: None = Depends(verify_admin_token),
):
    """
    Delete one verified inactive document from Pinecone.

    WordPress remains responsible for its durable operation ledger,
    private PDF cleanup, and registry deletion.

    Active documents are never deleted through this endpoint.
    """

    document_id = validate_identifier(document_id, "document_id")
    logical_document_id = validate_identifier(
        logical_document_id, "logical_document_id"
    )
    request_id = validate_identifier(request_id, "request_id")

    if version < 1 or expected_vectors < 1:
        raise HTTPException(
            status_code=400,
            detail="version and expected_vectors must be positive.",
        )

    try:
        # Step 1: Verify exact vector count and ownership.
        ownership = verify_document_ownership(
            document_id=document_id,
            logical_document_id=logical_document_id,
            version=version,
            request_id=request_id,
            expected_vectors=expected_vectors,
        )

        if ownership.get("verified") is not True:
            raise HTTPException(
                status_code=409,
                detail="Document ownership verification failed.",
            )

        # Step 2: Fetch all vector metadata and reject active documents.
        metadata = get_document_vector_metadata(document_id)

        if len(metadata) != expected_vectors:
            raise HTTPException(
                status_code=409,
                detail="Vector count changed during deletion pre-check.",
            )

        active_vectors = [
            vector_id
            for vector_id, values in metadata.items()
            if values.get("is_active") is not False
        ]

        if active_vectors:
            raise HTTPException(
                status_code=409,
                detail="Active or unverified vectors cannot be deleted.",
            )

        # Step 3: Delete only this document's vectors.
        delete_vectors(document_id)

        # Step 4: Verify that no vectors remain.
        remaining_ids = list_document_vector_ids(document_id)

        if remaining_ids:
            logger.error(
                "Document deletion incomplete. document_id=%s remaining=%s",
                document_id,
                len(remaining_ids),
            )

            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Pinecone deletion could not be fully verified.",
                    "document_id": document_id,
                    "remaining_vectors": len(remaining_ids),
                    "reconciliation_required": True,
                },
            )

        return {
            "status": "success",
            "document_id": document_id,
            "logical_document_id": logical_document_id,
            "request_id": request_id,
            "deleted_vectors": expected_vectors,
            "remaining_vectors": 0,
            "verified": True,
            "message": "Inactive document vectors deleted and verified.",
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=409,
            detail=str(error),
        )

    except Exception:
        logger.exception(
            "Document deletion failed. request_id=%s document_id=%s",
            request_id,
            document_id,
        )

        raise HTTPException(
            status_code=502,
            detail={
                "message": "Document deletion failed. Reconciliation is required.",
                "document_id": document_id,
                "request_id": request_id,
                "reconciliation_required": True,
            },
        )
    

@app.post("/api/v1/documents/delete/verify-absence")
def verify_deleted_document_absence(
    document_id: str = Form(...),
    logical_document_id: str = Form(...),
    version: int = Form(...),
    expected_vectors: int = Form(...),
    request_id: str = Form(...),
    _: None = Depends(verify_admin_token),
):
    """
    Read-only reconciliation check after WordPress has a persisted delete
    operation with an ambiguous backend outcome.

    This endpoint does not perform deletion and cannot prove that this
    particular request caused the absence. WordPress must only call it when
    its durable operation ledger already contains a prior delete attempt.
    """
    document_id = validate_identifier(document_id, "document_id")
    logical_document_id = validate_identifier(
        logical_document_id, "logical_document_id"
    )
    request_id = validate_identifier(request_id, "request_id")

    if version < 1 or expected_vectors < 1:
        raise HTTPException(
            status_code=400,
            detail="version and expected_vectors must be positive.",
        )

    try:
        result = verify_document_absence(document_id)

        if result.get("verified_absence") is not True:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Document vectors are still present.",
                    "document_id": document_id,
                    "remaining_vectors": result["remaining_vectors"],
                    "reconciliation_required": True,
                },
            )

        return {
            "status": "success",
            "document_id": document_id,
            "logical_document_id": logical_document_id,
            "version": version,
            "request_id": request_id,
            "outcome": "already_absent",
            "deleted_vectors": 0,
            "remaining_vectors": 0,
            "verified": True,
            "verified_absence": True,
            "message": (
                "No vectors remain for this document. This read-only result "
                "must be reconciled against the prior WordPress delete operation."
            ),
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))

    except Exception:
        logger.exception(
            "Document absence verification failed. request_id=%s document_id=%s",
            request_id,
            document_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Unable to verify document absence in Pinecone.",
                "document_id": document_id,
                "request_id": request_id,
                "reconciliation_required": True,
            },
        )



@app.post("/api/v1/documents/deactivate")
def deactivate_document(
    document_id: str = Form(...),
    logical_document_id: str = Form(...),
    version: int = Form(...),
    expected_vectors: int = Form(...),
    request_id: str = Form(...),
    upload_request_id: str = Form(...),
    _: None = Depends(verify_admin_token),
):
    """Deactivate one owned document and verify every Pinecone vector.

    This operation does not delete vectors or files. WordPress owns the
    durable operation ledger and updates its registry only after this
    endpoint returns a fully verified inactive state.
    """
    document_id = validate_identifier(document_id, "document_id")
    logical_document_id = validate_identifier(
        logical_document_id, "logical_document_id"
    )
    request_id = validate_identifier(request_id, "request_id")
    upload_request_id = validate_identifier(upload_request_id, "upload_request_id")

    if version < 1 or expected_vectors < 1:
        raise HTTPException(
            status_code=400,
            detail="version and expected_vectors must be positive.",
        )

    try:
        ownership = verify_document_ownership(
            document_id=document_id,
            logical_document_id=logical_document_id,
            version=version,
            request_id=upload_request_id,
            expected_vectors=expected_vectors,
        )
        # Idempotent retry: a prior request may have completed in Pinecone
        # while its HTTP response was lost. Accept only a fully verified
        # inactive state for the same owned document.
        try:
            inactive_state = verify_document_active_state(
                document_id=document_id,
                expected_active=False,
                expected_vectors=expected_vectors,
            )
            already_inactive = True
            update_result = {"total_updated": 0}
        except ValueError:
            # Do not mutate a mixed/unknown state. Only a fully active
            # document may transition to inactive.
            verify_document_active_state(
                document_id=document_id,
                expected_active=True,
                expected_vectors=expected_vectors,
            )
            update_result = set_document_active(
                document_id=document_id,
                is_active=False,
                rollback_to=True,
            )
            inactive_state = verify_document_active_state(
                document_id=document_id,
                expected_active=False,
                expected_vectors=expected_vectors,
            )
            already_inactive = False

        return {
            "status": "success",
            "document_id": document_id,
            "logical_document_id": logical_document_id,
            "version": version,
            "request_id": request_id,
            "upload_request_id": upload_request_id,
            "is_active": False,
            "already_inactive": already_inactive,
            "total_updated": update_result["total_updated"],
            "verified": True,
            "verification": {
                "ownership": ownership,
                "inactive_state": inactive_state,
            },
            "message": "Document deactivated and verified successfully.",
        }
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except Exception as error:
        logger.exception(
            "Document deactivation failed. request_id=%s document_id=%s",
            request_id,
            document_id,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Document deactivation could not be verified.",
                "request_id": request_id,
                "document_id": document_id,
                "reconciliation_required": True,
            },
        ) from error


@app.post("/api/v1/documents/activate")
def activate_document_version(
    document_id: str = Form(...),
    logical_document_id: str = Form(...),
    version: int = Form(...),
    expected_vectors: int = Form(...),
    request_id: str = Form(...),
    previous_document_id: str | None = Form(default=None),
    _: None = Depends(verify_admin_token),
):
    """Activate a fully indexed document version and verify final vector state.

    WordPress remains responsible for the durable operation ledger and for
    coordinating its MySQL registry transaction with this Pinecone operation.
    Pinecone vector updates are not atomic; recovery is compensating best effort.
    """

    document_id = validate_identifier(document_id, "document_id")
    logical_document_id = validate_identifier(
        logical_document_id, "logical_document_id"
    )
    request_id = validate_identifier(request_id, "request_id")

    if version < 1 or expected_vectors < 1:
        raise HTTPException(
            status_code=400,
            detail="version and expected_vectors must be positive.",
        )

    if previous_document_id is not None:
        previous_document_id = validate_identifier(
            previous_document_id, "previous_document_id"
        )
        if previous_document_id == document_id:
            raise HTTPException(
                status_code=400,
                detail="New and previous document IDs must be different.",
            )

    # Verify ownership before changing any active flags.
    try:
        new_verification = verify_document_ownership(
            document_id=document_id,
            logical_document_id=logical_document_id,
            version=version,
            request_id=request_id,
            expected_vectors=expected_vectors,
        )

        previous_verification = None
        if previous_document_id:
            previous_verification = verify_document_ownership(
                document_id=previous_document_id,
                logical_document_id=logical_document_id,
            )

    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))

    new_activation_completed = False

    try:
        if previous_document_id:
            set_document_active(
                document_id=previous_document_id,
                is_active=False,
                rollback_to=True,
            )

        new_activation = set_document_active(
            document_id=document_id,
            is_active=True,
            rollback_to=False,
        )
        new_activation_completed = True

        # Verify the actual state visible in Pinecone after all updates.
        new_active_verification = verify_document_active_state(
            document_id=document_id,
            expected_active=True,
            expected_vectors=expected_vectors,
        )

        previous_active_verification = None
        if previous_document_id:
            previous_active_verification = verify_document_active_state(
                document_id=previous_document_id,
                expected_active=False,
                expected_vectors=previous_verification["actual_vectors"],
            )

        return {
            "status": "success",
            "document_id": document_id,
            "logical_document_id": logical_document_id,
            "version": version,
            "request_id": request_id,
            "previous_document_id": previous_document_id,
            "total_updated": new_activation["total_updated"],
            "verification": {
                "new_document_ownership": new_verification,
                "previous_document_ownership": previous_verification,
                "new_document_active_state": new_active_verification,
                "previous_document_active_state": previous_active_verification,
            },
            "message": "Document version activated and verified successfully.",
        }

    except Exception as activation_error:
        logger.exception(
            "Document activation failed. request_id=%s document_id=%s",
            request_id,
            document_id,
        )

        recovery_errors = []

        # If activation completed but final verification failed, deactivate
        # the new version before restoring the previous version.
        if new_activation_completed:
            try:
                set_document_active(
                    document_id=document_id,
                    is_active=False,
                    rollback_to=True,
                )
            except Exception as error:
                recovery_errors.append(
                    {"document_id": document_id, "error": str(error)}
                )

        # Restore the previous version independently, even if new-version
        # recovery failed.
        if previous_document_id:
            try:
                set_document_active(
                    document_id=previous_document_id,
                    is_active=True,
                    rollback_to=False,
                )
            except Exception as error:
                recovery_errors.append(
                    {"document_id": previous_document_id, "error": str(error)}
                )

        detail = {
            "message": "Document activation failed.",
            "request_id": request_id,
            "document_id": document_id,
            "reconciliation_required": bool(recovery_errors),
        }

        if recovery_errors:
            detail["recovery_errors"] = recovery_errors
            detail["message"] = (
                "Activation failed and one or more recovery actions failed. "
                "Manual vector reconciliation is required."
            )
        else:
            detail["message"] = (
                "Activation failed. Compensating recovery actions completed; "
                "verify the resulting state before retrying."
            )

        raise HTTPException(status_code=500, detail=detail) from activation_error


@app.post("/api/v1/chat")
def public_policy_chat(
    payload: PublicChatRequest,
    request: Request,
    _: None = Depends(verify_chat_token),
    x_revon_visitor_ip: str | None = Header(
        default=None,
        alias="X-RevOn-Visitor-IP",
    ),
):
    """Answer a public visitor using active policy documents only.

    The browser talks to WordPress. WordPress calls this endpoint server-side
    using REVON_CHAT_TOKEN and may pass the visitor IP for per-visitor limits.
    This endpoint never accepts document IDs, admin operations, or arbitrary
    retrieval filters from the public caller.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    # The visitor IP header is trusted only because this endpoint requires the
    # private WordPress-to-backend credential. Fall back to the direct peer.
    visitor_key = (x_revon_visitor_ip or "").strip()
    if not visitor_key and request.client:
        visitor_key = request.client.host
    if not visitor_key:
        visitor_key = "unknown"

    enforce_chat_rate_limit(visitor_key)

    try:
        # Import lazily so document-management endpoints remain independent
        # from chat provider initialization and tests can isolate the service.
        from retrieval_service import answer_policy_question

        result = answer_policy_question(question=question, top_k=5)
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("The policy answer service returned no answer.")

        raw_sources = result.get("sources", [])
        sources = []
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            sources.append({
                "filename": str(source.get("filename") or "Unknown document"),
                "page": source.get("page_number"),
                "score": source.get("score"),
            })

        return {
            "status": "success",
            "answer": answer.strip(),
            "sources": sources,
            "total_matches": int(result.get("total_matches", len(sources))),
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception:
        logger.exception("Public policy chat failed.")
        raise HTTPException(
            status_code=502,
            detail="The policy assistant is temporarily unavailable. Please try again.",
        )

