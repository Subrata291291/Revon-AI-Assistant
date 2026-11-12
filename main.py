import os
import secrets
import logging

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
    Depends,
    Query,
)
from dotenv import load_dotenv

load_dotenv()

from ingestion_service import ingest_pdf
from vector_store import (
    set_document_active,
    verify_document_vector_count,
    verify_document_ownership,
    verify_document_active_state,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("revon-policy-chatbot")

app = FastAPI(
    title="RevOn Policy AI Chatbot",
    description="AI-powered policy chatbot for RevOn",
    version="1.3.1",
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
