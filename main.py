
import os
import secrets

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Header,
    Depends
)
from dotenv import load_dotenv
load_dotenv()

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from ingestion_service import ingest_pdf


app = FastAPI(
    title="RevOn Policy AI Chatbot",
    description="AI-powered policy chatbot for RevOn",
    version="1.0.0"
)


def verify_admin_token(
    x_revon_admin_token: str | None = Header(
        default=None,
        alias="X-RevOn-Admin-Token"
    )
):
    expected_token = os.getenv("REVON_ADMIN_TOKEN")

    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Document upload service is not configured."
        )

    if not x_revon_admin_token or not secrets.compare_digest(
        x_revon_admin_token,
        expected_token
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication token."
        )


@app.get("/health")
def health_check():
    return {
        "status": "success",
        "message": "RevOn Policy AI Backend is running!"
    }


@app.get("/")
def home():
    return {
        "message": "Welcome to RevOn Policy AI Chatbot"
    }


@app.post("/api/v1/documents/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    _: None = Depends(verify_admin_token)
):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="PDF size must not exceed 10 MB."
        )

    try:
        result = ingest_pdf(
            file_bytes=file_bytes,
            filename=file.filename
        )

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed."
        )
