from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from ingestion_service import ingest_pdf

app = FastAPI(
    title="RevOn Policy AI Chatbot",
    description="AI-powered policy chatbot for RevOn",
    version="1.0.0"
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
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
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

    # Basic upload size limit: 10 MB
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

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(error)}"
        )
