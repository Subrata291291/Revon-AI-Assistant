import uuid

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from embedding_service import generate_embeddings
from vector_store import upsert_vectors


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    logical_document_id: str,
    version: int,
    document_id: str = None,
    request_id: str = None
):
    """
    Version-aware, retry-friendly PDF ingestion pipeline.

    Steps:
    1. Validate input.
    2. Extract PDF text.
    3. Create chunks.
    4. Generate embeddings.
    5. Upsert vectors as inactive.

    document_id must remain stable when retrying the same upload.
    request_id identifies the WordPress-coordinated operation.

    Activation is deliberately handled separately, after WordPress
    successfully saves the document registry.

    IMPORTANT:
    This function deliberately does not automatically delete vectors
    after an upsert error. A network timeout can happen after Pinecone
    has accepted some or all vectors. Deleting them automatically could
    destroy data from an operation that should instead be reconciled.
    """

    if not file_bytes:
        raise ValueError("PDF file is empty.")

    if not filename or not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    if not isinstance(logical_document_id, str) or not logical_document_id.strip():
        raise ValueError("Logical document ID is required.")

    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError("Version must be a positive integer.")

    if document_id is None:
        document_id = str(uuid.uuid4())
    elif not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID must be a non-empty string.")

    if request_id is not None:
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("Request ID must be a non-empty string.")
        request_id = request_id.strip()

    logical_document_id = logical_document_id.strip()
    document_id = document_id.strip()

    try:
        # STEP 1: PDF Text Extraction
        print("\n[1/4] Extracting PDF text...")

        extracted_pages = extract_text_from_pdf(file_bytes)

        if not extracted_pages:
            raise ValueError("No readable text found in this PDF.")

        total_pages = len(extracted_pages)
        print("Pages extracted:", total_pages)

        # STEP 2: Text Chunking
        print("\n[2/4] Creating text chunks...")

        chunks = create_chunks(extracted_pages)

        if not chunks:
            raise ValueError("No text chunks generated.")

        total_chunks = len(chunks)
        print("Total chunks:", total_chunks)

        # STEP 3: Gemini Embeddings
        print("\n[3/4] Generating Gemini embeddings...")

        embedded_chunks = generate_embeddings(chunks)

        if len(embedded_chunks) != total_chunks:
            raise RuntimeError(
                "Embedding count does not match chunk count."
            )

        print("Embeddings generated:", len(embedded_chunks))

        # STEP 4: Pinecone Upsert
        print("\n[4/4] Uploading inactive vectors to Pinecone...")

        upsert_result = upsert_vectors(
            embedded_chunks=embedded_chunks,
            document_id=document_id,
            logical_document_id=logical_document_id,
            version=version,
            filename=filename,
            request_id=request_id,
            expected_vectors=total_chunks
        )

        if upsert_result.get("total_upserted") != total_chunks:
            raise RuntimeError(
                "Not all document chunks were uploaded. "
                f"Expected {total_chunks}, "
                f"received {upsert_result.get('total_upserted')}."
            )

        print("\nPDF indexing completed successfully.")
        print("Activation is pending WordPress registry save.")

        return {
            "status": "success",
            "document_id": document_id,
            "logical_document_id": logical_document_id,
            "request_id": request_id,
            "version": version,
            "filename": filename,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_vectors": upsert_result["total_upserted"],
            "expected_vectors": total_chunks,
            "is_active": False,
            "activation_status": "pending"
        }

    except ValueError:
        # Input/PDF validation errors are deterministic and safe to report.
        raise

    except Exception as error:
        # Do not delete vectors here. The upsert may have succeeded remotely
        # even if this process received a timeout or connection error.
        raise RuntimeError(
            "PDF ingestion did not complete successfully. "
            "Any vectors written under this document_id have been preserved "
            "for status verification or reconciliation. "
            f"document_id={document_id}; request_id={request_id or 'not-provided'}. "
            f"Cause: {error}"
        ) from error
