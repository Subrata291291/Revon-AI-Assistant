import uuid

from chunker import create_chunks
from embedding_service import generate_embeddings, iter_embeddings
from pdf_processor import extract_text_from_pdf
from vector_store import list_document_vector_ids, upsert_vectors


def _expected_vector_id(document_id: str, chunk: dict) -> str:
    return (
        f"{document_id}-"
        f"page-{chunk['page_number']}-"
        f"chunk-{chunk['chunk_index']}"
    )


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    logical_document_id: str,
    version: int,
    document_id: str = None,
    request_id: str = None,
):
    """
    Version-aware, retry-friendly PDF ingestion pipeline.

    Steps:
    1. Validate input.
    2. Extract PDF text.
    3. Create chunks deterministically.
    4. Detect existing deterministic vector IDs for this document (resume support).
    5. Generate embeddings only for missing chunks (rate-limited, retry-aware batches).
    6. Upsert embedded batches incrementally as inactive vectors (idempotent deterministic IDs).

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

        # STEP 3: Resume detection (safe/idempotent). Do not delete or reset existing vectors.
        print("\n[3/4] Generating Gemini embeddings (with resume-safe incremental upserts)...")
        existing_ids = set()
        try:
            existing_ids = set(list_document_vector_ids(document_id))
        except Exception:
            # If listing fails, proceed with full attempt; upserts remain deterministic/idempotent.
            existing_ids = set()

        if existing_ids:
            print(f"Existing vectors detected for document_id={document_id}: {len(existing_ids)}")

        missing_chunks = [
            chunk for chunk in chunks
            if _expected_vector_id(document_id, chunk) not in existing_ids
        ]
        already_present = total_chunks - len(missing_chunks)
        if already_present > 0:
            print(f"Resuming ingestion: {already_present}/{total_chunks} vectors already present; {len(missing_chunks)} remaining.")

        newly_upserted = 0
        if missing_chunks:
            for embedded_batch in iter_embeddings(missing_chunks):
                if not embedded_batch:
                    continue
                upsert_result = upsert_vectors(
                    embedded_chunks=embedded_batch,
                    document_id=document_id,
                    logical_document_id=logical_document_id,
                    version=version,
                    filename=filename,
                    request_id=request_id,
                    expected_vectors=len(embedded_batch),
                )
                batch_upserted = int(upsert_result.get("total_upserted") or 0)
                if batch_upserted != len(embedded_batch):
                    raise RuntimeError(
                        "Not all embedded chunks were uploaded for a batch. "
                        f"Expected {len(embedded_batch)}, received {batch_upserted}."
                    )
                newly_upserted += batch_upserted
        else:
            print("All expected vectors already present for this document_id; skipping embedding/upsert.")

        # STEP 4: Final verification view (best-effort count for response)
        print("\n[4/4] Finalizing ingestion result...")
        progress_count = min(total_chunks, already_present + newly_upserted)
        final_existing_count = progress_count
        try:
            # Re-list for an up-to-date count after incremental upserts/resume.
            relisted = set(list_document_vector_ids(document_id))
            final_existing_count = max(len(relisted), progress_count)
        except Exception:
            final_existing_count = progress_count

        if newly_upserted == 0 and final_existing_count < total_chunks:
            # No progress made and state is incomplete; report as not completed.
            raise RuntimeError(
                "Ingestion incomplete: no new vectors were uploaded and the expected vector set is not fully present. "
                f"Present={final_existing_count}, expected={total_chunks}."
            )

        print("\nPDF indexing completed successfully (or reconciled).")
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
            "total_vectors": final_existing_count,
            "expected_vectors": total_chunks,
            "vectors_already_present": already_present,
            "vectors_newly_upserted": newly_upserted,
            "is_active": False,
            "activation_status": "pending",
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

