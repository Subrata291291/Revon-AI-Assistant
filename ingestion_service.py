import uuid

from pdf_processor import extract_text_from_pdf
from chunker import create_chunks
from embedding_service import generate_embeddings
from vector_store import upsert_vectors, delete_vectors


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    document_id: str = None
):
    """
    Complete PDF ingestion pipeline.

    Steps:
    1. Extract text from PDF.
    2. Split text into chunks.
    3. Generate Gemini embeddings.
    4. Store vectors in Pinecone.

    If indexing fails, attempt to remove
    partially uploaded vectors.
    """

    if not file_bytes:
        raise ValueError("PDF file is empty.")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    # Generate a unique ID for this ingestion
    if document_id is None:
        document_id = str(uuid.uuid4())

    indexing_started = False

    try:

        # ==================================
        # STEP 1: PDF Text Extraction
        # ==================================

        print("\n[1/4] Extracting PDF text...")

        extracted_pages = extract_text_from_pdf(
            file_bytes
        )

        if not extracted_pages:
            raise ValueError(
                "No readable text found in this PDF."
            )

        total_pages = len(extracted_pages)

        print("Pages extracted:", total_pages)


        # ==================================
        # STEP 2: Text Chunking
        # ==================================

        print("\n[2/4] Creating text chunks...")

        chunks = create_chunks(extracted_pages)

        if not chunks:
            raise ValueError(
                "No text chunks generated."
            )

        total_chunks = len(chunks)

        print("Total chunks:", total_chunks)


        # ==================================
        # STEP 3: Gemini Embeddings
        # ==================================

        print("\n[3/4] Generating Gemini embeddings...")

        embedded_chunks = generate_embeddings(
            chunks
        )

        if len(embedded_chunks) != total_chunks:
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        print(
            "Embeddings generated:",
            len(embedded_chunks)
        )


        # ==================================
        # STEP 4: Pinecone Upsert
        # ==================================

        print("\n[4/4] Uploading vectors to Pinecone...")

        indexing_started = True

        upsert_result = upsert_vectors(
            embedded_chunks=embedded_chunks,
            document_id=document_id,
            filename=filename
        )

        if upsert_result["total_upserted"] != total_chunks:
            raise RuntimeError(
                "Not all document chunks were uploaded."
            )

        print("\nPDF ingestion completed successfully!")

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "total_vectors": upsert_result["total_upserted"]
        }

    except Exception as error:

        # Remove partially uploaded vectors
        if indexing_started:

            print(
                "\nIngestion failed. "
                "Attempting vector cleanup..."
            )

            try:
                delete_vectors(document_id)

                print("Partial vectors cleaned up.")

            except Exception as cleanup_error:

                print(
                    "WARNING: Vector cleanup failed:",
                    str(cleanup_error)
                )

        raise RuntimeError(
            f"PDF ingestion failed: {str(error)}"
        ) from error
