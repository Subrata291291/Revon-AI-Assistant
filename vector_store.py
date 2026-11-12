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
    filename: str
):
    """
    Store embedded document chunks in Pinecone.

    Each vector contains:
    - Unique vector ID
    - Embedding values
    - Document metadata
    """

    if not embedded_chunks:
        raise ValueError("No embedded chunks provided")

    base_url = get_base_url()
    headers = get_headers()

    total_upserted = 0

    with httpx.Client(timeout=60.0) as client:

        for start in range(0, len(embedded_chunks), BATCH_SIZE):

            batch = embedded_chunks[
                start:start + BATCH_SIZE
            ]

            vectors = []

            for chunk in batch:

                embedding = chunk["embedding"]

                if len(embedding) != DIMENSION:
                    raise ValueError(
                        f"Expected {DIMENSION} dimensions, "
                        f"received {len(embedding)}"
                    )

                page_number = chunk["page_number"]
                chunk_index = chunk["chunk_index"]

                vector_id = (
                    f"{document_id}-"
                    f"page-{page_number}-"
                    f"chunk-{chunk_index}"
                )

                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk["text"]
                }

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

            total_upserted += result.get(
                "upsertedCount", 0
            )

            print(
                f"Upserted batch: "
                f"{total_upserted} vectors"
            )

    return {
        "status": "success",
        "document_id": document_id,
        "total_upserted": total_upserted
    }


# ==========================================
# 3. Query Vectors
# ==========================================

def query_vectors(
    query_embedding: list[float],
    top_k: int = 5,
    document_id: str = None
):
    """
    Search relevant document chunks using
    a query embedding.
    """

    if len(query_embedding) != DIMENSION:
        raise ValueError(
            f"Expected {DIMENSION} dimensions, "
            f"received {len(query_embedding)}"
        )

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    base_url = get_base_url()
    headers = get_headers()

    payload = {
        "vector": query_embedding,
        "topK": top_k,
        "includeMetadata": True,
        "namespace": NAMESPACE
    }

    if document_id:
        payload["filter"] = {
            "document_id": {
                "$eq": document_id
            }
        }

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
# 4. Delete Document Vectors
# ==========================================

def delete_vectors(document_id: str):
    """
    Delete all vectors belonging to a document.
    """

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
