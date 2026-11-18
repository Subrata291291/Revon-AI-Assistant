import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

# Number of separate texts sent in one API call.
EMBEDDING_BATCH_SIZE = 20

# Maximum retries after the initial attempt.
MAX_RETRIES = 5

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=api_key)


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text.
    """

    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


def _generate_embedding_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate separate embeddings for multiple texts in one API call.

    Each text is wrapped in its own Content object because
    gemini-embedding-2 requires explicit separate content entries.
    """

    contents = [
        types.Content(
            parts=[types.Part(text=text)]
        )
        for text in texts
    ]

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=EMBEDDING_DIMENSION
                )
            )

            embeddings = response.embeddings or []

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Embedding count mismatch. "
                    f"Expected {len(texts)}, received {len(embeddings)}."
                )

            return [embedding.values for embedding in embeddings]

        except Exception as error:
            error_message = str(error).upper()

            is_rate_limit_error = (
                getattr(error, "code", None) == 429
                or "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            if not is_rate_limit_error or attempt >= MAX_RETRIES:
                raise

            delay = min(2 ** attempt, 60)

            print(
                f"Gemini rate limit reached. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"after {delay} seconds."
            )

            time.sleep(delay)

    raise RuntimeError("Embedding generation failed after retries.")


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for document chunks using batch requests.
    """

    embedded_chunks = []

    total_chunks = len(chunks)

    for start in range(0, total_chunks, EMBEDDING_BATCH_SIZE):

        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]

        batch_number = (start // EMBEDDING_BATCH_SIZE) + 1
        total_batches = (
            total_chunks + EMBEDDING_BATCH_SIZE - 1
        ) // EMBEDDING_BATCH_SIZE

        print(
            f"Generating embeddings: "
            f"Batch {batch_number}/{total_batches} "
            f"({len(batch)} chunks)"
        )

        texts = [chunk["text"] for chunk in batch]

        vectors = _generate_embedding_batch(texts)

        for chunk, vector in zip(batch, vectors, strict=True):

            embedded_chunks.append({
                "page_number": chunk["page_number"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "embedding": vector
            })

    if len(embedded_chunks) != total_chunks:
        raise RuntimeError(
            "Final embedding count mismatch. "
            f"Expected {total_chunks}, "
            f"received {len(embedded_chunks)}."
        )

    return embedded_chunks