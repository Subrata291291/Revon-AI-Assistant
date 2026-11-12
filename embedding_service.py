import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIMENSION = 768

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=api_key)


def generate_embedding(text: str) -> list[float]:
    """
    Generate a vector embedding for a single text.
    """

    response = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )

    return response.embeddings[0].values


def generate_embeddings(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for multiple document chunks.
    """

    embedded_chunks = []

    for chunk in chunks:
        vector = generate_embedding(chunk["text"])

        embedded_chunks.append({
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "embedding": vector
        })

    return embedded_chunks