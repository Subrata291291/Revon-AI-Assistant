
from embedding_service import generate_embedding
from vector_store import query_vectors
from groq_service import generate_response


def retrieve_context(question: str, top_k: int = 5):
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    # Step 1: Convert user question into an embedding
    query_embedding = generate_embedding(question)

    # Step 2: Search relevant chunks in Pinecone
    matches = query_vectors(
        query_embedding=query_embedding,
        top_k=top_k
    )

    # Step 3: Prepare context and source information
    context_parts = []
    sources = []

    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})

        text = metadata.get("text", "")
        filename = metadata.get("filename", "Unknown document")
        page_number = metadata.get("page_number", "Unknown")

        if not text.strip():
            continue

        context_parts.append(
            f"[Source {index}]\n"
            f"Filename: {filename}\n"
            f"Page: {page_number}\n"
            f"Content: {text}"
        )

        sources.append({
            "filename": filename,
            "page_number": page_number,
            "score": match.get("score")
        })

    return {
        "context": "\n\n".join(context_parts),
        "sources": sources,
        "total_matches": len(matches)
    }


def answer_policy_question(question: str, top_k: int = 5):

    retrieved = retrieve_context(question, top_k)

    if not retrieved["context"]:
        return {
            "answer": (
                "I could not find relevant information "
                "in the available policy documents."
            ),
            "sources": [],
            "total_matches": 0
        }

    system_prompt = """
You are RevOn's official policy assistant.

Follow these rules strictly:

1. Answer using only the provided policy context.
2. Never invent company policies, rules, dates, or benefits.
3. If the context does not contain sufficient information,
   clearly say that the available policy documents do not
   provide enough information.
4. Do not use external knowledge to fill missing information.
5. Cite supporting sources using their exact source labels,
   such as [Source 1] or [Source 2].
6. Never fabricate source references.
7. Be professional, clear, and concise.
"""

    user_prompt = f"""
Policy Context:

{retrieved["context"]}

User Question:

{question}

Provide an answer based only on the policy context.
Include source references in your answer.
"""

    answer = generate_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    return {
        "answer": answer,
        "sources": retrieved["sources"],
        "total_matches": retrieved["total_matches"]
    }
