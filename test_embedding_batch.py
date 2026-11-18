
from embedding_service import generate_embeddings

chunks = [
    {
        "page_number": (i // 3) + 1,
        "chunk_index": i,
        "text": (
            f"This is a synthetic test document chunk number {i}. "
            "It contains sample business policy information for testing."
        )
    }
    for i in range(139)
]

result = generate_embeddings(chunks)

print("\n========== BATCH TEST RESULT ==========")
print("Input chunks:", len(chunks))
print("Output embeddings:", len(result))
print("Vector dimension:", len(result[0]["embedding"]))
print("First chunk index:", result[0]["chunk_index"])
print("Last chunk index:", result[-1]["chunk_index"])
print(
    "All metadata preserved:",
    all(
        result[i]["chunk_index"] == chunks[i]["chunk_index"]
        for i in range(len(chunks))
    )
)
