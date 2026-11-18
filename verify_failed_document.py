
from vector_store import list_document_vector_ids

DOCUMENT_ID = "6b61bea0-7fc5-47c7-bab1-0dccc02d3490"

print("\nChecking Pinecone document vectors...")
print("Document ID:", DOCUMENT_ID)

vector_ids = list_document_vector_ids(DOCUMENT_ID)

print("\n========== VERIFICATION RESULT ==========")
print("Namespace: revon-policies")
print("Document ID:", DOCUMENT_ID)
print("Total vectors found:", len(vector_ids))

if not vector_ids:
    print("\nRESULT: NO VECTORS FOUND")
else:
    print("\nRESULT: VECTORS FOUND")
    print("\nVector IDs:")
    for vector_id in vector_ids:
        print("-", vector_id)

print("\n=========================================")
