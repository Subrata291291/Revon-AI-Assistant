
import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
NAMESPACE = "revon-policies"

api_key = os.getenv("PINECONE_API_KEY")

if not api_key or not INDEX_NAME:
    raise RuntimeError("Pinecone configuration missing")

pc = Pinecone(api_key=api_key)
index = pc.Index(INDEX_NAME)

# Verify current data before deletion
stats = index.describe_index_stats()
namespace_stats = stats.namespaces.get(NAMESPACE)

count = namespace_stats.vector_count if namespace_stats else 0

print("Index:", INDEX_NAME)
print("Namespace:", NAMESPACE)
print("Vectors to delete:", count)

if count == 0:
    print("Namespace is already empty. Nothing to delete.")
    raise SystemExit(0)

confirmation = input(
    f"\nType DELETE {NAMESPACE} to permanently delete these {count} vectors: "
)

if confirmation != f"DELETE {NAMESPACE}":
    print("Deletion cancelled.")
    raise SystemExit(0)

# Delete only the target namespace
index.delete(delete_all=True, namespace=NAMESPACE)

print("\nDeletion request submitted.")

# Verify after deletion
updated_stats = index.describe_index_stats()
updated_namespace = updated_stats.namespaces.get(NAMESPACE)

remaining = updated_namespace.vector_count if updated_namespace else 0

print("Remaining vectors:", remaining)

if remaining == 0:
    print("SUCCESS: Policy namespace is empty.")
else:
    print("WARNING: Some vectors may still remain. Verify again.")
