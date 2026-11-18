
import os
from pinecone import Pinecone

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise RuntimeError("PINECONE_API_KEY is not loaded")

pc = Pinecone(api_key=api_key)

index = pc.Index("revon-policy-index")

stats = index.describe_index_stats()

print("\nPINECONE INDEX:", "revon-policy-index")
print("\nINDEX STATISTICS:")
print(stats)

namespace = "revon-policies"
namespace_stats = stats.namespaces.get(namespace)

print(f"\nTARGET NAMESPACE: {namespace}")

if namespace_stats:
    print("VECTOR COUNT:", namespace_stats.vector_count)
else:
    print("Namespace not found or empty.")
