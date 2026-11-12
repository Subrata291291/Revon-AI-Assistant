import os
import time
import httpx

from dotenv import load_dotenv

load_dotenv()


# ==========================================
# 1. Pinecone Configuration
# ==========================================

API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = "revon-policy-index"

DIMENSION = 768
METRIC = "cosine"

BASE_URL = "https://api.pinecone.io"

HEADERS = {
    "Api-Key": API_KEY or "",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-Pinecone-Api-Version": "2025-10",
}


# ==========================================
# 2. Check Existing Index
# ==========================================

def get_index_details(client):

    response = client.get(
        f"{BASE_URL}/indexes/{INDEX_NAME}",
        headers=HEADERS
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()

    return response.json()


# ==========================================
# 3. Create Pinecone Index
# ==========================================

def create_index(client):

    print("\nChecking Pinecone index...")

    existing_index = get_index_details(client)

    if existing_index:

        print("Index already exists!")

        if existing_index["dimension"] != DIMENSION:
            raise ValueError(
                f"Dimension mismatch! "
                f"Expected {DIMENSION}, "
                f"Found {existing_index['dimension']}"
            )

        if existing_index["metric"] != METRIC:
            raise ValueError(
                f"Metric mismatch! "
                f"Expected {METRIC}, "
                f"Found {existing_index['metric']}"
            )

        print("Dimension verified:", DIMENSION)
        print("Metric verified:", METRIC)

        return existing_index

    print("Index not found. Creating a new index...")

    payload = {
        "name": INDEX_NAME,
        "vector_type": "dense",
        "dimension": DIMENSION,
        "metric": METRIC,
        "spec": {
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    }

    response = client.post(
        f"{BASE_URL}/indexes",
        headers=HEADERS,
        json=payload
    )

    response.raise_for_status()

    result = response.json()

    print("\nIndex creation request successful!")
    print("Name:", result["name"])
    print("Dimension:", result["dimension"])
    print("Metric:", result["metric"])
    print("Initial state:", result["status"]["state"])

    return result


# ==========================================
# 4. Wait Until Index Is Ready
# ==========================================

def wait_for_index(client):

    print("\nWaiting for Pinecone index to become Ready...")

    max_attempts = 30

    for attempt in range(1, max_attempts + 1):

        index_data = get_index_details(client)

        if index_data is None:
            raise RuntimeError("Pinecone index was not found.")

        state = index_data["status"]["state"]

        print(
            f"Attempt {attempt}/{max_attempts} "
            f"- Status: {state}"
        )

        if state == "Ready":

            print("\nPinecone index is READY!")

            print("Name:", index_data["name"])
            print("Dimension:", index_data["dimension"])
            print("Metric:", index_data["metric"])
            print("Host:", index_data["host"])

            return index_data

        time.sleep(5)

    raise TimeoutError(
        "Index did not become Ready within the allowed time."
    )


# ==========================================
# 5. Main Function
# ==========================================

def main():

    if not API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is missing from .env"
        )

    with httpx.Client(timeout=30.0) as client:

        create_index(client)

        index_data = wait_for_index(client)

        print("\n================================")
        print("PINECONE CONFIGURATION COMPLETE")
        print("================================")

        print("Index:", index_data["name"])
        print("Dimension:", index_data["dimension"])
        print("Metric:", index_data["metric"])
        print("Host:", index_data["host"])


if __name__ == "__main__":
    main()
