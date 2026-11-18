
from dotenv import load_dotenv
from langsmith import traceable

# Load environment variables from .env
load_dotenv()


@traceable(
    name="revon-langsmith-connection-test",
    run_type="chain"
)
def test_langsmith_connection():
    return {
        "status": "connected",
        "project": "revon-policy-chatbot",
        "message": "RevOn LangSmith tracing is working!"
    }


if __name__ == "__main__":
    result = test_langsmith_connection()
    print("LangSmith test result:", result)
