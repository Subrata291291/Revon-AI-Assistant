
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing")

client = Groq(
    api_key=GROQ_API_KEY,
    timeout=60.0,
    max_retries=2
)


def generate_response(
    system_prompt: str,
    user_prompt: str
) -> str:

    if not system_prompt.strip():
        raise ValueError("System prompt cannot be empty")

    if not user_prompt.strip():
        raise ValueError("User prompt cannot be empty")

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1500,
        reasoning_effort="medium"
    )

    response = completion.choices[0].message.content

    if not response:
        raise RuntimeError("Groq returned an empty response")

    return response.strip()
