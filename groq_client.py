import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DEFAULT_MODEL = "llama-3.3-70b-versatile"


def call_llm(messages, model: str = DEFAULT_MODEL, temperature: float = 0.2) -> str:
    """
    Simple wrapper around the Groq chat completion API.

    messages: list of {"role": "...", "content": "..."}
    """
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content
