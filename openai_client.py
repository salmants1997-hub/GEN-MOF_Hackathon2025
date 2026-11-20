import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from openai import OpenAI


load_dotenv(Path(__file__).with_name(".env"))
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")


def call_llm(messages, model: str = DEFAULT_MODEL) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key and os.getenv("OPENAI_ENV_FILE"):
        key = dotenv_values(Path(os.environ["OPENAI_ENV_FILE"]).expanduser()).get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Configure OPENAI_API_KEY or OPENAI_ENV_FILE for GenMOF.")
    with OpenAI(api_key=key) as client:
        response = client.responses.create(model=model, input=messages, store=False)
    if response.status != "completed" or not response.output_text:
        raise RuntimeError(f"OpenAI returned no complete text response (status {response.status}).")
    return response.output_text
