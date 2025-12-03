from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(provider: str, model: str, api_key: str, temperature: float = 0.2):
    provider = provider.lower()
    if provider == "openai":
        if not api_key:
            raise ValueError("OpenAI API key missing")
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq API key missing")
        return ChatGroq(api_key=api_key, model=model, temperature=temperature)
    elif provider in ("gemini", "google"):
        if not api_key:
            raise ValueError("Google API key missing")
        return ChatGoogleGenerativeAI(
            api_key=api_key, model=model, temperature=temperature
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
