from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENROUTER_API_KEY: str = Field(..., description="API Key for OpenRouter")
    
    # Hardcoded API URL
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"

    # Council members - list of OpenRouter model identifiers
    COUNCIL_MODELS: List[str] = [
        "openai/gpt-5.1",
        "google/gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.5",
        "x-ai/grok-4",
    ]

    # Chairman model - synthesizes final response
    CHAIRMAN_MODEL: str = "google/gemini-3-pro-preview"
