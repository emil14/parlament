from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENROUTER_API_KEY: str = Field(..., description="API Key for OpenRouter")
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    COUNCIL_MODELS: List[str] = [
        "openai/gpt-5.1",
        "google/gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.5",
        "x-ai/grok-4",
    ]
    CHAIRMAN_MODEL: str = "google/gemini-3-pro-preview"
