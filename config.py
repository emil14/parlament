from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    OPENROUTER_API_KEY: str = Field(..., description="API Key for OpenRouter")
    OPENROUTER_API_URL: str = Field("https://openrouter.ai/api/v1/chat/completions", description="OpenRouter API URL")

settings = Settings()
