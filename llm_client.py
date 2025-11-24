import asyncio
import logging
from typing import Any

import httpx

from config import Settings

logger = logging.getLogger(__name__)

async def query_model(
    model: str,
    messages: list[dict[str, str]],
    settings: Settings,
    timeout: float = 120.0
) -> dict[str, Any]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        settings: Application settings with API key
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # OpenRouter API expects 'model' in the body
    payload = {
        "model": model,
        "messages": messages,
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            settings.OPENROUTER_API_URL,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get('choices'):
            raise ValueError(f"No choices returned from model {model}: {data}")
                
        message = data['choices'][0]['message']
        return {
            'content': message.get('content'),
            'reasoning_details': message.get('reasoning_details')
        }

async def query_models_parallel(
    models: list[str],
    messages: list[dict[str, str]],
    settings: Settings
) -> dict[str, dict[str, Any]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenRouter model identifiers
        messages: List of message dicts to send to each model
        settings: Application settings with API key

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    # Create tasks for all models
    tasks = [query_model(model, messages, settings) for model in models]
    
    # Wait for all to complete
    responses = await asyncio.gather(*tasks)
    
    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}

