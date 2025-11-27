"""FastAPI server exposing council as OpenAI-compatible API."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import Settings
from council import run_full_council

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Parliament Council API", version="0.1.0")

# Initialize settings at startup
settings = Settings()  # pyright: ignore[reportCallIssue]


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="parliament/council", description="Model identifier")
    messages: list[ChatMessage] = Field(..., description="List of chat messages")
    temperature: float = Field(default=1.0, description="Temperature (not used by council)")
    max_tokens: int | None = Field(default=None, description="Max tokens (not used by council)")
    stream: bool = Field(default=False, description="Streaming not supported")


class Choice(BaseModel):
    """OpenAI-compatible choice object."""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """OpenAI-compatible usage object."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = "chatcmpl-council"
    object: str = "chat.completion"
    created: int = 0
    model: str = "parliament/council"
    choices: list[Choice]
    usage: Usage


def extract_user_query(messages: list[ChatMessage]) -> str:
    """
    Extract the user query from messages.
    
    For simplicity, we concatenate all user messages.
    In a production system, you might want to handle conversation history differently.
    """
    user_messages = [msg.content for msg in messages if msg.role == "user"]
    if not user_messages:
        raise ValueError("No user messages found in request")
    
    # Combine all user messages
    return "\n\n".join(user_messages)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    OpenAI-compatible chat completions endpoint.
    
    This endpoint accepts standard OpenAI format requests and routes them
    through the council system, returning the Stage 3 synthesized result.
    """
    try:
        logger.info(f"Received chat completion request with {len(request.messages)} messages")
        
        # Extract user query from messages
        user_query = extract_user_query(request.messages)
        
        logger.info("Running council process...")
        
        # Run the full council process
        council_result = await run_full_council(user_query, settings)
        
        # Extract the final synthesized response from Stage 3
        final_response = council_result.stage3_result.response
        
        # Use real accumulated usage from all stages
        total_usage = council_result.total_usage
        
        logger.info(
            f"Council process completed successfully. "
            f"Total tokens: {total_usage['total_tokens']} "
            f"(prompt: {total_usage['prompt_tokens']}, completion: {total_usage['completion_tokens']})"
        )
        
        # Build OpenAI-compatible response
        response_message = ChatMessage(
            role="assistant",
            content=final_response
        )
        
        choice = Choice(
            index=0,
            message=response_message,
            finish_reason="stop"
        )
        
        usage = Usage(
            prompt_tokens=total_usage['prompt_tokens'],
            completion_tokens=total_usage['completion_tokens'],
            total_tokens=total_usage['total_tokens']
        )
        
        return ChatCompletionResponse(
            id="chatcmpl-council",
            object="chat.completion",
            created=0,
            model=request.model,
            choices=[choice],
            usage=usage
        )
        
    except Exception as e:
        logger.error(f"Error processing chat completion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": "parliament/council",
                "object": "model",
                "created": 0,
                "owned_by": "parliament"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

