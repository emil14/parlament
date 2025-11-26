"""Parliament - LLM Council System.

Can run as a test script or start the API server.
"""

import asyncio
import logging
import sys

import uvicorn

from llm_client import query_model
from config import Settings

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_council():
    """Test the council system directly."""
    logger.info("Starting parliament test...")
    
    # Initialize settings
    settings = Settings()  # pyright: ignore[reportCallIssue]
    
    response = await query_model("openai/gpt-4o-mini", [{"role": "user", "content": "Hello!"}], settings)
    logger.info(f"Response: {response['content']}")
    
    logger.info("Parliament test completed.")


def run_server():
    """Start the FastAPI server."""
    logger.info("Starting Parliament Council API server...")
    logger.info("Server will be available at http://localhost:8000")
    logger.info("API endpoint: http://localhost:8000/v1/chat/completions")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    else:
        asyncio.run(test_council())
