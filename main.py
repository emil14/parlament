import asyncio
import logging

from llm_client import query_model
from config import Settings

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting parliament...")
    
    # Initialize settings
    settings = Settings()  # pyright: ignore[reportCallIssue]
    
    response = await query_model("openai/gpt-4o-mini", [{"role": "user", "content": "Hello!"}], settings)
    logger.info(f"Response: {response['content']}")
    
    logger.info("Parliament initialized.")

if __name__ == "__main__":
    asyncio.run(main())
