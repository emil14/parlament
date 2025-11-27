"""Parliament - LLM Council System.

Can run as a test script or start the API server.
"""

import logging
import uvicorn

# Configure logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_server():
    """Start the FastAPI server."""
    logger.info("Starting Parliament Council API server...")
    logger.info("Server will be available at http://localhost:8000")
    logger.info("API endpoint: http://localhost:8000/v1/chat/completions")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_server()
