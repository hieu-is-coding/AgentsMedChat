"""
Configuration template for MedChat application.
Copy this file to config.py and fill in your actual values.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Gemini API Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Qdrant Configuration
QDRANT_URL = os.getenv("SERVICE_URL_QDRANT")
QDRANT_API_KEY = os.getenv("SERVICE_PASSWORD_QDRANTAPIKEY")

# Gemini Model Configuration
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSION = 1536

# Application Configuration
APP_NAME = "MedChat"
DEBUG = False
LOG_LEVEL = "INFO"

# Qdrant Collection Configuration
MEDICAL_COLLECTION_NAME = "MedChat-RAG"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Agent Configuration
MAX_AGENT_ITERATIONS = 10
AGENT_TIMEOUT = 300  # seconds

# RAG Configuration
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.5

# Validation
def validate_config():
    """Validate that all required configuration values are set."""
    required_keys = ["GOOGLE_API_KEY"]
    missing_keys = [key for key in required_keys if not globals().get(key)]
    
    if missing_keys:
        raise ValueError(f"Missing required configuration: {', '.join(missing_keys)}")
    
    return True
