import logging
import os
import uuid
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

from src.medchat import MedChat
from config_template import (
    GOOGLE_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    GEMINI_MODEL,
    MEDICAL_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    LOG_LEVEL,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global MedChat instance
medchat_instance: Optional[MedChat] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI to handle startup and shutdown events.
    """
    global medchat_instance
    try:
        logger.info("Initializing MedChat API...")
        medchat_instance = MedChat(
            google_api_key=GOOGLE_API_KEY,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            gemini_model=GEMINI_MODEL,
            collection_name=MEDICAL_COLLECTION_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
        )
        # Perform health check on startup
        health = medchat_instance.health_check()
        logger.info(f"Startup Health Check: {health}")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize MedChat: {e}")
        raise
    finally:
        logger.info("Shutting down MedChat API...")
        # Cleanup if necessary

app = FastAPI(
    title="MedChat API",
    description="API for the Multi-Agent Medical Chatbot",
    version="1.0.0",
    lifespan=lifespan
)

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")

class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    agent_type: str
    retrieved_documents: List[SourceDocument] = []
    search_results: List[SearchResult] = []
    thinking_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, bool]

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {
        "message": "Welcome to MedChat API",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the system components."""
    if not medchat_instance:
        raise HTTPException(status_code=503, detail="MedChat system not initialized")
    
    health = medchat_instance.health_check()
    status = "healthy" if all(health.values()) else "degraded"
    return HealthResponse(status=status, components=health)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request."""
    if not medchat_instance:
        raise HTTPException(status_code=503, detail="MedChat system not initialized")

    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        result = medchat_instance.process_query(request.query, session_id=session_id)
        
        # Transform internal result to API response
        retrieved_docs = []
        if "retrieved_documents" in result:
            for doc in result["retrieved_documents"]:
                retrieved_docs.append(SourceDocument(
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0)
                ))
        
        search_results = []
        if "search_results" in result:
            for res in result["search_results"]:
                search_results.append(SearchResult(
                    title=res.get("title", ""),
                    link=res.get("link", ""),
                    snippet=res.get("snippet", "")
                ))

        return ChatResponse(
            answer=result.get("answer", ""),
            session_id=session_id,
            agent_type=result.get("agent_type", "unknown"),
            retrieved_documents=retrieved_docs,
            search_results=search_results,
            thinking_time=result.get("thinking_time")
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if not medchat_instance:
        raise HTTPException(status_code=503, detail="MedChat system not initialized")
        
    medchat_instance.clear_conversation_history(session_id)
    return {"message": "History cleared", "session_id": session_id}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
