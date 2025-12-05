"""
Qdrant Data Pipeline Module

This module handles the retrieval pipeline for medical documents using an existing Qdrant collection.
It integrates with an embedding model and Qdrant vector database.
"""

import os
import logging
from typing import List, Tuple, Optional, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

class CustomGeminiEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Custom wrapper for Gemini embeddings to support 1536 dimensions
    by concatenating the 768-dimensional vector with itself.
    """
    def embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeddings = super().embed_documents(texts, **kwargs)
        # Only pad if dimension is 768
        return [list(emb) + list(emb) if len(emb) == 768 else emb for emb in embeddings]

    def embed_query(self, text: str, **kwargs) -> List[float]:
        embedding = super().embed_query(text, **kwargs)
        # Only pad if dimension is 768
        if len(embedding) == 768:
            return list(embedding) + list(embedding)
        return embedding

class QdrantPipeline:
    """
    Manages the Qdrant vector database pipeline for medical document retrieval.
    Connects to an existing collection 'MedChat-RAG'.
    """

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "MedChat-RAG",
        embedding_model: Optional[Embeddings] = None,
        embedding_dimension: int = 1536,
        vector_name: str = "dense",
    ):
        """
        Initialize the Qdrant pipeline.

        Args:
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: API key for Qdrant Cloud
            collection_name: Name of the Qdrant collection (default: "MedChat-RAG")
            embedding_model: LangChain Embeddings interface. 
                             IMPORTANT: Must match the dimension of the collection (1536).
            embedding_dimension: Dimension of embeddings (default: 1536)
            vector_name: Name of the vector in the collection (default: "dense")
        """
        # Load from env if not provided
        self.qdrant_url = qdrant_url or os.getenv("SERVICE_URL_QDRANT")
        self.qdrant_api_key = qdrant_api_key or os.getenv("SERVICE_PASSWORD_QDRANTAPIKEY")
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.vector_name = vector_name

        # Initialize embeddings
        if embedding_model:
            self.embeddings = embedding_model
        else:
            # Fallback to Gemini with custom padding to 1536
            logger.info("Initializing CustomGeminiEmbeddings (768 -> 1536 padding)")
            self.embeddings = CustomGeminiEmbeddings(
                model="models/embedding-001", 
                # google_api_key=os.getenv("GOOGLE_API_KEY"), # Rely on env var
            )

        # Initialize Qdrant client
        self.client = self._init_qdrant_client()

        logger.info(f"Qdrant pipeline initialized for collection: {collection_name}")

    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client with proper configuration."""
        try:
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                port=443,
                prefer_grpc=False,
                timeout=60,
            )
            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[models.Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform semantic search using dense vectors.

        Args:
            query: The search query string
            k: Number of results to return
            filter: Optional Qdrant filter
            score_threshold: Optional minimum score threshold

        Returns:
            List of (Document, score) tuples
        """
        try:
            logger.info(f"Performing search for: {query}")
            
            # 1. Generate Embedding
            query_vector = self.embeddings.embed_query(query)
            
            # Check dimension
            if len(query_vector) != self.embedding_dimension:
                logger.warning(
                    f"Query vector dimension ({len(query_vector)}) does not match "
                    f"configured dimension ({self.embedding_dimension}). Search may fail."
                )

            # 2. Execute Search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=models.NamedVector(
                    name=self.vector_name,
                    vector=query_vector
                ),
                query_filter=filter,
                limit=k,
                score_threshold=score_threshold,
                with_payload=True,
            )
            
            # 3. Format Results
            formatted_results = []
            for point in results:
                # Map specific payload fields to Document
                payload = point.payload or {}
                
                # Extract main content
                page_content = payload.get("text", "")
                
                # Extract metadata
                metadata = {
                    "book_name": payload.get("book_name"),
                    "author": payload.get("author"),
                    "publish_year": payload.get("publish_year"),
                    "page_number": payload.get("page_number"),
                    "pdf_id": payload.get("pdf_id"),
                    "keywords": payload.get("keywords"),
                    "language": payload.get("language"),
                    # Keep original payload just in case
                    "_original_payload": payload 
                }
                
                doc = Document(
                    page_content=page_content,
                    metadata=metadata
                )
                formatted_results.append((doc, point.score))
                
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def get_collection_info(self) -> dict:
        """Get information about the current collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
                "config": str(collection_info.config.params)
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

