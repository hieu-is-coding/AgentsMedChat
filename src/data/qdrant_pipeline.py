"""
Qdrant Data Pipeline Module

This module handles the embedding and vector storage pipeline for medical documents.
It integrates with Google's Gemini embedding model and Qdrant vector database.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.pydantic_v1 import root_validator, SecretStr
from langchain_core.utils import get_from_dict_or_env
from langchain_google_genai._common import get_client_info
from langchain_google_genai._genai_extension import build_generative_service

logger = logging.getLogger(__name__)


class SafeGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    A wrapper around GoogleGenerativeAIEmbeddings that handles SecretStr correctly.
    This fixes a bug where SecretStr is passed to the google-auth library which expects str.
    """
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates params and passes them to google-generativeai package."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        
        # FIX: Convert SecretStr to string
        if isinstance(google_api_key, SecretStr):
            google_api_key = google_api_key.get_secret_value()
            
        client_info = get_client_info("GoogleGenerativeAIEmbeddings")

        values["client"] = build_generative_service(
            credentials=values.get("credentials"),
            api_key=google_api_key,
            client_info=client_info,
            client_options=values.get("client_options"),
        )
        return values

    def embed_documents(self, texts: List[str], *args, **kwargs) -> List[List[float]]:
        results = super().embed_documents(texts, *args, **kwargs)
        return [list(res) for res in results]

    def embed_query(self, text: str, *args, **kwargs) -> List[float]:
        result = super().embed_query(text, *args, **kwargs)
        return list(result)



class QdrantPipeline:
    """
    Manages the Qdrant vector database pipeline for medical document storage and retrieval.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        collection_name: str = "rag-opensource",
        embedding_dimension: int = 3072,
    ):
        """
        Initialize the Qdrant pipeline.

        Args:
            qdrant_url: URL of the Qdrant server
            qdrant_api_key: API key for Qdrant Cloud (optional)
            google_api_key: Google API key for embeddings
            collection_name: Name of the Qdrant collection
            embedding_dimension: Dimension of embeddings (default: 1536 for Gemini)
        """
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        # Initialize embeddings
        self.embeddings = SafeGoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.google_api_key,
        )

        # Initialize Qdrant client
        self.client = self._init_qdrant_client()

        # Initialize vector store
        self.vector_store = self._init_vector_store()

        logger.info(f"Qdrant pipeline initialized with collection: {collection_name}")

    def _init_qdrant_client(self) -> QdrantClient:
        """Initialize Qdrant client with proper configuration."""
        try:
            if self.qdrant_api_key:
                # Qdrant Cloud
                client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    prefer_grpc=True,
                )
            else:
                # Local or on-premise
                client = QdrantClient(url=self.qdrant_url, prefer_grpc=True)

            logger.info(f"Connected to Qdrant at {self.qdrant_url}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _init_vector_store(self) -> QdrantVectorStore:
        """Initialize or get existing Qdrant vector store."""
        try:
            # Try to get existing collection
            collection_info = self.client.get_collection(self.collection_name)
            
            # Check for dimension mismatch
            current_size = collection_info.config.params.vectors.size
            if current_size != self.embedding_dimension:
                logger.warning(
                    f"Collection {self.collection_name} has dimension {current_size}, "
                    f"but config specifies {self.embedding_dimension}. Recreating collection."
                )
                self.client.delete_collection(self.collection_name)
                raise ValueError("Collection deleted due to dimension mismatch")
                
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            # Create new collection if it doesn't exist or was deleted
            logger.info(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        return vector_store

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 10,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            batch_size: Number of documents to process at once

        Returns:
            List of document IDs
        """
        try:
            logger.info(f"Adding {len(documents)} documents to vector store")

            # Add documents in batches
            doc_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                ids = self.vector_store.add_documents(documents=batch)
                doc_ids.extend(ids)
                logger.info(f"Added batch {i // batch_size + 1} ({len(ids)} documents)")

            logger.info(f"Successfully added {len(doc_ids)} documents")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search on the vector store.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of (Document, similarity_score) tuples
        """
        try:
            logger.info(f"Searching for: {query}")

            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
            )

            # Filter by threshold if provided
            if score_threshold:
                results = [
                    (doc, score)
                    for doc, score in results
                    if score >= score_threshold
                ]
                logger.info(
                    f"Found {len(results)} results above threshold {score_threshold}"
                )
            else:
                logger.info(f"Found {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise

    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.

        Args:
            doc_ids: List of document IDs to delete

        Returns:
            True if successful
        """
        try:
            self.vector_store.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise

    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._init_vector_store()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise
