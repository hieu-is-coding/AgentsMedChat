"""
RAG Agent Module

This module implements the Retrieval-Augmented Generation (RAG) agent
that retrieves relevant medical documents and generates responses.
"""

import logging
from typing import List, Tuple, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    RAG Agent that retrieves documents from Qdrant and generates responses
    augmented with retrieved context.
    """

    def __init__(
        self,
        qdrant_pipeline,
        google_api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        top_k: int = 5,
    ):
        """
        Initialize the RAG Agent.

        Args:
            qdrant_pipeline: QdrantPipeline instance
            google_api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
            top_k: Number of documents to retrieve
        """
        self.qdrant_pipeline = qdrant_pipeline
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            # google_api_key=google_api_key, # Rely on env var
        )

        # Define the RAG prompt template
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a medical assistant helping medical students. 
            
Based on the following retrieved medical documents, answer the user's question comprehensively and accurately.

Retrieved Documents:
{context}

User Question: {question}

Please provide a detailed answer based on the retrieved documents. If the documents don't contain relevant information, 
indicate that and provide general medical knowledge if appropriate. Always cite the sources of your information."""
        )

        # Create the RAG chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"RAG Agent initialized with model: {model_name}")

    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents from the vector store using dense search.

        Args:
            query: User query
            k: Number of documents to retrieve (uses default if not specified)

        Returns:
            List of (Document, similarity_score) tuples
        """
        k = k or self.top_k

        try:
            logger.info(f"Retrieving {k} documents for query: {query}")
            # Use the new search method (formerly hybrid_search)
            results = self.qdrant_pipeline.search(query=query, k=k)
            logger.info(f"Retrieved {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def format_context(
        self,
        retrieved_docs: List[Tuple[Document, float]],
    ) -> str:
        """
        Format retrieved documents into context string with rich metadata.

        Args:
            retrieved_docs: List of (Document, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, (doc, score) in enumerate(retrieved_docs, 1):
            # Extract metadata
            meta = doc.metadata
            book = meta.get("book_name", "Unknown Book")
            author = meta.get("author", "Unknown Author")
            year = meta.get("publish_year", "N/A")
            page = meta.get("page_number", "N/A")
            
            content = doc.page_content
            
            # Create a citation header
            citation = f"{book} ({year}), by {author}, p. {page}"

            context_parts.append(
                f"Document {i} (Relevance: {score:.4f}):\n"
                f"Source: {citation}\n"
                f"Content: {content}\n"
            )

        return "\n\n".join(context_parts)

    def answer_question(
        self,
        question: str,
        k: Optional[int] = None,
    ) -> dict:
        """
        Answer a question using RAG.

        Args:
            question: User question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer and retrieved documents
        """
        try:
            logger.info(f"Processing question: {question}")

            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, k)

            # Format context
            context = self.format_context(retrieved_docs)

            # Generate answer
            answer = self.chain.invoke({
                "context": context,
                "question": question,
            })

            logger.info("Answer generated successfully")

            return {
                "question": question,
                "answer": answer,
                "retrieved_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                    for doc, score in retrieved_docs
                ],
                "context_used": context,
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    def stream_answer(
        self,
        question: str,
        k: Optional[int] = None,
    ):
        """
        Stream answer for a question (for real-time UI updates).

        Args:
            question: User question
            k: Number of documents to retrieve

        Yields:
            Chunks of the answer
        """
        try:
            logger.info(f"Streaming answer for: {question}")

            # Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, k)

            # Format context
            context = self.format_context(retrieved_docs)

            # Stream the answer
            for chunk in self.chain.stream({
                "context": context,
                "question": question,
            }):
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            raise

