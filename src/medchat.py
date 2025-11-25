"""
MedChat Main Application Module

This module coordinates all agents and provides the main interface
for the multi-agent chatbot system.
"""

import logging
import os
from typing import Dict, Optional, Generator
from src.agents.orchestration_agent import OrchestrationAgent, AgentType
from src.agents.rag_agent import RAGAgent
from src.agents.search_agent import SearchAgent
from src.agents.report_agent import ReportAgent
from src.data.qdrant_pipeline import QdrantPipeline

logger = logging.getLogger(__name__)


class MedChat:
    """
    Main MedChat application that coordinates all agents.
    """

    def __init__(
        self,
        google_api_key: str,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        collection_name: str = "rag-opensource",
        embedding_dimension: int = 3072,
    ):
        """
        Initialize MedChat application.

        Args:
            google_api_key: Google API key for Gemini and Search
            qdrant_url: URL of Qdrant server
            qdrant_api_key: API key for Qdrant Cloud (optional)
            gemini_model: Gemini model to use
            collection_name: Name of Qdrant collection
            embedding_dimension: Dimension of embeddings
        """
        self.google_api_key = google_api_key
        self.qdrant_url = qdrant_url
        self.qdrant_api_key = qdrant_api_key
        self.gemini_model = gemini_model
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        logger.info("Initializing MedChat application...")

        # Initialize Qdrant pipeline
        try:
            self.qdrant_pipeline = QdrantPipeline(
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                google_api_key=google_api_key,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
            )
            logger.info("Qdrant pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant pipeline: {e}")
            raise

        # Initialize agents
        try:
            self.orchestration_agent = OrchestrationAgent(
                google_api_key=google_api_key,
                model_name=gemini_model,
            )
            logger.info("Orchestration agent initialized")

            self.rag_agent = RAGAgent(
                vector_store=self.qdrant_pipeline.vector_store,
                google_api_key=google_api_key,
                model_name=gemini_model,
            )
            logger.info("RAG agent initialized")

            self.search_agent = SearchAgent(
                google_api_key=google_api_key,
                model_name=gemini_model,
            )
            logger.info("Search agent initialized")

            self.report_agent = ReportAgent(
                google_api_key=google_api_key,
                model_name=gemini_model,
            )
            logger.info("Report agent initialized")

        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

        logger.info("MedChat application initialized successfully")

    def process_query(self, query: str) -> Dict:
        """
        Process a user query through the multi-agent system.

        Args:
            query: User query

        Returns:
            Dictionary with response and metadata
        """
        try:
            logger.info(f"Processing query: {query}")

            # Step 1: Route query using orchestration agent
            routing_info = self.orchestration_agent.process_query(query)
            agent_type = AgentType(routing_info["agent_type"])

            logger.info(f"Routed to agent: {agent_type.value}")

            # Step 2: Execute appropriate agent
            if agent_type == AgentType.RAG:
                result = self.rag_agent.answer_question(
                    question=routing_info["query_refinement"]
                )

            elif agent_type == AgentType.SEARCH:
                result = self.search_agent.answer_question(
                    question=routing_info["query_refinement"]
                )

            elif agent_type == AgentType.REPORT:
                # For reports, we might want to combine RAG and search
                rag_result = self.rag_agent.answer_question(
                    question=routing_info["query_refinement"]
                )
                search_result = self.search_agent.answer_question(
                    question=routing_info["query_refinement"]
                )

                report = self.report_agent.generate_summary_report(
                    query=query,
                    rag_results=rag_result,
                    search_results=search_result,
                )

                result = {
                    "question": query,
                    "answer": report,
                    "agent_type": "report",
                    "rag_results": rag_result,
                    "search_results": search_result,
                }

            else:  # GENERAL
                # Use RAG as default for general queries
                result = self.rag_agent.answer_question(
                    question=routing_info["query_refinement"]
                )

            # Step 3: Add metadata
            result["routing_info"] = routing_info
            result["agent_type"] = agent_type.value

            # Add to conversation history
            self.orchestration_agent.add_to_history(
                role="assistant",
                content=result.get("answer", "")[:200],
                agent_type=agent_type.value,
            )

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def stream_query(self, query: str) -> Generator:
        """
        Stream response for a query (for real-time UI updates).

        Args:
            query: User query

        Yields:
            Chunks of the response
        """
        try:
            logger.info(f"Streaming query: {query}")

            # Route query
            routing_info = self.orchestration_agent.process_query(query)
            agent_type = AgentType(routing_info["agent_type"])

            # Stream from appropriate agent
            if agent_type == AgentType.RAG:
                for chunk in self.rag_agent.stream_answer(
                    question=routing_info["query_refinement"]
                ):
                    yield chunk

            elif agent_type == AgentType.SEARCH:
                for chunk in self.search_agent.stream_answer(
                    question=routing_info["query_refinement"]
                ):
                    yield chunk

            elif agent_type == AgentType.REPORT:
                # Generate report and stream
                rag_result = self.rag_agent.answer_question(
                    question=routing_info["query_refinement"]
                )
                search_result = self.search_agent.answer_question(
                    question=routing_info["query_refinement"]
                )

                for chunk in self.report_agent.stream_report(
                    topic=query,
                    information=f"{rag_result.get('answer', '')}\n\n{search_result.get('answer', '')}",
                ):
                    yield chunk

            else:  # GENERAL
                for chunk in self.rag_agent.stream_answer(
                    question=routing_info["query_refinement"]
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise

    def get_vector_store_info(self) -> Dict:
        """Get information about the vector store."""
        try:
            return self.qdrant_pipeline.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            raise

    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.orchestration_agent.clear_history()
        logger.info("Conversation history cleared")

    def health_check(self) -> Dict:
        """
        Perform a health check on all components.

        Returns:
            Dictionary with health status of each component
        """
        health_status = {
            "qdrant": False,
            "orchestration_agent": False,
            "rag_agent": False,
            "search_agent": False,
            "report_agent": False,
        }

        try:
            # Check Qdrant
            self.qdrant_pipeline.get_collection_info()
            health_status["qdrant"] = True
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")

        # Check agents (they should be initialized)
        health_status["orchestration_agent"] = self.orchestration_agent is not None
        health_status["rag_agent"] = self.rag_agent is not None
        health_status["search_agent"] = self.search_agent is not None
        health_status["report_agent"] = self.report_agent is not None

        return health_status
