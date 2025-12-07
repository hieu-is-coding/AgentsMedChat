"""
MedChat Main Application Module

This module coordinates all agents and provides the main interface
for the multi-agent chatbot system.
"""

import logging
import os
import time
from typing import Dict, Optional, Generator
from src.agents.orchestration_agent import OrchestrationAgent, AgentType
from src.agents.rag_agent import RAGAgent
from src.agents.search_agent import SearchAgent
from src.agents.report_agent import ReportAgent
from src.data.qdrant_pipeline import QdrantPipeline
from src.memory.supabase_memory import SupabaseMemory

logger = logging.getLogger(__name__)


class MedChat:
    """
    Main MedChat application that coordinates all agents.
    """

    def __init__(
        self,
        google_api_key: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        gemini_model: str = "gemini-2.0-flash",
        collection_name: str = "MedChat-RAG",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize MedChat application.

        Args:
            google_api_key: Google API key for Gemini and Search
            qdrant_url: URL of Qdrant server
            qdrant_api_key: API key for Qdrant Cloud
            gemini_model: Gemini model to use
            collection_name: Name of Qdrant collection
            embedding_dimension: Dimension of embeddings
        """
        self.google_api_key = google_api_key
        self.qdrant_url = qdrant_url or os.getenv("SERVICE_URL_QDRANT")
        self.qdrant_api_key = qdrant_api_key or os.getenv("SERVICE_PASSWORD_QDRANTAPIKEY")
        self.gemini_model = gemini_model
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension

        logger.info("Initializing MedChat application...")

        # Initialize Qdrant pipeline
        try:
            self.qdrant_pipeline = QdrantPipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=collection_name,
                embedding_dimension=embedding_dimension,
            )
            logger.info("Qdrant pipeline initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant pipeline: {e}")
            raise

        # Initialize Supabase Memory
        try:
            self.supabase_memory = SupabaseMemory()
            logger.info("Supabase memory initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Supabase memory: {e}")
            self.supabase_memory = None

        # Initialize agents
        try:
            self.orchestration_agent = OrchestrationAgent(
                google_api_key=google_api_key,
                supabase_memory=self.supabase_memory,
                model_name=gemini_model,
            )
            logger.info("Orchestration agent initialized")

            self.rag_agent = RAGAgent(
                qdrant_pipeline=self.qdrant_pipeline,
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

    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a user query through the multi-agent system.

        Args:
            query: User query
            session_id: Session ID for memory

        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            logger.info(f"Processing query: {query}")

            # Step 1: Route query using orchestration agent
            routing_info = self.orchestration_agent.process_query(query, session_id=session_id)
            agent_type = AgentType(routing_info["agent_type"])
            refined_query = routing_info["query_refinement"]
            is_medical = routing_info.get("is_medical", True)
            direct_response = routing_info.get("direct_response")

            logger.info(f"Routed to agent: {agent_type.value}")

            # Handle non-medical or direct response cases
            if direct_response:
                logger.info("Returning direct response from orchestration agent")
                end_time = time.time()
                thinking_time = end_time - start_time
                self.orchestration_agent.add_to_history(
                    role="assistant",
                    content=direct_response,
                    agent_type="orchestration",
                    session_id=session_id,
                    thinking_time=thinking_time,
                )
                return {
                    "question": query,
                    "answer": direct_response,
                    "routing_info": routing_info,
                    "agent_type": "orchestration"
                }

            if not is_medical:
                # Fallback if no direct response provided but marked as non-medical
                msg = "I am a medical assistant and cannot answer non-medical questions. Please ask about health topics."
                end_time = time.time()
                thinking_time = end_time - start_time
                self.orchestration_agent.add_to_history(
                    role="assistant",
                    content=msg,
                    agent_type="orchestration",
                    session_id=session_id,
                    thinking_time=thinking_time,
                )
                return {
                    "question": query,
                    "answer": msg,
                    "routing_info": routing_info,
                    "agent_type": "orchestration"
                }

            # Step 2: Execute appropriate workflow
            if agent_type == AgentType.GENERAL:
                # For general queries, use simple RAG or direct answer
                result = self.rag_agent.answer_question(question=refined_query)
                
            else:
                # Smart Workflow for Medical Queries (RAG -> Sufficiency -> Search -> Report)
                logger.info("Executing Smart Medical Workflow")
                
                # 1. RAG Retrieval
                rag_result = self.rag_agent.answer_question(question=refined_query)
                
                # 2. Sufficiency Check
                sufficiency = None
                # sufficiency = self.orchestration_agent.check_sufficiency(
                #     query=refined_query,
                #     context=rag_result.get("context_used", "")
                # )
                
                search_result = None
                # if not sufficiency.is_sufficient:
                #     logger.info(f"RAG insufficient: {sufficiency.reasoning}. Performing search.")
                #     # 3. Search Fallback
                #     search_result = self.search_agent.answer_question(question=refined_query)
                # else:
                #     logger.info("RAG sufficient. Skipping search.")



                # 4. Final Answer Generation (Report or Short Answer)
                if routing_info.get("requires_report", False):
                    logger.info("Generating comprehensive report")
                    report = self.report_agent.generate_summary_report(
                        query=query,
                        rag_results=rag_result,
                        search_results=search_result,
                    )
                else:
                    logger.info("Generating short answer with citations")
                    report = self.report_agent.generate_short_answer(
                        query=query,
                        rag_results=rag_result,
                        search_results=search_result,
                    )
                
                result = {
                    "question": query,
                    "answer": report,
                    "retrieved_documents": rag_result.get("retrieved_documents", []),
                    "search_results": search_result.get("search_results", []) if search_result else [],
                    "sufficiency_check": sufficiency.dict() if sufficiency else None
                }

            # Step 3: Add metadata
            result["routing_info"] = routing_info
            result["agent_type"] = agent_type.value

            # Add to conversation history
            end_time = time.time()
            thinking_time = end_time - start_time
            
            self.orchestration_agent.add_to_history(
                role="assistant",
                content=result.get("answer", "")[:200],
                agent_type=agent_type.value,
                session_id=session_id,
                thinking_time=thinking_time,
            )
            
            result["thinking_time"] = thinking_time

            logger.info("Query processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def stream_query(self, query: str, session_id: Optional[str] = None) -> Generator:
        """
        Stream response for a query (for real-time UI updates).

        Args:
            query: User query
            session_id: Session ID

        Yields:
            Chunks of the response
        """
        try:
            logger.info(f"Streaming query: {query}")

            # Route query
            routing_info = self.orchestration_agent.process_query(query, session_id=session_id)
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

    def clear_conversation_history(self, session_id: Optional[str] = None) -> None:
        """Clear the conversation history."""
        self.orchestration_agent.clear_history(session_id=session_id)
        logger.info("Conversation history cleared")

    def get_all_sessions(self) -> list[str]:
        """Retrieve all available session IDs."""
        if self.supabase_memory:
            return self.supabase_memory.get_all_sessions()
        return []

    def get_session_history(self, session_id: str) -> list[dict]:
        """Retrieve history for a specific session."""
        if self.supabase_memory:
            return self.supabase_memory.get_history(session_id=session_id)
        return []

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
            "supabase_memory": False,
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
        
        # Check Supabase
        health_status["supabase_memory"] = self.supabase_memory is not None

        return health_status
