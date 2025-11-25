"""
Orchestration Agent Module

This module implements the Orchestration Agent that coordinates between
different specialized agents (RAG, Search, Report) based on user queries.
"""

import logging
import json
from typing import Dict, Optional, List
from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    """Enumeration of available agent types."""

    RAG = "rag"
    SEARCH = "search"
    REPORT = "report"
    GENERAL = "general"


class AgentDecision(BaseModel):
    """Model for agent decision output."""

    agent_type: AgentType = Field(
        description="Which agent should handle this query"
    )
    reasoning: str = Field(description="Reasoning for the agent selection")
    requires_report: bool = Field(
        default=False,
        description="Whether a formal report should be generated",
    )
    query_refinement: str = Field(
        description="Refined version of the query for the selected agent"
    )


class OrchestrationAgent:
    """
    Orchestration Agent that routes queries to appropriate specialized agents.
    """

    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5,
    ):
        """
        Initialize the Orchestration Agent.

        Args:
            google_api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
        )

        # Define the routing prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an intelligent query router for a medical chatbot system.

Your job is to analyze user queries and decide which agent should handle them:
- RAG Agent: For questions about medical knowledge base, diseases, treatments, anatomy
- Search Agent: For current medical news, recent research, latest guidelines
- Report Agent: For generating formal medical reports or comprehensive summaries
- General: For general conversation or clarifications

Analyze the following query and determine:
1. Which agent is best suited to answer it
2. Your reasoning for the selection
3. Whether a formal report should be generated
4. A refined version of the query for the selected agent

Query: {query}

Respond with a JSON object containing: agent_type, reasoning, requires_report, and query_refinement."""
        )

        # Create the parser
        self.parser = JsonOutputParser(pydantic_object=AgentDecision)

        # Create the routing chain
        self.chain = (
            self.prompt_template
            | self.llm
            | self.parser
        )

        # Conversation history for context
        self.conversation_history: List[Dict] = []

        logger.info(f"Orchestration Agent initialized with model: {model_name}")

    def decide_agent(self, query: str) -> AgentDecision:
        """
        Decide which agent should handle the query.

        Args:
            query: User query

        Returns:
            AgentDecision object with routing information
        """
        try:
            logger.info(f"Routing query: {query}")

            # Get routing decision
            decision = self.chain.invoke({"query": query})

            logger.info(f"Routed to agent: {decision.agent_type}")
            logger.info(f"Reasoning: {decision.reasoning}")

            return decision

        except Exception as e:
            logger.error(f"Error in agent routing: {e}")
            # Default to RAG agent on error
            return AgentDecision(
                agent_type=AgentType.RAG,
                reasoning="Error in routing, defaulting to RAG agent",
                requires_report=False,
                query_refinement=query,
            )

    def add_to_history(
        self,
        role: str,
        content: str,
        agent_type: Optional[str] = None,
    ) -> None:
        """
        Add message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
            agent_type: Which agent handled this (if assistant)
        """
        self.conversation_history.append({
            "role": role,
            "content": content,
            "agent_type": agent_type,
        })

    def get_conversation_context(self, last_n: int = 5) -> str:
        """
        Get recent conversation history as context.

        Args:
            last_n: Number of recent messages to include

        Returns:
            Formatted conversation context
        """
        recent = self.conversation_history[-last_n:]
        context_parts = []

        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"][:200]  # Limit content
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def process_query(self, query: str) -> Dict:
        """
        Process a query and return routing information.

        Args:
            query: User query

        Returns:
            Dictionary with routing information
        """
        try:
            # Get agent decision
            decision = self.decide_agent(query)

            # Add to history
            self.add_to_history("user", query)

            return {
                "agent_type": decision.agent_type.value,
                "reasoning": decision.reasoning,
                "requires_report": decision.requires_report,
                "query_refinement": decision.query_refinement,
                "conversation_context": self.get_conversation_context(),
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def handle_multi_turn_conversation(
        self,
        query: str,
        previous_context: Optional[str] = None,
    ) -> Dict:
        """
        Handle multi-turn conversations with context awareness.

        Args:
            query: Current user query
            previous_context: Context from previous turns

        Returns:
            Routing information with conversation context
        """
        try:
            # Enhance query with previous context if available
            enhanced_query = query
            if previous_context:
                enhanced_query = f"Previous context: {previous_context}\n\nCurrent query: {query}"

            return self.process_query(enhanced_query)

        except Exception as e:
            logger.error(f"Error handling multi-turn conversation: {e}")
            raise
