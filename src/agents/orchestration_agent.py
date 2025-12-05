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
    is_medical: bool = Field(
        description="Whether the query is related to medical/health topics"
    )
    direct_response: Optional[str] = Field(
        description="Direct response for non-medical queries or clarification requests"
    )


class SufficiencyCheck(BaseModel):
    """Model for sufficiency check output."""

    is_sufficient: bool = Field(
        description="Whether the provided context is sufficient to answer the query"
    )
    reasoning: str = Field(description="Reasoning for the sufficiency decision")
    missing_information: Optional[str] = Field(
        description="What information is missing, if any"
    )
    confidence_score: float = Field(
        description="Confidence score (0.0-1.0) based on context relevance"
    )


class OrchestrationAgent:
    """
    Orchestration Agent that routes queries to appropriate specialized agents.
    """

    def __init__(
        self,
        google_api_key: str,
        supabase_memory = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5,
    ):
        """
        Initialize the Orchestration Agent.

        Args:
            google_api_key: Google API key for Gemini
            supabase_memory: SupabaseMemory instance
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
        """
        self.google_api_key = google_api_key
        self.supabase_memory = supabase_memory
        self.model_name = model_name
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            # google_api_key=google_api_key, # Rely on env var to avoid SecretStr issue
        )

        # Define the routing prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an intelligent query router for a medical chatbot system.

Your job is to analyze user queries and decide how to handle them.

1. **Input Classification**:
   - **Medical/Health Questions**: Proceed to agent selection.
   - **Non-Medical Questions**: Do NOT use any tools. Politely decline (e.g., "I am a medical assistant and cannot answer non-medical questions. Please ask about health topics.").
   - **Self-Description**: If asked about yourself, provide a short description.

2. **Question Analysis (For Medical Questions)**:
   - **Vague Topic** (e.g., "Tell me about diabetes"): Expand the query to include core topics (definition, causes, symptoms, general management).
   - **Vague Dosage/Calculation** (e.g., "How to calculate dosage?"): Provide general principles and common clinical examples. Do NOT ask for clarification.
   - **Missing/Unclear Info**: If the question is truly unintelligible, ask for clarification in `direct_response`.

3. **Agent Selection & Output Format**:
   - **RAG Agent**: For specific questions about medical knowledge base, diseases, treatments, anatomy. (Output: Short Answer)
   - **Search Agent**: For current medical news, recent research, latest guidelines. (Output: Short Answer)
   - **Report Agent**: ONLY if the user explicitly asks for a "report", "summary", "comprehensive overview", or "detailed analysis". (Output: Long Report)
   - **General**: For general conversation or clarifications.

Analyze the following query and determine:
1. Is it medical?
2. Which agent is best suited?
3. Reasoning?
4. Requires report? (True ONLY if user asks for report/summary/comprehensive overview)
5. Refined query (expanded if vague)?
6. Direct response (if non-medical or clarification needed)?

Query: {query}

Respond with a JSON object containing: 
- agent_type (string enum: rag, search, report, general)
- reasoning (string)
- requires_report (boolean)
- query_refinement (string)
- is_medical (boolean)
- direct_response (string or null)"""
        )

        # Create the parser
        self.parser = JsonOutputParser(pydantic_object=AgentDecision)

        # Create the routing chain
        self.chain = (
            self.prompt_template
            | self.llm
            | self.parser
        )

        # Define sufficiency check prompt
        self.sufficiency_prompt = ChatPromptTemplate.from_template(
            """You are a medical information evaluator.
            
Analyze the User Query and the Retrieved Context. Determine if the context contains sufficient information to comprehensively and accurately answer the query.

User Query: {query}

Retrieved Context:
{context}

Retrieval Scores (if available): {scores}

Consider:
1. Does the context directly address the core of the query?
2. Is the information specific enough?
3. Are there major gaps?
4. Do the retrieval scores indicate high relevance?

Respond with a JSON object containing:
- is_sufficient: boolean
- reasoning: string explanation
- missing_information: string description of what is missing (or null if sufficient)
- confidence_score: float (0.0-1.0) reflecting your confidence in the sufficiency"""
        )
        
        self.sufficiency_parser = JsonOutputParser(pydantic_object=SufficiencyCheck)
        
        self.sufficiency_chain = (
            self.sufficiency_prompt
            | self.llm
            | self.sufficiency_parser
        )

        # Conversation history fallback (if supabase not available)
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
            decision_dict = self.chain.invoke({"query": query})
            decision = AgentDecision(**decision_dict)

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
                is_medical=True,
                direct_response=None,
            )

    def check_sufficiency(
        self, 
        query: str, 
        context: str,
        retrieval_scores: Optional[List[float]] = None
    ) -> SufficiencyCheck:
        """
        Check if the retrieved context is sufficient to answer the query.

        Args:
            query: User query
            context: Retrieved context from RAG
            retrieval_scores: Optional list of retrieval scores

        Returns:
            SufficiencyCheck object
        """
        try:
            logger.info(f"Checking sufficiency for query: {query}")
            
            scores_str = str(retrieval_scores) if retrieval_scores else "Not available"
            
            result_dict = self.sufficiency_chain.invoke({
                "query": query,
                "context": context,
                "scores": scores_str
            })
            result = SufficiencyCheck(**result_dict)
            
            logger.info(f"Sufficiency check: {result.is_sufficient} (Confidence: {result.confidence_score})")
            return result
            
        except Exception as e:
            logger.error(f"Error in sufficiency check: {e}")
            # Default to sufficient to avoid unnecessary searches if check fails
            return SufficiencyCheck(
                is_sufficient=True,
                reasoning="Error in sufficiency check, defaulting to sufficient",
                missing_information=None,
                confidence_score=0.5
            )


    def add_to_history(
        self,
        role: str,
        content: str,
        agent_type: Optional[str] = None,
        session_id: Optional[str] = None,
        thinking_time: Optional[float] = None,
    ) -> None:
        """
        Add message to conversation history.

        Args:
            role: "user" or "assistant"
            content: Message content
            agent_type: Which agent handled this (if assistant)
            session_id: Session ID for Supabase
            thinking_time: Time taken to process the query
        """
        # Save to Supabase if available and session_id provided
        if self.supabase_memory and session_id:
            try:
                metadata = {"agent_type": agent_type} if agent_type else {}
                self.supabase_memory.add_message(
                    session_id=session_id,
                    role=role,
                    content=content,
                    metadata=metadata,
                    thinking_time=thinking_time
                )
            except Exception as e:
                logger.error(f"Failed to save to Supabase: {e}")
                
        # Fallback to in-memory
        self.conversation_history.append({
            "role": role,
            "content": content,
            "agent_type": agent_type,
        })

    def get_conversation_context(self, session_id: Optional[str] = None, last_n: int = 5) -> str:
        """
        Get recent conversation history as context.

        Args:
            session_id: Session ID for Supabase
            last_n: Number of recent messages to include

        Returns:
            Formatted conversation context
        """
        recent = []
        
        if self.supabase_memory and session_id:
            try:
                history = self.supabase_memory.get_history(session_id, limit=last_n)
                # Convert Supabase format to local format if needed, or just use it
                # Supabase returns list of dicts with 'role', 'content'
                recent = history
            except Exception as e:
                logger.error(f"Failed to fetch from Supabase: {e}")
                recent = self.conversation_history[-last_n:]
        else:
            recent = self.conversation_history[-last_n:]

        context_parts = []

        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"][:200]  # Limit content
            context_parts.append(f"{role}: {content}")

        return "\n".join(context_parts)

    def clear_history(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history."""
        if self.supabase_memory and session_id:
            try:
                self.supabase_memory.clear_history(session_id)
            except Exception as e:
                logger.error(f"Failed to clear Supabase history: {e}")
                
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict:
        """
        Process a query and return routing information.

        Args:
            query: User query
            session_id: Session ID

        Returns:
            Dictionary with routing information
        """
        try:
            # Get agent decision
            decision = self.decide_agent(query)

            # Add to history
            self.add_to_history("user", query, session_id=session_id)

            return {
                "agent_type": decision.agent_type.value,
                "reasoning": decision.reasoning,
                "requires_report": decision.requires_report,
                "query_refinement": decision.query_refinement,
                "is_medical": decision.is_medical,
                "direct_response": decision.direct_response,
                "conversation_context": self.get_conversation_context(session_id=session_id),
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def handle_multi_turn_conversation(
        self,
        query: str,
        previous_context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Handle multi-turn conversations with context awareness.

        Args:
            query: Current user query
            previous_context: Context from previous turns
            session_id: Session ID

        Returns:
            Routing information with conversation context
        """
        try:
            # Enhance query with previous context if available
            enhanced_query = query
            if previous_context:
                enhanced_query = f"Previous context: {previous_context}\n\nCurrent query: {query}"

            return self.process_query(enhanced_query, session_id=session_id)

        except Exception as e:
            logger.error(f"Error handling multi-turn conversation: {e}")
            raise
