"""
Test script for MedChat multi-agent system.

This script demonstrates how to:
1. Initialize MedChat
2. Test each agent
3. Verify multi-agent workflow
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.medchat import MedChat
from config_template import (
    GOOGLE_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    GEMINI_MODEL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_rag_agent(medchat: MedChat):
    """Test the RAG agent."""
    logger.info("\n" + "="*60)
    logger.info("Testing RAG Agent")
    logger.info("="*60)

    query = "What are the symptoms of hypertension?"

    try:
        logger.info(f"Query: {query}")
        result = medchat.rag_agent.answer_question(query)

        logger.info(f"Answer: {result['answer'][:200]}...")
        logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")

        for i, doc in enumerate(result["retrieved_documents"], 1):
            logger.info(f"  Document {i}: {doc['metadata'].get('source', 'Unknown')}")

    except Exception as e:
        logger.error(f"Error testing RAG agent: {e}", exc_info=True)


def test_search_agent(medchat: MedChat):
    """Test the Search agent."""
    logger.info("\n" + "="*60)
    logger.info("Testing Search Agent")
    logger.info("="*60)

    query = "Latest treatment guidelines for diabetes 2025"

    try:
        logger.info(f"Query: {query}")
        result = medchat.search_agent.answer_question(query)

        logger.info(f"Answer: {result['answer'][:200]}...")
        logger.info(f"Found {len(result['search_results'])} search results")

        for i, search_result in enumerate(result["search_results"][:3], 1):
            logger.info(f"  Result {i}: {search_result.get('title', 'No title')}")

    except Exception as e:
        logger.error(f"Error testing Search agent: {e}", exc_info=True)


def test_orchestration_agent(medchat: MedChat):
    """Test the Orchestration agent."""
    logger.info("\n" + "="*60)
    logger.info("Testing Orchestration Agent")
    logger.info("="*60)

    test_queries = [
        "What is diabetes?",  # Should route to RAG
        "What are the latest cancer treatments?",  # Should route to Search
        "Generate a comprehensive report on hypertension",  # Should route to Report
    ]

    for query in test_queries:
        try:
            logger.info(f"\nQuery: {query}")
            routing_info = medchat.orchestration_agent.process_query(query)

            logger.info(f"Routed to: {routing_info['agent_type']}")
            logger.info(f"Reasoning: {routing_info['reasoning']}")
            logger.info(f"Requires report: {routing_info['requires_report']}")

        except Exception as e:
            logger.error(f"Error testing orchestration: {e}", exc_info=True)


def test_multi_agent_workflow(medchat: MedChat):
    """Test the complete multi-agent workflow."""
    logger.info("\n" + "="*60)
    logger.info("Testing Multi-Agent Workflow")
    logger.info("="*60)

    query = "What are the risk factors and treatment options for heart disease?"

    try:
        logger.info(f"Query: {query}")
        result = medchat.process_query(query)

        logger.info(f"Agent used: {result['agent_type']}")
        logger.info(f"Answer: {result['answer'][:300]}...")

        if "retrieved_documents" in result:
            logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")

        if "search_results" in result:
            logger.info(f"Found {len(result['search_results'])} search results")

    except Exception as e:
        logger.error(f"Error testing multi-agent workflow: {e}", exc_info=True)


def test_health_check(medchat: MedChat):
    """Test system health check."""
    logger.info("\n" + "="*60)
    logger.info("System Health Check")
    logger.info("="*60)

    try:
        health = medchat.health_check()

        for component, status in health.items():
            status_str = "✓ OK" if status else "✗ FAILED"
            logger.info(f"{component}: {status_str}")

    except Exception as e:
        logger.error(f"Error during health check: {e}", exc_info=True)


def main():
    """Main test function."""
    logger.info("Initializing MedChat for testing...")

    try:
        medchat = MedChat(
            google_api_key=GOOGLE_API_KEY,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            gemini_model=GEMINI_MODEL,
        )

        logger.info("MedChat initialized successfully\n")

        # Run tests
        test_health_check(medchat)
        test_rag_agent(medchat)
        test_search_agent(medchat)
        test_orchestration_agent(medchat)
        test_multi_agent_workflow(medchat)

        logger.info("\n" + "="*60)
        logger.info("All tests completed!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Error initializing MedChat: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
