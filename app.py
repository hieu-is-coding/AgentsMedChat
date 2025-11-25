"""
MedChat Streamlit Application

Interactive UI for the multi-agent medical chatbot system.
"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st
from streamlit_chat import message

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.medchat import MedChat
from config_template import (
    GOOGLE_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    GEMINI_MODEL,
    APP_NAME,
    DEBUG,
    LOG_LEVEL,
    MEDICAL_COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .agent-rag {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .agent-search {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .agent-report {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_medchat():
    """Initialize MedChat application (cached)."""
    try:
        logger.info("Initializing MedChat...")
        medchat = MedChat(
            google_api_key=GOOGLE_API_KEY,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
            gemini_model=GEMINI_MODEL,
            collection_name=MEDICAL_COLLECTION_NAME,
            embedding_dimension=EMBEDDING_DIMENSION,
        )

        # Perform health check
        health = medchat.health_check()
        logger.info(f"Health check: {health}")

        return medchat

    except Exception as e:
        logger.error(f"Failed to initialize MedChat: {e}")
        st.error(f"Failed to initialize MedChat: {e}")
        st.stop()


def main():
    """Main Streamlit application."""
    # Header
    col1, col2 = st.columns([0.1, 0.9])
    with col1:
        st.markdown("🏥")
    with col2:
        st.title(APP_NAME)

    st.markdown(
        "A multi-agent AI chatbot designed to help medical students with questions about medicine, "
        "current research, and clinical information."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Clear conversation button
        if st.button("Clear Conversation", key="clear_conv"):
            st.session_state.messages = []
            st.session_state.medchat.clear_conversation_history()
            st.success("Conversation cleared!")

        # Vector store info
        st.subheader("Vector Store Status")
        try:
            medchat = st.session_state.medchat
            store_info = medchat.get_vector_store_info()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", store_info.get("points_count", "N/A"))
            with col2:
                st.metric("Status", store_info.get("status", "Unknown"))

        except Exception as e:
            st.warning(f"Could not fetch vector store info: {e}")

        # Debug info
        if DEBUG:
            st.subheader("Debug Info")
            st.write(f"Model: {GEMINI_MODEL}")
            st.write(f"Qdrant URL: {QDRANT_URL}")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "medchat" not in st.session_state:
        st.session_state.medchat = initialize_medchat()

    # Display chat history
    st.subheader("Conversation")
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(
                    msg["content"],
                    is_user=True,
                    key=f"msg_{i}",
                )
            else:
                agent_type = msg.get("agent_type", "unknown")
                agent_badge = f'<span class="agent-badge agent-{agent_type}">{agent_type.upper()}</span>'

                message(
                    msg["content"] + agent_badge,
                    is_user=False,
                    key=f"msg_{i}",
                    allow_html=True,
                )

    # Input area
    st.subheader("Ask a Question")

    col1, col2 = st.columns([0.9, 0.1])

    with col1:
        user_input = st.text_input(
            "Your question:",
            placeholder="Ask about medical conditions, treatments, recent research...",
            key="user_input",
        )

    with col2:
        submit_button = st.button("Send", key="send_button", use_container_width=True)

    # Process user input
    if submit_button and user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })

        # Show thinking indicator
        with st.spinner("Thinking..."):
            try:
                medchat = st.session_state.medchat

                # Process query
                result = medchat.process_query(user_input)

                # Extract response
                response = result.get("answer", "No response generated")
                agent_type = result.get("agent_type", "unknown")

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "agent_type": agent_type,
                })

                # Show additional information if available
                if "retrieved_documents" in result:
                    with st.expander("📚 Retrieved Documents"):
                        for i, doc in enumerate(result["retrieved_documents"], 1):
                            st.write(f"**Document {i}** (Score: {doc['score']:.2f})")
                            st.write(f"Source: {doc['metadata'].get('source', 'Unknown')}")
                            st.write(doc["content"][:300] + "...")

                if "search_results" in result:
                    with st.expander("🔍 Search Results"):
                        for i, search_result in enumerate(result["search_results"], 1):
                            st.write(f"**Result {i}**")
                            st.write(f"Title: {search_result.get('title', 'No title')}")
                            st.write(f"Link: {search_result.get('link', 'No link')}")
                            st.write(search_result.get("snippet", "No snippet")[:300] + "...")

                # Rerun to update chat display
                st.rerun()

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"Error processing query: {e}")

    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8rem;">
        <p>MedChat is an AI-powered educational tool for medical students.</p>
        <p>Always consult with qualified medical professionals for clinical decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
