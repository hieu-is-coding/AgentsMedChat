# MedChat: Multi-Agent Chatbot for Medical Students

A sophisticated multi-agent AI system designed to assist medical students with comprehensive medical information, current research, and clinical guidance. MedChat leverages **Google Gemini 2.0**, **LangChain**, **Qdrant**, and **Streamlit** to provide intelligent, context-aware responses.

## 🚀 Installation

### 1. Clone and Setup

```bash
cd AgentsMedChat
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file based on `config_template.py`:

```python
# Google Gemini API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Model Configuration
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=models/gemini-embedding-001
```

## 🏗️ Architecture

```
User Query
    ↓
[Orchestration Agent] → Routes to appropriate agent
    ↓
    ├─→ [RAG Agent] → Qdrant Vector Store
    │       ↓
    │   Retrieved Documents
    │
    ├─→ [Search Agent] → Google Search API
    │       ↓
    │   Web Results
    │
    └─→ [Report Agent] → Generates Report
            ↓
        Formatted Response
            ↓
        Streamlit UI
```

## 📋 Prerequisites

- Python 3.11+
- Google API Key (for Gemini and Search)
- Google Custom Search Engine ID
- Qdrant instance (local, Docker, or cloud)

### 3. Set Up Qdrant

#### Qdrant Cloud
Sign up at [Qdrant Cloud](https://cloud.qdrant.io/) and get your API key.

### 4. Load Medical Documents

```bash
python setup_qdrant.py
```

This script will:
- Initialize Qdrant collection
- Load sample medical documents
- Test similarity search functionality

## 💻 Usage

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Run Tests

```bash
python test_medchat.py
```

This will test:
- RAG Agent functionality
- Search Agent functionality
- Orchestration Agent routing
- Multi-agent workflow
- System health checks


## 📁 Project Structure

```
MedChat/
├── src/
│   ├── agents/
│   │   ├── orchestration_agent.py    # Query routing and decision-making
│   │   ├── rag_agent.py              # Knowledge base retrieval
│   │   ├── search_agent.py           # Web search integration
│   │   └── report_agent.py           # Report generation
│   ├── data/
│   │   └── qdrant_pipeline.py        # Vector database operations
│   ├── utils/
│   │   └── document_processor.py     # Document loading and chunking
│   └── medchat.py                    # Main application coordinator
├── app.py                             # Streamlit UI
├── setup_qdrant.py                   # Qdrant initialization script
├── test_medchat.py                   # System testing script
├── config_template.py                # Configuration template
├── requirements.txt                  # Python dependencies
├── ARCHITECTURE.md                   # Detailed architecture documentation
└── README.md                         # This file
```

## 🤖 Agent Descriptions

### Orchestration Agent
Routes queries based on intent analysis:
- **RAG**: Knowledge base questions
- **Search**: Current information requests
- **Report**: Formal report generation
- **General**: Fallback for unclear queries

### RAG Agent
Retrieves and augments responses with knowledge base:
- Similarity search in Qdrant
- Context-aware response generation
- Source citation and tracking

### Search Agent
Performs web searches for current information:
- Google Search API integration
- Result synthesis and summarization
- Source attribution

### Report Agent
Generates comprehensive medical reports:
- Structured report generation
- Multi-source information synthesis
- Professional formatting
- File export capabilities
