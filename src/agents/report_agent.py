"""
Report Agent Module

This module implements the Report Agent that generates comprehensive,
well-formatted medical reports based on information from other agents.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class ReportAgent:
    """
    Report Agent that generates comprehensive medical reports.
    """

    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.5,
    ):
        """
        Initialize the Report Agent.

        Args:
            google_api_key: Google API key for Gemini
            model_name: Name of the Gemini model to use

        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature

        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            # google_api_key=google_api_key, # Rely on env var
        )

        # Define the report generation prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a medical report writer. Generate a comprehensive, well-structured medical report.

Topic: {topic}

Information to include:
{information}

Sources and References:
{sources}

**Instructions:**

1. **Style and Structure:**
   * Use bullet points (*) to structure lists, steps, or analysis of complex information.
   * Use **bold** to emphasize key terms, diagnoses, and important concepts.
   * Write clearly and professionally, using your best judgment on length and detail based on the complexity of the question.
   * Each answer must end with a clear summary or conclusion.

2. **Special Handling:**
   * **DO NOT diagnose the patient.** Instead, guide the user through a general procedure that a physician might use. Combine theory with illustrative examples (e.g., "For a condition like this, a typical diagnostic procedure includes: 1. Patient history, 2. Physical examination, 3. Specific tests such as...").
   * **Health record:** Provide general medical information about the findings. (e.g., "This presentation is often associated with...").

3. **Final Format (Markdown):**
   * Structure your summary answer.
   * **Citation and Linking:** Use the following format to cite evidence and link to references (All key answers need to have the citation):
     * Argument will look like: Arguments/Evidence sentences in answer <sup>[[1]](#ref1)</sup>
     * References will look like: <a id="ref1">1.</a> Reference in APA 7 style for sentences above
   * Add a **References** section: list all sources in APA version 7 format in order (Separate each by spacing or newline).
   * **Final output:** Your final output to the user should be just the language same as user input. Do not add any other text.

Please generate the report following these strict guidelines."""
        )

        # Define the short answer prompt
        self.short_answer_prompt_template = ChatPromptTemplate.from_template(
            """You are a medical assistant providing a concise and accurate answer.

Topic: {topic}

Information to include:
{information}

Sources and References:
{sources}

**Instructions:**

1. **Style:**
   * Be concise and direct. Do NOT generate a full report with sections.
   * Focus on answering the specific question asked.
   * Use **bold** for key terms.

2. **Safety:**
   * **DO NOT diagnose.** Provide general medical information.

3. **Citations (CRITICAL):**
   * You MUST cite your sources for every factual claim.
   * Use the following format:
     * Sentence with claim <sup>[[1]](#ref1)</sup>
     * References section at the bottom: <a id="ref1">1.</a> Reference details
   * Add a **References** section at the very end.

4. **Final output:**
   * Just the answer and references. No conversational filler.

Please generate the short answer following these guidelines."""
        )

        # Create the report chain
        self.chain = (
            self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Create the short answer chain
        self.short_answer_chain = (
            self.short_answer_prompt_template
            | self.llm
            | StrOutputParser()
        )

        logger.info(f"Report Agent initialized with model: {model_name}")

    def generate_report(
        self,
        topic: str,
        information: str,
        sources: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a comprehensive medical report.

        Args:
            topic: Report topic
            information: Information to include in the report
            sources: List of sources/references

        Returns:
            Generated report as string
        """
        try:
            logger.info(f"Generating report for topic: {topic}")

            # Format sources
            sources_text = "\n".join(sources) if sources else "No sources provided"

            # Generate report
            report = self.chain.invoke({
                "topic": topic,
                "information": information,
                "sources": sources_text,
            })

            logger.info("Report generated successfully")
            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def stream_report(
        self,
        topic: str,
        information: str,
        sources: Optional[List[str]] = None,
    ):
        """
        Stream report generation (for real-time UI updates).

        Args:
            topic: Report topic
            information: Information to include
            sources: List of sources

        Yields:
            Chunks of the report
        """
        try:
            logger.info(f"Streaming report for topic: {topic}")

            # Format sources
            sources_text = "\n".join(sources) if sources else "No sources provided"

            # Stream the report
            for chunk in self.chain.stream({
                "topic": topic,
                "information": information,
                "sources": sources_text,
            }):
                yield chunk

        except Exception as e:
            logger.error(f"Error streaming report: {e}")
            raise

    def generate_short_answer(
        self,
        query: str,
        rag_results: Optional[Dict] = None,
        search_results: Optional[Dict] = None,
    ) -> str:
        """
        Generate a short answer combining RAG and search results.

        Args:
            query: Original user query
            rag_results: Results from RAG agent
            search_results: Results from Search agent

        Returns:
            Generated short answer
        """
        try:
            logger.info(f"Generating short answer for query: {query}")

            # Compile information from different sources
            information_parts = []

            if rag_results:
                information_parts.append(
                    f"Knowledge Base Information:\n{rag_results.get('answer', '')}"
                )

            if search_results:
                information_parts.append(
                    f"Recent Research and News:\n{search_results.get('answer', '')}"
                )

            information = "\n\n".join(information_parts)

            # Compile sources with rich metadata
            sources = []
            if rag_results:
                for doc in rag_results.get("retrieved_documents", []):
                    meta = doc.get("metadata", {})
                    
                    # Try to construct a rich citation
                    book = meta.get("book_name")
                    author = meta.get("author")
                    year = meta.get("publish_year")
                    page = meta.get("page_number")
                    
                    if book:
                        citation = f"{book}"
                        if year:
                            citation += f" ({year})"
                        if author:
                            citation += f", by {author}"
                        if page:
                            citation += f", p. {page}"
                    else:
                        # Fallback to source field or unknown
                        citation = meta.get("source", "Unknown Source")
                        
                    if citation not in sources:
                        sources.append(citation)

            if search_results:
                for result in search_results.get("search_results", []):
                    title = result.get("title", "Unknown Title")
                    link = result.get("link", "Unknown Link")
                    citation = f"{title} - {link}"
                    if citation not in sources:
                        sources.append(citation)

            sources_text = "\n".join(sources) if sources else "No sources provided"

            # Generate short answer
            answer = self.short_answer_chain.invoke({
                "topic": query,
                "information": information,
                "sources": sources_text,
            })

            return answer

        except Exception as e:
            logger.error(f"Error generating short answer: {e}")
            raise

    def generate_summary_report(
        self,
        query: str,
        rag_results: Optional[Dict] = None,
        search_results: Optional[Dict] = None,
    ) -> str:
        """
        Generate a summary report combining RAG and search results.

        Args:
            query: Original user query
            rag_results: Results from RAG agent
            search_results: Results from Search agent

        Returns:
            Generated summary report
        """
        try:
            logger.info(f"Generating summary report for query: {query}")

            # Compile information from different sources
            information_parts = []

            if rag_results:
                information_parts.append(
                    f"Knowledge Base Information:\n{rag_results.get('answer', '')}"
                )

            if search_results:
                information_parts.append(
                    f"Recent Research and News:\n{search_results.get('answer', '')}"
                )

            information = "\n\n".join(information_parts)

            # Compile sources with rich metadata
            sources = []
            if rag_results:
                for doc in rag_results.get("retrieved_documents", []):
                    meta = doc.get("metadata", {})
                    
                    # Try to construct a rich citation
                    book = meta.get("book_name")
                    author = meta.get("author")
                    year = meta.get("publish_year")
                    page = meta.get("page_number")
                    
                    if book:
                        citation = f"{book}"
                        if year:
                            citation += f" ({year})"
                        if author:
                            citation += f", by {author}"
                        if page:
                            citation += f", p. {page}"
                    else:
                        # Fallback to source field or unknown
                        citation = meta.get("source", "Unknown Source")
                        
                    if citation not in sources:
                        sources.append(citation)

            if search_results:
                for result in search_results.get("search_results", []):
                    title = result.get("title", "Unknown Title")
                    link = result.get("link", "Unknown Link")
                    citation = f"{title} - {link}"
                    if citation not in sources:
                        sources.append(citation)

            # Generate report
            report = self.generate_report(
                topic=query,
                information=information,
                sources=sources,
            )

            return report

        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise


    def format_report_with_metadata(
        self,
        report_content: str,
        query: str,
        timestamp: Optional[str] = None,
    ) -> str:
        """
        Format report with metadata.

        Args:
            report_content: The report content
            query: Original query
            timestamp: Report generation timestamp

        Returns:
            Formatted report with metadata
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""
================================================================================
MEDCHAT MEDICAL REPORT
================================================================================
Query: {query}
Generated: {timestamp}
================================================================================

"""

        footer = """
================================================================================
This report was generated by MedChat, an AI-powered medical information system.
It is intended for educational purposes for medical students.
Always consult with qualified medical professionals for clinical decisions.
================================================================================
"""

        return header + report_content + footer

    def export_report_to_file(
        self,
        report_content: str,
        filename: str,
        include_metadata: bool = True,
    ) -> str:
        """
        Export report to a file.

        Args:
            report_content: Report content
            filename: Output filename
            include_metadata: Whether to include metadata in the file

        Returns:
            Path to the saved file
        """
        try:
            logger.info(f"Exporting report to {filename}")

            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_content)

            logger.info(f"Report exported successfully to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
