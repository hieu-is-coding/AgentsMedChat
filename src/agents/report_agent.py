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

        # Define the report generation prompt
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a medical report writer. Generate a comprehensive, well-structured medical report.

Topic: {topic}

Information to include:
{information}

Sources and References:
{sources}

Please generate a professional medical report with the following structure:
1. Executive Summary
2. Introduction
3. Main Content (organized by relevant subtopics)
4. Key Findings
5. Clinical Implications
6. Recommendations
7. References

Ensure the report is accurate, well-cited, and suitable for medical students."""
        )

        # Create the report chain
        self.chain = (
            self.prompt_template
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

            # Compile sources
            sources = []
            if rag_results:
                for doc in rag_results.get("retrieved_documents", []):
                    source = doc.get("metadata", {}).get("source", "Unknown")
                    if source not in sources:
                        sources.append(source)

            if search_results:
                for result in search_results.get("search_results", []):
                    link = result.get("link", "Unknown")
                    if link not in sources:
                        sources.append(link)

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
