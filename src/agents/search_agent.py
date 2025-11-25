"""
Search Agent Module

This module implements the Search Agent that performs web searches
to retrieve current medical information and research using Gemini's native Google Search tool.
"""

import logging
from typing import List, Dict, Optional, Generator
from google.ai.generativelanguage_v1beta import GenerativeServiceClient
from google.ai.generativelanguage_v1beta import GenerateContentRequest
from google.ai.generativelanguage_v1beta.types import Content, Part, Tool as GenAITool

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Search Agent that performs web searches for current medical information
    using Gemini's native Google Search capabilities.
    """

    def __init__(
        self,
        google_api_key: str,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.7,
    ):
        """
        Initialize the Search Agent.

        Args:
            google_api_key: Google API key
            model_name: Name of the Gemini model to use
            temperature: Temperature for model generation
        """
        self.google_api_key = google_api_key
        self.model_name = model_name
        self.temperature = temperature

        # Initialize the GenerativeServiceClient
        self.client = GenerativeServiceClient(client_options={"api_key": google_api_key})
        
        # Configure the tool
        self.tool = GenAITool(google_search=GenAITool.GoogleSearch())

    def answer_question(self, question: str) -> dict:
        """
        Answer a question using web search.

        Args:
            question: User question

        Returns:
            Dictionary with answer and search results
        """
        try:
            logger.info(f"Processing question: {question}")

            # Construct the request
            content = Content(parts=[Part(text=question)])
            request = GenerateContentRequest(
                model=f"models/{self.model_name}",
                contents=[content],
                tools=[self.tool]
            )

            # Generate content
            response = self.client.generate_content(request)
            
            # Extract answer
            answer = ""
            if response.candidates and response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text

            # Extract grounding metadata
            search_results = []
            if response.candidates and hasattr(response.candidates[0], 'grounding_metadata'):
                gm = response.candidates[0].grounding_metadata
                for chunk in gm.grounding_chunks:
                    if chunk.web:
                        search_results.append({
                            "title": chunk.web.title,
                            "link": chunk.web.uri,
                            "snippet": "" # Snippet is not always available in chunks, sometimes in entry point or just implicit
                        })

            return {
                "question": question,
                "answer": answer,
                "search_results": search_results,
                "formatted_results": self._format_search_results(search_results)
            }

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results for display."""
        if not results:
            return "No search results available."
        
        formatted = []
        for i, res in enumerate(results, 1):
            formatted.append(f"{i}. [{res.get('title', 'Unknown')}]({res.get('link', '#')})")
        return "\n".join(formatted)

    def stream_answer(self, question: str) -> Generator:
        """
        Stream answer for a question.
        """
        try:
            logger.info(f"Streaming answer for: {question}")
            
            # Construct the request
            content = Content(parts=[Part(text=question)])
            request = GenerateContentRequest(
                model=f"models/{self.model_name}",
                contents=[content],
                tools=[self.tool]
            )
            
            # Stream content
            for chunk in self.client.stream_generate_content(request):
                if chunk.candidates and chunk.candidates[0].content.parts:
                    yield chunk.candidates[0].content.parts[0].text
                
        except Exception as e:
            logger.error(f"Error streaming answer: {e}")
            raise

