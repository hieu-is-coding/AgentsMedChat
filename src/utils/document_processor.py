"""
Document Processing Utilities

This module provides utilities for loading, processing, and chunking medical documents
for use with the RAG pipeline.
"""

import logging
import os
from typing import List, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles loading and processing of medical documents.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_pdf(self, file_path: str) -> str:
        """
        Load text from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        try:
            logger.info(f"Loading PDF: {file_path}")
            reader = PdfReader(file_path)
            text = ""
            for page_num, page in enumerate(reader.pages):
                text += page.extract_text()
                logger.debug(f"Extracted page {page_num + 1}")

            logger.info(f"Successfully loaded {len(reader.pages)} pages from {file_path}")
            return text

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise

    def load_text_file(self, file_path: str) -> str:
        """
        Load text from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            File content
        """
        try:
            logger.info(f"Loading text file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"Successfully loaded {len(content)} characters from {file_path}")
            return content

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            raise

    def load_directory(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load all documents from a directory.

        Args:
            directory_path: Path to the directory
            file_extensions: List of file extensions to load (e.g., ['.pdf', '.txt'])

        Returns:
            List of Document objects
        """
        if file_extensions is None:
            file_extensions = [".pdf", ".txt"]

        documents = []
        directory = Path(directory_path)

        try:
            logger.info(f"Loading documents from directory: {directory_path}")

            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                    try:
                        if file_path.suffix.lower() == ".pdf":
                            content = self.load_pdf(str(file_path))
                        else:
                            content = self.load_text_file(str(file_path))

                        # Create document with metadata
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                            },
                        )
                        documents.append(doc)

                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
                        continue

            logger.info(f"Loaded {len(documents)} documents from {directory_path}")
            return documents

        except Exception as e:
            logger.error(f"Error loading directory {directory_path}: {e}")
            raise

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        try:
            logger.info(f"Chunking {len(documents)} documents")

            chunked_docs = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)
                logger.debug(
                    f"Document {doc.metadata.get('filename')} split into {len(chunks)} chunks"
                )

                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    )
                    chunked_docs.append(chunk_doc)

            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            return chunked_docs

        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise

    def process_documents(
        self,
        directory_path: str,
        file_extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Complete pipeline: load documents from directory and chunk them.

        Args:
            directory_path: Path to the directory containing documents
            file_extensions: List of file extensions to load

        Returns:
            List of chunked Document objects ready for embedding
        """
        try:
            # Load documents
            documents = self.load_directory(directory_path, file_extensions)

            # Chunk documents
            chunked_documents = self.chunk_documents(documents)

            return chunked_documents

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            raise
