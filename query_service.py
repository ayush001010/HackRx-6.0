import os
import pickle
from typing import List, Tuple
from langchain_core.documents import Document

from document_manager import DocumentManager
from retriever import VectorStoreProvider
from models import QueryResponse, GeneratedQueries
from response_generator import ResponseGenerator

class QueryService:
    """
    A service class to orchestrate the RAG process:
    - Downloads documents
    - Manages a cache of processed vector stores
    - Generates responses based on retrieved information and questions.
    """
    def __init__(self):
        """
        Initializes the service with an in-memory cache.
        """
        
        self._cache = {}  # In-memory cache to store processed vector stores

    def process_queries(
        self,
        document_url: str,
        questions: List[str],
        background_tasks
    ) -> List[Tuple[QueryResponse, GeneratedQueries]]:
        """
        Processes a list of questions against a document URL.
        This method uses a cache to avoid re-processing the same document multiple times.
        """
        # Step 1: Check cache for existing vector store
        if document_url in self._cache:
            print("Using cached vector store for document...")
            vector_store_provider = self._cache[document_url]
        else:
            print("Processing new document and building vector store...")
            # Step 2: If not in cache, process the document
            document_manager = DocumentManager(document_url)
            vector_store_provider = VectorStoreProvider(document_manager)

            # Step 3: Store the new vector store in the cache
            self._cache[document_url] = vector_store_provider
            print("Vector store added to cache.")

            # Add a background task to clean up the temporary file after processing
            background_tasks.add_task(document_manager.cleanup)

        # Step 4: Create a ResponseGenerator with the vector store
        response_generator = ResponseGenerator(vector_store_provider)

        # Step 5: Process all questions and generate responses
        results = [
            response_generator.generate_response(question) for question in questions
        ]

        return results
