from typing import List, Tuple
from models import FinalAnswer
from document_manager import DocumentManager
from retriever import VectorStoreProvider
from workflow import RAGWorkflow

class QueryService:
    """
    A service class to orchestrate the RAG process:
    - Downloads documents
    - Manages a cache of processed vector stores
    - Generates responses based on retrieved information and questions.
    """
    def __init__(self):
        self.llm = RAGWorkflow()

    def process_queries(
        self,
        document_url: str,
        questions: List[str]
    ) -> List[FinalAnswer]:
        """
        Processes a list of questions against a document URL.
        """
        print("Processing new document and building vector store...")
        document_manager = DocumentManager(document_url)
        retriever = VectorStoreProvider(document_manager).retriever
        print("retriever created....\ncalling llm")

        results = []
        for question in questions:
            results.append(self.llm.invoke(question,retriever))

        return results
